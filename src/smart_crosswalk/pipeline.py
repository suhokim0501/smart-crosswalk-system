"""Compose video -> tracking -> filtering -> classification -> signal estimate."""

from dataclasses import asdict
from pathlib import Path

from .backend import YoloBackend
from .classification import classify_tracks
from .config import AppConfig
from .geometry import compute_homography, ground_position
from .timing import evaluate_tracks
from .tracking import filter_observations, filter_short_tracks
from .types import Observation, Tracks
from .video import inspect_video, iter_frames


def run_analysis(config: AppConfig, backend=None, annotated_output: Path | None = None) -> dict:
    """Analyze a local clip whose t=0 is assumed to be the start of green."""
    if not config.video.is_file():
        raise FileNotFoundError(f"Local video file not found: {config.video}")
    metadata = inspect_video(config.video)
    crosswalk = config.crosswalk
    if metadata.frame_count / metadata.fps < crosswalk.decision_time_s:
        raise ValueError("Video ends before decision_time_s; use a longer clip or an earlier decision.")
    matrix = compute_homography(crosswalk.pixel_points, crosswalk.ground_points_cm,
                                crosswalk.calibration_size, crosswalk.frame_size)
    backend = backend if backend is not None else YoloBackend(config)
    frames = {}
    tracks: Tracks = {}
    processed_frames = 0
    last_timestamp = None
    invalid_projections = 0
    for index, timestamp, image in iter_frames(config.video, metadata, crosswalk.frame_size,
                                              config.tracking.stride, crosswalk.decision_time_s):
        processed_frames += 1
        last_timestamp = timestamp
        detections = backend.track(image)
        if detections:
            frames[index] = image
        for detection in detections:
            try:
                position = ground_position(detection.bbox_xywh, matrix)
            except ValueError:
                invalid_projections += 1
                continue
            observation = Observation(index, timestamp, detection.bbox_xywh, position)
            tracks.setdefault(detection.track_id, []).append(observation)
    tolerance = config.tracking.stride / metadata.fps
    if last_timestamp is None or last_timestamp < crosswalk.decision_time_s - tolerance - 1e-9:
        raise ValueError("Could not decode video frames through the decision window.")
    selected = filter_observations(tracks, config.tracking.roi_cm, crosswalk.frame_size[1])
    selected = filter_short_tracks(selected, processed_frames, config.tracking.min_track_ratio)
    probabilities = classify_tracks(selected, frames, backend.predict, config.classification)
    decision = evaluate_tracks(selected, probabilities, crosswalk, config.tracking, config.classification)
    if annotated_output is not None:
        from .visualization import render_video
        render_video(config.video, annotated_output, metadata, crosswalk.frame_size,
                     config.tracking.stride, crosswalk.decision_time_s, selected, decision.tracks)
    warnings = []
    if not any(t.reason == "eligible" for t in decision.tracks):
        warnings.append("No eligible tracks; zero extension is not a pedestrian-safety assessment.")
    if invalid_projections:
        warnings.append(f"Skipped {invalid_projections} observations with invalid ground projections.")
    return {
        "source": "video",
        "video": config.video.name,
        "video_metadata": asdict(metadata),
        "processed_frames": processed_frames,
        "tracked_objects": len(tracks),
        "selected_objects": len(selected),
        "annotated_video": annotated_output.name if annotated_output else None,
        "units": {"distance": "cm", "speed": "cm/s", "time": "s"},
        "decision": decision.to_dict(),
        "warnings": warnings,
    }
