"""Optional annotated video export, separated from detection and decisions."""

from pathlib import Path

import numpy as np

from .types import Tracks
from .video import VideoMetadata, iter_frames, require_opencv


def track_color(track_id: int) -> tuple[int, int, int]:
    """Stable BGR colors without modifying NumPy's global random state."""
    return tuple(int(value) for value in np.random.default_rng(track_id).integers(64, 256, size=3))


def render_video(video_path: Path, output_path: Path, metadata: VideoMetadata,
                 frame_size, stride: int, until_s: float, tracks: Tracks, decisions) -> None:
    """Write sampled frames through the decision window, with cm/s labels."""
    cv2 = require_opencv()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             metadata.fps / stride, tuple(frame_size))
    by_frame = {}
    labels = {item.track_id: item for item in decisions}
    for track_id, observations in tracks.items():
        for observation in observations:
            by_frame.setdefault(observation.frame_index, []).append((track_id, observation))
    try:
        if not writer.isOpened():
            raise ValueError(f"Cannot create annotated video: {output_path}")
        for index, _, image in iter_frames(video_path, metadata, frame_size, stride, until_s):
            for track_id, observation in by_frame.get(index, []):
                x, y, width, height = observation.bbox_xywh
                top_left = (int(x - width / 2), int(y - height / 2))
                bottom_right = (int(x + width / 2), int(y + height / 2))
                color = track_color(track_id)
                decision = labels.get(track_id)
                label = f"ID {track_id}"
                if decision is not None:
                    if decision.class_name:
                        label += f" {decision.class_name}"
                    if decision.speed_cm_s is not None:
                        label += f" {decision.speed_cm_s:.1f} cm/s"
                cv2.rectangle(image, top_left, bottom_right, color, 2)
                cv2.putText(image, label, (max(0, top_left[0]), max(15, top_left[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            writer.write(image)
    finally:
        writer.release()
