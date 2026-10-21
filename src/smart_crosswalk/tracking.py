"""Track selection and constant-velocity estimation, independent of YOLO."""

import math
import numpy as np

from .types import Observation, Tracks


def filter_observations(tracks: Tracks, roi_cm, max_bottom_px: float) -> Tracks:
    """Return new lists; discard empty tracks and observations outside the ROI."""
    min_x, min_y, max_x, max_y = roi_cm
    filtered = {}
    for track_id, observations in tracks.items():
        selected = [o for o in observations
                    if min_x <= o.ground_xy_cm[0] <= max_x
                    and min_y <= o.ground_xy_cm[1] <= max_y
                    and o.bbox_xywh[1] + o.bbox_xywh[3] / 2 <= max_bottom_px]
        if selected:
            filtered[track_id] = sorted(selected, key=lambda o: o.timestamp_s)
    return filtered


def filter_short_tracks(tracks: Tracks, processed_frames: int, min_ratio: float) -> Tracks:
    if processed_frames < 0 or not 0 < min_ratio <= 1:
        raise ValueError("Require nonnegative frame count and min_ratio in (0, 1].")
    minimum = max(2, math.ceil(processed_frames * min_ratio))
    return {track_id: list(observations) for track_id, observations in tracks.items()
            if len(observations) >= minimum}


def estimate_speed_cm_s(observations: list[Observation]) -> float | None:
    """OLS y(t) slope, negated because the destination lies at ground y=0."""
    if len(observations) < 2:
        return None
    times = np.asarray([o.timestamp_s for o in observations], dtype=float)
    positions = np.asarray([o.ground_xy_cm[1] for o in observations], dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(positions).all():
        return None
    centered = times - times.mean()
    denominator = float(centered @ centered)
    if denominator <= np.finfo(float).eps:
        return None
    return float(-(centered @ (positions - positions.mean())) / denominator)
