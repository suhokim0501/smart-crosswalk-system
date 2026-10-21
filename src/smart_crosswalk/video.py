"""Local video reading, timestamping and bounded pedestrian crops."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


def require_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Video analysis requires: pip install -e ".[vision]"') from exc
    return cv2


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


def inspect_video(path: Path) -> VideoMetadata:
    cv2 = require_opencv()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        metadata = VideoMetadata(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if not math.isfinite(metadata.fps) or metadata.fps <= 0 or min(metadata.width, metadata.height, metadata.frame_count) <= 0:
            raise ValueError("Video must have positive dimensions, frame count and FPS.")
        return metadata
    finally:
        capture.release()


def iter_frames(path: Path, metadata: VideoMetadata, frame_size, stride: int, until_s: float):
    """Yield (source frame index, seconds, BGR image); never sample after cutoff."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if not math.isfinite(until_s) or until_s < 0:
        raise ValueError("until_s must be finite and nonnegative.")
    cv2 = require_opencv()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        index = 0
        while index / metadata.fps <= until_s + 1e-9:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, index / metadata.fps, cv2.resize(frame, tuple(frame_size))
            index += 1
    finally:
        capture.release()


def crop_bbox(image: np.ndarray, bbox_xywh) -> np.ndarray | None:
    """Clip to image bounds so negative indices cannot wrap to the other edge."""
    x, y, width, height = (float(v) for v in bbox_xywh)
    if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
        return None
    image_height, image_width = image.shape[:2]
    left = max(0, min(image_width, math.floor(x - width / 2)))
    right = max(0, min(image_width, math.ceil(x + width / 2)))
    top = max(0, min(image_height, math.floor(y - height / 2)))
    bottom = max(0, min(image_height, math.ceil(y + height / 2)))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right].copy()
