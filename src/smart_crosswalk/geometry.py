"""Pixel/ground projection using normalized Direct Linear Transform."""

import numpy as np


def _normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance <= np.finfo(float).eps:
        raise ValueError("Calibration points must not coincide.")
    scale = np.sqrt(2.0) / distance
    transform = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
    normalized = (points - center) * scale
    return normalized, transform


def compute_homography(pixel_points, ground_points, calibration_size, frame_size) -> np.ndarray:
    """Scale x/y independently, then fit a projective transform to all pairs."""
    source = np.asarray(pixel_points, dtype=float)
    target = np.asarray(ground_points, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape or len(source) < 4:
        raise ValueError("Need at least four matching 2D calibration points.")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Calibration points must be finite.")
    calibration_size = np.asarray(calibration_size, dtype=float)
    frame_size = np.asarray(frame_size, dtype=float)
    if any(size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any()
           for size in (calibration_size, frame_size)):
        raise ValueError("Calibration and frame sizes must have two positive dimensions.")
    source = source * frame_size / calibration_size
    source, source_transform = _normalize(source)
    target, target_transform = _normalize(target)
    rows = []
    for (x, y), (u, v) in zip(source, target):
        rows.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    design = np.asarray(rows)
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("Degenerate calibration: use points spanning a plane.")
    _, _, vh = np.linalg.svd(design, full_matrices=True)
    matrix = np.linalg.inv(target_transform) @ vh[-1].reshape(3, 3) @ source_transform
    norm = np.linalg.norm(matrix)
    if not np.isfinite(matrix).all() or norm == 0 or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Could not compute an invertible homography.")
    return matrix / norm


def transform_point(point, matrix: np.ndarray) -> tuple[float, float]:
    projected = matrix @ np.array([point[0], point[1], 1.0])
    if not np.isfinite(projected).all() or abs(projected[2]) < 1e-12:
        raise ValueError("Point projects to infinity or contains invalid coordinates.")
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def ground_position(bbox_xywh, matrix: np.ndarray) -> tuple[float, float]:
    """Use the bounding box bottom center as the ground contact estimate."""
    x, y, _, height = bbox_xywh
    return transform_point((x, y + height / 2), matrix)
