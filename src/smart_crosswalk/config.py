"""Load and validate the small, explicit TOML configuration."""

import math
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CrosswalkConfig:
    green_time_s: float
    decision_time_s: float
    calibration_size: tuple[int, int]
    frame_size: tuple[int, int]
    size_cm: tuple[float, float]
    pixel_points: tuple[tuple[float, float], ...]
    ground_points_cm: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if not (0 < self.decision_time_s < self.green_time_s < math.inf):
            raise ValueError("Require 0 < decision_time_s < green_time_s.")
        for name in ("calibration_size", "frame_size", "size_cm"):
            values = getattr(self, name)
            if len(values) != 2 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} must contain two positive dimensions.")
        for name in ("calibration_size", "frame_size"):
            if any(isinstance(v, bool) or not isinstance(v, int) for v in getattr(self, name)):
                raise ValueError(f"{name} must use integer pixel dimensions.")
        if len(self.pixel_points) != len(self.ground_points_cm) or len(self.pixel_points) < 4:
            raise ValueError("Calibration needs at least four matching point pairs.")
        for point in (*self.pixel_points, *self.ground_points_cm):
            if len(point) != 2 or any(not math.isfinite(v) for v in point):
                raise ValueError("Calibration points must be finite 2D coordinates.")


@dataclass(frozen=True)
class TrackingConfig:
    stride: int = 1
    device: str = "cpu"
    tracker: str = "bytetrack.yaml"
    roi_cm: tuple[float, float, float, float] = (-100.0, 600.0, 900.0, 1500.0)
    min_track_ratio: float = 0.2
    min_speed_cm_s: float = 1.0
    max_speed_cm_s: float = 120.0
    max_observation_age_s: float = 2.0

    def __post_init__(self):
        if isinstance(self.stride, bool) or not isinstance(self.stride, int) or self.stride < 1:
            raise ValueError("stride must be a positive integer.")
        if not 0 < self.min_track_ratio <= 1:
            raise ValueError("min_track_ratio must be in (0, 1].")
        if not 0 < self.min_speed_cm_s <= self.max_speed_cm_s < math.inf:
            raise ValueError("Require 0 < min_speed_cm_s <= max_speed_cm_s.")
        if not 0 <= self.max_observation_age_s < math.inf:
            raise ValueError("max_observation_age_s must be finite and nonnegative.")
        if len(self.roi_cm) != 4 or not all(math.isfinite(v) for v in self.roi_cm):
            raise ValueError("roi_cm must contain four finite coordinates.")
        x1, y1, x2, y2 = self.roi_cm
        if x1 >= x2 or y1 >= y2:
            raise ValueError("roi_cm must be [min_x, min_y, max_x, max_y].")
        if not self.device or not self.tracker:
            raise ValueError("device and tracker must be nonempty.")


@dataclass(frozen=True)
class ClassificationConfig:
    samples_per_track: int = 10
    batch_size: int = 32
    model_class_names: tuple[str, ...] = ("0", "1", "2", "3", "4")
    class_names: tuple[str, ...] = ("crutches", "person", "walking frame", "walking stick", "wheelchair")
    person_class_id: int = 1
    person_threshold: float = 0.9

    def __post_init__(self):
        for name in ("samples_per_track", "batch_size", "person_class_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.samples_per_track < 1 or self.batch_size < 1:
            raise ValueError("Classification sample count and batch size must be positive.")
        if not self.class_names or any(not isinstance(n, str) or not n for n in self.class_names):
            raise ValueError("class_names must contain nonempty strings.")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique.")
        if (len(self.model_class_names) != len(self.class_names)
                or any(not isinstance(name, str) or not name for name in self.model_class_names)
                or len(set(self.model_class_names)) != len(self.model_class_names)):
            raise ValueError("model_class_names must provide one unique raw name per semantic class.")
        if not 0 <= self.person_class_id < len(self.class_names) or len(self.class_names) < 2:
            raise ValueError("person_class_id must index one of at least two classes.")
        if not 0 <= self.person_threshold <= 1:
            raise ValueError("person_threshold must be in [0, 1].")


@dataclass(frozen=True)
class AppConfig:
    video: Path
    detector: Path
    classifier: Path
    crosswalk: CrosswalkConfig
    tracking: TrackingConfig
    classification: ClassificationConfig


def load_config(path: Path) -> AppConfig:
    """Resolve asset paths relative to the config and reject misspelled fields."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    allowed = {"paths", "crosswalk", "tracking", "classification"}
    if set(data) - allowed or not {"paths", "crosswalk"} <= set(data):
        raise ValueError("Expected paths/crosswalk and optional tracking/classification tables.")
    paths = data["paths"]
    if set(paths) != {"video", "detector", "classifier"}:
        raise ValueError("paths must define video, detector and classifier.")
    if any(not isinstance(v, str) or not v.strip() for v in paths.values()):
        raise ValueError("Asset paths must be nonempty strings.")
    crosswalk = dict(data["crosswalk"])
    for name in ("calibration_size", "frame_size", "size_cm"):
        if name in crosswalk:
            crosswalk[name] = tuple(crosswalk[name])
    for name in ("pixel_points", "ground_points_cm"):
        if name in crosswalk:
            crosswalk[name] = tuple(tuple(point) for point in crosswalk[name])
    tracking = dict(data.get("tracking", {}))
    if "roi_cm" in tracking:
        tracking["roi_cm"] = tuple(tracking["roi_cm"])
    classification = dict(data.get("classification", {}))
    for name in ("class_names", "model_class_names"):
        if name in classification:
            classification[name] = tuple(classification[name])
    try:
        return AppConfig(
            **{key: (path.parent / value).resolve() for key, value in paths.items()},
            crosswalk=CrosswalkConfig(**crosswalk),
            tracking=TrackingConfig(**tracking),
            classification=ClassificationConfig(**classification),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid configuration field or type: {exc}") from exc
