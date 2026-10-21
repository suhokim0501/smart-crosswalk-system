"""Plain data shared by the vision, motion and decision stages."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Observation:
    frame_index: int
    timestamp_s: float
    bbox_xywh: tuple[float, float, float, float]
    ground_xy_cm: tuple[float, float]


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackDecision:
    track_id: int
    class_name: str | None
    speed_cm_s: float | None
    required_extension_s: float | None
    reason: str


@dataclass(frozen=True)
class CrossingDecision:
    decision_time_s: float
    remaining_green_s: float
    additional_time_s: int
    controlling_track_id: int | None
    tracks: tuple[TrackDecision, ...]

    def to_dict(self) -> dict:
        return asdict(self)


Tracks = dict[int, list[Observation]]
