"""Deterministic synthetic tracks: no cameras, model weights or network calls."""

import numpy as np

from .config import ClassificationConfig, CrosswalkConfig, TrackingConfig
from .timing import evaluate_tracks
from .types import Observation


def run_demo() -> dict:
    crosswalk = CrosswalkConfig(
        green_time_s=21, decision_time_s=10,
        calibration_size=(640, 360), frame_size=(640, 360), size_cm=(800, 1400),
        pixel_points=((0, 0), (640, 0), (640, 360), (0, 360)),
        ground_points_cm=((0, 0), (800, 0), (800, 1400), (0, 1400)),
    )
    tracks = {}
    for track_id, initial_y, speed in ((1, 1250, 50), (2, 1200, 80), (3, 900, 0)):
        tracks[track_id] = [Observation(i, float(i), (100, 100, 20, 40), (400, initial_y - speed*i))
                            for i in range(11)]
    probabilities = {
        1: np.array([0.01, 0.01, 0.01, 0.01, 0.96]),
        2: np.array([0.01, 0.96, 0.01, 0.01, 0.01]),
        3: np.array([0.01, 0.01, 0.01, 0.01, 0.96]),
    }
    decision = evaluate_tracks(tracks, probabilities, crosswalk, TrackingConfig(), ClassificationConfig())
    return {
        "source": "synthetic_demo",
        "description": "Artificial trajectories; not model accuracy or real-world performance.",
        "units": {"distance": "cm", "speed": "cm/s", "time": "s"},
        "expected_additional_time_s": 4,
        "decision": decision.to_dict(),
    }
