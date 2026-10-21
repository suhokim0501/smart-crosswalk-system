"""Estimate remaining crossing time and a whole-second signal extension."""

import math

from .classification import choose_class
from .config import ClassificationConfig, CrosswalkConfig, TrackingConfig
from .tracking import estimate_speed_cm_s
from .types import CrossingDecision, TrackDecision, Tracks


def estimate_extension_s(observations, speed_cm_s: float, decision_time_s: float,
                         green_time_s: float) -> float | None:
    """Extrapolate the last observed y to decision time; never search backwards forever."""
    if not (math.isfinite(speed_cm_s) and speed_cm_s > 0):
        return None
    if not (math.isfinite(decision_time_s) and math.isfinite(green_time_s)
            and 0 <= decision_time_s < green_time_s):
        raise ValueError("Require 0 <= decision_time_s < green_time_s.")
    past = [o for o in observations if o.timestamp_s <= decision_time_s]
    if not past:
        return None
    latest = max(past, key=lambda o: o.timestamp_s)
    distance = max(0.0, latest.ground_xy_cm[1] - speed_cm_s * (decision_time_s - latest.timestamp_s))
    return max(0.0, distance / speed_cm_s - (green_time_s - decision_time_s))


def evaluate_tracks(tracks: Tracks, probabilities: dict, crosswalk: CrosswalkConfig,
                    tracking: TrackingConfig, classification: ClassificationConfig) -> CrossingDecision:
    """Report explicit rejection reasons as well as eligible crossing estimates."""
    decisions = []
    for track_id in sorted(tracks):
        # Future observations must not affect either speed or the current position.
        observations = [o for o in tracks[track_id] if o.timestamp_s <= crosswalk.decision_time_s]
        speed = estimate_speed_cm_s(observations)
        label = None
        extension = None
        if not observations:
            reason = "no_observation_at_decision"
        elif crosswalk.decision_time_s - max(o.timestamp_s for o in observations) > tracking.max_observation_age_s:
            reason = "stale_track"
        elif speed is None or not tracking.min_speed_cm_s <= speed <= tracking.max_speed_cm_s:
            reason = "invalid_speed"
        elif track_id not in probabilities:
            reason = "no_classification"
        else:
            class_id = choose_class(probabilities[track_id], classification)
            label = classification.class_names[class_id]
            if class_id == classification.person_class_id:
                reason = "person_class_excluded"
            else:
                extension = estimate_extension_s(observations, speed, crosswalk.decision_time_s,
                                                  crosswalk.green_time_s)
                reason = "eligible"
        decisions.append(TrackDecision(track_id, label, speed, extension, reason))
    eligible = [d for d in decisions if d.required_extension_s is not None and d.required_extension_s > 0]
    controlling = max(eligible, key=lambda d: d.required_extension_s) if eligible else None
    # Choose the largest raw estimate BEFORE rounding; avoid changing the selected ID.
    extension = math.ceil(round(controlling.required_extension_s, 9)) if controlling else 0
    return CrossingDecision(
        decision_time_s=crosswalk.decision_time_s,
        remaining_green_s=crosswalk.green_time_s - crosswalk.decision_time_s,
        additional_time_s=extension,
        controlling_track_id=controlling.track_id if controlling else None,
        tracks=tuple(decisions),
    )
