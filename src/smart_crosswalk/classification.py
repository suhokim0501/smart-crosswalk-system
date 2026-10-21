"""Uniform crop sampling and the prototype's person-threshold decision rule."""

from collections.abc import Callable, Sequence
import numpy as np

from .config import ClassificationConfig
from .types import Tracks
from .video import crop_bbox


def sample_evenly(items: Sequence, count: int) -> list:
    """Use distinct samples when fewer than count are available."""
    if count < 1:
        raise ValueError("Sample count must be positive.")
    if not items:
        return []
    indices = np.linspace(0, len(items) - 1, min(count, len(items)), dtype=int)
    return [items[int(index)] for index in indices]


def choose_class(probabilities, config: ClassificationConfig) -> int:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(config.class_names),) or not np.isfinite(probabilities).all():
        raise ValueError("Classifier probability count must match configured labels.")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError("Expected normalized class probabilities in [0, 1].")
    person = config.person_class_id
    if probabilities[person] > config.person_threshold:
        return person
    candidates = probabilities.copy()
    candidates[person] = -np.inf
    return int(np.argmax(candidates))


def classify_tracks(tracks: Tracks, frames: dict[int, np.ndarray], predict: Callable,
                    config: ClassificationConfig) -> dict[int, np.ndarray]:
    """Batch sampled crops and average probabilities per track, without globals."""
    averages = {}
    for track_id, observations in tracks.items():
        crops = []
        for observation in observations:
            image = frames.get(observation.frame_index)
            if image is not None:
                crop = crop_bbox(image, observation.bbox_xywh)
                if crop is not None:
                    crops.append(crop)
        samples = sample_evenly(crops, config.samples_per_track)
        if not samples:
            continue
        rows = []
        for offset in range(0, len(samples), config.batch_size):
            batch = samples[offset:offset + config.batch_size]
            prediction = np.asarray(predict(batch), dtype=float)
            if prediction.shape != (len(batch), len(config.class_names)):
                raise ValueError("Classifier returned an unexpected batch or label count.")
            for row in prediction:
                choose_class(row, config)
            rows.extend(prediction)
        averages[track_id] = np.mean(rows, axis=0)
    return averages
