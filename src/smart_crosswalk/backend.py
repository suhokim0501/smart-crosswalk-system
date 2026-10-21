"""Keep the optional Ultralytics dependency behind one small adapter."""

from pathlib import Path

import numpy as np

from .config import AppConfig
from .types import Detection


def _label_key(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


class YoloBackend:
    """One fresh instance per video, so persistent tracker IDs cannot leak between runs."""

    def __init__(self, config: AppConfig):
        for path in (config.detector, config.classifier):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Local model file not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Video analysis requires: pip install -e ".[vision]"') from exc
        self.config = config
        self.detector = YOLO(str(config.detector), task="detect")
        self.classifier = YOLO(str(config.classifier), task="classify")
        names = self.classifier.names
        actual = [names[index] for index in range(len(names))]
        expected = config.classification.model_class_names
        if [_label_key(n) for n in actual] != [_label_key(n) for n in expected]:
            raise ValueError(f"Classifier labels {actual} differ from configured labels {expected}.")

    def track(self, image: np.ndarray) -> list[Detection]:
        """Track exactly the resized frame that will later supply classifier crops."""
        results = self.detector.track(
            source=image,
            persist=True,
            tracker=self.config.tracking.tracker,
            device=self.config.tracking.device,
            verbose=False,
            save=False,
        )
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []
        boxes = results[0].boxes
        identifiers = boxes.id.cpu().numpy().astype(int)
        coordinates = boxes.xywh.cpu().numpy()
        return [Detection(int(track_id), tuple(float(v) for v in bbox))
                for track_id, bbox in zip(identifiers, coordinates)]

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        results = self.classifier.predict(
            source=crops,
            device=self.config.tracking.device,
            verbose=False,
            save=False,
        )
        if any(result.probs is None for result in results):
            raise ValueError("Classification model did not return probabilities.")
        return np.asarray([result.probs.data.cpu().numpy() for result in results])
