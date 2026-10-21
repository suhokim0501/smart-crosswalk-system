"""Model adapter and video export contracts, using local test doubles only."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from smart_crosswalk.backend import YoloBackend
from smart_crosswalk.config import load_config
from smart_crosswalk.types import Observation, TrackDecision
from smart_crosswalk.video import VideoMetadata
from smart_crosswalk.visualization import render_video, track_color

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class BackendTests(unittest.TestCase):
    def test_adapter_converts_tensor_outputs_and_uses_persistent_ids(self):
        config = load_config(ROOT / "configs/crosswalk.example.toml")
        detector = Mock()
        classifier = Mock()
        classifier.names = dict(enumerate(config.classification.model_class_names))
        detector.track.return_value = [SimpleNamespace(boxes=SimpleNamespace(
            id=Tensor([7]), xywh=Tensor([[100, 200, 20, 40]])))]
        classifier.predict.return_value = [SimpleNamespace(probs=SimpleNamespace(
            data=Tensor([0.01, 0.01, 0.01, 0.01, 0.96])))]
        loader = Mock(side_effect=[detector, classifier])
        with patch("pathlib.Path.is_file", return_value=True), \
             patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=loader)}):
            backend = YoloBackend(config)
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            detection = backend.track(image)[0]
            self.assertEqual(detection.track_id, 7)
            self.assertEqual(detection.bbox_xywh, (100, 200, 20, 40))
            self.assertIs(detector.track.call_args.kwargs["source"], image)
            self.assertTrue(detector.track.call_args.kwargs["persist"])
            self.assertEqual(backend.predict([image]).shape, (1, 5))

    def test_wrong_label_order_is_rejected(self):
        config = load_config(ROOT / "configs/crosswalk.example.toml")
        classifier = SimpleNamespace(names=dict(enumerate(reversed(config.classification.model_class_names))))
        loader = Mock(side_effect=[Mock(), classifier])
        with patch("pathlib.Path.is_file", return_value=True), \
             patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=loader)}), \
             self.assertRaisesRegex(ValueError, "Classifier labels"):
            YoloBackend(config)


class VideoExportTests(unittest.TestCase):
    def test_video_uses_sampled_fps_and_centimeter_labels(self):
        writer = Mock()
        writer.isOpened.return_value = True
        cv2 = SimpleNamespace(VideoWriter=Mock(return_value=writer), VideoWriter_fourcc=Mock(return_value=0),
                              rectangle=Mock(), putText=Mock(), FONT_HERSHEY_SIMPLEX=0)
        observation = Observation(2, 0.5, (10, 10, 10, 10), (400, 900))
        decision = TrackDecision(7, "wheelchair", 50, 4, "eligible")
        frames = iter([(2, 0.5, np.zeros((20, 20, 3), dtype=np.uint8))])
        with tempfile.TemporaryDirectory() as directory, \
             patch("smart_crosswalk.visualization.require_opencv", return_value=cv2), \
             patch("smart_crosswalk.visualization.iter_frames", return_value=frames):
            render_video(Path("input.mp4"), Path(directory) / "annotated.mp4", VideoMetadata(4, 20, 20, 40),
                         (20, 20), 2, 10, {7: [observation]}, (decision,))
        self.assertEqual(cv2.VideoWriter.call_args.args[2], 2)
        self.assertIn("50.0 cm/s", cv2.putText.call_args.args[1])
        writer.write.assert_called_once()
        writer.release.assert_called_once()

    def test_writer_failure_releases_resource(self):
        writer = Mock()
        writer.isOpened.return_value = False
        cv2 = SimpleNamespace(VideoWriter=Mock(return_value=writer), VideoWriter_fourcc=Mock(return_value=0))
        with tempfile.TemporaryDirectory() as directory, \
             patch("smart_crosswalk.visualization.require_opencv", return_value=cv2), self.assertRaises(ValueError):
            render_video(Path("input.mp4"), Path(directory) / "out.mp4", VideoMetadata(4, 20, 20, 40),
                         (20, 20), 1, 10, {}, ())
        writer.release.assert_called_once()

    def test_color_is_stable_without_changing_global_randomness(self):
        np.random.seed(100)
        expected = np.random.random()
        np.random.seed(100)
        self.assertEqual(track_color(7), track_color(7))
        self.assertEqual(np.random.random(), expected)


if __name__ == "__main__":
    unittest.main()
