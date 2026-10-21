"""Exercise the full pipeline with known coordinates and a fake vision backend."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from smart_crosswalk.config import load_config
from smart_crosswalk.pipeline import run_analysis
from smart_crosswalk.types import Detection
from smart_crosswalk.video import VideoMetadata, iter_frames

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, empty=False):
        self.index = 0
        self.empty = empty
        self.predictions = 0

    def track(self, image):
        # Ground mapping in tests is identity: bbox bottom is y=1250-50*t.
        timestamp = float(image[0, 0, 0])
        self.index += 1
        return [] if self.empty else [Detection(7, (400, 1240 - 50*timestamp, 10, 20))]

    def predict(self, crops):
        self.predictions += 1
        return np.tile([0.01, 0.01, 0.01, 0.01, 0.96], (len(crops), 1))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        video = Path(self.directory.name) / "local.mp4"
        video.touch()
        config = load_config(ROOT / "configs/crosswalk.example.toml")
        points = ((0, 0), (1600, 0), (1600, 1600), (0, 1600))
        crosswalk = replace(config.crosswalk, frame_size=(1600, 1600), calibration_size=(1600, 1600),
                            pixel_points=points, ground_points_cm=points)
        self.config = replace(config, video=video, crosswalk=crosswalk)
        self.metadata = VideoMetadata(4, 1600, 1600, 84)

    def frames(self, *args):
        # Tiny read-only broadcast arrays preserve coordinates without large allocations.
        return iter((i, float(i), np.broadcast_to(np.array([[[i]*3]], dtype=np.uint8), (1600, 1600, 3)))
                    for i in range(11))

    def test_full_pipeline_controls_known_slow_track(self):
        with patch("smart_crosswalk.pipeline.inspect_video", return_value=self.metadata), \
             patch("smart_crosswalk.pipeline.iter_frames", side_effect=self.frames):
            report = run_analysis(self.config, FakeBackend())
        self.assertEqual(report["processed_frames"], 11)
        self.assertEqual(report["decision"]["additional_time_s"], 4)
        self.assertEqual(report["decision"]["controlling_track_id"], 7)
        self.assertEqual(report["warnings"], [])

    def test_no_detections_skips_classifier(self):
        backend = FakeBackend(empty=True)
        with patch("smart_crosswalk.pipeline.inspect_video", return_value=self.metadata), \
             patch("smart_crosswalk.pipeline.iter_frames", side_effect=self.frames):
            report = run_analysis(self.config, backend)
        self.assertEqual(backend.predictions, 0)
        self.assertEqual(report["decision"]["additional_time_s"], 0)
        self.assertTrue(report["warnings"])

    def test_missing_video_fails_before_loading_models(self):
        with patch("smart_crosswalk.pipeline.YoloBackend") as loader, self.assertRaises(FileNotFoundError):
            run_analysis(replace(self.config, video=Path(self.directory.name) / "missing.mp4"))
        loader.assert_not_called()

    def test_short_video_fails_before_loading_models(self):
        with patch("smart_crosswalk.pipeline.inspect_video", return_value=replace(self.metadata, frame_count=20)), \
             patch("smart_crosswalk.pipeline.YoloBackend") as loader, self.assertRaises(ValueError):
            run_analysis(self.config)
        loader.assert_not_called()

    def test_truncated_decode_fails_instead_of_reporting_success(self):
        frames = list(self.frames())[:3]
        with patch("smart_crosswalk.pipeline.inspect_video", return_value=self.metadata), \
             patch("smart_crosswalk.pipeline.iter_frames", return_value=iter(frames)), self.assertRaises(ValueError):
            run_analysis(self.config, FakeBackend())


class FakeCapture:
    def __init__(self):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index == 7:
            return False, None
        frame = np.full((2, 2, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class VideoSamplingTests(unittest.TestCase):
    def test_stride_timestamp_and_eof_terminate_consistently(self):
        from types import SimpleNamespace
        capture = FakeCapture()
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, resize=lambda frame, _: frame)
        with patch("smart_crosswalk.video.require_opencv", return_value=cv2):
            frames = list(iter_frames(Path("fake.mp4"), VideoMetadata(4, 2, 2, 7), (2, 2), 2, 10))
        self.assertEqual([i for i, _, _ in frames], [0, 2, 4, 6])
        self.assertEqual([t for _, t, _ in frames], [0, 0.5, 1.0, 1.5])
        self.assertTrue(capture.released)

    def test_no_frame_after_decision_time(self):
        from types import SimpleNamespace
        capture = FakeCapture()
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, resize=lambda frame, _: frame)
        with patch("smart_crosswalk.video.require_opencv", return_value=cv2):
            frames = list(iter_frames(Path("fake.mp4"), VideoMetadata(4, 2, 2, 7), (2, 2), 1, 0.5))
        self.assertEqual([i for i, _, _ in frames], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
