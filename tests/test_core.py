"""Regression coverage for coordinate, sampling, speed and extension failures."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from smart_crosswalk.classification import choose_class, classify_tracks, sample_evenly
from smart_crosswalk.config import ClassificationConfig, TrackingConfig, load_config
from smart_crosswalk.demo import run_demo
from smart_crosswalk.geometry import compute_homography, ground_position, transform_point
from smart_crosswalk.reporting import write_report
from smart_crosswalk.timing import estimate_extension_s, evaluate_tracks
from smart_crosswalk.tracking import estimate_speed_cm_s, filter_observations, filter_short_tracks
from smart_crosswalk.types import Observation
from smart_crosswalk.video import crop_bbox

ROOT = Path(__file__).resolve().parents[1]


def observation(t, y, index=None, x=400):
    return Observation(int(t) if index is None else index, float(t), (5, 5, 4, 4), (x, y))


class GeometryTests(unittest.TestCase):
    def test_nonuniform_image_resize_and_bottom_center(self):
        source = [(0, 0), (100, 0), (100, 100), (0, 100)]
        target = [(0, 0), (1000, 0), (1000, 2000), (0, 2000)]
        matrix = compute_homography(source, target, (100, 100), (200, 50))
        np.testing.assert_allclose(transform_point((100, 25), matrix), (500, 1000), atol=1e-8)
        np.testing.assert_allclose(ground_position((100, 20, 10, 10), matrix), (500, 1000), atol=1e-8)

    def test_collinear_and_nonfinite_points_rejected(self):
        points = [(0, 0), (1, 0), (2, 0), (3, 0)]
        with self.assertRaises(ValueError):
            compute_homography(points, points, (10, 10), (10, 10))
        with self.assertRaises(ValueError):
            compute_homography([(0, 0), (1, 0), (1, 1), (np.nan, 1)], points, (10, 10), (10, 10))

    def test_four_point_projective_mapping(self):
        source = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=float)
        expected = np.array([[2, 0.2, 10], [0.1, 3, 5], [0.001, 0.002, 1]])
        target = [transform_point(point, expected) for point in source]
        matrix = compute_homography(source, target, (100, 80), (100, 80))
        np.testing.assert_allclose(transform_point((23, 44), matrix), transform_point((23, 44), expected), atol=1e-8)

    def test_original_calibration_is_nondegenerate(self):
        config = load_config(ROOT / "configs/crosswalk.example.toml").crosswalk
        matrix = compute_homography(config.pixel_points, config.ground_points_cm,
                                    config.calibration_size, config.frame_size)
        self.assertTrue(np.isfinite(matrix).all())


class TrackingTests(unittest.TestCase):
    def test_speed_uses_timestamps_even_when_frames_skipped(self):
        sparse = [observation(0, 1000, 0), observation(0.5, 975, 2), observation(1.5, 925, 6)]
        self.assertAlmostEqual(estimate_speed_cm_s(sparse), 50)

    def test_short_empty_and_duplicate_timestamp_tracks(self):
        self.assertIsNone(estimate_speed_cm_s([]))
        self.assertIsNone(estimate_speed_cm_s([observation(0, 1000)]))
        self.assertIsNone(estimate_speed_cm_s([observation(0, 1000), observation(0, 900)]))

    def test_filters_do_not_mutate_and_remove_empty_tracks(self):
        tracks = {1: [observation(0, 1000), observation(1, 900, x=2000)],
                  2: [observation(0, 2000)]}
        selected = filter_observations(tracks, (0, 0, 800, 1400), 360)
        self.assertEqual(len(tracks[1]), 2)
        self.assertIn(2, tracks)
        self.assertEqual(len(selected[1]), 1)
        self.assertNotIn(2, selected)

    def test_min_track_ratio_applies_to_processed_frames_once(self):
        tracks = {1: [observation(i, 1000 - i) for i in range(3)]}
        self.assertEqual(filter_short_tracks(tracks, 20, 0.2), {})
        self.assertIn(1, filter_short_tracks(tracks, 10, 0.2))


class ClassificationTests(unittest.TestCase):
    def test_clipped_crop_cannot_wrap_negative_indices(self):
        image = np.ones((10, 10, 3), dtype=np.uint8)
        self.assertEqual(crop_bbox(image, (0, 0, 6, 6)).shape, (3, 3, 3))
        self.assertIsNone(crop_bbox(image, (-20, -20, 2, 2)))
        self.assertIsNone(crop_bbox(image, (5, 5, 0, 2)))

    def test_even_samples_handle_empty_and_short_tracks(self):
        self.assertEqual(sample_evenly([], 10), [])
        self.assertEqual(sample_evenly([1, 2], 10), [1, 2])
        self.assertEqual(sample_evenly(list(range(20)), 3), [0, 9, 19])

    def test_person_threshold_preserves_original_rule(self):
        config = ClassificationConfig()
        self.assertEqual(choose_class([0.01, 0.96, 0.01, 0.01, 0.01], config), 1)
        self.assertEqual(choose_class([0.01, 0.8, 0.01, 0.01, 0.17], config), 4)

    def test_bad_probabilities_fail_explicitly(self):
        for row in ([0.5, 0.5], [float("nan")] * 5, [1] * 5, [-0.1, 0.6, 0.2, 0.2, 0.1]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                choose_class(row, ClassificationConfig())

    def test_batch_predictions_keep_the_right_track(self):
        frames = {i: np.ones((10, 10, 3), dtype=np.uint8) for i in range(4)}
        tracks = {8: [observation(i, 1000) for i in range(4)], 9: []}
        batch_sizes = []
        def predict(batch):
            batch_sizes.append(len(batch))
            return np.tile([0.01, 0.01, 0.01, 0.01, 0.96], (len(batch), 1))
        averages = classify_tracks(tracks, frames, predict, ClassificationConfig(batch_size=2))
        self.assertEqual(batch_sizes, [2, 2])
        self.assertEqual(set(averages), {8})
        self.assertEqual(choose_class(averages[8], ClassificationConfig()), 4)

    def test_empty_tracks_never_call_model(self):
        def unexpected(_):
            self.fail("Model should not run for empty tracks")
        self.assertEqual(classify_tracks({}, {}, unexpected, ClassificationConfig()), {})


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/crosswalk.example.toml")
        self.wheelchair = np.array([0.01, 0.01, 0.01, 0.01, 0.96])

    def test_demo_extension_and_exclusions(self):
        report = run_demo()
        decision = report["decision"]
        self.assertEqual(decision["additional_time_s"], 4)
        self.assertEqual(decision["controlling_track_id"], 1)
        self.assertEqual([t["reason"] for t in decision["tracks"]],
                         ["eligible", "person_class_excluded", "invalid_speed"])

    def test_empty_tracks_have_no_extension(self):
        decision = evaluate_tracks({}, {}, self.config.crosswalk, self.config.tracking, self.config.classification)
        self.assertEqual(decision.additional_time_s, 0)
        self.assertIsNone(decision.controlling_track_id)

    def test_no_past_observation_does_not_loop(self):
        self.assertIsNone(estimate_extension_s([observation(12, 1000)], 50, 10, 21))
        self.assertIsNone(estimate_extension_s([], 50, 10, 21))
        self.assertIsNone(estimate_extension_s([observation(9, 1000)], 0, 10, 21))

    def test_lost_frame_is_extrapolated(self):
        self.assertAlmostEqual(estimate_extension_s([observation(9, 800)], 50, 10, 21), 4)

    def test_future_data_cannot_change_velocity(self):
        tracks = {1: [observation(0, 1250), observation(10, 750), observation(11, -999999)]}
        decision = evaluate_tracks(tracks, {1: self.wheelchair}, self.config.crosswalk,
                                    self.config.tracking, self.config.classification)
        self.assertEqual(decision.additional_time_s, 4)
        self.assertAlmostEqual(decision.tracks[0].speed_cm_s, 50)

    def test_stale_and_reverse_tracks_are_explicit(self):
        tracks = {1: [observation(0, 1000), observation(5, 750)],
                  2: [observation(0, 750), observation(10, 1000)]}
        decision = evaluate_tracks(tracks, {1: self.wheelchair, 2: self.wheelchair},
                                    self.config.crosswalk, self.config.tracking, self.config.classification)
        self.assertEqual([d.reason for d in decision.tracks], ["stale_track", "invalid_speed"])

    def test_maximum_is_selected_before_rounding(self):
        tracks = {1: [observation(0, 1210), observation(10, 710)],
                  2: [observation(0, 1230), observation(10, 730)]}
        decision = evaluate_tracks(tracks, {1: self.wheelchair, 2: self.wheelchair},
                                    self.config.crosswalk, self.config.tracking, self.config.classification)
        self.assertEqual(decision.additional_time_s, 4)
        self.assertEqual(decision.controlling_track_id, 2)

    def test_already_crossed_needs_no_extension(self):
        self.assertEqual(estimate_extension_s([observation(9, 20)], 50, 10, 21), 0)


class ConfigurationAndReportTests(unittest.TestCase):
    def test_asset_paths_are_relative_to_config(self):
        config = load_config(ROOT / "configs/crosswalk.example.toml")
        self.assertEqual(config.detector, ROOT / "models/human_detect.pt")

    def test_invalid_tracking_and_classifier_configuration(self):
        for overrides in ({"stride": 0}, {"stride": 1.5}, {"min_track_ratio": 0},
                          {"max_speed_cm_s": 0}, {"roi_cm": (1, 1, 0, 0)}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                TrackingConfig(**overrides)
        with self.assertRaises(ValueError):
            ClassificationConfig(samples_per_track=0)
        with self.assertRaises(ValueError):
            ClassificationConfig(model_class_names=("0", "1"))

    def test_invalid_crosswalk_configuration(self):
        config = load_config(ROOT / "configs/crosswalk.example.toml").crosswalk
        for overrides in ({"green_time_s": 5}, {"frame_size": (0, 360)}, {"frame_size": (640.5, 360)}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(config, **overrides)

    def test_json_and_two_value_signal_file(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, signal = write_report(run_demo(), Path(directory))
            self.assertEqual(json.loads(summary.read_text())["decision"]["additional_time_s"], 4)
            self.assertEqual(signal.read_text(), "1 4\n")


if __name__ == "__main__":
    unittest.main()
