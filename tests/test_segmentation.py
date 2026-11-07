"""Failure-path regressions; these tests do not download or load model weights."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from segmentation.medsam2_inference import MedSAM2Error, MedSAM2Segmenter, SegmentationResult


class SegmenterTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.boxes = [np.array([2, 12, 28, 30], dtype=np.float32)]

    def test_missing_configuration_cannot_produce_a_fallback_result(self):
        with patch.dict(os.environ, {}, clear=True):
            segmenter = MedSAM2Segmenter()
        self.assertFalse(segmenter.available)
        self.assertIn("MEDSAM2_CHECKPOINT", segmenter.load_error)
        with self.assertRaises(MedSAM2Error):
            segmenter.segment(self.image, self.boxes)

    def test_missing_checkpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MEDSAM2_CHECKPOINT": str(Path(tmp) / "missing.pt")}):
                with self.assertLogs("segmentation.medsam2_inference", level="ERROR"):
                    segmenter = MedSAM2Segmenter()
        self.assertFalse(segmenter.available)
        self.assertIn("FileNotFoundError", segmenter.load_error)

    def test_model_build_failure_is_not_a_successful_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            with patch("sam2.build_sam.build_sam2", side_effect=RuntimeError("invalid checkpoint")):
                with patch.dict(os.environ, {"MEDSAM2_CHECKPOINT": checkpoint.name}):
                    with self.assertLogs("segmentation.medsam2_inference", level="ERROR"):
                        segmenter = MedSAM2Segmenter()
        self.assertFalse(segmenter.available)
        self.assertEqual(segmenter.backend, "MedSAM2 unavailable")
        self.assertIn("invalid checkpoint", segmenter.load_error)

    def test_predict_failure_clears_predictor_state_and_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            segmenter = MedSAM2Segmenter()
        predictor = Mock()
        predictor.predict.side_effect = RuntimeError("CUDA out of memory")
        segmenter.predictor = predictor
        with self.assertLogs("segmentation.medsam2_inference", level="ERROR"):
            with self.assertRaisesRegex(MedSAM2Error, "CUDA out of memory"):
                segmenter.segment(self.image, self.boxes)
        predictor.reset_predictor.assert_called_once()

    def test_empty_boxes_do_not_call_the_predictor(self):
        with patch.dict(os.environ, {}, clear=True):
            segmenter = MedSAM2Segmenter()
        segmenter.predictor = Mock()
        with self.assertRaisesRegex(MedSAM2Error, "No YOLO target boxes"):
            segmenter.segment(self.image, [])
        segmenter.predictor.set_image.assert_not_called()


class SegmentationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"MEDSAM2_CHECKPOINT": ""}), patch("ultralytics.YOLO"):
            cls.app = importlib.import_module("app")

    def setUp(self):
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.face = {"box": np.array([2, 2, 28, 30]), "class": "no_mask", "conf": 0.9}

    def test_no_yolo_detections_explains_why_segmentation_did_not_run(self):
        with patch.object(self.app, "detect", return_value=([], 0, 0)):
            with patch.object(self.app.segmenter, "segment") as segment:
                _, stats, note, rows = self.app.run_segmentation(self.image, 0.25, False)
        segment.assert_not_called()
        self.assertIn("YOLO did not detect any faces", note)
        self.assertIn("Not run", stats)
        self.assertEqual(rows, [])

    def test_only_no_mask_faces_explains_the_checkbox(self):
        with patch.object(self.app, "detect", return_value=([self.face], 0, 0)):
            with patch.object(self.app.segmenter, "segment") as segment:
                _, _, note, rows = self.app.run_segmentation(self.image, 0.25, False)
        segment.assert_not_called()
        self.assertIn("enable the 'No Mask' option", note)
        self.assertEqual(rows, [])

    def test_inference_failure_clears_previous_success_outputs(self):
        with patch.object(self.app, "detect", return_value=([self.face], 0, 0)):
            with patch.object(self.app.segmenter, "segment", side_effect=MedSAM2Error("model failed")):
                overlay, stats, note, rows = self.app.run_segmentation(self.image, 0.25, True)
        self.assertIsNone(overlay)
        self.assertIn("Not run", stats)
        self.assertIn("Segmentation failed", note)
        self.assertIn("model failed", note)
        self.assertEqual(rows, [])

    def test_cleared_upload_resets_result(self):
        overlay, stats, note, rows = self.app.run_segmentation(None, 0.25, False)
        self.assertIsNone(overlay)
        self.assertIn("Not run", stats)
        self.assertIn("Upload an image", note)
        self.assertEqual(rows, [])

    def test_no_mask_checkbox_passes_the_face_to_the_model(self):
        mask = np.ones(self.image.shape[:2], dtype=bool)
        result = SegmentationResult(self.image, [mask], "MedSAM2 (cpu)")
        with patch.object(self.app, "detect", return_value=([self.face], 0, 0)):
            with patch.object(self.app.segmenter, "segment", return_value=result) as segment:
                overlay, _, note, rows = self.app.run_segmentation(self.image, 0.25, True)
        segment.assert_called_once()
        self.assertIsNotNone(overlay)
        self.assertIn("inference completed", note)
        self.assertEqual(rows[0][2], 32 * 32)

    def test_empty_prediction_is_not_counted_as_a_segmented_region(self):
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        result = SegmentationResult(self.image, [mask], "MedSAM2 (cpu)")
        with patch.object(self.app, "detect", return_value=([self.face], 0, 0)):
            with patch.object(self.app.segmenter, "segment", return_value=result):
                _, stats, note, rows = self.app.run_segmentation(self.image, 0.25, True)
        self.assertIn("empty masks", note)
        self.assertIn('stat-value">0<', stats)
        self.assertEqual(rows[0][2], 0)


if __name__ == "__main__":
    unittest.main()
