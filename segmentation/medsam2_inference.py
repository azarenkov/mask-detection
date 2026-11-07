"""Box-prompted segmentation of the face-mask region.

YOLO11s gives a face box; we shrink it to the lower part of the face
(nose → chin, where a medical mask sits) and use that as the box prompt.

Requires the official MedSAM2 package and MEDSAM2_CHECKPOINT pointing to
MedSAM2_latest.pt from https://huggingface.co/wanglab/MedSAM2.
Loading or inference errors are reported explicitly; there is no fallback model.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

# Fraction of the face box (from the top) where the mask region starts.
MASK_REGION_TOP = 0.42
LOGGER = logging.getLogger(__name__)


class MedSAM2Error(RuntimeError):
    """MedSAM2 could not load or complete inference."""


@dataclass
class SegmentationResult:
    overlay: np.ndarray          # RGB image with masks drawn
    masks: list[np.ndarray]      # one boolean HxW mask per prompted box
    backend: str


def mask_region_box(face_box, image_shape) -> np.ndarray:
    """Lower part of a face box, where a medical mask is expected."""
    x1, y1, x2, y2 = face_box
    h, w = image_shape[:2]
    top = y1 + (y2 - y1) * MASK_REGION_TOP
    pad = (x2 - x1) * 0.03
    box = np.array([x1 - pad, top, x2 + pad, y2 + pad])
    box[[0, 2]] = box[[0, 2]].clip(0, w - 1)
    box[[1, 3]] = box[[1, 3]].clip(0, h - 1)
    return box.astype(np.float32)


class MedSAM2Segmenter:
    def __init__(self):
        self.predictor = None
        self.backend = "MedSAM2 unavailable"
        self.load_error = None
        # SAM2ImagePredictor stores image embeddings between set_image/predict.
        self._lock = Lock()
        ckpt = os.environ.get("MEDSAM2_CHECKPOINT")
        if ckpt:
            self._try_load_medsam2(ckpt, os.environ.get("MEDSAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml"))
        else:
            self.load_error = "MEDSAM2_CHECKPOINT is not set. Start the app with start.ps1 or configure the checkpoint path."

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def _try_load_medsam2(self, ckpt: str, config: str):
        try:
            checkpoint = Path(ckpt).expanduser()
            if not checkpoint.is_absolute():
                raise ValueError("MEDSAM2_CHECKPOINT must be an absolute path")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"MedSAM2 checkpoint not found: {checkpoint}")
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = os.environ.get("MASKVISION_DEVICE", "cpu").strip().lower()
            if device not in {"cpu", "cuda", "mps"}:
                raise ValueError("MASKVISION_DEVICE must be cpu, cuda or mps")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is unavailable")
            model = build_sam2(config, str(checkpoint), device=device)
            self.predictor = SAM2ImagePredictor(model)
            self.backend = f"MedSAM2 ({device})"
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("MedSAM2 failed to load; segmentation is disabled")

    def _segment_medsam2(self, image: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        with self._lock, torch.inference_mode():
            try:
                self.predictor.set_image(image)
                masks = []
                for box in boxes:
                    pred, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
                    if pred.shape != (1, *image.shape[:2]) or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid MedSAM2 mask shape or values: {pred.shape}")
                    masks.append(pred[0].astype(bool))
                return masks
            finally:
                self.predictor.reset_predictor()

    def segment(self, image: np.ndarray, boxes: list[np.ndarray]) -> SegmentationResult:
        if not boxes:
            raise MedSAM2Error("No YOLO target boxes; segmentation was not run.")
        if not self.available:
            raise MedSAM2Error(f"MedSAM2 is unavailable: {self.load_error}")
        try:
            masks = self._segment_medsam2(image, boxes)
        except Exception as exc:
            LOGGER.exception("MedSAM2 inference failed")
            raise MedSAM2Error(f"MedSAM2 inference failed: {type(exc).__name__}: {exc}") from exc
        return SegmentationResult(draw_masks(image, masks, boxes), masks, self.backend)


def draw_masks(image: np.ndarray, masks: list[np.ndarray], boxes: list[np.ndarray]) -> np.ndarray:
    out = image.copy()
    color = np.array([0, 200, 255], np.uint8)
    for mask, box in zip(masks, boxes):
        out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)
    return out
