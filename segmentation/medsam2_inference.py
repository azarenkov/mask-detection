"""Box-prompted segmentation of the face-mask region.

YOLO11s gives a face box; we shrink it to the lower part of the face
(nose → chin, where a medical mask sits) and use that as the box prompt.

Backends, in order of preference:
  1. MedSAM2 (SAM2 architecture) — used when the `sam2` package is installed
     and MEDSAM2_CHECKPOINT points to a checkpoint (e.g. MedSAM2_latest.pt
     from https://huggingface.co/wanglab/MedSAM2).
  2. OpenCV GrabCut — CPU fallback so the demo always works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# Fraction of the face box (from the top) where the mask region starts.
MASK_REGION_TOP = 0.42


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
        self.backend = "GrabCut (CPU fallback)"
        self.load_error = None
        ckpt = os.environ.get("MEDSAM2_CHECKPOINT")
        if ckpt:
            self._try_load_medsam2(ckpt, os.environ.get("MEDSAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml"))

    def _try_load_medsam2(self, ckpt: str, config: str):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            model = build_sam2(config, ckpt, device=device)
            self.predictor = SAM2ImagePredictor(model)
            self.backend = f"MedSAM2 ({device})"
        except Exception as exc:  # keep the app usable without MedSAM2
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _segment_medsam2(self, image: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        self.predictor.set_image(image)
        masks = []
        for box in boxes:
            pred, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
            masks.append(pred[0].astype(bool))
        return masks

    @staticmethod
    def _segment_grabcut(image: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            if x2 - x1 < 4 or y2 - y1 < 4:
                masks.append(np.zeros(image.shape[:2], bool))
                continue
            gc_mask = np.zeros(image.shape[:2], np.uint8)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, gc_mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            masks.append(_largest_component(fg))
        return masks

    def segment(self, image: np.ndarray, boxes: list[np.ndarray]) -> SegmentationResult:
        if not boxes:
            return SegmentationResult(image.copy(), [], self.backend)
        if self.predictor is not None:
            masks = self._segment_medsam2(image, boxes)
        else:
            masks = self._segment_grabcut(image, boxes)
        return SegmentationResult(draw_masks(image, masks, boxes), masks, self.backend)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


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
