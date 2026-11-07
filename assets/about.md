### About the project

**MaskVision AI** is the web demo for *Assignment 2 — Face Mask Detection*.

| Component | Model | Role |
|---|---|---|
| Detection | YOLO11s (fine-tuned, `models/best.pt`) | Finds faces and classifies them as `mask` / `no_mask` |
| Segmentation | MedSAM2 (box-prompted) | Predicts the mask region inside the detected face; reports loading/inference errors explicitly |

**Pipeline:** YOLO11s detects faces in the uploaded image or webcam photo. MedSAM2 uses each lower-face box to predict a mask, which is displayed as an overlay.

#### Comparison with Assignment 1

| | Assignment 1 | Assignment 2 (YOLO11s) |
|---|---|---|
| Model | _fill in_ | YOLO11s |
| Test mAP@50 | _fill in_ | 0.956 |
| Test precision / recall | _fill in_ | 0.864 / 0.970 |

#### Ethics
All participants gave consent for their photos. Faces from the test participants
were never used for training.
