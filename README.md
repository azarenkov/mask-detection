# MaskVision AI — Face Mask Detection & Segmentation

Gradio web app for Assignment 2: YOLO11s (fine-tuned) + MedSAM2 + SadTalker.

## Run locally

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python app.py          # → http://127.0.0.1:7860
```

## Tabs

| Tab | Model | What it shows |
|---|---|---|
| Image Detection | YOLO11s | Upload / webcam → boxes, `Mask` / `No Mask`, confidence, inference time |
| Segmentation | MedSAM2 (GrabCut fallback) | Mask region highlighted inside each detected face |
| PSA Video | SadTalker | `assets/psa_video.mp4` (see `psa/sadtalker_colab.md`) |
| Performance | YOLO11s | Validation / independent-test metrics, training curves, PR curve, confusion matrix |
| About | — | Project description, comparison with Assignment 1 |

## Enabling the real MedSAM2

Without extra setup the Segmentation tab uses OpenCV GrabCut on the same box prompt.
To use MedSAM2 (GPU recommended):

```bash
git clone https://github.com/bowang-lab/MedSAM2.git && uv pip install -e ./MedSAM2
huggingface-cli download wanglab/MedSAM2 MedSAM2_latest.pt --local-dir checkpoints
export MEDSAM2_CHECKPOINT=checkpoints/MedSAM2_latest.pt
export MEDSAM2_CONFIG=configs/sam2.1_hiera_t512.yaml
python app.py
```

The tab shows which backend is active. The box prompt is the lower part of the face box
(`MASK_REGION_TOP` in `segmentation/medsam2_inference.py`), so the model outlines the mask,
not the whole face. MedSAM2 is trained on medical scans, so check the outlines visually
and compare with the fallback in the report.

## Structure

```
app.py                         Gradio UI + YOLO11s inference
models/best.pt                 trained YOLO11s weights (Colab run)
segmentation/medsam2_inference.py
assets/results/                metrics + plots from training
assets/psa_video.mp4           SadTalker video (add it yourself)
assets/about.md                About tab text
psa/sadtalker_colab.md         how to generate the PSA video
```

## Deploy to Hugging Face Spaces

Create a Gradio Space, push this folder (track `models/best.pt` with Git LFS). The free CPU
tier runs detection + GrabCut; MedSAM2 needs a GPU Space.
