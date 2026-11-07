# MaskVision AI — Face Mask Detection & Segmentation

Gradio web app for Assignment 2: YOLO11s (fine-tuned) + MedSAM2.

## Run locally on CPU

The environment and MedSAM2 checkpoint on this computer are already installed. Run from
the repository directory. Windows PowerShell:

```powershell
$env:MASKVISION_DEVICE = 'cpu'
$env:CUDA_VISIBLE_DEVICES = '-1'
.\start.ps1
```

Linux, after completing the installation below:

```bash
source .venv/bin/activate
export MASKVISION_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MEDSAM2_CHECKPOINT="$(pwd)/checkpoints/MedSAM2_latest.pt"
export MEDSAM2_CONFIG=configs/sam2.1_hiera_t512.yaml
python app.py
```

Open **http://127.0.0.1:7860**. Segmentation must show **Backend: MedSAM2 (cpu)**.
The CPU setting applies to both YOLO and MedSAM2; OpenMP/MKL use at most two threads.

## Tabs

| Tab | Model | What it shows |
|---|---|---|
| Image Detection | YOLO11s | Upload / webcam → boxes, `Mask` / `No Mask`, confidence, inference time |
| Segmentation | MedSAM2 | Upload → YOLO face boxes → mask prediction → overlay; explicit errors if the model is unavailable |
| Performance | YOLO11s | Validation / independent-test metrics, training curves, PR curve, confusion matrix |
| About | — | Project description, comparison with Assignment 1 |

## Which YOLO weights are active?

The site uses **A (`A_no_flip`, `fliplr=0.0`)** through `models/best.pt`.
Variant **B (`B_flip_50`, `fliplr=0.5`)** is not loaded.

The file matches the GitHub model blob `892aeee7884f94b687586e39789403baed2417e6`
at commit `9ae1238b5b3517257716a583287ca704cdcf63c0`. Its SHA256 is
`63d71d157f6b817dcc1acfd338d37b0be96ac84bc06ee85ee45e6f4490e20ea3`.
All **503 indexed tensor-storage payloads** match the ablation archive's A checkpoint;
the B payloads do not match. The A archive file has a different whole-file hash because
its checkpoint metadata differs. This identifies the learned weights without renaming
or replacing the site's model file.

`assets/results/model_provenance.json` records this comparison. Both metric columns
in `experiment.json` also match A's `comparison.csv` within 0.000001. Performance labels
the table and plots as A, and verifies the loaded checkpoint hash before describing
these results as active-model metrics. Training completed 28 epochs, with a limit of 50.

| A metrics | Validation (110 images) | Independent test (39 images) |
|---|---|---|
| Precision | 0.988 | 0.864 |
| Recall | 0.961 | 0.970 |
| mAP@50 | 0.983 | 0.956 |
| mAP@50-95 | 0.863 | 0.686 |

In raw Ultralytics CSV names such as `metrics/precision(B)`, `(B)` means **bounding boxes**;
it does not identify ablation variant B. These are YOLO detection metrics and do not
measure MedSAM2 segmentation accuracy.

To reproduce the checkpoint audit without running either model:

```powershell
git ls-remote origin HEAD
.\.venv\Scripts\python.exe -m scripts.audit_yolo_weights --archive 'C:\path\to\MaskVision_Ablation.zip' --github-commit 9ae1238b5b3517257716a583287ca704cdcf63c0
```

Supply the commit returned by the GitHub check and the actual ablation archive path.

## Enabling the real MedSAM2

Segmentation requires the official MedSAM2 package and trained weights in the **same Python
environment as the app**. There is no GrabCut fallback. Detection remains available when
MedSAM2 cannot load; the Segmentation tab displays the actual error and produces no mask.

For this Windows checkout, the application environment is `.venv` (Python 3.12).
Install from the repository directory in PowerShell:

```powershell
# For a fresh checkout, create .venv with Python 3.12 first:
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt huggingface_hub setuptools
git clone https://github.com/bowang-lab/MedSAM2.git MedSAM2
# Install the real model without its optional CUDA extension.
$env:SAM2_BUILD_CUDA = '0'
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e ./MedSAM2
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('wanglab/MedSAM2', 'MedSAM2_latest.pt', local_dir='checkpoints')"
$env:MASKVISION_DEVICE = 'cpu'
$env:CUDA_VISIBLE_DEVICES = '-1'
.\start.ps1
```

`start.ps1` uses the local `.venv` and sets these process environment variables **before**
Python starts, unless you already supplied custom values:

```powershell
$env:MEDSAM2_CHECKPOINT = Join-Path $PWD 'checkpoints\MedSAM2_latest.pt' # absolute path
$env:MEDSAM2_CONFIG = 'configs/sam2.1_hiera_t512.yaml'
```

For a fresh Linux checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt huggingface_hub setuptools
git clone https://github.com/bowang-lab/MedSAM2.git
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e ./MedSAM2
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('wanglab/MedSAM2', 'MedSAM2_latest.pt', local_dir='checkpoints')"
export MEDSAM2_CHECKPOINT="$(pwd)/checkpoints/MedSAM2_latest.pt"
export MEDSAM2_CONFIG=configs/sam2.1_hiera_t512.yaml
export MASKVISION_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python app.py
```

The app defaults to **CPU for both YOLO and MedSAM2** because GPU activity makes this
laptop unresponsive. The launcher hides CUDA from the process and limits OpenMP/MKL
to two threads. CPU inference is slower; it still uses the real MedSAM2 model.
`MASKVISION_DEVICE` selects `cpu`, `cuda` or `mps`; do not enable CUDA on this laptop
until the freezing issue is resolved.

Restart the app after installation or environment changes. Open `http://127.0.0.1:7860`
and select **Segmentation**: a loaded model shows **Backend: MedSAM2 (cpu)** by default.
Uploading a photo runs YOLO and then
MedSAM2 automatically; the **Segment** button reruns it with your chosen options.

If YOLO finds no faces, the tab explains that segmentation was not run and suggests a
clearer photo or a lower confidence threshold. If all detections are `No Mask`, enable
the corresponding checkbox to segment their lower faces. A load/inference failure clears
previous masks, rows and success counters and displays the error. Empty predictions are
reported as empty, not counted as segmented regions.

The box prompt is the lower part of the face box (`MASK_REGION_TOP` in
`segmentation/medsam2_inference.py`). MedSAM2 was trained on medical images; a box prompt
does not guarantee an accurate face-mask outline. Check predictions visually. Confidence
and coverage are not measured segmentation accuracy without ground-truth masks.

## Checks before recording a demo

The running app was checked through its Gradio API on CPU:

| Input | Detection | Segmentation behavior |
|---|---|---|
| Unmasked test photo, person_04 photo_11 | `No Mask`, confidence 0.938 | Skipped by default with a clear explanation; lower-face segmentation works when explicitly enabled |
| Wall crop of a real photo, with no faces | No detections | Skipped with the message `YOLO did not detect any faces` and an empty result table |

**Observed model error:** another unmasked test photo (person_04 photo_10) was classified
as `Mask` with confidence 0.912. As a result MedSAM2 ran on that face. This is a YOLO false
positive; the app does not independently prove that the segmented region is a physical
mask. The local check report `.local/pre_recording_checks.json` preserves both unmasked
examples and the no-face result. It is a functional check, not a new accuracy evaluation.

Regression checks: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
To run the real pipeline on the exported photo archive, set the model environment variables
and run `python -m scripts.verify_medsam2 --archive path/to/batch.zip` from the repository
directory. It saves overlays and a JSON report in `.local/verification`.

## Structure

```
app.py                         Gradio UI + YOLO11s inference
models/best.pt                 trained YOLO11s weights (Colab run)
segmentation/medsam2_inference.py
assets/results/                metrics + plots from training
assets/results/model_provenance.json   active YOLO variant, checkpoint hashes and audit evidence
assets/about.md                About tab text
```

## Deploy to Hugging Face Spaces

Create a Gradio Space, push this folder (track `models/best.pt` with Git LFS), install
MedSAM2 in its Python environment, download the checkpoint, and set the two environment
variables above. A GPU Space is recommended; CPU inference is supported but slower.
This checkout is currently intended to run locally on the Windows computer.
