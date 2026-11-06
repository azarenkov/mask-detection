"""MaskVision AI — Face Mask Detection & Segmentation (Assignment 2).

Run:  python app.py   →  http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

from segmentation.medsam2_inference import MedSAM2Segmenter, mask_region_box

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best.pt"
RESULTS_DIR = ROOT / "assets" / "results"
PSA_VIDEO = ROOT / "assets" / "psa_video.mp4"

CLASS_COLORS = {"mask": (34, 197, 94), "no_mask": (239, 68, 68)}  # RGB
CLASS_LABELS = {"mask": "Mask", "no_mask": "No Mask"}

model = YOLO(str(MODEL_PATH))
segmenter = MedSAM2Segmenter()


def detect(image: np.ndarray, conf: float):
    """Run YOLO11s and return (detections, inference_ms)."""
    start = time.perf_counter()
    # Ultralytics expects numpy images in BGR; Gradio provides RGB.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = model.predict(bgr, conf=conf, imgsz=640, agnostic_nms=True, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    detections = []
    for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(),
                               result.boxes.cls.cpu().numpy().astype(int),
                               result.boxes.conf.cpu().numpy()):
        detections.append({"box": box, "class": model.names[cls], "conf": float(score)})
    return detections, elapsed_ms, result.speed.get("inference", elapsed_ms)


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image.copy()
    thickness = max(2, round(min(image.shape[:2]) / 250))
    font_scale = max(0.6, min(image.shape[:2]) / 900)
    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(int)
        color = CLASS_COLORS[det["class"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{CLASS_LABELS[det['class']]} {det['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(y1, th + base + 4)
        cv2.rectangle(out, (x1, ty - th - base - 4), (x1 + tw + 8, ty), color, -1)
        cv2.putText(out, label, (x1 + 4, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def stat_card(title: str, value: str, hint: str = "") -> str:
    return f'<div class="stat"><div class="stat-title">{title}</div><div class="stat-value">{value}</div><div class="stat-hint">{hint}</div></div>'


def stats_html(detections, total_ms, model_ms) -> str:
    n_mask = sum(d["class"] == "mask" for d in detections)
    n_no = len(detections) - n_mask
    return '<div class="stats">' + "".join([
        stat_card("Detected faces", str(len(detections))),
        stat_card("With mask", str(n_mask), "class: mask"),
        stat_card("Without mask", str(n_no), "class: no_mask"),
        stat_card("Inference time", f"{model_ms:.0f} ms", f"end-to-end {total_ms:.0f} ms"),
    ]) + "</div>"


def run_detection(image, conf):
    if image is None:
        raise gr.Error("Upload an image or take a photo with the webcam first.")
    image = image[..., :3]
    detections, total_ms, model_ms = detect(image, conf)
    table = [[i + 1, CLASS_LABELS[d["class"]], round(d["conf"], 3),
              ", ".join(str(int(v)) for v in d["box"])] for i, d in enumerate(detections)]
    return draw_detections(image, detections), stats_html(detections, total_ms, model_ms), table


def run_segmentation(image, conf, include_no_mask):
    if image is None:
        raise gr.Error("Upload an image or take a photo with the webcam first.")
    image = image[..., :3]
    detections, _, _ = detect(image, conf)
    targets = [d for d in detections if d["class"] == "mask" or include_no_mask]
    boxes = [mask_region_box(d["box"], image.shape) for d in targets]
    start = time.perf_counter()
    seg = segmenter.segment(image, boxes)
    seg_ms = (time.perf_counter() - start) * 1000

    rows = []
    for d, m in zip(targets, seg.masks):
        x1, y1, x2, y2 = d["box"]
        face_area = max((x2 - x1) * (y2 - y1), 1)
        rows.append([CLASS_LABELS[d["class"]], round(d["conf"], 3), int(m.sum()),
                     f"{100 * m.sum() / face_area:.1f}%"])

    backend_note = f"**Backend:** {seg.backend}"
    if segmenter.load_error:
        backend_note += f"  \n⚠️ MedSAM2 could not be loaded, using fallback: `{segmenter.load_error}`"
    if not targets:
        backend_note += "  \nNo faces with a mask were found — nothing to segment."
    html = '<div class="stats">' + stat_card("Segmented regions", str(len(seg.masks))) + \
        stat_card("Segmentation time", f"{seg_ms:.0f} ms", seg.backend) + "</div>"
    return draw_detections(seg.overlay, detections), html, backend_note, rows


def load_experiment() -> dict:
    path = RESULTS_DIR / "experiment.json"
    return json.loads(path.read_text()) if path.exists() else {}


def metrics_markdown(exp: dict) -> str:
    if not exp:
        return "_experiment.json not found._"
    v, t = exp["validation"], exp["independent_test"]
    split = exp["dataset_checks"]["validation"]["splits"]
    args = exp["train_args"]
    return f"""
| Metric | Validation (person_03) | Independent test (person_04, person_05) |
|---|---|---|
| Precision | {v['precision']:.3f} | {t['precision']:.3f} |
| Recall | {v['recall']:.3f} | {t['recall']:.3f} |
| mAP@50 | {v['mAP50']:.3f} | {t['mAP50']:.3f} |
| mAP@50-95 | {v['mAP50-95']:.3f} | {t['mAP50-95']:.3f} |

**Dataset:** {exp['dataset_checks']['images_found']} images, 2 classes (`mask`, `no_mask`) —
train {split['train']['images']} / val {split['val']['images']} / test {split['test']['images']}.
Splits are made **by participant**, so test faces were never seen in training.

**Training:** YOLO11s, {args['epochs']} epochs (early-stopping patience {args['patience']}),
imgsz {args['imgsz']}, batch {args['batch']}, seed {args['seed']}, all augmentations disabled
(mosaic, flips, HSV, scale = 0) — baseline for the ablation study.
"""


CSS = """
.hero {padding: 28px 28px 20px; border-radius: 16px; margin-bottom: 8px;
  background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%); color: #f8fafc;}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -0.02em; color: #f8fafc;}
.hero p {margin: 6px 0 0; color: #99f6e4; font-size: 1.05rem;}
.hero .chips {margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap;}
.hero .chip {font-size: 0.8rem; padding: 3px 10px; border-radius: 999px;
  border: 1px solid #2dd4bf55; color: #ccfbf1;}
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;}
.stat {padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);}
.stat-title {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7;}
.stat-value {font-size: 1.6rem; font-weight: 700; margin-top: 2px;}
.stat-hint {font-size: 0.75rem; opacity: 0.6;}
"""

HERO = """
<div class="hero">
  <h1>MaskVision AI</h1>
  <p>AI-powered Face Mask Detection &amp; Segmentation</p>
  <div class="chips">
    <span class="chip">YOLO11s detection</span><span class="chip">MedSAM2 segmentation</span>
    <span class="chip">SadTalker PSA video</span><span class="chip">Assignment 2</span>
  </div>
</div>
"""

EMPTY_STATS = '<div class="stats">' + stat_card("Detected faces", "—") + stat_card("With mask", "—") + \
    stat_card("Without mask", "—") + stat_card("Inference time", "— ms") + "</div>"


def build_ui() -> gr.Blocks:
    exp = load_experiment()
    with gr.Blocks(title="MaskVision AI") as demo:
        gr.HTML(HERO)

        with gr.Tab("Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    det_in = gr.Image(label="Upload an image or use webcam", sources=["upload", "webcam", "clipboard"], type="numpy")
                    det_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                    det_btn = gr.Button("Detect masks", variant="primary")
                with gr.Column(scale=1):
                    det_out = gr.Image(label="Detection result", interactive=False)
                    det_stats = gr.HTML(EMPTY_STATS)
            det_table = gr.Dataframe(headers=["#", "Class", "Confidence", "Box (x1, y1, x2, y2)"],
                                     label="Detections", interactive=False)
            det_btn.click(run_detection, [det_in, det_conf], [det_out, det_stats, det_table])
            det_in.upload(run_detection, [det_in, det_conf], [det_out, det_stats, det_table])

        with gr.Tab("Segmentation"):
            gr.Markdown("YOLO11s finds the face → the lower part of the box (nose to chin) is passed as a "
                        "**box prompt** to the segmentation model → the mask region is highlighted.")
            with gr.Row():
                with gr.Column(scale=1):
                    seg_in = gr.Image(label="Upload an image or use webcam", sources=["upload", "webcam", "clipboard"], type="numpy")
                    seg_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                    seg_all = gr.Checkbox(False, label="Also segment the lower face of 'No Mask' detections")
                    seg_btn = gr.Button("Segment", variant="primary")
                with gr.Column(scale=1):
                    seg_out = gr.Image(label="Segmentation result", interactive=False)
                    seg_stats = gr.HTML()
                    seg_note = gr.Markdown(f"**Backend:** {segmenter.backend}")
            seg_table = gr.Dataframe(headers=["Class", "Confidence", "Mask pixels", "Share of face box"],
                                     label="Segmented regions", interactive=False)
            seg_btn.click(run_segmentation, [seg_in, seg_conf, seg_all], [seg_out, seg_stats, seg_note, seg_table])

        with gr.Tab("PSA Video"):
            gr.Markdown("### Public-service announcement\nAn educational talking-head video generated with "
                        "**SadTalker** from a single portrait (used with the participant's consent) and a recorded script.")
            if PSA_VIDEO.exists():
                gr.Video(str(PSA_VIDEO), label="PSA video", interactive=False)
            else:
                gr.Markdown("> ⚠️ `assets/psa_video.mp4` is not there yet. Generate it with SadTalker in Colab "
                            "(see `psa/sadtalker_colab.md`) and put the file into `assets/`, then restart the app.")

        with gr.Tab("Performance"):
            gr.Markdown("### YOLO11s results")
            gr.Markdown(metrics_markdown(exp))
            with gr.Row():
                for name, label in [("results.png", "Training curves"),
                                    ("test_BoxPR_curve.png", "Precision–Recall (test)"),
                                    ("test_confusion_matrix_normalized.png", "Confusion matrix (test)")]:
                    if (RESULTS_DIR / name).exists():
                        gr.Image(str(RESULTS_DIR / name), label=label, interactive=False)

        with gr.Tab("About"):
            gr.Markdown((ROOT / "assets" / "about.md").read_text() if (ROOT / "assets" / "about.md").exists() else "")
    return demo


if __name__ == "__main__":
    build_ui().launch(css=CSS, theme=gr.themes.Soft(primary_hue="teal"))
