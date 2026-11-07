"""Run the real application pipeline on an exported batch of input photos.

Usage (after configuring MEDSAM2_CHECKPOINT and MEDSAM2_CONFIG):
    python -m scripts.verify_medsam2 --archive path/to/batch.zip
Outputs stay in ignored .local/verification; this does not measure accuracy.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from PIL import Image
import torch

import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".local/verification"))
    args = parser.parse_args()
    if not app.segmenter.available:
        raise RuntimeError(app.segmenter.load_error)
    args.output.mkdir(parents=True, exist_ok=True)
    with Path(os.environ["MEDSAM2_CHECKPOINT"]).open("rb") as checkpoint:
        checkpoint_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()

    report = {
        "backend": app.segmenter.backend,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_sha256": checkpoint_sha256,
        "config": os.environ.get("MEDSAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml"),
        "note": "Real pipeline smoke test; no ground-truth masks, no accuracy measurement.",
        "images": [],
    }
    with zipfile.ZipFile(args.archive) as archive:
        inputs = sorted(name for name in archive.namelist() if re.fullmatch(r"\d{3}/input\.png", name))
        if not inputs:
            raise ValueError("Archive contains no NNN/input.png photos")
        for name in inputs:
            image = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("RGB"))
            start = time.perf_counter()
            overlay, _, note, rows = app.run_segmentation(image, 0.25, False)
            success = overlay is not None and bool(rows) and any(row[2] > 0 for row in rows)
            entry = {
                "input": name,
                "success": success,
                "seconds": round(time.perf_counter() - start, 3),
                "mask_pixels": [row[2] for row in rows],
                "status": note,
            }
            if success:
                output = args.output / f"{Path(name).parent.name}_overlay.png"
                Image.fromarray(overlay).save(output)
                entry["overlay"] = str(output)
            report["images"].append(entry)
            print(json.dumps(entry), flush=True)

    # A blank image must report the missing YOLO detection, without segmentation.
    _, stats, note, rows = app.run_segmentation(np.zeros((640, 640, 3), np.uint8), 0.25, False)
    report["no_detection_check"] = not rows and "YOLO did not detect any faces" in note and "Not run" in stats
    report["passed"] = sum(item["success"] for item in report["images"])
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{len(inputs)} photos segmented. Report: {report_path}", flush=True)
    if report["passed"] != len(inputs) or not report["no_detection_check"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
