"""Identify the site's A/B weights without importing ML libraries or unpickling.

Compare every indexed tensor-storage payload because checkpoint containers can
have different training metadata and therefore different whole-file hashes.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def tensor_storages(checkpoint: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(checkpoint)) as archive:
        storages = {
            name.split("/data/", 1)[1]: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist() if "/data/" in name and not name.endswith("/")
        }
    if not storages:
        raise ValueError("Checkpoint contains no tensor storages")
    return storages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--github-commit", required=True, help="Commit verified with git ls-remote origin HEAD")
    args = parser.parse_args()
    checkpoint = (ROOT / "models/best.pt").read_bytes()
    active = tensor_storages(checkpoint)
    matches = []
    with zipfile.ZipFile(args.archive) as archive:
        entries = [name for name in archive.namelist() if name.endswith("/weights/best.pt")]
        for name in entries:
            candidate = archive.read(name)
            if tensor_storages(candidate) == active:
                matches.append((name, candidate))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one matching A/B checkpoint, found {len(matches)}")
        name, matched = matches[0]
        variant = name.split("/")[-3]
        comparison_name = name.split("/training/", 1)[0] + "/comparison.csv"
        comparison = list(csv.DictReader(io.StringIO(archive.read(comparison_name).decode("utf-8-sig"))))
    experiment = json.loads((ROOT / "assets/results/experiment.json").read_text(encoding="utf-8"))
    # Verify both metric columns before attaching this identity to their UI.
    for source, target in (("val", "validation"), ("test", "independent_test")):
        row = next(row for row in comparison if row["variant"] == variant and row["split"] == source)
        for csv_key, json_key in (("precision", "precision"), ("recall", "recall"),
                                  ("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")):
            if abs(float(row[csv_key]) - experiment[target][json_key]) > 1e-6:
                raise ValueError(f"Saved {target}/{json_key} does not match {variant}")
    provenance = {
        "variant": variant,
        "display_name": {"A_no_flip": "A (no horizontal flip)", "B_flip_50": "B (50% horizontal flip)"}[variant],
        "checkpoint_path": "models/best.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "github_commit": args.github_commit,
        "github_blob": hashlib.sha1(f"blob {len(checkpoint)}\0".encode() + checkpoint).hexdigest(),
        "ablation_archive": args.archive.name,
        "ablation_entry": name,
        "ablation_checkpoint_sha256": hashlib.sha256(matched).hexdigest(),
        "matching_tensor_storages": len(active),
        "tensor_storage_fingerprint": hashlib.sha256(json.dumps(active, sort_keys=True).encode()).hexdigest(),
        "epochs_completed": int(row["epochs_completed"]),
        "metrics_verified": True,
        "note": "All indexed tensor-storage payloads match; checkpoint container metadata differs. No pickle was executed.",
    }
    output = ROOT / "assets/results/model_provenance.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
