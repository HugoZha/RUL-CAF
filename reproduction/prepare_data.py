#!/usr/bin/env python3
"""Download C-MAPSS, verify checksums, and materialise the exact data splits.

The raw NASA C-MAPSS files are ~43 MB uncompressed, so they are distributed
as a GitHub release asset rather than committed to git:

    https://github.com/HugoZha/RUL-CAF/releases/tag/data-v1

This script downloads ``c-mapss.zip`` into ``data/``, optionally verifies it
against ``data_manifest.json``, and writes the exact 85/15 window split
indices to ``splits/FD00X_split.npz``.

Usage
-----
    python prepare_data.py                      # download (if missing) + splits
    python prepare_data.py --verify             # also check SHA256 manifest
    python prepare_data.py --data-dir /path/to/c-mapss   # use local copy
    python prepare_data.py --skip-download      # only (re)build splits
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from configs import DATASETS, FIXED_CONFIG, BEST_CONFIG  # noqa: E402
from src.data_loader import build_split_indices, save_split  # noqa: E402

RELEASE_URL = ("https://github.com/HugoZha/RUL-CAF/releases/download/"
               "data-v1/c-mapss.zip")
EXPECTED_SIZE = 12015668  # bytes, as reported by the GitHub release API


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == EXPECTED_SIZE:
        print(f"[data] using existing {dest}")
        return
    print(f"[data] downloading {RELEASE_URL} -> {dest}")
    with urllib.request.urlopen(RELEASE_URL, timeout=300) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    size = dest.stat().st_size
    if size != EXPECTED_SIZE:
        print(f"[data] WARNING: downloaded {size} bytes, expected {EXPECTED_SIZE}")


def extract_zip(zpath: Path, data_dir: Path) -> Path:
    """Extract and return the directory that actually holds train_FD001.txt."""
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(data_dir)
    candidates = [p.parent for p in data_dir.rglob("train_FD001.txt")]
    if not candidates:
        raise RuntimeError(f"train_FD001.txt not found under {data_dir}")
    found = candidates[0]
    if found != data_dir:
        # Flatten nested archive layouts (e.g. c-mapss/train_FD001.txt).
        for p in found.iterdir():
            target = data_dir / p.name
            if p.is_file() and not target.exists():
                shutil.move(str(p), str(target))
        shutil.rmtree(found, ignore_errors=True)
    return data_dir


def verify_manifest(data_dir: Path) -> None:
    mpath = HERE / "data_manifest.json"
    if not mpath.exists():
        print("[data] no data_manifest.json, skipping checksum verification")
        return
    manifest = json.load(open(mpath))
    ok = True
    for name, expected in manifest.items():
        p = data_dir / name
        if not p.exists():
            print(f"[data] MISSING {name}")
            ok = False
            continue
        actual = sha256(p)
        if actual != expected:
            print(f"[data] CHECKSUM MISMATCH {name}: {actual} != {expected}")
            ok = False
    print(f"[data] SHA256 verification: {'PASS' if ok else 'FAIL'} ({len(manifest)} files)")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--split-dir", default=str(HERE / "splits"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "train_FD001.txt").exists() and not args.skip_download:
        zpath = data_dir.parent / "c-mapss.zip"
        download_zip(zpath)
        data_dir = extract_zip(zpath, data_dir)
    if not (data_dir / "train_FD001.txt").exists():
        raise SystemExit(f"No C-MAPSS files in {data_dir}; pass --data-dir or allow download.")

    if args.verify:
        verify_manifest(data_dir)

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        cfg = BEST_CONFIG[ds]
        split = build_split_indices(str(data_dir), dataset_id=ds,
                                    window_size=cfg["window_size"],
                                    cap=FIXED_CONFIG["rul_cap"],
                                    norm_method=FIXED_CONFIG["norm_method"],
                                    val_ratio=FIXED_CONFIG["val_ratio"],
                                    split_seed=FIXED_CONFIG["split_seed"])
        m = split["manifest"]
        print(f"[split] {m['dataset']}: {m['n_windows_total']} windows -> "
              f"{m['n_train_windows']} train / {m['n_val_windows']} val, "
              f"{m['n_test_engines']} test engines, {m['n_features']} features")
        save_split(split_dir / f"FD00{ds}_split.npz", split)


if __name__ == "__main__":
    main()
