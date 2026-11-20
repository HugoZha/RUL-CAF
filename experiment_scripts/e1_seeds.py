#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e1_seeds.py — Multi-seed validation of the condition-adaptive configuration
(paper E1: stability audit + mean±std tables).

Configuration = exp_final_patent BEST_CONFIG (per-dataset adaptive):
  FD001 ws=40 L=1 | FD002 ws=20 L=3 | FD003 ws=30 L=1 | FD004 ws=60 L=3
Seeds: 7, 123, 2026, 3407 for every dataset (seed 42 already archived in
exp_final_patent; results pooled in analysis).

"Seed" controls model init + data order (torch/np). The train/val split is
held fixed at random_state=42 for comparability with all archived runs.

Outputs per run: results/e1_seeds/FD00X/seed_S/{result.json, predictions.npy, model_best.pt}
Aggregation: results/e1_seeds/summary.json (appended after each run)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from exp_final_patent import PureRULModel, BEST_CONFIG, FIXED_CONFIG  # noqa: E402
from src.data_loader import preprocess_pipeline                     # noqa: E402

OUT_ROOT = ROOT / "results" / "e1_seeds"
SEEDS = [7, 123, 2026, 3407]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def phm08_score(pred, true):
    """NASA PHM08 asymmetric scoring function."""
    d = pred - true
    s = 0.0
    for di in d:
        if di < 0:
            s += math.exp(-di / 13.0) - 1
        else:
            s += math.exp(di / 10.0) - 1
    return s


def run_one(dataset_idx, seed):
    ds_name = f"FD00{dataset_idx}"
    out_dir = OUT_ROOT / ds_name / f"seed_{seed}"
    if (out_dir / "result.json").exists():
        print(f"  [skip] {out_dir} exists", flush=True)
        return json.load(open(out_dir / "result.json"))
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    seed_everything(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = BEST_CONFIG[dataset_idx]
    F = FIXED_CONFIG

    print(f"\n[{ds_name} seed={seed}] ws={cfg['window_size']} L={cfg['n_encoder_layers']}",
          flush=True)

    pipe = preprocess_pipeline(str(ROOT / "data"), dataset_id=dataset_idx,
                               window_size=cfg["window_size"], cap=F["rul_cap"],
                               val_ratio=F["val_ratio"], random_seed=42)
    X_train, y_train = pipe["X_train"], pipe["y_train"]
    X_val, y_val = pipe["X_val"], pipe["y_val"]
    X_test, y_test = pipe["X_test"], pipe["y_test"]
    n_features = pipe["n_features"]

    model = PureRULModel(
        n_features=n_features, hidden_dim=F["hidden_dim"],
        num_heads=cfg["num_heads"], num_layers=F["gru_num_layers"],
        dropout=F["dropout"], sensor_dropout_p=F["sensor_dropout_p"],
        n_encoder_layers=cfg["n_encoder_layers"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).float()),
        batch_size=F["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=F["lr"],
                                  weight_decay=F["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=F["epochs"])
    mse = nn.MSELoss()

    best_val, best_state, best_ep, wait = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, F["epochs"] + 1):
        model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss = mse(model(X_b).squeeze(-1), y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), F["grad_clip"])
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(X_val), 512):
                xb = torch.from_numpy(X_val[i:i + 512]).float().to(device)
                vp.append(model(xb).squeeze(-1).cpu())
            vpred = torch.cat(vp).numpy()
        val_rmse = float(np.sqrt(np.mean((vpred - y_val) ** 2)))
        if val_rmse < best_val:
            best_val, best_ep, wait = val_rmse, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d} val={val_rmse:.2f} best={best_val:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if wait >= F["patience"]:
            print(f"  early stop ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        tp = []
        for i in range(0, len(X_test), 256):
            xb = torch.from_numpy(X_test[i:i + 256]).float().to(device)
            tp.append(model(xb).squeeze(-1).cpu())
        pred = torch.cat(tp).numpy()

    test_rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    test_mae = float(np.mean(np.abs(pred - y_test)))
    test_r2 = float(1 - np.sum((y_test - pred) ** 2) /
                    (np.sum((y_test - y_test.mean()) ** 2) + 1e-8))
    score = phm08_score(pred, y_test)

    segs = {}
    for lo, hi in [(0, 30), (30, 60), (60, 100), (100, 9999)]:
        m = (y_test >= lo) & (y_test < hi)
        if m.sum() > 0:
            segs[f"{lo}-{hi}"] = {"n": int(m.sum()),
                                  "rmse": float(np.sqrt(np.mean((pred[m] - y_test[m]) ** 2))),
                                  "mae": float(np.mean(np.abs(pred[m] - y_test[m])))}

    result = {
        "experiment": "E1 multi-seed (condition-adaptive config)",
        "dataset": ds_name, "seed": seed,
        "window_size": cfg["window_size"],
        "n_encoder_layers": cfg["n_encoder_layers"],
        "num_heads": cfg["num_heads"],
        "config": {**F, **cfg},
        "params": n_params,
        "best_val_rmse": best_val, "best_epoch": best_ep,
        "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2,
        "phm08_score": score,
        "pred_std": float(np.std(pred)),
        "collapsed": bool(test_rmse > 40),
        "val_test_gap_pct": (test_rmse - best_val) / best_val * 100,
        "segment_errors": segs,
        "training_time_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(out_dir / "predictions.npy", pred)
    torch.save({"model_state": best_state, "seed": seed,
                "dataset": ds_name}, out_dir / "model_best.pt")
    print(f"  DONE {ds_name} seed={seed}: RMSE={test_rmse:.2f} "
          f"collapsed={result['collapsed']} score={score:.0f}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="1,2,3,4")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    ds_list = [int(x) for x in args.datasets.split(",")]
    seed_list = [int(x) for x in args.seeds.split(",")]

    summary_path = OUT_ROOT / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for ds in ds_list:
        for seed in seed_list:
            r = run_one(ds, seed)
            summary.setdefault(f"FD00{ds}", {})[f"seed_{seed}"] = {
                k: r[k] for k in ("test_rmse", "test_mae", "test_r2",
                                  "phm08_score", "pred_std", "collapsed",
                                  "best_val_rmse")}
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
    print("\nAll done. Summary ->", summary_path, flush=True)
