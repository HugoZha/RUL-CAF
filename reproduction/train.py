#!/usr/bin/env python3
"""Train and evaluate ONE CAS-Fuse run on one C-MAPSS subset.

The loop is a line-for-line port of ``paper_exp/e1_seeds.py`` (the script that
produced the manuscript's E1 tables).  With default arguments it reproduces
the reported numbers on the same device/PyTorch version.

Examples
--------
    python train.py --dataset 1 --seed 42
    python train.py --dataset 4 --seed 7 --split-file splits/FD004_split.npz
    python train.py --dataset 2 --seed 123 --epochs 20 --patience 3 --out /tmp/smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from configs import BEST_CONFIG, FIXED_CONFIG, TRIPWIRE  # noqa: E402
from src.data_loader import preprocess_pipeline, load_split  # noqa: E402
from src.metrics import (attention_entropy, check_tripwires,  # noqa: E402
                         phm08_score, regression_metrics)
from src.model import build_model  # noqa: E402


def seed_everything(seed: int) -> None:
    """Same seeding as the archived experiment: init + data order; split fixed at 42."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        # The paper was produced on Apple M-series (MPS), CPU fallback otherwise.
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def run_one(dataset_id: int, seed: int, out_root=None, split_file=None,
            epochs=None, patience=None, device_pref="auto", save_model=False,
            verbose=True, data_dir=None) -> dict:
    """Run one train/val/test cycle; return a result dict and write ``result.json``."""
    ds = f"FD00{dataset_id}"
    cfg = BEST_CONFIG[dataset_id]
    F = dict(FIXED_CONFIG)
    if epochs is not None:
        F["epochs"] = epochs
    if patience is not None:
        F["patience"] = patience

    out_root = Path(out_root) if out_root else HERE / "runs"
    out_dir = out_root / ds / f"seed_{seed}"

    t_start = time.time()
    seed_everything(seed)                       # (1) must precede model init
    device = resolve_device(device_pref)

    split = load_split(split_file) if split_file else None
    pipe = preprocess_pipeline(
        str(data_dir) if data_dir else str(HERE / "data"), dataset_id=dataset_id,
        window_size=cfg["window_size"], cap=F["rul_cap"],
        norm_method=F["norm_method"], val_ratio=F["val_ratio"],
        random_seed=F["split_seed"], split_indices=split)
    X_train, y_train = pipe["X_train"], pipe["y_train"]
    X_val, y_val = pipe["X_val"], pipe["y_val"]
    X_test, y_test = pipe["X_test"], pipe["y_test"]

    # (2) model init consumes the seeded torch RNG stream
    model = build_model(pipe["n_features"], cfg, F).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # (3) DataLoader shuffle consumes the same stream, epoch by epoch
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).float()),
        batch_size=F["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=F["lr"],
                                  weight_decay=F["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=F["epochs"])
    mse = nn.MSELoss()

    best_val, best_state, best_ep, wait = float("inf"), None, 0, 0
    history = []
    plateau_streak = 0
    plateau_flag = False
    val_std = float(np.std(y_val))

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
        val_pred_std = float(np.std(vpred))

        # Tripwire C: plateau near std(y_val) with near-constant predictions.
        if (abs(val_rmse - val_std) <= TRIPWIRE["plateau_epsilon"]
                and val_pred_std < TRIPWIRE["plateau_pred_std"]):
            plateau_streak += 1
        else:
            plateau_streak = 0
        if plateau_streak >= TRIPWIRE["plateau_epochs"]:
            plateau_flag = True

        history.append({"epoch": ep, "val_rmse": val_rmse,
                        "val_pred_std": val_pred_std,
                        "plateau_streak": plateau_streak})
        if val_rmse < best_val:
            best_val, best_ep, wait = val_rmse, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  ep{ep:3d} val={val_rmse:.2f} best={best_val:.2f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        if wait >= F["patience"]:
            if verbose:
                print(f"  early stop at ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        tp = []
        for i in range(0, len(X_test), 256):
            xb = torch.from_numpy(X_test[i:i + 256]).float().to(device)
            tp.append(model(xb).squeeze(-1).cpu())
        pred = torch.cat(tp).numpy()

        # One extra hooked forward pass for Tripwire B (after training: no RNG
        # effect on the reported predictions).
        attn_vals = []
        for i in range(0, min(len(X_val), 256), 64):
            xb = torch.from_numpy(X_val[i:i + 64]).float().to(device)
            _, attn = model(xb, return_attn=True)
            attn_vals.append(attention_entropy(attn))
        ent = float(np.mean(attn_vals)) if attn_vals else 0.0

    metrics = regression_metrics(pred, y_test)
    test_rmse = metrics["rmse"]
    trip = check_tripwires(test_rmse, metrics["pred_std"], y_test, ent, TRIPWIRE)

    result = {
        "experiment": "E1 multi-seed (condition-adaptive CAS-Fuse config)",
        "dataset": ds, "seed": seed,
        "window_size": cfg["window_size"],
        "n_encoder_layers": cfg["n_encoder_layers"],
        "num_heads": cfg["num_heads"],
        "config": {**F, **cfg},
        "device": str(device),
        "params": n_params,
        "best_val_rmse": best_val, "best_epoch": best_ep,
        "test_rmse": test_rmse, "test_mae": metrics["mae"],
        "test_r2": metrics["r2"], "pred_std": metrics["pred_std"],
        "phm08_score": phm08_score(pred, y_test),
        "attention_entropy": ent,
        "tripwires": trip,
        "collapsed": bool(trip["collapsed"] or test_rmse > 40),
        "tripwire_C_plateau": bool(plateau_flag),
        "val_test_gap_pct": (test_rmse - best_val) / best_val * 100,
        "segment_errors": metrics["segment_errors"],
        "training_time_sec": round(time.time() - t_start, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.save(out_dir / "predictions.npy", pred)
    if save_model:
        torch.save({"model_state": best_state, "seed": seed, "dataset": ds},
                   out_dir / "model_best.pt")

    if verbose:
        print(f"  DONE {ds} seed={seed}: RMSE={test_rmse:.2f} "
              f"PHM08={result['phm08_score']:.0f} collapsed={result['collapsed']} "
              f"entropy={ent:.4f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-file", default=None,
                    help="splits/FD00X_split.npz for a bit-identical split")
    ap.add_argument("--out", default=None, help="output root (default: reproduction/runs)")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--patience", type=int, default=None, help="override early-stop patience")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--data-dir", default=None,
                    help="C-MAPSS directory (default: reproduction/data)")
    args = ap.parse_args()

    r = run_one(args.dataset, args.seed, out_root=args.out,
                split_file=args.split_file, epochs=args.epochs,
                patience=args.patience, device_pref=args.device,
                save_model=args.save_model, data_dir=args.data_dir)
    print(json.dumps({k: r[k] for k in
                      ("dataset", "seed", "test_rmse", "test_mae", "test_r2",
                       "phm08_score", "pred_std", "attention_entropy", "collapsed")},
                     indent=2))


if __name__ == "__main__":
    main()
