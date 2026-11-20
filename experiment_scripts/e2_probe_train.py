#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2_probe_train.py — Instrumented training runs for collapse-mechanism evidence
(paper Sec. 4.3, H3 dynamics + P1/P2/P3 predictions).

Per-epoch instrumentation on a fixed probe batch (first 256 validation windows):
  - attention entropy per encoder layer (eval mode)
  - condition-probe accuracy (logistic, 5-fold CV on the probe batch)
  - RUL-probe R2 (ridge, 5-fold CV on the probe batch)
  - prediction std (constancy detector)
Per-epoch during training:
  - gradient L2 norm per parameter group (input_proj / tcn / gru / encoder / head),
    measured pre-clipping
  - train loss, val RMSE

Usage:
    python e2_probe_train.py --config FD002_L2
    python e2_probe_train.py --config all           # all configs in order
    python e2_probe_train.py --list
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from exp_final_patent import PureRULModel            # noqa: E402
from src.data_loader import preprocess_pipeline       # noqa: E402

OUT_ROOT = ROOT / "results" / "e2_probe"

# tag -> (dataset_id, ws, L, seed, note)
CONFIGS = {
    "FD002_L2":  (2, 20, 2, 42,   "expect deterministic collapse (P10)"),
    "FD002_L3":  (2, 20, 3, 42,   "healthy control"),
    "FD002_L2_s7": (2, 20, 2, 7,  "collapse robustness, second seed"),
    "FD004_L1":  (4, 60, 1, 42,   "expect deterministic collapse (P10)"),
    "FD004_L3":  (4, 60, 3, 42,   "healthy control (P10 seed42-like)"),
    "FD004_L3_s123": (4, 60, 3, 123, "bistable probe (collapsed in original)"),
}

TRAIN = dict(lr=1e-4, weight_decay=1e-4, batch_size=32, epochs=150,
             patience=20, grad_clip=1.0, cap=125, val_ratio=0.15,
             hidden_dim=128, gru_layers=2, num_heads=4, dropout=0.3,
             sensor_dropout_p=0.1)
N_COND = {1: 1, 2: 6, 3: 1, 4: 6}
PROBE_N = 256


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def attn_entropy_layer(attn):
    """attn: (B,H,L,L) causal. Mean normalized entropy over valid rows."""
    B, H, L, _ = attn.shape
    ents = []
    for t in range(L):
        row = attn[:, :, t, :t + 1]
        row = row / (row.sum(-1, keepdim=True) + 1e-12)
        ent = -(row * (row + 1e-12).log2()).sum(-1)
        norm = np.log2(t + 1)
        if norm > 0:
            ents.append((ent / norm).mean().item())
    return float(np.mean(ents))


def probe_metrics(h_pool, y, cond_labels):
    hz = StandardScaler().fit_transform(h_pool)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    out = {}
    reg = Ridge(alpha=10.0)
    rp = cross_val_predict(reg, hz, y, cv=cv)
    out["rul_r2_cv"] = float(1 - ((y - rp) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    if cond_labels is not None:
        clf = LogisticRegression(max_iter=1000)
        cp = cross_val_predict(clf, hz, cond_labels, cv=cv)
        out["cond_acc_cv"] = float((cp == cond_labels).mean())
    return out


def run_config(tag, epochs_override=None):
    ds_id, ws, L, seed, note = CONFIGS[tag]
    epochs = epochs_override or TRAIN["epochs"]
    patience = 3 if epochs_override else TRAIN["patience"]
    print(f"\n{'='*70}\n  E2 probe run: {tag}  (FD00{ds_id}, ws={ws}, L={L}, seed={seed})\n"
          f"  note: {note}\n{'='*70}", flush=True)
    out_dir = OUT_ROOT / tag
    if (out_dir / "probe_log.json").exists():
        print(f"  [skip] {out_dir}/probe_log.json exists")
        return
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    seed_everything(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    pipe = preprocess_pipeline(str(ROOT / "data"), dataset_id=ds_id, window_size=ws,
                               cap=TRAIN["cap"], val_ratio=TRAIN["val_ratio"],
                               random_seed=seed)
    X_train, y_train = pipe["X_train"], pipe["y_train"]
    X_val, y_val = pipe["X_val"], pipe["y_val"]
    n_features = pipe["n_features"]
    feat_cols = pipe["feature_cols"]

    # fixed probe batch + condition labels
    X_probe = torch.from_numpy(X_val[:PROBE_N]).float().to(device)
    y_probe = y_val[:PROBE_N]
    cond_probe = None
    op_idx = [i for i, c in enumerate(feat_cols) if c.startswith("op_setting")]
    if N_COND[ds_id] > 1 and len(op_idx) == 3:
        last_step = X_val[:PROBE_N, -1, op_idx].astype(np.float64)
        cond_probe = KMeans(n_clusters=N_COND[ds_id], n_init=10,
                            random_state=0).fit_predict(last_step)

    model = PureRULModel(n_features=n_features, hidden_dim=TRAIN["hidden_dim"],
                         num_heads=TRAIN["num_heads"], num_layers=TRAIN["gru_layers"],
                         dropout=TRAIN["dropout"],
                         sensor_dropout_p=TRAIN["sensor_dropout_p"],
                         n_encoder_layers=L).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # parameter groups for gradient norms
    groups = {
        "input_proj": list(model.input_proj.parameters()),
        "tcn": [p for l in model.tcn_layers for p in l.parameters()],
        "gru": list(model.gru.parameters()) + list(model.gru_proj.parameters()),
        "encoder": [p for l in model.encoder_layers for p in l.parameters()],
        "head": list(model.rul_head.parameters()),
    }

    # attention hooks (eval-time measurement on probe batch)
    attn_caps = []
    def attn_hook(module, inp, out):
        attn_caps.append(out[1].detach().cpu())
    handles = [l.attn_sublayer.register_forward_hook(attn_hook)
               for l in model.encoder_layers]
    feat_caps = []
    def feat_hook(module, inp, out):
        feat_caps.append(inp[0].detach().cpu())
    handles.append(model.rul_head[0].register_forward_hook(feat_hook))

    def measure_probe():
        model.eval()
        attn_caps.clear(); feat_caps.clear()
        with torch.no_grad():
            preds = model(X_probe).squeeze(-1).cpu().numpy()
        h_pool = feat_caps[0].numpy()
        ents = [attn_entropy_layer(a) for a in attn_caps]
        pm = probe_metrics(h_pool, y_probe, cond_probe)
        return {
            "pred_std": float(np.std(preds)),
            "pred_mean": float(np.mean(preds)),
            "attn_entropy": ents,
            "h_pool": h_pool,
            **pm,
        }

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(),
                                      torch.from_numpy(y_train).float()),
                        batch_size=TRAIN["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                                  weight_decay=TRAIN["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    mse = nn.MSELoss()

    log = {"config": {"tag": tag, "dataset": f"FD00{ds_id}", "ws": ws, "L": L,
                      "seed": seed, "note": note, "params": n_params,
                      "n_train": len(X_train), "n_val": len(X_val),
                      "n_probe": int(PROBE_N)},
           "epochs": []}
    feat_snapshots = {}

    best_val, best_state, wait, t0 = float("inf"), None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        grad_sq = {k: 0.0 for k in groups}
        n_batches = 0
        ep_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss = mse(model(X_b).squeeze(-1), y_b)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for k, params in groups.items():
                    s = 0.0
                    for p in params:
                        if p.grad is not None:
                            s += float(p.grad.norm() ** 2)
                    grad_sq[k] += math.sqrt(s)
            n_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step()
            ep_loss += loss.item() * len(X_b)
        scheduler.step()

        # validation rmse
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(X_val), 512):
                xb = torch.from_numpy(X_val[i:i + 512]).float().to(device)
                vp.append(model(xb).squeeze(-1).cpu())
            vpred = torch.cat(vp).numpy()
        val_rmse = float(np.sqrt(np.mean((vpred - y_val) ** 2)))

        pm = measure_probe()
        h = pm.pop("h_pool")
        if ep % 5 == 1 or ep == 1:
            feat_snapshots[f"ep{ep:03d}"] = h
        rec = {"ep": ep, "train_loss": ep_loss / len(X_train),
               "val_rmse": val_rmse,
               "grad_norm": {k: v / n_batches for k, v in grad_sq.items()},
               **pm}
        log["epochs"].append(rec)
        is_best = val_rmse < best_val
        if is_best:
            best_val, wait = val_rmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 5 == 0 or is_best or ep == 1:
            print(f"  ep{ep:3d} loss={rec['train_loss']:8.2f} val={val_rmse:6.2f} "
                  f"attnEnt={np.mean(pm['attn_entropy']):.4f} "
                  f"predStd={pm['pred_std']:7.3f} "
                  f"cond={pm.get('cond_acc_cv', float('nan')):.2f} "
                  f"rulR2={pm['rul_r2_cv']:.2f} "
                  f"g_enc={rec['grad_norm']['encoder']:.3f} "
                  f"g_tcn={rec['grad_norm']['tcn']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if wait >= patience:
            print(f"  early stop ep{ep}", flush=True)
            break

    for h_ in handles:
        h_.remove()

    # final test evaluation
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    X_test, y_test = pipe["X_test"], pipe["y_test"]
    with torch.no_grad():
        tp = []
        for i in range(0, len(X_test), 256):
            xb = torch.from_numpy(X_test[i:i + 256]).float().to(device)
            tp.append(model(xb).squeeze(-1).cpu())
        tpred = torch.cat(tp).numpy()
    test_rmse = float(np.sqrt(np.mean((tpred - y_test) ** 2)))
    log["final"] = {"best_val_rmse": best_val, "test_rmse": test_rmse,
                    "collapsed": bool(test_rmse > 40),
                    "pred_std_test": float(np.std(tpred)),
                    "train_time_sec": round(time.time() - t0, 1)}

    torch.save({"model_state": best_state, "config": log["config"],
                "final": log["final"]}, out_dir / "model_best.pt")
    np.savez(out_dir / "feat_snapshots.npz", **feat_snapshots)
    np.save(out_dir / "test_pred.npy", tpred)
    with open(out_dir / "probe_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  DONE {tag}: test_rmse={test_rmse:.2f} collapsed={log['final']['collapsed']}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=0)
    args = ap.parse_args()
    if args.list:
        for k, v in CONFIGS.items():
            print(f"{k:<15} FD00{v[0]} ws={v[1]} L={v[2]} seed={v[3]}  {v[4]}")
        sys.exit(0)
    tags = list(CONFIGS) if args.config == "all" else args.config.split(",")
    for t in tags:
        run_config(t, epochs_override=(args.epochs or None))
