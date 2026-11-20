#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_offline.py — Offline probing of saved checkpoints (paper Sec. 4, H1 & H2).

For each checkpoint (healthy / collapsed), on the deterministic TEST split:
  H1 (condition-identity shortcut):
    - penultimate pooled features h_pool (B, 256)
    - operating-condition id recovered by K-Means (k = #conditions) on the
      op-setting channels of the last window timestep
    - silhouette(h_pool | condition)  vs  silhouette(h_pool | RUL bucket)
    - linear probes: condition classification accuracy, RUL regression R2
  H2 (attention entropy collapse):
    - per-layer attention entropy normalized by log(#valid causal positions)
    - mean max attention weight
  Constancy fingerprint:
    - std of predictions, RMSE vs recorded, segment errors

Outputs: paper_exp/probe_results.json, paper_exp/probe_feats/<tag>.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from exp_final_patent import PureRULModel          # noqa: E402
from src.data_loader import preprocess_pipeline     # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FEAT_DIR = OUT_DIR / "probe_feats"
if not FEAT_DIR.exists():
    FEAT_DIR.mkdir()

# tag -> (dataset_id, ws, ckpt_path, expectation)
CKPTS = {
    "FD002_L1": (2, 20, ROOT / "results/exp_p10_no_adv/FD002/layers_1/model_best.pt", "healthy"),
    "FD002_L2": (2, 20, ROOT / "results/exp_p10_no_adv/FD002/layers_2/model_best.pt", "collapsed"),
    "FD002_L3": (2, 20, ROOT / "results/exp_p10_no_adv/FD002/layers_3/model_best.pt", "healthy"),
    "FD002_L4": (2, 20, ROOT / "results/exp_p10_no_adv/FD002/layers_4/model_best.pt", "collapsed"),
    "FD004_L1": (4, 60, ROOT / "results/exp_p10_no_adv/FD004/layers_1/model_best.pt", "collapsed"),
    "FD004_L2": (4, 60, ROOT / "results/exp_p10_no_adv/FD004/layers_2/model_best.pt", "healthy"),
    "FD004_L3": (4, 60, ROOT / "results/exp_p10_no_adv/FD004/layers_3/model_best.pt", "healthy"),
    "FD004_L4": (4, 60, ROOT / "results/exp_p10_no_adv/FD004/layers_4/model_best.pt", "healthy"),
    "FD004_L3_final": (4, 60, ROOT / "results/exp_final_patent/FD004/model_best.pt", "verify"),
    "FD001_L1_final": (1, 40, ROOT / "results/exp_final_patent/FD001/model_best.pt", "healthy(single-cond)"),
    "FD002_L3_final": (2, 20, ROOT / "results/exp_final_patent/FD002/model_best.pt", "healthy"),
    "FD003_L1_final": (3, 30, ROOT / "results/exp_final_patent/FD003/model_best.pt", "healthy(single-cond)"),
}

N_CONDITIONS = {1: 1, 2: 6, 3: 1, 4: 6}


def causal_attention_entropy(attn):
    """attn: (B, H, L, L) with causal mask applied (invalid entries ~0).
    Returns mean normalized entropy over heads & valid query positions."""
    B, H, L, _ = attn.shape
    ents, maxws = [], []
    for t in range(L):
        row = attn[:, :, t, :t + 1]                       # (B, H, t+1)
        row = row / (row.sum(-1, keepdim=True) + 1e-12)
        ent = -(row * (row + 1e-12).log2()).sum(-1)        # (B, H)
        norm = np.log2(t + 1)
        if norm > 0:
            ents.append((ent / norm).mean().item())
        maxws.append(row.max(-1).values.mean().item())
    return float(np.mean(ents)), float(np.mean(maxws))


def probe_model(tag, dataset_id, ws, ckpt_path, expectation):
    print(f"\n=== {tag}  (FD00{dataset_id}, ws={ws}, expect={expectation}) ===")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    L_enc = ckpt.get("n_encoder_layers")

    pipe = preprocess_pipeline(str(ROOT / "data"), dataset_id=dataset_id,
                               window_size=ws, cap=125, val_ratio=0.15,
                               random_seed=42)
    X_test, y_test = pipe["X_test"], pipe["y_test"]
    n_features = pipe["n_features"]
    feat_cols = pipe["feature_cols"]

    model = PureRULModel(n_features=n_features, hidden_dim=128, num_heads=4,
                         num_layers=2, dropout=0.3, sensor_dropout_p=0.1,
                         n_encoder_layers=L_enc)
    model.load_state_dict(state)
    model.eval()

    # ---- hooks: attention weights + penultimate features ----
    attn_store, feat_store = [], []
    def attn_hook(module, inp, out):
        attn_store.append(out[1].detach())
    def feat_hook(module, inp, out):
        feat_store.append(inp[0].detach())
    handles = [layer.attn_sublayer.register_forward_hook(attn_hook)
               for layer in model.encoder_layers]
    handles.append(model.rul_head[0].register_forward_hook(feat_hook))

    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), 128):
            X_b = torch.from_numpy(X_test[i:i + 128]).float()
            preds.append(model(X_b).squeeze(-1))
    for h in handles:
        h.remove()
    preds = torch.cat(preds).numpy()
    h_pool = torch.cat(feat_store).numpy()                # (N, 256)
    # attn_store: list over batches of list-per-layer; reorganize
    n_batches = (len(X_test) + 127) // 128
    per_layer_attn = [[] for _ in range(L_enc)]
    for b in range(n_batches):
        for l in range(L_enc):
            per_layer_attn[l].append(attn_store[b * L_enc + l])
    per_layer_attn = [torch.cat(chunks) for chunks in per_layer_attn]  # (N,H,L,L) each

    # ---- metrics ----
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    res = {
        "tag": tag, "dataset": f"FD00{dataset_id}", "ws": ws, "L": L_enc,
        "expectation": expectation,
        "test_rmse": rmse,
        "pred_std": float(np.std(preds)),
        "pred_mean": float(np.mean(preds)),
        "y_std": float(np.std(y_test)),
        "n_test": int(len(y_test)),
    }

    # H2: attention entropy
    attn_stats = []
    for l, A in enumerate(per_layer_attn):
        ent, maxw = causal_attention_entropy(A)
        attn_stats.append({"layer": l + 1, "norm_entropy": ent, "mean_max_weight": maxw})
    res["attention"] = attn_stats
    res["attn_entropy_mean"] = float(np.mean([a["norm_entropy"] for a in attn_stats]))

    # H1: condition labels from op settings of the last window timestep
    op_idx = [i for i, c in enumerate(feat_cols) if c.startswith("op_setting")]
    n_cond = N_CONDITIONS[dataset_id]
    if n_cond > 1 and len(op_idx) == 3:
        last_step = X_test[:, -1, op_idx].astype(np.float64)   # (N, 3) normalized
        km = KMeans(n_clusters=n_cond, n_init=10, random_state=0).fit(last_step)
        cond_labels = km.labels_
        # cluster purity proxy: variance ratio of settings between/within clusters
        res["cond_cluster_inertia"] = float(km.inertia_)
    else:
        cond_labels = None

    # RUL buckets for silhouette
    buckets = np.digitize(y_test, [30, 60, 100])

    hz = StandardScaler().fit_transform(h_pool)
    res["silhouette_rul_bucket"] = float(silhouette_score(hz, buckets))

    if cond_labels is not None:
        res["silhouette_condition"] = float(silhouette_score(hz, cond_labels))
        # cross-validated linear probes (5-fold)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        cond_pred = cross_val_predict(clf, hz, cond_labels, cv=cv)
        res["probe_cond_acc_cv"] = float((cond_pred == cond_labels).mean())
        reg = Ridge(alpha=10.0)
        rul_pred_cv = cross_val_predict(reg, hz, y_test, cv=cv)
        ss_res = ((y_test - rul_pred_cv) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        res["probe_rul_r2_cv"] = float(1 - ss_res / ss_tot)
        if res["pred_std"] > 1.0:
            gm = preds.mean()
            ss_between = sum(len(preds[cond_labels == c]) *
                             (preds[cond_labels == c].mean() - gm) ** 2
                             for c in range(n_cond))
            ss_total = ((preds - gm) ** 2).sum()
            res["eta2_pred_by_cond"] = float(ss_between / (ss_total + 1e-12))
        # mean capped train RUL per condition (what a condition-shortcut predicts)
        train_df = pd.read_csv(ROOT / "data" / f"train_FD00{dataset_id}.txt",
                               sep=r"\s+", header=None)
        # recover condition of each TRAIN row via the fitted kmeans
        op_raw = train_df.iloc[:, [2, 3, 4]].values
        norm = pipe["normalizer"]
        opn = np.stack([
            (op_raw[:, j] - norm.scalers[c]["min"]) /
            (norm.scalers[c]["max"] - norm.scalers[c]["min"] + 1e-8)
            for j, c in enumerate(["op_setting_1", "op_setting_2", "op_setting_3"])
        ], axis=1)
        train_cond = km.predict(opn.astype(np.float64))
        max_cycles = train_df.groupby(0)[1].max()
        rul_train_cap = (max_cycles.loc[train_df[0]].values - train_df[1].values)
        rul_train_cap = np.minimum(rul_train_cap, 125)
        res["train_rul_cap_mean"] = float(rul_train_cap.mean())
        res["train_rul_cap_mean_per_cond"] = {
            str(c): float(rul_train_cap[train_cond == c].mean())
            for c in range(n_cond)}
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        reg = Ridge(alpha=10.0)
        rul_pred_cv = cross_val_predict(reg, hz, y_test, cv=cv)
        ss_res = ((y_test - rul_pred_cv) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        res["probe_rul_r2_cv"] = float(1 - ss_res / ss_tot)

    np.savez(FEAT_DIR / f"{tag}.npz", h_pool=h_pool, preds=preds, y=y_test,
             cond=cond_labels if cond_labels is not None else np.zeros_like(y_test, dtype=int),
             buckets=buckets)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in res.items() if k != "attention"}, ensure_ascii=False))
    for a in attn_stats:
        print(f"    attn L{a['layer']}: norm_entropy={a['norm_entropy']:.4f} "
              f"max_w={a['mean_max_weight']:.4f}")
    return res


if __name__ == "__main__":
    all_res = []
    for tag, (ds, ws, path, exp) in CKPTS.items():
        if not path.exists():
            print(f"[skip] {tag}: {path} not found")
            continue
        all_res.append(probe_model(tag, ds, ws, path, exp))
    with open(OUT_DIR / "probe_results.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved {len(all_res)} probe results -> probe_results.json")
