#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_collapse.py — Scan all experiment result JSONs in RULPredictor_V2/results,
build the collapse matrix for paper Sec. 3.4 (prevalence study).

Collapse criterion: test_rmse > 40 (the constant-prediction regime; healthy runs
are in the 13-25 range on all datasets).

Outputs:
  paper_exp/collapse_matrix.csv      — one row per run
  paper_exp/collapse_summary.json    — aggregate stats + key tables for the paper
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # RULPredictor_V2
RESULTS = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent       # paper_exp
if not OUT_DIR.exists():
    OUT_DIR.mkdir()

COLLAPSE_THR = 40.0

rows = []

def extract_from_json(path: Path):
    """Extract a run record from a result JSON."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    if not isinstance(d, dict) or "test_rmse" not in d:
        return None
    fam = path.relative_to(RESULTS).parts[0]
    rel = path.relative_to(RESULTS).parts[1:]
    dataset = d.get("dataset", "")
    if not dataset.startswith("FD"):
        m = re.search(r"FD00(\d)", str(path))
        dataset = f"FD00{m.group(1)}" if m else "?"

    rec = {
        "family": fam,
        "path": str(path.relative_to(RESULTS)),
        "dataset": dataset,
        "test_rmse": float(d.get("test_rmse", np.nan)),
        "best_val_rmse": float(d.get("best_val_rmse", np.nan)),
        "test_mae": d.get("test_mae"),
        "test_r2": d.get("test_r2"),
        "ws": d.get("window_size"),
        "heads": d.get("num_heads"),
        "L": d.get("n_encoder_layers"),
        "params": d.get("params"),
        "adversarial": d.get("config", {}).get("adversarial"),
        "best_epoch": d.get("best_epoch"),
        "early_stop_epoch": d.get("early_stop_epoch"),
    }
    # try to parse variant (ws_30, layers_2, seed_123, A0..) from parent dir name
    parent = path.parent.name
    m = re.match(r"ws_(\d+)", parent)
    if m: rec["ws"] = int(m.group(1))
    m = re.match(r"layers_(\d+)", parent)
    if m: rec["L"] = int(m.group(1))
    m = re.match(r"seed_(\d+)", parent)
    if m: rec["seed"] = int(m.group(1))
    else: rec["seed"] = d.get("seed", 42)
    m = re.match(r"(A\d)", parent)
    if m: rec["variant"] = m.group(1)
    m = re.match(r"h_(\d+)", parent)
    if m: rec["heads"] = int(m.group(1))
    return rec

# ---- walk all result JSONs ----
for p in sorted(RESULTS.rglob("*.json")):
    if p.name == "summary.json":
        # flatten nested summaries (e.g. exp_final_patent)
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception:
            continue
        for key, d in (s.items() if isinstance(s, dict) else []):
            if isinstance(d, dict) and "test_rmse" in d:
                rec = extract_from_json(p)
                if rec:
                    rec.update({
                        "dataset": d.get("dataset", key),
                        "test_rmse": float(d["test_rmse"]),
                        "best_val_rmse": float(d.get("best_val_rmse", np.nan)),
                        "ws": d.get("window_size"), "L": d.get("n_encoder_layers"),
                        "heads": d.get("num_heads"),
                    })
                    rows.append(rec)
            # seed-nested (FD004 in exp_final_patent summary)
            if isinstance(d, dict):
                for sk, sd in d.items():
                    if isinstance(sd, dict) and "test_rmse" in sd:
                        rec = extract_from_json(p)
                        if rec:
                            rec.update({
                                "dataset": sd.get("dataset", f"{key}/{sk}"),
                                "test_rmse": float(sd["test_rmse"]),
                                "best_val_rmse": float(sd.get("best_val_rmse", np.nan)),
                                "ws": sd.get("window_size"), "L": sd.get("n_encoder_layers"),
                                "seed": sk.replace("seed_", ""),
                            })
                            rows.append(rec)
        continue
    rec = extract_from_json(p)
    if rec:
        rows.append(rec)

df = pd.DataFrame(rows)
df["collapsed"] = df["test_rmse"] > COLLAPSE_THR

# ---- constant-prediction fingerprint: std of test RUL per dataset ----
fingerprint = {}
for ds_id in [1, 2, 3, 4]:
    rul = pd.read_csv(ROOT / "data" / f"RUL_FD00{ds_id}.txt", sep=r"\s+", header=None).values.ravel()
    rul_cap = np.minimum(rul, 125)
    fingerprint[f"FD00{ds_id}"] = {
        "n_test_engines": int(len(rul)),
        "std_rul_raw": float(np.std(rul)),
        "std_rul_cap125": float(np.std(rul_cap)),
        "rmse_if_predict_capped_mean": float(np.sqrt(np.mean((rul_cap - rul_cap.mean()) ** 2))),
    }

# ---- aggregate tables ----
summary = {
    "n_runs_total": int(len(df)),
    "n_collapsed": int(df["collapsed"].sum()),
    "collapse_rate_overall": float(df["collapsed"].mean()),
    "collapse_by_dataset": df.groupby("dataset")["collapsed"].agg(["count", "sum", "mean"]).to_dict("index"),
    "collapse_by_family": df.groupby("family")["collapsed"].agg(["count", "sum", "mean"]).to_dict("index"),
    "constant_prediction_fingerprint": fingerprint,
    "collapsed_runs": df[df["collapsed"]][
        ["family", "path", "dataset", "ws", "L", "heads", "seed", "test_rmse", "best_val_rmse", "early_stop_epoch"]
    ].to_dict("records"),
}

# ---- key table for paper: FD002/FD004 (dataset, ws, L) -> collapsed? ----
key = df[df["dataset"].isin(["FD002", "FD004"])].copy()
key["cfg"] = key.apply(lambda r: f"ws={r['ws']},L={r['L']},h={r['heads']}", axis=1)
per_cfg = key.groupby(["dataset", "cfg"]).agg(
    n_runs=("collapsed", "count"),
    n_collapse=("collapsed", "sum"),
    min_rmse=("test_rmse", "min"),
    max_rmse=("test_rmse", "max"),
).reset_index()
summary["fd24_config_table"] = per_cfg.to_dict("records")

df.to_csv(OUT_DIR / "collapse_matrix.csv", index=False)
with open(OUT_DIR / "collapse_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---- console report ----
print(f"Total runs scanned: {len(df)}")
print(f"Collapsed (test RMSE > {COLLAPSE_THR}): {df['collapsed'].sum()} "
      f"({100*df['collapsed'].mean():.1f}%)")
print("\n--- Collapse by dataset ---")
print(df.groupby("dataset")["collapsed"].agg(["count", "sum", "mean"]).round(3))
print("\n--- Constant-prediction fingerprint (test RUL stats) ---")
for k, v in fingerprint.items():
    print(f"  {k}: std(cap125)={v['std_rul_cap125']:.2f}  "
          f"RMSE(const=mean)={v['rmse_if_predict_capped_mean']:.2f}")
print("\n--- FD002/FD004 config-level table (n_collapse/n_runs, min-max RMSE) ---")
for _, r in per_cfg.iterrows():
    print(f"  {r['dataset']} {r['cfg']}: {r['n_collapse']}/{r['n_runs']} collapsed, "
          f"RMSE {r['min_rmse']:.2f}-{r['max_rmse']:.2f}")
