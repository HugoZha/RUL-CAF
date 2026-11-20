#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_ablation_phm08.py — PHM08 scores for all ablation runs (paper Table 10).

Loads each of the 32 saved checkpoints (8 ablations A0–A7 × 4 datasets),
rebuilds the corresponding AblationRULModel, forwards the deterministic test
set, computes the asymmetric PHM08 score, and cross-verifies the recomputed
RMSE against the archived result.json.

Output: paper_exp/ablation_phm08.json  {A0..A7: {FD001..: {rmse, score, ...}}}
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/Users/hugozha/project/RULPredictor_V2")
sys.path.insert(0, str(ROOT))
from exp_ablation import AblationRULModel, ABLATION_CONFIGS   # noqa: E402
from exp_final_patent import BEST_CONFIG                      # noqa: E402
from src.data_loader import preprocess_pipeline               # noqa: E402

OUT = ROOT / "paper_exp" / "ablation_phm08.json"


def phm08(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    return float(sum(math.exp(-di / 13.0) - 1 if di < 0
                     else math.exp(di / 10.0) - 1 for di in d))


def main():
    results = {}
    pipes = {}   # cache per dataset
    for aid, cfg in ABLATION_CONFIGS.items():
        results[aid] = {}
        for ds in [1, 2, 3, 4]:
            run_dir = ROOT / "results/exp_ablation" / aid / f"FD00{ds}"
            ckpt_p = run_dir / "model_best.pt"
            res_p = run_dir / "result.json"
            if not ckpt_p.exists():
                print(f"[skip] {aid}/FD00{ds}: no checkpoint")
                continue
            if ds not in pipes:
                bc = BEST_CONFIG[ds]
                pipes[ds] = preprocess_pipeline(
                    str(ROOT / "data"), dataset_id=ds,
                    window_size=bc["window_size"], cap=125,
                    val_ratio=0.15, random_seed=42)
            pipe = pipes[ds]
            X_test, y_test = pipe["X_test"], pipe["y_test"]

            ck = torch.load(ckpt_p, map_location="cpu")
            L_enc = ck.get("n_encoder_layers",
                           BEST_CONFIG[ds]["n_encoder_layers"])
            model = AblationRULModel(
                n_features=pipe["n_features"], hidden_dim=128, num_heads=4,
                num_layers=2, dropout=0.3, sensor_dropout_p=0.1,
                n_encoder_layers=L_enc, **{k: v for k, v in cfg.items()
                                           if k.startswith("use_")})
            model.load_state_dict(ck["model_state"])
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(X_test), 256):
                    xb = torch.from_numpy(X_test[i:i + 256]).float()
                    preds.append(model(xb).squeeze(-1))
            pred = torch.cat(preds).numpy()
            rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
            score = phm08(pred, y_test)

            archived = json.load(open(res_p)) if res_p.exists() else {}
            arch_rmse = archived.get("test_rmse")
            ok = (arch_rmse is not None and abs(arch_rmse - rmse) < 0.05)
            results[aid][f"FD00{ds}"] = {
                "rmse": rmse, "phm08_score": score,
                "archived_rmse": arch_rmse, "rmse_match": bool(ok),
            }
            flag = "OK" if ok else f"MISMATCH(arch={arch_rmse})"
            print(f"{aid} FD00{ds}: RMSE={rmse:6.2f}  Score={score:9.1f}  [{flag}]")

    # per-ablation means
    print("\n--- Table 10 row data (per ablation) ---")
    for aid, dsr in results.items():
        scores = [v["phm08_score"] for v in dsr.values()]
        rmses = [v["rmse"] for v in dsr.values()]
        print(f"{aid}: mean RMSE={np.mean(rmses):.2f}  "
              f"scores={[round(s,1) for s in scores]}")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
