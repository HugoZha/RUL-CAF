#!/usr/bin/env python3
"""Five-seed CAS-Fuse stability audit (manuscript E1 / Tables 8-9 / Sec. 5.3).

Runs all datasets x seeds through the identical training loop, then aggregates
mean +/- std over *surviving* seeds and the collapse rate.  Written as a
wrapper around ``train.run_one`` so no training logic is duplicated.

Examples
--------
    python audit.py --datasets 1,2,3,4 --seeds 42,7,123,2026,3407
    python audit.py --datasets 1 --seeds 42 --epochs 20 --patience 3  # smoke test
"""

import argparse
import json
from pathlib import Path

import numpy as np

from configs import AUDIT_SEEDS, DATASETS  # noqa: E402
from train import run_one  # noqa: E402

HERE = Path(__file__).resolve().parent


def aggregate(results: list) -> dict:
    rmses = np.array([r["test_rmse"] for r in results], dtype=float)
    maes = np.array([r["test_mae"] for r in results], dtype=float)
    scores = np.array([r["phm08_score"] for r in results], dtype=float)
    collapsed = np.array([r["collapsed"] for r in results], dtype=bool)
    surv = ~collapsed
    return {
        "n_total": int(len(results)),
        "n_collapsed": int(collapsed.sum()),
        "collapse_rate": float(collapsed.mean()),
        "surv_rmse_mean": float(rmses[surv].mean()) if surv.any() else None,
        "surv_rmse_std": float(rmses[surv].std(ddof=0)) if surv.any() else None,
        "surv_mae_mean": float(maes[surv].mean()) if surv.any() else None,
        "surv_score_mean": float(scores[surv].mean()) if surv.any() else None,
        "collapsed_rmse": [float(x) for x in rmses[collapsed]],
        "per_seed": {
            f"seed_{r['seed']}": {
                "test_rmse": round(r["test_rmse"], 4),
                "test_mae": round(r["test_mae"], 4),
                "test_r2": round(r["test_r2"], 4),
                "phm08_score": round(r["phm08_score"], 2),
                "pred_std": r["pred_std"],
                "attention_entropy": round(r["attention_entropy"], 4),
                "collapsed": r["collapsed"],
            }
            for r in results
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(str(d) for d in DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in AUDIT_SEEDS))
    ap.add_argument("--split-dir", default=str(HERE / "splits"),
                    help="directory holding FD00X_split.npz (or 'none')")
    ap.add_argument("--out", default=str(HERE / "runs"))
    ap.add_argument("--result-file", default=str(HERE / "results" / "e1_aggregate_reproduced.json"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--data-dir", default=None,
                    help="C-MAPSS directory (default: reproduction/data)")
    args = ap.parse_args()

    datasets = [int(x) for x in args.datasets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    summary = {}
    for ds in datasets:
        per_seed = []
        for seed in seeds:
            split_file = None
            if args.split_dir.lower() != "none":
                candidate = Path(args.split_dir) / f"FD00{ds}_split.npz"
                split_file = str(candidate) if candidate.exists() else None
            r = run_one(ds, seed, out_root=args.out, split_file=split_file,
                        epochs=args.epochs, patience=args.patience,
                        device_pref=args.device, save_model=args.save_model,
                        data_dir=args.data_dir)
            per_seed.append(r)
        summary[f"FD00{ds}"] = aggregate(per_seed)
        print(f"\n[audit] FD00{ds}: {summary[f'FD00{ds}']['n_collapsed']}/"
              f"{summary[f'FD00{ds}']['n_total']} collapsed, "
              f"surv RMSE {summary[f'FD00{ds}']['surv_rmse_mean']} "
              f"+/- {summary[f'FD00{ds}']['surv_rmse_std']}\n", flush=True)

    out = Path(args.result_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[audit] wrote {out}")

    reference = HERE / "results" / "e1_aggregate.json"
    if reference.exists() and args.epochs is None:
        ref = json.load(open(reference))
        print("\n[audit] comparison with archived manuscript numbers:")
        for ds, agg in summary.items():
            r = ref.get(ds, {})
            print(f"  {ds}: RMSE reproduced {agg['surv_rmse_mean']:.3f} vs "
                  f"archived {r.get('surv_rmse_mean', float('nan')):.3f} | "
                  f"collapse {agg['n_collapsed']}/{agg['n_total']} vs "
                  f"{r.get('n_collapsed')}/{r.get('n_total')}")


if __name__ == "__main__":
    main()
