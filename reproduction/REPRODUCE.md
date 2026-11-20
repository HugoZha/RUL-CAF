# Full reproduction guide — RUL-CAF

This folder contains everything needed to reproduce the experimental results
of the manuscript *"Constant predictions in multi-condition aero-engine RUL
models: causes, early detection, and condition-adaptive network design"*:

- the **data** (NASA C-MAPSS, distributed via the `data-v1` GitHub release),
- the **exact train/validation split indices** for all four subsets,
- the **model / training / evaluation / stability-audit code**, and
- the **archived result files** behind Tables 7–10 and Figs. 2–9.

All code here is a self-contained extraction of the scripts that actually
produced the manuscript numbers. It is not a reimplementation from the paper
text: with the pinned library versions and the same device, the training
trajectory is bit-identical (verified: FD001 / seed 7, epoch-1 validation RMSE
= 77.56 in both the original log and this package).

---

## 1. What is reproduced, and where

| Manuscript item | Produced by | Archived reference |
| --- | --- | --- |
| Sec. 3.1 protocol (cap / normalisation / split / test) | `src/data_loader.py`, `prepare_data.py` | `splits/FD00X_split.npz` |
| Sec. 5.1 CAS-Fuse configurations (w, L) | `configs.py` | `configs.py` |
| Tables 8–9, five-seed audit (RMSE, PHM08, collapse rate) | `audit.py` → `train.py` | `results/e1_aggregate.json`, `results/e1_seed_summary.json` |
| Sec. 5.3 R5 tripwires A–C | `src/metrics.py`, `train.py` | `results/collapse_summary.json` |
| Table 10 ablations, PHM08 re-scoring | archived outputs | `results/ablation_phm08.json`, `results/phm08_scores_checkpoint.json` |
| Sec. 4 mechanism probes (attention entropy, condition probe) | `src/metrics.py` | `results/probe_results.json`, `results/e2_escape_analysis.json` |
| Sec. 4.6 collapse matrix (archive-wide scan) | archived outputs | `results/collapse_matrix.csv`, `results/collapse_summary.json` |

---

## 2. Prerequisites

```bash
cd reproduction
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

Tested with Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, pandas 2.3.3,
scikit-learn 1.6.1 on macOS / Apple M-series (MPS backend). CPU works but is
~2–3× slower. The scripts auto-select MPS when available, otherwise CPU;
use `--device cpu|mps|cuda` to override.

**Time budget.** One full CAS-Fuse run: 60–130 min on FD001–FD003 and
70–350 min on FD004 (Apple M-series, MPS). The complete five-seed audit over
four subsets (20 runs) takes roughly two days on one consumer device. A
two-epoch smoke test takes under a minute.

---

## 3. Step 1 — obtain the C-MAPSS data

The raw files are ~43 MB uncompressed, so they are kept outside git:

```bash
python3 prepare_data.py --verify
```

This downloads `c-mapss.zip` from
<https://github.com/HugoZha/RUL-CAF/releases/tag/data-v1>, extracts it into
`reproduction/data/`, verifies the SHA256 of all 13 text files against
`data_manifest.json`, and builds the split files.

If you already have NASA's original C-MAPSS files:

```bash
python3 prepare_data.py --data-dir /path/to/cmapss --skip-download --verify
```

The dataset is U.S. Government public-domain material distributed by the NASA
Prognostics Data Repository; cite Saxena & Goebel (2008) and use at your own
risk (see the repository README).

---

## 4. Step 2 — the exact dataset split

`prepare_data.py` writes one `splits/FD00X_split.npz` per subset containing
`train_idx`, `val_idx` and a JSON manifest. The rule (Sec. 3.1 / 6.1) is:

1. Compute training RUL as `max_cycle − cycle`, clipped at **cap = 125**.
2. Select features dynamically: 3 operating settings plus sensors whose
   training variance exceeds `1e-6` (17 / 24 / 18 / 24 features on
   FD001 / FD002 / FD003 / FD004).
3. Min-max normalise using **training statistics only**.
4. Slide a window of length `w` over every training engine with step 1; the
   label is the RUL at the window's last cycle.
5. Split those windows **85/15** with `sklearn.train_test_split(...,
   random_state=42)` → `train_idx` / `val_idx`.
6. Test set: the **last window of each test engine**, one prediction per
   engine, capped RUL from `RUL_FD00X.txt`. Engines with fewer rows than `w`
   are skipped, exactly as in the original experiments.

Resulting partitions (all four subsets):

| Subset | w | Windows total | Train | Val | Test engines | Features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 40 | 16,731 | 14,221 | 2,510 | 96 | 17 |
| FD002 | 20 | 48,819 | 41,496 | 7,323 | 259 | 24 |
| FD003 | 30 | 21,820 | 18,547 | 3,273 | 100 | 18 |
| FD004 | 60 | 46,558 | 39,574 | 6,984 | 248 | 24 |

The indices are shipped so the split can be audited without relying on the
scikit-learn version. To rebuild them from raw files alone, rerun
`prepare_data.py`; `src/data_loader.build_split_indices()` exposes the recipe.

---

## 5. Step 3 — smoke test (≈1 min)

```bash
python3 train.py --dataset 1 --seed 7 --epochs 1 --patience 1 \
    --split-file splits/FD001_split.npz
```

Expected console output includes:

```
[preprocess] FD001: train=14221 val=2510 test=96 | features=17 | window=40 | cap=125
  ep  1 val=77.56 best=77.56 ...
```

Outputs are written to `runs/FD001/seed_7/`: `result.json`, `history.json`
(per-epoch RMSE / prediction-std / plateau counter) and `predictions.npy`.
The short run is expected to collapse — collapse after 1–2 epochs is exactly
the phenomenon the paper studies, not a bug.

---

## 6. Step 4 — one full training / evaluation run

```bash
python3 train.py --dataset 1 --seed 42 --split-file splits/FD001_split.npz
python3 train.py --dataset 4 --seed 7  --split-file splits/FD004_split.npz
```

The configuration is taken from `configs.py` (do not edit it when verifying):

| Subset | w | Transformer layers L | Heads | Hidden | Dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | 40 | 1 | 4 | 128 | 0.3 |
| FD002 | 20 | 3 | 4 | 128 | 0.3 |
| FD003 | 30 | 1 | 4 | 128 | 0.3 |
| FD004 | 60 | 3 | 4 | 128 | 0.3 |

Shared: AdamW (lr 1e-4, weight decay 1e-4), cosine schedule, MSE loss,
batch 32, up to 150 epochs, early stopping patience 20, gradient clip 1.0,
sensor dropout 0.1, difference augmentation, dual avg+max pooling. Parameter
counts: **1.077 M** (single-condition) and **1.474 M** (multi-condition).

`train.py` reports RMSE, MAE, R², prediction std, segment-wise RMSE/MAE, the
NASA PHM08 asymmetric score, mean normalised attention entropy, and all three
R5 tripwire flags, and saves them to `result.json`.

---

## 7. Step 5 — the five-seed stability audit (Tables 8–9)

```bash
python3 audit.py --datasets 1,2,3,4 --seeds 42,7,123,2026,3407
```

This runs all 20 configurations in sequence and writes
`results/e1_aggregate_reproduced.json`, printing a side-by-side comparison
against the archived numbers. Expected outcome (archived values):

| Subset | Surviving-seed RMSE | Collapse rate | PHM08 (survivors) |
| --- | ---: | ---: | ---: |
| FD001 | 13.66 ± 0.10 | 0/5 | 307 |
| FD002 | 19.43 ± 0.44 | 0/5 | 2 855 |
| FD003 | 16.09 ± 0.52 | 0/5 | 1 050 |
| FD004 | 19.53 ± 0.07 | 6/8 | 1 783–2 448 |

The FD004 collapse rate is **not** a failure of the reproduction: FD004 sits
in a bistable region where six of eight seeds converge to the closed-form
constant predictor (RMSE ≈ 43.7, prediction std < 1e-4). The paper's R5
protocol is to detect such runs with the tripwires, reseed, and report the
surviving-seed statistics. Small numeric differences are possible if you
change device (`--device cpu` vs MPS) or library versions; use the pinned
versions for exact agreement.

To reproduce just one subset or a quick partial audit:

```bash
python3 audit.py --datasets 1 --seeds 7,123 --epochs 20 --patience 3
```

(With truncated epochs the run is under-trained and will be flagged collapsed;
this only checks that the plumbing works.)

---

## 8. Step 6 — interpret the stability tripwires

Implemented in `src/metrics.py` and called at the end of every run:

- **Tripwire A** — prediction std < 0.01 cycles, or test RMSE within 2 % of
  the constant-predictor RMSE `std(y_test)` (Eq. (1)).
- **Tripwire B** — mean normalised attention entropy > 0.99
  (healthy: 0.70–0.93; collapsed: ≈1.0000).
- **Tripwire C** — during training, validation RMSE stays within 0.5 of
  `std(y_val)` while prediction std < 0.1 for five consecutive epochs;
  recorded as `tripwire_C_plateau` in `history.json`-adjacent `result.json`.

A run is flagged `collapsed` if any of A/B/C fires or RMSE > 40. Archived
collapse statistics for the 161-run archive are in
`results/collapse_matrix.csv` and `results/collapse_summary.json`;
per-epoch mechanism probes are in `results/probe_results.json` and
`results/e2_escape_analysis.json`.

---

## 9. File map

```
reproduction/
├── README.md                    quick orientation
├── REPRODUCE.md / REPRODUCE_CN.md  this tutorial (EN / 中文)
├── requirements.txt             pinned environment
├── configs.py                   CAS-Fuse configurations + tripwire thresholds
├── data_manifest.json           SHA256 of the 13 C-MAPSS text files
├── prepare_data.py              download / verify / build splits
├── splits/FD00X_split.npz       exact 85/15 window indices + manifest
├── src/
│   ├── data_loader.py           C-MAPSS loading, capping, scaling, windows
│   ├── model.py                 TCN + BiGRU + Transformer (PureRULModel)
│   └── metrics.py               RMSE/MAE/R²/PHM08 + R5 tripwires
├── train.py                     one train/evaluate run
├── audit.py                     multi-seed audit + aggregation
├── run_all.sh                   end-to-end driver
└── results/                     archived outputs behind the paper's tables
```

---

## 10. Troubleshooting

- **"No C-MAPSS files"** — run `prepare_data.py` first, or pass `--data-dir`.
- **Download is slow** — the release asset is hosted on GitHub's asset CDN;
  resume with a download manager, or copy NASA's original files into
  `data/` and pass `--skip-download`.
- **Different numbers** — check the pinned versions and use MPS; CPU/MPS and
  CUDA differ slightly in floating point. Also ensure you pass the shipped
  `--split-file` so the partition is identical.
- **FD004 keeps collapsing** — expected; this is the paper's central finding.
  Increase the number of seeds (R5 audit-and-reseed) rather than changing the
  configuration.
