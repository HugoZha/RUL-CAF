# Reproduction package

Self-contained code, data pointers, exact dataset splits and archived results
for the RUL-CAF manuscript.

**Start here:** [REPRODUCE.md](REPRODUCE.md) (English) or
[REPRODUCE_CN.md](REPRODUCE_CN.md) (中文) — full end-to-end tutorial.

Quick start:

```bash
pip install -r requirements.txt
python3 prepare_data.py --verify          # download + verify + build splits
./run_all.sh --smoke                       # ~1 min sanity check
python3 audit.py                           # full five-seed audit (~2 days)
```

Key contents:

- `src/data_loader.py` — C-MAPSS loading, RUL capping, scaling, sliding windows
- `src/model.py` — CAS-Fuse backbone (TCN + BiGRU + Transformer)
- `src/metrics.py` — RMSE/MAE/R²/PHM08 and the three R5 collapse tripwires
- `configs.py` — exact per-dataset configurations and thresholds
- `splits/FD00X_split.npz` — exact 85/15 window indices (seed 42)
- `results/` — archived outputs behind the manuscript's tables and figures

The raw C-MAPSS text files are ~43 MB and live in the
[`data-v1` release](https://github.com/HugoZha/RUL-CAF/releases/tag/data-v1),
not in git. `data_manifest.json` holds their SHA256 checksums.
