#!/bin/zsh
# run_all.sh — end-to-end reproduction driver.
#   ./run_all.sh          full five-seed audit (~2 days on Apple M-series)
#   ./run_all.sh --smoke  data + 1-epoch sanity check (~1 min)
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

echo "[run_all] $(date): preparing data and splits"
$PY prepare_data.py --verify

if [[ "$1" == "--smoke" ]]; then
    echo "[run_all] smoke test: FD001, seed 7, 1 epoch"
    $PY train.py --dataset 1 --seed 7 --epochs 1 --patience 1 \
        --split-file splits/FD001_split.npz
    echo "[run_all] smoke test done"
    exit 0
fi

echo "[run_all] $(date): five-seed CAS-Fuse audit (20 runs)"
$PY audit.py --datasets 1,2,3,4 --seeds 42,7,123,2026,3407
echo "[run_all] $(date): done. See results/e1_aggregate_reproduced.json"
