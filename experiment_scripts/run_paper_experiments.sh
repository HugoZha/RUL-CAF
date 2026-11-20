#!/bin/zsh
# run_paper_experiments.sh — E1 (multi-seed audit) then E2 (mechanism probes)
# Launched detached; logs to paper_exp/logs/
cd /Users/hugozha/project/RULPredictor_V2
mkdir -p paper_exp/logs

echo "[driver] start $(date)" >> paper_exp/logs/driver.log

echo "[driver] E1: 16 runs (4 datasets x 4 seeds)" >> paper_exp/logs/driver.log
/usr/bin/python3 paper_exp/e1_seeds.py --datasets 1,2,3,4 --seeds 7,123,2026,3407 \
    >> paper_exp/logs/e1.log 2>&1
echo "[driver] E1 finished $(date)" >> paper_exp/logs/driver.log

echo "[driver] E2: 6 instrumented runs" >> paper_exp/logs/driver.log
/usr/bin/python3 paper_exp/e2_probe_train.py --config all \
    >> paper_exp/logs/e2.log 2>&1
echo "[driver] E2 finished $(date)" >> paper_exp/logs/driver.log
echo "[driver] ALL DONE $(date)" >> paper_exp/logs/driver.log
