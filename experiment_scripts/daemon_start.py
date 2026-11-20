#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daemon_start.py — Double-fork daemonized supervisor for the paper experiment
chain (E1 multi-seed audit -> E2 mechanism probes).

Why: processes launched from the assistant session are killed when the host
app reaps its process tree (observed twice). A double-fork daemon detaches
from the session/process group and is reparented to launchd, surviving app
restarts. The supervisor loop resumes automatically: both e1_seeds.py and
e2_probe_train.py skip runs whose result files already exist.

Completeness = all 16 E1 results + all 6 E2 probe logs exist.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "paper_exp" / "logs"

E1_DATASETS = [1, 2, 3, 4]
E1_SEEDS = [7, 123, 2026, 3407]
E2_TAGS = ["FD002_L2", "FD002_L3", "FD002_L2_s7",
           "FD004_L1", "FD004_L3", "FD004_L3_s123"]


def e1_done():
    return all((ROOT / f"results/e1_seeds/FD00{ds}/seed_{s}/result.json").exists()
               for ds in E1_DATASETS for s in E1_SEEDS)


def e2_done():
    return all((ROOT / f"results/e2_probe/{tag}/probe_log.json").exists()
               for tag in E2_TAGS)


def log(msg):
    with open(LOG / "driver.log", "a") as f:
        f.write(f"[supervisor] {msg} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    os.chdir(ROOT)
    while True:
        if e1_done() and e2_done():
            log("ALL EXPERIMENTS COMPLETE.")
            return
        log("(re)starting experiment chain")
        with open(LOG / "e1.log", "a") as f1:
            subprocess.run(
                ["/usr/bin/python3", "paper_exp/e1_seeds.py",
                 "--datasets", ",".join(map(str, E1_DATASETS)),
                 "--seeds", ",".join(map(str, E1_SEEDS))],
                stdout=f1, stderr=subprocess.STDOUT)
        with open(LOG / "e2.log", "a") as f2:
            subprocess.run(
                ["/usr/bin/python3", "paper_exp/e2_probe_train.py", "--config", "all"],
                stdout=f2, stderr=subprocess.STDOUT)
        log("chain iteration returned; re-checking completeness in 60s")
        time.sleep(60)


if __name__ == "__main__":
    LOG.mkdir(parents=True, exist_ok=True)
    # ---- double-fork daemonization ----
    if os.fork() > 0:
        sys.exit(0)                     # parent exits
    os.setsid()                          # new session, no controlling tty
    if os.fork() > 0:
        sys.exit(0)                     # first child exits; daemon reparented to launchd
    sys.stdout.flush(); sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    logf = os.open(str(LOG / "daemon.log"),
                   os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(logf, 1)
    os.dup2(logf, 2)
    log(f"daemon started, pid={os.getpid()}, ppid={os.getppid()}")
    main()
