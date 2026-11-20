#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_paper_figures.py -- data-driven figures (2, 3, 4, 5, 9).

Every colour comes from fig_palette.py.  Fig. 1 and Fig. 6 are drawn as
editable PowerPoint files by gen_fig1_fig6_pptx.py.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))
from fig_palette import (NAVY, TEAL, AMBER, CORAL, GRAY, INK, WHITE,
                         NAVY_L, TEAL_L, AMBER_L, CORAL_L, AXIS,
                         COND6, CMAP_SEQ, ramp, rc_params, clean)

ROOT = Path("/Users/hugozha/project/RULPredictor_V2")
OUT = Path("/Users/hugozha/WorkBuddy/2026-09-27-12-28-14/paper/figures")
if not OUT.exists():
    OUT.mkdir(parents=True)

plt.rcParams.update(rc_params())


def e2log(tag):
    return json.load(open(ROOT / f"results/e2_probe/{tag}/probe_log.json"))


# ---------------- Fig 2: fingerprint ----------------
def fig2():
    log_h = e2log("FD002_L3")
    log_c = e2log("FD002_L2_s7")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    ax.plot([e["ep"] for e in log_h["epochs"]],
            [e["val_rmse"] for e in log_h["epochs"]],
            color=TEAL, lw=1.6, label="Surviving run (L=3)", zorder=3)
    ax.plot([e["ep"] for e in log_c["epochs"]],
            [e["val_rmse"] for e in log_c["epochs"]],
            color=CORAL, lw=1.6, ls="--", label="Collapsing run (L=2, seed 7)", zorder=3)
    ax.axhline(41.97, color=GRAY, lw=0.9, ls=":")
    ax.text(101, 43.8, r"$\mathrm{std}(y_{val})=41.97$", fontsize=8, color=GRAY)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_ylim(10, 75)
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    ax.set_title("(a) Validation trajectory")

    ax = axes[1]
    for tag, color, lbl in [("FD002_L3", TEAL, "Surviving (RMSE 18.6)"),
                            ("FD002_L2", CORAL, "Failed run (RMSE 43.8)")]:
        z = np.load(ROOT / f"paper_exp/probe_feats/{tag}.npz")
        ax.scatter(z["y"], z["preds"], s=9, alpha=0.55, color=color, label=lbl,
                   edgecolors="none", zorder=3)
    lim = [0, 130]
    ax.plot(lim, lim, color=GRAY, lw=0.8, ls=":", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    clean(ax, grid=None)
    ax.set_title("(b) Test predictions, FD002")
    fig.savefig(OUT / "fig2_fingerprint.png")
    fig.savefig(OUT / "fig2_fingerprint.pdf")
    plt.close(fig)


# ---------------- Fig 3: prevalence ----------------
def fig3():
    df = pd.read_csv(ROOT / "paper_exp/collapse_matrix.csv")
    cmap = ramp(CMAP_SEQ, "collapse")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.subplots_adjust(wspace=0.38, top=0.80)

    pooled = {
        ("FD002", 1): (0, 1), ("FD002", 2): (2, 3), ("FD002", 3): (0, 3), ("FD002", 4): (1, 1),
        ("FD004", 1): (1, 2), ("FD004", 2): (0, 1), ("FD004", 3): (6, 8), ("FD004", 4): (0, 1),
    }
    ax = axes[0]
    data = np.full((2, 4), np.nan)
    annot = [[""] * 4 for _ in range(2)]
    for i, ds in enumerate(["FD002", "FD004"]):
        for L in [1, 2, 3, 4]:
            c, n = pooled[(ds, L)]
            data[i, L - 1] = c / n
            annot[i][L - 1] = f"{c}/{n}"
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(4):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=9,
                    color=INK if data[i, j] < 0.6 else WHITE, weight="bold")
    ax.set_xticks(range(4), [f"L={x}" for x in [1, 2, 3, 4]])
    ax.set_yticks(range(2), ["FD002\n(ws=20)", "FD004\n(ws=60)"])
    ax.set_title("(a) Encoder depth", fontsize=9, pad=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    p7 = df[(df.family == "exp_p7_window_search")]
    ws_idx = {20: 0, 30: 1, 40: 2, 60: 3}
    ax = axes[1]
    data = np.full((2, 4), np.nan)
    annot = [[""] * 4 for _ in range(2)]
    for _, r in p7.iterrows():
        if r.dataset in ("FD002", "FD004") and not np.isnan(r.ws):
            i = 0 if r.dataset == "FD002" else 1
            j = ws_idx.get(int(r.ws))
            if j is None:
                continue
            data[i, j] = 1.0 if r.collapsed else 0.0
            annot[i][j] = f"{r.test_rmse:.1f}"
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(4):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8,
                        color=INK if data[i, j] < 0.6 else WHITE, weight="bold")
    ax.set_xticks(range(4), [str(w) for w in [20, 30, 40, 60]])
    ax.set_yticks(range(2), ["", ""])   # row labels already given in (a)
    ax.set_title("(b) Window size", fontsize=9, pad=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im if False else plt.cm.ScalarMappable(cmap=cmap),
                        ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Failure rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "fig3_prevalence.png")
    fig.savefig(OUT / "fig3_prevalence.pdf")
    plt.close(fig)


# ---------------- Fig 4: t-SNE ----------------
def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.0))
    for row, (tag, status) in enumerate([("FD002_L3", "Surviving"),
                                         ("FD002_L2", "Failed")]):
        z = np.load(ROOT / f"paper_exp/probe_feats/{tag}.npz")
        emb = TSNE(n_components=2, init="pca", random_state=0,
                   perplexity=30).fit_transform(z["h_pool"])
        for col, (key, ttl) in enumerate([("cond", "operating condition"),
                                          ("buckets", "RUL bucket")]):
            ax = axes[row, col]
            if key == "cond":
                for c in range(6):
                    m = z[key] == c
                    ax.scatter(emb[m, 0], emb[m, 1], s=8, color=COND6[c],
                               alpha=0.85, edgecolors="none",
                               label=f"cond {c+1}" if (row == 0 and col == 0) else None)
                if row == 0 and col == 0:
                    ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.1)
            else:
                sc = ax.scatter(emb[:, 0], emb[:, 1], c=z[key], s=8,
                                cmap=ramp((NAVY, TEAL, AMBER, AMBER_L), "heat"),
                                alpha=0.85, edgecolors="none")
                cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
                cb.ax.tick_params(labelsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{status} — by {ttl}", fontsize=9)
    fig.savefig(OUT / "fig4_tsne.png")
    fig.savefig(OUT / "fig4_tsne.pdf")
    plt.close(fig)


# ---------------- Fig 5: escape race ----------------
def fig5():
    runs = [("FD002_L2", "FD002 L=2 s42", False),
            ("FD002_L3", "FD002 L=3 s42", False),
            ("FD002_L2_s7", "FD002 L=2 s7", True),
            ("FD004_L1", "FD004 L=1 s42", False),
            ("FD004_L3", "FD004 L=3 s42", False),
            ("FD004_L3_s123", "FD004 L=3 s123", False)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for tag, lbl, coll in runs:
        log = e2log(tag)
        eps = log["epochs"]
        ep = [e["ep"] for e in eps]
        val = [e["val_rmse"] for e in eps]
        ent = [np.mean(e["attn_entropy"]) for e in eps]
        gt = [e["grad_norm"]["tcn"] for e in eps]
        cond = [e.get("cond_acc_cv", np.nan) for e in eps]
        color = CORAL if coll else TEAL
        style = "--" if coll else "-"
        axes[0, 0].plot(ep, val, style, color=color, lw=1.3, label=lbl, zorder=3)
        axes[0, 1].plot(ep, ent, style, color=color, lw=1.3, zorder=3)
        axes[1, 0].semilogy(ep, gt, style, color=color, lw=1.3, zorder=3)
        axes[1, 1].plot(ep, cond, style, color=color, lw=1.3, zorder=3)
    axes[0, 0].axhline(41.97, color=GRAY, lw=0.8, ls=":")
    axes[0, 0].set_ylabel("Val RMSE")
    axes[0, 0].set_ylim(10, 75)
    axes[0, 0].set_title("(a) Validation RMSE")
    axes[0, 1].axhline(1.0, color=GRAY, lw=0.8, ls=":")
    axes[0, 1].set_ylabel("Attention entropy (norm.)")
    axes[0, 1].set_title("(b) Attention entropy")
    axes[1, 0].set_ylabel("TCN gradient norm")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_title("(c) TCN gradient (log scale)")
    axes[1, 1].set_ylabel("Condition probe acc. (CV)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_title("(d) Condition decoding")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes.flat:
        clean(ax, grid=None)
    fig.savefig(OUT / "fig5_escape_race.png")
    fig.savefig(OUT / "fig5_escape_race.pdf")
    plt.close(fig)


# ---------------- Fig 9 -> Fig 7: seed strip ----------------
def fig9():
    agg = json.load(open(ROOT / "paper_exp/e1_aggregate.json"))
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    rng = np.random.RandomState(0)
    for i, ds in enumerate(datasets):
        surv = [v for v in agg[ds]["per_seed"].values() if v < 40]
        coll = [v for v in agg[ds]["per_seed"].values() if v >= 40]
        x0 = i * 1.0
        ax.scatter(rng.rand(len(surv)) * 0.22 - 0.11 + x0, surv, s=30,
                   color=TEAL, zorder=3, edgecolors=WHITE, linewidths=0.6)
        if coll:
            ax.scatter(rng.rand(len(coll)) * 0.22 - 0.11 + x0 + 0.24, coll,
                       s=34, color=CORAL, zorder=3, marker="x", lw=1.4)
        if surv:
            ax.hlines(np.mean(surv), x0 - 0.16, x0 + 0.16, color=NAVY, lw=1.8, zorder=4)
    ax.set_xticks(range(4), datasets)
    ax.set_ylabel("Test RMSE per seed")
    ax.set_ylim(10, 50)
    ax.scatter([], [], s=30, color=TEAL, label="surviving seed")
    ax.scatter([], [], s=30, color=CORAL, marker="x", label="failed seed")
    ax.hlines([], 0, 0, color=NAVY, lw=1.8, label="surviving mean")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    clean(ax, grid="y")
    fig.savefig(OUT / "fig9_seed_boxplot.png")
    fig.savefig(OUT / "fig9_seed_boxplot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig2(); print("fig2 done")
    fig3(); print("fig3 done")
    fig4(); print("fig4 done")
    fig5(); print("fig5 done")
    fig9(); print("fig9 done")
    print("All ->", OUT)
