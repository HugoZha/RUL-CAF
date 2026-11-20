#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_paper_figures_extra.py -- Fig. 8 (layer-1 attention heatmaps).

Fig. 1 and Fig. 6 are drawn as editable PowerPoint files by
gen_fig1_fig6_pptx.py; keeping those two out of this script avoids two
competing definitions of the same figure.  All colours come from
fig_palette.py.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fig_palette import (CMAP_HEAT, ramp, rc_params, clean)

ROOT = Path("/Users/hugozha/project/RULPredictor_V2")
OUT = Path("/Users/hugozha/WorkBuddy/2026-09-27-12-28-14/paper/figures")
plt.rcParams.update(rc_params())


def fig8():
    sys.path.insert(0, str(ROOT))
    from exp_final_patent import PureRULModel
    from src.data_loader import preprocess_pipeline

    pipe = preprocess_pipeline(str(ROOT / "data"), dataset_id=2, window_size=20,
                               cap=125, val_ratio=0.15, random_seed=42)
    X_test = pipe["X_test"]
    cmap = ramp(CMAP_HEAT, "family_heat")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, (tag, title) in zip(axes, [
            ("FD002_L3", "Surviving run (L=3)"),
            ("FD002_L2", "Failed run (L=2)")]):
        ck = torch.load(ROOT / f"results/exp_p10_no_adv/FD002/"
                        f"{'layers_3' if 'L3' in tag else 'layers_2'}/model_best.pt",
                        map_location="cpu")
        L = ck.get("n_encoder_layers") or ck["config"]["L"]
        model = PureRULModel(n_features=pipe["n_features"], hidden_dim=128,
                             num_heads=4, num_layers=2, dropout=0.3,
                             sensor_dropout_p=0.1, n_encoder_layers=L)
        model.load_state_dict(ck["model_state"])
        model.eval()
        caps = []
        model.encoder_layers[0].attn_sublayer.register_forward_hook(
            lambda m, i, o: caps.append(o[1].detach()))
        with torch.no_grad():
            model(torch.from_numpy(X_test[:32]).float())
        A_disp = torch.cat(caps).mean(dim=(0, 1)).numpy()
        caps_full = []
        model.encoder_layers[0].attn_sublayer.register_forward_hook(
            lambda m, i, o: caps_full.append(o[1].detach()))
        with torch.no_grad():
            for i in range(0, len(X_test), 128):
                model(torch.from_numpy(X_test[i:i + 128]).float())
        A_per = torch.cat(caps_full).numpy()
        Lmax = np.log2(np.arange(1, 21))
        H = -(A_per * np.log2(np.clip(A_per, 1e-12, None))).sum(-1)
        # skip the first query position (log2(1) = 0) to avoid a 0/0
        ent = float((H[..., 1:] / Lmax[1:]).mean())
        im = ax.imshow(A_disp, cmap=cmap, vmin=0, vmax=1.0)
        ax.set_title(f"{title}\nmean normalised row entropy {ent:.4f}", fontsize=8.5)
        ax.set_xlabel("Key position (cycle)")
        ax.set_ylabel("Query position")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "fig8_attention_maps.png")
    fig.savefig(OUT / "fig8_attention_maps.pdf")
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    fig8()
    print("fig8 done ->", OUT)
