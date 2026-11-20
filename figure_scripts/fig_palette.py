#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_palette.py -- single source of truth for every colour in the paper figures.

All eight figures draw from the same five-colour family.  No figure script may
hard-code a hex value of its own; import from here instead.

  NAVY  #1F3B57   TEAL  #2A9D8F   AMBER #E9C46A
  CORAL #E76F51   GRAY  #8D99AE   INK   #2B2B2B

The four pastel fills are the family's light variants; grid/axis lines are
neutral greys derived from GRAY.  COND6 and the two ramps below are built by
interpolating *between* family colours, so even the multi-hue encodings stay
inside one continuous palette.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors

# ---- family -------------------------------------------------------------
NAVY = "#1F3B57"
TEAL = "#2A9D8F"
AMBER = "#E9C46A"
CORAL = "#E76F51"
GRAY = "#8D99AE"
INK = "#2B2B2B"
WHITE = "#FFFFFF"

# ---- light variants -----------------------------------------------------
NAVY_L = "#DDE4EE"
TEAL_L = "#DDF0EC"
AMBER_L = "#FCF1D8"
CORAL_L = "#FBE3DC"
NEUTRAL_L = "#EEF1F4"
GRID = "#E8EBEF"
AXIS = "#B9C0CC"

PRIMARY = (NAVY, TEAL, AMBER, CORAL, GRAY)


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _mix(a, b, t):
    ca, cb = _hex_rgb(a), _hex_rgb(b)
    return "#%02X%02X%02X" % tuple(round(255 * (x + (y - x) * t))
                                   for x, y in zip(ca, cb))


def ramp(colors, name="family", n=256):
    """A LinearSegmentedColormap built only from family colours."""
    return mcolors.LinearSegmentedColormap.from_list(name, list(colors), N=n)


# Six ordered operating conditions: sampled along the NAVY -> TEAL -> AMBER ->
# CORAL family ramp, so the six scatter colours read as one scale, not six
# unrelated swatches.
_ANCHORS = [NAVY, TEAL, AMBER, CORAL]
COND6 = []
for i in range(6):
    u = i / 5.0 * (len(_ANCHORS) - 1)
    k = min(int(u), len(_ANCHORS) - 2)
    COND6.append(_mix(_ANCHORS[k], _ANCHORS[k + 1], u - k))

# Sequential encodings used by the heatmap-style figures.
CMAP_HEAT = (NAVY, TEAL, AMBER, AMBER_L)      # attention maps (Fig. 8)
CMAP_SEQ = (TEAL_L, AMBER, CORAL)             # collapse-rate grids (Fig. 3)

# Status colours, reused verbatim across all figures.
OK = TEAL        # surviving / healthy
FAIL = CORAL     # collapsed / at-risk
NEUTRAL = GRAY   # reference lines, axes


def rc_params():
    """Shared matplotlib rc so every figure uses identical typography."""
    return {
        "font.family": "sans-serif",
        # DejaVu Sans is bundled with matplotlib and covers the Unicode
        # arrows / set symbols used in the schematic figures; putting it first
        # keeps every figure identical and avoids missing-glyph boxes.
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": WHITE,
    }


def clean(ax, grid="y"):
    """Flat PPT-style axes: no top/right spines, faint grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)


if __name__ == "__main__":
    print("family:", PRIMARY)
    print("cond6 :", COND6)
