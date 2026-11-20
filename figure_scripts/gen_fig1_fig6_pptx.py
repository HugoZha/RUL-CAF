#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_fig1_fig6_pptx.py -- Fig. 1 and Fig. 6 rebuilt as editable PowerPoint.

One geometry definition drives two outputs:

  * ``<name>.pptx``  -- native shapes, text boxes and connectors.  Every arrow
    is attached to a shape connection site via begin_connect/end_connect, so
    in PowerPoint the arrow stays glued to the box edge when the box is moved.
  * ``<name>.png``   -- the same geometry rasterised with matplotlib for the
    LaTeX manuscript (high DPI, identical coordinates).

No colour literals here: everything comes from fig_palette.py.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
from fig_palette import (NAVY, TEAL, AMBER, CORAL, GRAY, INK, WHITE,
                         NAVY_L, TEAL_L, AMBER_L, CORAL_L, NEUTRAL_L,
                         AXIS, rc_params)

HERE = Path(__file__).parent
OUT = HERE / "figures"
PPT_OUT = HERE / "figures_ppt"
for d in (OUT, PPT_OUT):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(rc_params())


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


# =====================================================================
# Geometry -- Fig. 1: hybrid temporal backbone
# =====================================================================
F1 = {
    # 5.8 / 11.6 == 5.4 / 10.8 == 0.5 in per logical unit, so widening the
    # canvas to the right does NOT move any box; it only creates room for the
    # two horizontal callout leaders.
    "size_in": (5.8, 7.0),
    "logic": (11.6, 15.6),
    "ylim": (2.35, 15.55),     # tight crop: no dead space above/below
    "cx": 5.4,
    "box_w": 5.4,              # wide enough for the longest label at 7.8 pt
    "rail1": 1.72,
    "rail2": 2.28,
    "font_pt": 7.8,
    # (cy, half_h, label, edge_colour, fill_colour)
    "boxes": [
        (14.9, 0.30, "Sensor window   X in R^(w x C)", GRAY, NEUTRAL_L),
        (13.8, 0.30, "First-order differencing (2C)", GRAY, AMBER_L),
        (12.7, 0.30, "1x1 conv projection + GELU", GRAY, WHITE),
        (11.6, 0.30, "Positional encoding + dropout", GRAY, WHITE),
        (10.1, 0.50, "TCN front-end\n5 causal dilated blocks (d=1...16)", TEAL, TEAL_L),
        (8.4, 0.50, "BiGRU x2 + linear projection", NAVY, NAVY_L),
        (6.7, 0.50, "Transformer encoder xL\n4-head causal attention", CORAL, CORAL_L),
        (5.35, 0.30, "Pooling [ avg ; max ]", GRAY, WHITE),
        (4.25, 0.30, "MLP head", GRAY, WHITE),
        (3.15, 0.30, "RUL estimate", NAVY, NAVY_L),
    ],
    "merge1": 9.25,     # between TCN (bottom 9.60) and BiGRU (top 8.90)
    "merge2": 7.55,     # between BiGRU (bottom 7.90) and Transformer (top 7.20)
    "merge_r": 0.13,
    "notes": [
        # (anchor box index, text, note CENTRE x, note w, colour).  The label
        # y is deliberately NOT a free parameter: both renderers place the
        # label at the anchor box's own centre y, so every leader is exactly
        # horizontal.  A filled dot on the box's right edge marks the exact
        # target of the arrowhead -- the association cannot be misread.
        (0, "window length  w\n1 condition: 40 / 30\n6 conditions: 20 / 60",
         10.15, 2.40, CORAL),
        (6, "encoder depth  L\n1 condition: 1\n6 conditions: 3",
         10.15, 2.40, CORAL),
    ],
    "note_h": 0.95,
    "note_pt": 7.0,
    "note_gap": 0.10,   # gap between the label's left edge and the arrow tail
}


# =====================================================================
# Geometry -- Fig. 6: qualitative stability map
# =====================================================================
F6 = {
    "size_in": (6.6, 4.1),
    "logic": (10.0, 100.0),
    # Negative margins hold the two axis labels ON the canvas, so the PPT and
    # the PNG use exactly the same coordinate system with no auto-placement.
    "xlim": (-2.0, 10.2),
    "ylim": (-14.0, 100.0),
    "b1": lambda x: 72 - 1.4 * x,
    "b2": lambda x: 38 - 2.0 * x,
    # (label, colour, text CENTRE x, text CENTRE y) -- shared verbatim by both
    # renderers so PPT and PNG cannot drift apart.
    "regions": [
        ("trains reliably", TEAL, 8.40, 88.0),
        ("depends on seed", NAVY, 7.60, 47.0),
        ("usually fails", CORAL, 1.60, 3.0),
    ],
    # (point x, point y, label, label centre x, label centre y, ha)
    # Each label sits within ~0.4 units of its dot so the association is
    # unambiguous; no two labels share a horizontal band.
    "points": [
        (0.8, 96.0, "FD001/003  (0/81 failed)", 1.30, 93.0, "left"),
        (5.9, 95.0, "FD002  L=3  (0/5)", 6.30, 97.0, "left"),
        (5.9, 40.0, "FD002  L=2  (2/3 recovered)", 5.55, 43.5, "right"),
        (5.9, 7.0, "FD002  L=4  (3/3 failed)", 5.55, 11.5, "right"),
        (9.2, 25.0, "FD004  L=3  (6/8)", 8.85, 30.0, "right"),
        (9.2, 60.0, "FD004  L=1  (1/2)", 8.85, 64.0, "right"),
    ],
    # (text, text centre x, text centre y, text w,
    #  arrow start x, arrow start y, arrow target x, arrow target y)
    "callouts": [
        ("no condition cue\nrecovers at once",
         2.35, 62.0, 2.60, 3.65, 64.0, 0.90, 93.0),
        ("recovery needs a sudden\nTCN gradient jump before\nearly stopping",
         2.45, 22.0, 3.00, 4.60, 18.0, 9.05, 26.0),
    ],
}


# =====================================================================
# Coordinate helpers (logical -> inches / matplotlib)
# =====================================================================
def make_map(size_in, logic, ylim=None, xlim=None):
    W, H = size_in
    LX, LY = logic
    y0, y1 = ylim if ylim is not None else (0.0, LY)
    x0, x1 = xlim if xlim is not None else (0.0, LX)
    return {
        "X": lambda x: Inches((x - x0) * W / (x1 - x0)),
        "Y": lambda y: Inches((y1 - y) * H / (y1 - y0)),
        "WI": lambda w: Inches(w * W / (x1 - x0)),
        "HI": lambda h: Inches(h * H / (y1 - y0)),
        "xlim": (x0, x1), "ylim": (y0, y1),
        "W": W, "H": H, "LX": LX, "LY": LY,
    }


# =====================================================================
# PowerPoint primitives
# =====================================================================
def ppt_box(slide, m, cx, cy, w, h, text, ec, fc, font_pt=8.2,
            rounded=True, shape_cache=None, key=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(
        shape_type, m["X"](cx - w / 2), m["Y"](cy + h / 2),
        m["WI"](w), m["HI"](h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fc)
    shp.line.color.rgb = rgb(ec)
    shp.line.width = Pt(0.9)
    try:
        shp.adjustments[0] = 0.08
    except (IndexError, ValueError):
        pass
    tf = shp.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size = Pt(font_pt)
        r.font.color.rgb = rgb(INK)
        r.font.name = "Arial"
    if shape_cache is not None and key is not None:
        shape_cache[key] = shp
    return shp


def ppt_line(slide, m, x1, y1, x2, y2, color=GRAY, width_pt=0.9,
             arrow=False, dash=None):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, m["X"](x1), m["Y"](y1),
        m["X"](x2), m["Y"](y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width_pt)
    if dash:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        ln = c._element.spPr.get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "sm")
        tail.set("len", "sm")
        ln.append(tail)
    return c


def ppt_connect(slide, m, src, src_site, dst, dst_site, color=GRAY,
                width_pt=0.9, arrow=True):
    """Connector glued to two shape connection sites (0=top,1=left,2=bottom,3=right)."""
    sb = src.left + src.width // 2, src.top + src.height
    db = dst.left + dst.width // 2, dst.top
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, sb[0], sb[1], db[0], db[1])
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width_pt)
    c.begin_connect(src, src_site)
    c.end_connect(dst, dst_site)
    if arrow:
        ln = c._element.spPr.get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "sm")
        tail.set("len", "sm")
        ln.append(tail)
    return c


def ppt_polygon(slide, m, points, fc, ec=None, width_pt=0.6):
    """Filled polygon from logical coordinates, built as a native freeform."""
    scale = 914400  # EMU per inch
    # convert logical points to inches first, then treat inches as local units
    pts_in = [(x * m["W"] / m["LX"], (m["LY"] - y) * m["H"] / m["LY"])
              for x, y in points]
    fb = slide.shapes.build_freeform(start_x=pts_in[0][0],
                                     start_y=pts_in[0][1], scale=scale)
    fb.add_line_segments(pts_in[1:], close=True)
    shp = fb.convert_to_shape()
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fc)
    if ec:
        shp.line.color.rgb = rgb(ec)
        shp.line.width = Pt(width_pt)
    else:
        shp.line.fill.background()
    return shp


def ppt_text(slide, m, x, y, w, h, text, color=INK, size_pt=7.4,
             align=PP_ALIGN.LEFT, bold=False):
    tb = slide.shapes.add_textbox(m["X"](x - w / 2), m["Y"](y + h / 2),
                                 m["WI"](w), m["HI"](h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.01)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size_pt)
        r.font.bold = bold
        r.font.color.rgb = rgb(color)
        r.font.name = "Arial"
    return tb


def new_ppt(size_in):
    prs = Presentation()
    prs.slide_width = Inches(size_in[0])
    prs.slide_height = Inches(size_in[1])
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(WHITE)
    return prs, slide


# =====================================================================
# Fig. 1 -- PPT
# =====================================================================
def fig1_ppt():
    g = F1
    m = make_map(g["size_in"], g["logic"], g["ylim"])
    prs, slide = new_ppt(g["size_in"])
    cache = {}

    for i, (cy, hh, label, ec, fc) in enumerate(g["boxes"]):
        ppt_box(slide, m, g["cx"], cy, g["box_w"], 2 * hh, label, ec, fc,
                font_pt=g["font_pt"] - (0.2 if "\n" in label else 0),
                shape_cache=cache, key=i)

    # merge nodes (circles) sitting ON the main chain
    for name, cy in (("m1", g["merge1"]), ("m2", g["merge2"])):
        c = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            m["X"](g["cx"] - g["merge_r"]), m["Y"](cy + g["merge_r"]),
            m["WI"](2 * g["merge_r"]), m["HI"](2 * g["merge_r"]))
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(WHITE)
        c.line.color.rgb = rgb(NAVY)
        c.line.width = Pt(0.9)
        tf = c.text_frame
        tf.clear()
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = "+"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = rgb(NAVY)
        cache[name] = c

    # main-chain arrows, split at the two merge nodes so arrows terminate on
    # the circle edge rather than passing underneath it
    B = g["boxes"]
    edges = []
    for i in range(len(B) - 1):
        y_top = B[i + 1][0] + B[i + 1][1]
        y_bot = B[i][0] - B[i][1]
        edges.append((y_bot, y_top))

    arrow_specs = []
    for i, (y0, y1) in enumerate(edges):
        if i == 4:      # TCN -> merge1 -> BiGRU
            arrow_specs.append((y0, g["merge1"] + g["merge_r"], GRAY))
            arrow_specs.append((g["merge1"] - g["merge_r"], y1, GRAY))
        elif i == 5:    # BiGRU -> merge2 -> Transformer
            arrow_specs.append((y0, g["merge2"] + g["merge_r"], GRAY))
            arrow_specs.append((g["merge2"] - g["merge_r"], y1, GRAY))
        else:
            arrow_specs.append((y0, y1, GRAY))
    for y0, y1, col in arrow_specs:
        ppt_line(slide, m, g["cx"], y0, g["cx"], y1, color=col,
                 width_pt=0.9, arrow=True)

    # residual rails: left edge of source -> horizontal -> vertical -> merge
    def rail(src_x, src_y, rail_x, merge_y, col):
        r = g["merge_r"]
        ppt_line(slide, m, src_x, src_y, rail_x, src_y, color=col, width_pt=0.9)
        ppt_line(slide, m, rail_x, src_y, rail_x, merge_y, color=col, width_pt=0.9)
        ppt_line(slide, m, rail_x, merge_y, g["cx"] - r, merge_y,
                 color=col, width_pt=0.9, arrow=True)

    src_left = g["cx"] - g["box_w"] / 2
    rail(src_left, B[3][0], g["rail1"], g["merge1"], TEAL)
    rail(g["cx"] - g["merge_r"], g["merge1"], g["rail2"], g["merge2"], NAVY)

    ppt_text(slide, m, g["rail1"] - 0.72, 10.1, 1.35, 0.8,
             "residual\npathways", color=GRAY, size_pt=7.4,
             align=PP_ALIGN.RIGHT)

    # Condition-adaptive callouts.  Each label sits at exactly the same height
    # as the box it describes, and a horizontal arrow runs from the label's
    # left edge to that box's right-edge midpoint.  end_connect() makes the
    # arrowhead stick to the box when the box is dragged in PowerPoint.
    for idx, text, nx, nw, col in g["notes"]:
        bx = g["cx"] + g["box_w"] / 2
        by = B[idx][0]
        ppt_text(slide, m, nx, by, nw, g["note_h"], text,
                 color=col, size_pt=g["note_pt"], align=PP_ALIGN.LEFT)
        tail_x = nx - nw / 2 - g["note_gap"]
        lead = ppt_line(slide, m, tail_x, by, bx, by,
                        color=col, width_pt=0.9, arrow=True)
        lead.end_connect(cache[idx], 3)   # 3 = right connection site
        # A filled dot on the target edge makes the arrow's destination
        # unambiguous, even if a viewer drags the box in PowerPoint.
        dot_r = 0.055
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, m["X"](bx - dot_r), m["Y"](by + dot_r),
            m["WI"](2 * dot_r), m["HI"](2 * dot_r))
        dot.fill.solid()
        dot.fill.fore_color.rgb = rgb(col)
        dot.line.fill.background()

    prs.save(PPT_OUT / "fig1_architecture.pptx")
    return m


# =====================================================================
# Fig. 1 -- PNG (same geometry)
# =====================================================================
def fig1_png(m):
    g = F1
    fig, ax = plt.subplots(figsize=g["size_in"])
    ax.set_xlim(0, g["logic"][0])
    ax.set_ylim(*g["ylim"])
    ax.axis("off")
    B, CX, W = g["boxes"], g["cx"], g["box_w"]

    for cy, hh, label, ec, fc in B:
        ax.add_patch(mpatches.FancyBboxPatch(
            (CX - W / 2, cy - hh), W, 2 * hh,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(CX, cy, label, ha="center", va="center",
                fontsize=g["font_pt"] - (0.2 if "\n" in label else 0),
                color=INK, zorder=4)

    def arr(x0, y0, x1, y1, col=GRAY, lw=0.9):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                    shrinkA=0, shrinkB=0), zorder=2)

    r = g["merge_r"]
    for name, cy in (("m1", g["merge1"]), ("m2", g["merge2"])):
        ax.add_patch(mpatches.Circle((CX, cy), r, fc=WHITE, ec=NAVY,
                                     lw=1.0, zorder=5))
        ax.text(CX, cy, "+", ha="center", va="center", fontsize=9,
                color=NAVY, weight="bold", zorder=6)

    for i in range(len(B) - 1):
        y0 = B[i][0] - B[i][1]
        y1 = B[i + 1][0] + B[i + 1][1]
        if i == 4:
            arr(CX, y0, CX, g["merge1"] + r)
            arr(CX, g["merge1"] - r, CX, y1)
        elif i == 5:
            arr(CX, y0, CX, g["merge2"] + r)
            arr(CX, g["merge2"] - r, CX, y1)
        else:
            arr(CX, y0, CX, y1)

    def rail(sx, sy, rx, my, col):
        ax.plot([sx, rx], [sy, sy], color=col, lw=0.9, zorder=2)
        ax.plot([rx, rx], [sy, my], color=col, lw=0.9, zorder=2)
        arr(rx, my, CX - r, my, col)

    rail(CX - W / 2, B[3][0], g["rail1"], g["merge1"], TEAL)
    rail(CX - r, g["merge1"], g["rail2"], g["merge2"], NAVY)
    ax.text(g["rail1"] - 0.17, 10.1, "residual\npathways", fontsize=7.4,
            color=GRAY, ha="right", va="center", style="italic")

    for idx, text, nx, nw, col in g["notes"]:
        by = B[idx][0]
        ax.text(nx - nw / 2, by, text, fontsize=g["note_pt"], color=col,
                ha="left", va="center", linespacing=1.3)
        arr(nx - nw / 2 - g["note_gap"], by, CX + W / 2, by, col, 0.9)
        ax.add_patch(mpatches.Circle((CX + W / 2, by), 0.055,
                                     fc=col, ec="none", zorder=8))

    fig.savefig(OUT / "fig1_architecture.png")
    fig.savefig(OUT / "fig1_architecture.pdf")
    plt.close(fig)


# =====================================================================
# Fig. 6 -- PPT
# =====================================================================
def fig6_ppt():
    g = F6
    m = make_map(g["size_in"], g["logic"], g["ylim"], g["xlim"])
    prs, slide = new_ppt(g["size_in"])
    xs = [0, 10]
    b1 = [g["b1"](x) for x in xs]
    b2 = [g["b2"](x) for x in xs]

    # three filled regions, native freeform polygons
    ppt_polygon(slide, m, [(0, 100), (10, 100), (10, b1[1]), (0, b1[0])], TEAL_L)
    ppt_polygon(slide, m, [(0, b1[0]), (10, b1[1]), (10, b2[1]), (0, b2[0])], AMBER_L)
    ppt_polygon(slide, m, [(0, b2[0]), (10, b2[1]), (10, 0), (0, 0)], CORAL_L)
    ppt_line(slide, m, 0, b1[0], 10, b1[1], color=GRAY, width_pt=0.7, dash=True)
    ppt_line(slide, m, 0, b2[0], 10, b2[1], color=GRAY, width_pt=0.7, dash=True)

    # region labels -- centred on the exact same coordinates as the PNG
    for label, col, tx, ty in g["regions"]:
        ppt_text(slide, m, tx, ty, 2.6, 6.0, label,
                 color=col, size_pt=9.0, bold=True, align=PP_ALIGN.CENTER)

    # data points + labels
    for px, py, lbl, lx, ly, ha in g["points"]:
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, m["X"](px - 0.11), m["Y"](py + 1.0),
            m["WI"](0.22), m["HI"](2.0))
        dot.fill.solid()
        dot.fill.fore_color.rgb = rgb(NAVY)
        dot.line.color.rgb = rgb(WHITE)
        dot.line.width = Pt(0.7)
        ppt_text(slide, m, lx, ly, 3.6, 5.0, lbl, color=NAVY, size_pt=7.2,
                 align=PP_ALIGN.LEFT if ha == "left" else PP_ALIGN.RIGHT)

    # explanatory callouts: short leader arrows ending on a data point
    for text, tx, ty, tw, sx, sy, ax_, ay_ in g["callouts"]:
        ppt_text(slide, m, tx, ty, tw, 9.0, text,
                 color=INK, size_pt=7.4, align=PP_ALIGN.CENTER)
        ppt_line(slide, m, sx, sy, ax_, ay_,
                 color=GRAY, width_pt=0.7, arrow=True)

    # axes, then labels placed in the reserved negative margins
    ppt_line(slide, m, 0, 0, 10, 0, color=AXIS, width_pt=0.8)
    ppt_line(slide, m, 0, 0, 0, 100, color=AXIS, width_pt=0.8)
    ppt_text(slide, m, 5.0, -8.0, 9.0, 7.0,
             "Operating-condition heterogeneity   ->",
             color=INK, size_pt=8.4, align=PP_ALIGN.CENTER)
    tb = slide.shapes.add_textbox(m["X"](-1.85), m["Y"](82),
                                  m["WI"](0.9), m["HI"](64))
    tb.rotation = 270
    tf = tb.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Chance of recovery before early stopping"
    r.font.size = Pt(8.4)
    r.font.color.rgb = rgb(INK)

    prs.save(PPT_OUT / "fig6_phase_diagram.pptx")
    return m


# =====================================================================
# Fig. 6 -- PNG (same geometry)
# =====================================================================
def fig6_png(m):
    g = F6
    fig, ax = plt.subplots(figsize=g["size_in"])
    xs = [0, 10]
    b1 = [g["b1"](x) for x in xs]
    b2 = [g["b2"](x) for x in xs]
    ax.fill_between(xs, b1, 100, color=TEAL_L, lw=0, zorder=0)
    ax.fill_between(xs, b2, b1, color=AMBER_L, lw=0, zorder=0)
    ax.fill_between(xs, 0, b2, color=CORAL_L, lw=0, zorder=0)
    ax.plot(xs, b1, color=GRAY, lw=0.8, ls="--", zorder=1)
    ax.plot(xs, b2, color=GRAY, lw=0.8, ls="--", zorder=1)

    for label, col, tx, ty in g["regions"]:
        ax.text(tx, ty, label, fontsize=9.0, color=col, weight="bold",
                ha="center", va="center", zorder=4)

    for px, py, lbl, lx, ly, ha in g["points"]:
        ax.scatter([px], [py], s=52, color=NAVY, zorder=5,
                   edgecolors=WHITE, linewidths=0.8)
        ax.text(lx, ly, lbl, fontsize=7.2, color=NAVY,
                ha=ha, va="center", zorder=6)

    for text, tx, ty, tw, sx, sy, ax_, ay_ in g["callouts"]:
        ax.text(tx, ty, text, fontsize=7.4, color=INK,
                ha="center", va="center", linespacing=1.35, zorder=6)
        ax.annotate("", xy=(ax_, ay_), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.7,
                                    shrinkA=0, shrinkB=3), zorder=5)

    ax.set_xlim(*g["xlim"])
    ax.set_ylim(*g["ylim"])
    ax.text(5.0, -8.0, "Operating-condition heterogeneity   ->",
            fontsize=8.4, color=INK, ha="center", va="center")
    ax.text(-1.30, 50.0, "Chance of recovery before early stopping",
            fontsize=8.4, color=INK, rotation=90, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "fig6_phase_diagram.png")
    fig.savefig(OUT / "fig6_phase_diagram.pdf")
    plt.close(fig)


if __name__ == "__main__":
    m1 = fig1_ppt()
    fig1_png(m1)
    print("fig1: pptx + png")
    m6 = fig6_ppt()
    fig6_png(m6)
    print("fig6: pptx + png")
    print("pptx ->", PPT_OUT)
    print("png  ->", OUT)
