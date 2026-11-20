#!/usr/bin/env python3
"""Generate Supplementary_Material.docx for the MSSP submission.

All items moved out of the condensed main manuscript are collected here.
Section / table / figure numbers carry the prefix S and match the pointers
in the main text (Sections S1, S3--S8; Tables S2--S6; Figs. S1--S3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("/Users/hugozha/WorkBuddy/2026-09-27-12-28-14")
V2 = ROOT / "V2"
FIG = V2 / "latex" / "figures"
BIB = V2 / "latex" / "bibliography_gen.tex"
EXP = Path("/Users/hugozha/project/RULPredictor_V2/paper_exp")
OUT = V2 / "Supplementary_Material.docx"

NAVY = "1F3B57"
NAVY_LIGHT = "E8EDF3"
TEAL_LIGHT = "E8F3F1"
GRAY_LIGHT = "F5F7F9"
INK = RGBColor(0x23, 0x2B, 0x33)
MUTED = RGBColor(0x5B, 0x66, 0x73)
ACCENT = RGBColor(0x1F, 0x3B, 0x57)


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def no_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    tr_pr.append(el)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def column_widths(header, rows, total_cm=16.5):
    """Content-aware widths that always sum to the printable page width."""
    scores = []
    for i in range(len(header)):
        vals = [str(header[i])] + [str(r[i]) for r in rows]
        avg = sum(len(v) for v in vals) / len(vals)
        mx = max(len(v) for v in vals)
        scores.append(min(max(0.6 * avg + 0.4 * mx, 5), 42))
    s = sum(scores)
    return [Cm(total_cm * v / s) for v in scores]


def cell_text(cell, text, *, bold=False, size=8.3, color=None, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    if align is not None:
        p.alignment = align
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.font.color.rgb = color or INK
    return p


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    a = OxmlElement("w:fldChar")
    a.set(qn("w:fldCharType"), "begin")
    b = OxmlElement("w:instrText")
    b.set(qn("xml:space"), "preserve")
    b.text = " PAGE "
    c = OxmlElement("w:fldChar")
    c.set(qn("w:fldCharType"), "end")
    run._r.append(a)
    run._r.append(b)
    run._r.append(c)


def heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.color.rgb = ACCENT
    r.font.size = Pt(14 if level == 1 else 11.5)
    r.bold = True
    p.paragraph_format.space_before = Pt(15 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    return p


def body(doc, text, *, size=10.5, italic=False, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.22
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.italic = italic
    r.font.color.rgb = INK
    return p


def bullets(doc, items, size=10.2):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.18
        r = p.add_run(it)
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r.font.color.rgb = INK


def caption(doc, tag, text, *, above=True, size=9.0):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8 if above else 3)
    p.paragraph_format.space_after = Pt(3 if above else 9)
    p.paragraph_format.keep_with_next = above
    r = p.add_run(tag + "  ")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.font.color.rgb = ACCENT
    r2 = p.add_run(text)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(size)
    r2.font.color.rgb = INK
    return p


def add_table(doc, tag, cap, header, rows, *, note=None, size=8.3,
              first_col_left=True):
    caption(doc, tag, cap, above=True)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = column_widths(header, rows)
    for i, w in enumerate(widths):
        t.columns[i].width = w
    hdr = t.rows[0]
    no_split(hdr)
    repeat_header(hdr)
    for i, h in enumerate(header):
        c = hdr.cells[i]
        c.width = widths[i]
        shade(c, NAVY)
        cell_text(c, h, bold=True, size=size, color=RGBColor(0xFF, 0xFF, 0xFF),
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        no_split(t.rows[-1])
        for i, val in enumerate(row):
            cells[i].width = widths[i]
            if ri % 2 == 1:
                shade(cells[i], GRAY_LIGHT)
            al = WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and first_col_left) \
                else WD_ALIGN_PARAGRAPH.CENTER
            cell_text(cells[i], val, size=size, align=al)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(9)
        r = p.add_run("Note. " + note)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.3)
        r.italic = True
        r.font.color.rgb = MUTED
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def add_figure(doc, filename, tag, cap, width_cm=14.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(FIG / filename), width=Cm(width_cm))
    caption(doc, tag, cap, above=False)


# --------------------------------------------------------------------------
# document setup
# --------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)
sec.top_margin = Cm(2.1)
sec.bottom_margin = Cm(2.0)
sec.header_distance = Cm(1.0)
sec.footer_distance = Cm(1.0)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(10.5)
normal.font.color.rgb = INK
normal.paragraph_format.space_after = Pt(6)

hp = sec.header.paragraphs[0]
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = hp.add_run("Supplementary Material  |  Why RUL models sometimes predict the same value for every engine")
hr.font.name = "Times New Roman"
hr.font.size = Pt(8)
hr.font.color.rgb = MUTED

fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = fp.add_run("Page ")
fr.font.name = "Times New Roman"
fr.font.size = Pt(8.5)
fr.font.color.rgb = MUTED
add_page_field(fp)

# --------------------------------------------------------------------------
# title block
# --------------------------------------------------------------------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(16)
r = p.add_run("SUPPLEMENTARY MATERIAL")
r.bold = True
r.font.name = "Times New Roman"
r.font.size = Pt(16)
r.font.color.rgb = ACCENT

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(12)
r = p.add_run("Why RUL models sometimes predict the same value for every engine: an "
              "audit of 161 training runs and condition-based design rules")
r.font.name = "Times New Roman"
r.font.size = Pt(12.5)
r.bold = True
r.font.color.rgb = INK

for line in [
    "Chi Zhang, Yuguo Zha*",
    "Department of Biomedical Engineering, College of Instrument Science and "
    "Optoelectronic Engineering, Nanchang Hangkong University, "
    "Nanchang 330063, Jiangxi, China",
    "* Corresponding author: hugozha@nchu.edu.cn",
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(line)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.font.color.rgb = MUTED

body(doc,
     "This document collects the supporting material that was moved out of "
     "the condensed main manuscript. Section, table and figure numbers carry "
     "the prefix S. The main text is self-contained; every result quoted there "
     "is tabulated here in full, together with the experimental settings, the "
     "seed-level run ledger and the derivations behind the reported formulas.",
     size=10, after=10)

heading(doc, "Contents", 2)
bullets(doc, [
    "S1  Full experimental setup (preprocessing, backbone, hyperparameters, metrics, instrumentation)",
    "S2  Data and configuration grids — Tables S1, S2; Fig. S1",
    "S3  Representation probes — Table S3; Fig. S2",
    "S4  Attention analysis — Table S4; Fig. S3",
    "S5  Accuracy comparison with prior work — Table S5",
    "S6  Component ablation and segment analysis — Table S6",
    "S7  Seed-level run ledger — Tables S7, S8",
    "S8  Derivations (the fixed output, entropy, PHM08, recovery criterion)",
    "S9  Reproducibility file map",
    "S10 Bibliographic details for verification",
], size=9.8)

doc.add_page_break()

# --------------------------------------------------------------------------
# S1
# --------------------------------------------------------------------------
heading(doc, "S1  Full experimental setup")

heading(doc, "S1.1  Preprocessing, labels and windowing", 2)
body(doc,
     "Training labels are the remaining cycles counted backwards from the "
     "recorded failure cycle of each training engine and clipped at r̄ = 125, "
     "which models the flat stretch of incipient degradation. Sensor channels with "
     "zero variance over the training set are removed; the three "
     "operating-setting channels are kept as inputs and reused to recover "
     "condition labels for the probes of Section S3. The remaining channels "
     "are min–max normalised with statistics fitted on the training windows "
     "only. Each engine trajectory is cut into overlapping windows of length "
     "w, which are split 85/15 into training and validation windows. The test "
     "set consists of the last window of each test engine, with the official "
     "C-MAPSS ground-truth RUL. The five-seed audit holds the train/validation "
     "split fixed across seeds; the instrumented re-runs of E2 also varied it.")

heading(doc, "S1.2  Backbone, layer by layer", 2)
add_table(
    doc, "Table S1a.",
    "Hybrid model specification; d = 128 throughout.",
    ["#", "Stage", "Operation", "Output", "Notes"],
    [
        ["1", "Difference augmentation",
         "concat(x_t, Δx_t), Δx_t = x_t − x_{t−1}, Δx_1 = x_1",
         "w × 2C", "Rates as well as levels"],
        ["2", "Input projection", "1×1 conv 2C → d, GELU, dropout",
         "w × d", "d = 128"],
        ["3", "Position encoding",
         "sinusoidal positional embeddings; sensor-channel dropout p = 0.1",
         "w × d", "Training only"],
        ["4", "TCN front-end",
         "5 causal dilated conv blocks, k = 3, dilations 1/2/4/8/16; "
         "layer norm, GELU, dropout, residual",
         "w × d", "Receptive field 31 cycles"],
        ["5", "BiGRU stage",
         "2-layer bidirectional GRU, hidden 128; linear projection; "
         "residual add to TCN output",
         "w × d", "Residual join (Fig. 1)"],
        ["6", "Transformer encoder",
         "L pre-norm layers, 4-head causal self-attention, d_ff = 4d, GELU",
         "w × d", "L ∈ {1, 2, 3, 4}"],
        ["7", "Prediction head",
         "concat[avg-pool(h); max-pool(h)] → 2-layer MLP → scalar",
         "1", "Penultimate h ∈ R^256"],
    ],
    note="Window size w and encoder depth L are the only condition-adaptive "
         "choices in CAS-Fuse. All other hyperparameters are shared across "
         "subsets.",
    size=8.0,
)

heading(doc, "S1.3  Training protocol and hardware", 2)
add_table(
    doc, "Table S1b.",
    "Training protocol shared by all archived and audited runs.",
    ["Item", "Value"],
    [
        ["Objective", "Mean squared error on capped RUL labels"],
        ["Optimiser", "AdamW, learning rate 1×10⁻⁴, weight decay 1×10⁻⁴"],
        ["Batch size", "32"],
        ["Schedule", "Cosine annealing, at most 150 epochs"],
        ["Early stopping", "Validation RMSE, patience 20"],
        ["Gradient clipping", "Global norm 1.0 (pre-clipping norms are logged in E2)"],
        ["Regularisation", "Dropout 0.1 on sensor channels, TCN and attention"],
        ["Parameters", "1.08 M (L = 1), 1.47 M (L = 3)"],
        ["Seed scope", "Initialisation and data order; validation split fixed "
                       "in E1, varied in E2"],
        ["Seeds (E1)", "42 (archived reference), 7, 123, 2026, 3407; "
                       "FD004 adds four archived runs"],
        ["Framework / device", "PyTorch 2.8, Apple M-series GPU (MPS)"],
        ["Runtime", "60–130 min (FD001/2/3), 70–350 min (FD004); "
                    "failed runs 30–70 min"],
        ["Total cost", "The 20-run audit took about two days on one consumer "
                       "device; instrumented runs add 50–100% overhead"],
    ],
    size=8.6,
)

heading(doc, "S1.4  Metrics", 2)
bullets(doc, [
    "RMSE = sqrt(mean_i (ŷ_i − y_i)²), the primary metric; R² is reported on "
    "the same capped labels.",
    "PHM08 asymmetric score: with d_i = ŷ_i − y_i, s = Σ_i "
    "[exp(−d_i/13) − 1] for d_i < 0 and [exp(d_i/10) − 1] for d_i ≥ 0. "
    "Late predictions are penalised more heavily; smaller is better.",
    "Failure rate: fraction of seeds flagged by Checks A–C of main-text "
    "Section 5.3. A run is counted as failed iff test RMSE > 40 in the "
    "archived prevalence study, a conservative threshold given Eq. (1).",
    "Prediction std σ_ŷ: standard deviation of the test-set predictions; "
    "below 10⁻³ cycles for every failed checkpoint.",
], size=10.0)

heading(doc, "S1.5  Instrumentation (E1 and E2)", 2)
body(doc,
     "E1 is the five-seed stability audit. Each run records test RMSE, MAE, "
     "PHM08 score, R², prediction std, the failure flag and the per-epoch "
     "validation curve. E2 re-trains six multi-condition configurations with "
     "per-epoch logging of: pre-clipping gradient L2 norms for the TCN, GRU, "
     "attention and head parameter groups; mean normalised attention entropy; "
     "condition- and RUL-probe accuracy on a fixed batch; validation RMSE; "
     "and prediction std. Probes are fitted on the penultimate representation "
     "h ∈ R^256: a 5-fold cross-validated linear classifier for the six "
     "operating conditions (accuracy) and a 5-fold cross-validated ridge "
     "regressor for RUL (R²). Condition labels come from k-means (k = 6) on "
     "the three operating-setting channels of the final window step, with "
     "cluster inertia below 10⁻³. An recovery event is the first epoch at which "
     "prediction std exceeds 0.1 and validation RMSE falls below "
     "std(y_val) − ε.")

# --------------------------------------------------------------------------
# S2
# --------------------------------------------------------------------------
heading(doc, "S2  Data and configuration grids")
body(doc,
     "The audit covers 161 archived runs. Four runs have no recorded subset "
     "label, so the per-subset breakdown in main-text Table 3 accounts for "
     "157 of them (45 + 43 + 36 + 33); all 161 are included in the overall "
     "failure count and in the formula verification.")

add_table(
    doc, "Table S1.",
    "C-MAPSS subsets, preprocessing output and CAS-Fuse configuration per subset.",
    ["Subset", "Cond.", "Faults", "Train eng.", "Test eng.", "Kept C",
     "Train win.", "CAS-Fuse (w, L)"],
    [
        ["FD001", "1", "1", "100", "96", "17", "14 221", "(40, 1)"],
        ["FD002", "6", "1", "260", "259", "24", "41 496", "(20, 3)"],
        ["FD003", "1", "2", "100", "100", "18", "18 547", "(30, 1)"],
        ["FD004", "6", "2", "249", "222", "24", "39 574", "(60, 3)"],
    ],
    note="Cond. = operating conditions; Kept C = sensor channels remaining "
         "after the variance filter; Train win. = training windows at the "
         "listed w. FD001 and FD003 have one condition, so their setting "
         "channels are constant and removed.",
    size=8.6,
)

add_table(
    doc, "Table S2.",
    "Controlled configuration search, single archived run per cell. "
    "(a) Encoder depth on the pure hybrid model backbone (test RMSE); "
    "(b) window size on the adversarial (GRL) backbone, FD002; "
    "(c) attention-head count, FD002.",
    ["Grid / setting", "1 / 20 / 2", "2 / 30 / 4", "3 / 40 / 8", "4 / 60 / —"],
    [
        ["(a) L — FD002 (w = 20)", "18.67", "43.85 (C)", "18.35", "43.85 (C)"],
        ["(a) L — FD004 (w = 60)", "43.71 (C)", "20.33", "19.46", "23.27"],
        ["(b) w — FD002, adversarial", "working", "failed", "working",
         "failed"],
        ["(c) h — FD002, adversarial", "failed", "working", "working", "—"],
    ],
    note="(C) = failed, verified against Eq. (1) of the main text. "
         "Column labels give the setting for each of the three grids "
         "respectively (L / w / h). The three GRL placements relocate the "
         "failure region but do not remove it. Failure cells reproduced "
         "across archived runs; seeded re-runs show high-probability "
         "seed dependence rather than deterministic failure (Table S8).",
    size=8.2,
)

add_figure(doc, "fig3_prevalence.png", "Fig. S1.",
           "Failure prevalence grids. (a) Encoder-depth grid on the pure "
           "backbone, cells showing failed/total runs pooled over the "
           "archive and seeded re-runs. (b) Window-size grid on the "
           "adversarial-backbone runs, cells showing test RMSE. Failure "
           "concentrates in configuration islands rather than growing "
           "monotonically with depth or window length.")

# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------
heading(doc, "S3  Representation probes")
body(doc,
     "Nine checkpoints from the controlled depth search (five failed, "
     "four working) plus two single-condition references are probed. Only "
     "the depth L or the seed changes within a subset, so differences in "
     "decodability track the outcome rather than the architecture.")

add_table(
    doc, "Table S3.",
    "Probing the penultimate representation (5-fold cross-validation). "
    "Cond. acc. is linear-probe accuracy for operating condition; "
    "RUL R² is ridge-probe R².",
    ["Model", "Subset", "L", "Status", "Test RMSE", "Cond. acc.", "RUL R²"],
    [
        ["FD002-L1", "FD002", "1", "working", "18.67", "0.32", "0.78"],
        ["FD002-L2", "FD002", "2", "failed", "43.85", "0.82", "0.58"],
        ["FD002-L3", "FD002", "3", "working", "18.35", "0.30", "0.81"],
        ["FD002-L4", "FD002", "4", "failed", "43.85", "0.78", "0.55"],
        ["FD004-L1", "FD004", "1", "failed", "43.71", "0.76", "0.09"],
        ["FD004-L2", "FD004", "2", "working", "20.33", "0.25", "0.68"],
        ["FD004-L3", "FD004", "3", "working", "19.46", "0.25", "0.76"],
        ["FD004-L4", "FD004", "4", "working", "23.27", "0.23", "0.66"],
        ["FD004-L3 (s2026)", "FD004", "3", "failed", "43.71", "0.62", "0.04"],
        ["FD001-L1 (ref.)", "FD001", "1", "working", "13.53", "n/a", "0.87"],
        ["FD003-L1 (ref.)", "FD003", "1", "working", "16.23", "n/a", "0.81"],
    ],
    note="Condition probes are undefined for the single-condition subsets. "
         "Per-condition label means on FD002 (86.7–87.1) and FD004 "
         "(92.7–93.3) barely differ from the global means, which is why a "
         "condition-organised representation has nothing better to output "
         "than the label mean.",
    size=8.3,
)

add_figure(doc, "fig4_tsne.png", "Fig. S2.",
           "t-SNE of the penultimate representation h for a working (top) "
           "and a failed (bottom) FD002 checkpoint, coloured by recovered "
           "operating condition (left) and RUL bucket (right). The failed "
           "embedding separates into condition bands while its RUL colouring "
           "is noise.")

body(doc,
     "The FD002–FD004 asymmetry is informative. On FD004 (w = 60) failure "
     "removes the RUL signal from h almost entirely (R² 0.04–0.09); on FD002 "
     "(w = 20) a linear RUL component remains (R² 0.55–0.58) although the "
     "prediction is still 82.57, because the attention stack has flattened "
     "the sequence and the head can no longer use it. Failure is therefore "
     "a failure of usable representation, not only of linear information "
     "content.")

# --------------------------------------------------------------------------
# S4
# --------------------------------------------------------------------------
heading(doc, "S4  Attention analysis")
body(doc,
     "For each layer, head, query position and test sample we compute the "
     "normalised entropy H = −Σ_j p_j log₂ p_j / log₂(t + 1) of the causal "
     "attention distribution, then average over heads, queries and samples. "
     "H = 1.0 means exactly uniform attention. Entropies are averaged "
     "per-head and per-sample rather than computed from an averaged attention "
     "matrix: by Jensen's inequality the latter is larger and cannot "
     "reproduce the observed pinning at 1.0000 (Section S8.3).")

add_table(
    doc, "Table S4.",
    "Mean normalised attention entropy (1.0 = uniform) and mean maximum "
    "attention weight, per checkpoint.",
    ["Model", "Status", "Layer-wise entropy", "Mean max. weight"],
    [
        ["FD002-L1", "working", "0.790", "0.38"],
        ["FD002-L2", "failed", "1.0000, 1.0000", "0.18"],
        ["FD002-L3", "working", "0.722 / 0.786 / 0.835", "0.44 / 0.38 / 0.35"],
        ["FD002-L4", "failed", "1.0000 ×4", "0.18"],
        ["FD004-L1", "failed", "0.9998", "0.08"],
        ["FD004-L2", "working", "0.775 / 0.884", "0.25 / 0.17"],
        ["FD004-L3", "working", "0.851 / 0.852 / 0.893", "0.20 / 0.21 / 0.15"],
        ["FD004-L3 (s2026)", "failed", "1.0000 ×3", "0.08"],
        ["FD001-L1", "working", "0.891", "0.21"],
        ["FD003-L1", "working", "0.931", "0.22"],
    ],
    note="Layer-wise values are listed in depth order. Failed checkpoints "
         "sit at 1.0000 to four decimals with the maximum weight at its "
         "uniform-theoretic value; working ones lie between 0.70 and 0.93. "
         "The single-condition references are the most uniform working "
         "models, which shows that uniform attention alone is not "
         "pathological when a residual temporal branch remains informative.",
    size=8.3,
)

add_figure(doc, "fig8_attention_maps.png", "Fig. S3.",
           "Layer-1 attention matrices (heads and test samples averaged) for "
           "the working (left) and failed (right) FD002 checkpoints of "
           "Table S3. Mean normalised row entropy: 0.7221 (working, "
           "matching Table S4) against 1.0000 (failed, exactly uniform).")

# --------------------------------------------------------------------------
# S5
# --------------------------------------------------------------------------
heading(doc, "S5  Accuracy comparison with prior work")
add_table(
    doc, "Table S5.",
    "Test RMSE on C-MAPSS. Literature rows are single-run values as reported "
    "by the cited papers; their failure behaviour is unknown or unreported. "
    "CAS-Fuse rows give mean ± std over working seeds with the failure "
    "rate in parentheses.",
    ["Method", "FD001", "FD002", "FD003", "FD004"],
    [
        ["DCNN", "12.61", "22.36", "12.64", "23.31"],
        ["LSTM", "12.10", "16.90", "12.14", "21.50"],
        ["Vanilla Transformer", "13.32", "19.83", "13.92", "21.88"],
        ["BiGRU-TSAM", "12.56", "18.94", "12.45", "20.47"],
        ["CNN-BiLSTM", "13.12", "13.92", "12.16", "20.24"],
        ["CNN-LSTM-SAM", "12.56", "15.25", "13.77", "18.63"],
        ["ATCN", "11.48", "15.82", "11.34", "17.80"],
        ["MS-DCNN", "11.44", "19.35", "11.67", "22.22"],
        ["VAE-Transformer", "12.05", "14.71", "12.11", "16.95"],
        ["DVGTformer", "11.33", "14.28", "11.89", "15.50"],
        ["TATFA-Transformer", "12.21", "15.07", "11.23", "18.81"],
        ["DKAMFormer", "10.68", "10.70", "10.52", "12.89"],
        ["CAS-Fuse, 5 seeds (ours)", "13.66 ± 0.10 (0/5)",
         "19.43 ± 0.44 (0/5)", "16.09 ± 0.52 (0/5)", "19.53 ± 0.07 (6/8)"],
        ["CAS-Fuse, best working", "13.53", "18.81", "15.46", "19.46"],
    ],
    note="FD004 additionally pools four archived runs of the same "
         "configuration (eight seeds in total). Bibliographic details for "
         "every literature row are listed in Section S10. Mean PHM08 scores "
         "over working seeds are 307 (FD001), 2 855 (FD002), 1 050 (FD003) "
         "and 1 783–2 448 (FD004); failed runs score 61 569–75 860.",
    size=8.1,
)

body(doc,
     "CAS-Fuse beats a vanilla Transformer on every subset and falls within "
     "the DCNN / BiGRU-TSAM / MS-DCNN band, but stays below specialised "
     "architectures such as DKAMFormer, which targets FD002/FD004 with "
     "dedicated multi-scale machinery. The claim of the main text is not "
     "per-subset state-of-the-art accuracy; it is verified multi-seed "
     "stability at competitive accuracy. None of the literature rows above "
     "reports a failure rate.")

# --------------------------------------------------------------------------
# S6
# --------------------------------------------------------------------------
heading(doc, "S6  Component ablation and segment analysis")
add_table(
    doc, "Table S6.",
    "Component ablation (test RMSE, single archived runs) with PHM08 scores "
    "recomputed from the saved checkpoints. (C) = failed, verified by the "
    "main-text Eq. (1) formula.",
    ["ID", "Removed", "FD001", "FD002", "FD003", "FD004", "Mean", "Score mean"],
    [
        ["A0", "— (full backbone)", "13.53", "19.73", "16.23", "19.60",
         "17.27", "3 425"],
        ["A1", "TCN front-end", "17.33", "43.84 (C)", "18.31", "24.21",
         "25.92", "21 060"],
        ["A2", "BiGRU", "13.84", "19.86", "15.20", "19.91", "17.20", "1 961"],
        ["A3", "Transformer encoder", "13.53", "19.04", "15.56",
         "43.68 (C)", "22.95", "16 149"],
        ["A4", "Difference features", "13.12", "19.00", "16.14",
         "43.73 (C)", "23.00", "16 466"],
        ["A5", "Max pooling", "12.85", "17.94", "16.02", "18.90",
         "16.43", "1 259"],
        ["A6", "Positional encoding", "18.21", "20.87", "17.41", "24.31",
         "20.20", "3 076"],
        ["A7", "Sensor dropout", "14.74", "19.88", "16.32", "20.28",
         "17.80", "2 351"],
    ],
    note="Relative changes quoted in the main text are computed against the "
         "A0 row, which is the ablation study's own reference run. Removing "
         "the TCN front-end makes FD002 fail; removing the Transformer "
         "encoder or the difference features makes FD004 fail — the branch that "
         "is load-bearing depends on window length and condition count. "
         "An accuracy-only reading would instead rank A5 as the best "
         "ablation.",
    size=8.0,
)

body(doc,
     "Segment errors of failed runs reproduce the constant-prediction "
     "signature on every subset: on FD002, RMSE is 69.9 / 38.5 / 10.7 / 38.2 "
     "across RUL buckets <30, 30–60, 60–100 and ≥100 (Section S8.2). "
     "Val–test gaps of working runs range from −2.3% to +25.3%, largest "
     "on FD003; this is a known generalisation limitation of windowed "
     "train/validation splits drawn from the same trajectories, retained here "
     "for comparability with the cited literature. The PHM08 column was "
     "computed by re-scoring the saved checkpoints with the definition in "
     "Section S1.4 (script: paper_exp/compute_ablation_phm08.py).")

# --------------------------------------------------------------------------
# S7
# --------------------------------------------------------------------------
heading(doc, "S7  Seed-level run ledger")
e1 = json.loads((EXP / "e1_aggregate.json").read_text())
e2 = json.loads((EXP / "e2_escape_analysis.json").read_text())

seed_cols = ["seed_7", "seed_123", "seed_2026", "seed_3407", "42(arch)"]
s7_rows = []
for name in ["FD001", "FD002", "FD003", "FD004"]:
    d = e1[name]
    per = d["per_seed"]
    extras = [f"{per[k]:.2f}" for k in per if k.startswith("arch")]
    row = [name] + [f"{per[k]:.2f}" for k in seed_cols] + [
        ", ".join(extras) if extras else "—",
        f"{d['n_collapsed']}/{d['n_total']}",
        f"{d['surv_rmse_mean']:.2f} ± {d['surv_rmse_std']:.2f}",
    ]
    s7_rows.append(row)

add_table(
    doc, "Table S7.",
    "E1 five-seed stability audit, per-seed test RMSE. FD004 pools four "
    "additional archived runs of the same configuration.",
    ["Subset", "s7", "s123", "s2026", "s3407", "s42 (arch)", "Extra archived",
     "Failed", "Working mean ± std"],
    s7_rows,
    note="Source: paper_exp/e1_aggregate.json. A seed controls "
         "initialisation and data order; the train/validation split is held "
         "fixed in E1. Failed FD004 values (43.67–43.72) match Eq. (1) "
         "to two decimals; the two working runs (19.46, 19.60) have std 0.07 "
         "and no intermediate outcome exists.",
    size=8.0,
)

run_name = {
    "FD002_L2": "FD002 L=2, s42",
    "FD002_L3": "FD002 L=3, s42",
    "FD002_L2_s7": "FD002 L=2, s7",
    "FD004_L1": "FD004 L=1, s42",
    "FD004_L3": "FD004 L=3, s42",
    "FD004_L3_s123": "FD004 L=3, s123",
}
s8_rows = []
for key, d in e2.items():
    esc = d.get("escape_ep")
    s8_rows.append([
        run_name.get(key, key),
        f"{d['final']:.2f}",
        "failed" if d["collapsed"] else "recovered",
        str(d["plateau_len"]),
        str(esc) if esc is not None else "never",
        f"×{d['grad_spike']:.0f}" if d.get("grad_spike") else "—",
        f"{d['ent_at_escape']:.4f}" if d.get("ent_at_escape") else "1.0000",
        f"{d['cond_before']:.2f}" if d.get("cond_before") is not None
        else f"{d.get('last_cond', 0):.2f}",
        f"{d['cond_after']:.2f}" if d.get("cond_after") is not None else "—",
    ])

add_table(
    doc, "Table S8.",
    "E2 instrumented re-runs: six multi-condition runs with per-epoch "
    "gradient, entropy and probe logging.",
    ["Run", "Final RMSE", "Outcome", "Flat (ep.)", "Recovery", "TCN jump",
     "H at recovery", "Cond. before", "Cond. after"],
    s8_rows,
    note="Source: paper_exp/e2_escape_analysis.json. Flat and Recovery are "
         "epoch numbers; the TCN jump is the pre-clipping gradient-norm "
         "ratio at recovery. H at recovery is the first entropy value below 1.0. "
         "For the failed run the final recorded entropy (1.0000), "
         "condition-probe accuracy (0.72) and encoder gradient (1.61) are "
         "shown in the last columns. E2 also varied the train/validation "
         "split seed; FD004 L=3 with seed 123 failed under the fixed "
         "split in E1 but recovered here.",
    size=8.0,
)

# --------------------------------------------------------------------------
# S8
# --------------------------------------------------------------------------
heading(doc, "S8  Derivations")

heading(doc, "S8.1  Constant predictor and closed-form test RMSE (Eq. (1))", 2)
body(doc,
     "If the network emits a constant c independent of the input, minimising "
     "squared error over the validation windows gives c* = ȳ_val, and the "
     "resulting validation RMSE is exactly std(y_val). This is why the "
     "failed validation curve pins at 41.97 on FD002 and 42.24 on FD004 "
     "within 10–15 epochs. The test labels have a different mean ȳ_test "
     "because test engines are on average more degraded, so")
p = body(doc,
         "MSE_test = E[(c* − y_test)²] = Var(y_test) + (ȳ_val − ȳ_test)²,   "
         "RMSE_failed = sqrt[ Var(y_test) + (ȳ_val − ȳ_test)² ]",
         size=10.5, after=8)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
body(doc,
     "With FD002 ȳ_val = 82.58, ȳ_test = 73.7 and std(y_val) = 41.97, this "
     "gives 43.85, matching every failed run to within 0.01 cycles. The "
     "roughly 1.9-cycle excess over the validation value is the squared bias "
     "term, which is why the failure reads as modest underfitting rather than "
     "as a distinct degenerate solution. The same calculation on FD004 gives "
     "43.71.")

heading(doc, "S8.2  Segment errors of a constant predictor", 2)
body(doc,
     "For a constant c, the RMSE within a bucket of true RUL values is "
     "sqrt[ Var(y | bucket) + (c − ȳ_bucket)² ], so buckets far from the "
     "label mean carry large errors and buckets near it carry small ones. "
     "On FD002 the observed values 69.9 (RUL < 30), 38.5 (30–60), 10.7 "
     "(60–100) and 38.2 (≥ 100) follow this U-shape around c = 82.57, which "
     "is a signature that cannot be produced by an input-dependent model.")

heading(doc, "S8.3  Attention entropy and the Jensen caveat", 2)
body(doc,
     "For a query at position t the causal distribution p_t has at most t+1 "
     "non-zero entries, so H(p_t) = −Σ_j p_tj log₂ p_tj is normalised by "
     "log₂(t+1). We report the mean of H over heads, queries and test "
     "samples. Computing the entropy of an averaged attention matrix instead "
     "would give H(E[p]) ≥ E[H(p)] by Jensen's inequality: averaging heads "
     "first can inflate the value even for non-uniform attention, and cannot "
     "reproduce the observed pinning at exactly 1.0000 in failed "
     "checkpoints. The per-head/per-sample convention is used in main-text "
     "Table 4, in Table S4 and in Fig. S3 (0.7221 versus 1.0000).")

heading(doc, "S8.4  PHM08 score and the 15–250× inflation", 2)
body(doc,
     "With d_i = ŷ_i − y_i, the score sums exp(−d_i/13) − 1 for d_i < 0 and "
     "exp(d_i/10) − 1 for d_i ≥ 0. A failed run predicts 82.6 for every "
     "engine, so late-prediction errors on low-RUL engines enter the steep "
     "exponential branch. Working seeds score 307–2 855 across subsets "
     "while failed runs score 61 569–75 860, i.e. 15–250 times worse than "
     "the RMSE ratio would suggest. Score, not RMSE, is the deployment "
     "metric.")

heading(doc, "S8.5  The patience race", 2)
body(doc,
     "Early stopping uses patience 20 on validation RMSE from the start of the "
     "flat stretch "
     "near epoch 5–10, while training that recovers does so in epochs 21–39. Working "
     "runs reach their recovery epoch only because micro-improvements in "
     "flat stretch validation RMSE (41.975 → 41.971 → …) intermittently reset the "
     "patience counter. Archived failed runs early-stopped at epochs "
     "31–38, inside that window, so at least some are plausibly recoveries "
     "truncated by the stopping rule. This is the basis of Check C: the "
     "flat stretch is visible from epoch ≈5, tens of epochs before early stopping "
     "fires, and an at-risk run can be killed and reseeded in minutes.")

# --------------------------------------------------------------------------
# S9
# --------------------------------------------------------------------------
heading(doc, "S9  Reproducibility file map")
add_table(
    doc, "Table S9.",
    "Paths are relative to the project root RULPredictor_V2/; the LaTeX "
    "sources and all figures are also mirrored under paper_submission/.",
    ["Item", "Path"],
    [
        ["Experiment driver", "paper_exp/run_paper_experiments.sh"],
        ["E1 five-seed audit", "paper_exp/e1_seeds.py"],
        ["E2 instrumented training", "paper_exp/e2_probe_train.py"],
        ["Supervised daemon (long runs)", "paper_exp/daemon_start.py"],
        ["Ablation PHM08 recomputation", "paper_exp/compute_ablation_phm08.py"],
        ["E1 aggregate results", "paper_exp/e1_aggregate.json"],
        ["E2 recovery analysis", "paper_exp/e2_escape_analysis.json"],
        ["161-run failure archive", "paper_exp/collapse_summary.json"],
        ["Checkpoint probe results", "paper_exp/probe_results.json"],
        ["Ablation scores", "paper_exp/ablation_phm08.json"],
        ["Run logs", "paper_exp/logs/"],
        ["Figure generation scripts", "paper_submission/figure_scripts/"],
        ["Editable Fig. 1 / Fig. 6 sources", "paper_submission/figures_ppt/"],
        ["Manuscript sources and PDF", "paper_submission/latex/"],
    ],
    note="All eight figures share a single palette defined in "
         "figure_scripts/fig_palette.py; the two schematic figures also have "
         "editable .pptx sources with connector-bound arrows.",
    size=8.4,
)

# --------------------------------------------------------------------------
# S10
# --------------------------------------------------------------------------
heading(doc, "S10  Bibliographic details for verification")
body(doc,
     "The following list reproduces the bibliography compiled for the "
     "manuscript, in citation-number order. Entries marked “supplementary "
     "only” support rows of Table S5 or background discussion that moved out "
     "of the condensed main text; before submission they should either be "
     "cited in the main text or removed. Author and venue details are "
     "reproduced as compiled and are pending the authors' final check.")

bib_text = BIB.read_text()
entries = re.findall(r"\\bibitem\{([^}]*)\}\s*(.*)", bib_text)
main_src = "".join((V2 / "latex" / f).read_text()
                   for f in ["main.tex", "sec1_intro.tex", "sec2_related.tex",
                             "sec3_phenomenon.tex", "sec4_mechanism.tex",
                             "sec5_method.tex", "sec6_experiments.tex",
                             "sec7_discussion.tex"])
cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", main_src)
         for k in grp.split(",")}
rows = []
for i, (key, text) in enumerate(entries, 1):
    clean = text.replace("``", '"').replace("''", '"')
    clean = re.sub(r"\\&", "&", clean)
    clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\['a-zA-Z]*\{?([a-zA-Z])\}?", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(".")
    if len(clean) > 165:
        clean = clean[:162] + "..."
    rows.append([f"[{i}]", key, clean,
                 "main text" if key in cited else "supplementary only"])

add_table(
    doc, "Table S10.",
    "Compiled bibliography in citation-number order, with citation status "
    "after the main-text condensation.",
    ["No.", "Citation key", "Entry (as compiled)", "Status"],
    rows,
    note="“Supplementary only” entries are not cited in the condensed main "
         "text; they are retained here so that the authors can verify, "
         "restore or delete them before submission.",
    size=7.8,
)

doc.save(OUT)
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
print(f"tables parsed from bibliography: {len(rows)}")
