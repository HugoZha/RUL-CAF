#!/usr/bin/env python3
"""Build a Word manuscript from the LaTeX sources plus the tectonic .aux file.

The .aux supplies authoritative cross-reference numbers (sections, figures,
tables, equations) and bibliography numbering, so the Word copy matches the
official cas-sc PDF exactly.
"""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = Path("/Users/hugozha/WorkBuddy/2026-09-27-12-28-14/submission_MSSP/01_manuscript")
OUT = BASE / "manuscript.docx"
FIGDIR = BASE / "figures"

# ---------------------------------------------------------------- mappings
aux = (BASE / "main_cas.aux").read_text(encoding="utf-8")
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{[^}]*\}\{([^}]*)\}", aux):
    key, num, kind = m.group(1), m.group(2), m.group(3)
    labels[key] = (num, kind.split(".")[0])
bibcites = {}
for m in re.finditer(r"\\bibcite\{([^}]*)\}\{([0-9]+)\}", aux):
    bibcites[m.group(1)] = m.group(2)

def refnum(key):
    if key in labels:
        return labels[key][0]
    if key in bibcites:
        return bibcites[key]
    return "??"

# ---------------------------------------------------------------- LaTeX->text
MATH_SIMPLE = {
    r"\%": "%", r"\&": "&", r"\_": "_", r"\$": "$",
    r"\times": "×", r"\dots": "…", r"\ldots": "…",
    r"\;": " ", r"\,": " ", r"\!": "", r"\ ": " ",
    r"\geq": "≥", r"\leq": "≤", r"\neq": "≠",
    r"\pm": "±", r"\approx": "≈", r"\to": "→",
    r"\infty": "∞", r"\alpha": "α", r"\beta": "β",
    r"\gamma": "γ", r"\delta": "δ", r"\sigma": "σ",
    r"\mu": "μ", r"\pi": "π", r"\theta": "θ",
    r"\mathrm{val}": "val", r"\mathrm{std}": "std",
}

def clean_math(s):
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", s)
    s = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\bar\{([^}]*)\}", r"\1-bar", s)
    s = re.sub(r"\\hat\{([^}]*)\}", r"\1-hat", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = re.sub(r"\\[a-zA-Z]+", lambda m: MATH_SIMPLE.get(m.group(0), m.group(0)), s)
    s = s.replace("{", "").replace("}", "")
    return s

def inline(text):
    """Convert inline LaTeX to plain decorated text (format via markers **/§)."""
    # protect escaped braces and specials
    text = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + ",".join(refnum(k.strip()) for k in m.group(1).split(",")) + "]", text)
    text = re.sub(r"\\citea\{([^}]*)\}", lambda m: "[" + ",".join(refnum(k.strip()) for k in m.group(1).split(",")) + "]", text)
    text = re.sub(r"\\ref\{([^}]*)\}", lambda m: refnum(m.group(1)), text)
    text = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + refnum(m.group(1)) + ")", text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"§\1§", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"§\1§", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    # inline math $...$
    def mathrep(m):
        return clean_math(m.group(1))
    text = re.sub(r"\$([^$]+)\$", mathrep, text)
    # leftover commands
    for k, v in MATH_SIMPLE.items():
        text = text.replace(k, v)
    text = re.sub(r"\\(label|cormark|cortext)\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("---", "—").replace("--", "–")
    text = text.replace("~", " ").replace("\\sep", "; ")
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    # LaTeX accents in names, e.g. Altch\'e -> Altché
    text = re.sub(r"\\'([a-zA-Z])", lambda m: {"e":"é","a":"á","o":"ó","i":"í","u":"ú","c":"ç","n":"ñ"}.get(m.group(1).lower(), m.group(1)), text)
    text = re.sub(r'\\[`^"]([a-zA-Z])', lambda m: m.group(1), text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def add_runs(par, text):
    """Add text with **bold** and §italic§ markers."""
    parts = re.split(r"(\*\*.*?\*\*|§.*?§)", text)
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            par.add_run(p[2:-2]).bold = True
        elif p.startswith("§") and p.endswith("§"):
            par.add_run(p[1:-1]).italic = True
        else:
            par.add_run(p)

# ---------------------------------------------------------------- doc setup
doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.27); sec.page_height = Inches(11.69)  # A4
sec.left_margin = Inches(1.0); sec.right_margin = Inches(1.0)
sec.top_margin = Inches(0.9); sec.bottom_margin = Inches(0.9)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
normal.paragraph_format.space_after = Pt(6)

for name, size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"; st.font.size = Pt(size); st.font.bold = True
    st.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)

def para(text="", style=None, align=None, space_after=None):
    p = doc.add_paragraph(style=style)
    if text:
        add_runs(p, text)
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p

# ---------------------------------------------------------------- title block
para("Why RUL models sometimes predict the same value for every engine: an audit of 161 training runs and condition-based design rules",
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10).runs[0].bold = True
para("Chi Zhang, Yuguo Zha*", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("Department of Biomedical Engineering, College of Instrument Science and Optoelectronic Engineering, Nanchang Hangkong University, Nanchang 330063, Jiangxi, China",
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2).runs[0].italic = True
para("* Corresponding author. E-mail: hugozha@nchu.edu.cn", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12).runs[0].font.size = Pt(9)

def heading(text, level):
    p = para(text, style=f"Heading {level}")
    return p

# abstract & keywords
p = para(); r = p.add_run("Abstract — "); r.bold = True
fm = (BASE / "frontmatter_cas.tex").read_text(encoding="utf-8")
absm = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", fm, re.S)
abstext = inline(absm.group(1)) if absm else ""
add_runs(p, " " + abstext)
p = para(space_after=12); r = p.add_run("Keywords: "); r.bold = True
kwm = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", fm, re.S)
kwtext = inline(kwm.group(1)) if kwm else ""
add_runs(p, " " + kwtext.replace(r"\sep", "; "))

# ---------------------------------------------------------------- body parse
sections = ["sec1_intro.tex", "sec2_related.tex", "sec3_phenomenon.tex",
            "sec4_mechanism.tex", "sec5_method.tex", "sec6_experiments.tex",
            "sec7_discussion.tex", "declarations.tex"]

def parse_table(env):
    """env text -> list of rows."""
    rows = []
    for line in env.splitlines():
        line = line.strip()
        if not line or line.startswith("\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
            continue
        if line.startswith("\\") or line.startswith("{"):
            continue
        cells = [inline(c) for c in line.split("&")]
        cells = [c.rstrip("\\").strip() for c in cells]
        if cells and any(cells):
            rows.append(cells)
    return rows

def flush_para(buf):
    txt = " ".join(x.strip() for x in buf if x.strip())
    if txt:
        para(inline(txt))
    buf.clear()

for fname in sections:
    src = (BASE / fname).read_text(encoding="utf-8")
    # insert a marker line before section/subsection commands
    src = re.sub(r"\\(section|subsection)(\*)?\{([^}]*)\}",
                 lambda m: f"\n@@SEC{m.group(1)}{m.group(2) or ''}:{m.group(3)}@@\n", src)
    buf = []
    i = 0
    lines = src.splitlines()
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        # section markers
        sm = re.match(r"@@SEC(section|subsection)(\*?):(.*)@@", s)
        if sm:
            flush_para(buf)
            lvl = 1 if sm.group(1) == "section" else 2
            title = sm.group(3).strip()
            # look ahead for the \label on the next non-empty line
            num = None
            k = i + 1
            while k < len(lines) and k <= i + 3:
                lm = re.match(r"\s*\\label\{([^}]*)\}", lines[k])
                if lm:
                    num = labels.get(lm.group(1), (None, None))[0]
                    break
                if lines[k].strip():
                    break
                k += 1
            if sm.group(2) == "*":
                p = para(title); p.runs[0].bold = True; p.runs[0].font.size = Pt(13)
            else:
                heading(f"{num}  {title}" if num else title, lvl)
            i += 1
            continue
        if s.startswith("\\label{"):
            i += 1; continue
        pm = re.match(r"\\paragraph\{([^}]*)\}", s)
        if pm:
            flush_para(buf)
            hp = para(pm.group(1)); hp.runs[0].bold = True
            i += 1; continue
        # figure / table environments
        m = re.match(r"\\begin\{(figure\*?|table\*?)\}", s)
        if m:
            flush_para(buf)
            envtype = m.group(1)
            endtag = "\\end{" + envtype + "}"
            j = i + 1
            body = []
            while j < len(lines) and lines[j].strip() != endtag:
                body.append(lines[j]); j += 1
            env = "\n".join(body)
            capm = re.search(r"\\caption\{(.*?)\}\s*(?:\\label\{[^}]*\})?", env, re.S)
            caption = inline(capm.group(1)) if capm else ""
            labelm = re.search(r"\\label\{([^}]*)\}", env)
            num = refnum(labelm.group(1)) if labelm else ""
            kind = "Figure" if envtype.startswith("figure") else "Table"
            capline = f"{kind} {num}. {caption}"
            if envtype.startswith("figure"):
                imgm = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", env)
                if imgm:
                    img = FIGDIR / Path(imgm.group(1)).name
                    if img.exists():
                        picp = doc.add_paragraph(); picp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        picp.add_run().add_picture(str(img), width=Inches(5.6))
                cp = para(capline, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
                for r in cp.runs: r.font.size = Pt(9); r.italic = True
            else:
                cp = para(capline, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=4)
                for r in cp.runs: r.font.size = Pt(9); r.bold = True
                trowsm = re.search(r"\\begin\{tabularx\}.*?\n(.*?)\\end\{tabularx\}", env, re.S)
                if trowsm:
                    rows = parse_table(trowsm.group(1))
                    if rows:
                        ncols = max(len(r) for r in rows)
                        tbl = doc.add_table(rows=0, cols=ncols)
                        tbl.style = "Table Grid"
                        for ri, row in enumerate(rows):
                            cells = tbl.add_row().cells
                            for ci in range(ncols):
                                val = row[ci] if ci < len(row) else ""
                                cells[ci].text = ""
                                pp = cells[ci].paragraphs[0]
                                add_runs(pp, val)
                                for r in pp.runs: r.font.size = Pt(8)
                                if ri == 0:
                                    for r in pp.runs: r.bold = True
                para("", space_after=8)
            i = j + 1
            continue
        # equation
        if s.startswith("\\begin{equation}"):
            flush_para(buf)
            endtag = "\\end{equation}"
            j = i + 1; body = []
            while j < len(lines) and lines[j].strip() != endtag:
                body.append(lines[j]); j += 1
            eq = " ".join(x.strip() for x in body if x.strip())
            eq = re.sub(r"\\label\{[^}]*\}", "", eq)
            labm = re.search(r"\\label\{([^}]*)\}", "\n".join(body))
            eqnum = refnum(labm.group(1)) if labm else ""
            ep = para(clean_math(eq), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
            for r in ep.runs: r.italic = True
            if eqnum:
                ep.add_run(f"    ({eqnum})")
            i = j + 1
            continue
        # enumerate
        if s.startswith("\\begin{enumerate}"):
            flush_para(buf)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("\\end{enumerate}"):
                it = re.match(r"\\item\s*(.*)", lines[j].strip())
                if it:
                    para(inline(it.group(1)), style="List Number")
                j += 1
            i = j + 1
            continue
        if s.startswith("\\end{") or s in ("", "%"):
            flush_para(buf); i += 1; continue
        if s.startswith("%") or s.startswith("\\centering") or s.startswith("\\footnotesize") or s.startswith("\\setlength"):
            i += 1; continue
        buf.append(line)
        i += 1
    flush_para(buf)

# ---------------------------------------------------------------- references
p = para("References", space_after=8)
p.runs[0].bold = True; p.runs[0].font.size = Pt(13)
bib = (BASE / "bibliography_gen.tex").read_text(encoding="utf-8")
for m in re.finditer(r"\\bibitem\{([^}]*)\}\s*(.*?)(?=\\bibitem|\Z)", bib, re.S):
    key, body = m.group(1), m.group(2).strip()
    num = bibcites.get(key, "?")
    txt = inline(body).replace("``", "“").replace("''", "”")
    pp = doc.add_paragraph()
    pp.paragraph_format.left_indent = Inches(0.3)
    pp.paragraph_format.first_line_indent = Inches(-0.3)
    pp.add_run(f"[{num}] ").bold = True
    add_runs(pp, txt)
    for r in pp.runs: r.font.size = Pt(9)

doc.save(OUT)
print("Wrote", OUT)
print("labels:", len(labels), "bibcites:", len(bibcites))
