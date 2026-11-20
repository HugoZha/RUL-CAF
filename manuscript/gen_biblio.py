#!/usr/bin/env python3
"""gen_biblio.py — refs.bib -> LaTeX thebibliography (numeric, elsarticle-num style)."""
import re
from pathlib import Path

SRC = Path(__file__).parent / "refs.bib"
OUT = Path(__file__).parent / "bibliography_gen.tex"

text = SRC.read_text()
entries = []
for m in re.finditer(r"@(\w+)\s*\{([^,]+),\s*(.*?)\n\}", text, re.S):
    etype, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
    fields = {}
    for fm in re.finditer(r"(\w+)\s*=\s*[{\"]?(.*?)[}\"]?\s*\n\s*\w+\s*=|\w+\s*=\s*[{\"].*?[}\"],?\s*$",
                          body, re.M):
        pass
    # simpler field parse: split top-level commas... use regex per field line
    for fm in re.finditer(r"(\w+)\s*=\s*[{](.*?)[}]\s*(?:,|$)", body, re.M):
        fields[fm.group(1).lower()] = fm.group(2).strip()
    entries.append((key, etype, fields))

def fmt_authors(raw):
    a = [x.strip() for x in raw.replace("\n", " ").split(" and ")]
    out = []
    for name in a:
        if "," in name:
            ln, fn = [x.strip() for x in name.split(",", 1)]
            out.append(f"{ln}, {fn[0]}.")
        else:
            parts = name.split()
            if len(parts) >= 2:
                out.append(f"{parts[-1]}, {parts[0][0]}.")
            else:
                out.append(name)
    s = ", ".join(out[:-1]) + " and " + out[-1] if len(out) > 1 else out[0]
    return s

def clean(t):
    return re.sub(r"[{}]", "", t)

items = []
for key, etype, f in entries:
    authors = fmt_authors(clean(f.get("author", "Anonymous")))
    title = clean(f.get("title", "")).rstrip(".")
    journal = clean(f.get("journal", ""))
    booktitle = clean(f.get("booktitle", ""))
    vol = clean(f.get("volume", ""))
    pages = clean(f.get("pages", "")).replace("--", "--")
    year = clean(f.get("year", ""))
    doi = clean(f.get("doi", ""))
    note = clean(f.get("note", ""))
    todo = " [TODO-VERIFY]" if "todo-verify" in note.lower() else ""
    seg = f"{authors}. ``{title}.'' "
    if etype == "article":
        seg += f"{journal} {vol}" + (f"({clean(f.get('number',''))})" if f.get("number") else "")
        seg += f", {pages}" if pages else ""
        seg += f", {year}."
    elif etype == "inproceedings":
        seg += f"In {booktitle}, {pages + ', ' if pages else ''}{year}."
    else:
        seg += f"{journal}, {year}."
    if doi:
        seg += f" DOI: {doi}."
    seg += todo
    items.append((key, seg))

# MSSP: "Number references in the order they appear in your article", and
# the submitted list must contain no uncited entries.  Derive the order from
# the manuscript rather than from refs.bib, so numbering stays correct after
# any edit to the prose.
import glob
HERE = Path(__file__).parent
body = (HERE / "main.tex").read_text()
body += "".join(Path(f).read_text()
                for f in sorted(glob.glob(str(HERE / "sec*.tex"))))
first_seen = []
for c in re.findall(r"\\cite\{([^}]*)\}", body):
    for k in c.split(","):
        k = k.strip()
        if k and k not in first_seen:
            first_seen.append(k)

by_key = dict(items)
ordered = [k for k in first_seen if k in by_key]
uncited = [k for k in by_key if k not in first_seen]
missing = [k for k in first_seen if k not in by_key]

out = ["\\begin{thebibliography}{99}", ""]
for key in ordered:
    out.append(f"\\bibitem{{{key}}} {by_key[key]}")
out.append("\\end{thebibliography}")
OUT.write_text("\n".join(out))

# Keep the unused entries on disk so no bibliographic data is lost; they are
# simply not part of the submitted reference list.
(HERE / "uncited_refs.tex").write_text(
    "% Entries in refs.bib that the manuscript does not cite.\n"
    "% Kept for reference; MSSP requires no uncited items.\n"
    + "\n".join(f"% \\bibitem{{{k}}} {by_key[k]}" for k in uncited) + "\n")

print(f"{OUT}: {len(ordered)} bibitems in order of first citation")
print(f"uncited ({len(uncited)}) kept in uncited_refs.tex")
if missing:
    print(f"WARNING cited but missing from refs.bib: {missing}")
