#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manuscript markdown -> docx (BMC submission format).

- Parses headings (#/##/###), paragraphs, tables (| cells), bold runs.
- Embeds figures (paper/figures/figN.png) before their Figure Legends.
- Writes paper/output/doc/manuscript.docx
"""
import re
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJ = Path(r"C:\Users\甄朝晖\Desktop\thyroid")
MD = PROJ / "paper" / "output" / "doc" / "manuscript.md"
OUT = PROJ / "paper" / "output" / "doc" / "manuscript.docx"
FIG_DIR = PROJ / "paper" / "figures"

# figure number -> actual file (fig9=subgroup, fig10=error cases)
FIG_FILES = {
    1: "fig1.png", 2: "fig2.png", 3: "fig3.png", 4: "fig4.png",
    5: "fig5.png", 6: "fig6.png", 7: "fig7.png", 8: "fig8.png",
    9: "fig9_subgroup_forest.png", 10: "fig10_error_cases.png",
}


def add_rich_text(par, text):
    """Add text with **bold** markers rendered as bold runs."""
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            run = par.add_run(p[2:-2])
            run.bold = True
        else:
            par.add_run(p)


def parse_table_line(line):
    """Parse a markdown table row into cells (strip leading/trailing |)."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_separator(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|?\s*$", line)) and "-" in line


def main():
    doc = Document()
    # base style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    lines = MD.read_text(encoding="utf-8").splitlines()
    i = 0
    pending_fig = None  # figure number to embed before next legend

    while i < len(lines):
        line = lines[i].rstrip()

        # skip the version-changelog block (removed) and blank leading
        if not line.strip():
            i += 1
            continue

        # Headings
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
            i += 1
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
            i += 1
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
            i += 1
            continue

        # Figure legend -> embed image first, then legend text
        m = re.match(r"\*\*Figure (\d+)\.\*\*(.*)", line)
        if m:
            fig_no = int(m.group(1))
            fname = FIG_FILES.get(fig_no)
            fpath = FIG_DIR / fname
            if fpath.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                try:
                    run.add_picture(str(fpath), width=Inches(5.5))
                except Exception as e:
                    print(f"  [warn] fig{fig_no} embed failed: {e}")
            else:
                print(f"  [warn] missing {fname}")
            # legend paragraph
            p = doc.add_paragraph()
            run = p.add_run(f"Figure {fig_no}.")
            run.bold = True
            add_rich_text(p, m.group(2))
            # consume wrapped legend lines
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith("**Figure"):
                add_rich_text(doc.add_paragraph(), lines[i].strip())
                i += 1
            continue

        # Table: collect consecutive table lines
        if line.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            header = parse_table_line(line)
            i += 2  # skip header + separator
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_table_line(lines[i]))
                i += 1
            table = doc.add_table(rows=1 + len(rows), cols=len(header))
            table.style = "Light Grid Accent 1"
            for j, h in enumerate(header):
                table.cell(0, j).text = h
            for r_i, row in enumerate(rows):
                for j in range(min(len(row), len(header))):
                    table.cell(r_i + 1, j).text = row[j]
            doc.add_paragraph()
            continue

        # Reference entries "N. text" -> hanging paragraph
        m = re.match(r"^(\d{1,2})\.\s+(.+)$", line)
        if m:
            p = doc.add_paragraph()
            run = p.add_run(m.group(1) + ". ")
            add_rich_text(p, m.group(2))
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            i += 1
            continue

        # Normal paragraph
        p = doc.add_paragraph()
        add_rich_text(p, line)
        i += 1

    doc.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
