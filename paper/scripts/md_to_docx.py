#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manuscript markdown -> docx (BMC submission format).

Fixed layout:
- Markdown paragraphs (blocks separated by blank lines) are merged into ONE
  Word paragraph (source soft line-wraps do not create separate paragraphs).
- Headings use built-in styles with consistent fonts.
- Tables get explicit column widths and a clean style.
- Figures are centered at a fixed width before their legends.
- References render as hanging-indent paragraphs.

Writes paper/output/doc/manuscript.docx (or given output).
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

PROJ = Path(r"C:\Users\甄朝晖\Desktop\thyroid")
MD = PROJ / "paper" / "output" / "doc" / "manuscript.md"
FIG_DIR = PROJ / "paper" / "figures"

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
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_separator(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|?\s*$", line)) and "-" in line


def set_cell_text(cell, text, bold=False):
    """Set cell text, clearing default paragraph first."""
    cell.text = ""
    p = cell.paragraphs[0]
    # strip markdown bold markers; render **...** as bold runs
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for seg in parts:
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**"):
            run = p.add_run(seg[2:-2])
            run.bold = True
        else:
            run = p.add_run(seg)
            run.bold = bold
    # strip footnote markers ^a^ ^b^ etc (footnote text is below the table)
    for r in p.runs:
        r.text = re.sub(r"\^\w\^", "", r.text)
    for r in p.runs:
        r.font.size = Pt(9)
        r.font.name = "Times New Roman"


def split_blocks(lines):
    """Split source lines into markdown blocks separated by blank lines.
    Returns list of (kind, payload):
      kind in {'heading','para','table','figure','ref','abbrev'}
    """
    blocks = []
    cur = []
    for ln in lines:
        if ln.strip():
            cur.append(ln)
        else:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
    if cur:
        blocks.append("\n".join(cur))

    out = []
    for blk in blocks:
        b = blk.strip()
        if b.startswith("## "):
            out.append(("h1", b[3:].strip()))
        elif b.startswith("### "):
            out.append(("h2", b[4:].strip()))
        elif b.startswith("# "):
            out.append(("title", b[2:].strip()))
        elif "|" in b and len(b.splitlines()) > 1 and is_separator(b.splitlines()[1]):
            out.append(("table", b))
        else:
            out.append(("para", b))
    return out


def main():
    md_path = MD
    out_path = PROJ / "paper" / "output" / "doc" / "manuscript.docx"
    if len(sys.argv) > 2:
        md_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
    elif len(sys.argv) > 1:
        out_path = Path(sys.argv[1])

    doc = Document()
    # Chinese manuscript -> use SimSun for CJK + Times New Roman for Latin
    is_zh = "zh" in out_path.stem.lower() or "中文" in out_path.name
    cn_font = "SimSun" if is_zh else "Times New Roman"

    # Normal style
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    # set East Asian font (SimSun) so Chinese renders properly
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", cn_font)
    # Heading styles
    for lvl, sz in [(1, 14), (2, 12)]:
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(sz)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr2 = st.element.get_or_add_rPr()
        rpr2.get_or_add_rFonts().set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", cn_font)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    blocks = split_blocks(lines)

    # Special handling: figure legends embed images. Since a figure legend is a
    # markdown block starting with '**Figure N.**', we detect it in the para pass.
    for kind, payload in blocks:
        if kind == "title":
            doc.add_heading(payload, level=0)
        elif kind == "h1":
            doc.add_heading(payload, level=1)
        elif kind == "h2":
            doc.add_heading(payload, level=2)
        elif kind == "table":
            blines = payload.splitlines()
            # find separator row
            sep_idx = None
            for j, bl in enumerate(blines):
                if is_separator(bl):
                    sep_idx = j
                    break
            if sep_idx is None:
                p = doc.add_paragraph()
                add_rich_text(p, payload)
                continue
            header = parse_table_line(blines[0])
            body = [parse_table_line(bl) for bl in blines[sep_idx + 1:] if bl.strip().startswith("|")]
            ncols = max(len(header), *(len(r) for r in body)) if body else len(header)
            table = doc.add_table(rows=1 + len(body), cols=ncols)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            # header
            for j in range(ncols):
                set_cell_text(table.cell(0, j), header[j] if j < len(header) else "", bold=True)
            for ri, row in enumerate(body):
                for j in range(ncols):
                    set_cell_text(table.cell(ri + 1, j), row[j] if j < len(row) else "")
            # footnote lines that follow the table (start with ^)
            # (handled as separate para blocks since they start with ^)
            doc.add_paragraph()
        else:  # para (or figure legend)
            m = re.match(r"\*\*Figure (\d+)\.\*\*(.*)", payload)
            if m:
                fig_no = int(m.group(1))
                fname = FIG_FILES.get(fig_no)
                fpath = FIG_DIR / fname
                if fpath.exists():
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = p.add_run()
                    try:
                        run.add_picture(str(fpath), width=Inches(5.0))
                    except Exception as e:
                        print(f"  [warn] fig{fig_no}: {e}")
                p = doc.add_paragraph()
                run = p.add_run(f"Figure {fig_no}.")
                run.bold = True
                add_rich_text(p, m.group(2))
            elif re.match(r"^\d{1,2}\.\s", payload):  # reference
                first_line, rest = payload.split("\n", 1) if "\n" in payload else (payload, "")
                mm = re.match(r"^(\d{1,2})\.\s+(.+)$", first_line)
                if mm:
                    p = doc.add_paragraph()
                    run = p.add_run(mm.group(1) + ". ")
                    add_rich_text(p, mm.group(2))
                    if rest:
                        add_rich_text(p, " " + " ".join(rest.split()))
                    p.paragraph_format.left_indent = Inches(0.35)
                    p.paragraph_format.first_line_indent = Inches(-0.35)
                else:
                    p = doc.add_paragraph()
                    add_rich_text(p, payload)
            elif payload.lstrip().startswith(">"):
                # blockquote (e.g. Chinese version note) -> italic, indented
                text = " ".join(payload.split())
                text = re.sub(r"^>\s*", "", text)
                p = doc.add_paragraph()
                run = p.add_run(text)
                run.italic = True
                run.font.size = Pt(9)
                p.paragraph_format.left_indent = Inches(0.35)
                p.paragraph_format.space_after = Pt(6)
            else:
                # normal paragraph: join all source lines with space
                text = " ".join(payload.split())
                p = doc.add_paragraph()
                add_rich_text(p, text)
                p.paragraph_format.space_after = Pt(6)

    doc.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
