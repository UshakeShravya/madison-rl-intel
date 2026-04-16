#!/usr/bin/env python3
"""
Build docs/technical_report.pdf from technical_report.md + architecture_diagram.md.

Usage:
    python docs/build_pdf.py
"""

from __future__ import annotations
import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, Preformatted,
)

# ── Page geometry ─────────────────────────────────────────────────────────────
PW, PH     = LETTER
MARGIN     = 0.85 * inch
CONTENT_W  = PW - 2 * MARGIN

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE  = HexColor('#1a2b4a')
MID_BLUE   = HexColor('#2d4a7a')
ACCENT     = HexColor('#3d6fa8')
LIGHT_BLUE = HexColor('#e8eef6')
CODE_BG    = HexColor('#f3f4f6')
CODE_BDR   = HexColor('#d0d7de')
TH_BG      = HexColor('#2d4a7a')
TR_ALT     = HexColor('#eef2f7')
LIGHT_GRY  = HexColor('#777777')
MID_GRY    = HexColor('#444444')

# ── Unicode → safe-ASCII substitution table ───────────────────────────────────
# ReportLab's built-in Helvetica / Courier are Latin-1; we map all non-Latin-1
# characters to readable ASCII equivalents before rendering.
_SUBS = str.maketrans({
    # Dashes / quotes
    '\u2014': '--',   '\u2013': '-',
    '\u2018': "'",    '\u2019': "'",
    '\u201c': '"',    '\u201d': '"',
    # Arithmetic / relational
    '\u00d7': 'x',    '\u2212': '-',
    '\u2265': '>=',   '\u2264': '<=',
    '\u2248': '~=',   '\u00b1': '+/-',
    '\u221e': 'inf',  '\u221a': 'sqrt',
    '\u2211': 'Sum',  '\u2208': 'in',
    '\u22c5': '*',    '\u00b7': '.',
    # Greek letters
    '\u03b1': 'alpha',  '\u03b2': 'beta',   '\u03b3': 'gamma',
    '\u03b4': 'delta',  '\u03b5': 'eps',    '\u03b8': 'theta',
    '\u03bb': 'lambda', '\u03bc': 'mu',     '\u03c0': 'pi',
    '\u03c3': 'sigma',  '\u03c4': 'tau',    '\u03a3': 'Sigma',
    # Superscripts
    '\u207b': '^-',  '\u00b9': '^1', '\u00b2': '^2', '\u00b3': '^3',
    '\u2074': '^4',  '\u2075': '^5', '\u2076': '^6',
    '\u2077': '^7',  '\u2078': '^8', '\u2079': '^9', '\u2070': '^0',
    # Arrows
    '\u2192': '->',  '\u2190': '<-',  '\u21d2': '=>',
    '\u25b8': '>',   '\u25c2': '<',
    # Math letters
    '\u211d': 'R',
    '\u00c2': 'A^',   # Â (A-hat in math)
    '\u00ca': 'E^',   # Ê (E-hat / expectation)
    # Box-drawing (used in ASCII diagrams inside code fences)
    '\u2554': '+',  '\u2557': '+',  '\u255a': '+',  '\u255d': '+',
    '\u2560': '+',  '\u2563': '+',  '\u2566': '+',  '\u2569': '+',  '\u256c': '+',
    '\u2551': '|',  '\u2550': '=',
    '\u250c': '+',  '\u2510': '+',  '\u2514': '+',  '\u2518': '+',
    '\u251c': '+',  '\u2524': '+',  '\u252c': '+',  '\u2534': '+',  '\u253c': '+',
    '\u2500': '-',  '\u2502': '|',
    # Pointer / triangle symbols
    '\u25bc': 'v',  '\u25b2': '^',
    '\u25c4': '<',  '\u25ba': '>',
    # Circled numbers
    '\u2460': '(1)', '\u2461': '(2)', '\u2462': '(3)', '\u2463': '(4)',
    '\u2464': '(5)', '\u2465': '(6)', '\u2466': '(7)', '\u2467': '(8)',
    '\u2468': '(9)',
    # Bullets / misc
    '\u2022': '*',
})


def clean(s: str) -> str:
    """Apply Unicode → ASCII substitutions."""
    return s.translate(_SUBS)


def xe(s: str) -> str:
    """Escape XML special characters for ReportLab Paragraph markup."""
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def md_inline(text: str) -> str:
    """Convert markdown inline formatting to ReportLab XML."""
    text = clean(text)
    text = xe(text)
    # Bold-italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', text)
    # Inline code  →  grey-background monospace
    text = re.sub(
        r'`([^`\n]+?)`',
        lambda m: (
            f'<font name="Courier" size="8" color="#24292e">'
            f'{xe(clean(m.group(1)))}</font>'
        ),
        text,
    )
    return text


# ── Paragraph styles ──────────────────────────────────────────────────────────
def _ps(name, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


S = {
    'h1': _ps('h1', fontName='Helvetica-Bold', fontSize=16,
              textColor=DARK_BLUE, spaceBefore=22, spaceAfter=8, leading=20),
    'h2': _ps('h2', fontName='Helvetica-Bold', fontSize=13,
              textColor=MID_BLUE, spaceBefore=15, spaceAfter=5, leading=17),
    'h3': _ps('h3', fontName='Helvetica-BoldOblique', fontSize=11,
              textColor=ACCENT, spaceBefore=11, spaceAfter=4, leading=15),
    'h4': _ps('h4', fontName='Helvetica-Bold', fontSize=10,
              textColor=MID_GRY, spaceBefore=8, spaceAfter=3, leading=14),
    'body': _ps('body', fontName='Helvetica', fontSize=10,
                spaceBefore=3, spaceAfter=3, leading=15, alignment=TA_JUSTIFY),
    'bullet': _ps('bullet', fontName='Helvetica', fontSize=10,
                  spaceBefore=2, spaceAfter=2, leading=14,
                  leftIndent=18, firstLineIndent=0),
    'num': _ps('num', fontName='Helvetica', fontSize=10,
               spaceBefore=2, spaceAfter=2, leading=14,
               leftIndent=22, firstLineIndent=-12),
    'sub_bullet': _ps('sub_bullet', fontName='Helvetica', fontSize=9.5,
                      spaceBefore=1, spaceAfter=1, leading=13,
                      leftIndent=36, firstLineIndent=0, textColor=MID_GRY),
    'th': _ps('th', fontName='Helvetica-Bold', fontSize=9,
              textColor=white, alignment=TA_CENTER, leading=12),
    'td': _ps('td', fontName='Helvetica', fontSize=9,
              alignment=TA_LEFT, leading=12),
    'td_ctr': _ps('td_ctr', fontName='Helvetica', fontSize=9,
                  alignment=TA_CENTER, leading=12),
    'footer': _ps('footer', fontName='Helvetica', fontSize=8,
                  textColor=LIGHT_GRY, alignment=TA_CENTER),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_code_block(lines: list[str], font_size: float = 7.0) -> list:
    """Render a fenced code block with grey background and border.

    Returns a list of Table flowables — long blocks are split into
    page-sized chunks so ReportLab never hits LayoutError.
    """
    # Usable frame height = page height minus top/bottom margins
    _frame_h = PH - MARGIN - (MARGIN + 0.15 * inch)
    _padding  = 20  # top + bottom cell padding in points
    _leading  = font_size * 1.38
    # Leave a 10-line safety margin so the last chunk fits comfortably
    _max_lines = max(30, int((_frame_h - _padding) / _leading) - 10)

    results: list = []
    for start in range(0, max(1, len(lines)), _max_lines):
        chunk = lines[start:start + _max_lines]
        text  = '\n'.join(clean(line) for line in chunk)
        pre_style = ParagraphStyle(
            'pre', fontName='Courier', fontSize=font_size,
            leading=_leading, textColor=HexColor('#24292e'),
        )
        pre = Preformatted(text, pre_style)
        tbl = Table([[pre]], colWidths=[CONTENT_W])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), CODE_BG),
            ('BOX',           (0, 0), (-1, -1), 0.6, CODE_BDR),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ]))
        results.append(tbl)
    return results


def make_table(rows: list[list[str]]) -> Table | None:
    """Build a styled Table. First row is the header."""
    if not rows:
        return None
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return None

    # Heuristic column widths: equal split
    col_w = CONTENT_W / n_cols

    data = []
    for i, row in enumerate(rows):
        # Pad short rows
        padded = list(row) + [''] * (n_cols - len(row))
        if i == 0:
            cells = [Paragraph(md_inline(c.strip()), S['th']) for c in padded]
        else:
            cells = [Paragraph(md_inline(c.strip()), S['td']) for c in padded]
        data.append(cells)

    tbl = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), TH_BG),
        ('TEXTCOLOR',     (0, 0), (-1,  0), white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 9),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, TR_ALT]),
        ('GRID',          (0, 0), (-1, -1), 0.5, HexColor('#c0c8d8')),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    return tbl


def section_rule() -> HRFlowable:
    return HRFlowable(width='100%', thickness=0.5,
                      color=HexColor('#c8d1dc'),
                      spaceAfter=4, spaceBefore=4)


# ── Markdown parser ───────────────────────────────────────────────────────────

def parse_md(md: str) -> list:
    """Convert markdown string to a list of ReportLab flowables."""
    flowables: list = []
    lines = md.split('\n')
    i = 0

    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()

        # ── Horizontal rule ──────────────────────────────────────────────────
        if re.match(r'^-{3,}$', line) or re.match(r'^\*{3,}$', line):
            flowables.append(section_rule())
            i += 1
            continue

        # ── Fenced code block ────────────────────────────────────────────────
        if line.startswith('```'):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            if code_lines:
                max_len = max((len(l) for l in code_lines), default=0)
                # Choose font size so longest line fits inside content width
                # Courier at pt: approx 0.6 * font_size pts per char
                fs = 7.5
                if max_len > 88:
                    fs = 6.5
                if max_len > 106:
                    fs = 5.5
                # make_code_block returns a list (may be split across pages)
                flowables.extend(make_code_block(code_lines, fs))
            continue

        # ── Markdown table ───────────────────────────────────────────────────
        if line.startswith('|') and line.endswith('|'):
            table_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row_line = lines[i].strip()
                cells = [c for c in row_line.split('|')[1:-1]]
                # Skip separator rows (|---|---|)
                if not all(re.match(r'^[-:| ]+$', c) for c in cells):
                    table_rows.append(cells)
                i += 1
            if table_rows:
                tbl = make_table(table_rows)
                if tbl:
                    flowables.append(Spacer(1, 4))
                    flowables.append(tbl)
                    flowables.append(Spacer(1, 6))
            continue

        # ── ATX Headers ──────────────────────────────────────────────────────
        m = re.match(r'^(#{1,5})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            text  = md_inline(m.group(2))
            style_map = {2: 'h1', 3: 'h2', 4: 'h3', 5: 'h4'}
            skey = style_map.get(level, 'h4')

            if level == 1:
                # Document title — already on title page, render as h1 here
                flowables.append(Paragraph(text, S['h1']))
            elif level == 2:
                # Major section — add decorative rule below
                flowables.append(Spacer(1, 6))
                flowables.append(Paragraph(text, S['h1']))
                flowables.append(HRFlowable(
                    width='100%', thickness=1.5, color=ACCENT,
                    spaceAfter=6, spaceBefore=2))
            else:
                flowables.append(Paragraph(text, S[skey]))
            i += 1
            continue

        # ── Numbered list item ────────────────────────────────────────────────
        m = re.match(r'^(\d+)\.\s+(.+)$', line)
        if m:
            text = md_inline(m.group(2))
            flowables.append(
                Paragraph(f'{m.group(1)}.&nbsp;&nbsp;{text}', S['num'])
            )
            i += 1
            continue

        # ── Bullet list item ─────────────────────────────────────────────────
        m = re.match(r'^[-*]\s+(.+)$', line)
        if m:
            text = md_inline(m.group(1))
            # Detect sub-bullets (indented 2+ spaces in the raw line)
            indent = len(raw) - len(raw.lstrip())
            style = S['sub_bullet'] if indent >= 4 else S['bullet']
            bullet_char = '&#x2022;' if indent < 4 else '&#x25e6;'
            flowables.append(Paragraph(f'{bullet_char}&nbsp;{text}', style))
            i += 1
            continue

        # ── Blank line ────────────────────────────────────────────────────────
        if not line:
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # ── Body paragraph ────────────────────────────────────────────────────
        # Collect wrapped continuation lines (stop at structural elements)
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt
                    or nxt.startswith('#')
                    or nxt.startswith('```')
                    or nxt.startswith('|')
                    or re.match(r'^[-*]\s+', nxt)
                    or re.match(r'^\d+\.\s+', nxt)
                    or re.match(r'^-{3,}$', nxt)):
                break
            para_lines.append(nxt)
            i += 1

        combined = ' '.join(para_lines)
        rendered = md_inline(combined)
        if rendered.strip():
            flowables.append(Paragraph(rendered, S['body']))

    return flowables


# ── Page footer ───────────────────────────────────────────────────────────────

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(LIGHT_GRY)
    txt = (
        f'Madison RL Intelligence Agent \u2014 Technical Report'
        f'\u2003|\u2003Page {doc.page}'
    )
    # Replace the em-dash and whitespace since we're drawing directly
    txt = txt.replace('\u2014', '--').replace('\u2003', '   ')
    canvas.drawCentredString(PW / 2, 0.45 * inch, txt)
    # Top header rule (pages > 1)
    if doc.page > 1:
        canvas.setStrokeColor(HexColor('#dde3ec'))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PH - MARGIN + 8, PW - MARGIN, PH - MARGIN + 8)
    canvas.restoreState()


# ── Title page builder ────────────────────────────────────────────────────────

def build_title_page() -> list:
    story = []
    story.append(Spacer(1, 0.9 * inch))

    # ── Main title banner ────────────────────────────────────────────────────
    banner_data = [[
        Paragraph(
            '<font size="22" color="#ffffff"><b>Madison RL Intelligence Agent</b></font>',
            _ps('banner', fontName='Helvetica-Bold', fontSize=22,
                textColor=white, alignment=TA_CENTER, leading=28),
        )
    ]]
    banner = Table(banner_data, colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), DARK_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
        ('LEFTPADDING',   (0, 0), (-1, -1), 20),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 20),
    ]))
    story.append(banner)

    # ── Subtitle strip ───────────────────────────────────────────────────────
    sub_data = [[
        Paragraph(
            'Technical Report',
            _ps('sub', fontName='Helvetica-Bold', fontSize=15,
                textColor=MID_BLUE, alignment=TA_CENTER, leading=20),
        )
    ]]
    sub_tbl = Table(sub_data, colWidths=[CONTENT_W])
    sub_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 0.22 * inch))

    # ── Description lines ────────────────────────────────────────────────────
    desc_style = _ps('desc', fontName='Helvetica', fontSize=10,
                     textColor=MID_GRY, alignment=TA_CENTER, leading=16)
    story.append(Paragraph(
        'Hierarchical Reinforcement Learning for Multi-Agent Research Orchestration',
        desc_style,
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        'PPO Meta-Controller  |  LinUCB Contextual Bandits  |  '
        'Transfer Learning  |  Novelty Bonus  |  Real Agent Execution',
        _ps('desc2', fontName='Helvetica-Oblique', fontSize=9,
            textColor=LIGHT_GRY, alignment=TA_CENTER, leading=14),
    ))
    story.append(Spacer(1, 0.22 * inch))
    story.append(HRFlowable(width='55%', thickness=1.2, color=ACCENT,
                             hAlign='CENTER', spaceAfter=0))
    story.append(Spacer(1, 0.22 * inch))

    # ── Key results table ────────────────────────────────────────────────────
    kr_rows = [
        ['Metric', 'Result'],
        ['Full System vs Random Baseline', '+291.3%  (27.48 vs 7.02 mean reward)'],
        ['Full System vs Heuristic', '+156.0%'],
        ['Few-Shot Transfer (K=1)', '+12.3% over from-scratch training'],
        ['Early Adaptation (ep 1-10)', '+17.6% advantage vs from-scratch'],
        ['Full System wins tight-budget env', '13.26 vs 12.78 (PPO Only)'],
        ['Experiments', '500 episodes x 3 seeds x 5 configurations'],
        ['Code & Data', 'github.com/UshakeShravya/madison-rl-intel'],
    ]
    kr_tbl = make_table(kr_rows)
    if kr_tbl:
        story.append(kr_tbl)

    story.append(PageBreak())
    return story


# ── Main build function ───────────────────────────────────────────────────────

def build_pdf(output: str = 'docs/technical_report.pdf') -> None:
    repo = Path(__file__).parent.parent  # project root

    report_md  = (repo / 'docs' / 'technical_report.md').read_text(encoding='utf-8')
    diagram_md = (repo / 'docs' / 'architecture_diagram.md').read_text(encoding='utf-8')

    # ── Merge documents ───────────────────────────────────────────────────────
    # Strip the H1 title from architecture_diagram.md (it's already a section)
    diag_lines = diagram_md.split('\n')
    if diag_lines and diag_lines[0].startswith('# '):
        diag_lines = diag_lines[1:]
    diag_body = '\n'.join(diag_lines).lstrip('\n')

    # Insert the architecture diagram as a new section right after
    # "## System Architecture" block (before "## Mathematical Formulation").
    insert_after  = '## Mathematical Formulation'
    arch_section  = (
        '\n\n---\n\n'
        '## Architecture Diagrams\n\n'
        + diag_body
        + '\n\n---\n\n'
    )

    if insert_after in report_md:
        combined_md = report_md.replace(
            insert_after,
            arch_section + insert_after,
            1,
        )
    else:
        combined_md = report_md + arch_section

    # ── Strip the document H1 (rendered on the title page instead) ───────────
    lines = combined_md.split('\n')
    body_start = 0
    for idx, l in enumerate(lines):
        if l.startswith('## '):
            body_start = idx
            break
    body_md = '\n'.join(lines[body_start:])

    # ── Build ReportLab story ─────────────────────────────────────────────────
    story: list = []
    story.extend(build_title_page())
    story.extend(parse_md(body_md))

    # ── Compile PDF ───────────────────────────────────────────────────────────
    out_path = str(repo / output) if not Path(output).is_absolute() else output
    doc = SimpleDocTemplate(
        out_path,
        pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN + 0.15 * inch,
        title='Madison RL Intelligence Agent: Technical Report',
        author='UshakeShravya',
        subject='Hierarchical RL for Multi-Agent Research Orchestration',
        creator='reportlab + docs/build_pdf.py',
    )
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f'PDF written -> {out_path}')
    print(f'Pages: check output')


if __name__ == '__main__':
    build_pdf()
