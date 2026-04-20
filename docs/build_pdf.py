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
from reportlab.graphics.shapes import (
    Drawing, Rect, String, Line, Polygon, Group,
)
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import colors as gcolors

# ── Page geometry ─────────────────────────────────────────────────────────────
PW, PH    = LETTER           # 612 x 792 points
MARGIN    = 0.85 * inch
CONTENT_W = PW - 2 * MARGIN  # ~468 pt

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE  = HexColor('#1a2b4a')
MID_BLUE   = HexColor('#2d4a7a')
ACCENT     = HexColor('#3d6fa8')
LIGHT_BLUE = HexColor('#e8eef6')
CODE_BG    = HexColor('#f0f2f5')   # slightly darker for contrast
CODE_BDR   = HexColor('#c8d0da')
TH_BG      = HexColor('#2d4a7a')
TR_ALT     = HexColor('#eef2f7')
LIGHT_GRY  = HexColor('#777777')
MID_GRY    = HexColor('#444444')

# ── Unicode → safe ASCII substitution table ───────────────────────────────────
# ReportLab built-in fonts (Helvetica / Courier) are Latin-1 only.
# Every non-Latin-1 codepoint must be mapped here or it will render as a blank.
_SUBS = str.maketrans({
    # Dashes / quotes
    '\u2014': '--',   '\u2013': '-',
    '\u2018': "'",    '\u2019': "'",
    '\u201c': '"',    '\u201d': '"',
    '\u2026': '...',

    # Arithmetic / relational
    '\u00d7': 'x',    '\u2212': '-',
    '\u2265': '>=',   '\u2264': '<=',
    '\u2248': '~=',   '\u00b1': '+/-',
    '\u221e': 'inf',  '\u221a': 'sqrt',
    '\u2211': 'Sum',  '\u2208': 'in',
    '\u22c5': '*',    '\u00b7': '.',
    '\u2260': '!=',   '\u2261': '===',
    '\u00f7': '/',    '\u00d7': 'x',
    '\u2297': '(x)',  '\u2295': '(+)',

    # Greek (lowercase)
    '\u03b1': 'alpha',   '\u03b2': 'beta',    '\u03b3': 'gamma',
    '\u03b4': 'delta',   '\u03b5': 'eps',     '\u03b6': 'zeta',
    '\u03b7': 'eta',     '\u03b8': 'theta',   '\u03b9': 'iota',
    '\u03ba': 'kappa',   '\u03bb': 'lambda',  '\u03bc': 'mu',
    '\u03bd': 'nu',      '\u03be': 'xi',      '\u03c0': 'pi',
    '\u03c1': 'rho',     '\u03c3': 'sigma',   '\u03c4': 'tau',
    '\u03c5': 'upsilon', '\u03c6': 'phi',     '\u03c7': 'chi',
    '\u03c8': 'psi',     '\u03c9': 'omega',
    # Greek (uppercase)
    '\u0393': 'Gamma',  '\u0394': 'Delta',   '\u0398': 'Theta',
    '\u039b': 'Lambda', '\u039e': 'Xi',      '\u03a0': 'Pi',
    '\u03a3': 'Sigma',  '\u03a6': 'Phi',     '\u03a8': 'Psi',
    '\u03a9': 'Omega',

    # Superscripts / subscripts
    '\u207b': '^-', '\u207a': '^+',
    '\u00b9': '^1', '\u00b2': '^2', '\u00b3': '^3',
    '\u2074': '^4', '\u2075': '^5', '\u2076': '^6',
    '\u2077': '^7', '\u2078': '^8', '\u2079': '^9', '\u2070': '^0',
    '\u2080': '_0', '\u2081': '_1', '\u2082': '_2', '\u2083': '_3',

    # Arrows
    '\u2192': '->',  '\u2190': '<-',  '\u2194': '<->',
    '\u21d2': '=>',  '\u21d0': '<=',  '\u21d4': '<=>',
    '\u2191': '^',   '\u2193': 'v',
    '\u21a6': '|->',
    '\u25b8': '>',   '\u25c2': '<',
    '\u25b6': '>',   '\u25c0': '<',

    # Math letters / symbols
    '\u211d': 'R',   '\u2115': 'N',   '\u2124': 'Z',
    '\u00c2': 'A^',  '\u00ca': 'E^',
    '\u2202': 'd',   '\u222b': 'int', '\u220f': 'Prod',
    '\u221d': 'prop',

    # ── Box-drawing ─────────────────────────────────────────────────────────
    # Double-line (used in ASCII architecture diagrams)
    '\u2554': '+',  '\u2557': '+',   # ╔ ╗
    '\u255a': '+',  '\u255d': '+',   # ╚ ╝
    '\u2560': '+',  '\u2563': '+',   # ╠ ╣
    '\u2566': '+',  '\u2569': '+',   # ╦ ╩
    '\u256c': '+',                   # ╬
    '\u2551': '|',  '\u2550': '=',   # ║ ═

    # Single-line corners / junctions
    '\u250c': '+',  '\u2510': '+',   # ┌ ┐
    '\u2514': '+',  '\u2518': '+',   # └ ┘
    '\u251c': '+',  '\u2524': '+',   # ├ ┤
    '\u252c': '+',  '\u2534': '+',   # ┬ ┴
    '\u253c': '+',                   # ┼
    # Single-line strokes
    '\u2500': '-',  '\u2502': '|',   # ─ │
    # Heavy strokes
    '\u2501': '=',  '\u2503': '|',   # ━ ┃
    # Dashed
    '\u2504': '-',  '\u2505': '-',
    '\u2506': '|',  '\u2507': '|',
    '\u254c': '-',  '\u254d': '=',
    '\u254e': '|',  '\u254f': '|',

    # Pointer / filled triangles
    '\u25bc': 'v',  '\u25b2': '^',
    '\u25c4': '<',  '\u25ba': '>',
    '\u25cf': '*',  '\u25cb': 'o',
    '\u25a0': '#',  '\u25a1': '[]',

    # Circled numbers ① – ⑨
    '\u2460': '(1)', '\u2461': '(2)', '\u2462': '(3)',
    '\u2463': '(4)', '\u2464': '(5)', '\u2465': '(6)',
    '\u2466': '(7)', '\u2467': '(8)', '\u2468': '(9)',
    '\u2469': '(10)',

    # Misc
    '\u2022': '*',   '\u2023': '>',
    '\u00a0': ' ',   # non-breaking space
    '\u200b': '',    # zero-width space
})


def clean(s: str) -> str:
    """Apply Unicode → ASCII substitutions, then drop any remaining non-Latin-1."""
    s = s.translate(_SUBS)
    # Final safety: encode to Latin-1, dropping anything still not representable
    return s.encode('latin-1', errors='ignore').decode('latin-1')


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
    # Inline code → monospace
    text = re.sub(
        r'`([^`\n]+?)`',
        lambda m: (
            f'<font name="Courier" size="8.5" color="#24292e">'
            f'{xe(clean(m.group(1)))}</font>'
        ),
        text,
    )
    return text


# ── Paragraph styles ──────────────────────────────────────────────────────────
def _ps(name, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


S = {
    # H2 in markdown → major section title
    'h1': _ps('h1', fontName='Helvetica-Bold', fontSize=16,
              textColor=DARK_BLUE, spaceBefore=32, spaceAfter=8, leading=21),
    # H3
    'h2': _ps('h2', fontName='Helvetica-Bold', fontSize=13,
              textColor=MID_BLUE, spaceBefore=22, spaceAfter=6, leading=18),
    # H4
    'h3': _ps('h3', fontName='Helvetica-BoldOblique', fontSize=11,
              textColor=ACCENT, spaceBefore=16, spaceAfter=5, leading=15),
    # H5
    'h4': _ps('h4', fontName='Helvetica-Bold', fontSize=10,
              textColor=MID_GRY, spaceBefore=12, spaceAfter=4, leading=14),
    'body': _ps('body', fontName='Helvetica', fontSize=10,
                spaceBefore=4, spaceAfter=4, leading=15.5, alignment=TA_JUSTIFY),
    'bullet': _ps('bullet', fontName='Helvetica', fontSize=10,
                  spaceBefore=2, spaceAfter=2, leading=14.5,
                  leftIndent=20, firstLineIndent=0),
    'num': _ps('num', fontName='Helvetica', fontSize=10,
               spaceBefore=2, spaceAfter=2, leading=14.5,
               leftIndent=24, firstLineIndent=-14),
    'sub_bullet': _ps('sub_bullet', fontName='Helvetica', fontSize=9.5,
                      spaceBefore=1, spaceAfter=1, leading=13,
                      leftIndent=38, firstLineIndent=0, textColor=MID_GRY),
    'th': _ps('th', fontName='Helvetica-Bold', fontSize=9,
              textColor=white, alignment=TA_CENTER, leading=12),
    'td': _ps('td', fontName='Helvetica', fontSize=9,
              alignment=TA_LEFT, leading=13),
    'td_ctr': _ps('td_ctr', fontName='Helvetica', fontSize=9,
                  alignment=TA_CENTER, leading=13),
    'footer': _ps('footer', fontName='Helvetica', fontSize=8,
                  textColor=LIGHT_GRY, alignment=TA_CENTER),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_widths(n_cols: int, header_row: list[str]) -> list[float]:
    """
    Return column widths that sum to CONTENT_W.
    For 2-col tables the first column is usually a label → 35/65 split.
    For 3-col, heuristic 25/37.5/37.5.
    Otherwise equal.
    """
    if n_cols == 2:
        return [CONTENT_W * 0.35, CONTENT_W * 0.65]
    if n_cols == 3:
        return [CONTENT_W * 0.25, CONTENT_W * 0.375, CONTENT_W * 0.375]
    if n_cols == 4:
        return [CONTENT_W * 0.22, CONTENT_W * 0.26, CONTENT_W * 0.26, CONTENT_W * 0.26]
    return [CONTENT_W / n_cols] * n_cols


def make_code_block(lines: list[str], font_size: float = 7.5) -> list:
    """Render a fenced code block with grey background and border.

    Returns a list of Table flowables — long blocks are split into
    page-sized chunks so ReportLab never hits LayoutError.
    """
    # Usable frame height = page height minus margins
    _frame_h  = PH - MARGIN - (MARGIN + 0.15 * inch)
    _v_pad    = 28   # top (14) + bottom (14) cell padding
    _leading  = font_size * 1.4
    # Safety margin of 8 lines so the last chunk fits with room to spare
    _max_lines = max(30, int((_frame_h - _v_pad) / _leading) - 8)

    results: list = []
    chunks = [lines[s:s + _max_lines]
              for s in range(0, max(1, len(lines)), _max_lines)]
    for chunk in chunks:
        text = '\n'.join(clean(line) for line in chunk)
        pre_style = ParagraphStyle(
            'pre', fontName='Courier', fontSize=font_size,
            leading=_leading, textColor=HexColor('#1e2126'),
        )
        pre = Preformatted(text, pre_style)
        tbl = Table([[pre]], colWidths=[CONTENT_W])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), CODE_BG),
            ('BOX',           (0, 0), (-1, -1), 0.8, CODE_BDR),
            ('TOPPADDING',    (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
            ('LEFTPADDING',   (0, 0), (-1, -1), 16),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ]))
        results.append(tbl)
    return results


def make_table(rows: list[list[str]]) -> Table | None:
    """Build a styled markdown table.  First row is treated as the header."""
    if not rows:
        return None
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return None

    col_widths = _col_widths(n_cols, rows[0])

    data = []
    for i, row in enumerate(rows):
        padded = list(row) + [''] * (n_cols - len(row))
        if i == 0:
            cells = [Paragraph(md_inline(c.strip()), S['th']) for c in padded]
        else:
            cells = [Paragraph(md_inline(c.strip()), S['td']) for c in padded]
        data.append(cells)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1,  0), TH_BG),
        ('TEXTCOLOR',      (0, 0), (-1,  0), white),
        ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1,  0), 9),
        ('FONTNAME',       (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',       (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, TR_ALT]),
        ('GRID',           (0, 0), (-1, -1), 0.5, HexColor('#c0c8d8')),
        ('VALIGN',         (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',     (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 7),
        ('LEFTPADDING',    (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 10),
    ]))
    return tbl


def section_rule() -> HRFlowable:
    return HRFlowable(width='100%', thickness=0.5,
                      color=HexColor('#c8d1dc'),
                      spaceAfter=6, spaceBefore=6)


# ── Markdown parser ───────────────────────────────────────────────────────────

def parse_md(md: str) -> list:
    """Convert a markdown string to a list of ReportLab flowables."""
    flowables: list = []
    lines = md.split('\n')
    i = 0

    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()

        # ── Horizontal rule ──────────────────────────────────────────────────
        if re.match(r'^-{3,}$', line) or re.match(r'^\*{3,}$', line):
            flowables.append(Spacer(1, 6))
            flowables.append(section_rule())
            flowables.append(Spacer(1, 4))
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
                # Pick font size so the widest line fits within CONTENT_W.
                # Courier character width ≈ 0.6 * fontSize points.
                fs = 8.0
                if max_len > 80:
                    fs = 7.0
                if max_len > 96:
                    fs = 6.0
                if max_len > 116:
                    fs = 5.5
                flowables.append(Spacer(1, 6))
                flowables.extend(make_code_block(code_lines, fs))
                flowables.append(Spacer(1, 8))
            continue

        # ── Markdown table ───────────────────────────────────────────────────
        if line.startswith('|') and line.endswith('|'):
            table_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row_line = lines[i].strip()
                cells = [c for c in row_line.split('|')[1:-1]]
                if not all(re.match(r'^[-:| ]+$', c) for c in cells):
                    table_rows.append(cells)
                i += 1
            if table_rows:
                tbl = make_table(table_rows)
                if tbl:
                    flowables.append(Spacer(1, 8))
                    flowables.append(tbl)
                    flowables.append(Spacer(1, 10))
            continue

        # ── ATX Headers ──────────────────────────────────────────────────────
        m = re.match(r'^(#{1,5})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            text  = md_inline(m.group(2))

            if level == 1:
                # Document title already on title page; render as h1 in body
                flowables.append(Spacer(1, 8))
                flowables.append(Paragraph(text, S['h1']))
            elif level == 2:
                # Major section — accent underline
                flowables.append(Spacer(1, 14))
                flowables.append(Paragraph(text, S['h1']))
                flowables.append(HRFlowable(
                    width='100%', thickness=2.0, color=ACCENT,
                    spaceAfter=8, spaceBefore=3))
            elif level == 3:
                flowables.append(Paragraph(text, S['h2']))
            elif level == 4:
                flowables.append(Paragraph(text, S['h3']))
            else:
                flowables.append(Paragraph(text, S['h4']))
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
        m = re.match(r'^[-*+]\s+(.+)$', line)
        if m:
            text   = md_inline(m.group(1))
            indent = len(raw) - len(raw.lstrip())
            style  = S['sub_bullet'] if indent >= 4 else S['bullet']
            bchar  = '&#x2022;' if indent < 4 else '&#x25e6;'
            flowables.append(Paragraph(f'{bchar}&nbsp;&nbsp;{text}', style))
            i += 1
            continue

        # ── Blank line ────────────────────────────────────────────────────────
        if not line:
            flowables.append(Spacer(1, 5))
            i += 1
            continue

        # ── Body paragraph ────────────────────────────────────────────────────
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt
                    or nxt.startswith('#')
                    or nxt.startswith('```')
                    or nxt.startswith('|')
                    or re.match(r'^[-*+]\s+', nxt)
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


# ── Page template: footer + top rule ─────────────────────────────────────────

def _on_page(canvas, doc):
    canvas.saveState()

    # Footer bar
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(LIGHT_GRY)
    footer_y = 0.45 * inch
    canvas.drawCentredString(
        PW / 2, footer_y,
        f'Madison RL Intelligence Agent  --  Technical Report  |  Page {doc.page}',
    )

    # Thin footer rule above the text
    canvas.setStrokeColor(HexColor('#dde3ec'))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, footer_y + 12, PW - MARGIN, footer_y + 12)

    # Top rule on every page except the title page
    if doc.page > 1:
        canvas.line(MARGIN, PH - MARGIN + 6, PW - MARGIN, PH - MARGIN + 6)

    canvas.restoreState()


# ── Visual Architecture Diagram ───────────────────────────────────────────────

def _hex(h: str):
    """Convert hex string to reportlab Color."""
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
    from reportlab.lib.colors import Color
    return Color(r, g, b)


def _box(d: Drawing, x, y, w, h, fill_hex, stroke_hex, label_lines,
         font_size=8, bold=False, label_color='#ffffff', corner=4):
    """Draw a rounded-corner box with centred multi-line text."""
    fill   = _hex(fill_hex)
    stroke = _hex(stroke_hex)
    font   = 'Helvetica-Bold' if bold else 'Helvetica'

    rect = Rect(x, y, w, h,
                fillColor=fill, strokeColor=stroke, strokeWidth=1,
                rx=corner, ry=corner)
    d.add(rect)

    # Centre text vertically
    n      = len(label_lines)
    lh     = font_size * 1.4
    y_top  = y + h/2 + (n * lh)/2 - lh * 0.75
    lc     = _hex(label_color)
    for i, txt in enumerate(label_lines):
        s = String(x + w/2, y_top - i * lh, txt,
                   fontName=font, fontSize=font_size,
                   fillColor=lc, textAnchor='middle')
        d.add(s)


def _arrow(d: Drawing, x1, y1, x2, y2, color='#555555', label='', font_size=7):
    """Draw a vertical/horizontal arrow with optional label."""
    c = _hex(color)
    d.add(Line(x1, y1, x2, y2, strokeColor=c, strokeWidth=1.2))
    # Arrowhead (small triangle pointing down if y2 < y1)
    if y2 < y1:  # going down
        tip = (x2, y2)
        pts = [tip[0]-4, tip[1]+8, tip[0]+4, tip[1]+8, tip[0], tip[1]]
        d.add(Polygon(pts, fillColor=c, strokeColor=c, strokeWidth=0))
    else:  # going up
        tip = (x2, y2)
        pts = [tip[0]-4, tip[1]-8, tip[0]+4, tip[1]-8, tip[0], tip[1]]
        d.add(Polygon(pts, fillColor=c, strokeColor=c, strokeWidth=0))
    if label:
        mid_y = (y1 + y2) / 2
        d.add(String(x1 + 6, mid_y, label,
                     fontName='Helvetica-Oblique', fontSize=font_size,
                     fillColor=_hex('#444444'), textAnchor='start'))


def _harrow(d: Drawing, x1, y1, x2, label='', font_size=7):
    """Horizontal arrow left→right."""
    c = _hex('#555555')
    d.add(Line(x1, y1, x2, y1, strokeColor=c, strokeWidth=1.0))
    tip = (x2, y1)
    pts = [tip[0]-7, tip[1]-3, tip[0]-7, tip[1]+3, tip[0], tip[1]]
    d.add(Polygon(pts, fillColor=c, strokeColor=c, strokeWidth=0))
    if label:
        d.add(String((x1+x2)/2, y1+3, label,
                     fontName='Helvetica-Oblique', fontSize=font_size,
                     fillColor=_hex('#444444'), textAnchor='middle'))


def build_arch_drawing() -> Drawing:
    """
    Build a proper visual architecture flowchart using ReportLab graphics.
    Shows the full two-level RL hierarchy: PPO -> Agent Selection -> Bandit -> Tools.
    Fits within CONTENT_W x 520 pts.
    """
    W = CONTENT_W   # ~468 pt
    H = 530.0
    d = Drawing(W, H)

    # Background
    d.add(Rect(0, 0, W, H, fillColor=_hex('#f8f9fb'),
               strokeColor=_hex('#dde3ec'), strokeWidth=0.5))

    # ── Row y-positions (from top = H, so y increases downward in our coords
    #    but ReportLab y=0 is bottom) ──────────────────────────────────────
    # We'll place from top:
    #   row0:  Query input       y = H-60  (box top)
    #   row1:  PPO controller    y = H-160
    #   row2:  Agents row        y = H-265
    #   row3:  Bandit            y = H-370
    #   row4:  Tools row         y = H-450
    #   row5:  Env + Reward      y = H-530 (bottom)

    pad = 8   # horizontal padding from edge

    # ── 0. Query input ────────────────────────────────────────────────────
    qy = H - 55
    qh = 40
    qw = W - 2*pad
    _box(d, pad, qy, qw, qh, '#1a2b4a', '#0f1e35',
         ['RESEARCH QUERY',
          '396-dim observation: query_emb(384) | coverage(8) | scalars(4)'],
         font_size=8, bold=False, label_color='#e8eef6')

    # arrow query -> PPO
    _arrow(d, W/2, qy, W/2, qy - 22, label='396-dim obs')

    # ── 1. PPO Meta-Controller ────────────────────────────────────────────
    py = H - 175
    ph = 80
    pw = W - 2*pad
    _box(d, pad, py, pw, ph, '#2d4a7a', '#1a2b4a',
         ['PPO META-CONTROLLER  (Strategic)',
          'Shared: Linear(396->256)->ReLU->Linear(256->256)->ReLU',
          'Policy Head: Linear(256->4) + Softmax     |     Value Head: Linear(256->1)',
          'PPO-clip objective | GAE advantages | Adam lr=3e-4 | entropy coeff=0.01'],
         font_size=7.5, bold=False, label_color='#ffffff')

    # arrow PPO -> agents
    _arrow(d, W/2, py, W/2, py - 22, label='agent_idx in {0,1,2,3}')

    # ── 2. Agent selection (4 boxes) ──────────────────────────────────────
    ay = H - 295
    ah = 52
    aw = (W - 2*pad - 9) / 4   # 4 boxes with 3 gaps of 3pt
    agent_data = [
        ('SEARCH', '#3d6fa8', ['web_scraper', 'api_client', 'acad_search']),
        ('EVALUATOR', '#2d7a5a', ['cred_scorer', 'web_scraper']),
        ('SYNTHESIS', '#7a4a2d', ['rel_extractor', 'pdf_parser']),
        ('DEEP DIVE', '#5a2d7a', ['web_scraper', 'api_client', 'rel_extr']),
    ]
    ax_positions = []
    for i, (name, color, tools) in enumerate(agent_data):
        ax = pad + i * (aw + 3)
        ax_positions.append(ax + aw/2)
        _box(d, ax, ay, aw, ah, color, '#1a2b4a',
             [name, ', '.join(tools[:2]), tools[2] if len(tools)>2 else ''],
             font_size=6.5, bold=True, label_color='#ffffff')

    # ── 3. Bandit ──────────────────────────────────────────────────────────
    by_ = H - 390
    bh  = 58
    bw  = W - 2*pad
    # arrow agents -> bandit
    _arrow(d, W/2, ay, W/2, by_ + bh, label='agent_name')

    _box(d, pad, by_, bw, bh, '#4a3d6f', '#2d1a4a',
         ['PER-AGENT CONTEXTUAL BANDIT  (Tactical)  -  LinUCB + Novelty Bonus',
          'score_i = theta_i^T * ctx + alpha * sqrt(ctx^T * A_i^-1 * ctx)  +  beta/sqrt(count+1)',
          'Context: first 128 dims of obs | Online update: Sherman-Morrison O(d^2)',
          '4 independent bandits (one per agent) | 6 tool arms each'],
         font_size=7.0, bold=False, label_color='#e8d8ff')

    # ── 4. Tool boxes ──────────────────────────────────────────────────────
    ty = H - 470
    th = 42
    tw = (W - 2*pad - 5*2) / 6
    tool_data = [
        ('web_scraper', '#Wikipedia', '#2a5a8a'),
        ('api_client', '#OpenAlex', '#2a6a5a'),
        ('acad_search', '#arXiv', '#6a4a2a'),
        ('pdf_parser', '#PDF/arXiv', '#5a3a7a'),
        ('rel_extractor', '#TF-IDF+Sem', '#3a6a4a'),
        ('cred_scorer', '#5-signal', '#7a4a4a'),
    ]
    _arrow(d, W/2, by_, W/2, ty + th, label='tool_idx')

    for i, (name, api, color) in enumerate(tool_data):
        tx = pad + i * (tw + 2)
        _box(d, tx, ty, tw, th, color, '#1a2b4a',
             [name, api],
             font_size=6.2, bold=False, label_color='#ffffff', corner=3)

    # ── 5. Environment + reward feedback ──────────────────────────────────
    ey = H - 530
    eh = 38
    ew = W - 2*pad
    _arrow(d, W/2, ty, W/2, ey + eh, label='Real API call')

    _box(d, pad, ey, ew, eh, '#1a4a2a', '#0f2a18',
         ['RESEARCH ENVIRONMENT  |  Reward: R = 1.0*cred + 0.5*cov - 0.3*lat + 0.4*div',
          'next_obs -> PPO (feedback loop)  |  Terminal: coverage > 0.9 or budget <= 0'],
         font_size=7.5, bold=False, label_color='#c8ffc8')

    # Feedback arrow (right side, going back up)
    fb_x = W - pad - 2
    d.add(Line(fb_x, ey + eh/2, fb_x, qy + qh/2,
               strokeColor=_hex('#cc4444'), strokeWidth=1.2,
               strokeDashArray=[4, 3]))
    d.add(String(fb_x - 3, (ey + qy)/2 + 15, 'next_obs',
                 fontName='Helvetica-Oblique', fontSize=6.5,
                 fillColor=_hex('#cc4444'), textAnchor='end'))
    # Arrowhead at top
    tip_y = qy + qh/2
    pts = [fb_x-4, tip_y-8, fb_x+4, tip_y-8, fb_x, tip_y]
    d.add(Polygon(pts, fillColor=_hex('#cc4444'),
                  strokeColor=_hex('#cc4444'), strokeWidth=0))

    return d


def arch_drawing_flowable() -> Table:
    """Wrap the architecture Drawing in a Table so it flows with the story."""
    drw = build_arch_drawing()
    from reportlab.platypus import Image
    import io
    from reportlab.graphics import renderSVG
    # Use renderPDF to get the drawing as a flowable directly
    from reportlab.platypus import Flowable

    class _DrawingFlowable(Flowable):
        def __init__(self, drawing):
            Flowable.__init__(self)
            self._drawing = drawing
            self.width  = drawing.width
            self.height = drawing.height

        def draw(self):
            renderPDF.draw(self._drawing, self.canv, 0, 0)

    return _DrawingFlowable(drw)


# ── Title page ────────────────────────────────────────────────────────────────

def build_title_page() -> list:
    story = []
    story.append(Spacer(1, 0.9 * inch))

    # Main title banner
    banner = Table([[
        Paragraph(
            '<font size="22" color="#ffffff"><b>Madison RL Intelligence Agent</b></font>',
            _ps('banner', fontName='Helvetica-Bold', fontSize=22,
                textColor=white, alignment=TA_CENTER, leading=28),
        )
    ]], colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), DARK_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 22),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 22),
        ('LEFTPADDING',   (0, 0), (-1, -1), 24),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 24),
    ]))
    story.append(banner)

    # Subtitle strip
    sub_tbl = Table([[
        Paragraph(
            'Technical Report',
            _ps('sub', fontName='Helvetica-Bold', fontSize=15,
                textColor=MID_BLUE, alignment=TA_CENTER, leading=20),
        )
    ]], colWidths=[CONTENT_W])
    sub_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_BLUE),
        ('TOPPADDING',    (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 11),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 0.25 * inch))

    desc_style = _ps('desc', fontName='Helvetica', fontSize=10,
                     textColor=MID_GRY, alignment=TA_CENTER, leading=16)
    story.append(Paragraph(
        'Hierarchical Reinforcement Learning for Multi-Agent Research Orchestration',
        desc_style,
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        'PPO Meta-Controller  |  LinUCB Contextual Bandits  |  '
        'Transfer Learning  |  Novelty Bonus  |  Real Agent Execution',
        _ps('desc2', fontName='Helvetica-Oblique', fontSize=9,
            textColor=LIGHT_GRY, alignment=TA_CENTER, leading=14),
    ))
    story.append(Spacer(1, 0.25 * inch))
    story.append(HRFlowable(width='55%', thickness=1.5, color=ACCENT,
                             hAlign='CENTER', spaceAfter=0))
    story.append(Spacer(1, 0.25 * inch))

    # Key results table (2 columns → 35/65 split)
    kr_rows = [
        ['Metric', 'Result'],
        ['Full System vs Random Baseline',   '+268%  (26.60 vs 7.23 mean reward)'],
        ['Full System vs Heuristic',         '+151%  (26.60 vs 10.60)'],
        ['Training Mean Reward (Last 50)',    '27.88  |  Peak 37.58  |  5,000 episodes'],
        ['Transfer Learning Advantage',      '+13.5%  (fine-tune vs from-scratch)'],
        ['Live Inference (Real APIs)',        'Wikipedia + arXiv + OpenAlex  |  0.57 avg coverage'],
        ['Experiments',                      '5,000 training episodes + 3-seed ablation x 5 configs'],
        ['Code & Data',                      'github.com/UshakeShravya/madison-rl-intel'],
    ]
    kr_tbl = make_table(kr_rows)
    if kr_tbl:
        story.append(kr_tbl)

    story.append(PageBreak())
    return story


# ── Main build ────────────────────────────────────────────────────────────────

def build_pdf(output: str = 'docs/technical_report.pdf') -> None:
    repo = Path(__file__).parent.parent  # project root

    report_md  = (repo / 'docs' / 'technical_report.md').read_text(encoding='utf-8')
    diagram_md = (repo / 'docs' / 'architecture_diagram.md').read_text(encoding='utf-8')

    # Strip the H1 title from architecture_diagram.md
    diag_lines = diagram_md.split('\n')
    if diag_lines and diag_lines[0].startswith('# '):
        diag_lines = diag_lines[1:]
    diag_body = '\n'.join(diag_lines).lstrip('\n')

    # Insert architecture section before "## Mathematical Formulation"
    insert_after = '## Mathematical Formulation'
    arch_section = (
        '\n\n---\n\n'
        '## Architecture Diagrams\n\n'
        + diag_body
        + '\n\n---\n\n'
    )

    combined_md = (
        report_md.replace(insert_after, arch_section + insert_after, 1)
        if insert_after in report_md
        else report_md + arch_section
    )

    # Drop the document H1 (already on title page)
    md_lines    = combined_md.split('\n')
    body_start  = next(
        (idx for idx, l in enumerate(md_lines) if l.startswith('## ')), 0
    )
    body_md = '\n'.join(md_lines[body_start:])

    # Build story — visual arch diagram goes right after title page
    story: list = []
    story.extend(build_title_page())

    # ── Visual Architecture Diagram (page 2) ──────────────────────────────
    story.append(Paragraph('System Architecture — Visual Overview', S['h1']))
    story.append(HRFlowable(width='100%', thickness=2.0, color=ACCENT,
                             spaceAfter=10, spaceBefore=3))
    story.append(Paragraph(
        'The two-level hierarchical control loop: PPO meta-controller selects the agent role; '
        'per-agent LinUCB contextual bandits select the API tool. '
        'A dashed red line shows the observation feedback from the environment back to the PPO controller.',
        S['body'],
    ))
    story.append(Spacer(1, 10))
    story.append(arch_drawing_flowable())
    story.append(Spacer(1, 12))

    # Key-numbers summary row
    kn_rows = [
        ['Component', 'Key Parameters'],
        ['PPO Controller',   'obs=396-dim | actions=4 | hidden=256 | lr=3e-4 | clip=0.2'],
        ['Contextual Bandit','LinUCB alpha=1.0 | novelty beta=0.1 | context=128-dim | arms=6'],
        ['Reward Function',  'R = 1.0*cred + 0.5*cov - 0.3*lat + 0.4*div + 0.6*(rel*cov)'],
        ['Training',         '5,000 episodes | 20 steps/ep | batch=64 | PPO epochs=4'],
        ['Live Inference',   'Wikipedia (REST) + OpenAlex + arXiv XML | sentence-transformers'],
    ]
    kn_tbl = make_table(kn_rows)
    if kn_tbl:
        story.append(kn_tbl)
    story.append(PageBreak())

    story.extend(parse_md(body_md))

    out_path = str(repo / output) if not Path(output).is_absolute() else output
    doc = SimpleDocTemplate(
        out_path,
        pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN + 0.2 * inch,
        title='Madison RL Intelligence Agent: Technical Report',
        author='UshakeShravya',
        subject='Hierarchical RL for Multi-Agent Research Orchestration',
        creator='reportlab + docs/build_pdf.py',
    )
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f'PDF written -> {out_path}')


if __name__ == '__main__':
    build_pdf()
