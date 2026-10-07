"""Adapter for the modified Qwen recognizer (salad).

salad (surendirakrishna/salad) is a modified Qwen vision-language model
(Qwen3.5 architecture) with an extrapolated vision tower. It always answers
with layout HTML (``<div data-label=... data-bbox=...>``) rather than the
plain text / table HTML / LaTeX that GLM-OCR returns per region. This module:

* builds requests the way the model expects (its OCR prompt, 28px-grid scaling,
  PNG), and pads small crops because thin line crops make it hallucinate;
* converts the HTML answer back into GLM-OCR's per-region content so the
  result formatter and the ocr.z.ai frontend work unchanged;
* detects runaway repetition so a looping generation can be cut short.

Prompt and ``scale_to_fit`` follow the model's own inference code.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from PIL import Image, ImageOps

FAMILIES = ("salad", "modified_qwen")

_ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td",
    "p", "th", "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li",
    "input", "a", "span", "img", "hr", "tbody", "small", "caption", "strong",
    "thead", "big", "code", "chem",
]
_ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border",
    "value", "style", "href", "alt", "align", "data-bbox", "data-label",
]

OCR_PROMPT = f"""
OCR this image to HTML.

Only use these tags {_ALLOWED_TAGS}, and these attributes {_ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()


_LAYOUT_LABELS = [
    "Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header",
    "Page-Footer", "Image", "Section-Header", "Table", "Text", "Complex-Block",
    "Code-Block", "Form", "Table-Of-Contents", "Figure", "Chemical-Block",
    "Diagram", "Bibliography", "Blank-Page",
]

# Whole-page prompt: same guidelines, answer as labelled layout blocks.
OCR_LAYOUT_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks.  Each layout block should be "
    "a div with the data-bbox attribute representing the bounding box of the block in "
    "x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the "
    "label for the block.\n\nUse the following labels:\n"
    + "\n".join(f"- {label}" for label in _LAYOUT_LABELS)
    + "\n\n"
    + OCR_PROMPT.split("\n\n", 1)[1]
)


def is_modified_qwen_family(family: Optional[str]) -> bool:
    return (family or "").lower() in FAMILIES


# ----------------------------------------------------------------------
# Request side
# ----------------------------------------------------------------------


def pad_region(img: Image.Image, padding: int, min_canvas: int = 0) -> Image.Image:
    """Add a white margin, then place small crops on a white page-like canvas.

    The model is trained on whole pages: a heading crop sent as a ~200x40 image
    makes it invent text (journal headers, numbered outlines). Pasting the
    crop at native scale onto a canvas of at least *min_canvas* px per side
    avoids that; upscaling the crop does not.
    """
    img = img.convert("RGB")
    if padding > 0:
        img = ImageOps.expand(img, border=padding, fill=(255, 255, 255))
    w, h = img.size
    if w >= min_canvas and h >= min_canvas:
        return img
    canvas = Image.new("RGB", (max(w, min_canvas), max(h, min_canvas)), "white")
    canvas.paste(img, (0, 0))
    return canvas


def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (1792, 28),
    grid_size: int = 28,
) -> Image.Image:
    """Resize to the model's pixel budget, snapped to its 28px patch grid."""
    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_size = (w_blocks * grid_size, h_blocks * grid_size)
    if new_size == (width, height):
        return img
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)


# ----------------------------------------------------------------------
# Response side: salad HTML -> GLM-OCR region content
# ----------------------------------------------------------------------

_LOOSE_DIV_RE = re.compile(r"</?div\b[^>]*>", re.IGNORECASE)
# A list item whose text already carries its own marker ("6.", "a)", "(ii)")
# would otherwise come out as "1. 6. ..." / "- a) ...".
_DOUBLE_MARKER_RE = re.compile(
    r"^(\s*)(?:[-*+]|\d+\.) +((?:\(?(?:\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,6})[.)])|[•·▪●○■])(?= )",
    re.MULTILINE,
)
# Whole-line bold on a heading ("## **Title**") is redundant.
_BOLD_HEADING_RE = re.compile(r"^(#{1,6} +)\*\*(.+?)\*\*\s*$", re.MULTILINE)


def _soup(html: str):
    import warnings

    from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        return BeautifulSoup(html, "html.parser")


def _block_bodies(html: str) -> list:
    """Return the inner HTML of each top-level layout div (or the whole answer)."""
    soup = _soup(html)
    divs = soup.find_all("div", recursive=False)
    if not divs:
        return [html]
    bodies = []
    for div in divs:
        if div.get("data-label") == "Blank-Page":
            continue
        for img in div.find_all("img"):
            img.decompose()
        for tag in div.find_all(True):
            for attr in [a for a in tag.attrs if a.startswith("data-")]:
                del tag[attr]
        bodies.append(str(div.decode_contents()))
    return bodies


def _column_count(table) -> int:
    widths = []
    for tr in table.find_all("tr"):
        widths.append(sum(int(c.get("colspan") or 1) for c in tr.find_all(["td", "th"])))
    return max(widths) if widths else 0


def _compact(table_html: str) -> str:
    """One-line table HTML: blank lines inside a table end the HTML block in
    Markdown, and the formatter turns single newlines into blank lines."""
    return re.sub(r">\s+<", "><", table_html).strip()


def merge_split_tables(tables: list) -> list:
    """Join consecutive tables with the same column count into one.

    Salad sometimes answers a table crop with a header-only <table> followed
    by a body <table>; one detected table region holds one table.
    """
    merged = []
    for t in tables:
        prev = merged[-1] if merged else None
        if prev is not None and _column_count(prev) == _column_count(t) > 0:
            body = prev.find("tbody")
            if body is None:
                body = _soup("<tbody></tbody>").tbody
                prev.append(body)
            for tr in t.find_all("tr"):
                body.append(tr.extract())
            continue
        merged.append(t)
    return merged


def _markdown(html: str) -> str:
    from markdownify import MarkdownConverter

    class _Converter(MarkdownConverter):
        def convert_math(self, el, text, parent_tags):
            text = el.get_text().strip()
            if el.get("display") == "block":
                return f"\n$$\n{text}\n$$\n"
            return f"${text}$"

        def convert_table(self, el, text, parent_tags):
            return "\n\n" + _compact(str(el)) + "\n\n"

        def convert_chem(self, el, text, parent_tags):
            return el.get_text()

    md = _Converter(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=False,
        escape_asterisks=False,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
    ).convert_soup(_soup(html))
    md = _DOUBLE_MARKER_RE.sub(r"\1\2", md)
    md = _BOLD_HEADING_RE.sub(r"\1\2", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def to_region_content(raw: Optional[str], task_type: str) -> str:
    """Convert a salad HTML answer into GLM-OCR region content."""
    if not raw:
        return ""
    raw = raw.strip()
    try:
        bodies = _block_bodies(raw)
        joined = "".join(bodies)

        if task_type == "table":
            tables = [t for t in _soup(joined).find_all("table") if t.find_parent("table") is None]
            if tables:
                return "\n".join(_compact(str(t)) for t in merge_split_tables(tables))

        if task_type == "formula":
            maths = _soup(joined).find_all("math")
            if maths:
                latex = "\n".join(m.get_text().strip() for m in maths)
                return f"$$\n{latex}\n$$"

        md = "\n\n".join(m for m in (_markdown(b) for b in bodies) if m)
        if task_type == "formula":
            md = md.strip().strip("$").strip()
            return f"$$\n{md}\n$$" if md else ""
        return md
    except Exception:
        # Never lose text because the HTML was malformed (e.g. cut off by a
        # loop abort): strip tags and keep the words.
        return re.sub(r"<[^>]+>", " ", _LOOSE_DIV_RE.sub("", raw)).strip()


# ----------------------------------------------------------------------
# Loop detection (used while streaming)
# ----------------------------------------------------------------------


def find_loop_start(text: str, min_span: int, max_unit: int = 400) -> Optional[int]:
    """Return where a runaway repetition at the end of *text* starts, or None.

    A loop is a unit repeated back-to-back at the tail whose repeated span
    covers at least *min_span* characters. *min_span* must be generous for
    tables: rows of identical cells (``<td>-</td>``) legitimately repeat.
    """
    n = len(text)
    if n < min_span:
        return None
    for unit in range(1, min(max_unit, n // 2) + 1):
        tail = text[n - unit:]
        pos = n - unit
        while pos - unit >= 0 and text[pos - unit:pos] == tail:
            pos -= unit
        span = n - pos
        # Need several repeats, not just a long unit seen twice.
        if span >= min_span and span // unit >= 4:
            return pos + unit  # keep one copy of the repeated unit
    return None
