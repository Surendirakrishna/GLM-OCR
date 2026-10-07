"""Second pass that fixes Indic script confusion on short regions.

Recognised one crop at a time, a short heading or list item has no context,
and salad confuses look-alike scripts: Kannada comes back in Telugu script,
Gujarati in Devanagari. Long paragraphs on the same page are read correctly.
So after a page is recognised, each region's output script is compared with
the script of the text around it; a region that disagrees is read again with
a hint naming the surrounding script ("Transcribe each word in the script it
is printed in" keeps genuinely different-script text, e.g. English, intact).
The new reading is kept only if it is in the surrounding script.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

# Unicode blocks of the scripts used for Indian languages.
INDIC_SCRIPTS = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    # Not an Indian script, but salad drifts into it on scanned Malayalam.
    "Sinhala": (0x0D80, 0x0DFF),
}

# Context needed before a region is second-guessed (characters of one script).
MIN_CONTEXT_CHARS = 40


def script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text or "":
        cp = ord(ch)
        if cp < 0x0900 or cp > 0x0DFF:
            continue
        for name, (lo, hi) in INDIC_SCRIPTS.items():
            if lo <= cp <= hi:
                counts[name] += 1
                break
    return counts


def has_native_digits(text: str) -> bool:
    """True if *text* uses an Indic script's own digits (০-৯, ०-९, ...)."""
    for ch in text or "":
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F and 0x66 <= (cp & 0x7F) <= 0x6F:
            return True
    return False


def dominant_script(text: str) -> Optional[str]:
    counts = script_counts(text)
    return counts.most_common(1)[0][0] if counts else None


def hint_for(script: str) -> str:
    return (
        f"\n* Language: this document is mainly written in {script} script. "
        "Transcribe each word in the script it is printed in."
    )


def _context_script(idx: int, counts: List[Counter]) -> Optional[str]:
    """Script of the other regions, weighted by length and reading-order distance."""
    weights: Counter = Counter()
    raw: Counter = Counter()
    for j, c in enumerate(counts):
        if j == idx:
            continue
        for script, n in c.items():
            weights[script] += n / (1 + abs(idx - j))
            raw[script] += n
    if not weights:
        return None
    script, _ = weights.most_common(1)[0]
    return script if raw[script] >= MIN_CONTEXT_CHARS else None


def table_shape(table_html: str) -> Tuple[int, int]:
    """(rows, columns) of an HTML table, counting colspans."""
    from glmocr.adapters.modified_qwen import _column_count, _soup

    tables = _soup(table_html or "").find_all("table")
    if not tables:
        return (0, 0)
    return (
        sum(len(t.find_all("tr")) for t in tables),
        max(_column_count(t) for t in tables),
    )


def fix_page(
    regions: List[Dict[str, Any]],
    recognize,
    max_workers: int = 8,
    reread_indic_tables: bool = False,
) -> int:
    """Second pass over one page's recognised regions (in reading order).

    * Text regions whose Indic script disagrees with their context are read
      again with a script hint; kept if the new reading is in that script.
    * With *reread_indic_tables*, tables using native Indic digits are read
      again from an enlarged crop (small digits like ৩/০ are misread at
      native scale); kept only if rows x columns are unchanged, because
      enlarging can flip how an ambiguous header is structured.

    Regions that can be re-read carry their crop as ``region["_crop"]``.
    *recognize(crop, task_type, prompt_extra, upscale)* returns the new
    content or None. Returns the number of regions changed.
    """
    ordered = sorted(regions, key=lambda r: r.get("index", 0))
    counts = [script_counts(r.get("content") or "") for r in ordered]

    jobs = []  # (region, kind, own, context)
    for i, region in enumerate(ordered):
        if region.get("_crop") is None or not counts[i]:
            continue
        own = counts[i].most_common(1)[0][0]
        if region.get("task_type") == "table":
            if reread_indic_tables and has_native_digits(region.get("content")):
                jobs.append((region, "table", own, own))
            continue
        context = _context_script(i, counts)
        if context and own != context:
            jobs.append((region, "script", own, context))
    if not jobs:
        return 0

    def rerun(job):
        region, kind, own, context = job
        try:
            if kind == "table":
                new = recognize(region["_crop"], "table", "", True)
            else:
                new = recognize(region["_crop"], region.get("task_type", "text"), hint_for(context), False)
        except Exception as e:  # keep the first reading
            logger.warning("Re-read failed: %s", e)
            return job, None
        return job, new

    changed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
        for (region, kind, own, context), new in ex.map(rerun, jobs):
            if not new or new == region.get("content"):
                continue
            if kind == "table":
                ok = table_shape(new) == table_shape(region.get("content"))
            else:
                ok = dominant_script(new) == context
            if ok:
                logger.info("%s fix: region %s (%s -> %s)", kind.capitalize(),
                            region.get("index"), own, context)
                region["content"] = new
                changed += 1
    return changed
