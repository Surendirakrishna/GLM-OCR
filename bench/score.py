"""Score OCR output on the bench pages (see make_pages.py).

Runs every page in DATA through a method, saves the markdown, and scores:

* text CER  - edit distance of all non-table text vs ground truth, after
  NFC normalisation, markdown stripping and whitespace collapsing;
* table cell accuracy - each GT table is matched to its best output table;
  both are expanded to a grid (rowspan/colspan filled in) and compared cell
  by cell (missing rows/cols count as wrong);
* table shape - fraction of GT tables whose expanded grid has the same
  rows x cols.

Methods:
  harness  POST each image to the glmocr harness (/glmocr/parse)
  native   salad on the full page with its layout prompt (no crops)

Usage: python bench/score.py harness|native [--data bench/data] [--tag NAME]
       python bench/score.py --compare TAG1 TAG2 ...
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import requests
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).parent
TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"<[^>]+>", " ", s)  # stray tags (<br>, <b>, <sup>)
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\([%_&#$])", r"\1", s)
    s = re.sub(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|>\s*)", "", s, flags=re.M)
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", s)
    s = s.replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm_cell(s: str) -> str:
    return norm(s).replace(" ", "")


# ----------------------------------------------------------------------
# HTML table -> grid
# ----------------------------------------------------------------------


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cur, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cur = []
        elif tag in ("td", "th"):
            a = dict(attrs)
            self.cell = {
                "text": "",
                "rs": int(a.get("rowspan") or 1),
                "cs": int(a.get("colspan") or 1),
            }
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += " "

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            if self.cur is None:
                self.cur = []
            self.cur.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.cur is not None:
            self.rows.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def table_grid(html_text: str) -> list[list[str]]:
    p = _TableParser()
    try:
        p.feed(html_text)
    except Exception:
        return []
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(p.rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(max(1, min(cell["rs"], 50))):
                for dc in range(max(1, min(cell["cs"], 50))):
                    grid[(r + dr, c + dc)] = norm_cell(cell["text"])
            c += max(1, cell["cs"])
    if not grid:
        return []
    nr = max(r for r, _ in grid) + 1
    nc = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(nc)] for r in range(nr)]


def grid_score(gt: list[list[str]], out: list[list[str]]) -> float:
    if not gt:
        return 0.0
    total = sum(len(r) for r in gt)
    hit = 0
    for r, row in enumerate(gt):
        for c, cell in enumerate(row):
            if r < len(out) and c < len(out[r]) and out[r][c] == cell:
                hit += 1
    extra = max(0, sum(len(r) for r in out) - total)
    return hit / (total + extra)


# ----------------------------------------------------------------------
# Scoring one page
# ----------------------------------------------------------------------


def score_page(md: str, gt: dict) -> dict:
    out_tables = TABLE_RE.findall(md)
    out_text = norm(TABLE_RE.sub(" ", md))
    gt_text = norm(" ".join(gt["text"]))
    cer = Levenshtein.distance(out_text, gt_text) / max(1, len(gt_text))

    out_grids = [table_grid(t) for t in out_tables]
    cell_scores, shape_ok = [], []
    for t in gt["tables"]:
        g = table_grid(t)
        best, best_grid = 0.0, None
        for og in out_grids:
            s = grid_score(g, og)
            if s >= best:
                best, best_grid = s, og
        cell_scores.append(best)
        shape_ok.append(
            bool(best_grid) and len(best_grid) == len(g) and len(best_grid[0]) == len(g[0])
        )
    return {
        "cer": round(min(cer, 1.0), 4),
        "cells": round(sum(cell_scores) / len(cell_scores), 4) if cell_scores else None,
        "shape": round(sum(shape_ok) / len(shape_ok), 4) if shape_ok else None,
        "n_out_tables": len(out_tables),
    }


# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------


def run_harness(path: Path, url: str) -> str:
    r = requests.post(url, json={"images": [str(path.resolve())]}, timeout=900)
    r.raise_for_status()
    return r.json().get("markdown_result") or ""


def run_native(path: Path, url: str) -> str:
    sys.path.insert(0, str(HERE.parent))
    from PIL import Image

    from glmocr.adapters import modified_qwen

    img = modified_qwen.scale_to_fit(Image.open(path).convert("RGB"))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    payload = {
        "model": "salad",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": modified_qwen.OCR_LAYOUT_PROMPT},
        ]}],
        "max_tokens": 12384, "temperature": 0, "top_p": 0.1,
    }
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return modified_qwen.to_region_content(raw, "text")


def summarise(rows: dict) -> dict:
    def mean(key, pred=lambda n: True):
        vals = [v[key] for n, v in rows.items() if v.get(key) is not None and pred(n)]
        return round(sum(vals) / len(vals), 4) if vals else None

    is_scan = lambda n: n.endswith("_scan")
    is_en = lambda n: n.startswith("en_")
    return {
        "cer_all": mean("cer"), "cells_all": mean("cells"), "shape_all": mean("shape"),
        "cer_en": mean("cer", is_en), "cer_indic": mean("cer", lambda n: not is_en(n)),
        "cells_en": mean("cells", is_en), "cells_indic": mean("cells", lambda n: not is_en(n)),
        "cer_clean": mean("cer", lambda n: not is_scan(n)), "cer_scan": mean("cer", is_scan),
        "cells_clean": mean("cells", lambda n: not is_scan(n)), "cells_scan": mean("cells", is_scan),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", nargs="?", choices=["harness", "native"])
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="substring filter on page names")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()

    results_dir = HERE / "results"
    if args.compare:
        keys = None
        for tag in args.compare:
            s = json.loads((results_dir / tag / "summary.json").read_text())["summary"]
            keys = keys or list(s)
            if tag == args.compare[0]:
                print("tag".ljust(22) + "".join(k.rjust(12) for k in keys))
            print(tag.ljust(22) + "".join(f"{s[k]:12.4f}" if s[k] is not None else " " * 12 for k in keys))
        return

    data = Path(args.data)
    tag = args.tag or args.method
    out_dir = results_dir / tag
    (out_dir / "md").mkdir(parents=True, exist_ok=True)
    url = args.url or {
        "harness": "http://127.0.0.1:5012/glmocr/parse",
        "native": "http://127.0.0.1:30000/v1/chat/completions",
    }[args.method]
    fn = run_harness if args.method == "harness" else run_native
    pages = sorted(p for p in data.glob("*.png") if not args.only or args.only in p.name)

    def work(p):
        t0 = time.time()
        try:
            md = fn(p, url)
        except Exception as e:  # keep going; an error scores as empty output
            print(f"  {p.stem}: ERROR {e}", file=sys.stderr)
            md = ""
        (out_dir / "md" / f"{p.stem}.md").write_text(md)
        gt = json.loads((data / "gt" / f"{p.stem}.json").read_text())
        s = score_page(md, gt)
        s["seconds"] = round(time.time() - t0, 1)
        return p.stem, s

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        rows = dict(ex.map(work, pages))
    for n, s in sorted(rows.items()):
        print(f"  {n:24s} cer={s['cer']:.4f}  cells={s['cells'] if s['cells'] is not None else '-':<7} "
              f"shape={s['shape'] if s['shape'] is not None else '-':<5} {s['seconds']}s")
    summary = summarise(rows)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "pages": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
