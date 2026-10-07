# GLM-OCR harness with salad

This checkout runs the GLM-OCR document pipeline and its ocr.z.ai web UI with
two models swapped out:

| Step | Upstream GLM-OCR | Here |
|---|---|---|
| Layout detection | PP-DocLayoutV3 (in process) | **docling-layout-heron-101** (docker service) |
| Region recognition | GLM-OCR | **salad** (`surendirakrishna/salad`, modified Qwen) |

It is tuned for **English, Indian languages and tables**. On the benchmark in
`bench/`, the harness reads Indic text with 0.8% character error and gets
97.6% of table cells right (see [Benchmark](#benchmark)).

```
PDF / image ──► backend :8010 ──► glmocr harness :5012 ──► heron :8600   (layout boxes)
   (UI :3006)    pages in            crop each region   ──► salad :30000  (HTML per region)
                 parallel            HTML → text / <table> / $$latex$$ ──► Markdown + JSON
```

## Quick start

```bash
apps/start-salad.sh
```

The script stops other GPU model containers (`sglang_mineru25_pro`,
the other salad variant), starts `sglang_salad_fp8` and
`docling-layout-heron101`,
waits for both, then starts the harness, backend and UI. Open
**http://localhost:3006**.

- **Remote server:** the UI is reached through an SSH tunnel. In VS Code use
  the Ports panel → *Forward a Port* → `3006`, or run
  `ssh -N -L 3006:localhost:3006 <user>@<server>` on your laptop. Only 3006
  needs forwarding; the UI proxies `/api` to the backend.
- **Stop:** Ctrl+C (or kill the script) stops the harness, backend and UI.
  The model containers keep running; `docker stop sglang_salad_fp8` frees the GPU.
- **Logs:** `apps/logs/{harness,backend,frontend}.log`.
- **Give the GPU back to MinerU:**
  `docker stop sglang_salad_fp8 && docker start sglang_mineru25_pro`.
  `apps/start-salad.sh` takes the GPU back.

### Services

| Service | Port | Notes |
|---|---|---|
| salad (SGLang, docker `sglang_salad_fp8`) | 30000 | `--served-model-name salad`, FP8 weights (`--quantization fp8`), ~26 GB VRAM incl. KV cache, max 20 concurrent requests. bf16 original: `sglang_salad` (`SALAD_CONTAINER=sglang_salad apps/start-salad.sh`) |
| heron layout (docker `docling-layout-heron101`) | 8600 | `POST /detect`, ~1 GB VRAM |
| glmocr harness (`configs/salad.yaml`) | 5012 | `POST /glmocr/parse {"images": [path, ...]}` |
| backend (FastAPI, `apps/backend`) | 8010 | env `layout_ocr_url` points at the harness |
| frontend (vite, `apps/frontend`) | 3006 | env `BACKEND_URL` sets the `/api` proxy target |

Only one VLM fits on the 32 GB GPU at a time. Environments: the SDK and
harness use `../.venv` (editable `glmocr`), the backend uses
`apps/backend/.venv`, and the frontend runs with `npx pnpm@9`.

### Call the harness directly

```bash
curl -s -X POST localhost:5012/glmocr/parse -H 'Content-Type: application/json' \
     -d '{"images": ["/abs/path/page.png"]}' | jq -r .markdown_result
```

## Configuration

Everything is in `configs/salad.yaml` (a copy of `glmocr/config.yaml`;
salad-specific lines are marked `[salad]`). Restart with
`apps/start-salad.sh` after editing.

| Setting | Value | Why |
|---|---|---|
| `ocr_api.api_port`, `model` | 30000, `salad` | the SGLang container |
| `ocr_api.loop_abort` | true | stream responses and cut a generation off when its tail repeats for more than `loop_min_span_text` (300) chars, or `loop_min_span_table` (4096) for tables, whose rows of identical cells repeat legitimately |
| `max_workers` | 20 | matches salad's `--max-running-requests` |
| `page_loader.model_family` | `salad` | modified Qwen prompt, 28px-grid scaling, PNG, HTML answer conversion (`modified_qwen` works too) |
| `page_loader.task_prompt_mapping` | null | every region uses the model's own "OCR this image to HTML" prompt |
| `temperature` / `top_p` / `repetition_penalty` | 0 / 0.1 / 1.0 | the model's recommended defaults |
| `page_loader.region_padding` | 32 | white margin around each crop |
| `page_loader.region_min_canvas` | 512 | small crops are pasted at native scale onto a ≥512×512 white canvas; bare heading crops (~200×40 px) made salad invent text ("Journal of Applied Mathematics", `1.1.1.1.N` runs) |
| `page_loader.max_page_side` | 3500 | page images larger than this (long side) are shrunk on load, as PDFs already are |
| `page_loader.script_fix` | true | second pass for look-alike Indic scripts (below) |
| `page_loader.table_min_width` | 2048 | tables written with native Indic digits are re-read enlarged to this width (below) |
| `layout.backend` | `heron` | `pp_doclayout` switches back to PP-DocLayoutV3 |
| `layout.api_url` | `http://127.0.0.1:8600/detect` | heron service |
| `layout.threshold` | 0.5 | heron's scores; PP-DocLayoutV3 used 0.3 |
| `layout.overlap_iou_threshold` / `overlap_contain_threshold` | 0.6 / 0.8 | duplicate and nested box removal |
| `layout.label_task_mapping` | `header`, `footer` under `text` | page headers/footers (running titles, page numbers, "Signature of Contractor") are kept as plain text; upstream discards them. Move them back under `abandon` to drop them |
| `server.port` | 5012 | harness port |

Backend environment variables (`apps/backend/app/utils/config.py`):
`layout_ocr_url` (harness URL), `OCR_PAGE_CONCURRENCY` (pages of one document
sent at once, default 6) and `OCR_PAGE_TIMEOUT` (seconds per page, default 600).

## How it works

### Recognition with salad (`glmocr/adapters/modified_qwen.py`)

salad is a modified Qwen vision-language model (Qwen3.5 architecture) with an
extrapolated vision tower. It always answers with layout HTML
(`<div data-label=… data-bbox=…>`). The adapter:

- builds requests the way the model expects (prompt, `scale_to_fit`, PNG) and
  applies the padding/canvas rules above;
- converts the HTML back into GLM-OCR region content (Markdown for text,
  `<table>` HTML for tables, `$$…$$` for formulas), so the result formatter
  and UI work unchanged;
- merges a header-only `<table>` followed by a body `<table>` with the same
  column count (salad sometimes splits one table like this);
- emits table HTML on one line, because blank lines inside an HTML table end
  the HTML block in Markdown;
- detects runaway repetition for `loop_abort`.

### Layout with heron (`glmocr/layout/heron_detector.py`)

The heron service returns score-sorted boxes with docling labels and no
reading order. The detector:

- maps docling labels to the PP-DocLayout names the pipeline keys on
  (`section_header` → `paragraph_title`, `title` → `doc_title`, `picture` →
  `image`, `formula` → `display_formula`, `document_index` → `content`,
  `page_header`/`page_footer` → `header`/`footer`); checkbox marks are
  dropped;
- removes duplicates (IoU ≥ 0.6) and any box ≥ 80% inside a larger box,
  whatever the labels. Heron nests boxes often ("(Online Only)" inside its
  title, tables inside the list paragraph around them), and reading both
  crops duplicated the text: on a 34-page tender PDF, 19 nested boxes and 9
  doubled tables went to 0 with no text lost;
- orders regions by recursive XY-cut (columns first, then rows).

### Indic second pass (`glmocr/adapters/script_fix.py`)

Read one crop at a time, a short heading or list item has no context, and
salad confuses look-alike scripts: Kannada comes back as Telugu, Gujarati as
Devanagari, and scanned Malayalam as Sinhala. Long paragraphs on the same page
are read correctly. After each page:

1. **Script fix.** A text region whose script differs from the text around it
   (weighted by length and reading-order distance) is read again with a hint
   naming that script. The new reading is kept only if it comes back in that
   script, so genuine English or second-language text is left alone.
2. **Native-digit tables.** At native scale salad misreads small Bengali
   digits (৩ as ০). Tables that use native Indic digits (০–৯, ०–९, …) are read
   again from a crop enlarged to 2048 px (at most 2×). The new reading is kept
   only if its rows × columns are unchanged: enlarging can flip how an
   ambiguous header is structured, which happened on an English annual-report
   table. Tables with ASCII digits are never enlarged.

Only the regions that need it are re-read, so pages without Indic script cost
nothing extra.

### Backend and UI

- The backend sends up to 6 pages of a document to the harness at once (one
  page at a time left most of salad's batch slots idle): a 4-page PDF went
  from 55 s to 31 s with identical output.
- Image links in the merged Markdown are relative (`/api/v1/tasks/file?…`),
  so they load through the UI's `/api` proxy on any port.
- The UI text is in English.

## Speed

Time per page is dominated by the longest region: salad writes each region
out token by token, and a big table is one long generation (a 180-cell table
is ~2,500 tokens of HTML). Pages, and regions within a page, already run in
parallel, so the levers are per-request decode speed and avoiding waste:

| Change | Effect |
|---|---|
| FP8 weights (`sglang_salad_fp8`) | decode 57 → 91 tokens/s per request; benchmark 69.5 → 48.5 s with the same accuracy (Indic CER 0.86% vs 0.89%, cells 97.6% both) |
| Page images capped at 3500 px (`page_loader.max_page_side`) | an A0-size scanned bill rendered at 6281×8889 px cost ~2–3 s per page in encoding and layout |
| Pages in parallel (backend) | 4-page PDF 55 → 31 s |

On a 3-page bill with two full-page tables the slowest page went from 59 s
to 40 s, and a BPCL table page from 31.5 s to 22 s. N-gram speculative
decoding (good for repetitive table HTML) crashes in SGLang 0.5.10 for
salad's hybrid Qwen3.5 architecture (`NgramVerifyInput has no attribute
topk`); worth retrying on a newer SGLang.

## Benchmark

`bench/make_pages.py` renders 10 page designs in Chromium (correct Indic
shaping), each saved clean and scan-degraded (grayscale, skew, blur, noise,
JPEG), with exact ground truth taken from the data that rendered the page:

- English: financial statement with merged headers, dense table with rowspan
  groups, borderless table plus list;
- Hindi–English bilingual notice with a totals row, Hindi two-column page;
- Tamil, Bengali (native digits), Telugu and Kannada pages with tables;
- Malayalam + Gujarati text page.

`bench/score.py` runs every page and reports text CER (NFC-normalised,
Markdown stripped, tables excluded) and table cell accuracy (rowspan/colspan
expanded into a grid, cell by cell) plus table shape.

```bash
../.venv/bin/python bench/make_pages.py                  # writes bench/data/
../.venv/bin/python bench/score.py harness --tag mytry   # needs the stack running
../.venv/bin/python bench/score.py native  --tag full    # salad on whole pages, no harness
../.venv/bin/python bench/score.py --compare base_harness v4 mytry
```

Options: `--only kn_` filters pages, `--workers` sets concurrency, `--url`
overrides the endpoint. Outputs go to `bench/results/<tag>/` (Markdown per
page and `summary.json`).

Results (20 pages):

| Run | Text CER, English | Text CER, Indic | Table cells | Table shape | Time |
|---|---|---|---|---|---|
| salad on whole pages, no harness (`base_native`) | 0.6% | 10.7% | 96.9% | 100% | 290 s |
| harness before Indic tuning (`base_harness`) | 0.0% | 3.5% | 91.0% | 94% | 63 s |
| **harness now** (`v4`) | **0.0%** | **0.8%** | **97.6%** | **100%** | 70 s |

The real English pages in the BPCL annual report give identical output before
and after the Indic tuning.

For the record, on olmOCR-bench old_scans (2026-09-30, with PP-DocLayoutV3),
salad scored 48.1 through the harness and 50.4 on whole pages, against 48.5
and 51.3 for the base modified Qwen model without the vision boost. The
harness has not been re-benchmarked there since the switch to heron.

## Troubleshooting

- **Jobs fail or hang, GPU shows MinerU:** only one VLM fits on the GPU. If
  someone starts `sglang_mineru25_pro`, salad is stopped and the harness has
  no recognizer. `docker ps` shows which is up; `apps/start-salad.sh` switches
  back (it stops MinerU).
- **UI did not come up, `ENOSPC` in `apps/logs/frontend.log`:** the system
  ran out of inotify file watches (another IDE server held ~65k), not disk
  space. The limit is raised to 524288 in
  `/etc/sysctl.d/60-inotify-watches.conf`. If a step fails, the start script
  stops the services it already started.
- **Killing a service by hand:** use anchored patterns, e.g.
  `pgrep -f "^/home/nandha/GLM/.venv/bin/python -m glmocr.server"`; an
  unanchored `pkill -f` matches your own shell's command line.

## Known limits

- A single heading that mixes two Indic scripts (e.g. Malayalam / Gujarati)
  keeps the second script wrong; the script fix only corrects whole regions.
- Handwriting is read poorly (a model limit).
- "Label: value" rows (ವಿಷಯ: …) inside one layout box can come out as all
  labels, then all values.
- En dashes (–) come out as hyphens.
- A single very large table bounds page time: it is one long generation
  (~26 s for a full-page table) that parallelism cannot shorten.

## Files changed from upstream

Based on [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) (Apache-2.0,
see `LICENSE`).

New: `apps/start-salad.sh`, `configs/salad.yaml`,
`glmocr/adapters/{modified_qwen,script_fix}.py`, `glmocr/layout/heron_detector.py`,
`bench/{make_pages,score}.py`.

Modified:

- SDK: `glmocr/config.py` (new settings), `dataloader/page_loader.py`
  (salad requests, page-size cap), `ocr_client.py` (streaming loop abort),
  `pipeline/_workers.py` and `pipeline/pipeline.py` (adapter, heron backend,
  second pass), `postprocess/result_formatter.py` (headings, headers/footers).
- Backend: `core/steps/layout_ocr.py` (parallel pages),
  `core/steps/merge_results.py` (relative image links), `utils/config.py`;
  unused imports removed.
- Frontend: `vite.config.ts` (`BACKEND_URL`), UI strings translated to
  English.

Removed as unused here: `examples/` (except `examples/source/GLM-4.5V.pdf` and `seal.png`,
the unit tests' samples), `skills/` (Zhipu cloud API helpers),
`.github/` (upstream release pipelines), `agent.md`, `README_zh.md`, the
Docker deploy files and the old `apps/start-local.sh` / `start-docker.sh`,
README images. All are recoverable with `git checkout HEAD -- <path>`.
`resources/PingFang.ttf` stays (layout visualisation font) and
`apps/backend/README.md` stays (the backend's packaging needs it).

The SDK's unit tests (`glmocr/tests/test_unit.py`, 157 tests) pass.
