"""Layout detection through the docling-layout-heron-101 HTTP service.

The service (docker ``docling-layout-heron101``, ``POST /detect``) runs
docling's RT-DETRv2 layout model and returns ``{"regions": [{"label",
"score", "bbox"}]}`` with bboxes already normalized to 0-1000, sorted by
score. Unlike PP-DocLayoutV3 it gives no reading order and does no overlap
cleanup, so this detector:

* maps docling labels onto the PP-DocLayout label names the rest of the
  pipeline keys on (``label_task_mapping``, the result formatter, the UI);
* drops duplicate and nested boxes;
* orders regions with a recursive XY-cut (columns first, then rows).
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.logging import get_logger
from glmocr.utils.visualization_utils import draw_layout_boxes

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)

# docling label -> PP-DocLayout label (see label_task_mapping in the config)
HERON_TO_PP_LABEL = {
    "text": "text",
    "list_item": "text",
    "key_value_region": "text",
    "form": "text",
    "title": "doc_title",
    "section_header": "paragraph_title",
    "caption": "figure_title",
    "footnote": "vision_footnote",
    "formula": "display_formula",
    "table": "table",
    "document_index": "content",
    "code": "algorithm",
    "picture": "image",
    "page_header": "header",
    "page_footer": "footer",
    # Checkbox marks sit inside text/form regions; the recognizer reads them.
    "checkbox_selected": "abandon",
    "checkbox_unselected": "abandon",
}


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def remove_overlaps(
    regions: List[Dict], iou_threshold: float, contain_threshold: float
) -> List[Dict]:
    """Drop duplicates (high IoU, keep higher score) and nested boxes.

    A box mostly inside a larger box is dropped, whatever the labels: the
    larger crop already contains its text, so reading both duplicates it.
    Heron nests boxes often: "(Online Only)" inside its title, a table inside
    the list paragraph around it (salad reads the table as part of the
    paragraph crop).
    """
    regions = sorted(regions, key=lambda r: -r["score"])
    kept: List[Dict] = []
    for r in regions:
        b = r["bbox_2d"]
        a = _area(b) or 1.0
        duplicate = False
        for k in kept:
            inter = _intersection(b, k["bbox_2d"])
            if not inter:
                continue
            union = a + _area(k["bbox_2d"]) - inter
            if inter / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(r)

    result = []
    for r in kept:
        b = r["bbox_2d"]
        a = _area(b) or 1.0
        nested = False
        for o in kept:
            if o is r:
                continue
            ob = o["bbox_2d"]
            if _area(ob) <= a:
                continue
            if _intersection(b, ob) / a < contain_threshold:
                continue
            nested = True
            break
        if not nested:
            result.append(r)
    return result


def _gaps(intervals: List[Tuple[float, float]], min_gap: float) -> List[float]:
    """Cut positions where the projection of *intervals* has empty gaps."""
    cuts = []
    spans = sorted(intervals)
    reach = spans[0][1]
    for start, end in spans[1:]:
        if start - reach >= min_gap:
            cuts.append((reach + start) / 2)
        reach = max(reach, end)
    return cuts


def xy_cut_order(regions: List[Dict], min_gap: float = 4.0) -> List[Dict]:
    """Reading order by recursive XY-cut: split into columns, else rows.

    Boxes are shrunk slightly for the projections so small overlaps from the
    detector don't hide the whitespace between columns or paragraphs.
    """

    def shrink(b):
        dx = min(3.0, (b[2] - b[0]) / 4)
        dy = min(3.0, (b[3] - b[1]) / 4)
        return b[0] + dx, b[1] + dy, b[2] - dx, b[3] - dy

    def split(items, axis):
        lo, hi = (0, 2) if axis == "x" else (1, 3)
        cuts = _gaps([(s[lo], s[hi]) for s, _ in items], min_gap)
        if not cuts:
            return None
        groups: List[List] = [[] for _ in range(len(cuts) + 1)]
        for s, r in items:
            centre = (s[lo] + s[hi]) / 2
            groups[sum(centre > c for c in cuts)].append((s, r))
        return [g for g in groups if g]

    def order(items):
        if len(items) <= 1:
            return [r for _, r in items]
        groups = split(items, "x") or split(items, "y")
        if not groups or len(groups) == 1:
            # No clean cut: fall back to top-to-bottom, left-to-right.
            return [r for _, r in sorted(items, key=lambda it: (it[0][1], it[0][0]))]
        out = []
        for g in groups:
            out.extend(order(g))
        return out

    return order([(shrink(r["bbox_2d"]), r) for r in regions])


class HeronLayoutDetector(BaseLayoutDetector):
    """Layout detector backed by the docling-layout-heron-101 service."""

    def __init__(self, config: "LayoutConfig"):
        super().__init__(config)
        self.api_url = config.api_url or "http://127.0.0.1:8600/detect"
        self.health_url = self.api_url.rsplit("/", 1)[0] + "/health"
        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class or {}
        self.label_task_mapping = config.label_task_mapping or {}
        self.batch_size = config.batch_size
        self.workers = max(1, config.workers)
        self.timeout = config.api_timeout
        self.iou_threshold = config.overlap_iou_threshold
        self.contain_threshold = config.overlap_contain_threshold
        self._session: Optional[requests.Session] = None

    def start(self):
        self._session = requests.Session()
        try:
            info = self._session.get(self.health_url, timeout=10).json()
            logger.info("Heron layout service: %s", info)
        except Exception as e:
            raise RuntimeError(
                f"docling-layout-heron service not reachable at {self.health_url}: {e}"
            ) from e

    def stop(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def _task_type(self, label: str) -> Optional[str]:
        for task, labels in self.label_task_mapping.items():
            if isinstance(labels, list) and label in labels:
                return task
        return None

    def _detect(self, image: Image.Image) -> List[Dict]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        # Ask for the lowest threshold in use; per-class thresholds apply below.
        min_threshold = min([self.threshold, *self.threshold_by_class.values()])
        resp = (self._session or requests).post(
            self.api_url,
            files={"file": ("page.png", buf.getvalue(), "image/png")},
            data={"threshold": str(min_threshold)},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        regions = []
        for det in resp.json().get("regions", []):
            heron_label = det["label"]
            label = HERON_TO_PP_LABEL.get(heron_label, "text")
            threshold = self.threshold_by_class.get(
                heron_label, self.threshold_by_class.get(label, self.threshold)
            )
            if det["score"] < threshold:
                continue
            task_type = self._task_type(label)
            if label == "abandon" or task_type is None or task_type == "abandon":
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            regions.append(
                {
                    "label": label,
                    "score": float(det["score"]),
                    "bbox_2d": [x1, y1, x2, y2],
                    "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "task_type": task_type,
                }
            )

        regions = remove_overlaps(regions, self.iou_threshold, self.contain_threshold)
        regions = xy_cut_order(regions)
        for i, r in enumerate(regions):
            r["index"] = i
        return regions

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ):
        pil_images = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(pil_images) or 1)) as ex:
            all_results = list(ex.map(self._detect, pil_images))

        vis_images: Dict[int, Image.Image] = {}
        if save_visualization:
            for i, (im, regions) in enumerate(zip(pil_images, all_results)):
                w, h = im.size
                boxes = [
                    {
                        "label": r["label"],
                        "score": r["score"],
                        "coordinate": [
                            r["bbox_2d"][0] * w / 1000,
                            r["bbox_2d"][1] * h / 1000,
                            r["bbox_2d"][2] * w / 1000,
                            r["bbox_2d"][3] * h / 1000,
                        ],
                    }
                    for r in regions
                ]
                vis_images[global_start_idx + i] = draw_layout_boxes(
                    image=np.array(im), boxes=boxes, use_polygon=False
                )
        return all_results, vis_images
