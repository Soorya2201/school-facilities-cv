"""Side-by-side comparison of predictions against the hand labels, with crops.

`metrics.json` says the pipeline is 59% accurate. It cannot say *which* pixels
produced any single answer. This module writes `out/results.csv`, where every
field carries its truth value, the prediction, the calibrated confidence, and a
path to a cropped image of the object that drove the detection.

Only two layers localise anything: OSM geometry (lon/lat rings, projected here
into the overview tile) and the classical CV detectors (pixel boxes already in
overview space). Fencing is a perimeter, not an object, so its crop cell is
always "not found" -- which is an honest statement about the evidence, not a gap
to be papered over.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from . import calibrate, config, labels

CROPS = config.OUT / "crops"
RESULTS_CSV = config.OUT / "results.csv"
RESULTS_XLSX = config.OUT / "results.xlsx"
NOT_FOUND = "not found"
NO_PREDICTION = "no_prediction"   # the pipeline abstained, rather than answered wrongly
THUMB_PX = 96            # embedded thumbnail size in the spreadsheet

# Which source's evidence localises each field. OSM is preferred where it exists
# because it is the layer that produced the count; CV is the only source for
# solar, and a fallback for pool and track.
OSM_EVIDENCE = {
    "portable_count": ("portable_count", "candidates"),
    "field_count": ("field_count", None),
    "court_count": ("court_count", None),
    "track_present": ("track_present", None),
    "pool_present": ("pool_present", None),
}
CV_EVIDENCE = {
    "solar_present": ("solar_present", "candidates"),
    "solar_area_m2": ("solar_area_m2", "candidates"),
    "pool_present": ("pool_present", "candidates"),
    "track_present": ("track_present", "best"),
}

# Below this calibrated confidence a prediction is not trusted enough to assert.
# The coverage-accuracy curve is what justifies the number: dropping the least
# confident fifth of predictions takes accuracy from 74% to 93%, so the pipeline
# genuinely knows which of its answers are worthless.
LOW_CONFIDENCE = 0.40
# What a rejected prediction falls back to. Asserting the negative is a policy
# choice, not a measurement -- it is scored against the labels like any other
# answer, and it is shaded in the spreadsheet so it never reads as a detection.
REJECTED_VALUE = {
    "solar_present": "False", "track_present": "False", "pool_present": "False",
    "portable_count": "0", "field_count": "0", "court_count": "0",
    "solar_area_m2": "0.0", "fencing_extent": "none", "fencing_type": "",
}

CROP_PAD = 0.30          # fraction of the object's size added on every side
MIN_CROP_PX = 96         # so a 20 m portable is not a 12-pixel smudge
OUT_MIN_PX = 256         # upscale small crops to something a human can judge


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def _mercator_y(lat: float) -> float:
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))


def geo_to_px(lon: float, lat: float, tile: Dict[str, Any]) -> Tuple[float, float]:
    """Project lon/lat into pixel coordinates of a cached tile.

    The tile is requested with a lon/lat bbox but rendered in EPSG:3857, so x is
    linear in longitude while y is linear in *Mercator* y, not in latitude.
    Treating y as linear in latitude puts crops progressively off-centre towards
    the edges of the image.
    """
    west, south, east, north = tile["bbox"]
    x = (lon - west) / (east - west) * tile["width"]
    y_n, y_s = _mercator_y(north), _mercator_y(south)
    y = (y_n - _mercator_y(lat)) / (y_n - y_s) * tile["height"]
    return x, y


def _box_from_bounds(bounds: Sequence[float], tile: Dict[str, Any]) -> List[float]:
    west, south, east, north = bounds
    x0, y0 = geo_to_px(west, north, tile)
    x1, y1 = geo_to_px(east, south, tile)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _pad_and_clamp(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(bw, bh) * (1.0 + 2 * CROP_PAD)
    side = max(side, float(MIN_CROP_PX))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x0 = max(0.0, min(x0, w - side))
    y0 = max(0.0, min(y0, h - side))
    side = min(side, float(min(w, h)))
    return int(x0), int(y0), int(x0 + side), int(y0 + side)


# --------------------------------------------------------------------------- #
# Evidence collection
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["school_id"]] = r
    return out


def _entries(signal: Optional[Dict[str, Any]], key: Optional[str]) -> List[Dict[str, Any]]:
    """Evidence entries for one field, whether the layer stores a list or a dict."""
    if not signal:
        return []
    ev = signal.get("evidence")
    if key is None:
        return [e for e in ev if isinstance(e, dict)] if isinstance(ev, list) else []
    if not isinstance(ev, dict):
        return []
    item = ev.get(key)
    if isinstance(item, list):
        return [e for e in item if isinstance(e, dict)]
    return [item] if isinstance(item, dict) else []


def objects_for(field: str, vector: Dict[str, Any], cv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Localisable detections for one field, preferring the OSM geometry."""
    if field in OSM_EVIDENCE:
        name, key = OSM_EVIDENCE[field]
        found = [e for e in _entries((vector.get("fields") or {}).get(name), key)
                 if e.get("bounds") or e.get("centroid")]
        if found:
            return found
    if field in CV_EVIDENCE:
        name, key = CV_EVIDENCE[field]
        return [e for e in _entries((cv.get("fields") or {}).get(name), key)
                if e.get("bbox_px")]
    return []


def crop_object(obj: Dict[str, Any], tile: Dict[str, Any], img: Image.Image,
                school_id: str, field: str, index: int) -> Optional[str]:
    """Write one crop and return its repo-relative path, or None if unlocatable."""
    if obj.get("bbox_px"):
        box = [float(v) for v in obj["bbox_px"]]
    elif obj.get("bounds"):
        box = _box_from_bounds(obj["bounds"], tile)
    elif obj.get("centroid"):
        x, y = geo_to_px(obj["centroid"][0], obj["centroid"][1], tile)
        box = [x, y, x, y]
    else:
        return None

    x0, y0, x1, y1 = _pad_and_clamp(box, img.width, img.height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = img.crop((x0, y0, x1, y1))
    if crop.width < OUT_MIN_PX:
        crop = crop.resize((OUT_MIN_PX, OUT_MIN_PX), Image.LANCZOS)
    CROPS.mkdir(parents=True, exist_ok=True)
    name = f"{school_id}_{field}_{index + 1}.jpg"
    crop.convert("RGB").save(CROPS / name, quality=88)
    return f"out/crops/{name}"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _columns() -> List[str]:
    cols = ["school_id", "school_name"]
    for f in config.FIELDS:
        cols += [f, f"{f}_pred", f"{f}_pred_confidence", f"{f}_crop", f"{f}_match"]
    # Last, after every field: the per-school roll-up reads naturally as a
    # conclusion rather than a header.
    return cols + ["score", "summary"]


def summarise(row: Dict[str, str]) -> str:
    """What the pipeline says this campus has, one facility per line.

    Reads the predictions, not the labels: this is the pipeline's account of the
    school. Anything absent or zero is left out rather than listed as a nil, so
    the cell is a description and not a checklist of negatives.
    """
    def val(field: str) -> Any:
        return labels.coerce_field(field, row.get(f"{field}_pred", ""))

    parts: List[str] = []
    if val("solar_present"):
        area = val("solar_area_m2")
        parts.append(f"solar - {area:.0f} m2" if area else "solar")
    for field, noun in (("portable_count", "portables"),
                        ("field_count", "fields"),
                        ("court_count", "courts")):
        n = val(field)
        if n:
            parts.append(f"{noun} - {n}")
    extent = val("fencing_extent")
    if extent and extent != "none":
        kind = val("fencing_type")
        parts.append(f"fencing - {extent}" + (f" ({kind})" if kind else ""))
    if val("track_present"):
        parts.append("track")
    if val("pool_present"):
        parts.append("pool")
    return "\n".join(parts) if parts else "nothing detected"


def _blank(v: Any) -> str:
    return "" if v is None else str(v)


def write_xlsx(rows: List[Dict[str, str]], cols: List[str],
               path: Path = RESULTS_XLSX) -> Optional[Path]:
    """The same table, with the crops themselves embedded rather than paths.

    A CSV cell is text and cannot hold an image, so the reviewable version of
    this report is a spreadsheet. Optional on purpose: if openpyxl is missing the
    CSV still writes, and the offline reproduction path stays dependency-light.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    except ImportError:
        return None

    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    # Excel's own gridlines box in every empty cell, which reads as structure
    # that isn't there. With them off, the only lines on the sheet are the ones
    # that mean something: a cell holding a value, and the break between schools.
    ws.sheet_view.showGridLines = False
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "C2"

    crop_cols = {f"{f}_crop" for f in config.FIELDS}
    green = PatternFill("solid", fgColor="D6EFD6")
    red = PatternFill("solid", fgColor="F8D7D7")
    amber = PatternFill("solid", fgColor="FBEBD0")   # abstained, not answered wrongly
    yellow = PatternFill("solid", fgColor="FFF2A8")  # predicted, then rejected as unsure
    match_fill = {"true": green, "false": red, NO_PREDICTION: amber}

    hair = Side(style="thin", color="C4C4C4")
    heavy = Side(style="medium", color="7F7F7F")
    box = Border(left=hair, right=hair, top=hair, bottom=hair)
    for c in ws[1]:
        c.border = Border(bottom=heavy)

    for r_i, row in enumerate(rows, start=2):
        has_image = False
        filled: List[Any] = []
        low_conf = row.get("__rejected__") or set()
        for c_i, col in enumerate(cols, start=1):
            value = row.get(col, "")
            cell = ws.cell(row=r_i, column=c_i)
            if col in crop_cols and value.endswith(".jpg"):
                p = config.ROOT / value
                if p.exists():
                    pic = XLImage(str(p))
                    pic.width = pic.height = THUMB_PX
                    pic.anchor = cell.coordinate
                    ws.add_image(pic)
                    has_image = True
                    filled.append(cell)
                    cell.border = box
                    continue
                cell.value = value
            else:
                cell.value = value
            if value:
                # Only cells that carry something get an outline. The blanks in a
                # continuation row are spacing, not data, so they stay invisible.
                cell.border = box
                filled.append(cell)
            if col.endswith("_match") and value in match_fill:
                cell.fill = match_fill[value]
            # A rejected guess is yellow across its prediction and confidence, so
            # it is visibly a suppressed detection rather than a confident "no".
            if col.endswith(("_pred", "_pred_confidence")):
                base = col.rsplit("_pred", 1)[0]
                if base in low_conf:
                    cell.fill = yellow
                    if not cell.border.left.style:
                        cell.border = box
                        filled.append(cell)
            if col == "summary":
                cell.alignment = Alignment(vertical="center", horizontal="left",
                                           wrap_text=True)
            else:
                cell.alignment = Alignment(vertical="center", horizontal="center")
        # Tall enough for the thumbnails, or for however many lines the summary
        # runs to, whichever needs more room.
        lines = row.get("summary", "").count("\n") + 1 if row.get("summary") else 0
        need = max(THUMB_PX * 0.78 if has_image else 0.0, lines * 13.0)
        if need:
            ws.row_dimensions[r_i].height = need
        # A school starts a new block: rule off the row above it, full width, so
        # the 25 schools read as 25 groups however many crop rows each has.
        if row.get("school_id") and r_i > 2:
            for c_i in range(1, len(cols) + 1):
                prev = ws.cell(row=r_i - 1, column=c_i)
                prev.border = Border(left=prev.border.left, right=prev.border.right,
                                     top=prev.border.top, bottom=heavy)

    for c_i, col in enumerate(cols, start=1):
        letter = ws.cell(row=1, column=c_i).column_letter
        if col in crop_cols:
            ws.column_dimensions[letter].width = THUMB_PX / 7.0
        elif col == "school_name":
            ws.column_dimensions[letter].width = 32
        elif col == "summary":
            ws.column_dimensions[letter].width = 22
        elif col == "score":
            ws.column_dimensions[letter].width = 8
        else:
            ws.column_dimensions[letter].width = max(11, min(len(col) + 2, 24))

    wb.save(path)
    return path


def build() -> Dict[str, Any]:
    truth = labels.load_truth()
    vector = _read_jsonl(config.PREDS / "tier0_vector.jsonl")
    cv = _read_jsonl(config.PREDS / "tier0_cv.jsonl")
    manifest = _read_jsonl(config.PREDS / "imagery_manifest.jsonl")

    with (config.OUT / "measurements.csv").open() as fh:
        measurements = list(csv.DictReader(fh))

    rows: List[Dict[str, str]] = []
    scored: Dict[str, Dict[str, int]] = {f: {"n": 0, "k": 0, "abstained": 0}
                                        for f in config.FIELDS}
    crops_made = 0
    crops_missing = 0
    rejected_total = 0

    for m in measurements:
        sid = m["school_id"]
        tile = (manifest.get(sid) or {}).get("overview")
        img = None
        if tile:
            p = config.IMAGERY_CACHE / f"{tile['cache_key']}.jpg"
            if p.exists():
                img = Image.open(p)

        t = truth.get(sid, {})
        crops: Dict[str, List[str]] = {}
        first: Dict[str, str] = {"school_id": sid, "school_name": m["school_name"]}

        rejected: set = set()
        for field in config.FIELDS:
            pred = m.get(field, "")
            conf_raw = m.get(f"{field}_confidence", "")
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = None
            # Too unsure to assert: fall back to the negative and mark it, so the
            # sheet shows a rejected guess rather than a confident absence.
            if pred != "" and conf is not None and conf < LOW_CONFIDENCE:
                pred = REJECTED_VALUE[field]
                rejected.add(field)

            first[field] = _blank(t.get(field))
            first[f"{field}_pred"] = pred
            first[f"{field}_pred_confidence"] = conf_raw

            coerced = labels.coerce_field(field, pred)
            ok = calibrate.is_correct(field, coerced, t.get(field))
            # "no_prediction" is not a third outcome in the scoring -- it still
            # counts against the pipeline, because a missing measurement is not a
            # correct one. It is called out so a wrong count can be told apart
            # from a campus where OSM simply had nothing mapped.
            if ok is None:
                first[f"{field}_match"] = ""
            elif coerced is None and field not in calibrate.BLANK_MEANS_NONE:
                first[f"{field}_match"] = NO_PREDICTION
            else:
                first[f"{field}_match"] = str(bool(ok)).lower()
            if ok is not None:
                scored[field]["n"] += 1
                scored[field]["k"] += int(ok)
                scored[field]["abstained"] += int(
                    coerced is None and field not in calibrate.BLANK_MEANS_NONE)

            paths: List[str] = []
            if img is not None:
                for i, obj in enumerate(objects_for(field, vector.get(sid, {}),
                                                    cv.get(sid, {}))):
                    got = crop_object(obj, tile, img, sid, field, i)
                    if got:
                        paths.append(got)
            crops[field] = paths
            crops_made += len(paths)
            if not paths:
                crops_missing += 1
            first[f"{field}_crop"] = paths[0] if paths else NOT_FOUND

        n_right = sum(1 for f in config.FIELDS if first[f"{f}_match"] == "true")
        first["score"] = f"{n_right}/{len(config.FIELDS)}"
        first["summary"] = summarise(first)
        first["__rejected__"] = rejected     # consumed by the spreadsheet writer
        rejected_total += len(rejected)
        rows.append(first)
        # Extra detections become extra rows carrying only their crop path, so
        # one row still reads as one school.
        for i in range(1, max((len(v) for v in crops.values()), default=1)):
            extra = {c: "" for c in _columns()}
            for field, paths in crops.items():
                if i < len(paths):
                    extra[f"{field}_crop"] = paths[i]
            if any(extra[c] for c in extra):
                rows.append(extra)

    cols = _columns()
    with RESULTS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    xlsx = write_xlsx(rows, cols)

    return {"rows": len(rows), "schools": len(measurements), "xlsx": xlsx,
            "crops": crops_made, "no_crop_cells": crops_missing,
            "rejected": rejected_total, "per_field": scored}


def main() -> Dict[str, Any]:
    info = build()
    print(f"wrote {RESULTS_CSV}")
    if info["xlsx"]:
        print(f"wrote {info['xlsx']}  (crops embedded in the cells)")
    else:
        print("     openpyxl not installed -- skipped results.xlsx")
    print(f"     {info['rows']} rows for {info['schools']} schools, "
          f"{info['crops']} crops in {CROPS}")
    print()
    print(f"{'field':<18} {'n':>4} {'correct':>8} {'accuracy':>9} {'no pred':>9}")
    tot_n = tot_k = tot_a = 0
    for field, d in info["per_field"].items():
        n, k, a = d["n"], d["k"], d["abstained"]
        tot_n += n
        tot_k += k
        tot_a += a
        acc = f"{k / n:.1%}" if n else "--"
        print(f"{field:<18} {n:>4} {k:>8} {acc:>9} {a or '':>9}")
    overall = f"{tot_k / tot_n:.1%}" if tot_n else "--"
    print(f"{'OVERALL':<18} {tot_n:>4} {tot_k:>8} {overall:>9} {tot_a:>9}")
    if info["rejected"]:
        print(f"\n{info['rejected']} predictions fell below confidence "
              f"{LOW_CONFIDENCE} and were rejected to the negative\n"
              f"(shaded yellow in the spreadsheet). They are scored as answers, "
              f"not excused.")
    if tot_a:
        wrong = tot_n - tot_k
        print(f"\n{tot_a} of the {wrong} not-correct cells are abstentions: the pipeline "
              f"produced no value at all\n(OSM had nothing mapped), rather than a wrong one. "
              f"They count against accuracy, since a\nmissing measurement is not a correct one.")
    return info
