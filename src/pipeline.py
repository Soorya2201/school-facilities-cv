"""Stage orchestration.

Stage A (CPU, internet)  : resolve campuses, fetch and cache imagery + OSM,
                           run the deterministic extractors, optionally call
                           the free-tier APIs. Weights are pre-staged here too.
Stage B (GPU, offline)   : open-weight model inference on the cached imagery.
Stage C (CPU, seconds)   : fuse, calibrate, write measurements.csv.

The split is not stylistic. CARC GPU nodes have restricted outbound network, so
anything that needs the internet -- imagery, Overpass, hosted APIs, HuggingFace
downloads -- has to happen in Stage A, and it would be wasteful to hold an A100
while waiting on a rate-limited HTTP call anyway.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import config, imagery, osm, prompts, tier0_cv, tier0_vector, tier2_api
from .fuse import fuse_school
from .geo import (bbox_from_center, bbox_of_ring, polygon_area_m2,
                  polygon_centroid, square_bbox)
from .resolve import Campus, load_schools, resolve_campus

# BDAPPV was trained at 400x400 on ~0.1 m/px imagery, so panels should be read
# from tight crops rather than the campus overview (~0.3 m/px).
#
# We fix the ground EXTENT and request the most pixels available, instead of
# demanding a target m/px: Esri enforces a per-location resolution ceiling and
# hard-fails below it, so a fixed 0.12 m/px request succeeds in Los Angeles and
# fails outright in rural Alabama. Fixing the extent means every school yields a
# usable crop; the resolution actually achieved is recorded per crop and flows
# into that school's confidence.
SOLAR_CROP_EXTENT_M = 120.0
CROP_PX = 1024
PERIM_CROP_EXTENT_M = 120.0


def _jsonl_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def _jsonl_read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Imagery plan
# --------------------------------------------------------------------------- #
def solar_crop_bboxes(cam: Campus, buildings: List[Dict[str, Any]],
                      max_crops: int = 3) -> List[List[float]]:
    """Crops centred on the largest on-campus buildings, at ~0.12 m/px.

    Rooftop panels occupy few pixels in a campus-wide view; feeding the
    segmenter its native scale matters more than feeding it more context.
    """
    half_m = SOLAR_CROP_EXTENT_M / 2.0
    centres: List[Tuple[float, Tuple[float, float]]] = []
    for b in buildings:
        ring = osm.ring_of(b)
        if len(ring) < 3:
            continue
        area = polygon_area_m2(ring)
        if area < 150:
            continue
        c = polygon_centroid(ring)
        from .geo import point_in_polygon
        if cam.ring and not point_in_polygon(c, cam.ring):
            continue
        centres.append((area, c))
    centres.sort(key=lambda t: -t[0])
    if not centres:
        c = cam.centroid
        return [square_bbox(bbox_from_center(c[1], c[0], half_m))]
    out = []
    for _, (lon, lat) in centres[:max_crops]:
        out.append(square_bbox(bbox_from_center(lat, lon, half_m)))
    return out


def perimeter_crop_bboxes(cam: Campus, n: int = 4,
                          half_m: float = PERIM_CROP_EXTENT_M / 2) -> List[List[float]]:
    """Crops at the cardinal extremes of the campus boundary, for fencing.

    Worth stating plainly: at 0.3-0.6 m/px chain-link wire is below the
    resolution limit. These crops let a model see gates, walls and shadow lines;
    they cannot make wire visible.
    """
    if not cam.ring:
        return []
    w, s, e, nn = bbox_of_ring(cam.ring)
    mid_lon, mid_lat = (w + e) / 2, (s + nn) / 2
    pts = [(mid_lon, nn), (e, mid_lat), (mid_lon, s), (w, mid_lat)][:n]
    return [square_bbox(bbox_from_center(lat, lon, half_m)) for lon, lat in pts]


# --------------------------------------------------------------------------- #
# Stage A
# --------------------------------------------------------------------------- #
def stage_a(offline: bool = False, limit: Optional[int] = None,
            use_api: bool = True, verbose: bool = True) -> Dict[str, Any]:
    rows = load_schools()
    if limit:
        rows = rows[:limit]

    campus_rows, vector_rows, cv_rows, api_rows, manifest = [], [], [], [], []
    releases = imagery.wayback_releases(offline=offline)

    for i, row in enumerate(rows, 1):
        t0 = time.time()
        sid = row["school_id"]
        cam = resolve_campus(row, offline=offline)
        campus_rows.append({**cam.to_dict(), "ring": cam.ring})

        ctx = None
        try:
            ctx = osm.fetch_campus_context(cam.lat, cam.lon,
                                           config.CAMPUS_SEARCH_RADIUS_M,
                                           offline=offline)
        except Exception:
            ctx = None
        buckets = osm.split(ctx["elements"]) if ctx else {
            "schools": [], "buildings": [], "barriers": [], "leisure": []}

        # --- Layer 0a: vector -------------------------------------------------
        vec = tier0_vector.extract_all(cam.ring, buckets)
        vector_rows.append({"school_id": sid, "fields": vec})

        # --- imagery ----------------------------------------------------------
        overview = imagery.fetch_esri(cam.bbox(), config.DEFAULT_TILE_PX,
                                      offline=offline)
        naip_meta = imagery.naip_metadata(cam.lat, cam.lon, offline=offline)
        solar_boxes = solar_crop_bboxes(cam, buckets["buildings"])
        solar_crops = [imagery.fetch_esri(b, CROP_PX, offline=offline)
                       for b in solar_boxes]
        perim = [imagery.fetch_esri(b, CROP_PX, offline=offline)
                 for b in perimeter_crop_bboxes(cam)]

        # Two dates for the temporal measurement, from Wayback's dated releases.
        t_new = t_old = None
        if releases:
            t_new = imagery.fetch_wayback(cam.bbox(), releases[0]["release"],
                                          releases[0]["date"],
                                          config.DEFAULT_TILE_PX, offline=offline)
            old = releases[-1] if len(releases) > 30 else releases[-1]
            t_old = imagery.fetch_wayback(cam.bbox(), old["release"], old["date"],
                                          config.DEFAULT_TILE_PX, offline=offline)

        manifest.append({
            "school_id": sid,
            "overview": overview.meta() if overview else None,
            "solar_crops": [c.meta() for c in solar_crops if c],
            "perimeter_crops": [c.meta() for c in perim if c],
            "wayback_new": t_new.meta() if t_new else None,
            "wayback_old": t_old.meta() if t_old else None,
            "naip": naip_meta,
        })

        # --- Layer 0b: classical CV ------------------------------------------
        if overview:
            import cv2
            bgr = cv2.cvtColor(np.array(overview.pil()), cv2.COLOR_RGB2BGR)
            cvres = tier0_cv.extract_all(bgr, cam.ring, overview.bbox,
                                         overview.meters_per_pixel)
            # Two-date measurement. Solar is the right attribute for this: it is
            # the one that genuinely changes on a five-year horizon. Both dates
            # go through the *same* detector so the comparison is like-for-like;
            # only the change between them is meaningful, not either absolute
            # value, because this detector is a weak single vote.
            temporal = {}
            for slot, img in (("t0", t_old), ("t1", t_new)):
                if img is None:
                    continue
                tb = cv2.cvtColor(np.array(img.pil()), cv2.COLOR_RGB2BGR)
                res = tier0_cv.extract_all(tb, cam.ring, img.bbox,
                                           img.meters_per_pixel)
                temporal[slot] = {
                    "date": img.date,
                    "solar_present": res["solar_present"]["value"],
                    "solar_area_m2": res["solar_area_m2"]["value"],
                    "m_per_px": round(img.meters_per_pixel, 4),
                }
            cv_rows.append({"school_id": sid, "fields": cvres,
                            "temporal": temporal})
        else:
            cv_rows.append({"school_id": sid, "fields": {}, "temporal": {},
                            "note": "no overview imagery"})

        # --- Layer 2: free-tier API judgment ----------------------------------
        if use_api and overview and tier2_api.available_providers():
            resp = tier2_api.call(
                prompts.survey_prompt(row["school_name"], row.get("level", ""),
                                      int(row.get("enrollment_2024") or 0)),
                overview.data, offline=offline)
            parsed = tier2_api.parse_json_response(resp.get("text", "")) if resp else None
            api_rows.append({"school_id": sid, "parsed": parsed,
                             "provider": (resp or {}).get("provider")})
        else:
            api_rows.append({"school_id": sid, "parsed": None, "provider": None})

        if verbose:
            date = (naip_meta.get("dates") or [{}])[-1].get("date")
            print(f"[{i:2d}/{len(rows)}] {row['school_name'][:32]:34s} "
                  f"{cam.method:20s} conf={cam.confidence:.2f} "
                  f"imgs={sum(1 for x in [overview,*solar_crops,*perim,t_new,t_old] if x):2d} "
                  f"naip={date} ({time.time()-t0:.0f}s)", flush=True)

    _jsonl_write(config.PREDS / "campus.jsonl", campus_rows)
    _jsonl_write(config.PREDS / "tier0_vector.jsonl", vector_rows)
    _jsonl_write(config.PREDS / "tier0_cv.jsonl", cv_rows)
    _jsonl_write(config.PREDS / "tier2_api.jsonl", api_rows)
    _jsonl_write(config.PREDS / "imagery_manifest.jsonl", manifest)
    return {"schools": len(rows), "api": tier2_api.status()}


# --------------------------------------------------------------------------- #
# Stage B (GPU)
# --------------------------------------------------------------------------- #
def gpu_stage(args) -> None:
    """Run one open-weight model family over the cached imagery."""
    import base64
    import cv2
    from . import tier1_gpu

    manifest = _jsonl_read(config.PREDS / "imagery_manifest.jsonl")
    campus = {r["school_id"]: r for r in _jsonl_read(config.PREDS / "campus.jsonl")}
    schools = {r["school_id"]: r for r in load_schools()}
    out: List[Dict[str, Any]] = []

    def _load(meta: Dict[str, Any]) -> Optional[np.ndarray]:
        if not meta:
            return None
        blob = imagery._img_cache.get_bytes(meta["cache_key"])
        if blob is None:
            return None
        import io
        from PIL import Image
        return np.array(Image.open(io.BytesIO(blob)).convert("RGB"))

    if args.task == "solar":
        seg = tier1_gpu.SolarSegmenter(device=args.device)
        for m in manifest:
            crops = []
            for c in m.get("solar_crops") or []:
                rgb = _load(c)
                if rgb is not None:
                    crops.append({"rgb": rgb, "m_per_px": c["m_per_px"],
                                  "bbox": c["bbox"]})
            if not crops:
                out.append({"school_id": m["school_id"], "fields": {},
                            "note": "no solar crops cached"})
                continue
            out.append({"school_id": m["school_id"],
                        "fields": tier1_gpu.solar_from_crops(seg, crops,
                                                             tta=args.tta)})
            print(f"solar {m['school_id']} done", flush=True)

    elif args.task == "owl":
        det = tier1_gpu.OwlDetector(device=args.device)
        for m in manifest:
            rgb = _load(m.get("overview"))
            if rgb is None:
                out.append({"school_id": m["school_id"], "fields": {}})
                continue
            dets = det.detect(rgb, tier1_gpu.OWLV2_QUERIES)
            mpp = (m.get("overview") or {}).get("m_per_px", 0.3)
            out.append({"school_id": m["school_id"],
                        "fields": tier1_gpu.owl_to_fields(dets, mpp)})
            print(f"owl {m['school_id']} n_det={len(dets)}", flush=True)

    elif args.task == "vlm":
        vlm = tier1_gpu.VllmVlm(args.vlm_url, args.vlm_model)
        for m in manifest:
            meta = m.get("overview")
            blob = imagery._img_cache.get_bytes(meta["cache_key"]) if meta else None
            if blob is None:
                out.append({"school_id": m["school_id"], "fields": {}})
                continue
            b64 = base64.b64encode(blob).decode()
            row = schools.get(m["school_id"], {})
            name = row.get("school_name", m["school_id"])
            fields: Dict[str, Any] = {}
            # Binary questions asked one at a time, because a yes/no logprob is
            # only meaningful when the next token really is the answer.
            for field in prompts.BINARY_QUESTIONS:
                try:
                    fields[field] = vlm.binary(b64, name, field)
                except Exception as exc:  # noqa: BLE001
                    fields[field] = {"value": None, "source": "vlm",
                                     "error": str(exc)}
            try:
                survey = vlm.survey(b64, name, row.get("level", ""),
                                    int(row.get("enrollment_2024") or 0))
                for k, v in (survey.get("parsed") or {}).items():
                    if k in config.FIELDS and k not in fields:
                        fields[k] = {"value": v, "source": "vlm_survey"}
            except Exception:
                pass
            out.append({"school_id": m["school_id"], "fields": fields})
            print(f"vlm {m['school_id']} done", flush=True)

    path = Path(args.out) if args.out else config.PREDS / f"tier1_{args.task}.jsonl"
    _jsonl_write(path, out)
    print(f"wrote {path} ({len(out)} rows)")


# --------------------------------------------------------------------------- #
# Stage C
# --------------------------------------------------------------------------- #
SOURCE_FILES = {
    "osm_vector": "tier0_vector.jsonl",
    "cv": "tier0_cv.jsonl",
    "api": "tier2_api.jsonl",
    "bdappv": "tier1_solar.jsonl",
    "owlv2": "tier1_owl.jsonl",
    "vlm": "tier1_vlm.jsonl",
}


def _load_sources() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """source -> school_id -> {field: signal}."""
    loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source, fname in SOURCE_FILES.items():
        rows = _jsonl_read(config.PREDS / fname)
        if not rows:
            continue
        by_school: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if source == "api":
                parsed = r.get("parsed") or {}
                by_school[r["school_id"]] = {
                    k: {"value": v, "source": f"api:{r.get('provider')}"}
                    for k, v in parsed.items() if k in config.FIELDS
                }
            else:
                by_school[r["school_id"]] = r.get("fields") or {}
        loaded[source] = by_school
    return loaded


def stage_c(truth_path=None) -> Dict[str, Any]:
    """Fuse every available source, then calibrate against the hand labels."""
    from .calibrate import evaluate, is_correct
    from .labels import load_truth

    campus = {r["school_id"]: r for r in _jsonl_read(config.PREDS / "campus.jsonl")}
    manifest = {r["school_id"]: r for r in
                _jsonl_read(config.PREDS / "imagery_manifest.jsonl")}
    sources = _load_sources()
    truth = load_truth(truth_path)
    schools = load_schools()
    temporal = {r["school_id"]: (r.get("temporal") or {})
                for r in _jsonl_read(config.PREDS / "tier0_cv.jsonl")}

    fused_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    calib_labels: List[bool] = []

    for row in schools:
        sid = row["school_id"]
        cam = campus.get(sid, {})
        cconf = float(cam.get("campus_confidence") or 0.3)
        per_source = {name: by_school.get(sid, {}) for name, by_school in sources.items()}
        man = manifest.get(sid, {})
        mpp = float(((man.get("overview") or {}).get("m_per_px")) or 0.3)
        fused = fuse_school(per_source, cconf, mpp)
        temp = temporal.get(sid, {})
        fused_rows.append({
            "school_id": sid, "school_name": row["school_name"],
            "campus": cam, "manifest": man, "fused": fused,
            **{f"{k}_{slot}": temp.get(slot, {}).get(k)
               for slot in ("t0", "t1")
               for k in ("solar_present", "solar_area_m2")},
        })
        t = truth.get(sid, {})
        for field, f in fused.items():
            ok = is_correct(field, f["value"], t.get(field))
            if ok is None:
                continue
            calib_rows.append({"school_id": sid, "field": field,
                               "features": f["features"]})
            calib_labels.append(ok)

    report = evaluate(calib_rows, calib_labels) if calib_rows else {
        "error": "no hand labels found -- run `python -m src.run label` first"}
    return {"fused": fused_rows, "report": report,
            "n_labeled_schools": len(truth)}
