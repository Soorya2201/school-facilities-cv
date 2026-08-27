"""Layer 0b: classical computer vision on the imagery.

These detectors are deliberately simple and deterministic. They are not meant to
beat a neural network; they are meant to be an *independent* opinion whose
failure modes are uncorrelated with a model's. Agreement between a colour-and-
shape heuristic and a VLM is far more informative than a VLM agreeing with
itself twice.

Everything returns areas in square metres, converted from pixels using the
image's own metres-per-pixel. No model is ever asked for a physical quantity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .geo import Ring, validate_scale_bar

# Pool water in RGB aerial imagery is a narrow, distinctive band: strongly
# saturated cyan-blue with high value. Very little else on a campus looks
# like this, which is why pools are the most reliable aerial detection.
# The value floor matters more than the hue range: dark navy play surfaces and
# blue shade sails share the hue of pool water but are much darker. Without a
# high V floor a rubber playground reads as a swimming pool.
POOL_HSV_LO = np.array([88, 70, 110], dtype=np.uint8)
POOL_HSV_HI = np.array([118, 255, 255], dtype=np.uint8)

# Running-track surfaces are typically red-brown rubber or dark asphalt.
TRACK_HSV_LO = np.array([0, 60, 50], dtype=np.uint8)
TRACK_HSV_HI = np.array([20, 255, 220], dtype=np.uint8)

# Photovoltaic panels: dark, low-saturation, slightly blue.
SOLAR_V_MAX = 105
SOLAR_S_MAX = 110
SOLAR_MIN_AREA_M2 = 12.0


def campus_mask(img_shape: Tuple[int, int], ring: Ring,
                bbox: Sequence[float]) -> np.ndarray:
    """Binary mask of the campus polygon in image pixel space."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not ring:
        mask[:] = 255
        return mask
    west, south, east, north = bbox
    pts = []
    for lon, lat in ring:
        x = (lon - west) / (east - west) * w
        y = (north - lat) / (north - south) * h
        pts.append([x, y])
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def _components(mask: np.ndarray, mpp: float, min_area_m2: float,
                max_area_m2: float) -> List[Dict[str, Any]]:
    """Connected components of a binary mask, measured in real-world units."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    px_area = mpp * mpp
    out = []
    for i in range(1, n):
        area_m2 = stats[i, cv2.CC_STAT_AREA] * px_area
        if not (min_area_m2 <= area_m2 <= max_area_m2):
            continue
        w_m = stats[i, cv2.CC_STAT_WIDTH] * mpp
        h_m = stats[i, cv2.CC_STAT_HEIGHT] * mpp
        fill = stats[i, cv2.CC_STAT_AREA] / max(
            1.0, stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT])
        out.append({
            "area_m2": round(area_m2, 1),
            "w_m": round(w_m, 1), "h_m": round(h_m, 1),
            "fill": round(float(fill), 3),
            "cx": float(cents[i][0]), "cy": float(cents[i][1]),
        })
    out.sort(key=lambda d: -d["area_m2"])
    return out


def detect_pool(bgr: np.ndarray, mask: np.ndarray, mpp: float) -> Dict[str, Any]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, POOL_HSV_LO, POOL_HSV_HI)
    blue = cv2.bitwise_and(blue, mask)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    comps = _components(blue, mpp, config.POOL_AREA_MIN_M2, config.POOL_AREA_MAX_M2)
    # Pools are convex, well filled, and not extremely elongated.
    pools = [c for c in comps
             if c["fill"] >= 0.55
             and max(c["w_m"], c["h_m"]) / max(1e-6, min(c["w_m"], c["h_m"])) <= 4.0]
    total_blue_m2 = sum(c["area_m2"] for c in comps)
    return {
        "value": bool(pools), "source": "cv_color",
        "evidence": {"candidates": pools[:5], "n_blue_blobs": len(comps),
                     "blue_area_m2": round(total_blue_m2, 1)},
        "score": min(1.0, len(pools) * 0.6) if pools else 0.0,
    }


def detect_track(bgr: np.ndarray, mask: np.ndarray, mpp: float) -> Dict[str, Any]:
    """Find a running track as an *annulus*, not merely a coloured blob.

    Baseball infield dirt has the same red-brown hue as track rubber, so colour
    alone produces confident false positives. What distinguishes a track is its
    geometry: a 400 m track is a ring roughly 176 x 92 m enclosing an infield,
    whose band is only ~6-12 m wide. So we require a contour with a hole, and we
    require the *band* (outer minus hole) to be a small fraction of the outer
    area. A solid dirt diamond fails that test.

    The 400 m perimeter also serves as an in-image scale bar: if the geometry
    says this loop is 800 m around, the metres-per-pixel conversion is wrong.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    surf = cv2.inRange(hsv, TRACK_HSV_LO, TRACK_HSV_HI)
    surf = cv2.bitwise_and(surf, mask)
    surf = cv2.morphologyEx(surf, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, hierarchy = cv2.findContours(surf, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Dict[str, Any]] = None
    rejected: List[str] = []
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            outer_m2 = cv2.contourArea(cnt) * mpp * mpp
            (_, _), (w, h), _ = cv2.minAreaRect(cnt)
            long_m, short_m = max(w, h) * mpp, min(w, h) * mpp
            if short_m <= 0:
                continue
            # Hard gates from the physical dimensions of a standard track.
            if not (8000.0 <= outer_m2 <= 30000.0):
                rejected.append(f"area {outer_m2:.0f} m2")
                continue
            if not (130.0 <= long_m <= 240.0 and 70.0 <= short_m <= 140.0):
                rejected.append(f"dims {long_m:.0f}x{short_m:.0f} m")
                continue

            child = hierarchy[0][i][2]
            hole_m2 = 0.0
            while child != -1:
                hole_m2 += cv2.contourArea(contours[child]) * mpp * mpp
                child = hierarchy[0][child][0]
            if hole_m2 <= 0:
                rejected.append("no infield")
                continue
            band_ratio = (outer_m2 - hole_m2) / outer_m2
            if not (0.10 <= band_ratio <= 0.60):
                rejected.append(f"band {band_ratio:.2f}")
                continue

            perim_m = cv2.arcLength(cnt, True) * mpp
            aspect = long_m / short_m
            score = 0.0
            score += 0.30 * max(0.0, 1.0 - abs(perim_m - config.TRACK_PERIMETER_M) / 200.0)
            score += 0.35 if 1.4 <= aspect <= 2.6 else 0.0
            score += 0.35 if 0.15 <= band_ratio <= 0.50 else 0.0
            cand = {"perimeter_m": round(perim_m), "outer_area_m2": round(outer_m2),
                    "infield_m2": round(hole_m2), "band_ratio": round(band_ratio, 3),
                    "long_m": round(long_m), "short_m": round(short_m),
                    "aspect": round(aspect, 2), "score": round(score, 3)}
            if best is None or score > best["score"]:
                best = cand

    scale_ok = None
    if best:
        ok, msg = validate_scale_bar(best["perimeter_m"], config.TRACK_PERIMETER_M,
                                     0.30, "track")
        scale_ok = {"ok": ok, "detail": msg}
    return {
        "value": bool(best and best["score"] >= 0.60), "source": "cv_shape",
        "evidence": {"best": best, "rejected": rejected[:5], "scale_check": scale_ok},
        "score": best["score"] if best else 0.0,
    }


def detect_solar(bgr: np.ndarray, mask: np.ndarray, mpp: float) -> Dict[str, Any]:
    """Dark, low-saturation, strongly rectilinear regions.

    This is the weakest of the three heuristics -- dark roofs and shadows share
    the colour signature -- so it is used as one vote, never alone. The area it
    returns comes from pixel counts, never from a model's estimate.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    dark = ((v < SOLAR_V_MAX) & (s < SOLAR_S_MAX)).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels, total = [], 0.0
    for cnt in contours:
        area_m2 = cv2.contourArea(cnt) * mpp * mpp
        if area_m2 < SOLAR_MIN_AREA_M2:
            continue
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        rect_area_m2 = rw * rh * mpp * mpp
        if rect_area_m2 <= 0:
            continue
        rectangularity = area_m2 / rect_area_m2
        long_m, short_m = max(rw, rh) * mpp, min(rw, rh) * mpp
        if short_m <= 0:
            continue
        # Panel arrays are strongly rectangular; tree shadows are not.
        if rectangularity < 0.70:
            continue
        # Photovoltaic panels are bluish; asphalt, tar roofs and shadow are
        # neutral or warm. Requiring B > R removes most dark-surface hits.
        comp_mask = np.zeros(dark.shape, dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
        mean_b, mean_g, mean_r = cv2.mean(bgr, mask=comp_mask)[:3]
        if mean_b <= mean_r + 4.0:
            continue
        panels.append({"area_m2": round(area_m2, 1),
                       "rect": round(float(rectangularity), 2),
                       "blueness": round(float(mean_b - mean_r), 1),
                       "dims_m": [round(long_m, 1), round(short_m, 1)]})
        total += area_m2
    panels.sort(key=lambda d: -d["area_m2"])
    return {
        "value": bool(panels), "source": "cv_color",
        "evidence": {"candidates": panels[:6], "n": len(panels)},
        "area_m2": round(total, 1),
        "score": min(1.0, total / 300.0) if panels else 0.0,
    }


def extract_all(bgr: np.ndarray, ring: Ring, bbox: Sequence[float],
                mpp: float) -> Dict[str, Dict[str, Any]]:
    mask = campus_mask(bgr.shape, ring, bbox)
    solar = detect_solar(bgr, mask, mpp)
    return {
        "pool_present": detect_pool(bgr, mask, mpp),
        "track_present": detect_track(bgr, mask, mpp),
        "solar_present": {k: v for k, v in solar.items() if k != "area_m2"},
        "solar_area_m2": {"value": solar["area_m2"], "source": "cv_color",
                          "evidence": solar["evidence"], "score": solar["score"]},
    }
