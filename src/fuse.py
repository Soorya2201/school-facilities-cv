"""Combine independent signals into one value per field, plus the features the
calibration step will turn into a confidence.

The design commitment: confidence is never a number a model asserts. It is a
learned function of *how much independent evidence agrees*, fitted against blind
hand labels. So this module's real output is not the fused value -- it is the
feature vector that explains how much to trust that value.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from . import config

# Prior weights on each source, used only to break ties when forming a value.
# They deliberately do NOT set the confidence -- calibration does that from data.
SOURCE_WEIGHT = {
    "osm_vector": 1.0,
    "cv_color": 0.6,
    "cv_shape": 0.7,
    "owlv2": 0.9,
    "bdappv_deeplab_google": 1.2,
    "bdappv_deeplab_ign": 1.0,
    "vlm": 1.1,
    "api": 1.1,
}

CATEGORICAL_FIELDS = {"fencing_extent", "fencing_type"}


def _weight(source: str) -> float:
    for prefix, w in SOURCE_WEIGHT.items():
        if source.startswith(prefix):
            return w
    return 0.8


def _usable(sig: Dict[str, Any]) -> bool:
    return sig is not None and sig.get("value") is not None


def fuse_field(field: str, signals: List[Dict[str, Any]],
               campus_confidence: float, m_per_px: float = 0.3) -> Dict[str, Any]:
    """Fuse one field across sources and emit its calibration features."""
    usable = [s for s in signals if _usable(s)]
    n_available = len(usable)

    if n_available == 0:
        return {
            "field": field, "value": None, "sources": [],
            "features": _features(field, [], None, campus_confidence, 0, m_per_px),
            "n_sources": 0, "agreement": None,
        }

    if field in config.BINARY_FIELDS:
        value, agreement = _fuse_binary(usable)
    elif field in config.COUNT_FIELDS:
        value, agreement = _fuse_count(field, usable)
    elif field in CATEGORICAL_FIELDS:
        value, agreement = _fuse_categorical(usable)
    elif field == "solar_area_m2":
        value, agreement = _fuse_area(usable)
    else:
        value, agreement = usable[0]["value"], 1.0

    return {
        "field": field,
        "value": value,
        "sources": [{"source": s.get("source"), "value": s.get("value"),
                     "score": s.get("score")} for s in usable],
        "features": _features(field, usable, agreement, campus_confidence,
                              n_available, m_per_px),
        "n_sources": n_available,
        "agreement": agreement,
    }


def _fuse_binary(usable: List[Dict[str, Any]]) -> Tuple[bool, float]:
    yes = sum(_weight(s.get("source", "")) for s in usable if bool(s["value"]))
    no = sum(_weight(s.get("source", "")) for s in usable if not bool(s["value"]))
    total = yes + no
    value = yes >= no if total > 0 else False
    winning = yes if value else no
    return bool(value), (winning / total if total > 0 else 0.0)


def _fuse_count(field: str, usable: List[Dict[str, Any]]) -> Tuple[int, float]:
    counts = [int(s["value"]) for s in usable]
    value = int(round(statistics.median(counts)))
    tol = config.COUNT_TOLERANCE.get(field, 1)
    within = sum(1 for c in counts if abs(c - value) <= tol)
    return value, within / len(counts)


def _fuse_categorical(usable: List[Dict[str, Any]]) -> Tuple[Any, float]:
    tally: Dict[Any, float] = {}
    for s in usable:
        tally[s["value"]] = tally.get(s["value"], 0.0) + _weight(s.get("source", ""))
    value = max(tally.items(), key=lambda kv: kv[1])[0]
    total = sum(tally.values())
    return value, (tally[value] / total if total else 0.0)


def _fuse_area(usable: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Prefer a segmentation-derived area; fall back to the colour heuristic.

    Areas are compared by ratio rather than difference: 40 m2 vs 60 m2 is a real
    disagreement, 1400 vs 1500 is not.
    """
    seg = [s for s in usable if str(s.get("source", "")).startswith("bdappv")]
    chosen = seg[0] if seg else max(usable, key=lambda s: _weight(s.get("source", "")))
    value = float(chosen["value"])
    areas = [float(s["value"]) for s in usable if float(s["value"]) > 0]
    if len(areas) < 2 or value <= 0:
        return round(value, 1), 1.0 if len(areas) <= 1 else 0.0
    ratios = [min(a, value) / max(a, value) for a in areas]
    return round(value, 1), sum(ratios) / len(ratios)


def _features(field: str, usable: List[Dict[str, Any]], agreement: Optional[float],
              campus_confidence: float, n_available: int,
              m_per_px: float = 0.3) -> Dict[str, float]:
    """The inputs to calibration.

    `n_sources` matters for more than accounting: because it is a feature, a
    provider going down or a quota running out makes confidence fall honestly
    and automatically, instead of the pipeline silently pretending it still had
    the same evidence.
    """
    scores = [float(s["score"]) for s in usable
              if s.get("score") is not None]
    tta = [float(s["tta_std"]) for s in usable if s.get("tta_std") is not None]
    p_yes = [float(s["p_yes"]) for s in usable if s.get("p_yes") is not None]
    osm = [s for s in usable if str(s.get("source", "")).startswith("osm")]

    return {
        "n_sources": float(n_available),
        "agreement": float(agreement) if agreement is not None else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
        "tta_std": (sum(tta) / len(tta)) if tta else 0.0,
        # Distance of P(yes) from 0.5: a confident model probability, from a
        # logprob rather than a self-report.
        "logprob_margin": max((abs(p - 0.5) * 2 for p in p_yes), default=0.0),
        "has_logprob": 1.0 if p_yes else 0.0,
        # Present-vs-absent asymmetry: OSM saying "there is one" is strong;
        # OSM being silent is weak. Calibration learns how weak.
        "osm_positive": 1.0 if any(s.get("osm_present") for s in osm) else 0.0,
        "osm_silent": 1.0 if (osm and not any(s.get("osm_present") for s in osm)) else 0.0,
        "campus_confidence": float(campus_confidence),
        # Available imagery resolution varies by roughly 2x between rural and
        # urban sites (measured: 0.31 m/px in rural Alabama vs 0.16 m/px in Los
        # Angeles). Coarser pixels mean small objects are harder to see and the
        # solar segmenter is further from its training distribution, so
        # resolution is a genuine driver of error rather than metadata.
        "m_per_px": float(m_per_px),
        "coarse_imagery": 1.0 if m_per_px > 0.28 else 0.0,
        "is_fencing": 1.0 if field.startswith("fencing") else 0.0,
        "is_count": 1.0 if field in config.COUNT_FIELDS else 0.0,
    }


FEATURE_NAMES = [
    "n_sources", "agreement", "max_score", "mean_score", "tta_std",
    "logprob_margin", "has_logprob", "osm_positive", "osm_silent",
    "campus_confidence", "m_per_px", "coarse_imagery", "is_fencing", "is_count",
]


def fuse_school(per_source: Dict[str, Dict[str, Dict[str, Any]]],
                campus_confidence: float,
                m_per_px: float = 0.3) -> Dict[str, Dict[str, Any]]:
    """`per_source[source_name][field] -> signal`  ->  `field -> fused`."""
    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for source_name, fields in per_source.items():
        if not fields:
            continue
        for field, sig in fields.items():
            if sig is None:
                continue
            sig = dict(sig)
            sig.setdefault("source", source_name)
            by_field.setdefault(field, []).append(sig)
    return {f: fuse_field(f, sigs, campus_confidence, m_per_px)
            for f, sigs in by_field.items()}
