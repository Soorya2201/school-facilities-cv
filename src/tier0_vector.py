"""Layer 0: measurement from free vector data (OpenStreetMap).

For three of the four attributes the best tool is not a vision model. A campus
polygon plus tagged building/leisure/barrier geometry answers portables,
athletics and fencing directly, deterministically, and for free.

The one thing to keep straight is *asymmetry of evidence*:

    an OSM tag being PRESENT is strong positive evidence;
    an OSM tag being ABSENT is weak evidence of absence,

because mapping recall varies enormously between dense urban areas and rural
ones. So this module never returns a confident "no". It returns a value plus the
evidence behind it, and lets `fuse` and `calibrate` learn what absence is worth.
"""
from __future__ import annotations

from typing import Any, Dict, List

from . import config, osm
from .geo import (Ring, bbox_of_ring, min_distance_to_lines, min_rotated_rect,
                  point_in_polygon, polygon_area_m2, polygon_centroid,
                  rectangularity, sample_ring)

# How close a mapped barrier must be to the campus boundary to count as fencing
# that boundary rather than being some unrelated nearby fence.
FENCE_SNAP_M = 18.0
PERIMETER_SAMPLES = 120

COURT_AREA_MIN_M2 = 150.0     # a tennis court is 261 m2, basketball 420 m2
COURT_AREA_MAX_M2 = 1400.0
COURT_SPORTS = {"basketball", "tennis", "volleyball", "netball", "handball",
                "badminton", "pickleball", "multi"}
FIELD_SPORTS = {"soccer", "football", "american_football", "baseball",
                "softball", "rugby", "field_hockey", "lacrosse", "cricket"}
TRACK_SPORTS = {"running", "athletics", "track"}


def _signal(value: Any, source: str, evidence: Any, present: bool,
            note: str = "") -> Dict[str, Any]:
    """One measurement from one source.

    `present` records whether the source had positive evidence at all, which is
    what lets the calibration step treat "OSM said no" differently from
    "OSM had nothing to say".
    """
    return {"value": value, "source": source, "evidence": evidence,
            "osm_present": present, "note": note}


def _geom(g: Ring) -> Dict[str, Any]:
    """Where an object is, in lon/lat, so `compare` can crop the pixels it came from."""
    lon, lat = polygon_centroid(g)
    return {"centroid": [round(lon, 7), round(lat, 7)],
            "bounds": [round(v, 7) for v in bbox_of_ring(g)]}


def _on_campus(el: Dict[str, Any], ring: Ring) -> bool:
    g = osm.ring_of(el)
    if not g:
        return False
    return point_in_polygon(polygon_centroid(g), ring)


# --------------------------------------------------------------------------- #
# B. Portable classrooms
# --------------------------------------------------------------------------- #
def portables(ring: Ring, buildings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count building footprints with portable-classroom geometry.

    Codebook: a portable is a detached, single-storey, rectangular building of
    roughly 40-250 m2 with an aspect ratio between 1.3 and 4.5. This excludes
    permanent wings (too large, usually L-shaped) and sheds (too small).
    """
    if not ring:
        return _signal(None, "osm_vector", {}, False, "no campus polygon")

    on_campus, candidates = 0, []
    for b in buildings:
        g = osm.ring_of(b)
        if len(g) < 3 or not point_in_polygon(polygon_centroid(g), ring):
            continue
        on_campus += 1
        area = polygon_area_m2(g)
        if not (config.PORTABLE_AREA_MIN_M2 <= area <= config.PORTABLE_AREA_MAX_M2):
            continue
        lo, sh = min_rotated_rect(g)
        if sh <= 0:
            continue
        aspect = lo / sh
        rect = rectangularity(g)
        tags = b.get("tags", {})
        # An explicit tag is decisive either way.
        tagged_portable = tags.get("building") in {"portable", "cabin", "container"}
        # Residential tags are decisive negatives: a campus polygon drawn
        # generously can enclose neighbouring houses, and a house has exactly
        # the area and aspect ratio of a portable classroom. Without this the
        # count silently absorbs the surrounding street.
        excluded = tags.get("building") in {
            "garage", "garages", "shed", "roof", "carport", "greenhouse",
            "toilets", "house", "residential", "apartments", "detached",
            "semidetached_house", "terrace", "bungalow", "hut", "service",
        }
        if excluded and not tagged_portable:
            continue
        if (config.PORTABLE_ASPECT_MIN <= aspect <= config.PORTABLE_ASPECT_MAX
                and rect >= config.PORTABLE_RECTANGULARITY_MIN) or tagged_portable:
            candidates.append({
                "area_m2": round(area), "aspect": round(aspect, 2),
                "rect": round(rect, 2), "dims_m": [round(lo, 1), round(sh, 1)],
                "tag": tags.get("building"), **_geom(g),
            })

    evidence = {"buildings_on_campus": on_campus, "candidates": candidates}
    # No mapped buildings at all means OSM is silent here, not that the count is
    # zero. That distinction is the whole point of `osm_present`.
    has_data = on_campus > 0
    return _signal(len(candidates) if has_data else None, "osm_vector", evidence,
                   has_data, "" if has_data else "no mapped buildings on campus")


# --------------------------------------------------------------------------- #
# D. Athletic facilities
# --------------------------------------------------------------------------- #
def athletics(ring: Ring, leisure: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not ring:
        empty = _signal(None, "osm_vector", {}, False, "no campus polygon")
        return {k: dict(empty) for k in
                ("track_present", "field_count", "court_count", "pool_present")}

    tracks, fields, courts, pools = [], [], [], []
    n_features = 0
    for el in leisure:
        g = osm.ring_of(el)
        if len(g) < 3 or not point_in_polygon(polygon_centroid(g), ring):
            continue
        tags = el.get("tags", {})
        kind = tags.get("leisure")
        sport = (tags.get("sport") or "").split(";")[0]
        area = polygon_area_m2(g)
        n_features += 1

        if kind == "track" or sport in TRACK_SPORTS:
            tracks.append({"area_m2": round(area), "sport": sport, **_geom(g)})
        elif kind == "swimming_pool" or sport == "swimming":
            if config.POOL_AREA_MIN_M2 <= area <= config.POOL_AREA_MAX_M2 or kind == "swimming_pool":
                pools.append({"area_m2": round(area), **_geom(g)})
        elif kind == "pitch":
            # A single huge "pitch" polygon is usually the whole athletic
            # complex drawn as one shape, not one field. Counting it as a field
            # is fine; counting it as several would be wrong.
            if area > config.PITCH_AREA_MAX_M2:
                fields.append({"area_m2": round(area), "sport": sport,
                               "note": "oversize_complex", **_geom(g)})
            elif sport in COURT_SPORTS or (COURT_AREA_MIN_M2 <= area <= COURT_AREA_MAX_M2
                                           and sport not in FIELD_SPORTS):
                courts.append({"area_m2": round(area), "sport": sport, **_geom(g)})
            elif area >= config.FULL_SIZE_FIELD_MIN_M2 or sport in FIELD_SPORTS:
                fields.append({"area_m2": round(area), "sport": sport, **_geom(g)})

    has_data = n_features > 0
    note = "" if has_data else "no mapped leisure features on campus"
    return {
        "track_present": _signal(bool(tracks) if has_data else None, "osm_vector",
                                 tracks, bool(tracks), note),
        "field_count": _signal(len(fields) if has_data else None, "osm_vector",
                               fields, bool(fields), note),
        "court_count": _signal(len(courts) if has_data else None, "osm_vector",
                               courts, bool(courts), note),
        "pool_present": _signal(bool(pools) if has_data else None, "osm_vector",
                                pools, bool(pools), note),
    }


# --------------------------------------------------------------------------- #
# C. Perimeter fencing
# --------------------------------------------------------------------------- #
def fencing(ring: Ring, barriers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fraction of the campus perimeter with a mapped barrier alongside it.

    Note the honesty constraint: nadir imagery cannot resolve chain-link wire at
    0.3-0.6 m/px, so OSM is the only source that can see fence *type* at all,
    and OSM fence mapping recall is low. A confident "none" is not available to
    us from this source and we do not pretend otherwise.
    """
    if not ring:
        empty = _signal(None, "osm_vector", {}, False, "no campus polygon")
        return {"fencing_extent": dict(empty), "fencing_type": dict(empty)}

    lines = [osm.ring_of(b) for b in barriers if len(osm.ring_of(b)) >= 2]
    types: Dict[str, int] = {}
    for b in barriers:
        tags = b.get("tags", {})
        t = tags.get("fence_type") or tags.get("barrier")
        if t:
            types[t] = types.get(t, 0) + 1

    if not lines:
        note = "no mapped barriers near campus"
        return {
            "fencing_extent": _signal(None, "osm_vector", {"covered_fraction": None},
                                      False, note),
            "fencing_type": _signal(None, "osm_vector", {}, False, note),
        }

    samples = sample_ring(ring, PERIMETER_SAMPLES)
    covered = sum(1 for p in samples
                  if min_distance_to_lines(p, lines) <= FENCE_SNAP_M)
    frac = covered / len(samples) if samples else 0.0
    if frac >= config.FENCE_FULL_MIN:
        extent = "full"
    elif frac >= config.FENCE_PARTIAL_MIN:
        extent = "partial"
    else:
        extent = "none"

    dominant = max(types.items(), key=lambda kv: kv[1])[0] if types else None
    # OSM's generic `barrier=fence` records that a fence exists, not what it is
    # made of. Reporting "chain-link" here would be a guess presented as
    # evidence -- the single easiest way to end up miscalibrated on this field.
    # Only report a material when a mapper actually specified one.
    GENERIC = {"fence", "barrier", "yes", None}
    material = dominant if dominant not in GENERIC else None
    # If nothing fences the perimeter, a material is not applicable.
    if frac < config.FENCE_PARTIAL_MIN:
        material = None
    evidence = {"covered_fraction": round(frac, 3), "samples": len(samples),
                "barrier_types": types, "n_barrier_ways": len(lines)}
    return {
        "fencing_extent": _signal(extent, "osm_vector", evidence, True),
        "fencing_type": _signal(
            material, "osm_vector", {"types": types}, material is not None,
            "" if material else "barrier mapped but material not specified"),
    }


def extract_all(ring: Ring, buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    out["portable_count"] = portables(ring, buckets.get("buildings", []))
    out.update(athletics(ring, buckets.get("leisure", [])))
    out.update(fencing(ring, buckets.get("barriers", [])))
    return out
