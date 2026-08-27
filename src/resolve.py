"""Step 1: work out where each campus actually is.

The task warns that CCD coordinates are "the federal record, not a survey" and
may be a mailing address or the district office. This step is therefore upstream
of everything else: if we do not know where the campus is, no other field can be
trusted, so `campus_confidence` is propagated into every downstream confidence.

Resolution ladder, most to least trustworthy:
  1. an OSM `amenity=school` polygon whose name matches the CCD name
  2. the nearest OSM school polygon containing/near the CCD point, unnamed
  3. the convex hull of a cluster of large building footprints
  4. a disc around the CCD point, sized by school level
"""
from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config, osm
from .geo import (Ring, bbox_from_center, bbox_of_ring, haversine_m,
                  polygon_area_m2, polygon_centroid, point_in_polygon)

# Abbreviations that appear in CCD names but not in OSM, and vice versa.
_ABBREV = {
    "elem": "elementary", "es": "elementary", "el": "elementary",
    "hs": "high", "sr": "senior", "jr": "junior",
    "ms": "middle", "sch": "school", "co": "county", "ctr": "center",
    "int": "intermediate", "acad": "academy", "ed": "education",
}
_STOP = {"school", "the", "of", "public", "district", "at"}


def normalize_name(name: str) -> List[str]:
    """Lowercase, expand abbreviations, drop punctuation and stopwords."""
    name = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    tokens = []
    for tok in name.split():
        tok = _ABBREV.get(tok, tok)
        if tok and tok not in _STOP:
            tokens.append(tok)
    return tokens


def name_similarity(a: str, b: str) -> float:
    """Token-overlap blended with sequence ratio.

    Deliberately dependency-free rather than pulling in rapidfuzz: school names
    differ mostly by abbreviation and word order, which token overlap handles.
    """
    ta, tb = normalize_name(a), normalize_name(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)
    seq = difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return 0.65 * jaccard + 0.35 * seq


@dataclass
class Campus:
    school_id: str
    name: str
    lat: float
    lon: float
    level: str
    enrollment: int
    ring: Ring = field(default_factory=list)
    area_m2: float = 0.0
    method: str = "none"
    confidence: float = 0.0
    name_match: float = 0.0
    offset_m: float = 0.0
    area_per_student: float = 0.0
    n_candidates: int = 0
    flags: List[str] = field(default_factory=list)

    @property
    def centroid(self):
        return polygon_centroid(self.ring) if self.ring else (self.lon, self.lat)

    def bbox(self, pad: float = 0.15) -> List[float]:
        """Imagery extent for this campus, with a floor on how small it can be.

        A degenerate polygon (e.g. a building-cluster hull of a few hundred
        square metres) would otherwise produce a ~20 m bbox, which is finer
        than any available imagery and simply returns nothing. Failing to
        measure because the *extent* was wrong is a silent, avoidable loss, so
        the box never shrinks below the level-appropriate fallback radius.
        """
        r = config.FALLBACK_RADIUS_M.get(self.level, 200.0)
        if not self.ring:
            return bbox_from_center(self.lat, self.lon, r * (1 + pad))

        w, s, e, n = bbox_of_ring(self.ring)
        dw, dh = (e - w) * pad, (n - s) * pad
        box = [w - dw, s - dh, e + dw, n + dh]

        lon_c, lat_c = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        width_m = haversine_m(box[0], lat_c, box[2], lat_c)
        height_m = haversine_m(lon_c, box[1], lon_c, box[3])
        min_half = 0.5 * r      # never image a box narrower than this
        if max(width_m, height_m) < 2 * min_half:
            return bbox_from_center(lat_c, lon_c, min_half)
        return box

    def to_dict(self) -> Dict[str, Any]:
        return {
            "school_id": self.school_id, "school_name": self.name,
            "campus_method": self.method,
            "campus_confidence": round(self.confidence, 3),
            "campus_area_m2": round(self.area_m2),
            "campus_name_match": round(self.name_match, 3),
            "campus_offset_m": round(self.offset_m, 1),
            "area_per_student": round(self.area_per_student, 1),
            "campus_flags": ";".join(self.flags),
        }


def load_schools(path=None) -> List[Dict[str, str]]:
    with open(path or config.SCHOOLS_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def _cluster_buildings(buildings: List[Dict], lat: float, lon: float,
                       radius_m: float) -> Optional[Ring]:
    """Convex hull of substantial buildings near the point.

    Used where OSM has building footprints but nobody has drawn the school
    boundary -- common in rural areas.
    """
    pts: List = []
    for b in buildings:
        ring = osm.ring_of(b)
        if len(ring) < 3:
            continue
        if polygon_area_m2(ring) < 200:      # sheds and garages are not campuses
            continue
        c = polygon_centroid(ring)
        if haversine_m(lon, lat, c[0], c[1]) <= radius_m:
            pts.extend(ring)
    if len(pts) < 6:
        return None
    from .geo import _convex_hull
    lat0 = sum(p[1] for p in pts) / len(pts)
    import math
    k = math.pi / 180.0 * 6378137.0
    cos0 = math.cos(math.radians(lat0))
    xy = [(p[0] * k * cos0, p[1] * k) for p in pts]
    hull_xy = _convex_hull(xy)
    return [(x / (k * cos0), y / k) for x, y in hull_xy]


def resolve_campus(row: Dict[str, str], offline: bool = False) -> Campus:
    lat, lon = float(row["latitude"]), float(row["longitude"])
    enrollment = int(row.get("enrollment_2024") or 0)
    cam = Campus(
        school_id=row["school_id"], name=row["school_name"], lat=lat, lon=lon,
        level=row.get("level", ""), enrollment=enrollment,
    )

    try:
        ctx = osm.fetch_campus_context(lat, lon, config.CAMPUS_SEARCH_RADIUS_M,
                                       offline=offline)
    except Exception as exc:  # noqa: BLE001
        cam.flags.append(f"osm_error:{type(exc).__name__}")
        ctx = None

    if not ctx:
        cam.method = "fallback_radius"
        cam.confidence = 0.30
        cam.flags.append("no_osm_context")
        _finish(cam)
        return cam

    buckets = osm.split(ctx["elements"])
    schools = [s for s in buckets["schools"] if len(osm.ring_of(s)) >= 3]
    cam.n_candidates = len(schools)

    scored = []
    for s in schools:
        ring = osm.ring_of(s)
        cen = polygon_centroid(ring)
        sim = name_similarity(cam.name, s.get("tags", {}).get("name", "") or "")
        dist = haversine_m(lon, lat, cen[0], cen[1])
        contains = point_in_polygon((lon, lat), ring)
        # Name agreement dominates; containment and proximity break ties.
        score = sim + (0.35 if contains else 0.0) - min(dist, 1000) / 4000.0
        scored.append((score, sim, dist, contains, ring))
    scored.sort(key=lambda t: -t[0])

    if scored and scored[0][1] >= config.NAME_MATCH_MIN:
        score, sim, dist, contains, ring = scored[0]
        cam.ring, cam.name_match, cam.offset_m = ring, sim, dist
        cam.method = "osm_name_match"
        cam.confidence = 0.72 + 0.23 * min(sim, 1.0)
    elif scored and (scored[0][3] or scored[0][2] < 250):
        score, sim, dist, contains, ring = scored[0]
        cam.ring, cam.name_match, cam.offset_m = ring, sim, dist
        cam.method = "osm_nearest_school"
        cam.confidence = 0.62 if contains else 0.52
        cam.flags.append("weak_name_match")
    else:
        hull = _cluster_buildings(buckets["buildings"], lat, lon, 300)
        if hull:
            cam.ring = hull
            cam.method = "building_cluster"
            cam.confidence = 0.45
            cam.offset_m = haversine_m(lon, lat, *polygon_centroid(hull))
            cam.flags.append("no_school_polygon")
        else:
            cam.method = "fallback_radius"
            cam.confidence = 0.30
            cam.flags.append("no_school_polygon")
            cam.flags.append("no_building_cluster")

    if cam.n_candidates > 1:
        cam.flags.append(f"multi_candidate:{cam.n_candidates}")
    _finish(cam)
    return cam


def _finish(cam: Campus) -> None:
    """Sanity checks that can only be applied once an extent exists."""
    if cam.ring:
        cam.area_m2 = polygon_area_m2(cam.ring)
    else:
        r = config.FALLBACK_RADIUS_M.get(cam.level, 200.0)
        cam.area_m2 = 3.14159 * r * r

    if cam.enrollment > 0:
        cam.area_per_student = cam.area_m2 / cam.enrollment
        if cam.area_per_student < config.AREA_PER_STUDENT_MIN:
            cam.flags.append("area_per_student_low")
            cam.confidence *= 0.75
        elif cam.area_per_student > config.AREA_PER_STUDENT_MAX:
            cam.flags.append("area_per_student_high")
            cam.confidence *= 0.75

    # A campus centroid far from the federal coordinate is exactly the case the
    # task warns about. It is not necessarily wrong, but it is worth flagging.
    if cam.offset_m > 400:
        cam.flags.append("far_from_ccd")
        cam.confidence *= 0.85

    cam.confidence = round(max(0.05, min(0.98, cam.confidence)), 3)
