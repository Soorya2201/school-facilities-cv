"""Geometry helpers.

Deliberately dependency-free (no shapely/geopandas): the amount of geometry this
pipeline needs is small, and every dependency removed is one less way the repo
fails to install on a reviewer's machine.

The metres-per-pixel conversion here is the single most dangerous piece of code
in the project. If it is wrong, every solar area is off by a constant factor and
*nothing raises an error*. See `validate_scale_bar`.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

EARTH_CIRCUMFERENCE_M = 40075016.686
EARTH_RADIUS_M = 6378137.0
# Web Mercator ground resolution at zoom 0, equator, 256px tiles.
BASE_RESOLUTION = EARTH_CIRCUMFERENCE_M / 256.0  # 156543.03392804097

Point = Tuple[float, float]  # (lon, lat)
Ring = List[Point]


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
def meters_per_pixel(lat: float, zoom: float, scale: int = 1) -> float:
    """Ground resolution of a Web Mercator tile pyramid at `lat`.

    `scale=2` means retina tiles: same ground extent, twice the pixels, so half
    the metres per pixel. Getting this factor wrong silently quadruples areas.
    """
    return BASE_RESOLUTION * math.cos(math.radians(lat)) / (2.0 ** zoom) / scale


def bbox_meters_per_pixel(bbox: Sequence[float], width_px: int, height_px: int) -> float:
    """Metres per pixel implied by an explicit lon/lat bbox rendered at a size.

    This is the path we actually use (Esri `export` takes a bbox, not a zoom), so
    it is derived from the bbox rather than assumed from a zoom level.
    """
    west, south, east, north = bbox
    lat_mid = (south + north) / 2.0
    width_m = (east - west) * (math.pi / 180.0) * EARTH_RADIUS_M * math.cos(math.radians(lat_mid))
    height_m = (north - south) * (math.pi / 180.0) * EARTH_RADIUS_M
    mpp_x = width_m / width_px
    mpp_y = height_m / height_px
    # A mismatch here means the bbox aspect and the pixel aspect disagree, which
    # is the classic symptom of an EPSG:3857 / 4326 mix-up.
    if abs(mpp_x - mpp_y) / max(mpp_x, mpp_y) > 0.02:
        raise ValueError(
            f"bbox/pixel aspect mismatch: {mpp_x:.4f} vs {mpp_y:.4f} m/px. "
            "Check that the bbox is lon/lat and the requested size matches its aspect."
        )
    return (mpp_x + mpp_y) / 2.0


def validate_scale_bar(measured_m: float, expected_m: float, tolerance: float,
                       what: str) -> Tuple[bool, str]:
    """Check a measured real-world length against a known reference.

    A running track is 400 m; a tennis court 23.77 x 10.97 m. If we detect one
    and our geometry says it is 800 m long, the scale maths is broken.
    """
    if expected_m <= 0:
        return False, "invalid reference"
    err = abs(measured_m - expected_m) / expected_m
    ok = err <= tolerance
    return ok, f"{what}: measured {measured_m:.1f} m vs expected {expected_m:.1f} m ({err:.1%} error)"


# --------------------------------------------------------------------------- #
# Distances and boxes
# --------------------------------------------------------------------------- #
def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bbox_from_center(lat: float, lon: float, half_width_m: float,
                     half_height_m: Optional[float] = None) -> List[float]:
    """(west, south, east, north) box of given half-extent around a point."""
    if half_height_m is None:
        half_height_m = half_width_m
    dlat = (half_height_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (half_width_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def square_bbox(bbox: Sequence[float]) -> List[float]:
    """Expand a bbox to square in *ground* units, so a square image is undistorted."""
    west, south, east, north = bbox
    lat_mid = (south + north) / 2.0
    w_m = (east - west) * (math.pi / 180) * EARTH_RADIUS_M * math.cos(math.radians(lat_mid))
    h_m = (north - south) * (math.pi / 180) * EARTH_RADIUS_M
    side = max(w_m, h_m)
    lon_c, lat_c = (west + east) / 2.0, lat_mid
    return bbox_from_center(lat_c, lon_c, side / 2.0)


def pad_bbox(bbox: Sequence[float], fraction: float) -> List[float]:
    west, south, east, north = bbox
    dw, dh = (east - west) * fraction, (north - south) * fraction
    return [west - dw, south - dh, east + dw, north + dh]


def latlon_to_tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


# --------------------------------------------------------------------------- #
# Polygons  (rings are lists of (lon, lat), implicitly closed)
# --------------------------------------------------------------------------- #
def _local_xy(ring: Ring) -> List[Tuple[float, float]]:
    """Project to a local equirectangular frame in metres. Fine at campus scale."""
    if not ring:
        return []
    lat0 = sum(p[1] for p in ring) / len(ring)
    k = math.pi / 180.0 * EARTH_RADIUS_M
    cos0 = math.cos(math.radians(lat0))
    return [(p[0] * k * cos0, p[1] * k) for p in ring]


def polygon_area_m2(ring: Ring) -> float:
    if len(ring) < 3:
        return 0.0
    pts = _local_xy(ring)
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_perimeter_m(ring: Ring) -> float:
    if len(ring) < 2:
        return 0.0
    pts = _local_xy(ring)
    return sum(
        math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))
    )


def polygon_centroid(ring: Ring) -> Point:
    if not ring:
        raise ValueError("empty ring")
    if len(ring) < 3:
        return (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))
    a = 0.0
    cx = cy = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        return (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def point_in_polygon(pt: Point, ring: Ring) -> bool:
    """Standard ray casting. Boundary cases are not important at this scale."""
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1
            if x < xint:
                inside = not inside
    return inside


def bbox_of_ring(ring: Ring) -> List[float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [min(xs), min(ys), max(xs), max(ys)]


def min_rotated_rect(ring: Ring) -> Tuple[float, float]:
    """(long_side_m, short_side_m) of the minimum-area enclosing rectangle.

    Rotating-callipers over the convex hull. Used to judge whether a building
    footprint has portable-classroom proportions.
    """
    pts = _local_xy(ring)
    if len(pts) < 3:
        return (0.0, 0.0)
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return (0.0, 0.0)
    best = None
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        ang = math.atan2(y2 - y1, x2 - x1)
        ca, sa = math.cos(-ang), math.sin(-ang)
        xs = [p[0] * ca - p[1] * sa for p in hull]
        ys = [p[0] * sa + p[1] * ca for p in hull]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if best is None or w * h < best[0] * best[1]:
            best = (w, h)
    w, h = best
    return (max(w, h), min(w, h))


def _convex_hull(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def rectangularity(ring: Ring) -> float:
    """Polygon area divided by its minimum enclosing rectangle area.

    1.0 is a perfect rectangle. Portable classrooms are boxes, so this separates
    them from L-shaped permanent buildings.
    """
    area = polygon_area_m2(ring)
    lo, sh = min_rotated_rect(ring)
    rect = lo * sh
    return area / rect if rect > 0 else 0.0


def sample_ring(ring: Ring, n: int) -> List[Point]:
    """`n` points spaced evenly along the perimeter of a ring."""
    if len(ring) < 2:
        return list(ring)
    pts = _local_xy(ring)
    segs = []
    total = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        d = math.dist(a, b)
        segs.append((i, d))
        total += d
    if total <= 0:
        return list(ring)
    out: List[Point] = []
    step = total / n
    target = 0.0
    acc = 0.0
    idx = 0
    for _ in range(n):
        while idx < len(segs) and acc + segs[idx][1] < target:
            acc += segs[idx][1]
            idx += 1
        if idx >= len(segs):
            break
        i, d = segs[idx]
        t = (target - acc) / d if d > 0 else 0.0
        a, b = ring[i], ring[(i + 1) % len(ring)]
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        target += step
    return out


def point_to_segment_m(p: Point, a: Point, b: Point) -> float:
    """Distance in metres from p to segment ab, in a local frame around p."""
    lat0 = p[1]
    k = math.pi / 180.0 * EARTH_RADIUS_M
    cos0 = math.cos(math.radians(lat0))
    px, py = p[0] * k * cos0, p[1] * k
    ax, ay = a[0] * k * cos0, a[1] * k
    bx, by = b[0] * k * cos0, b[1] * k
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def min_distance_to_lines(p: Point, lines: Iterable[Ring]) -> float:
    best = float("inf")
    for line in lines:
        for i in range(len(line) - 1):
            d = point_to_segment_m(p, line[i], line[i + 1])
            if d < best:
                best = d
    return best
