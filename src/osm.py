"""OpenStreetMap access via Overpass.

OSM is used two ways, and the distinction matters for calibration:
  * as the source of the campus polygon (Step 1), and
  * as an *independent labeller* for athletics, fencing and building geometry.

Critical asymmetry, applied throughout: an OSM tag being **present** is strong
positive evidence, but a tag being **absent** is weak evidence of absence,
because mapping recall varies enormously between urban and rural areas. The
calibration step is given `osm_present` and urbanicity separately so it can
learn that recall rather than us assuming it.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from . import config
from .cache import Cache, key_of, retry
from .geo import Ring, bbox_from_center

_cache = Cache(config.OSM_CACHE)
_last_call = [0.0]
MIN_INTERVAL_S = 1.2  # be polite: the public instance allows 2 slots per IP

# Overpass returns HTTP 406 for the default python-requests User-Agent. Failing
# to set this looks exactly like a network hang once mirror fallback kicks in,
# so it is worth an explicit constant.
HEADERS = {
    "User-Agent": "school-facilities-cv/0.1 (USC Dornsife research; contact sv57690@usc.edu)",
    "Content-Type": "text/plain",
}
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 90


def _post(query: str) -> Dict[str, Any]:
    """POST to the first Overpass mirror that answers.

    Mirrors are tried in order and a failure is recorded rather than raised, so
    one dead mirror degrades latency instead of killing the run.
    """
    errors = []
    for endpoint in config.OVERPASS_ENDPOINTS:
        gap = time.time() - _last_call[0]
        if gap < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - gap)
        try:
            r = requests.post(endpoint, data=query.encode(), headers=HEADERS,
                              timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
            _last_call[0] = time.time()
            if r.status_code == 200:
                payload = r.json()
                if _looks_like_full_planet(payload):
                    payload["_served_by"] = endpoint
                    return payload
                # A regional mirror answering a US query returns 200 with an
                # empty element list. Accepting that would silently mean
                # "no buildings, no fences" rather than "source unavailable".
                errors.append(f"{endpoint}: 200 but empty (regional mirror?)")
            else:
                errors.append(f"{endpoint}: HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint}: {type(exc).__name__}")
    raise RuntimeError("all Overpass mirrors failed: " + "; ".join(errors))


def _looks_like_full_planet(payload: Dict[str, Any]) -> bool:
    """Reject a mirror whose database does not actually cover the query area.

    Any US school campus has *some* mapped feature within a few hundred metres.
    A completely empty response means the mirror is regional, not that the
    campus is empty.
    """
    return bool(payload.get("elements"))


def campus_query(lat: float, lon: float, radius_m: float) -> str:
    """Everything we need about one campus in a single round trip.

    Two deliberate choices here:
      * a *bbox* filter rather than `around:` -- Overpass evaluates bbox filters
        far faster, and at campus scale the difference between a box and a disc
        is immaterial;
      * one batched query per school rather than four separate ones, because 25
        schools x 4 queries would trip the rate limit while 25 will not.
    """
    bb = bbox_from_center(lat, lon, radius_m)
    # Overpass bbox order is (south, west, north, east).
    box = f"{bb[1]:.6f},{bb[0]:.6f},{bb[3]:.6f},{bb[2]:.6f}"
    return f"""[out:json][timeout:60];
(
  way["amenity"="school"]({box});
  relation["amenity"="school"]({box});
  way["building"]({box});
  way["barrier"]({box});
  way["leisure"]({box});
);
out tags geom;"""


def fetch_campus_context(lat: float, lon: float, radius_m: float,
                         offline: bool = False) -> Optional[Dict[str, Any]]:
    k = key_of("overpass_ctx_v1", round(lat, 6), round(lon, 6), int(radius_m))
    return _cache.json_or_fetch(
        k, lambda: retry(lambda: _post(campus_query(lat, lon, radius_m)),
                         label="overpass"),
        offline=offline,
    )


def ring_of(element: Dict[str, Any]) -> Ring:
    """Overpass `geom` -> our (lon, lat) ring."""
    return [(p["lon"], p["lat"]) for p in element.get("geometry") or []]


def split(elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket raw Overpass elements by the role they play downstream."""
    out: Dict[str, List[Dict[str, Any]]] = {
        "schools": [], "buildings": [], "barriers": [], "leisure": []
    }
    for el in elements:
        tags = el.get("tags") or {}
        if not el.get("geometry"):
            continue
        if tags.get("amenity") == "school":
            out["schools"].append(el)
        elif "building" in tags:
            out["buildings"].append(el)
        elif "barrier" in tags:
            out["barriers"].append(el)
        elif "leisure" in tags or "sport" in tags:
            out["leisure"].append(el)
    return out


def stats() -> Dict[str, int]:
    return _cache.stats()
