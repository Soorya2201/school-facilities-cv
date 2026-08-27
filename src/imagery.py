"""Imagery acquisition. Every source here is keyless and free.

  * Esri World Imagery  -- primary, sub-metre, arbitrary bbox via `export`
  * Esri Wayback        -- 190+ dated releases back to 2014, for imagery vintage
                           and the two-date (temporal) measurement
  * NAIP                -- independent third view; its catalogue carries exact
                           acquisition dates, and its NIR band gives a free
                           vegetation index for canopy-occlusion flags

Images are cached by a content hash of the request, so re-runs are free and a
reviewer can reproduce the run with no network at all.
"""
from __future__ import annotations

import datetime as _dt
import io
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image

from . import config
from .cache import Cache, key_of, retry
from .geo import EARTH_RADIUS_M, bbox_meters_per_pixel, square_bbox

_img_cache = Cache(config.IMAGERY_CACHE, suffix=".jpg")
_meta_cache = Cache(config.IMAGERY_CACHE, suffix=".json")

HEADERS = {"User-Agent": "school-facilities-cv/0.1 (USC Dornsife research)"}
TIMEOUT = (10, 60)


class ImageResult:
    """An image plus everything needed to convert pixels to metres."""

    def __init__(self, data: bytes, bbox: Sequence[float], width: int, height: int,
                 source: str, date: Optional[str], cache_key: str):
        self.data = data
        self.bbox = list(bbox)
        self.width, self.height = width, height
        self.source = source
        self.date = date
        self.cache_key = cache_key

    @property
    def meters_per_pixel(self) -> float:
        return bbox_meters_per_pixel(self.bbox, self.width, self.height)

    def pil(self) -> Image.Image:
        return Image.open(io.BytesIO(self.data)).convert("RGB")

    def meta(self) -> Dict[str, Any]:
        return {"source": self.source, "date": self.date, "bbox": self.bbox,
                "width": self.width, "height": self.height,
                "m_per_px": round(self.meters_per_pixel, 4),
                "cache_key": self.cache_key}


# --------------------------------------------------------------------------- #
# Esri World Imagery (primary)
# --------------------------------------------------------------------------- #
# Pixel sizes tried, largest first. Esri World Imagery enforces a per-location
# maximum resolution and returns HTTP 500 -- deterministically, not
# transiently -- for anything finer. Measured in rural Alabama: a 200 m extent
# succeeds at 768 px (0.26 m/px) but hard-fails at 1024 px (0.195 m/px), while
# the same request succeeds at 1024 px in Los Angeles.
#
# The ladder walks the PIXEL COUNT down while holding the ground extent fixed,
# rather than widening the box. Keeping the extent means the campus stays framed
# and the geometry stays correct; we simply record the coarser metres-per-pixel
# we actually got, which then flows into the confidence for that school.
SIZE_LADDER = [1024, 768, 512, 384, 256]


def fetch_esri(bbox: Sequence[float], size: int = config.DEFAULT_TILE_PX,
               offline: bool = False) -> Optional[ImageResult]:
    # A square pixel request must describe a square *ground* extent, or the
    # resulting metres-per-pixel differs by axis and every area is wrong. The
    # aspect assertion in geo.bbox_meters_per_pixel catches this, so square the
    # box here rather than let callers get it right by accident.
    bbox = square_bbox(bbox)
    ladder = [s for s in SIZE_LADDER if s <= size] or [size]

    # A cached result from any rung of the ladder is a hit.
    for candidate in ladder:
        k = key_of("esri_v1", [round(v, 6) for v in bbox], candidate)
        data = _img_cache.get_bytes(k)
        if data is not None:
            return ImageResult(data, bbox, candidate, candidate,
                               "esri_world_imagery", None, k)
    if offline:
        return None

    for candidate in ladder:
        k = key_of("esri_v1", [round(v, 6) for v in bbox], candidate)
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "bboxSR": 4326, "imageSR": 3857,
            "size": f"{candidate},{candidate}",
            "format": "jpg", "f": "image",
        }
        try:
            r = requests.get(config.ESRI_EXPORT, params=params, headers=HEADERS,
                             timeout=TIMEOUT)
        except Exception:
            continue
        if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
            _img_cache.put_bytes(k, r.content)
            return ImageResult(r.content, bbox, candidate, candidate,
                               "esri_world_imagery", None, k)
        # HTTP 500 here means "no imagery at this resolution", so step down.
        # Any other status is also worth stepping down for -- there is no
        # benefit to retrying the same rung.
    return None


# --------------------------------------------------------------------------- #
# Esri Wayback (dated historical imagery)
# --------------------------------------------------------------------------- #
def wayback_releases(offline: bool = False) -> List[Dict[str, Any]]:
    """All Wayback releases, newest first, as {release, date}."""
    k = key_of("wayback_releases_v1")

    def _go():
        r = requests.get(config.WAYBACK_CONFIG, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        out = []
        for rel, meta in r.json().items():
            title = meta.get("itemTitle", "")
            # "World Imagery (Wayback 2023-05-17)"
            date = title.split("Wayback ")[-1].rstrip(")") if "Wayback " in title else None
            if date:
                out.append({"release": rel, "date": date})
        out.sort(key=lambda d: d["date"], reverse=True)
        return out

    return _meta_cache.json_or_fetch(k, lambda: retry(_go, label="wayback config"),
                                     offline=offline) or []


def zoom_for_bbox(bbox: Sequence[float], target_px: int) -> int:
    """Smallest zoom whose mosaic is at least `target_px` across."""
    west, south, east, north = bbox
    lat_mid = (south + north) / 2.0
    width_m = (east - west) * (math.pi / 180) * EARTH_RADIUS_M * math.cos(math.radians(lat_mid))
    for z in range(21, 12, -1):
        mpp = 156543.03392804097 * math.cos(math.radians(lat_mid)) / (2 ** z)
        if width_m / mpp <= target_px * 1.6:
            return z
    return 18


def _tile_pixel(lat: float, lon: float, z: int) -> Tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x * 256.0, y * 256.0


def fetch_xyz_mosaic(bbox: Sequence[float], url_template: str, source: str,
                     date: Optional[str], target_px: int = config.DEFAULT_TILE_PX,
                     offline: bool = False, **fmt) -> Optional[ImageResult]:
    """Stitch XYZ tiles covering `bbox`, then crop to the exact bbox.

    Cropping to the bbox (rather than returning whole tiles) is what keeps the
    metres-per-pixel of the result exactly derivable from the bbox.
    """
    bbox = square_bbox(bbox)
    k = key_of("xyz_v1", source, date, [round(v, 6) for v in bbox], target_px)
    data = _img_cache.get_bytes(k)
    if data is not None:
        img = Image.open(io.BytesIO(data))
        return ImageResult(data, bbox, img.width, img.height, source, date, k)
    if offline:
        return None

    west, south, east, north = bbox
    z = zoom_for_bbox(bbox, target_px)
    x0f, y0f = _tile_pixel(north, west, z)      # top-left in global pixels
    x1f, y1f = _tile_pixel(south, east, z)      # bottom-right
    tx0, ty0 = int(x0f // 256), int(y0f // 256)
    tx1, ty1 = int(x1f // 256), int(y1f // 256)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64:
        z -= 1
        x0f, y0f = _tile_pixel(north, west, z)
        x1f, y1f = _tile_pixel(south, east, z)
        tx0, ty0 = int(x0f // 256), int(y0f // 256)
        tx1, ty1 = int(x1f // 256), int(y1f // 256)

    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    got = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = url_template.format(z=z, x=tx, y=ty, **fmt)
            try:
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                if r.status_code != 200 or len(r.content) < 100:
                    continue
                tile = Image.open(io.BytesIO(r.content)).convert("RGB")
                mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
                got += 1
            except Exception:  # noqa: BLE001 - a missing tile is not fatal
                continue
    if got == 0:
        return None

    crop = mosaic.crop((int(x0f - tx0 * 256), int(y0f - ty0 * 256),
                        int(x1f - tx0 * 256), int(y1f - ty0 * 256)))
    if crop.width < 8 or crop.height < 8:
        return None
    if crop.width > target_px:
        crop = crop.resize((target_px, int(target_px * crop.height / crop.width)),
                           Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=88)
    data = buf.getvalue()
    _img_cache.put_bytes(k, data)
    return ImageResult(data, bbox, crop.width, crop.height, source, date, k)


def fetch_wayback(bbox: Sequence[float], release: str, date: str,
                  target_px: int = config.DEFAULT_TILE_PX,
                  offline: bool = False) -> Optional[ImageResult]:
    return fetch_xyz_mosaic(bbox, config.WAYBACK_TILE, f"wayback_{date}", date,
                            target_px=target_px, offline=offline, release=release)


# --------------------------------------------------------------------------- #
# NAIP
# --------------------------------------------------------------------------- #
def naip_metadata(lat: float, lon: float, offline: bool = False) -> Dict[str, Any]:
    """Acquisition dates for NAIP rasters covering a point.

    Gives an imagery *vintage* that is a real acquisition date rather than a
    publication date, which is what the deliverable asks for.
    """
    k = key_of("naip_meta_v2", round(lat, 5), round(lon, 5))

    def _go():
        params = {
            "geometry": f'{{"x":{lon},"y":{lat},"spatialReference":{{"wkid":4326}}}}',
            "geometryType": "esriGeometryPoint", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "where": "Category=1",
            "outFields": "Year,acquisition_date,resolution_value,State",
            "returnGeometry": "false", "f": "json",
        }
        r = requests.get(config.NAIP_IMAGESERVER + "/query", params=params,
                         headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        feats = r.json().get("features", [])
        dates = []
        for f in feats:
            a = f.get("attributes", {})
            ms = a.get("acquisition_date")
            if ms:
                dates.append({
                    "year": a.get("Year"),
                    "date": _dt.datetime.utcfromtimestamp(ms / 1000).date().isoformat(),
                    "res_m": a.get("resolution_value"),
                })
        dates.sort(key=lambda d: d["date"] or "")
        return {"n": len(feats), "dates": dates[:8]}

    try:
        return _meta_cache.json_or_fetch(k, lambda: retry(_go, attempts=3,
                                                          label="naip meta"),
                                          offline=offline) or {"n": 0, "dates": []}
    except Exception:
        return {"n": 0, "dates": []}


def fetch_naip(bbox: Sequence[float], size: int = config.DEFAULT_TILE_PX,
               offline: bool = False) -> Optional[ImageResult]:
    bbox = square_bbox(bbox)
    k = key_of("naip_img_v1", [round(v, 6) for v in bbox], size)
    data = _img_cache.get_bytes(k)
    if data is None:
        if offline:
            return None
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox), "bboxSR": 4326,
            "imageSR": 3857, "size": f"{size},{size}", "format": "jpg", "f": "image",
        }

        def _go() -> bytes:
            r = requests.get(config.NAIP_IMAGESERVER + "/exportImage",
                             params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            if r.content[:2] != b"\xff\xd8":
                raise RuntimeError("non-image response")
            return r.content

        try:
            data = retry(_go, attempts=3, label="naip image")
        except Exception:
            return None       # NAIP is a supporting source, never load-bearing
        _img_cache.put_bytes(k, data)
    return ImageResult(data, bbox, size, size, "naip", None, k)


def stats() -> Dict[str, int]:
    return {"img": _img_cache.stats(), "meta": _meta_cache.stats()}
