"""Central configuration: paths, endpoints, and tunable thresholds.

Every magic number that affects a measurement lives here so the memo can cite it
and a reviewer can change it in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = Path(os.environ.get("SCHOOLCV_CACHE", ROOT / "cache"))
IMAGERY_CACHE = CACHE / "imagery"
API_CACHE = CACHE / "api"
OSM_CACHE = CACHE / "osm"
PREDS = ROOT / "preds"
OUT = ROOT / "out"
LABELS = ROOT / "labels"

for _d in (IMAGERY_CACHE, API_CACHE, OSM_CACHE, PREDS, OUT, LABELS):
    _d.mkdir(parents=True, exist_ok=True)

SCHOOLS_CSV = DATA / "schools_sample.csv"

# --- Imagery endpoints (all keyless; verified working) ----------------------
ESRI_EXPORT = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)
WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_TILE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
    "WMTS/1.0.0/default028mm/MapServer/tile/{release}/{z}/{y}/{x}"
)
NAIP_IMAGESERVER = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
)

# Overpass mirrors, tried in order. The main instance rate-limits at 2 slots/IP
# and will drop connections outright under load, so a fallback list is required
# rather than optional -- we hit exactly that during development.
#
# Only full-planet instances belong here. Regional mirrors (overpass.osm.ch,
# openhistoricalmap) answer HTTP 200 with an *empty* result set for US queries,
# which would silently produce "no buildings, no fences" instead of an error.
# That is why `osm._post` also validates the response shape, not just the status.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# --- Campus resolution ------------------------------------------------------
CAMPUS_SEARCH_RADIUS_M = 800       # how far from the CCD point we look
NAME_MATCH_MIN = 0.55              # below this a name match is not trusted
# Plausible campus area per enrolled student. Outside this range we flag.
AREA_PER_STUDENT_MIN = 5.0
AREA_PER_STUDENT_MAX = 300.0
# Fallback campus radius when nothing else resolves, by school level.
FALLBACK_RADIUS_M = {"Primary": 150.0, "Middle": 200.0, "High": 260.0}

# --- Portables --------------------------------------------------------------
PORTABLE_AREA_MIN_M2 = 40.0
PORTABLE_AREA_MAX_M2 = 250.0
PORTABLE_ASPECT_MIN = 1.3
PORTABLE_ASPECT_MAX = 4.5
PORTABLE_RECTANGULARITY_MIN = 0.80

# --- Athletics --------------------------------------------------------------
FULL_SIZE_FIELD_MIN_M2 = 90.0 * 45.0   # codebook: a "full-size" field
PITCH_AREA_MAX_M2 = 30000.0            # above this it is a complex, not a field
POOL_AREA_MIN_M2 = 100.0
POOL_AREA_MAX_M2 = 1250.0
TRACK_PERIMETER_M = 400.0              # in-image scale reference

# --- Solar ------------------------------------------------------------------
SOLAR_MASK_THRESHOLD = 0.5
SOLAR_MAX_ROOF_FRACTION = 0.60         # above this, flag as implausible

# --- Fencing ----------------------------------------------------------------
FENCE_FULL_MIN = 0.80                  # fraction of perimeter covered
FENCE_PARTIAL_MIN = 0.20

# --- Imagery geometry -------------------------------------------------------
DEFAULT_TILE_PX = 1024
SCALE_BAR_TOLERANCE = 0.10             # assert measured vs expected within 10%

# --- Fields we measure ------------------------------------------------------
FIELDS = [
    "solar_present",
    "solar_area_m2",
    "portable_count",
    "fencing_extent",
    "fencing_type",
    "track_present",
    "field_count",
    "court_count",
    "pool_present",
]
# Fields scored as binary correct/incorrect for calibration purposes.
BINARY_FIELDS = ["solar_present", "track_present", "pool_present"]
COUNT_FIELDS = ["portable_count", "field_count", "court_count"]
COUNT_TOLERANCE = {"portable_count": 1, "field_count": 1, "court_count": 1}
