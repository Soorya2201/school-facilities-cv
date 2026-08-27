"""Tests for the invariants that fail *silently* if broken.

The priority here is not coverage. It is the small number of errors that would
corrupt every measurement without raising anything: the metres-per-pixel
conversion, the bbox/pixel aspect agreement, and the scoring rules that decide
whether a prediction counts as correct.

Run:  python -m pytest tests/ -q      (or: python tests/test_pipeline.py)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.calibrate import (Calibrator, coverage_accuracy_curve,
                           expected_calibration_error, is_correct,
                           wilson_interval)
from src.fuse import FEATURE_NAMES, fuse_field
from src.geo import (bbox_from_center, bbox_meters_per_pixel, haversine_m,
                     meters_per_pixel, min_rotated_rect, point_in_polygon,
                     polygon_area_m2, polygon_perimeter_m, rectangularity,
                     square_bbox, validate_scale_bar)
from src.resolve import name_similarity


def test_meters_per_pixel_matches_known_values():
    # Web Mercator ground resolution at zoom 0 on the equator.
    assert abs(meters_per_pixel(0, 0) - 156543.03392) < 0.01
    # Retina tiles halve the ground size of a pixel. Getting this factor wrong
    # scales every area by 4 with no error raised.
    assert abs(meters_per_pixel(34, 19, scale=2)
               - meters_per_pixel(34, 19) / 2) < 1e-9


def test_bbox_resolution_matches_requested_extent():
    bbox = square_bbox(bbox_from_center(34.0612, -118.4115, 250))  # 500 m box
    mpp = bbox_meters_per_pixel(bbox, 1024, 1024)
    assert abs(mpp - 500 / 1024) < 0.01


def test_non_square_bbox_in_square_image_is_rejected():
    """The guard that caught a real bug: a 317x277 m campus rendered into a
    square image has different m/px per axis, so every area would be wrong."""
    bbox = [-121.37391, 36.82795, -121.37036, 36.83043]
    try:
        bbox_meters_per_pixel(bbox, 1024, 1024)
    except ValueError:
        return
    raise AssertionError("aspect mismatch was not detected")


def test_square_bbox_makes_ground_extent_square():
    bbox = square_bbox([-121.37391, 36.82795, -121.37036, 36.83043])
    lat_mid = (bbox[1] + bbox[3]) / 2
    lon_mid = (bbox[0] + bbox[2]) / 2
    w = haversine_m(bbox[0], lat_mid, bbox[2], lat_mid)
    h = haversine_m(lon_mid, bbox[1], lon_mid, bbox[3])
    assert abs(w - h) / max(w, h) < 0.02


def test_scale_bar_rejects_a_doubled_scale():
    # The failure mode this exists to catch: geometry off by exactly 2x.
    ok, _ = validate_scale_bar(800.0, config.TRACK_PERIMETER_M, 0.10, "track")
    assert not ok
    ok, _ = validate_scale_bar(398.0, config.TRACK_PERIMETER_M, 0.10, "track")
    assert ok


def test_polygon_area_and_perimeter_are_metric():
    # ~100 m square at the equator, where 1 degree ~ 111.32 km.
    d = 100.0 / 111320.0
    ring = [(0.0, 0.0), (d, 0.0), (d, d), (0.0, d)]
    assert abs(polygon_area_m2(ring) - 10000) < 100
    assert abs(polygon_perimeter_m(ring) - 400) < 5


def test_rectangularity_separates_boxes_from_l_shapes():
    d = 100.0 / 111320.0
    box = [(0.0, 0.0), (d, 0.0), (d, d), (0.0, d)]
    l_shape = [(0.0, 0.0), (d, 0.0), (d, d / 2), (d / 2, d / 2),
               (d / 2, d), (0.0, d)]
    assert rectangularity(box) > 0.98
    assert rectangularity(l_shape) < 0.85


def test_min_rotated_rect_is_orientation_invariant():
    d = 100.0 / 111320.0
    axis_aligned = [(0.0, 0.0), (2 * d, 0.0), (2 * d, d), (0.0, d)]
    lo, sh = min_rotated_rect(axis_aligned)
    theta = math.radians(37)
    rotated = [(x * math.cos(theta) - y * math.sin(theta),
                x * math.sin(theta) + y * math.cos(theta))
               for x, y in axis_aligned]
    lo2, sh2 = min_rotated_rect(rotated)
    assert abs(lo - lo2) / lo < 0.05 and abs(sh - sh2) / sh < 0.05


def test_point_in_polygon():
    ring = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    assert point_in_polygon((0.0, 0.0), ring)
    assert not point_in_polygon((2.0, 0.0), ring)


def test_name_matching_separates_the_right_school_from_a_neighbour():
    assert name_similarity("Beverly Hills High", "Beverly Hills High School") > 0.9
    assert name_similarity("Beverly Hills High", "Good Shepherd Catholic School") < 0.3
    assert name_similarity("HILLDALE LOWER ES",
                           "Hilldale Lower Elementary School") > 0.9


def test_scoring_rules():
    assert is_correct("pool_present", True, True)
    assert not is_correct("pool_present", True, False)
    # Counts carry an explicit tolerance: +/-1 is the honest resolution.
    assert is_correct("portable_count", 4, 5)
    assert not is_correct("portable_count", 4, 7)
    # Areas are judged by ratio, not absolute difference.
    assert is_correct("solar_area_m2", 900.0, 1200.0)
    assert not is_correct("solar_area_m2", 100.0, 1200.0)
    # An unlabelled field is not scoreable, which is different from being wrong.
    assert is_correct("pool_present", True, None) is None
    # A field nothing could measure is wrong, not excused.
    assert is_correct("pool_present", None, True) is False


def test_agreement_raises_confidence_features():
    agree = [{"source": "osm_vector", "value": True, "osm_present": True},
             {"source": "owlv2", "value": True, "score": 0.9},
             {"source": "cv_color", "value": True, "score": 0.7}]
    split = [{"source": "osm_vector", "value": False, "osm_present": False},
             {"source": "owlv2", "value": True, "score": 0.5},
             {"source": "cv_color", "value": False, "score": 0.3}]
    a = fuse_field("pool_present", agree, 0.95)
    b = fuse_field("pool_present", split, 0.95)
    assert a["features"]["agreement"] > b["features"]["agreement"]
    assert set(a["features"]) == set(FEATURE_NAMES)


def test_losing_a_source_lowers_evidence_rather_than_hiding_it():
    """A provider going down must reduce confidence, not silently keep it."""
    many = [{"source": "osm_vector", "value": True, "osm_present": True},
            {"source": "owlv2", "value": True, "score": 0.9},
            {"source": "vlm", "value": True, "score": 0.8}]
    few = many[:1]
    assert (fuse_field("pool_present", many, 0.9)["features"]["n_sources"]
            > fuse_field("pool_present", few, 0.9)["features"]["n_sources"])


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(18, 25)
    assert lo < 18 / 25 < hi
    assert (hi - lo) > 0.2          # n=25 really is this uncertain
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_calibrator_never_claims_certainty():
    rows = [{"school_id": f"S{i}", "field": "f",
             "features": {k: (0.9 if i % 2 else 0.1) for k in FEATURE_NAMES}}
            for i in range(40)]
    labels = [bool(i % 2) for i in range(40)]
    probs = Calibrator().fit(rows, labels).predict(rows)
    assert probs.min() >= 0.02 and probs.max() <= 0.98


def test_degenerate_fold_falls_back_to_base_rate():
    """An all-correct training fold has no gradient; emit the base rate rather
    than a fabricated probability."""
    rows = [{"school_id": f"S{i}", "field": "f",
             "features": {k: 0.5 for k in FEATURE_NAMES}} for i in range(20)]
    cal = Calibrator().fit(rows, [True] * 20)
    assert cal.model is None
    assert abs(cal.predict(rows)[0] - 1.0) < 1e-9


def test_ece_is_zero_for_perfect_calibration():
    probs = [0.1] * 100 + [0.9] * 100
    correct = [False] * 90 + [True] * 10 + [False] * 10 + [True] * 90
    ece, table = expected_calibration_error(probs, correct, bins=5)
    assert ece < 0.02
    assert sum(b["n"] for b in table) == 200


def test_coverage_curve_is_ordered_by_confidence():
    probs = [i / 100 for i in range(100)]
    correct = [p > 0.5 for p in probs]
    curve = coverage_accuracy_curve(probs, correct)
    assert curve[0]["coverage"] == 1.0
    # Dropping the least confident half must not lower accuracy.
    assert curve[-1]["accuracy"] >= curve[0]["accuracy"]


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
