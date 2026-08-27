"""Write measurements.csv and metrics.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config

# Confidence at or below which a row is flagged for human review. Abstention is
# a feature: the brief explicitly prefers a pipeline that knows what it does not
# know over one that is more often right but silent about its failures.
FLAG_THRESHOLD = 0.55

CSV_COLUMNS = (
    ["school_id", "school_name"]
    + [c for f in config.FIELDS for c in (f, f + "_confidence")]
    + ["solar_present_t0", "solar_area_m2_t0", "imagery_date_t0",
       "solar_present_t1", "solar_area_m2_t1", "imagery_date_t1",
       "solar_change_m2", "solar_change_plausible",
       "imagery_date", "imagery_source", "imagery_m_per_px",
       "campus_method", "campus_confidence", "campus_area_m2", "campus_flags",
       "n_sources_median", "flag_for_review", "failure_reason"]
)


def _imagery_date(manifest: Dict[str, Any]) -> Any:
    """Prefer a NAIP acquisition date; fall back to a Wayback release date.

    These are different things and the distinction is reported rather than
    smoothed over: NAIP gives the date the imagery was *flown*, Wayback gives
    the date a basemap version was *published*.
    """
    naip = (manifest.get("naip") or {}).get("dates") or []
    if naip:
        return naip[-1].get("date"), "naip_acquisition"
    wb = manifest.get("wayback_new") or {}
    if wb.get("date"):
        return wb["date"], "wayback_release"
    return None, None


def _solar_change(row: Dict[str, Any]) -> Dict[str, Any]:
    """Change in panel area between the two dated observations, with a physical
    plausibility check.

    Installed PV is essentially never removed, so a large *decrease* is far more
    likely to be detector noise (shadow, wet asphalt, seasonal lighting) than a
    real decommissioning. Reporting such a change without flagging it would be
    presenting a measurement artefact as a finding, so the direction of change
    is checked rather than just its magnitude.
    """
    a0, a1 = row.get("solar_area_m2_t0"), row.get("solar_area_m2_t1")
    if a0 is None or a1 is None:
        return {"solar_change_m2": None, "solar_change_plausible": None}
    delta = float(a1) - float(a0)
    plausible = True
    if float(a0) > 0 and delta < 0 and abs(delta) / float(a0) > 0.5:
        plausible = False        # a >50% loss of installed PV is not credible
    return {"solar_change_m2": round(delta, 1),
            "solar_change_plausible": plausible}


def write_measurements(fused_rows: List[Dict[str, Any]],
                       probabilities: Optional[Dict[tuple, float]] = None,
                       out_path: Optional[Path] = None) -> Path:
    out_path = Path(out_path) if out_path else config.OUT / "measurements.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = probabilities or {}

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in fused_rows:
            sid = row["school_id"]
            fused = row["fused"]
            cam = row.get("campus", {})
            man = row.get("manifest", {})
            rec: Dict[str, Any] = {"school_id": sid,
                                   "school_name": row.get("school_name")}
            confs, nsrc = [], []
            for field in config.FIELDS:
                f = fused.get(field)
                if not f:
                    rec[field] = None
                    rec[field + "_confidence"] = None
                    continue
                rec[field] = f["value"]
                c = probabilities.get((sid, field))
                if c is None:
                    # No calibration fitted yet: fall back to the raw agreement
                    # signal, and say so rather than inventing a number.
                    c = f["features"].get("agreement", 0.0) * f["features"].get(
                        "campus_confidence", 1.0)
                rec[field + "_confidence"] = round(float(c), 3)
                if f["value"] is not None:
                    confs.append(float(c))
                nsrc.append(f["n_sources"])

            date, kind = _imagery_date(man)
            ov = man.get("overview") or {}
            rec["imagery_date"] = date
            rec["imagery_source"] = kind or ov.get("source")
            rec["imagery_m_per_px"] = ov.get("m_per_px")
            for slot, key in (("t0", "wayback_old"), ("t1", "wayback_new")):
                wb = man.get(key) or {}
                rec[f"imagery_date_{slot}"] = wb.get("date")
                rec[f"solar_present_{slot}"] = row.get(f"solar_present_{slot}")
                rec[f"solar_area_m2_{slot}"] = row.get(f"solar_area_m2_{slot}")
            rec.update(_solar_change(row))

            rec["campus_method"] = cam.get("campus_method")
            rec["campus_confidence"] = cam.get("campus_confidence")
            rec["campus_area_m2"] = cam.get("campus_area_m2")
            rec["campus_flags"] = cam.get("campus_flags")
            rec["n_sources_median"] = int(np.median(nsrc)) if nsrc else 0
            mean_conf = float(np.mean(confs)) if confs else 0.0
            rec["flag_for_review"] = bool(mean_conf < FLAG_THRESHOLD or not confs)
            missing = [f for f in config.FIELDS if rec.get(f) is None]
            rec["failure_reason"] = (
                "no source could measure: " + ",".join(missing) if missing else "")
            w.writerow(rec)
    return out_path


def write_metrics(report: Dict[str, Any], extra: Optional[Dict[str, Any]] = None,
                  out_path: Optional[Path] = None) -> Path:
    out_path = Path(out_path) if out_path else config.OUT / "metrics.json"
    payload = dict(report)
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def reliability_plot(report: Dict[str, Any],
                     out_path: Optional[Path] = None) -> Optional[Path]:
    """Reliability diagram plus the coverage-accuracy curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rel = [b for b in report.get("reliability", []) if b.get("n")]
    if not rel:
        return None
    out_path = Path(out_path) if out_path else config.OUT / "reliability.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    xs = [b["mean_confidence"] for b in rel]
    ys = [b["observed_accuracy"] for b in rel]
    ns = [b["n"] for b in rel]
    ax1.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect calibration")
    ax1.scatter(xs, ys, s=[20 + 6 * n for n in ns], color="#2a6", zorder=3)
    for b in rel:
        lo, hi = b.get("accuracy_ci", [None, None])
        if lo is not None:
            ax1.plot([b["mean_confidence"]] * 2, [lo, hi], color="#2a6", lw=1.4)
    ax1.set_xlabel("predicted confidence")
    ax1.set_ylabel("observed accuracy")
    ax1.set_title(f"Reliability  (ECE={report.get('ece')}, Brier={report.get('brier')})")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.legend(fontsize=8)

    cov = report.get("coverage_accuracy", [])
    if cov:
        cx = [c["coverage"] for c in cov]
        cy = [c["accuracy"] for c in cov]
        ax2.plot(cx, cy, "-o", color="#36c")
        for c in cov:
            ax2.plot([c["coverage"]] * 2, c["accuracy_ci"], color="#36c", lw=1.2)
        ax2.set_xlabel("coverage (fraction retained)")
        ax2.set_ylabel("accuracy on retained")
        ax2.set_title("Accuracy vs coverage after flagging")
        ax2.invert_xaxis()
    fig.suptitle("Out-of-fold (leave-one-school-out)", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
