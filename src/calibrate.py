"""Turn agreement features into calibrated probabilities, and measure whether
they are actually calibrated.

The thing this module exists to avoid: fitting the confidence model on the same
25 schools whose accuracy we then report. That inflates every number. So every
reported figure here comes from **leave-one-school-out cross-validation** -- the
model that scores a school was never trained on it. Grouping is by school, not
by observation, because fields within one school share a campus polygon and an
image and are therefore not independent.

Reported: reliability diagram, ECE, Brier score, per-field accuracy with Wilson
intervals, and a coverage-accuracy curve.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .fuse import FEATURE_NAMES


# --------------------------------------------------------------------------- #
# Scoring a prediction against a label
# --------------------------------------------------------------------------- #
# Fields where "left empty" is an answer rather than a gap. A blank fence
# material means no identifiable material was seen -- which is the usual case,
# since nadir imagery cannot resolve what a fence is made of. Treating blank as
# unlabelled instead would drop almost every school from this field's n.
BLANK_MEANS_NONE = {"fencing_type"}


def _blank_as_none(v: Any) -> str:
    return "none" if v is None or str(v).strip() == "" else str(v).strip().lower()


def is_correct(field: str, predicted: Any, truth: Any) -> Optional[bool]:
    """Whether a prediction counts as correct. None means "not scoreable"."""
    if field in BLANK_MEANS_NONE:
        return _blank_as_none(predicted) == _blank_as_none(truth)
    if truth is None or (isinstance(truth, str) and truth.strip() == ""):
        return None
    if predicted is None:
        return False
    if field in config.BINARY_FIELDS:
        return bool(predicted) == bool(truth)
    if field in config.COUNT_FIELDS:
        try:
            tol = config.COUNT_TOLERANCE.get(field, 1)
            return abs(int(predicted) - int(truth)) <= tol
        except (TypeError, ValueError):
            return False
    if field == "solar_area_m2":
        try:
            p, t = float(predicted), float(truth)
        except (TypeError, ValueError):
            return False
        if t <= 0:
            return p <= 0
        # Areas are judged on ratio, not absolute error: being within a factor
        # of two is the honest resolution of this measurement.
        return 0.5 <= (p / t if p > 0 else 0.0) <= 2.0
    return str(predicted).strip().lower() == str(truth).strip().lower()


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval. With n=25 the bands are wide, and saying so is
    part of the point rather than a caveat to bury."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# The calibrator
# --------------------------------------------------------------------------- #
class Calibrator:
    """Logistic (Platt) mapping from agreement features to P(correct).

    Logistic rather than isotonic by default: with ~200 observations, isotonic
    overfits badly, while a regularised linear model in 12 features is stable
    and its coefficients are interpretable in the memo.
    """

    def __init__(self, C: float = 0.5):
        self.C = C
        self.model = None
        self.feature_names = list(FEATURE_NAMES)
        self.fallback_rate = 0.5

    def _matrix(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([[float(r["features"].get(k, 0.0))
                          for k in self.feature_names] for r in rows],
                        dtype=float)

    def fit(self, rows: List[Dict[str, Any]], labels: List[bool]) -> "Calibrator":
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(labels, dtype=int)
        self.fallback_rate = float(y.mean()) if len(y) else 0.5
        # A degenerate training fold (all right or all wrong) has no gradient to
        # fit; fall back to the base rate rather than emit a fake probability.
        if len(y) < 12 or y.min() == y.max():
            self.model = None
            return self
        X = self._matrix(rows)
        self.model = LogisticRegression(C=self.C, max_iter=2000,
                                        class_weight="balanced")
        self.model.fit(X, y)
        return self

    def predict(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.array([])
        if self.model is None:
            return np.full(len(rows), self.fallback_rate)
        p = self.model.predict_proba(self._matrix(rows))[:, 1]
        # Never emit 0 or 1: we do not have the evidence for certainty.
        return np.clip(p, 0.02, 0.98)

    def coefficients(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {n: round(float(c), 3)
                for n, c in zip(self.feature_names, self.model.coef_[0])}


def leave_one_school_out(rows: List[Dict[str, Any]],
                         labels: List[bool]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Out-of-fold calibrated probabilities, grouped by school.

    Grouping by school (not by row) is essential: fields within a school share a
    campus polygon and an image, so splitting them across folds would leak.
    """
    schools = sorted({r["school_id"] for r in rows})
    out = np.zeros(len(rows))
    for held in schools:
        tr_idx = [i for i, r in enumerate(rows) if r["school_id"] != held]
        te_idx = [i for i, r in enumerate(rows) if r["school_id"] == held]
        if not te_idx:
            continue
        cal = Calibrator().fit([rows[i] for i in tr_idx],
                               [labels[i] for i in tr_idx])
        out[te_idx] = cal.predict([rows[i] for i in te_idx])
    full = Calibrator().fit(rows, labels)
    return out, full.coefficients()


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: Sequence[float], correct: Sequence[bool],
                               bins: int = 5) -> Tuple[float, List[Dict[str, Any]]]:
    """ECE plus the reliability diagram's bins.

    A field marked 0.9 should be right about 90% of the time; ECE is the
    average gap between those two numbers, weighted by how many predictions
    land in each bin.
    """
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, table = 0.0, []
    n = len(probs)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        cnt = int(sel.sum())
        if cnt == 0:
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0,
                          "mean_confidence": None, "observed_accuracy": None})
            continue
        conf = float(probs[sel].mean())
        acc = float(correct[sel].mean())
        lo_ci, hi_ci = wilson_interval(int(correct[sel].sum()), cnt)
        ece += (cnt / n) * abs(acc - conf)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": cnt,
                      "mean_confidence": round(conf, 3),
                      "observed_accuracy": round(acc, 3),
                      "accuracy_ci": [round(lo_ci, 3), round(hi_ci, 3)]})
    return float(ece), table


def brier_score(probs: Sequence[float], correct: Sequence[bool]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((p - y) ** 2))


def coverage_accuracy_curve(probs: Sequence[float], correct: Sequence[bool],
                            steps: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
                            ) -> List[Dict[str, Any]]:
    """Accuracy on the retained fraction after flagging the least confident.

    This is the curve that answers the brief's own framing: right 70% of the
    time and knowing which 30% to flag.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=bool)
    order = np.argsort(-p)
    rows = []
    for cov in steps:
        k = max(1, int(round(cov * len(p))))
        sel = order[:k]
        acc = float(y[sel].mean())
        lo, hi = wilson_interval(int(y[sel].sum()), k)
        rows.append({"coverage": cov, "n": k, "accuracy": round(acc, 3),
                     "accuracy_ci": [round(lo, 3), round(hi, 3)],
                     "min_confidence": round(float(p[sel].min()), 3)})
    return rows


def per_field_accuracy(rows: List[Dict[str, Any]],
                       correct: Sequence[bool]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r, ok in zip(rows, correct):
        d = out.setdefault(r["field"], {"n": 0, "k": 0})
        d["n"] += 1
        d["k"] += int(bool(ok))
    for field, d in out.items():
        lo, hi = wilson_interval(d["k"], d["n"])
        d["accuracy"] = round(d["k"] / d["n"], 3) if d["n"] else None
        d["accuracy_ci"] = [round(lo, 3), round(hi, 3)]
    return out


def evaluate(rows: List[Dict[str, Any]], labels: List[bool]) -> Dict[str, Any]:
    """Full calibration report, entirely out-of-fold."""
    if not rows:
        return {"error": "no scoreable observations"}
    probs, coefs = leave_one_school_out(rows, labels)
    ece, reliability = expected_calibration_error(probs, labels)
    return {
        "n_observations": len(rows),
        "n_schools": len({r["school_id"] for r in rows}),
        "base_accuracy": round(float(np.mean(labels)), 3),
        "ece": round(ece, 4),
        "brier": round(brier_score(probs, labels), 4),
        "reliability": reliability,
        "coverage_accuracy": coverage_accuracy_curve(probs, labels),
        "per_field": per_field_accuracy(rows, labels),
        "coefficients": coefs,
        "validation": "leave-one-school-out cross-validation",
        "out_of_fold_probs": [round(float(p), 4) for p in probs],
    }
