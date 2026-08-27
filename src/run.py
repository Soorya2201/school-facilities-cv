"""Command-line entry point.

    python -m src.run stage-a          # fetch + deterministic extraction (needs net)
    python -m src.run label            # build the blind labelling tool
    python -m src.run stage-c          # fuse + calibrate + write outputs
    python -m src.run all --offline    # reproduce everything from the cache

`--offline` is the default path in the README: it needs no API key, no GPU and
no network, and reproduces measurements.csv from the committed cache. That is
the reviewer's first command, so it is the one that must never break.
"""
from __future__ import annotations

import argparse
import json
import sys


def cmd_stage_a(args) -> int:
    from .pipeline import stage_a
    info = stage_a(offline=args.offline, limit=args.limit,
                   use_api=not args.no_api)
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_label(args) -> int:
    from .labels import build_label_tool, intra_rater_agreement
    if args.agreement:
        print(json.dumps(intra_rater_agreement(), indent=2))
        return 0
    path = build_label_tool()
    print(f"labelling tool: {path}")
    print("Open it in a browser, label every school, then click "
          "'download truth.csv' and save it to labels/truth.csv")
    print("Label BEFORE running stage-c, so the labels stay blind.")
    return 0


def cmd_stage_c(args) -> int:
    from .pipeline import stage_c
    from .report import reliability_plot, write_measurements, write_metrics
    from .labels import intra_rater_agreement
    from . import tier2_api

    result = stage_c(getattr(args, 'truth', None))
    report = result["report"]

    probs = {}
    if "out_of_fold_probs" in report:
        # Re-derive the (school, field) keys in the same order evaluate() used.
        from .calibrate import is_correct
        from .labels import load_truth
        truth = load_truth(getattr(args, 'truth', None))
        keys = []
        for row in result["fused"]:
            t = truth.get(row["school_id"], {})
            for field, f in row["fused"].items():
                if is_correct(field, f["value"], t.get(field)) is not None:
                    keys.append((row["school_id"], field))
        probs = dict(zip(keys, report["out_of_fold_probs"]))

    csv_path = write_measurements(result["fused"], probs)
    metrics_path = write_metrics(report, {
        "n_labeled_schools": result["n_labeled_schools"],
        "intra_rater": intra_rater_agreement(),
        "api_status": tier2_api.status(),
    })
    plot = reliability_plot(report)

    print(f"wrote {csv_path}")
    print(f"wrote {metrics_path}")
    if plot:
        print(f"wrote {plot}")
    if "error" in report:
        print(f"\nNOTE: {report['error']}")
        print("Confidences fall back to the raw agreement signal until labels exist.")
    else:
        print(f"\nECE={report['ece']}  Brier={report['brier']}  "
              f"base accuracy={report['base_accuracy']}  "
              f"({report['n_observations']} obs / {report['n_schools']} schools, "
              f"{report['validation']})")
    return 0


def cmd_all(args) -> int:
    rc = cmd_stage_a(args)
    return rc or cmd_stage_c(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="src.run",
                                 description="School facilities measurement pipeline")
    ap.add_argument("--offline", action="store_true",
                    help="use only the committed cache: no network, no keys")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N schools (for smoke tests)")
    ap.add_argument("--no-api", action="store_true",
                    help="skip the free-tier API layer entirely")
    ap.add_argument("--truth", default=None,
                    help="alternate label CSV (default: labels/truth.csv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage-a", help="fetch imagery/OSM and run L0 extractors")
    p_label = sub.add_parser("label", help="build the blind labelling tool")
    p_label.add_argument("--agreement", action="store_true",
                         help="report intra-rater agreement instead")
    sub.add_parser("stage-c", help="fuse, calibrate and write outputs")
    sub.add_parser("all", help="stage-a then stage-c")

    args = ap.parse_args(argv)
    return {"stage-a": cmd_stage_a, "label": cmd_label,
            "stage-c": cmd_stage_c, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
