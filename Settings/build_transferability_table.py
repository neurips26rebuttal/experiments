#!/usr/bin/env python3
"""Build transferability tables from the results.json tree.

    python3 src/build_transferability_table.py --results-dir results
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dirs import parse_run_path

METRICS = ("attack_success_rate", "adversarial_accuracy", "clean_accuracy",
           "lpips", "ssim", "psnr", "l2_norm", "linf_norm")


def parse_meta(rel):
    """<run directory>/results.json -> its fields.
    """
    return parse_run_path(rel)


def mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("-o", "--out-dir", default="results/tables")
    p.add_argument("--metric", default="attack_success_rate", choices=METRICS,
                   help="metric written into the source x target matrices")
    p.add_argument("--percent", action="store_true",
                   help="scale rate-like metrics by 100")
    args = p.parse_args()

    found = []
    for root, _d, files in os.walk(args.results_dir):
        if "results.json" in files:
            found.append(os.path.join(root, "results.json"))
    if not found:
        print(f"No results.json under {args.results_dir!r}", file=sys.stderr)
        return 1
    print(f"Found {len(found)} results.json file(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    scale = 100.0 if args.percent else 1.0
    summary = []

    for path in sorted(found):
        rel = os.path.relpath(path, args.results_dir)
        meta = parse_meta(rel)
        try:
            with open(path) as f:
                res = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {rel}: {e}", file=sys.stderr)
            continue
        if not isinstance(res, dict) or not res:
            continue

        sources = list(res.keys())
        targets = sorted({t for s in sources for t in (res.get(s) or {})})
        if not targets:
            continue

        diag, off, n_missing = [], [], 0
        rows = []
        for s in sources:
            row = [s]
            for t in targets:
                cell = (res.get(s) or {}).get(t) or {}
                v = cell.get(args.metric)
                if v is None:
                    n_missing += 1
                    row.append("")
                else:
                    v = float(v) * scale
                    row.append(f"{v:.6g}")
                    (diag if s == t else off).append(v)
            rows.append(row)

        tag = "_".join(x for x in (meta["dataset"], meta["case"], meta["group"],
                                   meta["eps"], meta["hparams"]) if x)
        mpath = os.path.join(args.out_dir, f"matrix_{tag}_{args.metric}.csv")
        with open(mpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source\\target"] + targets)
            w.writerows(rows)

        summary.append({
            **meta, "metric": args.metric,
            "n_sources": len(sources), "n_targets": len(targets),
            "white_box": f"{mean(diag):.6g}" if mean(diag) is not None else "",
            "transfer": f"{mean(off):.6g}" if mean(off) is not None else "",
            "n_missing": n_missing,
            "matrix": os.path.basename(mpath),
        })
        print(f"  {tag:<52} {len(sources)}x{len(targets)}"
              + (f"  ({n_missing} missing)" if n_missing else ""))

    spath = os.path.join(args.out_dir, "summary.csv")
    cols = ["dataset", "case", "group", "eps", "hparams", "metric",
            "n_sources", "n_targets", "white_box", "transfer", "n_missing",
            "matrix"]
    with open(spath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {len(summary)} matri{'x' if len(summary)==1 else 'ces'} "
          f"and {spath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())