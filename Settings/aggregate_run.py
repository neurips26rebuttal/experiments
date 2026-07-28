#!/usr/bin/env python3
"""Aggregate one results tree into files AT ITS ROOT.

    python3 src/aggregate_run.py results

The tree is flat: one directory per run, named by run_dirs.run_dir_name(), plus
one shared runtime/ directory --

    <root>/runtime/runtime_*.json                     raw, source in the json meta
    <root>/<run dir>/<dataset>_results_detailed.json  per-run metrics

so the run's dataset, case, model family and epsilon are recovered from the
directory NAME with run_dirs.parse_run_dir(), the inverse of what wrote it.

This script walks the tree and writes, per dataset, directly in <root>:

    runtime_<dataset>_<accel>.csv / .json    rows = (source, case)  x  phase
    timing_<dataset>_<accel>.csv / .json     rows = (model, case, phase), ms/image
    metrics_<dataset>.csv / .json            rows = (source, case, model) x metric

The accelerator stays in the runtime FILENAME so a100 and h100 numbers can
never be silently mixed. Repeated runs of the same (source, case) sum their
seconds and sample counts; attack_ms_per_sample stays comparable.

Runs made with --timing are kept in a SEPARATE table and excluded from the
runtime_* one. They are batch-size-1 measurements whose seconds mean something
different, and they carry the raw per-image durations, so the timing table
reports a real distribution (mean/median/std/min/max/p95) per source model
rather than a ratio of two totals.

Metric columns are the union over all rows (case4/AutoAttack rows have no
eps_dgf, for instance); a metric a row does not have is left empty, not 0.

The imagenet transferability results.json files have a different shape and
their own aggregator (src/aggregate_results.py); this script only picks up
the *_results_detailed.json summaries.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import attack_name
from run_dirs import parse_run_dir
from runtime_log import stats_ms

PHASES = ("attack", "metrics", "load")

# Preferred metric order; anything not listed is appended alphabetically.
METRIC_ORDER = [
    "clean_accuracy", "adversarial_accuracy", "attack_success_rate",
    "lpips_mean", "lpips_std", "ssim_mean", "ssim_std",
    "psnr_mean", "psnr_std",
    "mean_l2_norm", "std_l2_norm", "mean_linf_norm", "std_linf_norm",
    "eps_dgf", "gabor_frame_norm",
]

def _case_key(name):
    hit = re.match(r"^case(\d+)$", name)
    return (0, int(hit.group(1)), "") if hit else (1, 0, name)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def _iter_runtime_json(root):
    """(path, parsed json, source) for every runtime_*.json under <root>."""
    central = os.path.normpath(os.path.join(root, "runtime"))
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) != "runtime":
            continue
        # New layout: all raw files in <root>/runtime, source recorded in the
        # json meta. Old layout: .../<case_eps>/<source>/runtime/, source is
        # the parent directory. Runs without either (imagenet) get "all".
        parent = os.path.basename(os.path.dirname(dirpath))
        is_central = os.path.normpath(dirpath) == central
        for fname in sorted(files):
            if not (fname.startswith("runtime") and fname.endswith(".json")):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  skipping unreadable {path}: {e}", file=sys.stderr)
                continue
            source = ((d.get("meta") or {}).get("model_source")
                      or ("all" if is_central else parent))
            yield path, d, source


def collect_runtime(root):
    """(ds, accel) -> (source, case) -> summed phases/samples/run count.

    --timing runs are excluded: they are batch-size-1 cost measurements, and
    summing their seconds into a production run's totals would corrupt both.
    They are aggregated separately by collect_timing().
    """
    agg = defaultdict(lambda: defaultdict(lambda: {
        **{p: 0.0 for p in PHASES}, "n_samples": 0, "runs": 0}))
    for _path, d, source in _iter_runtime_json(root):
        if d.get("timing_mode"):
            continue
        ds = d.get("dataset", "unknown")
        accel = d.get("env", {}).get("accelerator", "unknown")
        for method, rec in (d.get("methods") or {}).items():
            if method == "_shared":
                continue
            row = agg[(ds, accel)][(source, method)]
            for p in PHASES:
                row[p] += float(rec.get(p) or 0.0)
            row["n_samples"] += int(rec.get("n_samples") or 0)
            row["runs"] += 1
    return agg


# ---------------------------------------------------------------------------
# Timing (--timing runs only)
# ---------------------------------------------------------------------------

def collect_timing(root):
    """(ds, accel) -> (source_model, case, phase) -> concatenated raw ms.

    Keyed on the SOURCE MODEL, not the model family: at batch size 1 the whole
    point is that a ResNet-18's per-image cost is never averaged into a
    WideResNet's. Raw millisecond lists are concatenated across array tasks and
    the statistics recomputed from the pooled list, so splitting a measurement
    over several jobs gives exactly the same numbers as running it in one.
    """
    agg = defaultdict(lambda: defaultdict(list))
    runs = defaultdict(set)
    for path, d, _source in _iter_runtime_json(root):
        if not d.get("timing_mode"):
            continue
        ds = d.get("dataset", "unknown")
        accel = d.get("env", {}).get("accelerator", "unknown")
        for method, rec in (d.get("methods") or {}).items():
            raw = ((rec.get("per_sample") or {}).get("raw_ms")) or {}
            for phase, by_src in raw.items():
                for src, vals in by_src.items():
                    if not vals:
                        continue
                    agg[(ds, accel)][(src, method, phase)].extend(
                        float(v) for v in vals)
                    runs[(ds, accel, src, method, phase)].add(
                        os.path.basename(path))
    return agg, runs


def write_timing(agg, runs, root):
    cols = ["source_model", "method", "name", "phase", "n_images",
            "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "p95_ms",
            "total_s", "runs"]
    written = []
    for (ds, accel), rows in sorted(agg.items()):
        stem = os.path.join(root, f"timing_{ds}_{accel}")
        keys = sorted(rows, key=lambda k: (_case_key(k[1]), k[2], k[0]))

        table, csv_rows = {}, []
        for src, method, phase in keys:
            vals = rows[(src, method, phase)]
            st = stats_ms(vals)
            rec = {
                "name": attack_name(method),
                **{f"{k}_ms": st[k] for k in
                   ("mean", "median", "std", "min", "max", "p95")},
                "n_images": st["n"],
                "total_s": round(sum(vals) / 1000.0, 6),
                "runs": len(runs[(ds, accel, src, method, phase)]),
            }
            table.setdefault(method, {}).setdefault(phase, {})[src] = rec
            csv_rows.append([src, method, rec["name"], phase, rec["n_images"]]
                            + [f"{rec[f'{k}_ms']:.4f}" for k in
                               ("mean", "median", "std", "min", "max", "p95")]
                            + [f"{rec['total_s']:.6f}", rec["runs"]])

        with open(stem + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(csv_rows)
        with open(stem + ".json", "w") as f:
            json.dump({"dataset": ds, "accelerator": accel,
                       "unit": "milliseconds per image, batch size 1",
                       "methods": table}, f, indent=2)
        written.append((stem, len(keys)))
    return written


def write_runtime(agg, root):
    written = []
    for (ds, accel), rows in sorted(agg.items()):
        stem = os.path.join(root, f"runtime_{ds}_{accel}")
        keys = sorted(rows, key=lambda sc: (sc[0], _case_key(sc[1])))

        table = {}
        for source, case in keys:
            r = rows[(source, case)]
            total = sum(r[p] for p in PHASES)
            n = r["n_samples"]
            # "name" not "attack": the attack PHASE column already owns that key
            table.setdefault(source, {})[case] = {
                "name": attack_name(case),
                **{p: round(r[p], 3) for p in PHASES},
                "total": round(total, 3),
                "n_samples": n,
                "attack_ms_per_sample": (round(1000.0 * r["attack"] / n, 4)
                                         if n else None),
                "runs": r["runs"],
            }

        with open(stem + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "method", "name"] + list(PHASES)
                       + ["total", "n_samples", "attack_ms_per_sample", "runs"])
            for source, case in keys:
                r = table[source][case]
                w.writerow([source, case, r["name"]]
                           + [f"{r[p]:.3f}" for p in PHASES]
                           + [f"{r['total']:.3f}", r["n_samples"],
                              f"{r['attack_ms_per_sample']:.4f}"
                              if r["attack_ms_per_sample"] is not None else "",
                              r["runs"]])

        with open(stem + ".json", "w") as f:
            json.dump({"dataset": ds, "accelerator": accel,
                       "phases": list(PHASES), "sources": table}, f, indent=2)
        written.append((stem, len(keys)))
    return written


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def collect_metrics(root):
    """ds -> list of {source, case, eps_tag, model, **metrics} rows."""
    by_ds = defaultdict(list)
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if not fname.endswith("_results_detailed.json"):
                continue
            ds = fname[: -len("_results_detailed.json")]
            # Flat tree: one run per directory, every field in its NAME.
            # parse_run_dir is the inverse of what the eval scripts wrote.
            run_dir = os.path.basename(dirpath)
            rd = parse_run_dir(run_dir)
            source, case_hint, eps_tag = rd["group"], rd["case"], rd["eps"]
            path = os.path.join(dirpath, fname)
            try:
                with open(path) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  skipping unreadable {path}: {e}", file=sys.stderr)
                continue
            for model, cases in d.items():
                if not isinstance(cases, dict):
                    continue
                for case, metrics in cases.items():
                    if not isinstance(metrics, dict):
                        continue
                    by_ds[ds].append({
                        "source": source, "case": case,
                        "attack": attack_name(case), "eps_tag": eps_tag,
                        "model": model,
                        **{k: v for k, v in metrics.items()},
                    })
                    if case_hint and case != case_hint:
                        print(f"  note: {path} holds {case!r} under a "
                              f"{run_dir!r} directory", file=sys.stderr)
    return by_ds


def write_metrics(by_ds, root):
    written = []
    id_cols = ["source", "case", "attack", "eps_tag", "model"]
    for ds, rows in sorted(by_ds.items()):
        metric_cols = set().union(*(r.keys() for r in rows)) - set(id_cols)
        ordered = ([m for m in METRIC_ORDER if m in metric_cols]
                   + sorted(metric_cols - set(METRIC_ORDER)))
        rows = sorted(rows, key=lambda r: (r["source"], _case_key(r["case"]),
                                           r["model"]))
        stem = os.path.join(root, f"metrics_{ds}")
        with open(stem + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(id_cols + ordered)
            for r in rows:
                w.writerow([r[c] for c in id_cols]
                           + [r.get(m, "") for m in ordered])
        with open(stem + ".json", "w") as f:
            json.dump({"dataset": ds, "columns": id_cols + ordered,
                       "rows": rows}, f, indent=2)
        written.append((stem, len(rows)))
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="results/smoke",
                   help="results tree to aggregate; outputs land directly here")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: {args.root!r} is not a directory", file=sys.stderr)
        return 1

    rt = collect_runtime(args.root)
    tm, tm_runs = collect_timing(args.root)
    mt = collect_metrics(args.root)
    if not rt and not mt and not tm:
        print(f"Nothing to aggregate under {args.root!r}.", file=sys.stderr)
        return 1

    for stem, n in write_runtime(rt, args.root):
        print(f"  wrote {stem}.csv / .json   ({n} source x case rows)")
    for stem, n in write_timing(tm, tm_runs, args.root):
        print(f"  wrote {stem}.csv / .json   ({n} model x case x phase rows, "
              f"ms per image)")
    for stem, n in write_metrics(mt, args.root):
        print(f"  wrote {stem}.csv / .json   ({n} model rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
