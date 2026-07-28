"""
Aggregate all results.json files under a results directory into a single CSV.

eval_imagenet.py writes one results.json per (group, case, epsilon, hparams)
combination, laid out as:

    <root>/transferability/<group>/<case>/<eps_tag>/<hp...>/results.json

where <hp...> is a nested path whose shape depends on the case:

    case1          a{a}_b{b}/{window}/gamma{g}/steps{n}
    case2, case3   steps{n}
    case4          aa_{version}_{norm}

Each results.json holds a nested {source_model: {target_model: {metric: value}}}
mapping, so this script emits ONE ROW PER (source, target) PAIR.

If an attack_stats.json sits next to results.json (case 1 only) its per-source
Gabor-frame diagnostics are merged onto every row for that source.

Usage:
    python3 src/aggregate_results.py
    python3 src/aggregate_results.py --results-dir ./results/imagenet
    python3 src/aggregate_results.py --results-dir ./results/imagenet -o summary.csv
"""

import argparse
import csv
import json
import os
import re


META_COLS = [
    "path", "group", "case", "epsilon", "eps_tag",
    "a", "b", "window", "gamma", "steps", "aa_version", "aa_norm",
    "source_model", "target_model", "is_self",
]

_RE_AB = re.compile(r"^a(?P<a>[^_]+)_b(?P<b>.+)$")
_RE_GAMMA = re.compile(r"^gamma(?P<gamma>.+)$")
_RE_STEPS = re.compile(r"^steps(?P<steps>\d+)$")
_RE_AA = re.compile(r"^aa_(?P<aa_version>[^_]+)_(?P<aa_norm>[^_]+)$")
_RE_EPS = re.compile(r"^(?P<eps>[0-9.]+)div255$")
_WINDOWS = {"Hann", "Blackman", "Gaussian"}


def parse_path_metadata(rel_path):
    """Pull group/case/epsilon/hparams out of the directory layout."""
    meta = {k: "" for k in META_COLS}
    meta["path"] = rel_path

    parts = [p for p in rel_path.split(os.sep) if p and p != "."]
    if parts and parts[0] == "transferability":
        parts = parts[1:]

    if parts:
        meta["group"] = parts[0]
    if len(parts) > 1:
        meta["case"] = parts[1]
    if len(parts) > 2:
        meta["eps_tag"] = parts[2]
        m = _RE_EPS.match(parts[2])
        if m:
            meta["epsilon"] = str(float(m.group("eps")) / 255.0)

    for part in parts[3:]:
        if part in _WINDOWS:
            meta["window"] = part
            continue
        for rx in (_RE_AB, _RE_GAMMA, _RE_STEPS, _RE_AA):
            m = rx.match(part)
            if m:
                meta.update(m.groupdict())
                break
    return meta


def load_attack_stats(root):
    path = os.path.join(root, "attack_stats.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description="Aggregate JSON results into one CSV")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Root results directory")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: <results-dir>/aggregated.csv)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or os.path.join(results_dir, "aggregated.csv")

    rows = []
    files_seen = 0

    for root, _, files in os.walk(results_dir):
        if "results.json" not in files:
            continue
        files_seen += 1
        fpath = os.path.join(root, "results.json")
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping unreadable {fpath}: {e}")
            continue

        meta = parse_path_metadata(os.path.relpath(root, results_dir))
        stats = load_attack_stats(root)

        for source, targets in data.items():
            if not isinstance(targets, dict):
                continue
            for target, metrics in targets.items():
                if not isinstance(metrics, dict):
                    continue
                row = dict(meta)
                row["source_model"] = source
                row["target_model"] = target
                row["is_self"] = int(source == target)
                for k, v in metrics.items():
                    row[k] = v
                for k, v in (stats.get(source) or {}).items():
                    row[f"attack_{k}"] = v
                rows.append(row)

    if not rows:
        print(f"No usable results.json files found under {results_dir} "
              f"({files_seen} file(s) scanned)")
        return

    metric_cols = sorted({k for r in rows for k in r if k not in META_COLS})
    all_cols = META_COLS + metric_cols

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (
            str(r["case"]), str(r["group"]), str(r["eps_tag"]),
            str(r.get("gamma", "")), str(r.get("steps", "")),
            str(r["source_model"]), str(r["target_model"]),
        )):
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows ({files_seen} results.json file(s), "
          f"{len(metric_cols)} metric columns) to {output_path}")


if __name__ == "__main__":
    main()
