"""Wall-clock accounting for the attack evaluations, shared by both datasets.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import socket
import statistics
import time
from contextlib import contextmanager

PHASES = ("load", "attack", "metrics")


def stats_ms(values):
    """Exact summary of a list of per-sample durations in milliseconds.
    """
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    k = min(n - 1, max(0, int(math.ceil(0.95 * n)) - 1))
    return {
        "n": n,
        "mean": round(sum(s) / n, 4),
        "median": round(statistics.median(s), 4),
        "std": round(statistics.pstdev(s), 4) if n > 1 else 0.0,
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "p95": round(s[k], 4),
    }


def _sync(device):
    """Block until queued CUDA work is done, so elapsed time is real."""
    if device is None or str(device).startswith("cpu"):
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def accelerator_tag():
    """Short tag for the accelerator: 'h100', 'a100', 'v100', ... else 'cpu'.
    """
    name = ""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    hay = " ".join([name,
                    os.environ.get("SLURM_JOB_CONSTRAINT", ""),
                    os.environ.get("SLURM_JOB_PARTITION", "")]).lower()
    for tag in ("h200", "h100", "a100", "v100", "l40s", "a40", "rtx8000", "p100"):
        if tag in hay.replace(" ", ""):
            return tag
    if name:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24]
    return "cpu"


class RuntimeLog:
    """Accumulates per-(method, phase) seconds and writes one JSON per run."""

    def __init__(self, dataset, device="cuda", meta=None, timing=False):
        self.dataset = dataset
        self.device = device
        self.t0 = time.perf_counter()
        self.records = {}          # method -> {phase -> seconds}
        self.samples = {}          # method -> n samples attacked
        self.records_src = {}      # method -> phase -> source -> seconds
        self.samples_src = {}      # method -> source -> n samples
        self.per_sample = {}
        self.per_batch = {}
        self.timing = bool(timing)
        self.meta = dict(meta or {})
        self.env = {
            "accelerator": accelerator_tag(),
            "gpu_name": self._gpu_name(),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_constraint": os.environ.get("SLURM_JOB_CONSTRAINT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
        }
        try:
            import torch
            self.env["torch"] = torch.__version__
        except Exception:
            pass

    @staticmethod
    def _gpu_name():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return None

    @contextmanager
    def phase(self, method, name, per_sample=False, source=None, n=None):
        """Time a phase. Re-entrant across calls: seconds accumulate."""
        if name not in PHASES:
            raise ValueError(f"unknown phase {name!r}, expected one of {PHASES}")
        _sync(self.device)
        t = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            dt = time.perf_counter() - t
            self.records.setdefault(method, {}).setdefault(name, 0.0)
            self.records[method][name] += dt
            if source:
                by = (self.records_src.setdefault(method, {})
                      .setdefault(name, {}))
                by[source] = by.get(source, 0.0) + dt
            if per_sample:
                (self.per_sample.setdefault(method, {}).setdefault(name, {})
                 .setdefault(source or "all", []).append(1000.0 * dt))
            if n:
                (self.per_batch.setdefault(method, {}).setdefault(name, {})
                 .setdefault(source or "all", []).append(1000.0 * dt / int(n)))

    def add_samples(self, method, n, source=None):
        """Count attacked samples; `source` also credits them to that model.
        """
        self.samples[method] = self.samples.get(method, 0) + int(n)
        if source:
            by = self.samples_src.setdefault(method, {})
            by[source] = by.get(source, 0) + int(n)

    def discard_warmup(self, method, source=None):
        """Drop everything recorded so far for (method, source): the warm-up.
        """
        key = source or "all"
        dropped = 0
        for by_src in (self.per_batch.get(method) or {}).values():
            by_src.pop(key, None)
        for ph, by_src in (self.per_sample.get(method) or {}).items():
            vals = by_src.pop(key, [])
            if not vals:
                continue
            self.records[method][ph] = max(
                0.0, self.records[method][ph] - sum(vals) / 1000.0)
            src_ph = (self.records_src.get(method) or {}).get(ph)
            if src_ph and key in src_ph:
                src_ph[key] = max(0.0, src_ph[key] - sum(vals) / 1000.0)
            if ph == "attack":
                dropped = len(vals)
        self.samples[method] = max(0, self.samples.get(method, 0) - dropped)
        src_n = self.samples_src.get(method) or {}
        if key in src_n:
            src_n[key] = max(0, src_n[key] - dropped)
        return dropped

    def _per_sample_block(self, method):
        """{phase: overall stats, by_source: {phase: {src: stats}}} or None."""
        ps = self.per_sample.get(method) or {}
        ps = {ph: {s: v for s, v in by_src.items() if v}
              for ph, by_src in ps.items()}
        ps = {ph: by_src for ph, by_src in ps.items() if by_src}
        if not ps:
            return None
        return {
            "overall": {ph: stats_ms([v for vals in by_src.values() for v in vals])
                        for ph, by_src in sorted(ps.items())},
            "by_source": {ph: {s: stats_ms(v) for s, v in sorted(by_src.items())}
                          for ph, by_src in sorted(ps.items())},
            "raw_ms": {ph: {s: [round(v, 4) for v in vals]
                            for s, vals in sorted(by_src.items())}
                       for ph, by_src in sorted(ps.items())},
        }

    def _by_source_block(self, method):
        """{source: {phase seconds, n_samples, attack_ms_per_sample}} or None.
        """
        phases = self.records_src.get(method) or {}
        counts = self.samples_src.get(method) or {}
        names = sorted(set(counts) | {s for by in phases.values() for s in by})
        if not names:
            return None
        out = {}
        for s in names:
            n = int(counts.get(s, 0))
            rec = {p: round((phases.get(p) or {}).get(s, 0.0), 6)
                   for p in PHASES}
            rec["n_samples"] = n
            rec["attack_ms_per_sample"] = (round(1000.0 * rec["attack"] / n, 4)
                                           if n else None)
            vals = (((self.per_batch.get(method) or {}).get("attack") or {})
                    .get(s))
            if vals:
                rec["n_batches"] = len(vals)
                rec["ms_per_sample_stats"] = stats_ms(vals)
            out[s] = rec
        return out

    def as_dict(self):
        total_wall = time.perf_counter() - self.t0
        methods = {}
        for m, ph in self.records.items():
            known = sum(ph.get(p, 0.0) for p in PHASES)
            n = self.samples.get(m, 0)
            methods[m] = {
                **{p: round(ph.get(p, 0.0), 6) for p in PHASES},
                "measured": round(known, 6),
                "n_samples": n,
                "attack_ms_per_sample": (round(1000.0 * ph.get("attack", 0.0) / n, 4)
                                         if n else None),
            }
            block = self._per_sample_block(m)
            if block:
                methods[m]["per_sample"] = block
            src = self._by_source_block(m)
            if src:
                methods[m]["by_source"] = src
        return {
            "dataset": self.dataset,
            "env": self.env,
            "meta": self.meta,
            "timing_mode": self.timing,
            "total_wall_s": round(total_wall, 6),
            "other_s": round(max(0.0, total_wall - sum(
                sum(p.get(k, 0.0) for k in PHASES) for p in self.records.values())), 6),
            "methods": methods,
        }

    def write(self, out_dir, filename="runtime.json"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
        return path

    def print_summary(self):
        d = self.as_dict()
        print(f"\n  Runtime [{d['dataset']} on {d['env']['accelerator']}] "
              f"total {d['total_wall_s']:.1f}s"
              f"{'  [--timing: batch size 1]' if d['timing_mode'] else ''}")
        for m, r in sorted(d["methods"].items()):
            per = f"{r['attack_ms_per_sample']:.1f} ms/sample" if r["attack_ms_per_sample"] else "-"
            print(f"    {m:<8} load {r['load']:7.1f}s  attack {r['attack']:8.1f}s  "
                  f"metrics {r['metrics']:7.1f}s   ({per})")
            for src, rec in sorted((r.get("by_source") or {}).items()):
                st = rec.get("ms_per_sample_stats")
                if not st:
                    continue
                print(f"      [{src:<28}] n={rec['n_samples']:<5} "
                      f"batches={st['n']:<4} median {st['median']:8.3f}  "
                      f"mean {st['mean']:8.3f}  std {st['std']:7.3f}  "
                      f"min {st['min']:7.3f}  max {st['max']:8.3f}  ms/sample")
            block = r.get("per_sample")
            if not block:
                continue
            for ph, st in block["overall"].items():
                print(f"      per-image {ph:<7} n={st['n']:<5} mean {st['mean']:8.3f}  "
                      f"median {st['median']:8.3f}  std {st['std']:7.3f}  "
                      f"min {st['min']:7.3f}  max {st['max']:8.3f}  "
                      f"p95 {st['p95']:8.3f}  ms")
            for src, st in block["by_source"].get("attack", {}).items():
                print(f"        attack [{src}] n={st['n']:<5} "
                      f"mean {st['mean']:8.3f}  median {st['median']:8.3f}  "
                      f"std {st['std']:7.3f}  ms/image")