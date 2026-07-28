"""Wall-clock accounting for the attack evaluations, shared by both datasets.

Imported by eval_imagenet.py and eval_cifar100.py so their cost numbers are
measured the same way and land in the same schema. bin/build_cost_matrix.py turns
the emitted JSON into one dataset x phase matrix per method, with the accelerator
in the filename.

Phases are deliberately coarse and non-overlapping:

    load     building/loading models and the Gabor operators
    attack   generating adversarial examples -- the number the paper reports
    metrics  LPIPS/SSIM/PSNR and the transferability sweep
    other    total minus the three above (I/O, checkpointing, image dumps)

CUDA is asynchronous, so every phase boundary synchronises. Without that the
timings measure kernel *launch*, not kernel *execution*, and the attack phase
looks ~free while some later phase absorbs its cost.

Both eval scripts bracket the SAME work: only the attacker call itself is
inside the `attack` phase. Data loading, host<->device copies and the
concatenation of the results are outside it, on both datasets, so the two
per-sample columns are directly comparable.

--timing mode (per_sample=True) additionally records the duration of every
individual bracket instead of only the running total. It is meant to be used
with batch size 1, where one bracket is exactly one image, and it turns
attack_ms_per_sample from a ratio of two aggregates into a measured
distribution: mean, median, std, min, max and p95, broken down per source
model, with the raw list kept so downstream tools can recompute anything.

Passing `n=` to phase() records the SAME bracket normalised to ms per sample
(dt/n), in every mode. That is what gives an ordinary batched run a median and
a standard deviation instead of the single ratio attack_ms_per_sample: at batch
size 20 over 200 samples there are ten brackets, so ten measurements of the
per-sample cost. It lands in by_source[src]["ms_per_sample_stats"] and is
DELIBERATELY not folded into the per_sample block above -- bin/report_cost.py
and src/aggregate_run.py read per_sample.raw_ms as per-IMAGE durations without
checking timing_mode, so putting per-batch numbers there would silently report
a batch's milliseconds as an image's.
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

    Nearest-rank percentile, population (not sample) standard deviation: these
    are the timings of every sample that ran, not a sample drawn from a larger
    population, so there is nothing to correct for.
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

    Read from the device name first because that is ground truth for what the
    job actually landed on; SLURM_JOB_CONSTRAINT/partition are only what was
    *requested* and can differ (e.g. gpu_p6 hosts more than one card type).
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
        # The same seconds and counts, kept per SOURCE MODEL. Filled in every
        # mode, not just --timing: a run over a roster of N models attacks each
        # of them in its own bracket, so the split already exists and throwing
        # it away was what forced one SLURM job per model to get a per-model
        # cost. Totals above are unchanged -- these are a decomposition of them.
        self.records_src = {}      # method -> phase -> source -> seconds
        self.samples_src = {}      # method -> source -> n samples
        # method -> phase -> source model -> [ms, ...]; only filled in --timing
        self.per_sample = {}
        # The same brackets normalised to ms PER SAMPLE (dt/n), filled whenever
        # phase() is given n=. Separate from per_sample because that block is
        # read elsewhere as per-image durations; these are per-batch means.
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
        """Time a phase. Re-entrant across calls: seconds accumulate.

        per_sample -- also keep this individual bracket's duration, not just
        add it to the running total. Only meaningful when the bracket holds
        exactly ONE sample, which is what --timing (batch size 1) guarantees;
        the eval scripts pass per_sample=args.timing so a normal run behaves
        and writes exactly as before.
        source -- which model produced the sample, so the per-image cost of a
        ResNet-18 is never silently averaged into a WideResNet's.
        n -- how many samples this bracket covers. Records dt/n, giving the
        per-sample cost a spread (median/std) in ordinary batched runs rather
        than the single aggregate ratio. Pass the batch size.
        """
        if name not in PHASES:
            raise ValueError(f"unknown phase {name!r}, expected one of {PHASES}")
        _sync(self.device)
        t = time.perf_counter()
        try:
            yield
        finally:
            # timed even when the body raises, so a crashed run still accounts
            # for the work it did rather than silently reporting zero
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

        Without the source the run can only ever report a roster average --
        seconds are per model but the denominator would not be, so the two
        could not be divided.
        """
        self.samples[method] = self.samples.get(method, 0) + int(n)
        if source:
            by = self.samples_src.setdefault(method, {})
            by[source] = by.get(source, 0) + int(n)

    def discard_warmup(self, method, source=None):
        """Drop everything recorded so far for (method, source): the warm-up.

        The first samples of a run pay for cuDNN autotuning, lazy CUDA context
        creation and allocator growth. At batch size 1 there are no other
        samples in the batch to amortise that over, so the first image can read
        several times the steady-state cost. The eval scripts therefore run the
        first --timing-warmup images through the real measured path and then
        call this to throw the numbers away.

        Seconds are SUBTRACTED from the phase totals rather than zeroed, so
        `measured`, n_samples and the per-sample statistics keep describing the
        same set of samples. Only usable while per-sample recording is on --
        without the raw list there is no way to know what to subtract.
        """
        key = source or "all"
        dropped = 0
        # the normalised copies describe the same brackets, so they have to go
        # too or the median would still be summarising the warm-up
        for by_src in (self.per_batch.get(method) or {}).values():
            by_src.pop(key, None)
        for ph, by_src in (self.per_sample.get(method) or {}).items():
            vals = by_src.pop(key, [])
            if not vals:
                continue
            self.records[method][ph] = max(
                0.0, self.records[method][ph] - sum(vals) / 1000.0)
            # the per-source decomposition has to lose exactly the same
            # seconds, or it would stop summing to the total it decomposes
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
        # a source whose samples were all discarded leaves an empty list behind
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

        Present in ordinary batched runs too, which is what lets ONE job over a
        roster of N models report N per-model costs instead of a single roster
        average. The seconds here sum to the method's totals; the per-sample
        figure is that model's own attack seconds over its own samples, at the
        run's batch size -- batch-amortised, exactly like the total, so the two
        are the same kind of number.

        ms_per_sample_stats is that same quantity as a DISTRIBUTION over the
        model's attack brackets (one per batch), present whenever the eval
        script passed n= to phase(). Its mean equals attack_ms_per_sample only
        when every batch is full; the last batch is usually short, so the two
        differ slightly by construction and the ratio remains the figure to
        quote for a total. The median and std are what this block adds.
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
            # 'other' is only meaningful for the run as a whole: it is whatever
            # wall-clock the phases did not claim (I/O, image dumps, teardown).
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
