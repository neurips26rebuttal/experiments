#!/usr/bin/env python3
"""One command, no configuration: fetch everything, then run every attack case.

    python3 run.py              # the whole sweep
    python3 run.py --dry-run    # print the 24 eval commands, execute none

Four steps:

  1. models   -> ./models        every weight the eval scripts can load
  2. data     -> ./data/CIFAR100 the eval scripts open it with download=False
  3. imagenet -> WARNING ONLY    ImageNet needs a licensed download, so a
                                 missing ./data/imagenet/val is a warning and
                                 the ImageNet half is skipped; CIFAR-100 still
                                 runs to completion
  4. cases    -> ./results       6 cases x 2 datasets x 2 model families = 24
                                 runs, then the source x target tables

Steps 1-3 need internet, step 4 does not.

The results tree is FLAT: one directory per run, directly under ./results/,
named for everything that identifies it --

    results/imagenet_case1_standard_eps4div255_a1_b112_Hann_gamma0.1_steps20/
    results/cifar100_case5_pretrained_eps8div255_N20_rho0.5_sigma16.0_steps10/

so `ls results/` is the index of the sweep, no two runs can collide, and
src/build_transferability_table.py rebuilds the tables from the names alone.
"""
import argparse
import os
import subprocess
import sys
import warnings

# The eval scripts live in src/ and import each other by bare name
# ("import paths"), so src/ must be on sys.path before importing any of them.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import paths  # noqa: E402  (must follow the sys.path insert above)

DOWNLOAD_MODELS = os.path.join(SRC_DIR, "download_models.py")
EVAL_IMAGENET = os.path.join(SRC_DIR, "eval_imagenet.py")
EVAL_CIFAR100 = os.path.join(SRC_DIR, "eval_cifar100.py")
AGGREGATE_RUN = os.path.join(SRC_DIR, "aggregate_run.py")
BUILD_TABLE = os.path.join(SRC_DIR, "build_transferability_table.py")

RESULTS_ROOT = "./results"           # repo-relative: every run is launched from
RUNTIME_DIR = "./results/runtime"    # REPO_ROOT, so these resolve inside it


# ============================================================================
# Pinned parameters -- shared
# ============================================================================

NUM_SAMPLES = 5000     # test images attacked per run, switch to 10000 for CIFAR100
NUM_WORKERS = 10       # dataloader workers
LPIPS_NET = "alex"     # perceptual-distance backbone used by the metrics

GAMMA = 0.1

# deprecated parameter, kept for backwards compatibility
TAU = 0.1


# ============================================================================
# Pinned parameters -- ImageNet
# ============================================================================

# Gabor lattice. Case 1 feeds this frame to the attack; cases 2-6 still BUILD
# one to render the spectrogram panels, so every case must carry the same
# lattice or the panels are drawn in incomparable frames.
IMAGENET_A = 1
IMAGENET_B = 112
IMAGENET_WINDOW = "Hann"

#: Linf budget for every eps-bounded ImageNet case (1-5). Case 6 is not
#: eps-bounded; see IMAGENET_CASES.
IMAGENET_EPS = 4 / 255

#: Spectrogram/image samples dumped per model.
IMAGENET_NUM_IMAGES = 40

# The two source-model families. eval_imagenet.py recognises which family a
# name belongs to and puts the family in each run's directory name, so both
# families can share one --output-dir without colliding.
IMAGENET_MODELS = {
    # torchvision, normally trained.
    "standard": ["resnet50", "densenet121", "mobilenet_v2", "efficientnet_b0",
                 "googlenet", "maxvit_t", "mnasnet1_0", "regnet_y_8gf"],
    # RobustBench Linf, adversarially trained.
    "adv": ["Amini2024MeanSparse_ConvNeXt-L", "Bai2024MixedNUTS",
            "Debenedetti2022Light_XCiT-M12", "Engstrom2019Robustness",
            "Liu2023Comprehensive_ConvNeXt-B",
            "RodriguezMunoz2024Characterizing_Swin-B", "Salman2020Do_R50",
            "Singh2023Revisiting_ViT-B-ConvStem"],
}

# One entry per case. `steps` is None for the methods that do not read
# --num-steps (AutoAttack schedules itself; AdvDrop counts with --advdrop-steps),
# and the flag is then omitted rather than passed and ignored. `batch` is per
# family because the adversarially-trained models are much larger.
IMAGENET_CASES = [
    {   # DGF-PGD (proposed)
        "case": "1", "epsilon": IMAGENET_EPS, "steps": 20,
        "batch": {"standard": 32, "adv": 32},
        "extra": ["--gamma", repr(GAMMA)],
    },
    {   # PGD -- step size is eps/steps, set internally
        "case": "2", "epsilon": IMAGENET_EPS, "steps": 20,
        "batch": {"standard": 32, "adv": 32},
        "extra": [],
    },
    {   # F-PGD (Fourier PGD) -- same step rule as case 2, no Gabor lattice
        "case": "3", "epsilon": IMAGENET_EPS, "steps": 20,
        "batch": {"standard": 32, "adv": 32},
        "extra": [],
    },
    {   # AutoAttack -- reads epsilon only; its schedule is internal
        "case": "4", "epsilon": IMAGENET_EPS, "steps": None,
        "batch": {"standard": 32, "adv": 32},
        "extra": ["--aa-norm", "Linf", "--aa-version", "standard"],
    },
    {   # SSA (Spectrum Simulation Attack) -- upstream baselines/SSA/attack.py
        # defaults: num_iter=10 N=20 rho=0.5 sigma=16 momentum off (MI/DI/TI
        # disabled). --ssa-steps is pinned alongside --num-steps because SSA
        # falls back to --num-steps only when --ssa-steps is absent.
        "case": "5", "epsilon": IMAGENET_EPS, "steps": 10,
        "batch": {"standard": 32, "adv": 32},
        "extra": ["--ssa-steps", "10", "--ssa-N", "20", "--ssa-rho", "0.5",
                  "--ssa-sigma", "16.0", "--ssa-momentum", "0.0"],
    },
    {   # AdvDrop (InfoDrop) -- q=60 is the paper setting (the upstream repo
        # defaults to 40), untargeted. epsilon=0 is a PLACEHOLDER: AdvDrop is
        # not eps-bounded and never reads it, but eval_imagenet.py names its
        # output directories with the epsilon, so it must be a real number.
        # Smaller batch: AdvDrop optimises a quantisation table per image.
        "case": "6", "epsilon": 0.0, "steps": None,
        "batch": {"standard": 20, "adv": 20},
        "extra": ["--advdrop-steps", "150", "--advdrop-q-size", "60",
                  "--advdrop-block-size", "8", "--advdrop-lr", "0.01"],
    },
]


# ============================================================================
# Pinned parameters -- CIFAR-100
# ============================================================================

CIFAR_A = 1
CIFAR_B = 16
CIFAR_WINDOW = "Hann"

#: Case 1 runs at the ORIGINAL published setting, which is a wider budget than
#: the baselines use; cases 2-5 share the standard 8/255. Deliberately two
#: constants rather than one, so neither can be changed by editing the other.
CIFAR_CASE1_EPS = 16 / 255
CIFAR_EPS = 8 / 255

CIFAR_NUM_IMAGES = 10

# eval_cifar100.py loads ONE family per process and picks it with
# --model-source, so the key here is that flag's value, not just a label. The
# family goes into the run's directory name, so both families of a case share
# an output directory without colliding.
CIFAR_MODELS = {
    "robustbench": ["Addepalli2021Towards_WRN34",
                    "Amini2024MeanSparse_S-WRN-70-16",
                    "Bai2023Improving_trades", "Chen2024Data_WRN_34_10",
                    "Cui2023Decoupled_WRN-34-10",
                    "Debenedetti2022Light_XCiT-L12", "Jia2022LAS-AT_34_10",
                    "Pang2022Robustness_WRN28_10"],
    # chenyaofo/pytorch-cifar-models backbones, normally trained.
    "pretrained": ["resnet44", "resnet56", "shufflenetv2_x0_5",
                   "shufflenetv2_x2_0", "mobilenetv2_x0_75",
                   "mobilenetv2_x1_0", "repvgg_a1", "repvgg_a2"],
}

CIFAR_CASES = [
    {   # DGF-PGD (proposed), original settings: eps=16/255 with absolute gamma
        "case": "1", "epsilon": CIFAR_CASE1_EPS, "steps": 20,
        "batch": {"robustbench": 256, "pretrained": 256},
        "extra": ["--gamma", repr(GAMMA), "--tau", repr(TAU)],
    },
    {   # PGD
        "case": "2", "epsilon": CIFAR_EPS, "steps": 20,
        "batch": {"robustbench": 256, "pretrained": 256},
        "extra": [],
    },
    {   # F-PGD
        "case": "3", "epsilon": CIFAR_EPS, "steps": 20,
        "batch": {"robustbench": 256, "pretrained": 256},
        "extra": [],
    },
    {   # AutoAttack
        "case": "4", "epsilon": CIFAR_EPS, "steps": None,
        "batch": {"robustbench": 256, "pretrained": 256},
        "extra": ["--aa-norm", "Linf", "--aa-version", "standard"],
    },
    {   # SSA -- same upstream defaults as the ImageNet row. The RobustBench
        # batch is a quarter of the others: SSA runs N=20 extra forward passes
        # per step, and this family holds WRN-70-16-scale models.
        "case": "5", "epsilon": CIFAR_EPS, "steps": 10,
        "batch": {"robustbench": 64, "pretrained": 256},
        "extra": ["--ssa-steps", "10", "--ssa-N", "20", "--ssa-rho", "0.5",
                  "--ssa-sigma", "16.0", "--ssa-momentum", "0.0"],
    },
    {   # AdvDrop -- epsilon=0 is the same placeholder the ImageNet row carries.
        # Omitting it entirely would be worse than it looks: eval_cifar100.py's
        # --epsilon default is 32/255, and that fabricated budget would be
        # written into the results file and the runtime JSON.
        "case": "6", "epsilon": 0.0, "steps": None,
        "batch": {"robustbench": 20, "pretrained": 20},
        "extra": ["--advdrop-steps", "150", "--advdrop-q-size", "60",
                  "--advdrop-block-size", "8", "--advdrop-lr", "0.01"],
    },
]


# ============================================================================
# Dependencies
# ============================================================================

# import name -> pip name, for the packages whose absence kills a step. Checked
# up front because otherwise each failure surfaces hours in, after the weights
# are on disk and a GPU has been sitting idle.
REQUIRED = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
    "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "tqdm": "tqdm", "lpips": "lpips", "pytorch_msssim": "pytorch-msssim",
    "robustbench": "robustbench",   # downloads AND loads the adv models
    "gdown": "gdown",               # RobustBench's Google-Drive weights
    "autoattack": "autoattack",     # case 4 only
}


def check_dependencies():
    """Report every missing package at once instead of one ImportError per run."""
    missing = [pkg for mod, pkg in REQUIRED.items()
               if not _importable(mod)]
    if missing:
        warnings.warn("missing packages: " + ", ".join(missing) +
                      "  ->  pip install " + " ".join(missing),
                      RuntimeWarning, stacklevel=2)
        print(f"\n  !! MISSING PACKAGES: {', '.join(missing)}")
        print(f"     pip install {' '.join(missing)}")
        print("     Steps needing them will fail; the rest still run.\n")
    return missing


def _importable(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# ============================================================================
# Helpers
# ============================================================================

def banner(text):
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}", flush=True)


def run(cmd):
    """Run a child process from the repo root and return its exit code.

    cwd matters: every output directory above is repo-relative ("./results"),
    so a run.py launched from elsewhere still writes into the checkout.
    """
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO_ROOT)


# ============================================================================
# Step 1 -- models
# ============================================================================

def download_models():
    """Fetch every weight into ./models via src/download_models.py.

    Delegated rather than reimplemented: that script owns the model lists, and
    they have to stay identical to the ones the eval scripts load. It downloads
    a superset of the rosters above (every RobustBench entry, all 18 chenyaofo
    backbones, all 18 torchvision checkpoints) and skips whatever is already on
    disk, so re-running run.py is cheap.

    A nonzero exit means SOME weight failed; the rest were still fetched. That
    is a warning, not a stop -- one flaky Google-Drive transfer should not block
    the CIFAR-100 half of the run, and the eval names the missing model when it
    gets there.
    """
    banner(f"[1/4] Downloading models into {paths.MODELS_DIR}")
    rc = run([sys.executable, "-u", DOWNLOAD_MODELS,
              "--models-dir", paths.MODELS_DIR, "--dataset", "all"])
    if rc != 0:
        warnings.warn(f"download_models.py exited {rc} -- at least one weight is "
                      f"missing from {paths.MODELS_DIR}. Scroll up for the "
                      f"per-model errors.", RuntimeWarning, stacklevel=2)
    return rc == 0


# ============================================================================
# Step 2 -- CIFAR-100
# ============================================================================

def download_cifar100():
    """Put cifar-100-python/ under ./data/CIFAR100.

    eval_cifar100.py opens the dataset with download=False on purpose (compute
    nodes have no internet), so without this step the eval dies at load time.
    One call covers both splits: torchvision fetches a single tarball holding
    train and test, and re-running only verifies the checksum.
    """
    banner(f"[2/4] Downloading CIFAR-100 into {paths.CIFAR100_ROOT}")
    import torchvision

    os.makedirs(paths.CIFAR100_ROOT, exist_ok=True)
    torchvision.datasets.CIFAR100(root=paths.CIFAR100_ROOT, train=False,
                                  download=True)
    marker = os.path.join(paths.CIFAR100_ROOT, "cifar-100-python")
    print(f"  CIFAR-100 ready: {marker}")
    return os.path.isdir(marker)


# ============================================================================
# Step 3 -- ImageNet (check only)
# ============================================================================

def check_imagenet():
    """True if an ImageFolder-style val split is present; warn and return False.

    ImageNet is not downloadable without credentials, so this cannot be an
    automated step -- and it must not be a hard failure either: a machine with
    only CIFAR-100 should still produce the CIFAR-100 half of the results.
    """
    root = paths.IMAGENET_ROOT
    banner(f"[3/4] Checking for ImageNet at {root}")
    # eval_imagenet.py accepts either <root>/val or a root that already IS val.
    val_dir = root if root.rstrip("/").endswith("val") else os.path.join(root, "val")

    # A val/ holding no class directories is as unusable as no val/ at all --
    # ImageFolder builds an empty dataset and the failure surfaces much later --
    # so emptiness is part of "does not exist".
    n_classes = 0
    if os.path.isdir(val_dir):
        n_classes = sum(1 for e in os.scandir(val_dir) if e.is_dir())
    if n_classes:
        print(f"  Found ImageNet val/ with {n_classes} class directories.")
        return True

    reason = "not found" if not os.path.isdir(val_dir) else "has no class subdirectories"
    warnings.warn(
        f"ImageNet val split {reason} at {val_dir!r}. ImageNet requires a "
        f"licensed download from https://image-net.org and cannot be fetched "
        f"automatically. SKIPPING all 12 ImageNet runs; the 12 CIFAR-100 runs "
        f"still execute. Expected layout: {val_dir}/<wnid>/*.JPEG (override the "
        f"location with the DGF_IMAGENET_ROOT environment variable).",
        RuntimeWarning, stacklevel=2)
    print(f"\n  !! ImageNet val/ {reason} at {val_dir}")
    print("     -> the 12 ImageNet runs are SKIPPED (CIFAR-100 still runs)\n")
    return False


# ============================================================================
# Step 4 -- every case
# ============================================================================

def imagenet_command(spec, family):
    """Resolve one (case, model family) pair into an eval_imagenet.py argv."""
    cmd = [sys.executable, "-u", EVAL_IMAGENET,
           "--attack-case", spec["case"],
           "--models", *IMAGENET_MODELS[family],
           "--data-root", paths.IMAGENET_ROOT,
           "--models-dir", paths.MODELS_DIR,
           # repr(), not str(): the eval records the epsilon it was given, and
           # 4/255 must round-trip exactly rather than through a shortened form.
           "--epsilon", repr(spec["epsilon"]),
           "--a", str(IMAGENET_A), "--b", str(IMAGENET_B),
           "--window-type", IMAGENET_WINDOW]
    if spec["steps"] is not None:
        cmd += ["--num-steps", str(spec["steps"])]
    cmd += ["--output-dir", RESULTS_ROOT,
            "--runtime-dir", RUNTIME_DIR,
            "--num-samples", str(NUM_SAMPLES),
            "--batch-size", str(spec["batch"][family]),
            "--num-workers", str(NUM_WORKERS),
            "--lpips-net", LPIPS_NET,
            "--num-images", str(IMAGENET_NUM_IMAGES),
            "--save-heatmaps"]
    return cmd + spec["extra"]


def cifar_command(spec, source):
    """Resolve one (case, model source) pair into an eval_cifar100.py argv."""
    cmd = [sys.executable, "-u", EVAL_CIFAR100,
           "--case", f"case{spec['case']}",
           "--model-source", source,
           "--models", *CIFAR_MODELS[source],
           "--data-root", paths.CIFAR100_ROOT,
           "--models-dir", paths.MODELS_DIR,
           "--epsilon", repr(spec["epsilon"]),
           "--a", str(CIFAR_A), "--b", str(CIFAR_B),
           "--window-type", CIFAR_WINDOW]
    if spec["steps"] is not None:
        cmd += ["--num-steps", str(spec["steps"])]
    cmd += ["--output-dir", RESULTS_ROOT,
            "--save-images", "--num-images", str(CIFAR_NUM_IMAGES),
            "--runtime-dir", RUNTIME_DIR,
            "--num-samples", str(NUM_SAMPLES),
            "--batch-size", str(spec["batch"][source]),
            "--num-workers", str(NUM_WORKERS),
            "--lpips-net", LPIPS_NET]
    return cmd + spec["extra"]


def build_all_runs(have_imagenet):
    """Every (label, argv) this script executes, in order.

    CIFAR-100 first so that a machine without ImageNet -- and a machine whose
    ImageNet runs are going to fail for some other reason -- still produces the
    complete CIFAR-100 table before anything can go wrong.
    """
    runs = []
    for spec in CIFAR_CASES:
        for source in CIFAR_MODELS:
            runs.append((f"cifar100 case{spec['case']} {source}",
                         cifar_command(spec, source)))
    if have_imagenet:
        for spec in IMAGENET_CASES:
            for family in IMAGENET_MODELS:
                runs.append((f"imagenet case{spec['case']} {family}",
                             imagenet_command(spec, family)))
    return runs


def run_all(runs, dry_run):
    """Execute every run, keeping going after a failure.

    One case failing (an OOM, a weight that never downloaded) must not cost the
    other 23 their results, so failures are collected and reported at the end
    rather than aborting the sweep.
    """
    failures = []
    for i, (label, cmd) in enumerate(runs):
        banner(f"[{i + 1}/{len(runs)}] {label}")
        print(f"[run] {' '.join(cmd)}", flush=True)
        if dry_run:
            continue
        rc = subprocess.call(cmd, cwd=REPO_ROOT)
        if rc != 0:
            failures.append((label, rc))
            print(f"\n  !! {label} FAILED (exit {rc}) -- continuing\n", flush=True)
    return failures


def aggregate():
    """Fold the raw per-run runtime/metrics JSON into the root-level tables.

    Derived data: a failure here must not fail the sweep whose results are
    already on disk.
    """
    banner(f"Aggregating {RESULTS_ROOT}")
    rc = run([sys.executable, AGGREGATE_RUN, RESULTS_ROOT])
    if rc != 0:
        warnings.warn(f"aggregate_run.py exited {rc}", RuntimeWarning, stacklevel=2)


def build_tables():
    """Build the source x target matrices and the summary.

    This is the deliverable: summary.csv splits the diagonal (white_box) from
    the off-diagonal mean (transfer), which a single average over the matrix
    would silently merge. --percent because the rates are reported in percent.
    """
    banner(f"Building transferability tables from {RESULTS_ROOT}")
    if not os.path.isdir(os.path.join(REPO_ROOT, RESULTS_ROOT)):
        warnings.warn(f"no results tree at {RESULTS_ROOT} -- nothing to tabulate",
                      RuntimeWarning, stacklevel=2)
        return False
    rc = run([sys.executable, BUILD_TABLE,
              "--results-dir", RESULTS_ROOT,
              "--out-dir", os.path.join(RESULTS_ROOT, "tables"),
              "--percent"])
    if rc == 0:
        print(f"\n  Tables: {os.path.join(RESULTS_ROOT, 'tables')}")
        print("    summary.csv   white_box (diagonal) vs transfer (off-diagonal)")
        print("    matrix_*.csv  full source x target per configuration")
    return rc == 0


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="print every resolved command, execute nothing "
                        "(still checks for the models and the datasets)")
    args = p.parse_args()

    print(f"repo    : {REPO_ROOT}")
    print(f"python  : {sys.executable}")
    print(f"models  : {paths.MODELS_DIR}")
    print(f"cifar   : {paths.CIFAR100_ROOT}")
    print(f"imagenet: {paths.IMAGENET_ROOT}")

    check_dependencies()

    if args.dry_run:
        banner("[1/4] and [2/4] Downloads: SKIPPED (--dry-run)")
    else:
        download_models()
        download_cifar100()

    have_imagenet = check_imagenet()

    banner("[4/4] Running all attack cases")
    runs = build_all_runs(have_imagenet)
    print(f"  {len(runs)} runs: 6 cases x "
          f"{'2 datasets x ' if have_imagenet else '1 dataset (cifar100) x '}"
          f"2 model families")
    failures = run_all(runs, args.dry_run)

    if not args.dry_run:
        aggregate()
        build_tables()

    banner("DONE" if not failures else f"DONE ({len(failures)} failed)")
    print(f"  runs      : {len(runs)}")
    print(f"  imagenet  : {'included' if have_imagenet else 'SKIPPED (no val/)'}")
    print(f"  results   : {RESULTS_ROOT}")
    for label, rc in failures:
        print(f"  FAILED    : {label} (exit {rc})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
