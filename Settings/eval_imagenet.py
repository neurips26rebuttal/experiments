"""
ImageNet Adversarial Attack Evaluation — Fixed Hyperparameters
==============================================================
Runs case 1-6 with hyperparameters supplied as CLI arguments.
Produces 20 spectrogram samples per model. No hparam sweeping.

Attack cases:
  --attack-case 1    DGF-PGD (Gabor frame, proposed)
  --attack-case 2    Standard PGD
  --attack-case 3    Fourier-based PGD
  --attack-case 4    AutoAttack
  --attack-case 5    SSA -- Spectrum Simulation Attack   (baselines/SSA)
  --attack-case 6    AdvDrop -- InfoDrop                 (baselines/AdvDrop)
"""
import gc
import math
import random
import re
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
import time
from contextlib import contextmanager
from typing import Dict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import paths
from transforms import *
from runtime_log import RuntimeLog
from run_dirs import case_hparam_token, format_eps_tag, run_dir_name
from dgf_pgd import DGFPGDAttack
from baseline_attacks import build_baseline_attacker
from evaluation_metrics import AdversarialMetrics

try:
    from pytorch_msssim import ssim as _ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================

# Max columns in the combined clean/adv/delta comparison strip (see M8).
MAX_COMPARISON_COLUMNS = 10

STANDARD_PRETRAINED_NAMES = [
    'resnet18', 'resnet50', 'vgg16_bn', 'densenet121',
    'mobilenet_v2', 'efficientnet_b0', 'convnext_tiny', 'vit_b_16',
    'alexnet', 'googlenet', 'inception_v3', 'maxvit_t',
    'mnasnet1_0', 'regnet_y_8gf', 'resnext50_32x4d',
    'shufflenet_v2_x1_0', 'swin_t', 'wide_resnet50_2',
]

# Paper-friendly display names for plots (raw keys → display labels).
# Any name not listed here falls back to the raw key.
DISPLAY_NAMES: Dict[str, str] = {
    # Standard torchvision models
    'resnet18':            'ResNet-18',
    'resnet50':            'ResNet-50',
    'vgg16_bn':            'VGG-16-BN',
    'densenet121':         'DenseNet-121',
    'mobilenet_v2':        'MobileNet-V2',
    'efficientnet_b0':     'EfficientNet-B0',
    'convnext_tiny':       'ConvNeXt-T',
    'vit_b_16':            'ViT-B/16',
    'alexnet':             'AlexNet',
    'googlenet':           'GoogLeNet',
    'inception_v3':        'Inception-V3',
    'maxvit_t':            'MaxViT-T',
    'mnasnet1_0':          'MNASNet-1.0',
    'regnet_y_8gf':        'RegNet-Y-8GF',
    'resnext50_32x4d':     'ResNeXt-50 32x4d',
    'shufflenet_v2_x1_0':  'ShuffleNet-V2',
    'swin_t':              'Swin-T',
    'wide_resnet50_2':     'WRN-50-2',
    # Adversarially-trained RobustBench models (matching download_models.py),
    # AuthorYY style to match the CIFAR-100 labels in eval_cifar100.py.
    'Amini2024MeanSparse_ConvNeXt-L':              'Amini24',
    'Liu2023Comprehensive_ConvNeXt-B':             'Liu23',
    'Bai2024MixedNUTS':                            'Bai24',
    'Debenedetti2022Light_XCiT-M12':               'Debenedetti22',
    'Engstrom2019Robustness':                      'Engstrom19',
    'RodriguezMunoz2024Characterizing_Swin-B':     'Rodriguez24',
    'Salman2020Do_R50':                            'Salman20',
    'Singh2023Revisiting_ViT-B-ConvStem':          'Singh23',
    'Wong2020Fast':                                'Wong20',
}


def display_name(key: str) -> str:
    """Return the paper-friendly label for a model key."""
    return DISPLAY_NAMES.get(key, key)


# ============================================================================
# NormalizedModel
# ============================================================================

class NormalizedModel(nn.Module):
    """Wraps a torchvision model with ImageNet mean/std normalization."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


# ============================================================================
# Hyperparameter folder name
# ============================================================================

class _NullRuntimeLog:
    """Stand-in so instrumentation is a no-op when no log was attached.

    Keeps the timing calls from being load-bearing: a caller that never built a
    RuntimeLog (a unit test, an audit script importing this module) behaves
    exactly as it did before instrumentation.
    """

    @contextmanager
    def phase(self, method, name, per_sample=False, source=None):
        yield

    def add_samples(self, method, n, source=None):
        pass

    def discard_warmup(self, method, source=None):
        return 0


def _rtlog(args):
    return getattr(args, "_rtlog", None) or _NullRuntimeLog()


@contextmanager
def _rng_preserved():
    """Restore every RNG on exit, so the block inside changes no result.

    The warm-up below runs the REAL attack, and several attacks draw randoms
    (cases 1/2 pass random_init=True on cifar, AutoAttack restarts, SSA's
    Gaussian noise). Without this the warm-up would advance the streams and
    every adversarial example after it would differ from the same run without
    warm-up -- which would break the bit-for-bit baseline parity that cases 5
    and 6 are held to.
    """
    cpu = torch.get_rng_state()
    cuda = (torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None)
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(np_state)
        random.setstate(py_state)


def warmup_attack(fn, args, source=None):
    """Run the attack `--warmup-batches` times, timed by nothing, then discard.

    The first batch a model sees pays for cuDNN autotuning, lazy CUDA context
    creation and allocator growth. In --timing mode that cost is removed by
    running warm-up IMAGES through the measured path and subtracting them
    afterwards (_timing_discard). An ordinary batched run has no such
    mechanism, so before this the whole warm-up landed inside the first timed
    bracket -- and, on the first model of a roster, dominated it: on the
    18-model ImageNet sweep alexnet (the cheapest model there is) reported
    28.86 ms/sample against resnet18's 5.12, because its first batch alone was
    5.0s of its 5.772s total.

    Running the real attack rather than a bare forward is deliberate: it is the
    only way to touch every kernel the measured path will use. `fn` is a
    zero-argument thunk so each call site passes its own signature (case 4
    calls AutoAttack, case 1 returns three values, the rest two).

    Nothing here is recorded: no rt.phase, no add_samples. The warm-up is
    outside the measurement by construction rather than subtracted from it.
    """
    n = int(getattr(args, "warmup_batches", 0) or 0)
    # --timing has its own warm-up (measured, then subtracted by
    # _timing_discard). Doing both would warm twice and measure neither better.
    if n <= 0 or getattr(args, "timing", False):
        return
    with _rng_preserved():
        for _ in range(n):
            fn()
    _sync_device(getattr(args, "device", None))
    if getattr(args, "verbose", False):
        print(f"  [warmup] {n} batch(es) discarded"
              f"{f' for {source}' if source else ''}")


def _sync_device(device):
    """Block on queued CUDA work so the warm-up finishes before timing starts."""
    if device is None or str(device).startswith("cpu"):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timing_discard(rt, method, source, batch_idx, args):
    """Drop the warm-up images once they are behind us (--timing only).

    Called at the end of every batch body so the warm-up goes through the exact
    same code path as a measured image rather than through a separate dry-run
    branch that could end up warming something else. Shared with
    eval_cifar100.py so both datasets discard identically.
    """
    if not getattr(args, "timing", False) or args.timing_warmup <= 0:
        return
    if batch_idx + 1 == args.timing_warmup:
        n = rt.discard_warmup(method, source)
        print(f"  [timing] discarded {n} warm-up image(s) for {source}; "
              f"measurement starts now")


def resolve_gamma(args):
    """Step size for the ascent, returned together with a human-readable reason.

    Every DGF-PGD step advances the iterate by EXACTLY gamma in the M_D-norm, so
    after k steps ||delta||_M <= k*gamma. gamma therefore lives in the SAME units
    as eps_dgf: an absolute value is only meaningful next to a known eps_dgf,
    which is what warn_if_constraint_inert() reports.

    The original implementation used the absolute 0.1, which is what an omitted
    --gamma still resolves to.
    """
    if args.gamma is not None:
        return float(args.gamma), "absolute (--gamma)"
    return 0.1, "original default"


def warn_if_constraint_inert(gamma, num_steps, eps_scale, epsilon, why):
    """Loudly flag a configuration in which the DGF constraint can never bind.

    If num_steps*gamma < eps_dgf the projection never activates and case 1
    degenerates to unconstrained preconditioned ascent under the [0,1] clamp --
    the perceptual budget stops being a budget. That produces numbers that look
    like a weak attack rather than like a misconfiguration, so it must be noisy.
    """
    eps_dgf = eps_scale * epsilon
    reach = num_steps * gamma
    print(f"  eps_scale={eps_scale:.6g}  eps={epsilon:.6g}  eps_dgf={eps_dgf:.6g}")
    print(f"  gamma={gamma:.6g} [{why}]  reach={reach:.6g}  eps_dgf={eps_dgf:.6g} "
          f"({reach / eps_dgf if eps_dgf else float('nan'):.3g}x budget)")
    if reach < eps_dgf:
        print(f"  !! WARNING: {num_steps} steps x gamma={gamma:.6g} reaches only "
              f"{reach:.6g} in the M_D-norm, below eps_dgf={eps_dgf:.6g}.\n"
              f"  !! The projection can NEVER activate: case 1 degenerates to "
              f"unconstrained ascent and the attack is far weaker than intended.\n"
              f"  !! Raise --gamma (2.5*eps_dgf/num_steps = "
              f"{2.5*eps_dgf/num_steps:.6g} is the standard PGD heuristic).")


# ============================================================================
# Command-line Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ImageNet Adversarial Attack Evaluation (fixed hparams)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--attack-case', type=str, nargs='+', default=['1', '2', '3'],
                        choices=['1', '2', '3', '4', '5', '6'],
                        help='Attack case(s) to run, e.g. --attack-case 1 2 3. '
                             '5=SSA (Spectrum Simulation Attack), '
                             '6=AdvDrop (InfoDrop).')
    parser.add_argument("--aa-norm", type=str, default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--aa-version", type=str, default="standard",
                        choices=["standard", "plus", "rand"])

    parser.add_argument("--data-root", type=str, default=paths.IMAGENET_ROOT,
                        help="ImageNet root containing val/ (default: the "
                             "repo's data/imagenet)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epsilon", type=float, required=True,
                        help="Perturbation budget (e.g. 8/255 = 0.03137)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Step size")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of PGD steps (K)")
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--b", type=int, default=112)
    parser.add_argument("--window-type", type=str, default="Hann",
                        choices=["Hann", "Blackman", "Gaussian"])

    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model names to evaluate. Names in "
                             "STANDARD_PRETRAINED_NAMES load from torchvision; "
                             "all others load from RobustBench.")
    parser.add_argument("--models-dir", type=str, default=paths.MODELS_DIR,
                        help="RobustBench weight tree, models/<dataset>/"
                             "<threat_model>/<name>.pt (default: the repo's "
                             "models/)")

    parser.add_argument("--output-dir", type=str, default="./results/imagenet")
    parser.add_argument("--runtime-dir", type=str, default=None,
                        help="Shared directory for the raw runtime_*.json "
                             "files (default: <output-dir>/runtime)")
    parser.add_argument("--lpips-net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of spectrogram/image samples to save per model")

    # --- Case 5: SSA (Spectrum Simulation Attack). Defaults mirror
    # --- baselines/SSA/attack.py. --num-steps is reused unless --ssa-steps set.
    parser.add_argument("--ssa-steps", type=int, default=None,
                        help="SSA iterations (case 5). Default: --num-steps.")
    parser.add_argument("--ssa-N", type=int, default=20,
                        help="SSA spectral copies averaged per step (case 5).")
    parser.add_argument("--ssa-rho", type=float, default=0.5,
                        help="SSA spectral-mask tuning factor (case 5).")
    parser.add_argument("--ssa-sigma", type=float, default=16.0,
                        help="SSA Gaussian-noise std in 0-255 units (case 5).")
    parser.add_argument("--ssa-momentum", type=float, default=0.0,
                        help="SSA MI-FGSM momentum (case 5). 0 = off, as upstream.")

    # --- Case 6: AdvDrop (InfoDrop). Defaults mirror
    # --- baselines/AdvDrop/infod_sample.py. epsilon does not apply to this attack.
    parser.add_argument("--advdrop-steps", type=int, default=150,
                        help="AdvDrop optimization steps (case 6).")
    parser.add_argument("--advdrop-q-size", type=float, default=60.0,
                        help="AdvDrop quantization-table upper bound (case 6). "
                             "Paper setting: 60. Upstream repo default is 40; "
                             "the vendored attack class keeps that default, "
                             "only this CLI default differs.")
    parser.add_argument("--advdrop-block-size", type=int, default=8,
                        help="AdvDrop DCT block size (case 6).")
    parser.add_argument("--advdrop-lr", type=float, default=0.01,
                        help="AdvDrop Adam learning rate (case 6).")

    # Timing
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Run case1's model call in fp32 instead of bf16 autocast "
        "(src/dgf_pgd.py:102). Cases 2/3 and the vendored baselines are "
        "already fp32, so the DEFAULT times the proposed method under a "
        "faster precision than everything it is compared against; pass this "
        "for a precision-matched cost comparison. No effect on cases 2-6.")
    parser.add_argument(
        "--timing", action="store_true",
        help="Cost-measurement mode: force batch size 1 so one timed bracket "
        "is exactly one image, record every sample individually (mean/median/"
        "std/min/max/p95 per source model, plus the raw list) and discard "
        "--timing-warmup images first. Measures the same work as "
        "eval_cifar100.py --timing, so the two per-image columns are "
        "comparable.")
    parser.add_argument(
        "--warmup-batches", type=int, default=1,
        help="Batches run through the real attack and thrown away before the "
        "timed loop starts, once per source model, so cuDNN autotuning and "
        "lazy CUDA init never land in a measured bracket. Costs one batch of "
        "wall-clock per model and changes no result: every RNG is restored "
        "afterwards. 0 restores the old (contaminated) behaviour. Ignored "
        "under --timing, which discards measured warm-up images instead.")
    parser.add_argument(
        "--timing-warmup", type=int, default=5,
        help="Images run through the real measured path and then thrown away, "
        "so cuDNN autotuning and lazy CUDA init never land in a kept sample. "
        "--num-samples is raised by this much so exactly --num-samples images "
        "are measured (--timing only)")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # One bracket must hold exactly one image or the per-sample distribution is
    # a distribution over batches. Forced, not merely defaulted: a --batch-size
    # inherited from the manifest would otherwise silently invalidate the run.
    if args.timing:
        if args.batch_size != 1:
            print(f"[timing] forcing --batch-size 1 (was {args.batch_size})")
        args.batch_size = 1
        args.timing_warmup = max(0, args.timing_warmup)
        # The run directory gets a "_timing" suffix (in _group_dir) so a
        # batch-size-1 cost run cannot overwrite a 1000-sample result with a
        # 200-sample one.
    return args


# ============================================================================
# Dataset Loading
# ============================================================================

def load_imagenet(args):
    print("\nLoading ImageNet dataset...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_dir = os.path.join(args.data_root, "val") if not args.data_root.endswith("/val") else args.data_root
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"ImageNet val/ not found at {val_dir}")

    testset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    print(f"  Found {len(testset)} images in {len(testset.classes)} classes")

    # --timing loads the warm-up images ON TOP of what was asked for, so
    # "--timing --num-samples 200" measures 200 images rather than 200 minus
    # whatever the warm-up consumed.
    n_want = args.num_samples
    if n_want is not None and getattr(args, "timing", False):
        n_want += args.timing_warmup
    if n_want is not None and n_want < len(testset):
        indices = np.random.choice(len(testset), n_want, replace=False)
        testset = Subset(testset, indices)

    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"  Loaded {len(testset)} test samples")
    if getattr(args, "timing", False):
        print(f"  [timing] {args.timing_warmup} warm-up + {args.num_samples} "
              f"measured, one image per batch")
    return loader


# ============================================================================
# Model Loading
# ============================================================================

def load_standard_pretrained_models(args):
    print("\nLoading standard pretrained ImageNet models...")
    # Weights come from the repo-local hub cache (models/torch_hub), populated
    # in advance by src/download_models.py -- compute nodes have no internet.
    hub = paths.point_torch_hub(args.models_dir)
    print(f"  torch.hub cache: {hub}")
    builders = {
        # --- existing ---
        'resnet18':           lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet50':           lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2),
        'vgg16_bn':           lambda: tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1),
        'densenet121':        lambda: tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1),
        'mobilenet_v2':       lambda: tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V2),
        'efficientnet_b0':    lambda: tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'convnext_tiny':      lambda: tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        'vit_b_16':           lambda: tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1),
        # --- new ---
        'alexnet':            lambda: tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1),
        'googlenet':          lambda: tv_models.googlenet(weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1, transform_input=False),
        'inception_v3':       lambda: tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False),
        'maxvit_t':           lambda: tv_models.maxvit_t(weights=tv_models.MaxVit_T_Weights.IMAGENET1K_V1),
        'mnasnet1_0':         lambda: tv_models.mnasnet1_0(weights=tv_models.MNASNet1_0_Weights.IMAGENET1K_V1),
        'regnet_y_8gf':       lambda: tv_models.regnet_y_8gf(weights=tv_models.RegNet_Y_8GF_Weights.IMAGENET1K_V2),
        'resnext50_32x4d':    lambda: tv_models.resnext50_32x4d(weights=tv_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
        'shufflenet_v2_x1_0': lambda: tv_models.shufflenet_v2_x1_0(weights=tv_models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
        'swin_t':             lambda: tv_models.swin_t(weights=tv_models.Swin_T_Weights.IMAGENET1K_V1),
        'wide_resnet50_2':    lambda: tv_models.wide_resnet50_2(weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    }
    target_names = args.models if args.models else list(builders.keys())
    # An explicitly requested name that does not exist is a manifest typo, and
    # skipping it silently would let a per-model cost row measure a DIFFERENT
    # model (or nothing) under the name the table will print. Only the implicit
    # "every builder" roster is allowed to skip.
    if args.models:
        unknown = [n for n in args.models if n not in builders]
        if unknown:
            raise ValueError(
                f"unknown model(s) {unknown} requested via --models; "
                f"known: {', '.join(sorted(builders))}")
    models_dict = {}
    for name in target_names:
        if name not in builders:
            continue
        try:
            print(f"  Loading {name}...")
            m = NormalizedModel(builders[name]()).to(args.device).eval()
            models_dict[name] = m
            print(f"    OK")
        except Exception as e:
            print(f"    Failed: {e}")
    if not models_dict:
        raise RuntimeError("No standard pretrained models loaded!")
    print(f"\n  Loaded {len(models_dict)} standard models")
    return models_dict


def load_imagenet_models(args):
    print("\nLoading ImageNet models from RobustBench...")
    try:
        from robustbench.utils import load_model as rb_load_model
        import robustbench.utils as _rb_utils
    except ImportError:
        raise ImportError("robustbench is required. Install with: pip install robustbench")

    try:
        import gdown

        def _download_gdrive_fixed(gdrive_id, fname_save):
            fname_save = str(fname_save)
            gdown.download(id=gdrive_id, output=fname_save, quiet=False)

        _rb_utils.download_gdrive = _download_gdrive_fixed
    except ImportError:
        pass

    def _canonical_model_name(weight_filename):
        if not weight_filename.endswith(".pt"):
            return None
        m = re.match(r"^(?P<name>.+)\.pt_m\d+\.pt$", weight_filename)
        if m:
            return m.group("name")
        m = re.match(r"^(?P<name>.+)_m\d+\.pt$", weight_filename)
        if m:
            return m.group("name")
        return weight_filename[:-3]

    base_dir = os.path.join(args.models_dir, "imagenet")
    candidates, seen = [], set()
    if os.path.isdir(base_dir):
        for tm in sorted(os.listdir(base_dir)):
            td = os.path.join(base_dir, tm)
            if not os.path.isdir(td):
                continue
            for f in sorted(os.listdir(td)):
                name = _canonical_model_name(f)
                if name is None:
                    continue
                if os.path.getsize(os.path.join(td, f)) < 100_000:
                    continue
                c = (name, tm)
                if c not in seen:
                    candidates.append(c)
                    seen.add(c)

    if not candidates:
        raise ValueError(f"No downloaded models found in {base_dir}")
    if args.models:
        candidates = [(n, t) for n, t in candidates if n in args.models]
    if not candidates:
        raise ValueError("No matching RobustBench models found.")

    print(f"  Loading {len(candidates)} RobustBench models:")
    models_dict = {}
    for model_name, threat_model in candidates:
        try:
            print(f"    {model_name} [{threat_model}]...")
            model = rb_load_model(model_name=model_name, dataset="imagenet",
                                  threat_model=threat_model,
                                  model_dir=args.models_dir).to(args.device)
            models_dict[model_name] = model.eval()
            print(f"      OK")
        except Exception as e:
            print(f"      Failed: {e}")

    if not models_dict:
        raise RuntimeError("No RobustBench models loaded!")
    print(f"\n  Loaded {len(models_dict)} RobustBench models")
    return models_dict


# ============================================================================
# Gabor Operators
# ============================================================================

def generate_gabor_operators(device, a, b, image_size=224, window_type="Hann"):
    """Build the Gabor operators exactly as the original implementation does.

    D comes from the Monte-Carlo row-sum estimate drawn from the global RNG, so
    D depends on how much randomness model construction consumed first.

    M = Psi^H D^-1 Psi: the original code called weights1() with D already
    inverted in place, and D_inv_1 is built from the same D, so M and D_inv_1
    coincide. eps_scale is exp(mean(log mu^2)/(2n)) times the volume prefactor,
    which measures as the constant ~108.42 at 224x224 -- the exp() term is
    0.9997, so eps_dgf carries essentially no information about M.
    """
    print(f"\nGenerating Gabor operators: a={a}, b={b}, window={window_type}, size={image_size}")
    H = image_size
    n = H * H

    Psi_2D = DGT(H, a=a, b=b, window=window_type)
    Psi_2D = Psi_2D / torch.linalg.norm(Psi_2D, dim=1, keepdim=True)

    S_2D = frameop_DGT(Psi_2D)
    sv = torch.linalg.svdvals(S_2D)
    sv_pos = sv[sv > 1e-10]
    cond_S = (sv_pos[0] / sv_pos[-1]).item() if sv_pos.numel() > 0 else float("inf")

    D_2D = diag_weights_from_mc_row_sums(Psi_2D, mode="down")

    D_inv_1_2D = dual_norm1(D_2D, Psi_2D)
    # Bit-for-bit reproduction of the original in-place D.pow_(-1) side effect:
    # weights1 received D^-1, making M identical to D_inv_1_2D.
    M_2D = weights1(D_2D.pow(-1), Psi_2D)

    Mherm_2D = 0.5 * (M_2D + M_2D.mH)
    jitter = float(1e-6) * Mherm_2D.abs().mean()
    Mherm_2D = Mherm_2D + jitter * torch.eye(H, device=Mherm_2D.device, dtype=Mherm_2D.dtype)
    torch.cuda.empty_cache()
    gc.collect()
    Mherm_2D = Mherm_2D.to(device)

    # The factorized projection consumes (mu_M, U_M) directly, so a failed
    # eigendecomposition leaves no usable path -- fail here rather than later.
    try:
        M64 = (Mherm_2D.to(torch.complex64) if Mherm_2D.is_complex()
               else Mherm_2D.to(torch.float64))
        mu64, U64 = torch.linalg.eigh(M64.cpu())
    except Exception as e:
        raise RuntimeError(
            f"eigh() failed on the Hermitian part of M; the factorized "
            f"projection cannot run without it: {e}"
        ) from e
    mu_M_2D = mu64.real.clamp_min(0.0).to(torch.float32).to(device)
    U_M_2D = U64.to(Mherm_2D.dtype).to(device)

    mu_safe = mu_M_2D.clamp_min(1e-8)
    eps_scale = torch.exp((torch.log(mu_safe ** 2).mean()) / (2 * n)).item()
    eps_scale = (math.sqrt((2 * n) / (math.e * math.pi))
                 * math.sqrt(math.pi * n) ** (1 / n)
                 * eps_scale)
    print(f"  eps_scale={eps_scale:.6g}, cond_S={cond_S:.4g}")

    return Psi_2D, D_inv_1_2D, M_2D, eps_scale, mu_M_2D, U_M_2D, cond_S


# ============================================================================
# Attacker / Metrics helpers
# ============================================================================

def _build_attacker(model, Psi_2D, D_inv_1_2D, M_2D,
                    eps_scale, mu_M_2D, U_M_2D,
                    epsilon, gamma, num_steps, case, args):
    return DGFPGDAttack(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        Psi_2D=Psi_2D,
        D_inv_1=D_inv_1_2D,
        M=M_2D,
        eps_scale=eps_scale,
        mu_M=mu_M_2D,
        U_M=U_M_2D,
        image_shape=(3, 224, 224),
        epsilon=epsilon,
        gamma=gamma,
        num_steps=num_steps,
        case=case,
        device=args.device,
        verbose=args.verbose,
        amp=not args.no_amp,
    )


def _build_metrics(args):
    return AdversarialMetrics(
        device=args.device,
        lpips_net=args.lpips_net,
        verbose=args.verbose,
    )


# ============================================================================
# Transferability Evaluation
# ============================================================================

def generate_adversarial_examples(source_model, source_name, attacker,
                                   testloader, device, case, verbose=False, args=None):
    """Attack every test image with `source_name` and return clean/adv/labels.

    The `attack` phase brackets ONLY the attacker call, exactly as
    eval_cifar100.py does. Dataloading (JPEG decode + resize, which is not
    cheap at 224x224), the host<->device copies and the final concatenation are
    deliberately outside it: charging them here made the ImageNet per-sample
    column measure something CIFAR's did not, and the two are printed side by
    side.
    """
    print(f"\nGenerating adversarial examples using {source_name}...")
    rt = _rtlog(args)
    timing = bool(getattr(args, "timing", False))

    if case == "case4":
        try:
            from autoattack import AutoAttack
        except ImportError:
            raise ImportError("autoattack not installed. pip install autoattack")
        adversary = AutoAttack(source_model, norm=args.aa_norm, eps=args.epsilon,
                               version=args.aa_version, device=device, verbose=verbose)
        if not timing:
            # Default path, unchanged: AutoAttack runs over the whole set in one
            # call and does its own internal batching. Only the collection loop
            # moved out of the bracket.
            all_clean, all_labels = [], []
            for x, y in tqdm(testloader, desc=f"Collecting data for {source_name}"):
                all_clean.append(x)
                all_labels.append(y)
            x_clean = torch.cat(all_clean, dim=0)
            y_true = torch.cat(all_labels, dim=0)
            # One batch, not the whole set: the warm-up only has to touch the
            # kernels, and AutoAttack over 200 images is the expensive one.
            _wb = min(args.batch_size, len(y_true))
            warmup_attack(lambda: adversary.run_standard_evaluation(
                x_clean[:_wb].to(device), y_true[:_wb].to(device), bs=_wb),
                args, source_name)
            with rt.phase(case, "attack", source=source_name,
                          n=len(y_true)):
                x_adv = adversary.run_standard_evaluation(
                    x_clean.to(device), y_true.to(device), bs=args.batch_size).cpu()
            rt.add_samples(case, len(y_true), source=source_name)
            print(f"  Generated {len(x_clean)} adversarial examples (AutoAttack)")
            return x_clean, x_adv, y_true, None
        # --timing: one call per image so each gets its own bracket. Per-sample
        # results are identical -- AutoAttack's cascade is per-sample and never
        # looks across the batch -- but the printed per-call summaries differ.
        all_clean, all_adv, all_labels = [], [], []
        for i, (x, y) in enumerate(tqdm(testloader, desc=f"Source: {source_name}")):
            x, y = x.to(device), y.to(device)
            with rt.phase(case, "attack", per_sample=True, source=source_name,
                          n=x.shape[0]):
                x_adv = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
            rt.add_samples(case, x.shape[0], source=source_name)
            all_clean.append(x.cpu())
            all_adv.append(x_adv.cpu())
            all_labels.append(y.cpu())
            _timing_discard(rt, case, source_name, i, args)
        x_clean = torch.cat(all_clean, dim=0)
        x_adv = torch.cat(all_adv, dim=0)
        y_true = torch.cat(all_labels, dim=0)
        print(f"  Generated {len(x_clean)} adversarial examples (AutoAttack)")
        return x_clean, x_adv, y_true, None

    attacker.model = source_model
    all_clean, all_adv, all_labels = [], [], []
    q_all = []          # case 1 only: Gabor-frame constraint value per sample
    source_model.eval()
    _warm = next(iter(testloader), None)
    if _warm is not None:
        _wx, _wy = _warm[0].to(device), _warm[1].to(device)
        warmup_attack(lambda: attacker(_wx, _wy, random_init=False),
                      args, source_name)
        del _wx, _wy
    for i, (x, y) in enumerate(tqdm(testloader, desc=f"Source: {source_name}")):
        x, y = x.to(device), y.to(device)
        with rt.phase(case, "attack", per_sample=timing, source=source_name,
                      n=x.shape[0]):
            if case == 'case1':
                x_adv, eps_dgf, last_delta_flat = attacker(x, y, random_init=False)
            else:
                x_adv, _ = attacker(x, y, random_init=False)
        rt.add_samples(case, x.shape[0], source=source_name)
        if case == 'case1':
            # outside the bracket: a diagnostic, not part of the attack
            q_all.append(attacker.gabor_constraint(last_delta_flat).cpu())
        all_clean.append(x.cpu())
        all_adv.append(x_adv.cpu())
        all_labels.append(y.cpu())
        _timing_discard(rt, case, source_name, i, args)
    x_clean = torch.cat(all_clean, dim=0)
    x_adv = torch.cat(all_adv, dim=0)
    y_true = torch.cat(all_labels, dim=0)

    attack_stats = None
    if case == 'case1' and q_all:
        q = torch.cat(q_all)
        eps2 = float(eps_dgf) ** 2
        attack_stats = {
            'eps_dgf': float(eps_dgf),
            'mean_gabor_frame_norm': float(q.sqrt().mean()),
            'max_gabor_frame_norm': float(q.sqrt().max()),
            'feasible_frac': float((q <= eps2 * (1 + 1e-6)).float().mean()),
        }
        print(f"  Gabor constraint: mean||d||_M={attack_stats['mean_gabor_frame_norm']:.6g} "
              f"eps_scale={attacker.eps_scale:.6g} "
              f"eps_dgf={attack_stats['eps_dgf']:.6g} "
              f"feasible={attack_stats['feasible_frac']*100:.1f}%")

    print(f"  Generated {len(x_clean)} adversarial examples")
    return x_clean, x_adv, y_true, attack_stats


def evaluate_transferability(target_model, x_clean, x_adv, y_true,
                              metrics_evaluator, device, batch_size=16):
    target_model.eval()
    # Collect every metric the evaluator returns. LPIPS/SSIM were previously
    # computed and discarded. `std_*` keys are skipped on purpose: a mean of
    # per-batch standard deviations is not the population standard deviation.
    # Sample-weighted, not a mean of per-batch means: the final batch is usually
    # short (8 of 16 at n=1000) and would otherwise count as much as a full one.
    tot, cnt = {}, {}
    n = len(x_clean)
    for i in range((n + batch_size - 1) // batch_size):
        s, e = i * batch_size, min((i + 1) * batch_size, n)
        bs = e - s
        bm = metrics_evaluator.compute_all_metrics(
            target_model, x_clean[s:e].to(device),
            x_adv[s:e].to(device), y_true[s:e].to(device))
        for k, v in bm.items():
            if v is None or k.startswith('std_'):
                continue
            tot[k] = tot.get(k, 0.0) + float(v) * bs
            cnt[k] = cnt.get(k, 0) + bs
    return {k: (tot[k] / cnt[k] if cnt.get(k) else None) for k in tot}


def evaluate_all_transferability(models_dict, attacker, metrics_evaluator,
                                  testloader, case, args, checkpoint_dir=None):
    results = {}
    attack_stats = {}
    model_names = list(models_dict.keys())

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        for source_name in model_names:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{source_name}.json")
            if not os.path.exists(ckpt_path):
                continue
            with open(ckpt_path) as f:
                cached = json.load(f)
            # A checkpoint written by a run with a different model set would
            # leave holes that crash matrix construction later. Only reuse a
            # checkpoint that covers exactly the current targets.
            missing = [t for t in model_names if t not in cached]
            if missing:
                print(f"  [resume] Ignoring stale checkpoint for {source_name} "
                      f"(missing {len(missing)} target(s): {', '.join(missing[:3])}"
                      f"{'...' if len(missing) > 3 else ''})")
                continue
            results[source_name] = cached
            print(f"  [resume] Loaded checkpoint for {source_name}")

    print("\n" + "=" * 80)
    print("TRANSFERABILITY EVALUATION".center(80))
    print("=" * 80)
    for source_name in model_names:
        if source_name in results:
            print(f"\nSOURCE: {source_name} [SKIPPED - checkpoint]")
            for target_name in model_names:
                m = results[source_name].get(target_name, {})
                tag = "[SELF]" if target_name == source_name else "      "
                asr = m.get('attack_success_rate')
                adv = m.get('adversarial_accuracy')
                asr_str = f"{asr*100:.1f}%" if asr is not None else "N/A"
                adv_str = f"{adv*100:.1f}%" if adv is not None else "N/A"
                print(f"  {tag} -> {target_name}: ASR={asr_str}, AdvAcc={adv_str}")
            continue

        results[source_name] = {}
        print(f"\nSOURCE: {source_name}")
        rt = _rtlog(args)
        # The attack phase and the sample count are opened INSIDE
        # generate_adversarial_examples, per batch, so that dataloading and the
        # device copies stay out of the measured attack time -- same bracket as
        # eval_cifar100.py. Do not re-wrap the call here or it double-counts.
        x_clean, x_adv, y_true, src_stats = generate_adversarial_examples(
            models_dict[source_name], source_name, attacker,
            testloader, args.device, case, args.verbose, args=args)
        if src_stats:
            attack_stats[source_name] = src_stats
        for target_name in model_names:
            with rt.phase(case, "metrics", source=source_name):
                m = evaluate_transferability(
                    models_dict[target_name],
                    x_clean, x_adv, y_true,
                    metrics_evaluator, args.device, args.batch_size)
            results[source_name][target_name] = m
            tag = "[SELF]" if target_name == source_name else "      "
            asr = m['attack_success_rate']
            adv = m['adversarial_accuracy']
            asr_str = f"{asr*100:.1f}%" if asr is not None else "N/A"
            adv_str = f"{adv*100:.1f}%" if adv is not None else "N/A"
            print(f"  {tag} -> {target_name}: ASR={asr_str}, AdvAcc={adv_str}")

        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{source_name}.json")
            serializable = {
                t: {k: float(v) if v is not None else None for k, v in mvals.items()}
                for t, mvals in results[source_name].items()
            }
            with open(ckpt_path, 'w') as f:
                json.dump(serializable, f, indent=2)

    if checkpoint_dir and attack_stats:
        with open(os.path.join(checkpoint_dir, 'attack_stats.json'), 'w') as f:
            json.dump(attack_stats, f, indent=2)

    return results


def create_transferability_matrices(results):
    names = list(results.keys())
    n = len(names)
    asr = np.zeros((n, n))
    acc = np.zeros((n, n))
    for i, s in enumerate(names):
        for j, t in enumerate(names):
            m = results.get(s, {}).get(t) or {}
            a_val = m.get('attack_success_rate')
            c_val = m.get('adversarial_accuracy')
            asr[i, j] = a_val * 100 if a_val is not None else np.nan
            acc[i, j] = c_val * 100 if c_val is not None else np.nan
    return asr, acc


def plot_transferability_heatmap(matrix, model_names, title, output_path,
                                  cmap='coolwarm', fmt='.1f'):
    plt.figure(figsize=(max(10, len(model_names)), max(8, len(model_names) - 2)))
    labels = [display_name(n) for n in model_names]
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 12}, vmin=0, vmax=100,
                linewidths=0.5, linecolor='gray')
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Target Model', fontsize=14, fontweight='bold')
    plt.ylabel('Source Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {output_path}")


def save_transferability_results(results, save_dir, case_label, save_heatmaps=True):
    os.makedirs(save_dir, exist_ok=True)
    names = list(results.keys())
    asr_m, acc_m = create_transferability_matrices(results)

    serializable = {
        s: {t: {k: float(v) if v is not None else None for k, v in m.items()}
            for t, m in tgt.items()}
        for s, tgt in results.items()
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    display = [display_name(n) for n in names]
    pd.DataFrame(asr_m, index=display, columns=display).to_csv(os.path.join(save_dir, 'asr.csv'))
    pd.DataFrame(acc_m, index=display, columns=display).to_csv(
        os.path.join(save_dir, 'accuracy.csv'))
    print(f"  Saved transferability results to {save_dir}")

    if save_heatmaps:
        plot_transferability_heatmap(
            asr_m, names, f'ASR (%) - {case_label}',
            os.path.join(save_dir, 'asr_heatmap.png'), cmap='coolwarm')
        plot_transferability_heatmap(
            acc_m, names, f'Adv Accuracy (%) - {case_label}',
            os.path.join(save_dir, 'accuracy_heatmap.png'), cmap='coolwarm')


# ============================================================================
# Image Generation and Saving
# ============================================================================

def _reshape_delta(delta, like):
    if delta.dim() in (1, 2) and delta.numel() == like.numel():
        return delta.reshape_as(like)
    return delta


def save_image_comparison(clean_images, adv_images, delta, labels,
                          predictions_clean, predictions_adv,
                          save_dir, case_name, model_name, num_images=20):
    os.makedirs(save_dir, exist_ok=True)
    delta = _reshape_delta(delta, clean_images)
    # Cap the side-by-side grid: at 4 in/column and dpi=150 a 40-image strip is
    # a 24000 px wide PNG. Individual images are still written in full by
    # save_individual_images().
    num_images = min(num_images, clean_images.shape[0], MAX_COMPARISON_COLUMNS)
    if num_images == 0:
        return

    fig, axes = plt.subplots(3, num_images, figsize=(4 * num_images, 12))
    if num_images == 1:
        axes = axes.reshape(3, 1)

    for i in range(num_images):
        clean_img = np.clip(clean_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        adv_img = np.clip(adv_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        delta_img = delta[i].cpu().permute(1, 2, 0).numpy()
        delta_disp = (delta_img - delta_img.min()) / (delta_img.max() - delta_img.min() + 1e-10)

        true_label = labels[i].item()
        pred_clean = predictions_clean[i].item()
        pred_adv = predictions_adv[i].item()

        axes[0, i].imshow(clean_img); axes[0, i].axis("off")
        axes[0, i].set_title(f"Clean\nTrue: {true_label}\nPred: {pred_clean}", fontsize=10)

        color = "red" if pred_adv != true_label else "green"
        axes[1, i].imshow(adv_img); axes[1, i].axis("off")
        axes[1, i].set_title(f"Adversarial\nTrue: {true_label}\nPred: {pred_adv}",
                             fontsize=10, color=color)

        axes[2, i].imshow(delta_disp); axes[2, i].axis("off")
        axes[2, i].set_title("δ (perturbation)", fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f"{model_name}_{case_name}_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison: {filepath}")


def save_individual_images(clean_images, adv_images, delta, labels,
                           save_dir, case_name, model_name, num_images=20):
    os.makedirs(save_dir, exist_ok=True)
    delta = _reshape_delta(delta, clean_images)
    num_images = min(num_images, clean_images.shape[0])

    for i in range(num_images):
        true_label = labels[i].item()
        clean_img = np.clip(clean_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        plt.figure(figsize=(4, 4)); plt.imshow(clean_img); plt.axis("off")
        plt.title(f"class_{true_label}", fontsize=12)
        plt.savefig(os.path.join(save_dir,
                    f"{model_name}_{case_name}_clean_{i:03d}_class{true_label}.png"),
                    dpi=150, bbox_inches="tight"); plt.close()

        adv_img = np.clip(adv_images[i].cpu().permute(1, 2, 0).numpy(), 0, 1)
        plt.figure(figsize=(4, 4)); plt.imshow(adv_img); plt.axis("off")
        plt.title(f"class_{true_label} (adv)", fontsize=12)
        plt.savefig(os.path.join(save_dir,
                    f"{model_name}_{case_name}_adv_{i:03d}_class{true_label}.png"),
                    dpi=150, bbox_inches="tight"); plt.close()

        delta_img = delta[i].cpu().permute(1, 2, 0).numpy()
        delta_disp = (delta_img - delta_img.min()) / (delta_img.max() - delta_img.min() + 1e-10)
        plt.figure(figsize=(4, 4)); plt.imshow(delta_disp); plt.axis("off")
        plt.title(f"class_{true_label} (delta)", fontsize=12)
        plt.savefig(os.path.join(save_dir,
                    f"{model_name}_{case_name}_delta_{i:03d}_class{true_label}.png"),
                    dpi=150, bbox_inches="tight"); plt.close()

    print(f"  Saved {num_images} clean/adv/delta images to: {save_dir}")


def save_gabor_spectrograms(clean_images, adv_images, delta, labels, Psi_2D,
                            save_dir, case_name, model_name, num_images=20):
    """2x3 grid: top = images (clean, adv, delta), bottom = 2D Gabor magnitude heatmaps."""
    if Psi_2D is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    delta = _reshape_delta(delta, clean_images)
    num_images = min(num_images, clean_images.shape[0])

    Psi_2D = Psi_2D.to(clean_images.device)
    n = clean_images.shape[2]
    N = Psi_2D.shape[0]
    Psi_bc = Psi_2D.view(1, N, n)
    PsiT_bc = Psi_2D.t().view(1, n, N)

    def gabor2d_avg_magnitude(x_chw):
        x_chw = x_chw.to(dtype=Psi_2D.dtype)
        tmp = torch.matmul(Psi_bc, x_chw)
        w = torch.matmul(tmp, PsiT_bc)
        return torch.abs(w).mean(dim=0).detach().cpu().numpy()

    def compute_psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return float("inf")
        max_val = max(float(a.max()), float(b.max()))
        if max_val == 0:
            return float("inf")
        return 20 * np.log10(max_val / np.sqrt(mse))

    case_labels = {
        "case1": "Proposed attack δ",
        "case2": "Standard PGD attack δ",
        "case3": "Fourier-based PGD attack δ",
        "case4": "AutoAttack δ",
        "case5": "SSA attack δ",
        "case6": "AdvDrop attack δ",
    }

    for idx in range(num_images):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        true_label = labels[idx].item()

        clean_img = np.clip(clean_images[idx].cpu().permute(1, 2, 0).numpy(), 0, 1)
        adv_img = np.clip(adv_images[idx].cpu().permute(1, 2, 0).numpy(), 0, 1)
        pert = delta[idx].cpu().permute(1, 2, 0).numpy()

        ssim_value = None
        if SSIM_AVAILABLE:
            try:
                with torch.no_grad():
                    ssim_value = _ssim(clean_images[idx:idx+1].to(Psi_2D.device).float(),
                                       adv_images[idx:idx+1].to(Psi_2D.device).float(),
                                       data_range=1.0, size_average=True).item()
            except Exception:
                ssim_value = None

        axes[0, 0].imshow(clean_img); axes[0, 0].axis("off")
        axes[0, 0].set_title(f"Clean image\nclass_{true_label}", fontsize=18)

        title = f"x_adv\nclass_{true_label}"
        if ssim_value is not None:
            title += f"\nSSIM: {ssim_value:.4f}"
        axes[0, 1].imshow(adv_img); axes[0, 1].axis("off")
        axes[0, 1].set_title(title, fontsize=18)

        pert_disp = (pert - pert.min()) / (pert.max() - pert.min() + 1e-10)
        axes[0, 2].imshow(pert_disp); axes[0, 2].axis("off")
        axes[0, 2].set_title(case_labels.get(case_name, "δ"), fontsize=18)

        clean_g = gabor2d_avg_magnitude(clean_images[idx])
        adv_g = gabor2d_avg_magnitude(adv_images[idx])
        delta_g = gabor2d_avg_magnitude(delta[idx])

        psnr_adv = compute_psnr(clean_g, adv_g)
        psnr_d = compute_psnr(clean_g, delta_g)

        im1 = axes[1, 0].imshow(clean_g, cmap="coolwarm", aspect="auto")
        axes[1, 0].set_title("PSNR: ∞ dB", fontsize=18); axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        im2 = axes[1, 1].imshow(adv_g, cmap="coolwarm", aspect="auto")
        axes[1, 1].set_title(f"PSNR: {psnr_adv:.4f} dB", fontsize=18); axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        im3 = axes[1, 2].imshow(delta_g, cmap="coolwarm", aspect="auto")
        axes[1, 2].set_title(f"PSNR: {psnr_d:.4f} dB", fontsize=18); axes[1, 2].axis("off")
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

        plt.tight_layout()
        filepath = os.path.join(
            save_dir, f"{model_name}_{case_name}_spectrogram_{idx:03d}_class{true_label}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved {num_images} spectrograms to: {save_dir}")


def generate_and_save_images(model, model_name, attacker, testloader, case, args,
                              save_dir, Psi_2D):
    num_images = args.num_images
    # --num-images 0 turns the dump off. It re-runs the attack on a fresh batch
    # purely to have pictures, which a cost sweep pays once per (model, case)
    # for output it never looks at -- and for case4 it is a second full
    # AutoAttack. The dump is unmeasured either way, so this only buys back
    # wall-clock, never changes a number.
    if num_images <= 0:
        return
    img_dir = save_dir
    os.makedirs(img_dir, exist_ok=True)

    if case == "case4":
        try:
            from autoattack import AutoAttack
        except ImportError:
            raise ImportError("autoattack not installed")
        collected_clean, collected_labels = [], []
        for x, y in testloader:
            collected_clean.append(x)
            collected_labels.append(y)
            if sum(len(t) for t in collected_clean) >= num_images:
                break
        x_clean = torch.cat(collected_clean, dim=0)[:num_images]
        y_true = torch.cat(collected_labels, dim=0)[:num_images]
        adversary = AutoAttack(model, norm=args.aa_norm, eps=args.epsilon,
                               version=args.aa_version, device=args.device,
                               verbose=args.verbose)
        x_adv = adversary.run_standard_evaluation(
            x_clean.to(args.device), y_true.to(args.device),
            bs=args.batch_size).cpu()
    else:
        attacker.model = model
        model.eval()
        collected_clean, collected_adv, collected_labels = [], [], []
        for x, y in testloader:
            x, y = x.to(args.device), y.to(args.device)
            if case == 'case1':
                x_adv_b, _, _ = attacker(x, y, random_init=False)
            else:
                x_adv_b, _ = attacker(x, y, random_init=False)
            collected_clean.append(x.cpu())
            collected_adv.append(x_adv_b.cpu())
            collected_labels.append(y.cpu())
            if sum(len(t) for t in collected_clean) >= num_images:
                break
        x_clean = torch.cat(collected_clean, dim=0)[:num_images]
        x_adv = torch.cat(collected_adv, dim=0)[:num_images]
        y_true = torch.cat(collected_labels, dim=0)[:num_images]

    model.eval()
    with torch.no_grad():
        pred_clean = model(x_clean.to(args.device)).argmax(dim=1).cpu()
        pred_adv = model(x_adv.to(args.device)).argmax(dim=1).cpu()
    delta = x_adv - x_clean

    save_image_comparison(x_clean, x_adv, delta, y_true, pred_clean, pred_adv,
                          img_dir, case, model_name, num_images)
    save_individual_images(x_clean, x_adv, delta, y_true,
                           img_dir, case, model_name, num_images)
    save_gabor_spectrograms(x_clean, x_adv, delta, y_true, Psi_2D,
                            img_dir, case, model_name, num_images)


# ============================================================================
# Main
# ============================================================================

def _group_dir(args, group_name, case, eps_tag, hp_subpath):
    name = run_dir_name("imagenet", group_name, case, eps_tag, hp_subpath)
    # A suffix rather than a nested folder, so the tree stays one level deep;
    # eval_cifar100.py does the same.
    if getattr(args, "timing", False):
        name += "_timing"
    return os.path.join(args.output_dir, name)


def _run_transferability_split(robust_models, standard_models, attacker,
                               metrics_evaluator, testloader, case, args,
                               eps_tag, hp_subpath, label, save_heatmaps):
    for group_name, group_models in (("adv", robust_models), ("standard", standard_models)):
        if not group_models:
            continue
        save_dir = _group_dir(args, group_name, case, eps_tag, hp_subpath)
        print(f"\n--- Transferability [{group_name}] ({len(group_models)} models) ---")
        results = evaluate_all_transferability(
            group_models, attacker, metrics_evaluator, testloader, case, args,
            checkpoint_dir=save_dir)
        save_transferability_results(
            results, save_dir, f"{label}_{group_name}", save_heatmaps)


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 80)
    print("ImageNet ADVERSARIAL EVALUATION".center(80))
    print("=" * 80)

    cases = [f"case{c}" for c in args.attack_case]

    epsilon = args.epsilon
    eps_tag = format_eps_tag(epsilon)

    testloader = load_imagenet(args)

    robust_models = {}
    standard_models = {}

    # Auto-detect: names in STANDARD_PRETRAINED_NAMES → standard loader,
    # everything else → RobustBench loader.
    std_requested = [n for n in args.models if n in STANDARD_PRETRAINED_NAMES]
    adv_requested = [n for n in args.models if n not in STANDARD_PRETRAINED_NAMES]
    print(f"\nAuto-detecting model groups from --models...")
    print(f"  Standard: {std_requested or '(none)'}")
    print(f"  Adv (RobustBench): {adv_requested or '(none)'}")
    # Model loading happens once and is shared by every case, so it is booked
    # against the pseudo-method "_shared" rather than charged to whichever case
    # happens to run first.
    # Everything a cost table needs to be auditable goes in the meta. Without
    # the roster a runtime file cannot be told apart from one produced on a
    # different model set, and src/aggregate_run.py sums same-(source, case)
    # files -- which is how a stale 6-model smoke run was once folded into a
    # 2-model case1 row unnoticed. The attack hyperparameters are here so
    # bin/report_cost.py can derive the model-pass count from the file itself
    # instead of trusting a manifest that may have moved on.
    rt = RuntimeLog("imagenet", device=args.device, timing=args.timing, meta={
        "a": args.a, "b": args.b, "window": args.window_type,
        "num_steps": args.num_steps,
        "epsilon": epsilon, "batch_size": args.batch_size,
        "timing": args.timing, "amp": not args.no_amp,
        "timing_warmup": args.timing_warmup if args.timing else 0,
        "warmup_batches": 0 if args.timing else args.warmup_batches,
        "num_samples": args.num_samples,
        "ssa_N": args.ssa_N, "ssa_steps": args.ssa_steps,
        "ssa_rho": args.ssa_rho, "ssa_sigma": args.ssa_sigma,
        "ssa_momentum": args.ssa_momentum,
        "advdrop_steps": args.advdrop_steps,
        "advdrop_q_size": args.advdrop_q_size,
        "advdrop_block_size": args.advdrop_block_size,
        "advdrop_lr": args.advdrop_lr,
        "aa_version": args.aa_version, "aa_norm": args.aa_norm})
    args._rtlog = rt

    with rt.phase("_shared", "load"):
        if std_requested:
            orig = args.models
            args.models = std_requested
            standard_models = load_standard_pretrained_models(args)
            args.models = orig
        if adv_requested:
            orig = args.models
            args.models = adv_requested
            robust_models = load_imagenet_models(args)
            args.models = orig

    all_models = {**robust_models, **standard_models}
    print(f"\nTotal models for transferability: {len(all_models)}")
    # recorded only now, once the roster is actually resolved
    rt.meta["source_models"] = sorted(all_models)
    rt.meta["n_source_models"] = len(all_models)
    # Parameter count per model, read off the model that actually ran rather
    # than from a table of published figures: bin/report_cost.py and
    # bin/runtime_to_csv.py attribute cost to capacity with this, so it has to
    # come from the same object whose forward pass was timed.
    rt.meta["model_params"] = {
        name: int(sum(p.numel() for p in m.parameters()))
        for name, m in all_models.items()}
    # Which family each model belongs to, from the same membership the output
    # directories use (_model_group), so a per-family aggregate never has to
    # re-derive it from a name list that could drift out of sync.
    rt.meta["model_family"] = {
        name: ("adv_trained" if name in robust_models else "pretrained")
        for name in all_models}

    image_gen_names = set(all_models.keys())

    def _model_group(name):
        return "adv" if name in robust_models else "standard"

    print(f"\n{'#'*80}")
    print(f"  EPSILON = {epsilon:.6f}  ({eps_tag})".center(80))
    print(f"{'#'*80}")

    for case in cases:
        print(f"\n{'='*80}")
        print(f"  CASE: {case}  |  eps={epsilon:.4f}".center(80))
        print(f"{'='*80}")

        # ------------------------------------------------------------------
        # Case 4: AutoAttack
        # ------------------------------------------------------------------
        if case == "case4":
            metrics_evaluator = _build_metrics(args)
            hp_subpath = case_hparam_token(case, args)
            label = f"{case}_aa_{args.aa_version}_{args.aa_norm}"

            _run_transferability_split(
                robust_models, standard_models, None, metrics_evaluator,
                testloader, case, args, eps_tag, hp_subpath, label,
                args.save_heatmaps)

            (Psi_2D_spec, *_) = generate_gabor_operators(
                args.device, args.a, args.b, 224, args.window_type)

            for mname in image_gen_names:
                model = all_models[mname]
                img_dir = os.path.join(
                    _group_dir(args, _model_group(mname), case, eps_tag, hp_subpath),
                    "images", mname)
                print(f"\n  Image gen [{mname}]")
                generate_and_save_images(model, mname, None, testloader, case,
                                         args, img_dir, Psi_2D_spec)

            del Psi_2D_spec, metrics_evaluator
            torch.cuda.empty_cache()
            gc.collect()

        # ------------------------------------------------------------------
        # Cases 5, 6: vendored baselines (SSA / AdvDrop). No Gabor operator.
        # ------------------------------------------------------------------
        elif case in ("case5", "case6"):
            metrics_evaluator = _build_metrics(args)
            # Every knob that changes the attack must reach the directory name:
            # save_dir IS checkpoint_dir, and resume only validates the target
            # model set (see evaluate_all_transferability), so two runs sharing
            # a path would silently reload each other's numbers.
            hp_subpath = case_hparam_token(case, args)
            label = f"{case}_{hp_subpath}"

            attacker = build_baseline_attacker(
                case, list(all_models.values())[0], args, 224)

            # _run_transferability_split -> evaluate_all_transferability already
            # opens rt.phase(case, "attack") and records sample counts per source.
            _run_transferability_split(
                robust_models, standard_models, attacker, metrics_evaluator,
                testloader, case, args, eps_tag, hp_subpath, label,
                args.save_heatmaps)

            # Spectrograms need a Gabor frame purely for display; build the spec
            # operator like case 4 does, without wiring it into the attack.
            (Psi_2D_spec, *_) = generate_gabor_operators(
                args.device, args.a, args.b, 224, args.window_type)

            for mname in image_gen_names:
                model = all_models[mname]
                img_dir = os.path.join(
                    _group_dir(args, _model_group(mname), case, eps_tag, hp_subpath),
                    "images", mname)
                print(f"\n  Image gen [{mname}]")
                generate_and_save_images(model, mname, attacker, testloader, case,
                                         args, img_dir, Psi_2D_spec)

            del attacker, metrics_evaluator, Psi_2D_spec
            torch.cuda.empty_cache()
            gc.collect()

        # ------------------------------------------------------------------
        # Cases 1, 2, 3: fixed hparams from CLI
        # ------------------------------------------------------------------
        else:
            a, b = args.a, args.b
            window = args.window_type
            num_steps = args.num_steps

            with rt.phase(case, "load"):
                (Psi_2D, D_inv_1_2D, M_2D,
                 eps_scale, mu_M_2D, U_M_2D, _
                 ) = generate_gabor_operators(args.device, a, b, 224, window)

            # Case 1 alone uses gamma (2/3 derive their step from eps/num_steps).
            gamma, gamma_why = resolve_gamma(args)
            if case == "case1":
                warn_if_constraint_inert(gamma, num_steps, eps_scale, epsilon,
                                         gamma_why)

            # Only the hyperparameters this method actually uses; case1 gets the
            # Gabor frame and gamma, cases 2/3 only num_steps.
            hp_subpath = case_hparam_token(
                case, args, a=a, b=b, window=window, gamma=gamma,
                num_steps=num_steps)
            label = f"{case}_{hp_subpath}"

            attacker = _build_attacker(
                list(all_models.values())[0],
                Psi_2D, D_inv_1_2D, M_2D,
                eps_scale, mu_M_2D, U_M_2D,
                epsilon, gamma, num_steps, case, args)
            metrics_evaluator = _build_metrics(args)

            _run_transferability_split(
                robust_models, standard_models, attacker, metrics_evaluator,
                testloader, case, args, eps_tag, hp_subpath, label,
                args.save_heatmaps)

            for mname in image_gen_names:
                model = all_models[mname]
                img_dir = os.path.join(
                    _group_dir(args, _model_group(mname), case, eps_tag, hp_subpath),
                    "images", mname)
                print(f"\n  Image gen [{mname}]")
                generate_and_save_images(model, mname, attacker, testloader, case,
                                         args, img_dir, Psi_2D)

            del attacker, metrics_evaluator
            del Psi_2D, D_inv_1_2D, M_2D, mu_M_2D, U_M_2D
            torch.cuda.empty_cache()
            gc.collect()

    # One raw runtime.json per run in a SHARED dir (--runtime-dir, set by
    # run_row.py to <results root>/runtime); dataset/cases/accelerator/job id
    # in the name so concurrent array tasks never overwrite each other.
    case_tag = "case" + "-".join(str(c) for c in args.attack_case)
    # "timing" in the name so a batch-size-1 cost run can never be folded into
    # a production run's totals by the aggregator
    # start time + pid in the name because repeat local passes are otherwise
    # identical: without them three runs collide on 'local_0' and the last wins
    rt_name = (f"runtime_imagenet_{case_tag}"
               f"{'_timing' if args.timing else ''}"
               f"_{rt.env['accelerator']}"
               f"_{rt.env.get('slurm_job_id') or 'local'}"
               f"_{rt.env.get('slurm_array_task_id') or '0'}"
               f"_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}"
               f"_{eps_tag}.json")
    rt_dir = args.runtime_dir or os.path.join(args.output_dir, "runtime")
    rt_path = rt.write(rt_dir, rt_name)
    rt.print_summary()
    print(f"  Runtime written to {rt_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
