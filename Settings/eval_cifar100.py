"""
CIFAR100 attack script
"""

import csv
import gc
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
import time
from typing import Dict, List

import paths
from transforms import *
from dgf_pgd import DGFPGDAttack
from eval_imagenet import (generate_gabor_operators, resolve_gamma,
                           warn_if_constraint_inert)
from run_dirs import case_hparam_token, format_eps_tag, run_dir_name
from baseline_attacks import build_baseline_attacker
from cases import attack_name
from runtime_log import RuntimeLog
from eval_imagenet import _rtlog, _timing_discard, warmup_attack
from evaluation_metrics import AdversarialMetrics
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt

try:
    from pytorch_msssim import ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print(
        "Warning: pytorch_msssim not available. SSIM values will not be shown in spectrograms."
    )


DISPLAY_NAMES: Dict[str, str] = {
    # RobustBench Linf CIFAR-100 models (--model-source robustbench)
    "Addepalli2021Towards_WRN34":           "Addepalli21",
    "Addepalli2022Efficient_WRN_34_10":     "Addepalli22",
    "Amini2024MeanSparse_S-WRN-70-16":      "Amini24",
    "Bai2023Improving_trades":              "Bai23",
    "Bai2024MixedNUTS":                     "Bai24",
    "Chen2024Data_WRN_34_10":               "Chen24",
    "Cui2020Learnable_34_20_LBGAT6":        "Cui20",
    "Cui2023Decoupled_WRN-34-10":           "Cui23",
    "Debenedetti2022Light_XCiT-L12":        "Debenedetti22",
    "Gowal2020Uncovering":                  "Gowal20",
    "Gowal2020Uncovering_extra":            "Gowal20-extra",
    "Jia2022LAS-AT_34_10":                  "Jia22",
    "Pang2022Robustness_WRN28_10":          "Pang22-WRN28",
    "Pang2022Robustness_WRN70_16":          "Pang22-WRN70",
    "Rebuffi2021Fixing_28_10_cutmix_ddpm":  "Rebuffi21-WRN28",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm":  "Rebuffi21-WRN70",
    "Rice2020Overfitting":                  "Rice20",
    "Wu2020Adversarial":                    "Wu20",
    # chenyaofo pretrained backbones (--model-source pretrained)
    "resnet20":            "ResNet20",
    "resnet32":            "ResNet32",
    "resnet44":            "ResNet44",
    "resnet56":            "ResNet56",
    "mobilenetv2_x0_5":    "MobileNet0.5",
    "mobilenetv2_x0_75":   "MobileNet0.75",
    "mobilenetv2_x1_0":    "MobileNet1.0",
    "mobilenetv2_x1_4":    "MobileNet1.4",
    "shufflenetv2_x0_5":   "ShuffleNet_x0_5",
    "shufflenetv2_x2_0":   "ShuffleNet_x2_0",
    "repvgg_a0":           "RepVGG_a0",
    "repvgg_a1":           "RepVGG_a1",
    "repvgg_a2":           "RepVGG_a2",
    "vgg11_bn":            "VGG11_bn",
    "vgg13_bn":            "VGG13_bn",
    "vgg16_bn":            "VGG16_bn",
    "vgg19_bn":            "VGG19_bn",
}


def display_name(key: str) -> str:
    """Return the paper-friendly label for a model key."""
    return DISPLAY_NAMES.get(key, key)


# ============================================================================
# Command-line Arguments
# ============================================================================


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CIFAR100 DGF-PGD Attack Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="robustbench",
        choices=["robustbench", "pretrained"],
        help="Model source: robustbench (Linf robust models) or pretrained (chenyaofo backbones)",
    )

    # Case selection
    parser.add_argument(
        "--case",
        type=str,
        choices=["case1", "case2", "case3", "case4", "case5", "case6"],
        help="Run Case 1/2/3 (DGF-PGD variants), Case 4 (AutoAttack), "
        "Case 5 (SSA) or Case 6 (AdvDrop). One case per run.",
    )

    # Data parameters
    parser.add_argument(
        "--data-root",
        type=str,
        default=paths.CIFAR100_ROOT,
        help="CIFAR100 root, i.e. the directory containing cifar-100-python/ "
        "(default: the repo's data/CIFAR100)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-samples", type=int, default=10000, help="Number of test samples"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Data loading workers"
    )

    # Attack parameters
    parser.add_argument(
        "--epsilon", type=float, default=32 / 255, help="Attack epsilon (L2 norm)"
    )
    parser.add_argument("--gamma", type=float, default=None,
                        help="Step size")
    parser.add_argument("--num-steps", type=int, default=10, help="PGD iterations")
    parser.add_argument(
        "--tau", type=float, default=0.1, help="Soft-thresholding parameter (Case 1) - DEPRECATED - ignored, kept for backwards compatibility"
    )
    parser.add_argument("--a", type=int, default=1, help="Time lattice parameter")
    parser.add_argument("--b", type=int, default=16, help="Frequency lattice parameter")
    parser.add_argument("--rho", type=float, default=1.0,
                        help="DEPRECATED, ignored. The operator no longer uses rho; "
                             "kept only so existing command lines still parse.")
    parser.add_argument(
        "--window-type",
        type=str,
        default="Hann",
        choices=["Hann", "Blackman", "Gaussian"],
        help="Type of Gabor window function",
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (default: all available)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=paths.MODELS_DIR,
        help="RobustBench weight tree, models/<dataset>/<threat_model>/"
        "<name>.pt (default: the repo's models/). --model-source pretrained "
        "uses <models-dir>/torch_hub as the torch.hub cache.",
    )

    # Case 4 (AutoAttack) hyperparameters; mirror eval_imagenet.py.
    parser.add_argument("--aa-norm", type=str, default="Linf",
                        choices=["Linf", "L2"],
                        help="AutoAttack threat model (case 4).")
    parser.add_argument("--aa-version", type=str, default="standard",
                        choices=["standard", "rand", "custom"],
                        help="AutoAttack version (case 4).")

    # Case 5 (SSA) hyperparameters; mirror baselines/SSA/attack.py.
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

    # Case 6 (AdvDrop) hyperparameters; mirror baselines/AdvDrop/infod_sample.py.
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

    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"]
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save clean and adversarial images"
    )
    parser.add_argument(
        "--num-images", type=int, default=10, help="Number of image pairs to save"
    )
    
    parser.add_argument(
        "--runtime-dir",
        type=str,
        default=None,
        help="Shared directory for the raw runtime_*.json files "
        "(default: <output-dir>/runtime)",
    )

    # Timing
    parser.add_argument(
        "--no-amp", action="store_true")
    parser.add_argument(
        "--timing", action="store_true"
    )
    parser.add_argument(
        "--warmup-batches", type=int, default=1
    )
    parser.add_argument(
        "--timing-warmup", type=int, default=5
    )

    # Other
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.timing:
        if args.batch_size != 1:
            print(f"[timing] forcing --batch-size 1 (was {args.batch_size})")
        args.batch_size = 1
        args.timing_warmup = max(0, args.timing_warmup)

    if not args.case:
        print("No case specified. Running default (Case 1).")
        args.case = "case1"

    run_dir = run_dir_name("cifar100", args.model_source, args.case,
                           format_eps_tag(args.epsilon),
                           case_hparam_token(args.case, args,
                                             a=args.a, b=args.b,
                                             window=args.window_type,
                                             gamma=resolve_gamma(args)[0],
                                             num_steps=args.num_steps))

    if args.timing:
        run_dir += "_timing"

    args.output_dir = os.path.join(args.output_dir, run_dir)
    args.image_dir = os.path.join(args.output_dir, "images")

    return args


# ============================================================================
# Dataset Loading
# ============================================================================


def load_cifar100(args):
    """Load CIFAR100 test data"""
    print("\nLoading CIFAR100 dataset...")

    transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False, transform=transform
    )

    n_want = args.num_samples
    if n_want is not None and getattr(args, "timing", False):
        n_want += args.timing_warmup
    if n_want is not None and n_want < len(testset):
        indices = np.random.choice(len(testset), n_want, replace=False)
        testset = Subset(testset, indices)

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Loaded {len(testset)} test samples")
    if getattr(args, "timing", False):
        print(f"  [timing] {args.timing_warmup} warm-up + {args.num_samples} "
              f"measured, one image per batch")
    return testloader


def load_cifar100_models(args):
    """Load pretrained CIFAR100 models"""

    if args.model_source == "robustbench":
        return load_robustbench_models(args)
    elif args.model_source == "pretrained":
        return load_pretrained_backbones(args)
    else:
        raise ValueError(f"Unknown model source: {args.model_source}")


def load_robustbench_models(args):
    """Load pretrained CIFAR100 models from RobustBench (Linf threat model)"""
    print("\nLoading CIFAR100 models from RobustBench (Linf threat model)...")
    print(f"  weights: {os.path.join(args.models_dir, 'cifar100', 'Linf')}")

    try:
        from robustbench.utils import load_model as rb_load_model
    except ImportError:
        print("\nERROR: robustbench not installed!")
        print("Install with: pip install robustbench")
        raise ImportError("robustbench is required for RobustBench models")

    model_names = [
        "Addepalli2021Towards_WRN34",
        "Addepalli2022Efficient_WRN_34_10",
        "Amini2024MeanSparse_S-WRN-70-16",
        "Bai2023Improving_trades",
        "Bai2024MixedNUTS",
        "Chen2024Data_WRN_34_10",
        "Cui2020Learnable_34_20_LBGAT6",
        "Cui2023Decoupled_WRN-34-10",
        "Debenedetti2022Light_XCiT-L12",
        "Gowal2020Uncovering",
        "Gowal2020Uncovering_extra",
        "Jia2022LAS-AT_34_10",
        "Pang2022Robustness_WRN28_10",
        "Pang2022Robustness_WRN70_16",
        "Rebuffi2021Fixing_28_10_cutmix_ddpm",
        "Rebuffi2021Fixing_70_16_cutmix_ddpm",
        "Rice2020Overfitting",
        "Wu2020Adversarial",
    ]

    # Filter if specific models requested
    if args.models:
        model_names = [m for m in model_names if m in args.models]

    if not model_names:
        raise ValueError(f"No valid models specified. Available: {model_names}")

    print(f"Loading {len(model_names)} models:")
    for name in model_names:
        print(f"  - {name}")

    models_dict = {}

    for model_name in model_names:
        try:
            print(f"\n  Loading {model_name}...")
            model = rb_load_model(
                model_name=model_name,
                dataset="cifar100",
                threat_model="Linf",
                model_dir=args.models_dir,
            ).to(args.device)
            models_dict[model_name] = model.eval()
            print(f"    ✓ Success")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:100]}")

    if not models_dict:
        raise RuntimeError("No models loaded successfully!")

    print(f"\n✓ Successfully loaded {len(models_dict)} models")
    return models_dict


def load_pretrained_backbones(args):
    """Load pretrained backbone models from chenyaofo/pytorch-cifar-models"""
    print("\nLoading pretrained backbone models from PyTorch Hub...")
    # Hub repo + checkpoints live in the repo-local cache (models/torch_hub),
    # populated in advance by src/download_models.py -- compute nodes have no
    # internet, so torch.hub must find everything already on disk.
    hub = paths.point_torch_hub(args.models_dir)
    print(f"  torch.hub cache: {hub}")

    # Available pretrained backbones
    available_backbones = {
        "mobilenetv2_x0_5": "cifar100_mobilenetv2_x0_5",
        "mobilenetv2_x1_4": "cifar100_mobilenetv2_x1_4",
        "mobilenetv2_x0_75": "cifar100_mobilenetv2_x0_75",
        "mobilenetv2_x1_0": "cifar100_mobilenetv2_x1_0",
        "shufflenetv2_x2_0": "cifar100_shufflenetv2_x2_0",
        "shufflenetv2_x0_5": "cifar100_shufflenetv2_x0_5",
        "repvgg_a0": "cifar100_repvgg_a0",
        "repvgg_a1": "cifar100_repvgg_a1",
        "repvgg_a2": "cifar100_repvgg_a2",
        "resnet20": "cifar100_resnet20",
        "resnet32": "cifar100_resnet32",
        "resnet44": "cifar100_resnet44",
        "resnet56": "cifar100_resnet56",
        "vgg11_bn": "cifar100_vgg11_bn",
        "vgg13_bn": "cifar100_vgg13_bn",
        "vgg16_bn": "cifar100_vgg16_bn",
        "vgg19_bn": "cifar100_vgg19_bn",
    }

    # Default backbones if none specified
    default_backbones = {
        "resnet44": "cifar100_resnet44",
        "resnet56": "cifar100_resnet56",
        "vgg13_bn": "cifar100_vgg13_bn",
        "vgg16_bn": "cifar100_vgg16_bn",
        "mobilenetv2_x0_75": "cifar100_mobilenetv2_x0_75",
        "mobilenetv2_x1_0": "cifar100_mobilenetv2_x1_0",
        "repvgg_a1": "cifar100_repvgg_a1",
        "repvgg_a2": "cifar100_repvgg_a2",
    }

    # Filter if specific models requested
    if args.models:
        backbones_to_load = {
            k: v
            for k, v in available_backbones.items()
            if k in args.models or any(m.lower() in k for m in args.models)
        }
        if not backbones_to_load:
            # Falling back to the eight defaults here would run a DIFFERENT
            # roster than the one asked for and record it under the requested
            # name -- a per-model cost row would then describe eight models.
            raise ValueError(
                f"none of --models {args.models} matches a pretrained CIFAR-100 "
                f"backbone; available: {', '.join(sorted(available_backbones))}")
    else:
        backbones_to_load = default_backbones

    print(f"Loading {len(backbones_to_load)} backbone models:")
    for name in backbones_to_load.keys():
        print(f"  - {name}")

    # PretrainedBackbone wrapper class
    class PretrainedBackbone(nn.Module):
        def __init__(self, pretrained_model):
            super(PretrainedBackbone, self).__init__()
            self.pretrained_model = pretrained_model

            # chenyaofo models expect ImageNet normalization
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        def forward(self, x):
            # Normalize input before passing to model
            x = self.normalize(x)

            classifier = self.pretrained_model(x)

            return classifier

    models_dict = {}

    for model_name, hub_name in backbones_to_load.items():
        try:
            print(f"\n  Loading {model_name}...")
            # skip_validation avoids the GitHub API call torch.hub makes when
            # (re)fetching -- with the repo already cached this stays offline.
            pretrained_model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", hub_name, pretrained=True,
                skip_validation=True,
            )
            # Wrap in PretrainedBackbone class
            backbone = PretrainedBackbone(pretrained_model)
            models_dict[model_name] = backbone.to(args.device).eval()
            print(f"    ✓ Success")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:100]}")

    if not models_dict:
        raise RuntimeError("No backbone models loaded successfully!")

    print(f"\n✓ Successfully loaded {len(models_dict)} backbone models")
    return models_dict


# ============================================================================
# Gabor Operators
# ============================================================================


def generate_gabor_operators_cifar100(device, a, b, window_type="Gaussian"):
    """CIFAR-100 (32x32) operators, delegated to the shared builder.

    This used to be a private copy of the operator maths. That copy carried every
    defect the imagenet path was fixed for -- it called dual_norm1() before
    weights1(), so the in-place D.pow_(-1) made M = Psi* D^-1 Psi (P1); it used
    the Monte-Carlo estimator for D (P2); and its eps_scale exponent was off by a
    factor n (P5). Keeping it would have made CIFAR-100 report a different --
    and wrong -- operator from ImageNet in the same paper.

    Worse, dual_norm1() is pure now, so running that copy verbatim against the
    current transforms.py would produce M = Psi* D Psi with D_inv_1 = Psi* D^-1
    Psi: a hybrid matching NEITHER the paper NOR the historical implementation.

    Both datasets now go through eval_imagenet.generate_gabor_operators, so the
    operator means the same thing at 32x32 as it does at 224x224.
    Psi_plus is still built here; only the spectrogram plots use it.
    """
    print(f"\nGenerating 2D Gabor operators for CIFAR-100 (32x32), a={a}, b={b}")
    (Psi_2D, D_inv_1_2D, M_2D, eps_scale,
     mu_M_2D, U_M_2D, cond_S) = generate_gabor_operators(
        device, a, b, image_size=32, window_type=window_type)
    Psi_plus_2D = torch.linalg.pinv(Psi_2D)
    return (Psi_2D, Psi_plus_2D, D_inv_1_2D, M_2D, eps_scale,
            mu_M_2D, U_M_2D, cond_S)


# ============================================================================
# Evaluation
# ============================================================================


def _warmup_from_loader(dataloader, args, model_name, fn):
    """Warm the attack on one batch off `dataloader`, before the timed loop.
    """
    batch = next(iter(dataloader), None)
    if batch is None:
        return
    images, labels = batch[0].to(args.device), batch[1].to(args.device)
    warmup_attack(lambda: fn(images, labels), args, model_name)


def _accumulate_transfer(models_dict, images, x_adv, labels, acc, args, run_case,
                         source=None):
    """Batchwise cross-model counts: how this source's adversarials transfer.
    """
    if not models_dict or len(models_dict) < 2:
        return
    rt = _rtlog(args)
    with rt.phase(run_case, "metrics", per_sample=getattr(args, "timing", False),
                  source=source), torch.no_grad():
        for tname, tmodel in models_dict.items():
            a = acc.setdefault(
                tname, {"clean_correct": 0, "adv_correct": 0, "n": 0})
            a["clean_correct"] += int(
                (tmodel(images).argmax(dim=1) == labels).sum().item())
            a["adv_correct"] += int(
                (tmodel(x_adv).argmax(dim=1) == labels).sum().item())
            a["n"] += int(labels.numel())


def _finalize_transfer(acc):
    out = {}
    for tname, a in acc.items():
        n = max(a["n"], 1)
        adv = a["adv_correct"] / n
        out[tname] = {
            "clean_accuracy": a["clean_correct"] / n,
            "adversarial_accuracy": adv,
            "attack_success_rate": 1.0 - adv,
            "n_samples": a["n"],
        }
    return out


def plot_transferability_heatmap(matrix, model_names, title, output_path,
                                 cmap="coolwarm"):
    plt.figure(figsize=(max(8, 1.2 * len(model_names) + 4),
                        max(6, 1.0 * len(model_names) + 3)))
    try:
        import seaborn as sns
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap,
                    xticklabels=model_names, yticklabels=model_names,
                    cbar_kws={"label": title}, vmin=0, vmax=100,
                    linewidths=0.5, linecolor="gray")
    except ImportError:
        plt.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
        plt.colorbar(label=title)
        plt.xticks(range(len(model_names)), model_names)
        plt.yticks(range(len(model_names)), model_names)
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                plt.text(j, i, f"{matrix[i][j]:.1f}",
                         ha="center", va="center", fontsize=9)
    plt.title(title)
    plt.xlabel("Target Model")
    plt.ylabel("Source Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_transferability_reports(transfer_by_source, args):
    names = [s for s in transfer_by_source if transfer_by_source[s]]
    if len(names) < 2:
        return
    label = f"{attack_name(args.case)} [{args.model_source}]"
    asr = np.array([[100.0 * transfer_by_source[s][t]["attack_success_rate"]
                     for t in names] for s in names])
    acc = np.array([[100.0 * transfer_by_source[s][t]["adversarial_accuracy"]
                     for t in names] for s in names])
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    for fname, m in (("transferability_asr.csv", asr),
                     ("transferability_accuracy.csv", acc)):
        with open(os.path.join(out, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source\\target"] + names)
            for i, s in enumerate(names):
                w.writerow([s] + [f"{m[i, j]:.4f}" for j in range(len(names))])

    with open(os.path.join(out, "transferability_detailed.json"), "w") as f:
        json.dump(transfer_by_source, f, indent=2)

    with open(os.path.join(out, "transferability_stats.txt"), "w") as f:
        f.write(f"Transferability statistics - {label}\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Source model':<40} {'Self ASR':>9} {'Mean tr.':>9} "
                f"{'Std':>7} {'Max':>7} {'Min':>7}\n")
        f.write("-" * 100 + "\n")
        for i, s in enumerate(names):
            off = [asr[i, j] for j in range(len(names)) if j != i]
            f.write(f"{s:<40} {asr[i, i]:>8.1f}% {np.mean(off):>8.1f}% "
                    f"{np.std(off):>6.1f} {np.max(off):>6.1f} "
                    f"{np.min(off):>6.1f}\n")

    try:
        plot_transferability_heatmap(
            asr, names, f"Attack Success Rate (%) - {label}",
            os.path.join(out, "transferability_asr_heatmap.png"), "RdYlGn_r")
        plot_transferability_heatmap(
            acc, names, f"Adversarial Accuracy (%) - {label}",
            os.path.join(out, "transferability_accuracy_heatmap.png"), "RdYlGn")
        print(f"  ✓ Transferability heatmaps saved to {out}")
    except Exception as e:
        print(f"  ✗ ERROR saving transferability heatmaps: {e}")

    print(f"\nTransferability ASR % ({label}); rows=source, cols=target, "
          f"* = self-attack:")
    colw = max(len(n) for n in names) + 2
    print(" " * colw + "".join(f"{n[:14]:>16}" for n in names))
    for i, s in enumerate(names):
        cells = "".join(
            f"{asr[i, j]:>15.1f}{'*' if i == j else ' '}"
            for j in range(len(names)))
        print(f"{s:<{colw}}" + cells)


def evaluate_model(
    Psi_2D,
    Psi_plus_2D,
    model,
    model_name,
    attacker,
    metrics_evaluator,
    dataloader,
    run_case,
    args,
    models_dict=None,
):
    """Evaluate attack on a single model"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    results = {}
    if attacker is not None:
        attacker.model = model

    transfer_acc = {}

    images_saved = False

    # Case 1
    if run_case == "case1":
        print(f"\nCase 1: Soft-thresholded Frame Attack")
        print("-" * 80)
        attacker.case = "case1"

        case1_metrics = []
        _warmup_from_loader(dataloader, args, model_name,
                            lambda i, l: attacker(i, l, random_init=True))
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 1", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            rt = _rtlog(args)
            _t = getattr(args, "timing", False)
            with rt.phase("case1", "attack", per_sample=_t, source=model_name,
                          n=images.shape[0]):
                x_adv, eps_dgf, last_delta = attacker(images, labels, random_init=True)
            rt.add_samples("case1", images.shape[0], source=model_name)
            with rt.phase("case1", "metrics", per_sample=_t, source=model_name):
                metrics = metrics_evaluator.compute_all_metrics(
                    model, images, x_adv, labels
                )
            metrics["gabor_frame_norm"] = float(
                attacker.gabor_constraint(last_delta).sqrt().mean().item())
            metrics["eps_dgf"] = float(eps_dgf)
            case1_metrics.append(metrics)
            _accumulate_transfer(models_dict, images, x_adv, labels,
                                 transfer_acc, args, run_case, source=model_name)
            _timing_discard(rt, "case1", model_name, batch_idx, args)

            # Save images from first batch only
            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)

                # Save comparison image
                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case1",
                    model_name,
                    args.num_images,
                )

                # Also save individual images
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case1",
                    model_name,
                    args.num_images,
                )

                # Save Gabor spectrograms for Case 1
                print("  Generating Gabor spectrograms...")
                print(f"    last_delta shape: {last_delta.shape}")
                print(f"    Psi shape: {Psi_2D.shape}")
                print(f"    save_dir: {args.image_dir}")

                print("  Computing x̃ = Ψ† S_τ(Ψx) for visualization...")
                x_tilde = None
                try:
                    B, C, n, _ = images.shape

                    N = Psi_2D.shape[0]

                    Psi_2D = Psi_2D.to(torch.complex128).to(args.device)

                    Psi_bc = Psi_2D.view(1, 1, N, n)
                    PsiT_bc = Psi_2D.t().view(1, 1, n, N)
                    z = torch.matmul(Psi_bc, images.to(dtype=Psi_2D.dtype))
                    z = torch.matmul(z, PsiT_bc)

                    Psi_plus_2D = Psi_plus_2D.to(torch.complex128).to(args.device)

                    Psi_plus_bc = Psi_plus_2D.view(1, 1, n, N)
                    Psi_plusT_bc = Psi_plus_2D.t().view(1, 1, N, n)

                    x_tilde = torch.matmul(Psi_plus_bc, z)
                    x_tilde = torch.matmul(x_tilde, Psi_plusT_bc).real

                    magnitude = torch.abs(z)
                    zeroed_coeffs = (magnitude < args.tau).sum().item()
                    total_coeffs = z.numel()

                    # Clamp x_tilde to valid image range [0, 1]
                    x_tilde = torch.clamp(x_tilde, 0.0, 1.0)

                    sparsity_percent = 100.0 * zeroed_coeffs / total_coeffs
                    print(
                        f"    x_tilde computed, range: [{x_tilde.min():.4f}, {x_tilde.max():.4f}]"
                    )
                    print(
                        f"    Sparsity: {sparsity_percent:.2f}% coefficients zeroed ({zeroed_coeffs}/{total_coeffs})"
                    )
                except Exception as e:
                    x_tilde = None
                    print(f"  ✗ ERROR computing x_tilde (visualization only): {e}")
                    import traceback

                    traceback.print_exc()

                try:
                    save_gabor_spectrograms(
                        images,
                        x_adv,
                        last_delta,
                        labels,
                        Psi_2D,
                        args.image_dir,
                        "case1",
                        model_name,
                        args.num_images,
                        x_tilde=x_tilde,  # Pass x_tilde for Case 1
                    )
                    print("  ✓ Spectrograms saved successfully for Case 1")
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")
                    import traceback

                    traceback.print_exc()

                images_saved = True

        results["case1"] = aggregate_metrics(case1_metrics)
        print_summary(results["case1"], "Case 1")

    # Case 2
    elif run_case == "case2":
        print(f"\nCase 2: L2 PGD Attack")
        print("-" * 80)
        attacker.case = "case2"

        case2_metrics = []
        _warmup_from_loader(dataloader, args, model_name,
                            lambda i, l: attacker(i, l, random_init=True))
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 2", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            rt = _rtlog(args)
            _t = getattr(args, "timing", False)
            with rt.phase("case2", "attack", per_sample=_t, source=model_name,
                          n=images.shape[0]):
                x_adv, last_delta = attacker(images, labels, random_init=True)
            rt.add_samples("case2", images.shape[0], source=model_name)
            with rt.phase("case2", "metrics", per_sample=_t, source=model_name):
                metrics = metrics_evaluator.compute_all_metrics(
                    model, images, x_adv, labels
                )
            case2_metrics.append(metrics)
            _accumulate_transfer(models_dict, images, x_adv, labels,
                                 transfer_acc, args, run_case, source=model_name)
            _timing_discard(rt, "case2", model_name, batch_idx, args)

            # Save images from first batch only (if not already saved from case1)
            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)

                # Save comparison image
                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case2",
                    model_name,
                    args.num_images,
                )

                # Also save individual images
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case2",
                    model_name,
                    args.num_images,
                )

                try:
                    save_gabor_spectrograms(
                        images,
                        x_adv,
                        last_delta,
                        labels,
                        Psi_2D,
                        args.image_dir,
                        "case2",
                        model_name,
                        args.num_images,
                        x_tilde=None,  
                    )
                    print("  ✓ Spectrograms saved successfully for Case 2")
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")

                images_saved = True

        results["case2"] = aggregate_metrics(case2_metrics)
        print_summary(results["case2"], "Case 2")

    elif run_case == "case3":
        print(f"\nCase 3: Fourier-based PGD Attack")
        print("-" * 80)
        attacker.case = "case3"

        case3_metrics = []
        _warmup_from_loader(dataloader, args, model_name,
                            lambda i, l: attacker(i, l, random_init=False))
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 3", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            rt = _rtlog(args)
            _t = getattr(args, "timing", False)
            with rt.phase("case3", "attack", per_sample=_t, source=model_name,
                          n=images.shape[0]):
                x_adv, last_delta = attacker(images, labels, random_init=False)
            rt.add_samples("case3", images.shape[0], source=model_name)
            with rt.phase("case3", "metrics", per_sample=_t, source=model_name):
                metrics = metrics_evaluator.compute_all_metrics(
                    model, images, x_adv, labels
                )
            case3_metrics.append(metrics)
            _accumulate_transfer(models_dict, images, x_adv, labels,
                                 transfer_acc, args, run_case, source=model_name)
            _timing_discard(rt, "case3", model_name, batch_idx, args)

            # Save images from first batch only (if not already saved from case1 or case2)
            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)

                # Save comparison image
                save_image_comparison(
                    images,
                    x_adv,
                    labels,
                    pred_clean,
                    pred_adv,
                    args.image_dir,
                    "case3",
                    model_name,
                    args.num_images,
                )

                # Also save individual images
                save_individual_images(
                    images,
                    x_adv,
                    labels,
                    args.image_dir,
                    "case3",
                    model_name,
                    args.num_images,
                )

                try:
                    save_gabor_spectrograms(
                        images,
                        x_adv,
                        last_delta,
                        labels,
                        Psi_2D,
                        args.image_dir,
                        "case3",
                        model_name,
                        args.num_images,
                        x_tilde=None,  # Pass x_tilde for Case 3
                    )
                    print("  ✓ Spectrograms saved successfully for Case 3")
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")

                images_saved = True

        results["case3"] = aggregate_metrics(case3_metrics)
        print_summary(results["case3"], "Case 3")

    # Case 4: AutoAttack
    elif run_case == "case4":
        print(f"\nCase 4: AutoAttack ({args.aa_version}, {args.aa_norm})")
        print("-" * 80)
        try:
            from autoattack import AutoAttack
        except ImportError:
            raise ImportError(
                "autoattack is required for case4. Install with: pip install autoattack"
            )

        adversary = AutoAttack(
            model, norm=args.aa_norm, eps=args.epsilon,
            version=args.aa_version, device=args.device,
            verbose=args.verbose,
        )

        case4_metrics = []
        _warmup_from_loader(
            dataloader, args, model_name,
            lambda i, l: adversary.run_standard_evaluation(i, l, bs=i.shape[0]))
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="Case 4", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            rt = _rtlog(args)
            _t = getattr(args, "timing", False)
            with rt.phase("case4", "attack", per_sample=_t, source=model_name,
                          n=images.shape[0]):
                x_adv = adversary.run_standard_evaluation(
                    images, labels, bs=images.shape[0]
                )
            rt.add_samples("case4", images.shape[0], source=model_name)
            with rt.phase("case4", "metrics", per_sample=_t, source=model_name):
                metrics = metrics_evaluator.compute_all_metrics(
                    model, images, x_adv, labels
                )
            case4_metrics.append(metrics)
            _accumulate_transfer(models_dict, images, x_adv, labels,
                                 transfer_acc, args, run_case, source=model_name)
            _timing_discard(rt, "case4", model_name, batch_idx, args)

            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    pred_clean = model(images).argmax(dim=1)
                    pred_adv = model(x_adv).argmax(dim=1)
                save_image_comparison(
                    images, x_adv, labels, pred_clean, pred_adv,
                    args.image_dir, "case4", model_name, args.num_images,
                )
                save_individual_images(
                    images, x_adv, labels,
                    args.image_dir, "case4", model_name, args.num_images,
                )
                try:
                    save_gabor_spectrograms(
                        images, x_adv, (x_adv - images), labels, Psi_2D,
                        args.image_dir, "case4", model_name, args.num_images,
                        x_tilde=None,
                    )
                    print("  ✓ Spectrograms saved successfully for Case 4")
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")
                images_saved = True

        results["case4"] = aggregate_metrics(case4_metrics)
        print_summary(results["case4"], "Case 4")

    # Cases 5 and 6: vendored baselines (SSA / AdvDrop)
    elif run_case in ("case5", "case6"):
        pretty = {"case5": "SSA (Spectrum Simulation Attack)",
                  "case6": "AdvDrop (InfoDrop)"}[run_case]
        print(f"\nCase {run_case[-1]}: {pretty}")
        print("-" * 80)

        case_metrics = []
        _warmup_from_loader(dataloader, args, model_name,
                            lambda i, l: attacker(i, l, random_init=False))
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc=f"Case {run_case[-1]}", leave=False)
        ):
            images, labels = images.to(args.device), labels.to(args.device)
            rt = _rtlog(args)
            _t = getattr(args, "timing", False)
            with rt.phase(run_case, "attack", per_sample=_t, source=model_name,
                          n=images.shape[0]):
                x_adv, _ = attacker(images, labels, random_init=False)
            rt.add_samples(run_case, images.shape[0], source=model_name)
            with rt.phase(run_case, "metrics", per_sample=_t, source=model_name):
                metrics = metrics_evaluator.compute_all_metrics(
                    model, images, x_adv, labels
                )
            case_metrics.append(metrics)
            _accumulate_transfer(models_dict, images, x_adv, labels,
                                 transfer_acc, args, run_case, source=model_name)
            _timing_discard(rt, run_case, model_name, batch_idx, args)

            if args.save_images and not images_saved and batch_idx == 0:
                with torch.no_grad():
                    pred_clean = model(images).argmax(dim=1)
                    pred_adv = model(x_adv).argmax(dim=1)
                save_image_comparison(
                    images, x_adv, labels, pred_clean, pred_adv,
                    args.image_dir, run_case, model_name, args.num_images,
                )
                save_individual_images(
                    images, x_adv, labels,
                    args.image_dir, run_case, model_name, args.num_images,
                )
                try:
                    save_gabor_spectrograms(
                        images, x_adv, (x_adv - images), labels, Psi_2D,
                        args.image_dir, run_case, model_name, args.num_images,
                        x_tilde=None,
                    )
                    print(f"  ✓ Spectrograms saved successfully for Case {run_case[-1]}")
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")
                images_saved = True

        results[run_case] = aggregate_metrics(case_metrics)
        print_summary(results[run_case], f"Case {run_case[-1]}")

    return results, _finalize_transfer(transfer_acc)


def save_image_comparison(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    predictions_clean: torch.Tensor,
    predictions_adv: torch.Tensor,
    save_dir: str,
    case_name: str,
    model_name: str,
    num_images: int = 10,
):
    """
    Save side-by-side comparison of clean and adversarial images

    Args:
        clean_images: Clean images (B, C, H, W) in [0, 1]
        adv_images: Adversarial images (B, C, H, W) in [0, 1]
        labels: True labels (B,)
        predictions_clean: Predictions on clean images (B,)
        predictions_adv: Predictions on adversarial images (B,)
        save_dir: Directory to save images
        case_name: 'case1' or 'case2'
        model_name: Name of the model
        num_images: Number of image pairs to save
    """
    os.makedirs(save_dir, exist_ok=True)

    # CIFAR-100 class names (fine labels)
    class_names = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm",
    ]

    # Limit to available images
    num_images = min(num_images, clean_images.shape[0])

    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # Convert images to numpy (C, H, W) -> (H, W, C)
        clean_img = clean_images[i].cpu().permute(1, 2, 0).numpy()
        adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()

        # Clip to [0, 1] for display
        clean_img = np.clip(clean_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)

        # Get labels and predictions
        true_label = class_names[labels[i].item()]
        pred_clean = class_names[predictions_clean[i].item()]
        pred_adv = class_names[predictions_adv[i].item()]

        # Display clean image
        axes[i, 0].imshow(clean_img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"Clean\nTrue: {true_label}\nPred: {pred_clean}", fontsize=10
        )

        # Display adversarial image
        axes[i, 1].imshow(adv_img)
        axes[i, 1].axis("off")
        color = "red" if pred_adv != true_label else "green"
        axes[i, 1].set_title(
            f"Adversarial\nTrue: {true_label}\nPred: {pred_adv}",
            fontsize=10,
            color=color,
        )

    plt.tight_layout()

    # Save figure
    filename = f"{model_name}_{case_name}_comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved comparison images to: {filepath}")


def save_individual_images(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    case_name: str,
    model_name: str,
    num_images: int = 10,
):
    """
    Save individual clean and adversarial images as separate files

    Args:
        clean_images: Clean images (B, C, H, W) in [0, 1]
        adv_images: Adversarial images (B, C, H, W) in [0, 1]
        labels: True labels (B,)
        save_dir: Directory to save images
        case_name: 'case1' or 'case2'
        model_name: Name of the model
        num_images: Number of image pairs to save
    """
    os.makedirs(save_dir, exist_ok=True)

    # CIFAR-100 class names (fine labels)
    class_names = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm",
    ]

    # Limit to available images
    num_images = min(num_images, clean_images.shape[0])

    for i in range(num_images):
        true_label = class_names[labels[i].item()]

        # Save clean image
        clean_img = clean_images[i].cpu().permute(1, 2, 0).numpy()
        clean_img = np.clip(clean_img, 0, 1)

        plt.figure(figsize=(3, 3))
        plt.imshow(clean_img)
        plt.axis("off")
        plt.title(f"{true_label}", fontsize=12)

        clean_filename = f"{model_name}_{case_name}_clean_{i:03d}_{true_label}.png"
        clean_filepath = os.path.join(save_dir, clean_filename)
        plt.savefig(clean_filepath, dpi=150, bbox_inches="tight")
        plt.close()

        # Save adversarial image
        adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, 0, 1)

        plt.figure(figsize=(3, 3))
        plt.imshow(adv_img)
        plt.axis("off")
        plt.title(f"{true_label} (adv)", fontsize=12)

        adv_filename = f"{model_name}_{case_name}_adv_{i:03d}_{true_label}.png"
        adv_filepath = os.path.join(save_dir, adv_filename)
        plt.savefig(adv_filepath, dpi=150, bbox_inches="tight")
        plt.close()

    print(
        f"  Saved {num_images} clean and {num_images} adversarial images to: {save_dir}"
    )

def save_gabor_spectrograms(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    delta: torch.Tensor,
    labels: torch.Tensor,
    Psi_2D: torch.Tensor,
    save_dir: str,
    case_name: str,
    model_name: str,
    num_images: int = 10,
    x_tilde: torch.Tensor = None,
):
    """
    Save Gabor magnitude spectrograms for clean and adversarial images
    """

    # Check if delta needs reshaping
    if delta.dim() == 1:
        # Delta is flattened, need to reshape to (B, C, H, W)
        # Infer B from clean_images
        B, C, H, W = clean_images.shape
        expected_size = B * C * H * W
        if delta.numel() == expected_size:
            delta = delta.reshape(B, C, H, W)
        else:
            raise ValueError(
                f"Delta size {delta.numel()} doesn't match expected {expected_size}"
            )
    elif delta.dim() == 2:
        # Delta is (B*C, H*W), reshape to (B, C, H, W)
        B, C, H, W = clean_images.shape
        delta = delta.reshape(B, C, H, W)

    os.makedirs(save_dir, exist_ok=True)

    # CIFAR-100 class names (fine labels)
    class_names = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm",
    ]

    # Limit to available images
    num_images = min(num_images, clean_images.shape[0])
    B, C, H, W = clean_images.shape
    n = H * W

    # Move Psi to same device as images
    Psi_2D = Psi_2D.to(clean_images.device)

    for idx in range(num_images):
        # try:
        # Create figure with 2 rows × 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # Get class name
        true_label = class_names[labels[idx].item()]

        # Convert images to numpy for display
        clean_img = clean_images[idx].cpu().permute(1, 2, 0).numpy()
        adv_img = adv_images[idx].cpu().permute(1, 2, 0).numpy()
        perturbation = delta[idx].cpu().permute(1, 2, 0).numpy()

        # Clip for display
        clean_img = np.clip(clean_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)

        ssim_value = None
        if SSIM_AVAILABLE:
            try:
                with torch.no_grad():
                    # SSIM expects (B, C, H, W) format
                    clean_tensor = clean_images[idx : idx + 1]  # (1, C, H, W)
                    adv_tensor = adv_images[idx : idx + 1]  # (1, C, H, W)

                    ssim_value = ssim(
                        clean_tensor, adv_tensor, data_range=1.0, size_average=True
                    ).item()
            except Exception as e:
                print(f"Warning: failed to compute SSIM for image {idx}: {e}")
                ssim_value = None

        # Row 1: Images (x_clean, x_tilde, x_adv, delta)
        axes[0, 0].imshow(clean_img)
        axes[0, 0].set_title(f"Clean image\n{true_label}", fontsize=18)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(adv_img)
        if ssim_value is not None:
            axes[0, 1].set_title(
                f"x_adv\n{true_label}\nSSIM: {ssim_value:.4f}", fontsize=18
            )
        else:
            axes[0, 1].set_title(f"x_adv\n{true_label}", fontsize=18)
        axes[0, 1].axis("off")

        # Perturbation (amplified for visibility)

        pert_display = perturbation - perturbation.min()
        pert_display = pert_display / (pert_display.max() + 1e-10)
        axes[0, 2].imshow(pert_display)

        # Canonical method name from cases.py; unknown cases fall back to the
        # raw case string instead of raising and killing the whole evaluation.
        axes[0, 2].set_title(f"{attack_name(case_name)} δ\n", fontsize=18)
        axes[0, 2].axis("off")

        # Row 2: Gabor Spectrograms
        # We'll use the average across RGB channels for visualization
        B, C, n, _ = clean_images.shape

        N = Psi_2D.shape[0]
        Psi_bc = Psi_2D.view(1, N, n)
        PsiT_bc = Psi_2D.t().view(1, n, N)

        def gabor2d_avg_magnitude(x_chw: torch.Tensor) -> np.ndarray:
            """
            x_chw: (C, H, W)
            returns: (N, N) numpy array of avg |Psi X Psi^T| over channels
            """
            x_chw = x_chw.to(dtype=Psi_2D.dtype)  # keep on same device as Psi
            # (1, N, H) @ (C, H, W) -> (C, N, W)  (broadcast over leading dim)
            tmp = torch.matmul(Psi_bc, x_chw)
            # (C, N, W) @ (1, W(=H), N) -> (C, N, N)
            w = torch.matmul(tmp, PsiT_bc)

            mag = torch.abs(w)  # (C, N, N)
            mag_avg = mag.mean(dim=0)  # (N, N)
            return mag_avg.detach().cpu().numpy()

        # Compute spectrogram magnitudes (avg over channels)
        clean_gabor_2d = gabor2d_avg_magnitude(clean_images[idx])  # (N, N)
        adv_gabor_2d = gabor2d_avg_magnitude(adv_images[idx])  # (N, N)
        delta_gabor_2d = gabor2d_avg_magnitude(delta[idx])  # (N, N)

        # PSNR between 2D spectrograms (compared to clean)
        def compute_psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float("inf")
            max_val = max(img1.max(), img2.max())
            if max_val == 0:
                return float("inf")
            return 20 * np.log10(max_val / np.sqrt(mse))

        # PSNR values (all compared to clean_gabor_avg)
        psnr_clean = float("inf")  # Self-comparison
        psnr_adv = compute_psnr(clean_gabor_2d, adv_gabor_2d)
        psnr_delta = compute_psnr(clean_gabor_2d, delta_gabor_2d)

        # Plot directly as (N, N). No padding / sqrt-grid reshaping needed anymore.
        im1 = axes[1, 0].imshow(clean_gabor_2d, cmap="coolwarm", aspect="auto")
        axes[1, 0].set_title(f"PSNR: ∞ dB", fontsize=18)
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        im3 = axes[1, 1].imshow(adv_gabor_2d, cmap="coolwarm", aspect="auto")
        axes[1, 1].set_title(f"PSNR: {psnr_adv:.4f} dB", fontsize=18)
        axes[1, 1].axis("off")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        im4 = axes[1, 2].imshow(delta_gabor_2d, cmap="coolwarm", aspect="auto")
        axes[1, 2].set_title(f"PSNR: {psnr_delta:.4f} dB", fontsize=18)
        axes[1, 2].axis("off")
        plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

        plt.tight_layout()

        # Save figure
        filename = f"{model_name}_{case_name}_spectrogram_{idx:03d}_{true_label}.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics across batches
    """
    aggregated = {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    for key in all_keys:
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        if values:
            # Average across batches (don't create additional _std suffix)
            aggregated[key] = np.mean(values)
        else:
            aggregated[key] = None

    return aggregated


def print_summary(metrics, case_name):
    """Print summary of key metrics"""
    print(f"\n{case_name} Summary:")
    print(f"  ASR:           {metrics['attack_success_rate']*100:>6.2f}%")
    print(f"  Clean Acc:     {metrics['clean_accuracy']*100:>6.2f}%")
    print(f"  Adv Acc:       {metrics['adversarial_accuracy']*100:>6.2f}%")
    print(
        f"  L2 Norm:       {metrics['mean_l2_norm']:>6.4f} ± {metrics.get('std_l2_norm', 0):>6.4f}"
    )
    print(
        f"  Linf Norm:       {metrics['mean_linf_norm']:>6.4f} ± {metrics.get('std_linf_norm', 0):>6.4f}"
    )

    if case_name == "case1":
        if metrics.get("mean_gabor_frame_norm") is not None:
            print(f"  Gabor ||w||_D:    {metrics['mean_gabor_frame_norm']:>6.8f}")
        if metrics.get("feasible_frac") is not None:
            print(f"  Feasibility region:   {metrics['feasible_frac']:>6.4f}")
    if metrics.get("lpips_mean") is not None:
        print(
            f"  LPIPS:         {metrics['lpips_mean']:>6.4f} ± {metrics.get('lpips_std', 0):>6.4f}"
        )
    if metrics.get("ssim_mean") is not None:
        print(
            f"  SSIM:          {metrics['ssim_mean']:>6.4f} ± {metrics.get('ssim_std', 0):>6.4f}"
        )
    if metrics.get("psnr_mean") is not None:
        print(
            f"  PSNR:          {metrics['psnr_mean']:>6.2f} ± {metrics.get('psnr_std', 0):>6.2f} dB"
        )


# ============================================================================
# Results Display and Saving
# ============================================================================


def print_results_table(all_results, args):
    """Print formatted results table"""
    print("\n" + "=" * 140)
    print("CIFAR100 DGF-PGD ATTACK RESULTS".center(140))
    print("=" * 140)

    cases_to_print = []
    if args.case == "case1":
        cases_to_print.append(("case1", "Case 1: Soft-thresholded Frame Attack"))
    elif args.case == "case2":
        cases_to_print.append(("case2", "Case 2: L2 PGD Attack"))
    elif args.case == "case3":
        cases_to_print.append(("case3", "Case 3: Fourier-based PGD Attack"))
    elif args.case == "case4":
        cases_to_print.append(("case4", "Case 4: AutoAttack"))
    elif args.case == "case5":
        cases_to_print.append(("case5", "Case 5: SSA (Spectrum Simulation Attack)"))
    elif args.case == "case6":
        cases_to_print.append(("case6", "Case 6: AdvDrop (InfoDrop)"))
    else:
        raise ValueError(f"Unknown case: {args.case}")

    for case_name, case_label in cases_to_print:
        print(f"\n{case_label}")
        print("-" * 140)
        print(
            f"{'Model':<30} {'ASR':>8} {'Clean':>8} {'Adv':>8} {'L2':>10} {'Linf':>10} {'||w||_D':>10} {'Feasible region':>10} {'LPIPS':>10} {'SSIM':>10}"
        )
        print("-" * 140)

        for model_name, results in all_results.items():
            if case_name in results:
                r = results[case_name]
                gabor_norm_str = (
                    f"{r['mean_gabor_frame_norm']:.8f}"
                    if r.get("mean_gabor_frame_norm") is not None
                    else "N/A"
                )
                gabor_feas_str = (
                    f"{r['feasible_frac']:.4f}"
                    if r.get("feasible_frac") is not None
                    else "N/A"
                )
                lpips_str = f"{r['lpips_mean']:.4f}" if r.get("lpips_mean") else "N/A"
                ssim_str = f"{r['ssim_mean']:.4f}" if r.get("ssim_mean") else "N/A"

                print(
                    f"{display_name(model_name):<30} "
                    f"{r['attack_success_rate']*100:>7.2f}% "
                    f"{r['clean_accuracy']*100:>7.2f}% "
                    f"{r['adversarial_accuracy']*100:>7.2f}% "
                    f"{r['mean_l2_norm']:>9.4f} "
                    f"{r['mean_linf_norm']:>9.4f} "
                    f"{gabor_norm_str:>9s} "
                    f"{gabor_feas_str:>9s}"
                    f"{lpips_str:>9s} "
                    f"{ssim_str:>9s}"
                )

    print("\n" + "=" * 160)


def save_results(all_results, args):
    """Save results to files"""
    os.makedirs(args.output_dir, exist_ok=True)

    # Summary file
    summary_file = os.path.join(args.output_dir, "cifar100_results.txt")
    with open(summary_file, "w") as f:
        f.write("CIFAR100 DGF-PGD Attack Results\n")
        f.write("=" * 160 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epsilon: {args.epsilon:.4f}\n")
        f.write(f"  Gamma: {args.gamma:.4f}\n")
        f.write(f"  Steps: {args.num_steps}\n")
        f.write(f"  Tau: {args.tau}\n")
        f.write(f"  Samples: {args.num_samples}\n\n")

        for case_name, case_label in [
            ("case1", "Case 1"),
            ("case2", "Case 2"),
            ("case3", "Case 3"),
            ("case4", "Case 4 (AutoAttack)"),
            ("case5", "Case 5 (SSA)"),
            ("case6", "Case 6 (AdvDrop)"),
        ]:
            if args.case != case_name:
                continue

            f.write(f"\n{case_label}\n")
            f.write("-" * 160 + "\n")
            f.write(
                f"{'Model':<30} {'ASR':>8} {'Clean':>8} {'Adv':>8} {'L2':>10} {'Linf':>10} {'||w||_D':>10} {'Feasible region':>10} {'LPIPS':>10} {'SSIM':>10} {'PSNR':>10}\n"
            )
            f.write("-" * 160 + "\n")

            for model_name, results in all_results.items():
                if case_name in results:
                    r = results[case_name]
                    gabor_norm_str = (
                        f"{r['mean_gabor_frame_norm']:.8f}"
                        if r.get("mean_gabor_frame_norm") is not None
                        else "N/A"
                    )
                    gabor_feas_str = (
                        f"{r['feasible_frac']:.4f}"
                        if r.get("feasible_frac") is not None
                        else "N/A"
                    )
                    lpips_str = (
                        f"{r['lpips_mean']:.4f}" if r.get("lpips_mean") else "N/A"
                    )
                    ssim_str = f"{r['ssim_mean']:.4f}" if r.get("ssim_mean") else "N/A"
                    psnr_str = f"{r['psnr_mean']:.2f}" if r.get("psnr_mean") else "N/A"

                    f.write(
                        f"{display_name(model_name):<30} "
                        f"{r['attack_success_rate']*100:>7.2f}% "
                        f"{r['clean_accuracy']*100:>7.2f}% "
                        f"{r['adversarial_accuracy']*100:>7.2f}% "
                        f"{r['mean_l2_norm']:>9.4f} "
                        f"{r['mean_linf_norm']:>9.4f} "
                        f"{gabor_norm_str:>9s} "
                        f"{gabor_feas_str:>9s}"
                        f"{lpips_str:>9s} "
                        f"{ssim_str:>9s} "
                        f"{psnr_str:>9s}\n"
                    )

    print(f"\n✓ Summary saved to {summary_file}")

    # Detailed JSON
    json_file = os.path.join(args.output_dir, "cifar100_results_detailed.json")
    serializable = {
        model: {
            case: {k: float(v) if v is not None else None for k, v in metrics.items()}
            for case, metrics in cases.items()
        }
        for model, cases in all_results.items()
    }
    with open(json_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"✓ Detailed results saved to {json_file}")


# ============================================================================
# Main
# ============================================================================


def main():
    """Main execution"""
    args = parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 80)
    print("CIFAR100 DGF-PGD ATTACK EVALUATION".center(80))
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Model Source: {args.model_source}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Cases: ", end="")

    gamma_str = f"{args.gamma:.4f}" if args.gamma is not None else "0.1 (default)"
    print(
        f"  Attack: ε={args.epsilon:.4f}, γ={gamma_str}, K={args.num_steps}, τ={args.tau}"
    )

    # Load data
    testloader = load_cifar100(args)

    # Load models
    models_dict = load_cifar100_models(args)

    epsilon = args.epsilon
    tau = args.tau
    a = args.a
    b = args.b
    rho = args.rho
    window = args.window_type

    rt = RuntimeLog("cifar100", device=args.device, timing=args.timing, meta={
        "a": a, "b": b, "window": window,
        "num_steps": args.num_steps, "epsilon": epsilon,
        "batch_size": args.batch_size,
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
        "aa_version": args.aa_version, "aa_norm": args.aa_norm,
        "source_models": sorted(models_dict),
        "n_source_models": len(models_dict),
        "model_params": {name: int(sum(p.numel() for p in m.parameters()))
                         for name, m in models_dict.items()},
        "model_family": {
            name: ("adv_trained" if args.model_source == "robustbench"
                   else "pretrained")
            for name in models_dict},
        "model_source": args.model_source, "case": args.case})
    args._rtlog = rt

    with rt.phase(args.case, "load"):
        (Psi_2D, Psi_plus_2D, D_inv_1_2D, M, eps_scale,
         mu_M_2D, U_M_2D, cond_S) = generate_gabor_operators_cifar100(
            args.device, a, b, window)

    gamma, gamma_why = resolve_gamma(args)
    args.gamma = gamma 
    if args.case == "case1":
        warn_if_constraint_inert(gamma, args.num_steps, eps_scale, epsilon, gamma_why)

    for case in [args.case]:

        print(f"Case {case[-1]}...")
        # Initialize attacker
        if case == "case4":
            print("\nUsing AutoAttack (adversary built per model)...")
            attacker = None
        elif case in ("case5", "case6"):
            print(f"\nInitializing baseline attacker ({case})...")
            attacker = build_baseline_attacker(
                case, list(models_dict.values())[0], args, image_size=32)
        else:
            print("\nInitializing DGF-PGD attacker...")
            attacker = DGFPGDAttack(
                model=list(models_dict.values())[0],
                loss_fn=nn.CrossEntropyLoss(),
                Psi_2D=Psi_2D,
                D_inv_1=D_inv_1_2D,
                M=M,
                eps_scale=eps_scale,
                mu_M=mu_M_2D,
                U_M=U_M_2D,
                image_shape=(3, 32, 32),
                epsilon=epsilon,
                gamma=gamma,
                num_steps=args.num_steps,
                case=args.case,
                device=args.device,
                verbose=args.verbose,
                amp=not args.no_amp,
            )

        # Initialize metrics evaluator
        print("Initializing metrics evaluator...")
        print(f"  M shape: {M.shape if M is not None else 'None'}")
        print(f"  2D Psi shape: {Psi_2D.shape if Psi_2D is not None else 'None'}")
        print(
            f"  2D Psi_plus shape: {Psi_plus_2D.shape if Psi_plus_2D is not None else 'None'}"
        )
        print(f"  Cond(S): {cond_S:.4f}")
        print(f"  tau: {args.tau}")
        print(f"  rho: {args.rho} (ignored)")
        print(f"  time param: {args.a}", f" frequency param: {args.b}")

        metrics_evaluator = AdversarialMetrics(
            device=args.device,
            lpips_net=args.lpips_net,
            verbose=args.verbose,
        )

        # Evaluate models
        all_results = {}
        transfer_by_source = {}
        for model_name, model in models_dict.items():
            results, transfer = evaluate_model(
                Psi_2D,
                Psi_plus_2D,
                model,
                model_name,
                attacker,
                metrics_evaluator,
                testloader,
                args.case,
                args,
                models_dict=models_dict,
            )
            all_results[model_name] = results
            transfer_by_source[model_name] = transfer

        # Print and save results
        print_results_table(all_results, args)
        save_results(all_results, args)
        try:
            save_transferability_reports(transfer_by_source, args)
        except Exception as e:
            print(f"  ✗ ERROR saving transferability reports: {e}")

        rt_name = (f"runtime_cifar100_{args.case}_{args.model_source}"
                   f"{'_timing' if args.timing else ''}"
                   f"_{rt.env['accelerator']}"
                   f"_{rt.env.get('slurm_job_id') or 'local'}"
                   f"_{rt.env.get('slurm_array_task_id') or '0'}"
                   f"_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.json")
        rt_dir = args.runtime_dir or os.path.join(args.output_dir, "runtime")
        rt_path = rt.write(rt_dir, rt_name)
        rt.print_summary()
        print(f"  Runtime written to {rt_path}")

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!".center(80))
        print("=" * 80)


if __name__ == "__main__":
    main()
