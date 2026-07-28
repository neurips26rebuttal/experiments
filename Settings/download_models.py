#!/usr/bin/env python3
"""
Download all models required by eval_imagenet.py and eval_cifar100.py.

Everything lands INSIDE the repo's models/ tree so that jobs running on
compute nodes without internet access find every weight already on disk:

  RobustBench imagenet  -> MODELS_DIR/imagenet/Linf/
  RobustBench cifar100  -> MODELS_DIR/cifar100/Linf/
  Standard pretrained   -> MODELS_DIR/torch_hub/checkpoints/  (torchvision)
  chenyaofo backbones   -> MODELS_DIR/torch_hub/              (hub repo + ckpts)

Run this once on a node WITH internet (e.g. the login/prepost node), then the
eval scripts -- which point torch.hub at MODELS_DIR/torch_hub via
paths.point_torch_hub() -- never touch the network.

  python3 src/download_models.py                  # everything
  python3 src/download_models.py --dataset cifar100
"""
import argparse
import os
import sys

import torch
import torchvision.models as tv_models

import paths


# ============================================================================
# Model lists
# ============================================================================

ROBUSTBENCH_MODELS = [
    "Amini2024MeanSparse_ConvNeXt-L",
    "Liu2023Comprehensive_ConvNeXt-B",
    "Bai2024MixedNUTS",
    "Debenedetti2022Light_XCiT-M12",
    "Engstrom2019Robustness",
    "RodriguezMunoz2024Characterizing_Swin-B",
    "Salman2020Do_R50",
    "Singh2023Revisiting_ViT-B-ConvStem",
    "Wong2020Fast",
]

# Must match load_robustbench_models() in eval_cifar100.py.
ROBUSTBENCH_CIFAR100_MODELS = [
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

# Must match load_pretrained_backbones() in eval_cifar100.py
# (--model-source pretrained). Fetched via torch.hub, so both the hub repo
# snapshot and each checkpoint end up in MODELS_DIR/torch_hub/.
CIFAR100_HUB_REPO = "chenyaofo/pytorch-cifar-models"
CIFAR100_HUB_BACKBONES = [
    "cifar100_mobilenetv2_x0_5",
    "cifar100_mobilenetv2_x0_75",
    "cifar100_mobilenetv2_x1_0",
    "cifar100_mobilenetv2_x1_4",
    "cifar100_shufflenetv2_x0_5",
    "cifar100_shufflenetv2_x2_0",
    "cifar100_repvgg_a0",
    "cifar100_repvgg_a1",
    "cifar100_repvgg_a2",
    "cifar100_resnet20",
    "cifar100_resnet32",
    "cifar100_resnet44",
    "cifar100_resnet56",
    "cifar100_vgg11_bn",
    "cifar100_vgg13_bn",
    "cifar100_vgg16_bn",
    "cifar100_vgg19_bn",
]

STANDARD_PRETRAINED = {
    # (weights_enum, builder)
    "resnet18":           (tv_models.ResNet18_Weights.IMAGENET1K_V1,
                           lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)),
    "resnet50":           (tv_models.ResNet50_Weights.IMAGENET1K_V2,
                           lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)),
    "vgg16_bn":           (tv_models.VGG16_BN_Weights.IMAGENET1K_V1,
                           lambda: tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1)),
    "densenet121":        (tv_models.DenseNet121_Weights.IMAGENET1K_V1,
                           lambda: tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)),
    "mobilenet_v2":       (tv_models.MobileNet_V2_Weights.IMAGENET1K_V2,
                           lambda: tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V2)),
    "efficientnet_b0":    (tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1,
                           lambda: tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)),
    "convnext_tiny":      (tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                           lambda: tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)),
    "vit_b_16":           (tv_models.ViT_B_16_Weights.IMAGENET1K_V1,
                           lambda: tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)),
    "alexnet":            (tv_models.AlexNet_Weights.IMAGENET1K_V1,
                           lambda: tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)),
    "googlenet":          (tv_models.GoogLeNet_Weights.IMAGENET1K_V1,
                           lambda: tv_models.googlenet(weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1, transform_input=False)),
    "inception_v3":       (tv_models.Inception_V3_Weights.IMAGENET1K_V1,
                           lambda: tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)),
    "maxvit_t":           (tv_models.MaxVit_T_Weights.IMAGENET1K_V1,
                           lambda: tv_models.maxvit_t(weights=tv_models.MaxVit_T_Weights.IMAGENET1K_V1)),
    "mnasnet1_0":         (tv_models.MNASNet1_0_Weights.IMAGENET1K_V1,
                           lambda: tv_models.mnasnet1_0(weights=tv_models.MNASNet1_0_Weights.IMAGENET1K_V1)),
    "regnet_y_8gf":       (tv_models.RegNet_Y_8GF_Weights.IMAGENET1K_V2,
                           lambda: tv_models.regnet_y_8gf(weights=tv_models.RegNet_Y_8GF_Weights.IMAGENET1K_V2)),
    "resnext50_32x4d":    (tv_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
                           lambda: tv_models.resnext50_32x4d(weights=tv_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)),
    "shufflenet_v2_x1_0": (tv_models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
                           lambda: tv_models.shufflenet_v2_x1_0(weights=tv_models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)),
    "swin_t":             (tv_models.Swin_T_Weights.IMAGENET1K_V1,
                           lambda: tv_models.swin_t(weights=tv_models.Swin_T_Weights.IMAGENET1K_V1)),
    "wide_resnet50_2":    (tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
                           lambda: tv_models.wide_resnet50_2(weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)),
}


# ============================================================================
# Existence checks
# ============================================================================

def _robustbench_exists(models_dir: str, dataset: str, threat_model: str, name: str) -> bool:
    """Return True if a non-empty .pt weight file for `name` is already on disk."""
    model_subdir = os.path.join(models_dir, dataset, threat_model)
    if not os.path.isdir(model_subdir):
        return False
    for fname in os.listdir(model_subdir):
        if not fname.endswith(".pt"):
            continue
        # Match exact name or sharded variants: name.pt, name_m1.pt, name.pt_m1.pt
        stem = fname
        for suffix in (".pt",):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem == name or stem.startswith(name + "_") or stem.startswith(name + ".pt_"):
            fpath = os.path.join(model_subdir, fname)
            if os.path.getsize(fpath) > 100_000:
                return True
    return False


def _torchvision_exists(weights_enum) -> bool:
    """Return True if the torchvision weight file is already in the hub cache."""
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    filename = os.path.basename(weights_enum.url)
    return os.path.isfile(os.path.join(cache_dir, filename))


# ============================================================================
# Download helpers
# ============================================================================

def download_robustbench_models(models_dir: str, dataset: str, model_names,
                                threat_model: str = "Linf"):
    try:
        from robustbench.utils import load_model as rb_load_model
        import robustbench.utils as _rb_utils
    except ImportError:
        print("ERROR: robustbench not installed. pip install robustbench")
        sys.exit(1)

    try:
        import gdown

        def _download_gdrive_fixed(gdrive_id, fname_save):
            fname_save = str(fname_save)
            print(f"  [gdown] downloading {os.path.basename(fname_save)} ...")
            gdown.download(id=gdrive_id, output=fname_save, quiet=False)

        _rb_utils.download_gdrive = _download_gdrive_fixed
    except ImportError:
        print("  Warning: gdown not installed — large models may fail to download.")

    print(f"\n{'='*60}")
    print(f"  Downloading {len(model_names)} RobustBench {dataset} models")
    print(f"  models_dir : {models_dir}")
    print(f"  threat_model: {threat_model}")
    print(f"{'='*60}")

    ok, skipped, failed = [], [], []
    for name in model_names:
        if _robustbench_exists(models_dir, dataset, threat_model, name):
            print(f"\n  -> {name} ... SKIPPED (already on disk)")
            skipped.append(name)
            ok.append(name)
            continue
        print(f"\n  -> {name} ...")
        try:
            rb_load_model(
                model_name=name,
                dataset=dataset,
                threat_model=threat_model,
                model_dir=models_dir,
            )
            print(f"     OK")
            ok.append(name)
        except Exception as e:
            print(f"     FAILED: {e}")
            failed.append((name, str(e)))

    print(f"\n  RobustBench {dataset}: {len(ok)} ready "
          f"({len(skipped)} skipped), {len(failed)} failed")
    return ok, failed


def download_cifar100_hub_backbones():
    """Cache the chenyaofo hub repo + every backbone checkpoint locally.

    torch.hub.load caches the GitHub repo snapshot and the weight file under
    torch.hub.get_dir(), which main() has already pointed at
    MODELS_DIR/torch_hub -- so after this, eval_cifar100.py --model-source
    pretrained works with no network at all.
    """
    print(f"\n{'='*60}")
    print(f"  Downloading {len(CIFAR100_HUB_BACKBONES)} CIFAR100 hub backbones")
    print(f"  ({CIFAR100_HUB_REPO} -> {torch.hub.get_dir()})")
    print(f"{'='*60}")

    ok, failed = [], []
    for hub_name in CIFAR100_HUB_BACKBONES:
        print(f"\n  -> {hub_name} ...")
        try:
            model = torch.hub.load(CIFAR100_HUB_REPO, hub_name,
                                   pretrained=True, skip_validation=True)
            del model
            print(f"     OK")
            ok.append(hub_name)
        except Exception as e:
            print(f"     FAILED: {e}")
            failed.append((hub_name, str(e)))

    print(f"\n  CIFAR100 backbones: {len(ok)} ready, {len(failed)} failed")
    return ok, failed


def download_standard_pretrained_models():
    print(f"\n{'='*60}")
    print(f"  Downloading {len(STANDARD_PRETRAINED)} standard torchvision models")
    print(f"  (cache: {os.path.join(torch.hub.get_dir(), 'checkpoints')})")
    print(f"{'='*60}")

    ok, skipped, failed = [], [], []
    for name, (weights_enum, builder) in STANDARD_PRETRAINED.items():
        if _torchvision_exists(weights_enum):
            print(f"\n  -> {name} ... SKIPPED (already cached)")
            skipped.append(name)
            ok.append(name)
            continue
        print(f"\n  -> {name} ...")
        try:
            model = builder()
            del model
            print(f"     OK")
            ok.append(name)
        except Exception as e:
            print(f"     FAILED: {e}")
            failed.append((name, str(e)))

    print(f"\n  Standard: {len(ok)} ready ({len(skipped)} skipped), {len(failed)} failed")
    return ok, failed


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all models required by eval_imagenet.py and "
                    "eval_cifar100.py into the repo's models/ tree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir", type=str, default=paths.MODELS_DIR,
        help="Directory where all weights will be saved "
             "(default: the repo's models/)",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "imagenet", "cifar100"],
        help="Which dataset's models to download",
    )
    parser.add_argument(
        "--threat-model", type=str, default="Linf", choices=["Linf", "L2"],
        help="RobustBench threat model",
    )
    parser.add_argument(
        "--skip-robustbench", action="store_true",
        help="Skip RobustBench model downloads",
    )
    parser.add_argument(
        "--skip-standard", action="store_true",
        help="Skip standard torchvision / hub backbone downloads",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    # All torch.hub traffic (torchvision checkpoints, chenyaofo repo+weights)
    # goes into <models-dir>/torch_hub -- the same place the eval scripts look.
    hub = paths.point_torch_hub(args.models_dir)
    print(f"torch.hub cache: {hub}")

    do_imagenet = args.dataset in ("all", "imagenet")
    do_cifar100 = args.dataset in ("all", "cifar100")

    groups = []  # (label, ok, failed)

    if not args.skip_robustbench:
        if do_imagenet:
            groups.append(("RobustBench imagenet",) + download_robustbench_models(
                args.models_dir, "imagenet", ROBUSTBENCH_MODELS, args.threat_model))
        if do_cifar100:
            groups.append(("RobustBench cifar100",) + download_robustbench_models(
                args.models_dir, "cifar100", ROBUSTBENCH_CIFAR100_MODELS,
                args.threat_model))

    if not args.skip_standard:
        if do_imagenet:
            groups.append(("Standard torchvision",)
                          + download_standard_pretrained_models())
        if do_cifar100:
            groups.append(("CIFAR100 backbones",)
                          + download_cifar100_hub_backbones())

    # Summary
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    any_failed = False
    for label, ok, failed in groups:
        print(f"  {label:<22}: {len(ok)} OK  |  {len(failed)} failed")
        for name, err in failed:
            print(f"    - {name}: {err[:80]}")
        any_failed = any_failed or bool(failed)

    if any_failed:
        print("\n  Some downloads failed — check errors above.")
        sys.exit(1)
    else:
        print("\n  All models downloaded successfully.")


if __name__ == "__main__":
    main()
