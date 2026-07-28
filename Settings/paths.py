"""Default on-disk locations for weights and datasets, resolved from the repo.

All the .py files live in src/, so the repo root is exactly one level up from
this file. Deriving the defaults from ``__file__`` rather than from the CWD means
``python src/eval_imagenet.py`` finds models/ and data/ no matter where it is
launched from, and a checkout that moves (laptop -> $WORK -> $SCRATCH) needs no
edit.

Resolution order for every location, most specific last:

    1. the value here (repo-relative)
    2. the environment variable named below  -- for a cluster run whose data
       lives outside the checkout
    3. the corresponding CLI flag (--models-dir / --data-root)

Expected layout:

    models/<dataset>/<threat_model>/<name>.pt   RobustBench's own convention,
                                                so `model_dir=MODELS_DIR` is
                                                all robustbench needs
    models/torch_hub/                           torch.hub cache (torchvision
                                                checkpoints + hub repos), kept
                                                inside the checkout so compute
                                                nodes WITHOUT internet find
                                                weights pre-downloaded by
                                                src/download_models.py
    data/imagenet/val/<wnid>/*.JPEG             ImageFolder-style val split
    data/CIFAR100/cifar-100-python/             torchvision CIFAR100 root
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)

#: RobustBench weight tree; also what --models-dir defaults to.
MODELS_DIR = os.environ.get("DGF_MODELS_DIR", os.path.join(REPO_ROOT, "models"))


def torch_hub_dir(models_dir=None):
    """The torch.hub cache used by this repo: <models_dir>/torch_hub."""
    return os.environ.get(
        "DGF_TORCH_HUB_DIR", os.path.join(models_dir or MODELS_DIR, "torch_hub")
    )


def point_torch_hub(models_dir=None):
    """Redirect torch.hub's cache into the repo's models/ tree.

    torch.hub defaults to ~/.cache/torch/hub, which cluster compute nodes may
    not share with the login node (and cannot populate themselves: no internet).
    Calling this before any torchvision/torch.hub weight load makes both the
    downloader (src/download_models.py, run online) and the eval scripts (run
    offline) agree on one location inside the checkout.
    """
    import torch

    d = torch_hub_dir(models_dir)
    os.makedirs(d, exist_ok=True)
    torch.hub.set_dir(d)
    return d

#: Parent of the per-dataset roots below.
DATA_DIR = os.environ.get("DGF_DATA_DIR", os.path.join(REPO_ROOT, "data"))

#: ImageNet root. Holds val/; eval_imagenet.py appends "val" itself.
#: DGF_DATA_ROOT is the older name for this and is still honoured.
IMAGENET_ROOT = os.environ.get(
    "DGF_IMAGENET_ROOT",
    os.environ.get("DGF_DATA_ROOT", os.path.join(DATA_DIR, "imagenet")),
)

#: CIFAR100 root, i.e. the directory that contains cifar-100-python/.
CIFAR100_ROOT = os.environ.get(
    "DGF_CIFAR100_ROOT", os.path.join(DATA_DIR, "CIFAR100")
)
