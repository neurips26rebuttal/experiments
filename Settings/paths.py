"""Default on-disk locations for weights and datasets, resolved from the repo.
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
    """
    import torch

    d = torch_hub_dir(models_dir)
    os.makedirs(d, exist_ok=True)
    torch.hub.set_dir(d)
    return d

DATA_DIR = os.environ.get("DGF_DATA_DIR", os.path.join(REPO_ROOT, "data"))
IMAGENET_ROOT = os.environ.get(
    "DGF_IMAGENET_ROOT",
    os.environ.get("DGF_DATA_ROOT", os.path.join(DATA_DIR, "imagenet")),
)
CIFAR100_ROOT = os.environ.get(
    "DGF_CIFAR100_ROOT", os.path.join(DATA_DIR, "CIFAR100")
)
