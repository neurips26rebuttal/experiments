"""
CIFAR100 DGF-PGD Attack Evaluation from CLAUDE
==================================

Comprehensive evaluation script for CIFAR100 that integrates:
- DGFPGDAttack (Case 1 and Case 2)
- AdversarialMetrics (30+ evaluation metrics)
- Pretrained CIFAR100 models from RobustBench

Usage:
    # Both cases
    python eval_CIFAR100.py --case1 --case2
    
    # Case 1 only
    python eval_CIFAR100.py --case1
    
    # Quick test
    python eval_CIFAR100.py --case1 --num-samples 100
"""
import math
import gc
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
from typing import Dict, List

# Import attack and metrics classes
from transforms import *
from dgf_pgd import DGFPGDAttack
from evaluation_metrics import AdversarialMetrics
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from autoattack import AutoAttack

try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not available. SSIM values will not be shown in spectrograms.")

# ============================================================================
# Command-line Arguments
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CIFAR100 DGF-PGD Attack Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-source',
        type=str,
        default='robustbench',
        choices=['robustbench', 'pretrained'],
        help='Model source: robustbench (L2 robust models) or pretrained (chenyaofo backbones)'
    )
    
    # Case selection
    parser.add_argument('--case', type=str, default='case1', help='Run Case 1, Case 2, Case 3 or Case 4 attack')
    
    # AutoAttack parameters (case4)
    parser.add_argument(
        "--aa-norm",
        type=str,
        default="Linf",
        choices=["Linf", "L2"],
        help="AutoAttack threat model norm (case4 only)",
    )
    parser.add_argument(
        "--aa-version",
        type=str,
        default="standard",
        choices=["standard", "plus", "rand"],
        help="AutoAttack version (case4 only)",
    )
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='./data/CIFAR100', help='CIFAR100 data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers')
    
    # Attack parameters
    parser.add_argument('--epsilon', type=float, default=16/255,
                       help='Attack epsilon (Linf norm)')
    parser.add_argument('--gamma', type=float, default=0.1, help='Step size')
    parser.add_argument('--num-steps', type=int, default=20, help='PGD iterations')
    parser.add_argument('--a', type=int, default=1, help='Time lattice parameter')
    parser.add_argument('--b', type=int, default=16, help='Frequency lattice parameter')
    parser.add_argument('--window-type', type=str, default='Hann',
                       choices=['Hann', 'Blackman', 'Gaussian'],
                       help='Type of Gabor window function')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to evaluate (default: all available)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./results_cifar100', help='Output directory')
    parser.add_argument('--lpips-net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'])
    parser.add_argument('--save-images', action='store_true', help='Save clean and adversarial images')
    parser.add_argument('--num-images', type=int, default=10, help='Number of image pairs to save')
    parser.add_argument('--image-dir', type=str, default='./results_cifar100/images', help='Directory to save images')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # If neither case specified, run both
    if not args.case:
        print("No case specified. Running default (Case 1).")
        args.case = 'case1'

    return args


# ============================================================================
# Dataset Loading
# ============================================================================

def load_CIFAR100(args):
    """Load CIFAR100 test data"""
    print("\nLoading CIFAR100 dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    testset = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    # Subset if requested
    if args.num_samples is not None and args.num_samples < len(testset):
        indices = np.random.choice(len(testset), args.num_samples, replace=False)
        testset = Subset(testset, indices)
    
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Loaded {len(testset)} test samples")
    return testloader


def load_CIFAR100_models(args):
    """Load pretrained CIFAR100 models"""
        
    if args.model_source == 'robustbench':
        return load_robustbench_models(args)
    elif args.model_source == 'pretrained':
        return load_pretrained_backbones(args)
    else:
        raise ValueError(f"Unknown model source: {args.model_source}")
    
def load_robustbench_models(args):
    """Load pretrained CIFAR100 models from RobustBench (Linf threat model)"""
    print("\nLoading CIFAR100 models from RobustBench (Linf threat model)...")
    
    try:
        from robustbench.utils import load_model as rb_load_model
    except ImportError:
        print("\nERROR: robustbench not installed!")
        print("Install with: pip install robustbench")
        raise ImportError("robustbench is required for RobustBench models")
    
    # Available RobustBench CIFAR100 Linf models
    model_names = [
        # 'Addepalli2021Towards_WRN34',
        # 'Amini2024MeanSparse_S-WRN-70-16',
        # 'Bai2024MixedNUTS',
        # 'Debenedetti2022Light_XCiT-L12',
        # 'Debenedetti2022Light_XCiT-M12',
        # 'Debenedetti2022Light_XCiT-S12',
        'Chen2024Data_WRN_34_10',
        # 'Cui2023Decoupled_WRN-34-10',
        # 'Debenedetti2022Light_XCiT-L12',
        # 'Jia2022LAS-AT_34_10',
        # 'Pang2022Robustness_WRN70_16',
        # 'Wu2020Adversarial'
        # 'Addepalli2021Towards_WRN34',
        # 'Bai2023Improving_trades',
        # 'Chen2024Data_WRN_34_10',
        # 'Pang2022Robustness_WRN70_16',
        # 'Wu2020Adversarial'
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
                threat_model="Linf"
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
        
    # Available pretrained backbones
    available_backbones = {
        'mobilenetv2_x0_5': 'cifar100_mobilenetv2_x0_5',
        'mobilenetv2_x1_4': 'cifar100_mobilenetv2_x1_4',
        'mobilenetv2_x0_75': 'cifar100_mobilenetv2_x0_75',
        'mobilenetv2_x1_0': 'cifar100_mobilenetv2_x1_0',
        'shufflenetv2_x2_0': 'cifar100_shufflenetv2_x2_0',
        'shufflenetv2_x0_5': 'cifar100_shufflenetv2_x0_5',
        'repvgg_a0': 'cifar100_repvgg_a0',
        'repvgg_a1': 'cifar100_repvgg_a1',
        'repvgg_a2': 'cifar100_repvgg_a2',
        'resnet20': 'cifar100_resnet20',
        'resnet32': 'cifar100_resnet32',
        'resnet44': 'cifar100_resnet44',
        'resnet56': 'cifar100_resnet56',
        'vgg11_bn': 'cifar100_vgg11_bn',
        'vgg13_bn': 'cifar100_vgg13_bn',
        'vgg16_bn': 'cifar100_vgg16_bn',
        'vgg19_bn': 'cifar100_vgg19_bn',
    }
    
    # Default backbones if none specified
    default_backbones = {
        'mobilenetv2_x1_4': 'cifar100_mobilenetv2_x1_4',
        'repvgg_a2': 'cifar100_repvgg_a2',
        'resnet56': 'cifar100_resnet56',
        'vgg16_bn': 'cifar100_vgg16_bn',
        'shufflenetv2_x0_5': 'cifar100_shufflenetv2_x0_5',
    }
    
    
    # Filter if specific models requested
    if args.models:
        backbones_to_load = {k: v for k, v in available_backbones.items() 
                            if k in args.models or any(m.lower() in k for m in args.models)}
        if not backbones_to_load:
            print(f"  No matching backbones found. Using defaults.")
            backbones_to_load = default_backbones
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
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
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
            pretrained_model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                hub_name,
                pretrained=True
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

def generate_gabor_operators_CIFAR100(device, a, b, window_type='Gaussian'):
    """
    Generate Gabor frame operators for CIFAR100 (32X32 per channel)
    
    Returns:
        Tuple of operators
    """

    print("\nGenerating 2D Gabor operators (and their kronecker products for CIFAR100...")
    
    # Parameters
    H, W = 32, 32
    n = H * W  # 1024
    
    print(f"  Dimensions: H={H}, W={W}, N={(n)/(a*b)}")
    
    print("  Generating Ψ...")
    Psi_2D = DGT(H, a=a, b=b, window=window_type)
    Psi_2D = Psi_2D/torch.linalg.norm(Psi_2D, dim = 1, keepdim=True)
    
    print("  Computing Ψ^+...")
    Psi_plus_2D = torch.linalg.pinv(Psi_2D)
    
    print("  Generating D...")
    D_2D = diag_weights_from_mc_row_sums(Psi_2D, mode='down')

    print("  Computing the condition number of the kronecker-product frame...")
    S_2D = frameop_DGT(Psi_2D)
    S_2D_inv = torch.linalg.inv(S_2D)
    cond_S = (torch.linalg.matrix_norm(S_2D, ord=2) * torch.linalg.matrix_norm(S_2D_inv, ord=2)).item()

    """
    Operators needed for case 1:
    a) M = Ψ* D Ψ -----> weights1
    b) Dual norm: Ψ* D^(-1) Ψ -----> dual_norm1
    """
    
    print("  Computing D^(-1)...")    
    D_inv_1_2D = dual_norm1(D_2D, Psi_2D)
    D_inv_1 = torch.kron(D_inv_1_2D, D_inv_1_2D)  # Full 1D diagonal operator

    print("  Computing M = Ψ* D Ψ ...")
    M_2D = weights1(D_2D, Psi_2D)
    
    Mherm_2D = 0.5 * (M_2D + M_2D.mH)

    jitter_scale = 1e-6
    jitter = float(jitter_scale) * Mherm_2D.abs().mean()
    Mherm_2D = Mherm_2D + jitter * torch.eye(H, device=Mherm_2D.device, dtype=Mherm_2D.dtype)
    torch.cuda.empty_cache()
    gc.collect()

    Mherm_2D = Mherm_2D.to(device)
    print('Calculating eigendecomposition...')  # Keep on CPU for stability in solves
    
    # --- Cache: eigendecomposition if possible (fast path) ---
    use_eig = False
    try:
        M64 = Mherm_2D.to(torch.complex64) if Mherm_2D.is_complex() else Mherm_2D.to(torch.float64)
        mu64, U64 = torch.linalg.eigh(M64.cpu())  # CPU is more robust
        mu_M = mu64.real.clamp_min(0.0).to(torch.float32).to(device)  # (n,) 
        U_M = U64.to(Mherm_2D.dtype).to(device)                         # (n,n) 
        use_eig = True
    except Exception as e:
        print("Warning: eigh failed; using solve-based projection fallback.", repr(e))
        use_eig = False

    # --- eps_scale via (clipped) geometric mean of eigenvalues ---
    # Requires self.mu_M if use_eig=True. If use_eig=False, we fall back to slogdet.
    if use_eig:
        mu = mu_M.to(device)  # (n,)
        mu_floor = 1e-8  # tune: 1e-10..1e-6 depending on conditioning
        mu_safe = mu.clamp_min(mu_floor)
        eps_scale = torch.exp((torch.log(mu_safe**2).mean())/ (2 * n)).item()
    else:
        # fallback: still can be unstable for near-singular M
        logdet = torch.slogdet(Mherm_2D).logabsdet
        eps_scale = torch.exp(logdet / 2 * ((n))).item()

    eps_scale = math.sqrt((2 * n)/(math.e * math.pi)) * math.sqrt(math.pi * n) ** (1/n) * eps_scale
    print(f"  eps_scale: {eps_scale}")

    M = torch.kron(M_2D, M_2D)  # Full 1D operator
    M_herm = torch.kron(Mherm_2D, Mherm_2D)
    mu_M = torch.kron(mu_M, mu_M) if use_eig else print("  Warning: mu_M not available without eigendecomposition.")
    U_M = torch.kron(U_M, U_M) if use_eig else print("  Warning: U_M not available without eigendecomposition.")

    # D_inv_1 = torch.linalg.inv(M)
    # print(D_inv_1)

    print("  Operators ready!")
    
    return Psi_2D, Psi_plus_2D, D_inv_1, M, eps_scale, mu_M, U_M, M_herm, cond_S, use_eig


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(Psi_2D, Psi_plus_2D, M, model, model_name, attacker, metrics_evaluator, dataloader, 
                   run_case, args):
    """Evaluate attack on a single model"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    results = {}
    if attacker is not None:
        attacker.model = model
    
    # Track if we've saved images for this model
    images_saved = False
    
    # Case 1
    if run_case == 'case1':
        print(f"\nCase 1: Soft-thresholded Frame Attack")
        print("-" * 80)
        attacker.case = 'case1'
        
        case1_metrics = []
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Case 1", leave=False)):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, eps_dgf, last_delta = attacker(images, labels, random_init=True)
            metrics = metrics_evaluator.compute_all_metrics_gabor(model, images, x_adv, labels, eps_dgf, last_delta)
            case1_metrics.append(metrics)
            
            # Save images from first batch only
            if batch_idx == 0:
                with torch.no_grad():
                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)
                    
                # Save comparison image
                save_image_comparison(
                    images, x_adv, labels, pred_clean, pred_adv,
                    args.image_dir, 'case1', model_name, args.num_images
                )
                
                # Also save individual images
                save_individual_images(
                    images, x_adv, labels,
                    args.image_dir, 'case1', model_name, args.num_images
                )
                
                # Save Gabor spectrograms for Case 1
                print("  Generating Gabor spectrograms...")
                print(f"    last_delta shape: {last_delta.shape}")
                print(f"    Psi shape: {Psi_2D.shape}")
                print(f"    save_dir: {args.image_dir}")
                
                last_delta = x_adv - images  # Ensure last_delta is defined for spectrograms
                try:
                    save_gabor_spectrograms(
                        images, x_adv, last_delta, labels, Psi_2D,
                        args.image_dir, 'case1', model_name, args.epsilon, args.gamma, args.num_images,
                        x_tilde=None
                    )
                    print("  ✓ Spectrograms saved successfully for Case 1")
                except Exception as e:
                    print(f"  ✗ ERROR saving spectrograms: {e}")
                    import traceback
                    traceback.print_exc()
                
                images_saved = True
        
        results['case1'] = aggregate_metrics(case1_metrics)
        print_summary(results['case1'], "Case 1")
    
    # Case 2
    elif run_case == 'case2':
        print(f"\nCase 2: L2 PGD Attack")
        print("-" * 80)
        attacker.case = 'case2'

        case2_metrics = []
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Case 2", leave=False)):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, last_delta = attacker(images, labels, random_init=True)
            metrics = metrics_evaluator.compute_all_metrics(model, images, x_adv, labels)
            case2_metrics.append(metrics)
            
            # Save images from first batch only (if not already saved from case1)
            if batch_idx == 0:
                with torch.no_grad():
                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)
                
                # Save comparison image
                save_image_comparison(
                    images, x_adv, labels, pred_clean, pred_adv,
                    args.image_dir, 'case2', model_name, args.num_images
                )
                
                # Also save individual images
                save_individual_images(
                    images, x_adv, labels,
                    args.image_dir, 'case2', model_name, args.num_images
                )
                
                save_gabor_spectrograms(
                        images, x_adv, last_delta, labels, Psi_2D,
                    args.image_dir, 'case2', model_name, args.epsilon, args.gamma, args.num_images,
                        x_tilde=None
                )
                print("  ✓ Spectrograms saved successfully for Case 2")

                images_saved = True

        results['case2'] = aggregate_metrics(case2_metrics)
        print_summary(results['case2'], "Case 2")

    elif run_case == 'case3':
        print(f"\nCase 3: Fourier-based PGD Attack")
        print("-" * 80)
        attacker.case = 'case3'

        case3_metrics = []
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Case 3", leave=False)):
            images, labels = images.to(args.device), labels.to(args.device)
            x_adv, last_delta = attacker(images, labels, random_init=False)
            metrics = metrics_evaluator.compute_all_metrics(model, images, x_adv, labels)
            case3_metrics.append(metrics)

            # Save images from first batch only (if not already saved from case1 or case2)
            if batch_idx == 0:
                with torch.no_grad():
                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)

                # Save comparison image
                save_image_comparison(
                    images, x_adv, labels, pred_clean, pred_adv,
                    args.image_dir, 'case3', model_name, args.num_images
                )

                # Also save individual images
                save_individual_images(
                    images, x_adv, labels,
                    args.image_dir, 'case3', model_name, args.num_images
                )

                save_gabor_spectrograms(
                        images, x_adv, last_delta, labels, Psi_2D,
                    args.image_dir, 'case3', model_name, args.epsilon, args.gamma, args.num_images,
                        x_tilde=None
                )
                print("  ✓ Spectrograms saved successfully for Case 3")

                images_saved = True

        results['case3'] = aggregate_metrics(case3_metrics)
        print_summary(results['case3'], "Case 3")

    elif run_case == 'case4':
        print(f"\nCase 4: AutoAttack")
        print("-" * 80) 
        case4_metrics = []
        all_images = []
        all_labels = [] 
        
        for images, labels in dataloader: 
            all_images.append(images) 
            all_labels.append(labels) 
        
        images_all = torch.cat(all_images, dim=0).to(args.device)
        labels_all = torch.cat(all_labels, dim=0).to(args.device)

        adversary = AutoAttack(model, norm=args.aa_norm, eps=args.epsilon, version=args.aa_version, device=args.device, verbose=args.verbose, )

        print( f"Running AutoAttack " f"(version={args.aa_version}, " f"norm={args.aa_norm}, " f"eps={args.epsilon:.4f})" )

        x_adv_all = adversary.run_standard_evaluation(images_all,labels_all, bs=args.batch_size)

        delta = x_adv_all - images_all

        print("\nAutoAttack diagnostics:")
        print("delta abs max :", delta.abs().max().item())
        print("delta abs mean:", delta.abs().mean().item())
        print("num changed pixels:",
            (delta.abs() > 1e-12).float().sum().item())

        num_samples = images_all.shape[0]

        for start in tqdm( range(0, num_samples, args.batch_size), desc="Case 4", leave=False ):
            end = min(start + args.batch_size, num_samples)

            images = images_all[start:end]
            labels = labels_all[start:end] 
            x_adv = x_adv_all[start:end]

            last_delta = x_adv - images
            print(last_delta.abs().max())
            print(last_delta.abs().mean())

            metrics = metrics_evaluator.compute_all_metrics( model, images, x_adv, labels)

            case4_metrics.append(metrics)

            if start == 0:
                with torch.no_grad():

                    outputs_clean = model(images)
                    outputs_adv = model(x_adv)
                    pred_clean = outputs_clean.argmax(dim=1)
                    pred_adv = outputs_adv.argmax(dim=1)
                
                save_image_comparison( images, x_adv, labels, pred_clean, pred_adv, args.image_dir, 'case4', model_name, args.num_images )

                save_individual_images( images, x_adv, labels, args.image_dir, 'case4', model_name, args.num_images )

                save_gabor_spectrograms( images, x_adv, last_delta, labels, Psi_2D, args.image_dir, 'case4', model_name, args.epsilon, args.gamma, args.num_images,
                        x_tilde=None)

        print(" ✓ Spectrograms saved successfully for Case 4")

        results['case4'] = aggregate_metrics(case4_metrics)
    
    return results

def save_image_comparison(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    predictions_clean: torch.Tensor,
    predictions_adv: torch.Tensor,
    save_dir: str,
    case_name: str,
    model_name: str,
    num_images: int = 10
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
    
    # CIFAR-100 class names
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm']
    
    # Limit to available images
    num_images = min(num_images, clean_images.shape[0])
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3*num_images))
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
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Clean\nTrue: {true_label}\nPred: {pred_clean}',
                            fontsize=10)
        
        # Display adversarial image
        axes[i, 1].imshow(adv_img)
        axes[i, 1].axis('off')
        color = 'red' if pred_adv != true_label else 'green'
        axes[i, 1].set_title(f'Adversarial\nTrue: {true_label}\nPred: {pred_adv}',
                            fontsize=10, color=color)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{model_name}_{case_name}_comparison.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison images to: {filepath}")


def save_individual_images(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    case_name: str,
    model_name: str,
    num_images: int = 10
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
    
    # CIFAR-10 class names
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm']
    
    # Limit to available images
    num_images = min(num_images, clean_images.shape[0])
    
    for i in range(num_images):
        true_label = class_names[labels[i].item()]
        
        # Save clean image
        clean_img = clean_images[i].cpu().permute(1, 2, 0).numpy()
        clean_img = np.clip(clean_img, 0, 1)
        
        plt.figure(figsize=(3, 3))
        plt.imshow(clean_img)
        plt.axis('off')
        plt.title(f'{true_label}', fontsize=12)
        
        clean_filename = f'{model_name}_{case_name}_clean_{i:03d}_{true_label}.png'
        clean_filepath = os.path.join(save_dir, clean_filename)
        plt.savefig(clean_filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save adversarial image
        adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, 0, 1)
        
        plt.figure(figsize=(3, 3))
        plt.imshow(adv_img)
        plt.axis('off')
        plt.title(f'{true_label} (adv)', fontsize=12)
        
        adv_filename = f'{model_name}_{case_name}_adv_{i:03d}_{true_label}.png'
        adv_filepath = os.path.join(save_dir, adv_filename)
        plt.savefig(adv_filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {num_images} clean and {num_images} adversarial images to: {save_dir}")


def save_gabor_spectrograms(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    delta: torch.Tensor,
    labels: torch.Tensor,
    Psi_2D: torch.Tensor,
    save_dir: str,
    case_name: str,
    model_name: str,
    epsilon: float,
    gamma: float,
    num_images: int = 10,
    x_tilde: torch.Tensor = None,
):
    """
    Save Gabor magnitude spectrograms for clean and adversarial images
    
    Displays:
    - Row 1: x_tilde (or clean if None) | Adversarial Image | Perturbation δ
    - Row 2: Clean Gabor |Ψx_tilde|| Adv Gabor |Ψx_adv|| Delta Gabor |Ψδ|
    
    Args:
        clean_images: Clean images (B, C, H, W) in [0, 1]
        adv_images: Adversarial images (B, C, H, W) in [0, 1]
        delta: Perturbation returned by attack (B, C, H, W)
        labels: True labels (B,)
        Psi: Gabor frame operator (N, n)
        save_dir: Directory to save spectrograms
        case_name: 'case1' or 'case2'
        model_name: Name of the model
        num_images: Number of spectrograms to save
        x_tilde: Soft-thresholded reconstruction for Case 1 (optional)
    """
    # print(f"[DEBUG] Entering save_gabor_spectrograms")
    # print(f"[DEBUG] save_dir: {save_dir}")
    # print(f"[DEBUG] num_images: {num_images}")
    # print(f"[DEBUG] delta shape: {delta.shape}")
    
    # Check if delta needs reshaping
    if delta.dim() == 1:
        # Delta is flattened, need to reshape to (B, C, H, W)
        # Infer B from clean_images
        B, C, H, W = clean_images.shape
        expected_size = B * C * H * W
        if delta.numel() == expected_size:
            delta = delta.reshape(B, C, H, W)
            # print(f"[DEBUG] Reshaped delta from 1D to {delta.shape}")
        else:
            raise ValueError(f"Delta size {delta.numel()} doesn't match expected {expected_size}")
    elif delta.dim() == 2:
        # Delta is (B*C, H*W), reshape to (B, C, H, W)
        B, C, H, W = clean_images.shape
        delta = delta.reshape(B, C, H, W)
        # print(f"[DEBUG] Reshaped delta from 2D to {delta.shape}")
    
    os.makedirs(save_dir, exist_ok=True)
    # print(f"[DEBUG] Created directory: {save_dir}")
    
    # CIFAR-100 class names
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm']
    
    # Limit to available images
    num_images = min(num_images, clean_images.shape[0])
    B, C, H, W = clean_images.shape
    n = H * W
    
    # print(f"[DEBUG] Processing {num_images} images")
    # print(f"[DEBUG] Image shape: B={B}, C={C}, H={H}, W={W}")
    
    # Move Psi to same device as images
    Psi_2D = Psi_2D.to(clean_images.device)
    # print(f"[DEBUG] Psi shape: {Psi_2D.shape}, device: {Psi_2D.device}")


    for idx in range(num_images):
        # print(f"[DEBUG] Processing image {idx+1}/{num_images}")
        
        # try:
        # Create figure with 2 rows × 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # Get class name
        true_label = class_names[labels[idx].item()]
        # Use x_tilde for Case 1, otherwise use clean as reference
        # reference_images = x_tilde if x_tilde is not None else clean_images
        
        # Convert images to numpy for display
        clean_img = clean_images[idx].cpu().permute(1, 2, 0).numpy()
        # ref_img = reference_images[idx].cpu().permute(1, 2, 0).numpy()
        adv_img = adv_images[idx].cpu().permute(1, 2, 0).numpy()
        perturbation = delta[idx].cpu().permute(1, 2, 0).numpy()
        
        # print(f"[DEBUG]   clean_img range: [{clean_img.min():.4f}, {clean_img.max():.4f}]")
        # print(f"[DEBUG]   ref_img range: [{ref_img.min():.4f}, {ref_img.max():.4f}]")
        # print(f"[DEBUG]   adv_img range: [{adv_img.min():.4f}, {adv_img.max():.4f}]")
        # print(f"[DEBUG]   perturbation range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")
        
        # Clip for display
        clean_img = np.clip(clean_img, 0, 1)
        # ref_img = np.clip(ref_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)

        ssim_value = None
        if SSIM_AVAILABLE:
            try:
                with torch.no_grad():
                    # SSIM expects (B, C, H, W) format
                    clean_tensor = clean_images[idx:idx+1]  # (1, C, H, W)
                    adv_tensor = adv_images[idx:idx+1]      # (1, C, H, W)
                    
                    ssim_value = ssim(
                        clean_tensor, adv_tensor,
                        data_range=1.0,
                        size_average=True
                    ).item()
                    print(f"[DEBUG]   SSIM (clean vs adv): {ssim_value:.4f}")
            except Exception as e:
                print(f"[WARNING] Failed to compute SSIM: {e}")
                ssim_value = None
        else: 
            print(f"[DEBUG] SSIM not available, skipping SSIM computation.")
        
        # Row 1: Images (x_clean, x_tilde, x_adv, delta)
        axes[0, 0].imshow(clean_img)
        axes[0, 0].set_title(f'Clean image\n{true_label}', fontsize=18)
        axes[0, 0].axis('off')
        
        # axes[0, 1].imshow(ref_img)
        # if x_tilde is not None:
        #     axes[0, 1].set_title(f'x̃ (soft-thresholded)\n{true_label}', fontsize=12)
        # else:
        #     axes[0, 1].set_title(f'x_clean\n{true_label}', fontsize=12)
        # axes[0, 1].axis('off')
        
        axes[0, 1].imshow(adv_img)
        if ssim_value is not None:
            axes[0, 1].set_title(f'x_adv\n{true_label}\nSSIM: {ssim_value:.4f}', fontsize=18)
        else:
            axes[0, 1].set_title(f'x_adv\n{true_label}', fontsize=18)
        axes[0, 1].axis('off')
        
        # Perturbation (amplified for visibility)
    
        pert_display = perturbation - perturbation.min()
        pert_display = pert_display / (pert_display.max() + 1e-10)
        axes[0, 2].imshow(pert_display)

        if case_name == 'case1':
            axes[0, 2].set_title('Proposed attack δ\n', fontsize=18)
        elif case_name == 'case2':
            axes[0, 2].set_title('Standard PGD attack δ\n', fontsize=18)
        elif case_name == 'case3':
            axes[0, 2].set_title('Fourier-based PGD attack δ\n', fontsize=18)
        elif case_name == 'case4':
            axes[0, 2].set_title( 'AutoAttack perturbation δ\n', fontsize=18 )
        else:
            raise ValueError(f"Unknown case_name: {case_name}")
        axes[0, 2].axis('off')

        # print(f"[DEBUG]   Row 1 (images) done")
        
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
            x_chw = x_chw.to(dtype=Psi_2D.dtype)              # keep on same device as Psi
            # (1, N, H) @ (C, H, W) -> (C, N, W)  (broadcast over leading dim)
            tmp = torch.matmul(Psi_bc, x_chw)
            # (C, N, W) @ (1, W(=H), N) -> (C, N, N)
            w = torch.matmul(tmp, PsiT_bc)

            mag = torch.abs(w)                                # (C, N, N)
            mag_avg = mag.mean(dim=0)                         # (N, N)
            return mag_avg.detach().cpu().numpy()
        
        # Compute spectrogram magnitudes (avg over channels)
        clean_gabor_2d = gabor2d_avg_magnitude(clean_images[idx])          # (N, N)
        # ref_gabor_2d   = gabor2d_avg_magnitude(reference_images[idx])      # (N, N)
        adv_gabor_2d   = gabor2d_avg_magnitude(adv_images[idx])            # (N, N)
        delta_gabor_2d = gabor2d_avg_magnitude(delta[idx])                 # (N, N)

        # print(f"[DEBUG]   2D separable Gabor transforms computed: output {N}x{N}")

        # PSNR between 2D spectrograms (compared to clean)
        def compute_psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            max_val = max(img1.max(), img2.max())
            if max_val == 0:
                return float('inf')
            return 20 * np.log10(max_val / np.sqrt(mse))
        
        # PSNR values (all compared to clean_gabor_avg)
        psnr_clean = float('inf')  # Self-comparison
        # psnr_ref   = compute_psnr(clean_gabor_2d, ref_gabor_2d)
        psnr_adv   = compute_psnr(clean_gabor_2d, adv_gabor_2d)
        psnr_delta = compute_psnr(clean_gabor_2d, delta_gabor_2d)
        
        # print(f"[DEBUG]   PSNR values: adv={psnr_adv:.4f}, delta={psnr_delta:.4f}")

        # Plot directly as (N, N). No padding / sqrt-grid reshaping needed anymore.
        im1 = axes[1, 0].imshow(clean_gabor_2d, cmap='plasma', aspect='auto')
        axes[1, 0].set_title(f'PSNR: ∞ dB', fontsize=18)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        # ref_label = 'x̃' if x_tilde is not None else 'x_clean'
        # im2 = axes[1, 1].imshow(ref_gabor_2d, cmap='viridis', aspect='auto')
        # axes[1, 1].set_title(f'Spectrogram of {ref_label}\nPSNR: {psnr_ref:.4f} dB', fontsize=11)
        # axes[1, 1].axis('off')
        # plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        im3 = axes[1, 1].imshow(adv_gabor_2d, cmap='plasma', aspect='auto')
        axes[1, 1].set_title(f'PSNR: {psnr_adv:.4f} dB', fontsize=18)
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        im4 = axes[1, 2].imshow(delta_gabor_2d, cmap='plasma', aspect='auto')
        axes[1, 2].set_title(f'PSNR: {psnr_delta:.4f} dB', fontsize=18)
        axes[1, 2].axis('off')
        plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

        # print(f"[DEBUG]   Row 2 (2D spectrograms) done")

        plt.tight_layout()
        
        # Save figure
        filename = f'{model_name}_{case_name}_{epsilon}_{gamma}_spectrogram_{idx:03d}_{true_label}.png'
        filepath = os.path.join(save_dir, filename)
        # print(f"[DEBUG]   Saving to: {filepath}")
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # print(f"[DEBUG]   ✓ Saved: {filename}")
            
        # except Exception as e:
        #     print(f"[ERROR] Failed to process image {idx}: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     plt.close('all')  # Clean up any open figures
    
    # print(f"[DEBUG] Completed! Saved {num_images} Gabor spectrograms to: {save_dir}")


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics across batches
    
    The new evaluation_metrics.py returns per-batch statistics:
    - mean_l2_norm, std_l2_norm (mean and std within each batch)
    - mean_gabor_frame_norm, std_gabor_frame_norm
    - lpips_mean, lpips_std
    - ssim_mean, ssim_std
    
    This function aggregates these across all batches by averaging.
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
    print(f"  L2 Norm:       {metrics['mean_l2_norm']:>6.4f} ± {metrics.get('std_l2_norm', 0):>6.4f}")
    print(f"  Linf Norm:       {metrics['mean_linf_norm']:>6.4f} ± {metrics.get('std_linf_norm', 0):>6.4f}")

    if case_name == "case1":
        if metrics.get('mean_gabor_frame_norm') is not None:
            print(f"  Gabor ||w||_D:    {metrics['mean_gabor_frame_norm']:>6.8f}")
        if metrics.get('feasible_frac') is not None:
            print(f"  Feasibility region:   {metrics['feasible_frac']:>6.4f}")
    if metrics.get('lpips_mean') is not None:
        print(f"  LPIPS:         {metrics['lpips_mean']:>6.4f} ± {metrics.get('lpips_std', 0):>6.4f}")
    if metrics.get('ssim_mean') is not None:
        print(f"  SSIM:          {metrics['ssim_mean']:>6.4f} ± {metrics.get('ssim_std', 0):>6.4f}")


# ============================================================================
# Results Display and Saving
# ============================================================================

def print_results_table(all_results, args):
    """Print formatted results table"""
    print("\n" + "=" * 140)
    print("CIFAR100 DGF-PGD ATTACK RESULTS".center(140))
    print("=" * 140)
    
    cases_to_print = []
    if args.case == 'case1':
        cases_to_print.append(('case1', 'Case 1: Frame Attack'))
    elif args.case == 'case2':
        cases_to_print.append(('case2', 'Case 2: Linf PGD Attack'))
    elif args.case == 'case3':
        cases_to_print.append(('case3', 'Case 3: Fourier-based PGD Attack'))
    elif args.case == 'case4':
        cases_to_print.append(('case4', 'Case 4: AutoAttack'))
    else:
        ValueError(f"Unknown case: {args.case}")
    
    for case_name, case_label in cases_to_print:
        print(f"\n{case_label}")
        print("-" * 140)
        print(f"{'Model':<30} {'ASR':>8} {'Clean':>8} {'Adv':>8} {'L2':>10} {'Linf':>10} {'||w||_D':>10} {'Feasible region':>10} {'LPIPS':>10} {'SSIM':>10}")
        print("-" * 140)
        
        for model_name, results in all_results.items():
            if case_name in results:
                r = results[case_name]
                gabor_norm_str = f"{r['mean_gabor_frame_norm']:.8f}" if r.get('mean_gabor_frame_norm') is not None else "N/A"
                gabor_feas_str = f"{r['feasible_frac']:.4f}" if r.get('feasible_frac') is not None else "N/A"
                lpips_str = f"{r['lpips_mean']:.4f}" if r.get('lpips_mean') else "N/A"
                ssim_str = f"{r['ssim_mean']:.4f}" if r.get('ssim_mean') else "N/A"
                
                print(f"{model_name:<30} "
                      f"{r['attack_success_rate']*100:>7.2f}% "
                      f"{r['clean_accuracy']*100:>7.2f}% "
                      f"{r['adversarial_accuracy']*100:>7.2f}% "
                      f"{r['mean_l2_norm']:>9.4f} "
                      f"{r['mean_linf_norm']:>9.4f} "
                      f"{gabor_norm_str:>9s} "
                      f"{gabor_feas_str:>9s}"
                      f"{lpips_str:>9s} "
                      f"{ssim_str:>9s}")
    
    print("\n" + "=" * 160)


def save_results(all_results, args):
    """Save results to files"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Summary file
    summary_file = os.path.join(args.output_dir, 'CIFAR100_results.txt')
    with open(summary_file, 'w') as f:
        f.write("CIFAR100 DGF-PGD Attack Results\n")
        f.write("=" * 160 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epsilon: {args.epsilon:.4f}\n")
        f.write(f"  Gamma: {args.gamma:.4f}\n")
        f.write(f"  Steps: {args.num_steps}\n")
        f.write(f"  Samples: {args.num_samples}\n\n")

        for case_name, case_label in [('case1', 'Case 1'), ('case2', 'Case 2'), ('case3', 'Case 3'), ('case4', 'Case 4')]:
            if not (args.case == 'case1' if case_name == 'case1' else args.case == 'case2' 
                    if case_name == 'case2' else args.case == 'case3' if case_name == 'case3' else args.case == 'case4'):
                continue
            
            f.write(f"\n{case_label}\n")
            f.write("-" * 160 + "\n")
            f.write(f"{'Model':<30} {'ASR':>8} {'Clean':>8} {'Adv':>8} {'L2':>10} {'Linf':>10} {'||w||_D':>10} {'Feasible region':>10} {'LPIPS':>10} {'SSIM':>10} {'PSNR':>10}\n")
            f.write("-" * 160 + "\n")
            
            for model_name, results in all_results.items():
                if case_name in results:
                    r = results[case_name]
                    gabor_norm_str = f"{r['mean_gabor_frame_norm']:.8f}" if r.get('mean_gabor_frame_norm') is not None else "N/A"
                    gabor_feas_str = f"{r['feasible_frac']:.4f}" if r.get('feasible_frac') is not None else "N/A"
                    lpips_str = f"{r['lpips_mean']:.4f}" if r.get('lpips_mean') else "N/A"
                    ssim_str = f"{r['ssim_mean']:.4f}" if r.get('ssim_mean') else "N/A"
                    psnr_str = f"{r['psnr_mean']:.2f}" if r.get('psnr_mean') else "N/A"
                    
                    f.write(f"{model_name:<30} "
                           f"{r['attack_success_rate']*100:>7.2f}% "
                           f"{r['clean_accuracy']*100:>7.2f}% "
                           f"{r['adversarial_accuracy']*100:>7.2f}% "
                           f"{r['mean_l2_norm']:>9.4f} "
                           f"{r['mean_linf_norm']:>9.4f} "
                           f"{gabor_norm_str:>9s} "
                           f"{gabor_feas_str:>9s}"
                           f"{lpips_str:>9s} "
                           f"{ssim_str:>9s} "
                           f"{psnr_str:>9s}\n")
    
    print(f"\n✓ Summary saved to {summary_file}")
    
    # Detailed JSON
    json_file = os.path.join(args.output_dir, 'CIFAR100_results_detailed.json')
    serializable = {
        model: {
            case: {k: float(v) if v is not None else None for k, v in metrics.items()}
            for case, metrics in cases.items()
        }
        for model, cases in all_results.items()
    }
    with open(json_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"✓ Detailed results saved to {json_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution"""
    args = parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 80)
    print("CIFAR100 DGF-PGD ATTACK EVALUATION".center(80))
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Model Source: {args.model_source}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Cases: ", end="")
    if args.case == 'case1': print("Case 1...")
    elif args.case == 'case2': print("Case 2...")
    elif args.case == 'case3': print("Case 3...")
    elif args.case == 'case4': print("Case 4...")
    else: raise ValueError("Invalid case selection")
    print(f"  Attack: ε={args.epsilon:.4f}, γ={args.gamma:.4f}, K={args.num_steps}")
    
    # Load data
    testloader = load_CIFAR100(args)
    
    # Load models
    models_dict = load_CIFAR100_models(args)

    epsilon = args.epsilon
    a = args.a
    b = args.b
    window = args.window_type
    
    case = args.case
    
    # Generate Gabor operators
    Psi_2D, Psi_plus_2D, D_inv_1, M, eps_scale, mu_M, U_M, M_herm, cond_S, use_eig \
            = generate_gabor_operators_CIFAR100(args.device, a, b, window)

    if case == "case4":
        print("\nCase 4 selected: Using AutoAttack for adversarial example generation.")
        print("No Gabor operators needed for Case 4.")
        attacker = None  # Not used for AutoAttack
        metrics_evaluator = AdversarialMetrics(
                device=args.device,
                lpips_net=args.lpips_net,
                verbose=args.verbose,
                M=None,
                case_type=4,
            )
        
        # Evaluate models
        all_results = {}

        for model_name, model in models_dict.items():
            results = evaluate_model(Psi_2D, Psi_plus_2D, M, model, model_name, attacker, metrics_evaluator,
                                    testloader, args.case, args)
            all_results[model_name] = results
        
        # Print and save results
        print_results_table(all_results, args)
        save_results(all_results, args)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!".center(80))
        print("=" * 80)

    else:   
    # Generate Gabor operators (needed for Case 1)
        print("\nGenerating Gabor operators...")
        
        # Initialize attacker
        print("\nInitializing DGF-PGD attacker...")
        attacker = DGFPGDAttack(
        model=list(models_dict.values())[0],
        loss_fn=nn.CrossEntropyLoss(),
        Psi_2D=Psi_2D, Psi_plus_2D=Psi_plus_2D,
        D_inv_1=D_inv_1,
        M=M, eps_scale=eps_scale, mu_M=mu_M, U_M=U_M, M_herm=M_herm, use_eig=use_eig,
        image_shape=(3, 32, 32), epsilon=epsilon, gamma=args.gamma,
        num_steps=args.num_steps, case=args.case,
        device=args.device, verbose=args.verbose
    )
        
        # Initialize metrics evaluator
        print("Initializing metrics evaluator...")
        metrics_evaluator = AdversarialMetrics(
            device=args.device,
            lpips_net=args.lpips_net,
            verbose=args.verbose,
            M=M,
            case_type=args.case
        )

        # Evaluate models
        all_results = {}
        for model_name, model in models_dict.items():
            results = evaluate_model(Psi_2D, Psi_plus_2D, M, model, model_name, attacker, metrics_evaluator,
                                    testloader, args.case, args)
            all_results[model_name] = results
        
        # Print and save results
        print_results_table(all_results, args)
        save_results(all_results, args)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!".center(80))
        print("=" * 80)


if __name__ == "__main__":
    main()