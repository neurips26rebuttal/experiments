"""
Simplified Evaluation Metrics for Adversarial Attacks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

# Optional imports with graceful degradation
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


class AdversarialMetrics:
    """
    Simplified adversarial attack evaluation metrics
    """

    def __init__(
        self,
        device: str = 'cuda',
        lpips_net: str = 'alex',
        verbose: bool = True,
    ):
        """
        Initialize metrics evaluator
        """
        self.device = device
        self.verbose = verbose

        # Initialize LPIPS if available
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(device)
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        else:
            self.lpips_model = None
            if verbose:
                print("Warning: LPIPS not available. Install with: pip install lpips")
        
        if not SSIM_AVAILABLE and verbose:
            print("Warning: SSIM not available. Install with: pip install pytorch-msssim")
    
    def compute_classification_metrics(
        self,
        model: nn.Module,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute classification accuracy metrics
        """
        model.eval()
        with torch.no_grad():
            # Clean predictions
            outputs_clean = model(x_clean)
            pred_clean = outputs_clean.argmax(dim=1)
            clean_correct = (pred_clean == y_true)
            
            # Adversarial predictions
            outputs_adv = model(x_adv)
            pred_adv = outputs_adv.argmax(dim=1)
            adv_correct = (pred_adv == y_true)
            
            attack_success = ~adv_correct  # Adversarial examples that fooled the model
            
            metrics = {
                'clean_accuracy': clean_correct.float().mean().item(),
                'adversarial_accuracy': adv_correct.float().mean().item(),
                'attack_success_rate': attack_success.float().mean().item()
            }
        
        return metrics
    
    def compute_perturbation_metrics(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute perturbation norms: L2, L∞, and Gabor frame norm
        """
        perturbation = x_adv - x_clean
        B = perturbation.shape[0]
        
        # L2 norm per sample: ||δ||_2
        l2_norms = torch.norm(perturbation.view(B, -1), p=2, dim=1)
        
        # L∞ norm per sample: ||δ||_∞
        linf_norms = torch.norm(perturbation.view(B, -1), p=float('inf'), dim=1)
        
        metrics = {
            'mean_l2_norm': l2_norms.mean().item(),
            'std_l2_norm': l2_norms.std().item(),
            'mean_linf_norm': linf_norms.mean().item(),
            'std_linf_norm': linf_norms.std().item()
        }
        
        return metrics
    

    def compute_perceptual_metrics(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute perceptual metrics: LPIPS, SSIM and PSNR
        """
        metrics = {}

        with torch.no_grad():
            mse = (x_adv - x_clean).pow(2).flatten(1).mean(dim=1)
            psnr_values = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
            metrics['psnr_mean'] = psnr_values.mean().item()
            metrics['psnr_std'] = psnr_values.std().item()
        
        # LPIPS
        if self.lpips_model is not None:
            with torch.no_grad():
                # LPIPS expects images in [-1, 1]
                x_clean_normalized = x_clean * 2 - 1
                x_adv_normalized = x_adv * 2 - 1
                
                lpips_values = self.lpips_model(x_clean_normalized, x_adv_normalized)
                lpips_values = lpips_values.view(-1)
                
                metrics['lpips_mean'] = lpips_values.mean().item()
                metrics['lpips_std'] = lpips_values.std().item()
        else:
            metrics['lpips_mean'] = None
            metrics['lpips_std'] = None
        
        # SSIM
        if SSIM_AVAILABLE:
            with torch.no_grad():
                # SSIM per sample
                B = x_clean.shape[0]
                ssim_values = []
                for i in range(B):
                    ssim_val = ssim(
                        x_clean[i:i+1], x_adv[i:i+1],
                        data_range=1.0, size_average=True
                    )
                    ssim_values.append(ssim_val.item())
                
                metrics['ssim_mean'] = np.mean(ssim_values)
                metrics['ssim_std'] = np.std(ssim_values)
        else:
            metrics['ssim_mean'] = None
            metrics['ssim_std'] = None
        
        return metrics
    
    def compute_all_metrics(
        self,
        model: nn.Module,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all metrics
        """
        metrics = {}

        # Classification metrics
        metrics.update(self.compute_classification_metrics(model, x_clean, x_adv, y_true))
        
        # Perturbation metrics
        metrics.update(self.compute_perturbation_metrics(x_clean, x_adv))

        # Perceptual metrics
        metrics.update(self.compute_perceptual_metrics(x_clean, x_adv))
        
        return metrics