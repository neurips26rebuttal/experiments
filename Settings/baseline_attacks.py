"""Baseline transfer attacks vendored from baselines/ as case 5 and case 6.

  case 5 -> SSAAttack   : Spectrum Simulation Attack (Long et al., ECCV 2022)
                          baselines/SSA/attack.py
  case 6 -> AdvDropAttack: AdvDrop / InfoDrop (Duan et al., ICCV 2021)
                          baselines/AdvDrop/infod_sample.py

Why vendored rather than imported from baselines/:
  * the originals hardcode `.cuda()` throughout (phi_diff, the 8x8 basis, the
    gaussian noise), so they crash on CPU and on a non-default GPU;
  * they carry heavy deps the eval harness does not need or have on the compute
    nodes -- pretrainedmodels, torchattacks, their own Normalize/loader;
  * they assume a model that takes [0,255]/[0,1]+own-Normalize input, whereas
    every model in eval_imagenet.py / eval_cifar100.py already wraps its own
    normalization and takes x in [0, 1].
The DCT / 8x8-JPEG math below is a faithful, device-safe port of
baselines/SSA/dct.py and baselines/AdvDrop/{compression,decompression,utils}.py;
the attack loops reproduce the originals' active code path.

Both classes match the calling convention the eval scripts already use for the
DGF-PGD attacker on cases 2/3, so they slot into the existing transferability
loop unchanged:

    attacker.model = source_model
    x_adv, aux = attacker(x, y, random_init=False)      # x, x_adv in [0, 1]

`aux` is always None (kept for tuple-unpacking symmetry with the other cases).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# SSA -- N-averaged spectral-transform DCT/IDCT (port of baselines/SSA/dct.py)
# These use x.dtype / x.device throughout, so they are already device-safe.
# ===========================================================================

def _dct(x, norm=None):
    """1-D DCT-II along the last dim (scipy fftpack convention)."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    V = Vc.real * torch.cos(k) - Vc.imag * torch.sin(k)
    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    return (2 * V).view(*x_shape)


def _idct(X, norm=None):
    """Inverse of :func:`_dct` (DCT-III), such that idct(dct(x)) == x."""
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, N) / 2
    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r, W_i = torch.cos(k), torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    tmp = torch.complex(real=V_r, imag=V_i)
    v = torch.fft.ifft(tmp)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]
    return x.view(*x_shape).real


def _dct_2d(x, norm=None):
    X1 = _dct(x, norm=norm)
    X2 = _dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def _idct_2d(X, norm=None):
    x1 = _idct(X, norm=norm)
    x2 = _idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


class SSAAttack:
    """Spectrum Simulation Attack -- case 5.

    Reproduces the active path of ``Spectrum_Simulation_Attack`` in
    baselines/SSA/attack.py: at each of ``num_steps`` iterations the gradient is
    averaged over ``N`` spectral copies of the input (add Gaussian noise, take the
    2-D DCT, multiply by a random spectral mask, invert), then an L-inf sign step
    is taken and the result is clipped to the eps-ball around the clean image.
    The optional MI / DI / TI-FGSM lines in the original are commented out there;
    we keep them off to match its reported "SSA" numbers, but expose ``momentum``
    so MI-FGSM can be switched on.

    Parity note: upstream hardcodes ``num_iter = 10`` (it ignores even its own
    ``--num_iter_set`` flag), so exact reproduction of the original scenario
    requires ``num_steps=10`` -- which is what the manifest passes. The step
    size is ``epsilon / num_steps``, matching upstream's ``eps / num_iter``.

    Notes on adaptation:
      * the original operates on 299x299 with its own Normalize(0.5, 0.5); here the
        model already normalizes and takes x in [0, 1], so we drop that layer and
        read the spatial size from the input;
      * ``sigma`` stays in 0-255 units (``sigma/255`` std) exactly as upstream;
      * gradients use autograd.grad, not loss.backward(), so no parameter grads
        are accumulated on the model.
    """

    def __init__(self, model, epsilon, num_steps=10, N=20, rho=0.5, sigma=16.0,
                 momentum=0.0, device="cuda", loss_fn=None, verbose=False, **_):
        self.model = model
        self.epsilon = float(epsilon)
        self.num_steps = int(num_steps)
        self.N = int(N)
        self.rho = float(rho)
        self.sigma = float(sigma)
        self.momentum = float(momentum)
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.verbose = verbose
        self.case = "case5"

    @torch.no_grad()
    def _clip(self, x, lo, hi):
        return torch.min(torch.max(x, lo), hi)

    def __call__(self, images, labels, random_init=False):
        images = images.to(self.device)
        labels = labels.to(self.device)
        eps = self.epsilon
        alpha = eps / max(self.num_steps, 1)
        lo = (images - eps).clamp(0.0, 1.0)
        hi = (images + eps).clamp(0.0, 1.0)

        x = images.clone()
        grad = torch.zeros_like(x)
        self.model.eval()
        for _ in range(self.num_steps):
            noise = torch.zeros_like(x)
            for _n in range(self.N):
                gauss = torch.randn_like(x) * (self.sigma / 255.0)
                x_dct = _dct_2d(x + gauss)
                mask = torch.rand_like(x) * 2 * self.rho + 1 - self.rho
                x_idct = _idct_2d(x_dct * mask).requires_grad_(True)
                logits = self.model(x_idct)
                loss = self.loss_fn(logits, labels)
                noise = noise + torch.autograd.grad(loss, x_idct)[0]
            noise = noise / self.N

            if self.momentum > 0:
                noise = noise / noise.abs().mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
                noise = self.momentum * grad + noise
                grad = noise

            x = (x + alpha * noise.sign()).detach()
            x = self._clip(x, lo, hi)
        return x.detach(), None


# ===========================================================================
# AdvDrop -- device-safe 8x8 JPEG-style DCT quantization
# (port of baselines/AdvDrop/{compression,decompression,utils}.py)
# ===========================================================================

def _dct_8x8_basis(device, dtype):
    """(8,8,8,8) forward-DCT basis and the (8,8) alpha scale, cached per device.

    Built in float64 and cast, exactly as upstream builds them with numpy
    doubles and assigns into a float32 array -- so the tables are bit-identical
    to baselines/AdvDrop/compression.py's.
    """
    idx = torch.arange(8, device=device, dtype=torch.float64)
    x, u = idx.view(8, 1), idx.view(1, 8)
    cos = torch.cos((2 * x + 1) * u * np.pi / 16)           # (x, u)
    basis = torch.einsum("xu,yv->xyuv", cos, cos)           # (x,y,u,v)
    alpha = torch.ones(8, device=device, dtype=torch.float64)
    alpha[0] = 1.0 / np.sqrt(2)
    scale = torch.outer(alpha, alpha) * 0.25
    return basis.to(dtype), scale.to(dtype)


def _idct_8x8_basis(device, dtype):
    idx = torch.arange(8, device=device, dtype=torch.float64)
    u, x = idx.view(8, 1), idx.view(1, 8)
    cos = torch.cos((2 * x + 1) * u * np.pi / 16)           # (u, x)
    basis = torch.einsum("ux,vy->uvxy", cos, cos)           # (u,v,x,y)
    alpha = torch.ones(8, device=device, dtype=torch.float64)
    alpha[0] = 1.0 / np.sqrt(2)
    scale = torch.outer(alpha, alpha)                       # (u, v)
    return basis.to(dtype), scale.to(dtype)


def _block_splitting(image, k=8):
    """(N, H, W) -> (N, H*W/k^2, k, k). Pads H,W up to a multiple of k."""
    n, h, w = image.shape
    dh = int(np.ceil(h / k) * k)
    dw = int(np.ceil(w / k) * k)
    padded = image.new_zeros(n, dh, dw)
    padded[:, :h, :w] = image
    reshaped = padded.view(n, dh // k, k, dw // k, k)
    return reshaped.permute(0, 1, 3, 2, 4).contiguous().view(n, -1, k, k)


def _block_merging(patches, height, width, k=8):
    dh = int(np.ceil(height / k) * k)
    dw = int(np.ceil(width / k) * k)
    n = patches.shape[0]
    reshaped = patches.view(n, dh // k, dw // k, k, k)
    merged = reshaped.permute(0, 1, 3, 2, 4).contiguous().view(n, dh, dw)
    return merged[:, :height, :width]


def _dct_8x8(image, basis, scale):
    return scale * torch.tensordot(image - 128, basis, dims=2)


def _idct_8x8(image, basis, scale):
    return 0.25 * torch.tensordot(image * scale, basis, dims=2) + 128


def _phi_diff(x, alpha):
    """Differentiable soft quantization (utils.phi_diff), device-safe."""
    alpha = torch.as_tensor(alpha, device=x.device, dtype=x.dtype).clamp(max=2.0 - 1e-6)
    s = 1.0 / (1.0 - alpha)
    k = torch.log(2.0 / alpha - 1.0)
    phi_x = torch.tanh((x - (torch.floor(x) + 0.5)) * k) * s
    return (phi_x + 1.0) / 2.0 + torch.floor(x)


class AdvDropAttack:
    """AdvDrop / InfoDrop -- case 6.

    Port of ``InfoDrop`` from baselines/AdvDrop/infod_sample.py. The perturbation
    is not an additive L-inf ball: per 8x8 DCT block and channel, a learnable
    quantization table (bounded in ``[5, q_size]``) is optimized to *drop*
    information while flipping the prediction. The table is trained with Adam and
    a per-step sign update; ``alpha`` anneals the soft-quantizer toward a hard
    rounding over the run.

    UNTARGETED only. The original's targeted branch is dropped rather than
    carried unused: nothing in this repo runs a targeted attack, and a targeted
    switch that no configuration sets is a path no result ever exercised.

    Adaptation: the original works on pixels in [0, 255] and its model carried a
    Normalize that divides by 255. Here the harness's model takes [0, 1], so we
    scale x up to [0, 255] for the DCT math and feed ``rgb/255`` to the model
    (unclamped during optimization, exactly as the original's Normalize layer
    does; only the returned image is clamped). ``epsilon`` does not apply to
    this attack and is ignored (accepted only so the constructor matches the
    others).

    ``batch_size`` mirrors the original constructor's argument: the early-stop
    success rate is ``count / batch_size`` -- so, like upstream, a smaller
    final batch can never trigger the early break. Pass None to fall back to
    the actual batch length.
    """

    def __init__(self, model, epsilon=None, steps=150, block_size=8, q_size=40,
                 lr=0.01, batch_size=None, device="cuda",
                 loss_fn=None, verbose=False, **_):
        self.model = model
        self.steps = int(steps)
        self.block_size = int(block_size)
        self.q_size = float(q_size)
        self.lr = float(lr)
        self.batch_size = int(batch_size) if batch_size else None
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.verbose = verbose
        self.factor_range = [5.0, self.q_size]
        self.alpha_range = [0.1, 1e-20]
        self.case = "case6"

    def __call__(self, images, labels, random_init=False):
        device = self.device
        k = self.block_size
        images = (images.to(device) * 255.0).clamp(0, 255)
        labels = labels.to(device)
        n, _, h, w = images.shape

        dtype = images.dtype
        fwd_basis, fwd_scale = _dct_8x8_basis(device, dtype)
        inv_basis, inv_scale = _idct_8x8_basis(device, dtype)

        block_n = int(np.ceil(h / k) * np.ceil(w / k))
        q_tables = {c: torch.full((n, block_n, k, k), self.q_size,
                                  device=device, dtype=dtype)
                    for c in ("y", "cb", "cr")}
        optimizer = torch.optim.Adam(list(q_tables.values()), lr=self.lr)

        alpha = torch.tensor(self.alpha_range[0], device=device, dtype=dtype)
        alpha_interval = torch.tensor(
            (self.alpha_range[1] - self.alpha_range[0]) / max(self.steps, 1),
            device=device, dtype=dtype)

        # channel-last; the three channels are quantized independently (the
        # original labels them y/cb/cr but applies no colour transform).
        comps_img = images.permute(0, 2, 3, 1)
        components = {"y": comps_img[..., 0],
                      "cb": comps_img[..., 1],
                      "cr": comps_img[..., 2]}

        self.model.eval()
        rgb = images
        for step in range(self.steps):
            for c in q_tables:
                q_tables[c].requires_grad_(True)

            up = {}
            for c, comp in components.items():
                blk = _block_splitting(comp, k)
                blk = _dct_8x8(blk, fwd_basis, fwd_scale)
                blk = _phi_diff(blk / q_tables[c], alpha)      # quantize
                blk = blk * q_tables[c]                         # dequantize
                blk = _idct_8x8(blk, inv_basis, inv_scale)
                up[c] = _block_merging(blk, h, w, k)

            rgb = torch.stack([up["y"], up["cb"], up["cr"]], dim=1)
            # unclamped, matching the original's Normalize(x/255) input path
            logits = self.model(rgb / 255.0)
            # untargeted only: ascend the loss of the true label
            cost = -self.loss_fn(logits, labels)

            optimizer.zero_grad()
            cost.backward()

            alpha = alpha + alpha_interval
            with torch.no_grad():
                for c in q_tables:
                    upd = q_tables[c].detach() - torch.sign(q_tables[c].grad)
                    q_tables[c] = upd.clamp(self.factor_range[0],
                                            self.factor_range[1])

            # early stop exactly as upstream: success count over the
            # *configured* batch size (a short final batch never breaks),
            # checked after the q-table update.
            with torch.no_grad():
                pred = logits.argmax(1)
                suc_rate = (pred != labels).sum().item() / (self.batch_size or n)
            if suc_rate >= 1:
                break

        adv = (rgb / 255.0).clamp(0.0, 1.0).detach()
        return adv, None


# ===========================================================================
# Factory used by the eval scripts
# ===========================================================================

def build_baseline_attacker(case, model, args, image_size):
    """Return the case-5/6 attacker, reading its hyperparameters off `args`.

    `image_size` is accepted for interface symmetry with the DGF operators; the
    attacks read the spatial size from each input batch, so both 224 and 32 work.
    """
    if case == "case5":
        return SSAAttack(
            model, epsilon=args.epsilon,
            num_steps=getattr(args, "ssa_steps", None) or args.num_steps,
            N=getattr(args, "ssa_N", 20),
            rho=getattr(args, "ssa_rho", 0.5),
            sigma=getattr(args, "ssa_sigma", 16.0),
            momentum=getattr(args, "ssa_momentum", 0.0),
            device=args.device, verbose=args.verbose)
    if case == "case6":
        return AdvDropAttack(
            model,
            steps=getattr(args, "advdrop_steps", 150),
            block_size=getattr(args, "advdrop_block_size", 8),
            q_size=getattr(args, "advdrop_q_size", 40),
            lr=getattr(args, "advdrop_lr", 0.01),
            batch_size=getattr(args, "batch_size", None),
            device=args.device, verbose=args.verbose)
    raise ValueError(f"build_baseline_attacker: unsupported case {case!r}")
