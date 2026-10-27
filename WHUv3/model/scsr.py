"""SCSR-V3: structure-conditioned, signed high-frequency residual for TIR.

This is the design in idea.md, not a reproduction of NS-FPN/FreqFiT/PFESA/SFP.
Haar uses the pytorch_wavelets convention: LL, LH (high along height),
HL (high along width), HH. Even grids need no boundary extension.
"""
import math
from contextlib import nullcontext

import torch
from torch import nn


def _disable_autocast(device_type):
    # CUDA training follows UAD's longstanding AMP API (also on older PyTorch).
    if device_type == 'cuda':
        return torch.cuda.amp.autocast(enabled=False)
    if device_type == 'cpu':
        cpu_amp = getattr(getattr(torch, 'cpu', None), 'amp', None)
        autocast = getattr(cpu_amp, 'autocast', None)
        return autocast(enabled=False) if autocast is not None else nullcontext()
    raise ValueError('SCSR supports the UAD CUDA path and CPU checks only')


def haar_dwt(x):
    """Orthogonal one-level DWT, NCHW -> four N,C,H/2,W/2 tensors."""
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar DWT requires a four-dimensional even spatial grid")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a + b + c + d) * 0.5, (a + b - c - d) * 0.5,
            (a - b + c - d) * 0.5, (a - b - c + d) * 0.5)


def haar_idwt(ll, lh, hl, hh):
    """Paired inverse; stacking interleaves the original row-major 2x2 cells."""
    if ll.ndim != 4 or any(t.shape != ll.shape for t in (lh, hl, hh)):
        raise ValueError("Haar IDWT requires four equally shaped NCHW subbands")
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


def smooth_bounded_residual(patches, delta, scale=0.2, eps=1e-6):
    """Frozen V3 operator, per sample; caller controls the FP32 precision scope.

    No detach or hard clipping: radial gradients remain nonzero above t=1.
    With epsilon the exact asymptotic bound is scale * (||P||_F + eps).
    """
    p_norm = patches.flatten(1).norm(dim=1).view(-1, 1, 1)
    delta_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1)
    return scale * delta / (1.0 + delta_norm / (p_norm + eps))


class SCSR(nn.Module):
    def __init__(self, grid_size=(16, 8), in_dim=768, spectral_dim=192,
                 condition_dim=64, gate_amplitude=0.5, residual_scale=0.2,
                 eps=1e-6):
        super().__init__()
        self.grid_size = tuple(grid_size)
        if len(self.grid_size) != 2 or any(v <= 0 or v % 2 for v in self.grid_size):
            raise ValueError("SCSR requires a positive even patch grid")
        if min(in_dim, spectral_dim, condition_dim) <= 0:
            raise ValueError("SCSR channel dimensions must be positive")
        if not all(math.isfinite(v) for v in (gate_amplitude, residual_scale, eps)):
            raise ValueError("SCSR scalar settings must be finite")
        if not (0 < gate_amplitude <= 1 and 0 < residual_scale <= 1 and eps > 0):
            raise ValueError("SCSR amplitudes must be in (0,1] and EPS positive")
        self.in_dim = in_dim
        self.gate_amplitude = gate_amplitude
        self.residual_scale = residual_scale
        self.eps = eps
        # Detached local diagnostics, excluded from parameters/checkpoints/DDP buffers.
        self.last_stats = None
        # Opt-in diagnostic callback; never registered as a parameter or buffer.
        self.diagnostic_callback = None
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, spectral_dim, bias=False)
        self.condition = nn.Sequential(
            nn.Conv2d(4 * spectral_dim, condition_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(condition_dim, condition_dim, 3, padding=1,
                      groups=condition_dim, bias=False),
            nn.GELU(),
            nn.Conv2d(condition_dim, 3 * spectral_dim, 1, bias=True),
        )
        self.up = nn.Linear(spectral_dim, in_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        for layer in (self.condition[0], self.condition[2]):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, patches):
        h, w = self.grid_size
        if patches.ndim != 3 or patches.shape[1:] != (h * w, self.in_dim):
            raise ValueError("SCSR expects B x {} x {} patches".format(h * w, self.in_dim))
        # AMP keeps master parameters FP32. Do not call model.half() for SCSR.
        with _disable_autocast(patches.device.type):
            p = patches.float()
            z = self.down(self.norm(p)).transpose(1, 2).reshape(p.shape[0], -1, h, w)
            ll, lh, hl, hh = haar_dwt(z)
            q = self.condition(torch.cat((ll, lh, hl, hh), dim=1))
            gates = (self.gate_amplitude * q.tanh()).chunk(3, dim=1)
            change = haar_idwt(torch.zeros_like(ll), gates[0] * lh,
                               gates[1] * hl, gates[2] * hh)
            delta = self.up(change.flatten(2).transpose(1, 2))
            p_norm = p.flatten(1).norm(dim=1).view(-1, 1, 1)
            residual = smooth_bounded_residual(p, delta, self.residual_scale, self.eps)
            if self.training:
                with torch.no_grad():
                    modulation = self.gate_amplitude * q.detach().tanh()
                    ratio = residual.detach().flatten(1).norm(dim=1) / (p_norm.flatten() + self.eps)
                    self.last_stats = torch.stack((modulation.abs().mean(),
                                                   modulation.abs().max(), ratio.mean()))
        cast_residual = residual.to(dtype=patches.dtype)
        updated = patches + cast_residual
        if self.diagnostic_callback is not None:
            self.diagnostic_callback({
                'p6_tir': patches, 'projected': z,
                'll': ll, 'lh': lh, 'hl': hl, 'hh': hh, 'q': q,
                'gate_increment': self.gate_amplitude * q.tanh(),
                'idwt_change': change, 'up_delta': delta,
                'residual_fp32': residual, 'residual_cast': cast_residual,
                'updated': updated})
        return updated
