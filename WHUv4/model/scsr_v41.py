"""SCSR-V4.1: TIR-only Haar modulation of Block 7's norm1 output.

No feature bottleneck, extra normalization, residual scale, or extra loss.
Subband names follow the explicit formulas below (LH: high along height).
"""
from contextlib import nullcontext

import torch
from torch import nn


def fp32_context(device):
    # CUDA API available in the server's PyTorch 1.12; CPU AMP is optional.
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(enabled=False)
    cpu_amp = getattr(getattr(torch, 'cpu', None), 'amp', None)
    autocast = getattr(cpu_amp, 'autocast', None)
    return autocast(enabled=False) if autocast is not None else nullcontext()


def haar_dwt(x):
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError('Haar requires BCHW with even spatial dimensions')
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 2, (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2


def haar_idwt(ll, lh, hl, hh):
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError('Haar subbands must have identical shapes')
    # Stack interleaved pixels without in-place operations on autograd views.
    a, b = (ll + lh + hl + hh) / 2, (ll + lh - hl - hh) / 2
    c, d = (ll - lh + hl - hh) / 2, (ll - lh - hl + hh) / 2
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


def rms(x):
    return x.detach().float().square().mean().sqrt()


class SCSRV41(nn.Module):
    VERSION = 'SCSR-V4.1'

    def __init__(self, cfg, modalities):
        super().__init__()
        fixed = dict(INSERT_AT_BLOCK=7, INSERT_POSITION='norm1_after_before_attention',
                     WAVELET='haar', LEVELS=1, EMBED_DIM=768, GRID_H=16, GRID_W=8,
                     LL_CONDITION_DIM=16, HF_SIGNED_DIM=4, CONDITION_HIDDEN_DIM=32,
                     LOW_RANK=4, GATE_AMPLITUDE=0.2, SCSR_WEIGHT_DECAY=0.0,
                     SCSR_LR_MULTIPLIER=1.0, U_INIT_STD=0.5, CONDITION_OUTPUT_INIT_STD=0.02,
                     EPS=1e-6)
        for key, expected in fixed.items():
            if getattr(cfg, key) != expected:
                raise ValueError('V4.1 freezes {}={}; got {}'.format(key, expected, getattr(cfg, key)))
        if len(set(modalities)) != len(modalities) or cfg.TARGET_MODALITY != 'Thermal' or 'Thermal' not in modalities:
            raise ValueError('V4.1 requires unique modalities including Thermal')
        if cfg.EPS <= 0 or cfg.U_INIT_STD <= 0 or cfg.CONDITION_OUTPUT_INIT_STD <= 0:
            raise ValueError('Initialization scales and EPS must be positive')
        self.target_id = list(modalities).index('Thermal') + 1
        self.num_modalities = len(modalities)
        self.eps = cfg.EPS
        self.amplitude = cfg.GATE_AMPLITUDE
        self.ll_proj = nn.Conv2d(768, 16, 1, bias=False)
        self.hf_proj = nn.ModuleList([nn.Conv2d(768, 4, 1, bias=False) for _ in range(3)])
        self.spatial = nn.Conv2d(31, 32, 3, padding=1, bias=False)
        self.activation = nn.GELU()
        self.output = nn.Conv2d(32, 12, 1, bias=False)
        self.U = nn.Parameter(torch.empty(3, 768, 4))
        for layer in [self.ll_proj, *self.hf_proj, self.spatial]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
        nn.init.normal_(self.output.weight, std=cfg.CONDITION_OUTPUT_INIT_STD)
        nn.init.normal_(self.U, std=cfg.U_INIT_STD)
        self.collect_stats = False
        self.last_stats = {}
        self.diagnostic_callback = None
        self.calibration = None
        self.calibrated = False
        assert sum(p.numel() for p in self.parameters()) == 40032

    def forward(self, tokens, modal_ids):
        if tokens.ndim != 3 or tokens.shape[1:] != (129, 768):
            raise ValueError('V4.1 expects [B,129,768] norm1 tokens on a 16x8 grid')
        if not isinstance(modal_ids, torch.Tensor) or modal_ids.dtype != torch.long:
            raise ValueError('Pass true per-sample modality IDs as a torch.long tensor')
        if modal_ids.shape != (tokens.shape[0],) or modal_ids.device != tokens.device:
            raise ValueError('Modality IDs must match the batch and device')
        if ((modal_ids < 1) | (modal_ids > self.num_modalities)).any():
            raise ValueError('Unknown sensor ID; evaluation mode is not a sensor ID')
        self.last_stats = {}
        mask = modal_ids == self.target_id
        if not mask.any():
            return tokens
        original = tokens[mask, 1:]
        with fp32_context(tokens.device):
            # ViT tokens transpose into a channels-last-like view. Keep the
            # convolution path explicitly NCHW, especially for old cuDNN/DDP
            # gradient layout handling of singleton 1x1 kernel dimensions.
            p = original.float().transpose(1, 2).reshape(-1, 768, 16, 8).contiguous()
            ll, *high = haar_dwt(p)
            energy = [h.square().mean(1, keepdim=True).add(self.eps).sqrt() for h in high]
            condition = torch.cat([self.ll_proj(ll)] + [proj(h) for proj, h in zip(self.hf_proj, high)] + energy, dim=1)
            v = self.output(self.activation(self.spatial(condition))).reshape(-1, 3, 4, 8, 4)
            t = torch.einsum('jcr,bjrhw->bjchw', self.U, v)
            m = self.amplitude * torch.tanh(t)
            updated_high = [(1 + m[:, j]) * h for j, h in enumerate(high)]
            updated_fp32 = haar_idwt(ll, *updated_high).flatten(2).transpose(1, 2)
            updated = updated_fp32.to(original.dtype)
            if self.collect_stats:
                denominator = original.detach().float().flatten(1).norm(dim=1) + self.eps
                ratio = (updated.detach().float() - original.detach().float()).flatten(1).norm(dim=1) / denominator
                stats = dict(v_rms=float(rms(v)), t_rms=float(rms(t)), t_max=float(t.detach().abs().max()),
                             t_saturation=float((t.detach().abs() > 2).float().mean()),
                             m_rms=float(rms(m)), r_attn_mean=float(ratio.mean()),
                             r_attn_max=float(ratio.max()), changed_ratio=float((updated != original).float().mean()))
                stats['norm1_input_dtype'] = str(original.dtype)
                stats['internal_dtype'] = str(v.dtype)
                stats['writeback_dtype'] = str(updated.dtype)
                stats['v_sample_rms_std'] = float(v.detach().flatten(1).square().mean(1).sqrt().std(unbiased=False))
                for j, name in enumerate(('lh', 'hl', 'hh')):
                    stats[name + '_m_mean'] = float(m[:, j].detach().mean())
                    stats[name + '_m_std'] = float(m[:, j].detach().std(unbiased=False))
                    stats[name + '_m_max'] = float(m[:, j].detach().abs().max())
                self.last_stats = stats
            if self.diagnostic_callback is not None:
                self.diagnostic_callback(dict(z_tir=original, ll=ll, lh=high[0], hl=high[1], hh=high[2],
                                              v=v, t=t, m=m, updated_fp32=updated_fp32, updated=updated))
        # Clone preserves CLS, sample order, and every RGB/NIR value exactly.
        result = tokens.clone()
        result[mask, 1:] = updated
        return result
