"""Frozen-model multi-layer Haar diagnostics.  This module is never imported by training."""
from contextlib import contextmanager, nullcontext
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from model.scsr_v41 import haar_dwt, haar_idwt, fp32_context


LAYERS = (3, 6, 7, 9, 12)
BANDS = ('LL', 'LH', 'HL', 'HH')


def patches_to_map(tokens):
    if tokens.ndim != 3 or tokens.shape[1:] != (129, 768):
        raise ValueError('Expected [B,129,768] tokens for the fixed 16x8 ViT grid')
    return tokens[:, 1:].float().transpose(1, 2).reshape(-1, 768, 16, 8).contiguous()


def decompose(tokens):
    with fp32_context(tokens.device):
        return haar_dwt(patches_to_map(tokens))


def descriptor(x, kind, all_bands=None):
    """Return independently normalized descriptors; energy never masquerades as pattern."""
    x = x.float()
    if kind == 'signed_spatial':
        value = x.flatten(1)
    elif kind == 'channel_rms':
        value = x.square().mean((2, 3)).add(1e-12).sqrt()
    elif kind == 'energy':
        value = x.square().mean((1, 2, 3), keepdim=False).sqrt().unsqueeze(1)
    elif kind == 'energy_ratio':
        if all_bands is None:
            raise ValueError('energy_ratio requires all four subbands')
        energies = torch.stack([v.float().square().mean((1, 2, 3)) for v in all_bands], 1)
        value = x.square().mean((1, 2, 3)).div(energies.sum(1).clamp_min(1e-12)).unsqueeze(1)
    else:
        raise ValueError('Unknown descriptor type ' + kind)
    return F.normalize(value, dim=1) if value.shape[1] > 1 else value


def squared_distance(a, b):
    return (a.float() - b.float()).square().sum(1)


def paired_distances(values, pairs, chunk_size=64):
    result = []
    for start in range(0, len(pairs), chunk_size):
        part = pairs[start:start + chunk_size]
        result.append(squared_distance(values[[a for a, _ in part]], values[[b for _, b in part]]).cpu())
    return torch.cat(result) if result else torch.empty(0)


def pair_auc(positive, negative):
    """AUC for smaller-distance-is-positive, with exact half credit for ties."""
    p, n = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    if not len(p) or not len(n):
        return float('nan')
    values = np.concatenate((-p, -n))
    labels = np.concatenate((np.ones(len(p), dtype=np.int64), np.zeros(len(n), dtype=np.int64)))
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def tensor_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().contiguous().cpu()
        digest.update(name.encode('utf-8')); digest.update(str(cpu.dtype).encode('ascii'))
        digest.update(np.asarray(cpu.shape, dtype=np.int64).tobytes()); digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


class LayerCapture(object):
    def __init__(self, model, layers=LAYERS, keep_grad=False):
        self.model, self.layers, self.keep_grad = model, tuple(layers), keep_grad
        self.values, self.handles = {}, []

    def __enter__(self):
        if self.handles:
            raise RuntimeError('LayerCapture cannot be entered twice')
        for layer in self.layers:
            block = self.model.base.blocks[layer - 1]
            def hook(module, inputs, output, layer=layer):
                self.values[layer] = output if self.keep_grad else output.detach()
            self.handles.append(block.norm1.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _seeded_sign(shape, seed, device):
    generator = torch.Generator(device='cpu'); generator.manual_seed(int(seed))
    return torch.randint(0, 2, shape, generator=generator, dtype=torch.float32).mul_(2).sub_(1).to(device)


def _matched_delta(reference, mode, seed):
    """Construct deterministic per-image controls with exactly matched L2 norm."""
    b, c, h, w = reference.shape
    if mode == 'random':
        direction = _seeded_sign(reference.shape, seed, reference.device)
    elif mode == 'spatial':
        direction = _seeded_sign((b, 1, h, w), seed, reference.device).expand_as(reference)
    elif mode == 'channel':
        direction = _seeded_sign((b, c, 1, 1), seed, reference.device).expand_as(reference)
    else:
        raise ValueError(mode)
    scale = reference.flatten(1).norm(dim=1) / direction.flatten(1).norm(dim=1).clamp_min(1e-12)
    return direction * scale[:, None, None, None]


class FrequencyIntervention(object):
    """Temporary norm1-output intervention; only selected TIR patches are replaced."""
    def __init__(self, model, layer, band=None, sign=1, alpha=0., modal_ids=None,
                 control='frequency', seed=0, collect=False):
        if layer not in LAYERS or band not in BANDS + (None,):
            raise ValueError('Unsupported layer/band')
        if sign not in (-1, 1) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError('sign must be +/-1 and alpha finite/nonnegative')
        self.model, self.layer, self.band = model, layer, band
        self.sign, self.alpha, self.modal_ids = sign, float(alpha), modal_ids
        self.control, self.seed, self.collect = control, int(seed), collect
        self.handle, self.records = None, []

    def _hook(self, module, inputs, output):
        ids = self.modal_ids
        if ids is None or ids.shape != (output.shape[0],) or ids.device != output.device:
            raise ValueError('Intervention modal_ids must match the current batch')
        mask = ids == 3
        if not mask.any() or self.control == 'none' or self.alpha == 0:
            return output
        original = output[mask, 1:]
        with fp32_context(output.device):
            patch = original.float().transpose(1, 2).reshape(-1, 768, 16, 8).contiguous()
            parts = list(haar_dwt(patch))
            if self.control in ('frequency', 'random_frequency'):
                index = BANDS.index(self.band)
                if self.control == 'frequency':
                    delta = self.sign * self.alpha * parts[index]
                else:
                    delta = _matched_delta(self.alpha * parts[index], 'random', self.seed)
                parts[index] = parts[index] + delta
                changed = haar_idwt(*parts)
            else:
                index = BANDS.index(self.band)
                frequency_delta = haar_idwt(*[
                    self.alpha * parts[i] if i == index else torch.zeros_like(parts[i]) for i in range(4)])
                delta = _matched_delta(frequency_delta, self.control, self.seed)
                changed = patch + self.sign * delta
            changed_tokens = changed.flatten(2).transpose(1, 2)
            ratio = ((changed_tokens - original.float()).flatten(1).norm(2, 1) /
                     output[mask].float().flatten(1).norm(2, 1).clamp_min(1e-12))
            if self.collect:
                self.records.extend(float(v) for v in ratio.detach().cpu())
        result = output.clone()
        result[mask, 1:] = changed_tokens.to(original.dtype)
        return result

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError('FrequencyIntervention cannot be entered twice')
        self.handle = self.model.base.blocks[self.layer - 1].norm1.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove(); self.handle = None


def calibration_alpha(tokens, band, target_ratio, cap):
    """Exact linear alpha for a batch-level attention-input L2 budget."""
    parts = decompose(tokens)
    delta = haar_idwt(*[parts[i] if i == BANDS.index(band) else torch.zeros_like(parts[i]) for i in range(4)])
    numerator = tokens.float().flatten(1).norm(2, 1)
    denominator = delta.flatten(1).norm(2, 1).clamp_min(1e-12)
    per_image = target_ratio * numerator / denominator
    raw = float(per_image.median())
    return min(raw, cap), raw


def relative_l2(a, b):
    return (b.float() - a.float()).flatten(1).norm(2, 1) / a.float().flatten(1).norm(2, 1).clamp_min(1e-12)


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(1), b.float().flatten(1), dim=1)


def reconstruct_v41_condition(module, captured, verify_v=True):
    """Recreate the production V4.1 condition omitted by its lightweight callback."""
    required = ('ll', 'lh', 'hl', 'hh', 'v')
    if any(key not in captured for key in required):
        raise ValueError('Incomplete production SCSR capture; keys={}'.format(sorted(captured)))
    device = module.ll_proj.weight.device
    with torch.no_grad(), fp32_context(device):
        ll = captured['ll'].to(device=device, dtype=torch.float32)
        high = [captured[key].to(device=device, dtype=torch.float32) for key in ('lh', 'hl', 'hh')]
        energy = [value.square().mean(1, keepdim=True).add(module.eps).sqrt() for value in high]
        condition = torch.cat([module.ll_proj(ll)] +
                              [projection(value) for projection, value in zip(module.hf_proj, high)] + energy, 1)
        rebuilt_v = module.output(module.activation(module.spatial(condition))).reshape(-1, 3, 4, 8, 4)
    expected_v = captured['v'].float().cpu()
    rebuilt_cpu = rebuilt_v.float().cpu()
    if verify_v and not torch.allclose(rebuilt_cpu, expected_v, atol=2e-6, rtol=2e-5):
        error = float((rebuilt_cpu - expected_v).abs().max())
        raise AssertionError('Reconstructed V4.1 condition does not reproduce captured V; max error={}'.format(error))
    return condition.detach().float().cpu(), rebuilt_cpu


@contextmanager
def no_parameter_mutation(model):
    before = tensor_sha256(model.state_dict())
    modes = {name: module.training for name, module in model.named_modules()}
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    try:
        model.eval()
        yield before
        if tensor_sha256(model.state_dict()) != before:
            raise AssertionError('Diagnostic mutated a model parameter or buffer')
        if any(parameter.grad is not grads[name] for name, parameter in model.named_parameters()):
            raise AssertionError('Diagnostic populated or replaced a parameter .grad buffer')
    finally:
        for name, module in model.named_modules():
            module.train(modes[name])

