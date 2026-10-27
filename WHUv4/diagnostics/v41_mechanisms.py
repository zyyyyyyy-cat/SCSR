"""Evaluation-only SCSR interventions. Production forward/training are untouched."""
from contextlib import contextmanager
from types import MethodType
import random

import numpy as np
import torch

from model.scsr_v41 import fp32_context, haar_dwt, haar_idwt
from .core import Capture, amp_context, diagnostic_session, difference_stats, tensor_stats
from .data import metadata, image_loader


@contextmanager
def parameter_intervention(module, state):
    if module.training:
        raise RuntimeError('Parameter interventions are eval-only')
    saved = {key: value.detach().clone() for key, value in module.state_dict().items()}
    try:
        module.load_state_dict(state, strict=True)
        yield
    finally:
        module.load_state_dict(saved, strict=True)


def intervention_forward(self, tokens, modal_ids, kind='normal', donor_v=None):
    """Mirror the frozen production operator; alter only the named mechanism."""
    if self.training:
        raise RuntimeError('Mechanism intervention cannot run during training')
    if tokens.ndim != 3 or tokens.shape[1:] != (129, 768):
        raise ValueError('Expected V4.1 norm1 [B,129,768] tokens')
    if (not isinstance(modal_ids, torch.Tensor) or modal_ids.dtype != torch.long or
            modal_ids.shape != (len(tokens),) or modal_ids.device != tokens.device):
        raise ValueError('Pass true per-sample Long sensor IDs on the input device')
    if ((modal_ids < 1) | (modal_ids > self.num_modalities)).any():
        raise ValueError('Unknown sensor ID')
    mask = modal_ids == self.target_id
    if not mask.any():
        return tokens
    original = tokens[mask, 1:]
    with fp32_context(tokens.device):
        p = original.float().transpose(1, 2).reshape(-1, 768, 16, 8).contiguous()
        ll, *high = haar_dwt(p)
        energy = [h.square().mean(1, keepdim=True).add(self.eps).sqrt() for h in high]
        low_condition = self.ll_proj(ll)
        if kind == 'zero_ll_condition':
            low_condition = torch.zeros_like(low_condition)
        condition = torch.cat([low_condition] + [proj(h) for proj, h in zip(self.hf_proj, high)] + energy, dim=1)
        own_v = self.output(self.activation(self.spatial(condition))).reshape(-1, 3, 4, 8, 4)
        v = own_v
        if kind == 'spatial_mean_v':
            v = own_v.mean((-2, -1), keepdim=True).expand_as(own_v)
        elif kind in ('donor_same_platform_v', 'donor_cross_platform_v'):
            if donor_v is None or donor_v.shape != own_v.shape:
                raise ValueError('Donor V must match the selected TIR samples, directions, rank and grid')
            v = donor_v.to(device=p.device, dtype=torch.float32)
        t = torch.einsum('jcr,bjrhw->bjchw', self.U, v)
        m = self.amplitude * torch.tanh(t)
        applied_m = m.clone()
        disabled = {'disable_lh': (0,), 'disable_hl': (1,), 'disable_hh': (2,), 'disable_all': (0, 1, 2)}.get(kind, ())
        for index in disabled:
            applied_m[:, index] = 0
        updated_high = [(1 + applied_m[:, j]) * h for j, h in enumerate(high)]
        updated_fp32 = haar_idwt(ll, *updated_high).flatten(2).transpose(1, 2)
        # All-disabled is analytically identity. Avoid mistaking Haar roundoff
        # for a mechanism effect; report the numerical shortcut in the manifest.
        if kind == 'disable_all':
            updated_fp32 = original.float()
        updated = updated_fp32.to(original.dtype)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(dict(z_tir=original, ll=ll, lh=high[0], hl=high[1], hh=high[2],
                                          condition=condition, own_v=own_v, v=v, t=t, m=m, applied_m=applied_m,
                                          updated_lh=updated_high[0], updated_hl=updated_high[1], updated_hh=updated_high[2],
                                          updated_fp32=updated_fp32, updated=updated))
    result = tokens.clone()
    result[mask, 1:] = updated
    return result


@contextmanager
def mechanism_intervention(module, kind='normal', donor_v=None):
    allowed = ('normal', 'spatial_mean_v', 'donor_same_platform_v', 'donor_cross_platform_v',
               'zero_ll_condition', 'disable_lh', 'disable_hl', 'disable_hh', 'disable_all')
    if kind not in allowed or module.training:
        raise ValueError('Unknown intervention or model not in eval state: ' + kind)
    had_local = 'forward' in module.__dict__
    old = module.__dict__.get('forward')
    def forward(this, tokens, modal_ids):
        return intervention_forward(this, tokens, modal_ids, kind, donor_v)
    module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        if had_local:
            module.forward = old
        else:
            del module.forward


class MechanismCapture(Capture):
    def backbone(self, name, tensor):
        if name in ('block7_norm1', 'block7_attention_input', 'block7_attention_output', 'block_7', 'final_cls'):
            super().backbone(name, tensor)

    def spectral(self, tensors):
        tensors = dict(tensors)
        # The production callback exposes H and M but not H'. Reproduce its
        # exact FP32 elementwise operator on the SAME device before offloading.
        # This avoids treating ideal FP64 M*H as the actual rounded increment.
        with fp32_context(tensors['m'].device):
            applied = tensors.get('applied_m', tensors['m'])
            for j, name in enumerate(('lh', 'hl', 'hh')):
                if 'updated_' + name not in tensors:
                    tensors['updated_' + name] = (1 + applied[:, j]) * tensors[name]
        super().spectral(tensors)


def capture_batch(model, images, device, use_amp=False, kind=None, donor_v=None):
    cap = MechanismCapture()
    ids = torch.full((len(images),), model.base.scsr_v41.target_id, dtype=torch.long, device=device)
    with torch.no_grad(), amp_context(device, use_amp), diagnostic_session(model, True, cap):
        if kind is None:
            model(images, mode=1, modal_ids=ids)
        else:
            with mechanism_intervention(model.base.scsr_v41, kind, donor_v):
                model(images, mode=1, modal_ids=ids)
    if not cap.internal or any(not value.isfinite().all() for value in cap.internal.values()):
        raise ValueError('Missing or nonfinite mechanism capture')
    if any(not value.isfinite().all() for value in cap.layers.values()):
        raise ValueError('Nonfinite backbone feature')
    return cap


def fixed_tir_samples(dataset, count, seed):
    if count < 4 or count % 2:
        raise ValueError('TIR mechanism samples must be even and at least 4')
    unique = {}
    for split in (dataset.query, dataset.gallery):
        for entry in split['Thermal']:
            unique[entry[0]] = entry
    rng = random.Random(seed)
    selected = []
    for ground in (True, False):
        pools = {}
        for entry in sorted(unique.values(), key=lambda entry: entry[0]):
            if (entry[2] < 5) == ground:
                pools.setdefault(entry[1], []).append(entry)
        if sum(map(len, pools.values())) < count // 2 or len(pools) < 2:
            raise ValueError('Insufficient TIR identities/images on platform ' + ('G' if ground else 'A'))
        ids = sorted(pools); rng.shuffle(ids)
        for pid in ids:
            rng.shuffle(pools[pid])
        chosen = []
        while len(chosen) < count // 2:
            for pid in ids:
                if pools[pid] and len(chosen) < count // 2:
                    chosen.append(pools[pid].pop())
        selected += chosen
    return selected


def fixed_donor_pairs(records, cross_platform=False):
    pairs = []
    for index, record in enumerate(records):
        candidates = [j for j, other in enumerate(records) if other['pid'] != record['pid'] and
                      ((other['platform'] != record['platform']) if cross_platform else
                       (other['platform'] == record['platform']))]
        if not candidates:
            raise ValueError('No different-identity donor satisfying the platform rule')
        # Dataset selection was seeded; this mapping is fixed and batch-independent.
        pairs.append(candidates[index % len(candidates)])
    return pairs


def gate_statistics(cap, records, group, sample_indices):
    rows = []
    for local, (index, record) in enumerate(zip(sample_indices, records)):
        row = dict(record, sample_index=int(index), group=group, row_type='statistics')
        for name in ('v', 't', 'm'):
            row.update({name + '_' + key: value for key, value in tensor_stats(cap.internal[name][local]).items()})
        t = cap.internal['t'][local]
        row['t_saturation_abs_gt_2'] = float((t.abs() > 2).float().mean())
        for j, direction in enumerate(('lh', 'hl', 'hh')):
            gate = cap.internal['m'][local, j].double()
            high = cap.internal[direction][local].double()
            delta = cap.internal['updated_' + direction][local].double() - high
            row.update({direction + '_m_' + key: value for key, value in tensor_stats(gate).items()})
            row[direction + '_m_mean'] = float(gate.mean())
            row[direction + '_m_std'] = float(gate.std(unbiased=False))
            row[direction + '_m_min'] = float(gate.min())
            row[direction + '_m_max'] = float(gate.max())
            row[direction + '_increment_norm'] = float(delta.norm())
            row[direction + '_increment_relative'] = float(delta.norm() / (high.norm() + 1e-6))
        rows.append(row)
    return rows


def paired_gate_rows(reference, other, records, reference_group, group, indices):
    rows = []
    for local, (index, record) in enumerate(zip(indices, records)):
        row = dict(record, sample_index=int(index), reference_group=reference_group, group=group, row_type='comparison')
        for name in ('v', 't', 'm'):
            for suffix, stats in [('all', difference_stats(other.internal[name][local], reference.internal[name][local]))] + [
                    (direction, difference_stats(other.internal[name][local, j], reference.internal[name][local, j]))
                    for j, direction in enumerate(('lh', 'hl', 'hh'))]:
                row.update({name + '_' + suffix + '_' + key: value for key, value in stats.items()})
        rows.append(row)
    return rows


def paired_feature_rows(reference, other, records, reference_group, group, indices):
    rows = []
    for name in reference.layers:
        a, b = other.layers[name], reference.layers[name]
        comparisons = [(name, a, b)]
        if a.ndim == 3:
            comparisons += [(name + '_cls', a[:, 0], b[:, 0]), (name + '_patch', a[:, 1:], b[:, 1:])]
        else:
            comparisons += [('retrieval_l2_cls', torch.nn.functional.normalize(a.float(), dim=1),
                             torch.nn.functional.normalize(b.float(), dim=1))]
        for label, x, y in comparisons:
            for local, (index, record) in enumerate(zip(indices, records)):
                row = dict(record, sample_index=int(index), reference_group=reference_group, group=group, layer=label)
                row.update(difference_stats(x[local], y[local]))
                rows.append(row)
    return rows


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return ranks


def gain_order_stats(a, b, top_k=32):
    a, b = a.detach().double().numpy(), b.detach().double().numpy()
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    denominator = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    k = min(top_k, len(a))
    top_a, top_b = np.argsort(-a, kind='mergesort')[:k], np.argsort(-b, kind='mergesort')[:k]
    return dict(gain_order_spearman=float(np.clip(float(ra @ rb) / denominator, -1., 1.)) if denominator > 0 else '',
                top32_gain_overlap=float(len(set(top_a) & set(top_b)) / k) if denominator > 0 else '',
                gain_order_degenerate=denominator == 0)


def input_dependence_rows(gates, records, same_pairs, cross_pairs):
    """Generator keeps the potentially large per-position CSV out of RAM."""
    for index, record in enumerate(records):
        for pairing, donor in (('same_platform_different_identity', same_pairs[index]),
                               ('cross_platform_different_identity', cross_pairs[index]),
                               ('same_image_different_position', index)):
            for j, direction in enumerate(('lh', 'hl', 'hh')):
                for h in range(8):
                    for w in range(4):
                        if pairing == 'same_image_different_position' and (h, w) == (0, 0):
                            continue
                        other_h, other_w = (0, 0) if pairing == 'same_image_different_position' else (h, w)
                        a, b = gates[index, j, :, h, w], gates[donor, j, :, other_h, other_w]
                        row = dict(record, sample_index=index, pairing=pairing, donor_index=donor,
                                   donor_pid=records[donor]['pid'], donor_platform=records[donor]['platform'],
                                   donor_camid=records[donor]['camid'],
                                   donor_path=records[donor]['path'], direction=direction, h=h, w=w,
                                   donor_h=other_h, donor_w=other_w)
                        row.update(difference_stats(a, b))
                        row.update(gain_order_stats(a, b))
                        yield row
