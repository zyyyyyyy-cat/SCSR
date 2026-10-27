"""V4.1 capture sites: norm1 -> modulated input -> attention -> final CLS."""
import csv
import hashlib
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from loss.wrt_loss import TripletLoss_WRT
from utils.scsr_v41_runtime import Capture, diagnostic_session


def amp_context(device, enabled):
    if enabled and device.type != 'cuda':
        raise ValueError('AMP diagnostics require CUDA')
    return torch.cuda.amp.autocast(enabled=True) if enabled else nullcontext()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(float(value), '.12e') if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    return value


def tensor_stats(tensor):
    x = tensor.detach().double().flatten()
    return dict(rms=float(x.square().mean().sqrt()), norm=float(x.norm()),
                max_abs=float(x.abs().max()), finite=bool(x.isfinite().all()))


def difference_stats(on, off, eps=1e-6):
    a, b = on.detach().double().flatten(), off.detach().double().flatten()
    delta, product = a - b, float(a.norm() * b.norm())
    return dict(abs_diff=float(delta.norm()), rel_diff=float(delta.norm()) / (float(b.norm()) + eps),
                cos_sim=float((a * b).sum()) / product if product > 0 else '',
                changed_ratio=float((a != b).double().mean()), equal_ratio=float((a == b).double().mean()))


def internal_rows(capture, records, eps):
    if not capture.internal:
        return []
    if len(records) != len(capture.internal['z_tir']):
        raise ValueError('TIR records/capture mismatch')
    rows = []
    for index, record in enumerate(records):
        row = dict(record)
        for name, value in capture.internal.items():
            row[name + '_dtype'] = str(value.dtype)
            row.update({name + '_' + key: result for key, result in tensor_stats(value[index]).items()})
        row.update({'actual_' + key: value for key, value in difference_stats(
            capture.internal['updated'][index], capture.internal['z_tir'][index], eps).items()})
        t, m = capture.internal['t'][index], capture.internal['m'][index]
        row['t_saturation'] = float((t.abs() > 2).float().mean())
        for j, direction in enumerate(('lh', 'hl', 'hh')):
            row[direction + '_m_mean'] = float(m[j].double().mean())
            row[direction + '_m_std'] = float(m[j].double().std(unbiased=False))
            row[direction + '_m_max'] = float(m[j].double().abs().max())
        rows.append(row)
    return rows


def layerwise_rows(on, off, records, eps):
    if set(on.layers) != set(off.layers):
        raise ValueError('ON/OFF capture sites differ')
    rows = []
    for layer in on.layers:
        a, b = on.layers[layer], off.layers[layer]
        comparisons = [(layer, a, b)]
        if a.ndim == 3:
            comparisons += [(layer + '_cls', a[:, 0], b[:, 0]), (layer + '_patch', a[:, 1:], b[:, 1:])]
        elif layer == 'final_cls':
            comparisons += [('retrieval_l2_cls', F.normalize(a.float(), dim=1), F.normalize(b.float(), dim=1))]
        for label, x, y in comparisons:
            if len(x) != len(records):
                raise ValueError('Layer records/capture mismatch')
            for index, record in enumerate(records):
                row = dict(record, layer=label, dtype_on=str(x.dtype), dtype_off=str(y.dtype))
                row.update(difference_stats(x[index], y[index], eps))
                rows.append(row)
    return rows


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_probe(model, images, target, batch_id, device, use_amp=False, loss_scale=1.0):
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('Diagnostic loss scale must be finite and positive')
    rows, summaries = [], []
    for loss_type in ('CE', 'WRT'):
        cap = Capture(gradients=True)
        with diagnostic_session(model, True, cap), torch.enable_grad(), amp_context(device, use_amp):
            score, final_cls, _ = model(images, label=target, mode=0)
            labels = target.repeat(len(images))
            loss = F.cross_entropy(score, labels) if loss_type == 'CE' else TripletLoss_WRT()(final_cls, labels)[0]
        targets = [('final_cls', final_cls)]
        targets += [(key, cap.layers[key]) for key in ('block7_norm1', 'block7_attention_input', 'block7_attention_output')]
        targets += [(key, cap.internal[key]) for key in ('z_tir', 'v', 't', 'm', 'updated_fp32', 'updated')]
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss * loss_scale, [v for _, v in targets + params], allow_unused=True)
        grad_map = {key: None if grad is None else grad.detach().float() / loss_scale
                    for (key, _), grad in zip(targets + params, grads)}
        total_norm = math.sqrt(sum(float(grad_map[key].double().norm()) ** 2 for key, _ in params if grad_map[key] is not None))
        clip = min(1.0, 1.0 / (total_norm + 1e-6)) if math.isfinite(total_norm) else None
        summaries.append(dict(batch=batch_id, loss_type=loss_type, loss=float(loss.detach()), global_grad_norm=total_norm,
                              hypothetical_clip_factor=clip, diagnostic_loss_scale=loss_scale))
        emitted = [(key, 'activation', value) for key, value in targets]
        emitted += [(key, 'parameter', value) for key, value in params if key.startswith('base.scsr_v41.')]
        for key, kind, value in emitted:
            grad = grad_map[key]
            row = dict(batch=batch_id, loss_type=loss_type, component=key, kind=kind, dtype=str(value.dtype),
                       value_rms=tensor_stats(value)['rms'], gradient_available=grad is not None,
                       diagnostic_loss_scale=loss_scale, global_grad_norm=total_norm, hypothetical_clip_factor=clip)
            if grad is not None:
                stats = tensor_stats(grad)
                row.update(grad_rms=stats['rms'], grad_norm=stats['norm'], finite_gradient=stats['finite'],
                           hypothetical_clipped_grad_rms=stats['rms'] * clip if clip is not None else '')
            rows.append(row)
        del cap, targets, params, grads, grad_map, emitted, score, final_cls, loss
    return rows, summaries
