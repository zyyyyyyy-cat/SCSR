"""Bounded feature capture and gradient probes, compatible with Python 3.8."""
import csv
import hashlib
import math
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from loss.wrt_loss import TripletLoss_WRT


@contextmanager
def diagnostic_session(model, enabled=True, capture=None):
    """Restore runtime switches even on error; never remove/rebuild modules."""
    if any(module.training for module in model.modules()):
        raise RuntimeError('Diagnostics require model.eval(), including BN/dropout')
    base = model.base
    if base.scsr is None:
        raise ValueError('Diagnostics require a loaded SCSR model')
    old = (base.scsr_runtime_enabled, base.diagnostic_callback,
           base.scsr.diagnostic_callback)
    base.scsr_runtime_enabled = enabled
    base.diagnostic_callback = capture.backbone if capture is not None else None
    base.scsr.diagnostic_callback = capture.spectral if capture is not None else None
    try:
        yield
    finally:
        (base.scsr_runtime_enabled, base.diagnostic_callback,
         base.scsr.diagnostic_callback) = old


def amp_context(device, enabled):
    if enabled and device.type != 'cuda':
        raise ValueError('--amp probes require CUDA; original CPU inference is FP32')
    return torch.cuda.amp.autocast(enabled=True) if enabled else nullcontext()


class Capture:
    def __init__(self, gradients=False):
        self.gradients = gradients
        self.layers = {}
        self.internal = {}

    def _keep(self, tensor):
        return tensor if self.gradients else tensor.detach().cpu().clone()

    def backbone(self, name, tensor):
        self.layers[name] = self._keep(tensor)

    def spectral(self, tensors):
        self.internal = {key: self._keep(value) for key, value in tensors.items()}


def write_csv(path, rows, fieldnames=None):
    # Scientific notation preserves tiny residual/feature differences.
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(float(value), '.12e')
                             if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})


def tensor_stats(tensor):
    x = tensor.detach().double().flatten()
    return {'rms': float(x.square().mean().sqrt()), 'norm': float(x.norm()),
            'max_abs': float(x.abs().max()), 'finite': bool(x.isfinite().all())}


def difference_stats(on, off, eps=1e-6):
    # Subtract in FP64 AFTER preserving the actual output values, to measure
    # representable differences accurately without changing model arithmetic.
    a, b = on.detach().double().flatten(), off.detach().double().flatten()
    delta = a - b
    absolute = float(delta.norm())
    norms = float(a.norm() * b.norm())
    return {'abs_diff': absolute, 'rel_diff': absolute / (float(b.norm()) + eps),
            'cos_sim': float((a * b).sum()) / norms if norms > 0 else '',
            'changed_ratio': float((a != b).double().mean()),
            'equal_ratio': float((a == b).double().mean())}


def internal_rows(capture, tir_records, eps):
    tensors = capture.internal
    if not tensors:
        return []
    if tensors['p6_tir'].shape[0] != len(tir_records):
        raise ValueError('TIR capture/metadata count mismatch')
    rows = []
    for i, record in enumerate(tir_records):
        row = dict(record)
        for name, tensor in tensors.items():
            row[name + '_dtype'] = str(tensor.dtype)
            for key, value in tensor_stats(tensor[i]).items():
                row[name + '_' + key] = value
        denominator = tensor_stats(tensors['p6_tir'][i])['norm'] + eps
        actual = (tensors['updated'][i].double() - tensors['p6_tir'][i].double())
        row['r_generated'] = row['residual_fp32_norm'] / denominator
        row['r_cast'] = row['residual_cast_norm'] / denominator
        row['r_effective'] = float(actual.norm()) / denominator
        row['actual_change_mean_abs'] = float(actual.abs().mean())
        row['actual_change_max_abs'] = float(actual.abs().max())
        row['changed_ratio'] = float((actual != 0).double().mean())
        row['cast_zero_ratio'] = float((tensors['residual_cast'][i] == 0).double().mean())
        for percentile in (0.5, 0.9, 0.99):
            # NumPy percentile avoids a dependency on newer torch.quantile APIs.
            row['actual_change_abs_p{}'.format(int(percentile * 100))] = float(
                np.percentile(actual.abs().cpu().numpy(), percentile * 100))
        for direction, gate in zip(('lh', 'hl', 'hh'), tensors['gate_increment'][i].chunk(3, dim=0)):
            row[direction + '_gate_abs_mean'] = float(gate.double().abs().mean())
            row[direction + '_gate_abs_max'] = float(gate.double().abs().max())
        rows.append(row)
    return rows


def layerwise_rows(on, off, records, eps):
    if set(on.layers) != set(off.layers):
        raise ValueError('ON/OFF captured different layers')
    rows = []
    comparisons = []
    for name in ('block_6', 'post_scsr', 'block_7', 'block_9', 'block_12'):
        if name not in on.layers:
            continue
        comparisons.append((name + '_cls', on.layers[name][:, 0], off.layers[name][:, 0]))
        if name in ('block_6', 'post_scsr'):
            comparisons.append((name + '_patch', on.layers[name][:, 1:], off.layers[name][:, 1:]))
    comparisons.append(('final_pre_bn_cls', on.layers['final_cls'], off.layers['final_cls']))
    comparisons.append(('retrieval_l2_cls', F.normalize(on.layers['final_cls'].float(), dim=1),
                        F.normalize(off.layers['final_cls'].float(), dim=1)))
    for layer, a, b in comparisons:
        if a.shape[0] != len(records):
            raise ValueError('Layer capture/metadata count mismatch')
        for i, record in enumerate(records):
            row = dict(record, layer=layer, dtype_on=str(a.dtype), dtype_off=str(b.dtype))
            row.update(difference_stats(a[i], b[i], eps))
            rows.append(row)
    return rows


def grouped_means(rows, fields, extra_keys=()):
    groups = {}
    for row in rows:
        for label in (row['modality_name'], row['modality_name'] + '-' + row['platform']):
            key = (label,) + tuple(row[k] for k in extra_keys)
            groups.setdefault(key, []).append(row)
    result = []
    for key, samples in sorted(groups.items()):
        row = dict(group=key[0], sample_count=len(samples))
        row.update(zip(extra_keys, key[1:]))
        for field in fields:
            values = [float(sample[field]) for sample in samples if sample.get(field, '') != '']
            row[field + '_mean'] = float(np.mean(values)) if values else None
        result.append(row)
    return result


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_full_checkpoint(model, path):
    model._check_checkpoint_config(str(path))
    try:
        state = torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    model._check_scsr_weights(state)
    # Unlike ordinary retrieval load_param, gradient diagnostics MUST restore classifier.
    model.load_state_dict(state, strict=True)


def gradient_probe(model, image_batches, target, batch_id, device, use_amp=False,
                   loss_scale=1.0):
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('Diagnostic gradient loss scale must be finite and positive')
    rows, loss_summary = [], []
    for loss_type in ('CE', 'WRT'):
        capture = Capture(gradients=True)
        with diagnostic_session(model, True, capture), torch.enable_grad(), amp_context(device, use_amp):
            score, final_cls, _ = model(image_batches, mode=0)
            labels = target.repeat(len(image_batches))
            loss = (F.cross_entropy(score, labels) if loss_type == 'CE'
                    else TripletLoss_WRT()(final_cls, labels)[0])
        targets = [('final_cls', final_cls), ('block6_tokens', capture.layers['block_6'])]
        targets += [(name, capture.internal[name]) for name in
                    ('p6_tir', 'projected', 'q', 'up_delta', 'residual_fp32', 'residual_cast', 'updated')]
        parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss * loss_scale,
                                    [value for _, value in targets + parameters], allow_unused=True)
        grad_map = {name: None if grad is None else grad.detach().float() / loss_scale
                    for (name, _), grad in zip(targets + parameters, grads)}
        param_norms = [float(grad_map[name].double().norm()) for name, _ in parameters
                       if grad_map[name] is not None]
        global_norm = math.sqrt(sum(value * value for value in param_norms))
        clip_factor = min(1.0, 1.0 / (global_norm + 1e-6)) if math.isfinite(global_norm) else None
        loss_summary.append(dict(batch=batch_id, loss_type=loss_type, loss=float(loss.detach()),
                                 diagnostic_loss_scale=loss_scale,
                                 global_grad_norm=global_norm, hypothetical_clip_factor=clip_factor))
        emitted = [('final_cls', 'activation', final_cls, grad_map['final_cls']),
                   ('block6_patch_all', 'activation', capture.layers['block_6'][:, 1:],
                    None if grad_map['block6_tokens'] is None else grad_map['block6_tokens'][:, 1:])]
        emitted += [(name, 'activation', value, grad_map[name]) for name, value in targets[2:]]
        emitted += [(name, 'parameter', p, grad_map[name]) for name, p in parameters
                    if name.startswith('base.scsr.')]
        for component, kind, value, grad in emitted:
            row = dict(batch=batch_id, loss_type=loss_type, component=component,
                       param_or_activation=kind, dtype=str(value.dtype),
                       value_rms=tensor_stats(value)['rms'],
                       finite_value=tensor_stats(value)['finite'],
                       gradient_available=grad is not None, diagnostic_loss_scale=loss_scale,
                       global_grad_norm=global_norm, hypothetical_clip_factor=clip_factor)
            if grad is not None:
                stats = tensor_stats(grad)
                row.update(grad_rms=stats['rms'], grad_norm=stats['norm'], finite_gradient=stats['finite'],
                           hypothetical_clipped_grad_rms=stats['rms'] * clip_factor
                           if clip_factor is not None else '')
            else:
                row.update(grad_rms='', grad_norm='', finite_gradient='', hypothetical_clipped_grad_rms='')
            rows.append(row)
        # autograd.grad does not populate parameter .grad or update any parameter.
        del capture, targets, grads, grad_map, parameters, emitted, score, final_cls, loss
    return rows, loss_summary
