"""Final-weight local CE/WRT/(optional)ProCA sensitivity; no optimizer/bank."""
import math

import torch
import torch.nn.functional as F

from loss.wrt_loss import TripletLoss_WRT
from processor.processor import PCA
from .core import Capture, amp_context, diagnostic_session, tensor_stats


def final_gradient_probe(model, images, target, batch_id, device, use_amp=False,
                         loss_scale=1.0, include_proca=False):
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('Gradient loss scale must be finite and positive')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Gradient probe must preserve BN and dropout eval state')
    all_params = [(key, p) for key, p in model.named_parameters() if p.requires_grad]
    if not all_params:
        raise ValueError('Temporarily restore original requires_grad flags before the optional probe')
    rows, summaries, parameter_gradients = [], [], {}
    kinds = ['CE', 'WRT'] + (['ProCA'] if include_proca else [])
    for loss_kind in kinds:
        cap = Capture(gradients=True)
        with torch.enable_grad(), diagnostic_session(model, True, cap), amp_context(device, use_amp):
            score, cls, _ = model(images, label=target, mode=0)
            labels = target.repeat(len(images))
            if loss_kind == 'CE':
                loss = F.cross_entropy(score, labels)
            elif loss_kind == 'WRT':
                loss = TripletLoss_WRT()(cls, labels)[0]
            else:
                loss = PCA(cls, target, len(images), group_size=model.cfg.DATALOADER.NUM_INSTANCE)[0]
        activations = [('block7_attention_input', cap.layers['block7_attention_input']), ('final_cls', cls)]
        grads = torch.autograd.grad(loss * loss_scale, [p for _, p in all_params] + [p for _, p in activations], allow_unused=True)
        normalized = [None if g is None else g.detach().float() / loss_scale for g in grads]
        total_norm = math.sqrt(sum(float(g.double().norm()) ** 2 for g in normalized[:len(all_params)] if g is not None))
        clip = min(1.0, 1.0 / (total_norm + 1e-6)) if math.isfinite(total_norm) else None
        summaries.append(dict(batch=batch_id, loss_type=loss_kind, loss=float(loss.detach()),
                              global_grad_norm=total_norm, hypothetical_clip_factor=clip,
                              gpd_included=False, bn_updated=False, optimizer_step=False))
        for (key, value), grad in zip(all_params + activations, normalized):
            if key not in dict(activations) and not key.startswith('base.scsr_v41.'):
                continue
            row = dict(batch=batch_id, loss_type=loss_kind, component=key,
                       kind='activation' if key in dict(activations) else 'parameter',
                       gradient_available=grad is not None, diagnostic_loss_scale=loss_scale,
                       global_grad_norm=total_norm, hypothetical_clip_factor=clip,
                       clipping_applies=key not in dict(activations),
                       clip_semantics='single-loss unweighted full-model parameter norm, max_norm=1; activation gradients are not clipped',
                       training_loss_weight={'CE': model.cfg.MODEL.ID_LOSS_WEIGHT,
                                             'WRT': model.cfg.MODEL.TRIPLET_LOSS_WEIGHT,
                                             'ProCA': model.cfg.MODEL.PCA_LOSS_WEIGHT}[loss_kind],
                       gradient_cosine_CE_WRT='', value_rms=tensor_stats(value)['rms'])
            if grad is None:
                row.update(grad_norm='', grad_rms='', finite_gradient='', post_clip_grad_norm='', post_clip_grad_rms='')
            else:
                stats = tensor_stats(grad)
                factor = clip if row['clipping_applies'] else 1.0
                row.update(grad_norm=stats['norm'], grad_rms=stats['rms'], finite_gradient=stats['finite'],
                           post_clip_grad_norm=stats['norm'] * factor if factor is not None else '',
                           post_clip_grad_rms=stats['rms'] * factor if factor is not None else '')
                if row['kind'] == 'parameter' and loss_kind in ('CE', 'WRT'):
                    parameter_gradients[(loss_kind, key)] = grad.detach().double().cpu().flatten()
            rows.append(row)
        del cap, score, cls, loss, grads, normalized, activations
    for row in rows:
        key = row['component']
        a, b = parameter_gradients.get(('CE', key)), parameter_gradients.get(('WRT', key))
        if a is not None and b is not None and a.isfinite().all() and b.isfinite().all():
            denom = float(a.norm() * b.norm())
            row['gradient_cosine_CE_WRT'] = float(a @ b) / denom if denom > 0 else ''
    return rows, summaries
