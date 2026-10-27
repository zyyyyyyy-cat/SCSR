"""Sparse statistics of unscaled gradients and actual SGD/AMP updates."""
import json
import math
import os

import torch

from .scsr_v41_runtime import short_on_off


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class TrainingMonitor:
    def __init__(self, cfg, model, logger):
        self.cfg, self.model, self.logger = cfg, model, logger
        self.module = model.base.scsr_v41
        self.params = dict(self.module.named_parameters())
        self.initial = {key: value.detach().clone() for key, value in self.params.items()}
        self.before, self.grad_pre, self.grad_post = {}, {}, {}
        self.path = os.path.join(cfg.OUTPUT_DIR or '.', 'scsr_v41_training_diagnostics.jsonl')

    def selected(self, attempt, updates):
        period = self.cfg.SCSR_V41.DIAGNOSTIC_PERIOD
        if period < 1:
            raise ValueError('DIAGNOSTIC_PERIOD must be positive')
        return attempt <= 3 or attempt % period == 0 or updates == self.cfg.SCSR_V41.EARLY_UPDATES - 1

    def snapshot(self):
        self.before = {key: value.detach().clone() for key, value in self.params.items()}
        self.forward_stats = dict(self.module.last_stats)

    def gradients(self, post_clip=False):
        stats = {}
        for key, param in self.params.items():
            grad = param.grad
            stats[key] = None if grad is None else dict(
                norm=float(grad.detach().float().norm()), rms=float(grad.detach().float().square().mean().sqrt()),
                finite=bool(grad.isfinite().all()))
        if post_clip:
            self.grad_post = stats
        else:
            self.grad_pre = stats

    def record(self, epoch, iteration, attempts, updates, skipped, did_step, scale_before, scale_after, images, loss_values=None):
        row = dict(epoch=epoch, iteration=iteration, attempted_steps=attempts, optimizer_updates=updates,
                   amp_skipped=skipped, did_step=did_step, scale_before=scale_before,
                   scale_after=scale_after, forward=self.forward_stats, losses=loss_values, parameters={})
        device = next(self.model.parameters()).device
        if device.type == 'cuda':
            row['cuda_memory'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        for key, param in self.params.items():
            value = param.detach().float()
            old, initial = self.before[key].float(), self.initial[key].float()
            row['parameters'][key] = dict(grad_pre_clip=self.grad_pre[key], grad_post_clip=self.grad_post[key],
                                          shape=list(param.shape), parameter_stride=list(param.stride()),
                                          gradient_stride=list(param.grad.stride()) if param.grad is not None else None,
                                          norm=float(value.norm()),
                                          step_abs_norm=float((value - old).norm()),
                                          changed_fraction=float((value != old).float().mean()),
                                          from_init_abs_norm=float((value - initial).norm()),
                                          step_relative=float((value - old).norm() / (old.norm() + self.module.eps)),
                                          from_init_relative=float((value - initial).norm() / (initial.norm() + self.module.eps)))
        if updates <= self.cfg.SCSR_V41.EARLY_UPDATES:
            row['paired_amp'] = short_on_off(self.model, images, self.cfg.DATASETS.MODALITIES, use_amp=True)
            row['paired_fp32'] = short_on_off(self.model, images, self.cfg.DATASETS.MODALITIES, use_amp=False)
        with open(self.path, 'a', encoding='utf-8') as stream:
            # Record overflow/NaN evidence without crashing the diagnostic writer.
            stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + '\n')
        self.logger.info('SCSR-V4.1 updates=%d skipped=%d V_RMS=%.5f T_RMS=%.5f r_attn=%.3e U_step=%.3e',
                         updates, skipped, self.forward_stats['v_rms'], self.forward_stats['t_rms'],
                         self.forward_stats['r_attn_mean'], row['parameters']['U']['step_relative'])
