"""Initialization and evaluation probes without sampler/RNG/BN side effects."""
import hashlib
import json
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Do not create CUDA contexts on the other ranks' GPUs merely to save RNG.
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@contextmanager
def evaluation_mode(model):
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, flag in flags:
            module.training = flag


@contextmanager
def diagnostic_session(model, enabled=True, capture=None):
    if any(module.training for module in model.modules()):
        raise RuntimeError('ON/OFF diagnostics require eval() on all modules')
    base = model.base
    module = base.scsr_v41
    if module is None:
        raise ValueError('V4.1 diagnostics require SCSR_V41.ENABLED=True')
    old = base.scsr_runtime_enabled, base.diagnostic_callback, module.diagnostic_callback
    base.scsr_runtime_enabled = enabled
    base.diagnostic_callback = capture.backbone if capture is not None else None
    module.diagnostic_callback = capture.spectral if capture is not None else None
    try:
        yield
    finally:
        base.scsr_runtime_enabled, base.diagnostic_callback, module.diagnostic_callback = old


class Capture:
    def __init__(self, gradients=False):
        self.gradients, self.layers, self.internal = gradients, {}, {}

    def keep(self, value):
        return value if self.gradients else value.detach().cpu().clone()

    def backbone(self, name, tensor):
        self.layers[name] = self.keep(tensor)

    def spectral(self, tensors):
        self.internal = {key: self.keep(value) for key, value in tensors.items()}


def fixed_calibration_batch(cfg, train_loader, rank, world_size):
    """Use dataset indices directly; never iterate/mutate the PKM sampler."""
    dataset = train_loader.dataset
    names = list(cfg.DATASETS.MODALITIES)
    if list(dataset.modalities) != names:
        raise ValueError('Dataset order does not match configured PKM order')
    pools = {}
    for name in names:
        pools[name] = {}
        for index, entry in enumerate(dataset.dataset[name]):
            if entry[3] != names.index(name) + 1:
                raise ValueError('Dataset has inconsistent modality IDs')
            pools[name].setdefault(entry[1], []).append(index)
    common = sorted(set.intersection(*(set(pools[name]) for name in names)))
    k = cfg.DATALOADER.NUM_INSTANCE
    positions = cfg.SOLVER.IMS_PER_BATCH // world_size
    if cfg.SOLVER.IMS_PER_BATCH % world_size or positions % k or positions < 2 * k:
        raise ValueError('Initialization needs the same valid local PKM P/K batch as training')
    p = positions // k
    if len(common) < p:
        raise ValueError('Too few common identities for initialization calibration')
    seed = cfg.SOLVER.SEED + 41001 + rank
    rng = random.Random(seed)
    selected = rng.sample(common, p)
    indices = []
    for pid in selected:
        by_modality = [rng.sample(pools[name][pid], k) if len(pools[name][pid]) >= k
                       else [rng.choice(pools[name][pid]) for _ in range(k)] for name in names]
        indices.extend(zip(*by_modality))
    random.seed(seed)
    np.random.seed(seed)
    # Training transforms run on CPU. Seeding torch.manual_seed here would also
    # modify every visible GPU's generator, although only this rank is active.
    torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
    samples = [dataset[tuple(index)] for index in indices]
    for index in indices:
        if len({dataset.dataset[name][idx][1] for name, idx in zip(names, index)}) != 1:
            raise ValueError('Calibration PKM identity alignment failed')
    images = [torch.stack([sample[0][m] for sample in samples]) for m in range(len(names))]
    records = [[dataset.dataset[name][index[m]][0] for index in indices] for m, name in enumerate(names)]
    digest = hashlib.sha256(json.dumps(records, ensure_ascii=False).encode('utf-8')).hexdigest()
    return images, dict(seed=seed, sample_paths_sha256=digest, identities=selected,
                        positions_per_modality=positions, images=sum(len(group) for group in images), paths=records)


def initial_scale_factor(values):
    """One output-conv multiplier; V and U*V are linear in that weight."""
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        raise RuntimeError('Nonfinite or zero initial V/T RMS')
    if all(0.05 <= value <= 0.10 for value in values.values()):
        return 1.0
    lower = max(0.05 / value for value in values.values())
    upper = min(0.10 / value for value in values.values())
    if lower > upper:
        raise RuntimeError('V/T scales cannot both be calibrated by the allowed output-conv-only adjustment')
    return min(max(0.075 / values['v_rms'], lower), upper)


def calibrate_before_optimizer(cfg, model, train_loader, local_rank, logger):
    module = model.base.scsr_v41
    if module is None:
        return
    if cfg.MODEL.PRETRAIN_CHOICE != 'imagenet':
        raise ValueError('Formal V4.1 training requires the same ImageNet pretrain as UAD')
    if module.calibrated:
        raise RuntimeError('Initialization calibration may run only once, before optimizer construction')
    ddp = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    device = torch.device('cuda', local_rank)
    model.to(device)
    old_collect = module.collect_stats
    try:
        with preserve_rng(), evaluation_mode(model), torch.no_grad():
            images, batch_meta = fixed_calibration_batch(cfg, train_loader, rank, world)
            images = [group.to(device) for group in images]
            probe_groups = [group[:1] for group in images]
            sensor_ids = torch.cat([torch.full((len(group),), m + 1, device=device, dtype=torch.long)
                                    for m, group in enumerate(images)])
            images = torch.cat(images)
            module.collect_stats = True

            def measure():
                with torch.cuda.amp.autocast(enabled=True):
                    model(images, mode=1, modal_ids=sensor_ids)
                local = dict(module.last_stats)
                # All ranks take the same number of collectives, outside forward.
                moment = torch.tensor([local['v_rms'] ** 2, local['t_rms'] ** 2], device=device, dtype=torch.float64)
                if ddp:
                    dist.all_reduce(moment)
                    moment /= world
                return local, dict(v_rms=float(moment[0].sqrt()), t_rms=float(moment[1].sqrt()))

            before_local, before = measure()
            factor = initial_scale_factor(before)
            if factor != 1.0:
                module.output.weight.mul_(factor)
            after_local, after = measure()
            valid = all(0.05 - 1e-6 <= value <= 0.10 + 1e-6 for value in after.values())
            valid = valid and after_local['t_saturation'] < 0.01 and after_local['changed_ratio'] > 0
            flag = torch.tensor(int(valid), device=device)
            if ddp:
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not flag.item():
                raise RuntimeError('V4.1 initialization verification failed; inspect per-rank V/T/gates')
            module.calibration = dict(verified=True, adjusted=factor != 1.0, output_scale_factor=factor,
                                      requested_output_std=cfg.SCSR_V41.CONDITION_OUTPUT_INIT_STD,
                                      effective_output_std=cfg.SCSR_V41.CONDITION_OUTPUT_INIT_STD * factor,
                                      measured_output_std=float(module.output.weight.std(unbiased=False)),
                                      before_global=before, after_global=after,
                                      local_before=before_local, local_after=after_local, batch=batch_meta)
            paired_amp = short_on_off(model, probe_groups, cfg.DATASETS.MODALITIES, use_amp=True)
            paired_fp32 = short_on_off(model, probe_groups, cfg.DATASETS.MODALITIES, use_amp=False)
            module.calibration.update(paired_amp=paired_amp, paired_fp32=paired_fp32)
            digest = hashlib.sha256()
            with open(cfg.MODEL.PRETRAIN_PATH, 'rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            module.calibration.update(pretrain_path=os.path.abspath(cfg.MODEL.PRETRAIN_PATH),
                                      pretrain_sha256=digest.hexdigest(), initialization_seed=cfg.SOLVER.SEED)
            readable = all(np.isfinite(probe[key]) and probe[key] > 0
                           for probe in (paired_amp, paired_fp32)
                           for key in ('block7_attention_output_tir_relative', 'final_cls_tir_relative'))
            flag = torch.tensor(int(readable), device=device)
            if ddp:
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not flag.item():
                raise RuntimeError('V4.1 initial modulation fails to reach attention/final CLS in FP32 or AMP')
            if paired_fp32['final_cls_tir_relative'] < 1e-7:
                logger.warning('V4.1 initial CLS change is near FP32 noise; review the calibration diagnostic before a full run')
            module.calibrated = True
            with open('{}/scsr_v41_init_rank{}.json'.format(cfg.OUTPUT_DIR or '.', rank), 'w', encoding='utf-8') as stream:
                json.dump(module.calibration, stream, indent=2, ensure_ascii=False, allow_nan=False)
            logger.info('SCSR-V4.1 initial calibration rank %d: global=%s, output factor=%.6f, r_attn=%.6e',
                        rank, after, factor, after_local['r_attn_mean'])
    finally:
        module.collect_stats = old_collect


def short_on_off(model, images, modalities, use_amp=True):
    """A sparse eval-only probe on the unwrapped model; no DDP collectives."""
    captures, outputs = {}, {}
    sample = torch.cat([group[:1].detach() for group in images])
    ids = torch.arange(1, len(modalities) + 1, dtype=torch.long, device=sample.device)
    with preserve_rng(), evaluation_mode(model), torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for state in ('ON', 'OFF'):
            cap = Capture()
            with diagnostic_session(model, enabled=state == 'ON', capture=cap):
                outputs[state] = model(sample, mode=1, modal_ids=ids).detach().float()
            captures[state] = cap
    result = {}
    tir = list(modalities).index('Thermal')
    for layer in ('block7_attention_input', 'block7_attention_output', 'final_cls'):
        a, b = captures['ON'].layers[layer].double(), captures['OFF'].layers[layer].double()
        result[layer + '_tir_relative'] = float((a[tir] - b[tir]).norm() / (b[tir].norm() + model.base.scsr_v41.eps))
        for index, name in enumerate(modalities):
            if index != tir and not torch.equal(a[index], b[index]):
                raise RuntimeError('Unexpected ON/OFF change for ' + name)
    return result
