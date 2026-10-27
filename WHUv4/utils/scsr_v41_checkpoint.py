"""Strict version/config checks for new V4.1 checkpoints; UAD stays legacy."""
import json
import os

import torch


def load_torch(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch 1.12
        return torch.load(path, map_location='cpu')


def signature(cfg):
    return dict(version='SCSR-V4.1',
                architecture={key: value for key, value in cfg.SCSR_V41.items()
                              if key not in ('DIAGNOSTIC_PERIOD', 'EARLY_UPDATES', 'PREFLIGHT_UPDATES')},
                modalities=list(cfg.DATASETS.MODALITIES),
                backbone=cfg.MODEL.TRANSFORMER_TYPE,
                input_size=list(cfg.INPUT.SIZE_TRAIN), stride=list(cfg.MODEL.STRIDE_SIZE))


def save_metadata(model, path, epoch, updates, skipped):
    if model.base.scsr_v41 is None:
        return
    payload = signature(model.cfg)
    payload.update(epoch=epoch, optimizer_updates=updates, amp_skipped=skipped,
                   calibration=model.base.scsr_v41.calibration, config_yaml=model.cfg.dump())
    with open(str(path) + '.json', 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_v41(model, path, include_classifier=False):
    sidecar = str(path) + '.json'
    if not os.path.isfile(sidecar):
        raise ValueError('V4.1 checkpoint requires its matching .pth.json version/config sidecar: ' + sidecar)
    with open(sidecar, encoding='utf-8') as stream:
        saved = json.load(stream)
    expected = signature(model.cfg)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError('Checkpoint {} mismatch: saved={!r}, current={!r}'.format(key, saved.get(key), value))
    calibration = saved.get('calibration')
    if not calibration or not calibration.get('verified'):
        raise ValueError('Checkpoint has no verified pre-optimizer V4.1 calibration')
    raw = load_torch(path)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in raw.items()}
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise ValueError('Incomplete or wrong-version V4.1 state_dict; missing={}, extra={}'.format(
            sorted(set(expected_state) - set(state)), sorted(set(state) - set(expected_state))))
    if not include_classifier:
        # Keep original UAD's evaluation classifier behavior; require its keys.
        state['classifier.weight'] = expected_state['classifier.weight']
    model.load_state_dict(state, strict=True)
    model.base.scsr_v41.calibration = calibration
    model.base.scsr_v41.calibrated = True
    print('Loaded verified SCSR-V4.1 checkpoint: {}'.format(path))
