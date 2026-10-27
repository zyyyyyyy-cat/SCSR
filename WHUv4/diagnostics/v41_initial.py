"""Recover calibrated initial controls without asserting unproven provenance."""
import hashlib
import json
from pathlib import Path

import torch

from model import make_model
from utils.scsr_v41_checkpoint import load_torch
from utils.scsr_v41_runtime import preserve_rng
from .core import difference_stats, tensor_stats


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_init_records(directory, fallback):
    files = sorted(Path(directory).glob('scsr_v41_init_rank*.json')) if directory else []
    records = [json.loads(path.read_text(encoding='utf-8')) for path in files]
    if not records:
        records = [fallback]
    for record in records:
        if not record.get('verified') or not record.get('output_scale_factor', 0) > 0:
            raise ValueError('Missing verified calibration and full-precision output_scale_factor')
        for key in ('initialization_seed', 'output_scale_factor', 'measured_output_std', 'pretrain_sha256'):
            if key in record and key in fallback and record[key] != fallback[key]:
                raise ValueError('Initialization record disagrees with checkpoint calibration: ' + key)
    return records, [dict(path=str(path), sha256=file_hash(path)) for path in files]


def initial_norm_reference(path):
    if not path or not Path(path).is_file():
        return None
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            params = row.get('parameters', {})
            if (row.get('optimizer_updates') == 0 and params and
                    all(value.get('from_init_abs_norm') == 0 for value in params.values())):
                return {key: value['norm'] for key, value in params.items()}
    return None


def reconstruct_initial(config, classes, camera_num, view_num, seed, factor):
    local = config.clone()
    local.defrost()
    local.MODEL.PRETRAIN_CHOICE = 'none'
    local.freeze()
    with preserve_rng():
        # Use the MODEL seed, not calibration Batch seed (42235 etc.).
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        # ImageNet load in this version only copies tensors / resizes pos_embed;
        # it consumes no RNG and does not touch SCSR. No pretrain file is needed.
        model = make_model(local, classes, camera_num, view_num)
        state = {key: value.detach().cpu().clone() for key, value in model.base.scsr_v41.state_dict().items()}
    state['output.weight'].mul_(factor)
    return state


def restore_initial_control(config, model, checkpoint_metadata, records_dir=None,
                            diagnostics_path=None, snapshot=None):
    manifest = dict(status='unavailable', exact_original_initialization=False,
                    note='No initial tensor hash was recorded by the original training code')
    try:
        calibration = checkpoint_metadata['calibration']
        records, sources = read_init_records(records_dir, calibration)
        seed = calibration.get('initialization_seed')
        if seed is None:
            raise ValueError('No recorded model initialization_seed; refusing to substitute calibration Batch seed')
        norms = initial_norm_reference(diagnostics_path)
        expected = model.base.scsr_v41.state_dict()
        if snapshot:
            raw = load_torch(snapshot)
            supplied_provenance = raw.get('provenance', {})
            raw = raw.get('state_dict', raw)
            raw = {key[7:] if key.startswith('module.') else key: value for key, value in raw.items()}
            if set(raw) == set(expected):
                state = raw
            else:
                state = {key[len('base.scsr_v41.'):]: value for key, value in raw.items()
                         if key.startswith('base.scsr_v41.')}
            manifest.update(control_type='provided_initial_snapshot', snapshot_sha256=file_hash(snapshot))
            if 'reconstructed' in supplied_provenance.get('control_type', ''):
                manifest['control_type'] = 'reused_reconstructed_initial_control'
                manifest['supplied_provenance'] = supplied_provenance
        else:
            state = reconstruct_initial(config, model.num_classes, 0, 0, int(seed), calibration['output_scale_factor'])
            manifest['control_type'] = 'reconstructed_initial_control'
        if set(state) != set(expected):
            raise ValueError('Initial state must contain all seven SCSR parameter tensors')
        state = {key: value.detach().cpu().clone() for key, value in state.items()}
        for key, value in state.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not value.isfinite().all():
                raise ValueError('Initial tensor shape/dtype/finite mismatch: ' + key)
        comparisons = []
        if norms:
            if set(norms) != set(state):
                raise ValueError('Zero-update norm reference is incomplete')
            for key, value in state.items():
                actual, reference = float(value.float().norm()), float(norms[key])
                matches = abs(actual - reference) <= 3e-7 + 3e-6 * abs(reference)
                comparisons.append(dict(parameter=key, reconstructed_norm=actual, reference_norm=reference, matches=matches))
                if not matches:
                    raise ValueError('Initial parameter norm check failed: ' + key)
        std = float(state['output.weight'].float().std(unbiased=False))
        target_std = calibration.get('measured_output_std')
        if target_std is None or abs(std - target_std) > 3e-7 + 3e-6 * abs(target_std):
            raise ValueError('Calibrated initial output weight std does not match the recorded statistic')
        digest = tensor_state_hash(state)
        expected_digest = calibration.get('initial_state_sha256')
        if expected_digest is not None and digest != expected_digest:
            raise ValueError('Recorded initial tensor SHA256 mismatch')
        manifest.update(status='ready', model_seed=int(seed), output_scale_factor=calibration['output_scale_factor'],
                        measured_output_std=std, recorded_output_std=target_std,
                        exact_original_initialization=expected_digest is not None,
                        norm_reference_available=norms is not None, norm_checks=comparisons,
                        initial_tensor_sha256=digest, initialization_record_sources=sources,
                        zero_update_reference_path=str(diagnostics_path) if norms else None,
                        zero_update_reference_sha256=file_hash(diagnostics_path) if norms else None,
                        statistical_checks_do_not_prove_elementwise_identity=True,
                        reconstructed_pretrain_load_skipped_because_it_consumes_no_rng=not bool(snapshot))
        return state, manifest
    except (ValueError, KeyError, RuntimeError, TypeError, OSError) as error:
        manifest['reason'] = str(error)
        return None, manifest


def parameter_change_rows(initial, trained, provenance):
    rows = []
    for key in sorted(initial):
        a, b = trained[key].detach().cpu(), initial[key]
        delta = a.double() - b.double()
        row = dict(parameter=key, numel=b.numel(), control_type=provenance['control_type'],
                   exact_original_initialization=provenance['exact_original_initialization'],
                   mean_abs_difference=float(delta.abs().mean()), max_abs_difference=float(delta.abs().max()),
                   delta_norm=float(delta.norm()), unchanged_fraction=float((a == b).double().mean()))
        row.update(difference_stats(a, b))
        row.update({'initial_' + k: v for k, v in tensor_stats(b).items()})
        row.update({'trained_' + k: v for k, v in tensor_stats(a).items()})
        rows.append(row)
    return rows
