"""Four fixed-checkpoint SCSR diagnostics. No optimizer, DDP or training run."""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from config import cfg
from model import make_model
from diagnostics.core import (Capture, amp_context, diagnostic_session, gradient_probe,
                              grouped_means, internal_rows, layerwise_rows,
                              load_full_checkpoint, state_digest, write_csv)
from diagnostics.data import (evaluation_entries, image_loader, load_dataset,
                              load_gradient_images, metadata, order_digest,
                              select_gradient_batches, select_queries)
from diagnostics.retrieval import paired_features, evaluate_on_off


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def json_safe(value):
    # A diagnostic must record a nonfinite gradient, not lose its manifest while
    # trying to serialize NaN/Inf. CSV finite flags carry the same information.
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    return value


def run(args, config):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run diagnose_scsr.py with python on one GPU, not torchrun')
    if not config.SCSR.ENABLED or config.SCSR.TARGET_MODALITY != 'Thermal':
        raise ValueError('The requested diagnostics require SCSR enabled for Thermal')
    if config.MODEL.SIE_CAMERA or config.MODEL.SIE_VIEW:
        raise ValueError('This diagnostic follows the supplied UAD config with SIE disabled')
    if config.SCSR.INSERT_AFTER_BLOCK != 6 or config.SCSR.IN_DIM != 768:
        raise ValueError('The requested L6/CLS diagnostics require the frozen V3 768D design')
    if list(config.INPUT.SIZE_TRAIN) != [256, 128] or list(config.INPUT.SIZE_TEST) != [256, 128]:
        raise ValueError('The requested diagnostics require the original 256x128 preprocessing')
    if (config.TEST.NECK_FEAT != 'before' or config.TEST.FEAT_NORM != 'yes'
            or config.TEST.RE_RANKING or config.TEST.TOP_K_EVAL != 0):
        raise ValueError('Use pre-BN, L2 normalization, no reranking and TOP_K_EVAL=0')
    if config.MODEL.METRIC_LOSS_TYPE != 'wrt' or config.MODEL.IF_LABELSMOOTH != 'off':
        raise ValueError('The gradient probe implements this experiment\'s CE(no smoothing)+WRT')
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or not Path(str(checkpoint) + '.yml').is_file():
        raise ValueError('Provide the SCSR checkpoint together with its .pth.yml configuration')
    epoch_match = re.search(r'_(\d+)\.pth$', checkpoint.name)
    checkpoint_epoch = int(epoch_match.group(1)) if epoch_match else None
    if checkpoint_epoch is not None and checkpoint_epoch != 120:
        raise ValueError('The supplied guidance requires the epoch-120 checkpoint')
    if args.uad_checkpoint and not Path(args.uad_checkpoint).is_file():
        raise ValueError('The optional UAD provenance checkpoint path does not exist')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in (
            'diagnostic_manifest.json', 'scsr_on_off_metrics.csv', 'scsr_on_off_query_comparison.csv',
            'scsr_internal_activation.csv', 'scsr_layerwise_feature_effect.csv', 'scsr_gradient_probe.csv')):
        raise ValueError('Use a fresh output directory so separate diagnostic runs do not mix')
    if min(args.batch_size, args.distance_chunk, args.probe_samples_per_modality) < 1:
        raise ValueError('Batch, distance chunk and sample counts must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu only for local checks')
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires CUDA')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint),
                    checkpoint_epoch_from_filename=checkpoint_epoch,
                    checkpoint_epoch_is_model_metadata=False,
                    checkpoint_config=str(checkpoint) + '.yml',
                    checkpoint_config_sha256=file_digest(str(checkpoint) + '.yml'),
                    uad_checkpoint=str(Path(args.uad_checkpoint).resolve()) if args.uad_checkpoint else None,
                    uad_checkpoint_used_for_on_off=False,
                    config=yaml.safe_load(config.dump()), args=vars(args),
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                     cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                     device=str(device), gpu=torch.cuda.get_device_name(device)
                                     if device.type == 'cuda' else None),
                    protocol=dict(feature='768D pre-BN CLS', normalization='L2',
                                  exclude='ALL gallery images from the query camera',
                                  reranking=False, TTA=False, top_k=0,
                                  sorting='torch.argsort, CPU FP32, existing Euclidean distance',
                                  distance_chunk=args.distance_chunk, amp=args.amp,
                                  platform='zero-based cameras 0..4=G; 5..6=A'),
                    gradient_semantics='eval-state CE and WRT separately, no optimizer or prototype update',
                    gpd_probe_skipped='PrototypeBank is not included in standard UAD model state_dict',
                    parameters_modified=False, bn_buffers_modified=False)
    manifest_path = output / 'diagnostic_manifest.json'

    def save_manifest():
        with open(manifest_path, 'w', encoding='utf-8') as stream:
            json.dump(json_safe(manifest), stream, ensure_ascii=False, indent=2, allow_nan=False)

    def progress(message):
        print(message, flush=True)

    save_manifest()
    model, before = None, None
    try:
        dataset = load_dataset(config)
        modalities = list(config.DATASETS.MODALITIES)
        queries, galleries = evaluation_entries(dataset, modalities)
        qrecords, grecords = ([metadata(entry, modalities) for entry in records]
                              for records in (queries, galleries))
        selected = select_queries(dataset, modalities, args.probe_samples_per_modality, args.seed)
        records = [metadata(entry, modalities) for entry in selected]
        fixed_batches = select_gradient_batches(dataset, modalities, args.gradient_batches,
                                                args.gradient_identities,
                                                config.DATALOADER.NUM_INSTANCE, args.seed)
        manifest['data'] = dict(query_count=len(queries), gallery_count=len(galleries),
                                query_order_sha256=order_digest(qrecords),
                                gallery_order_sha256=order_digest(grecords),
                                train_identity_count=dataset.num_train_pids,
                                probe_samples=records,
                                gradient_batches=[dict(batch=i + 1, targets=labels,
                                    samples=[[metadata(entry, modalities) for entry in group] for group in groups])
                                    for i, (groups, labels) in enumerate(fixed_batches)])
        model = make_model(config, dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids)
        if len(model.base.blocks) != 12:
            raise ValueError('Layerwise diagnostics require the original 12-block ViT')
        load_full_checkpoint(model, checkpoint)
        model.to(device).eval()
        before = state_digest(model)
        manifest['model_state_sha256_before'] = before
        manifest['parameter_count'] = sum(p.numel() for p in model.parameters())
        manifest['scsr_parameter_count'] = sum(p.numel() for p in model.base.scsr.parameters())
        manifest['classifier_restored'] = True
        save_manifest()

        if '1' in args.experiments:
            progress('Experiment 1: paired ON/OFF full query and gallery extraction')
            qf = paired_features(model, image_loader(queries, config, args.batch_size, args.workers),
                                 qrecords, device, args.amp, progress)
            gf = paired_features(model, image_loader(galleries, config, args.batch_size, args.workers),
                                 grecords, device, args.amp, progress)
            metrics, comparison, summary = evaluate_on_off(
                qf, gf, qrecords, grecords, args.distance_chunk, progress)
            write_csv(output / 'scsr_on_off_metrics.csv', metrics)
            write_csv(output / 'scsr_on_off_query_comparison.csv', comparison)
            manifest['on_off_summary'] = summary
            del qf, gf, metrics, comparison
            save_manifest()

        if '2' in args.experiments or '3' in args.experiments:
            progress('Experiments 2/3: bounded capture on fixed representative queries')
            activations, effects = [], []
            loader = image_loader(selected, config, args.batch_size, args.workers)
            for images, indices in loader:
                batch_records = [records[int(i)] for i in indices]
                mods = torch.tensor([row['modal_id'] for row in batch_records], dtype=torch.long, device=device)
                images = images.to(device)
                on, off = Capture(), Capture()
                with torch.no_grad(), amp_context(device, args.amp):
                    with diagnostic_session(model, True, on):
                        model(images, mode=1, modal_ids=mods)
                    if '3' in args.experiments:
                        with diagnostic_session(model, False, off):
                            model(images, mode=1, modal_ids=mods)
                if '2' in args.experiments:
                    tir = [row for row in batch_records if row['modal_id'] == model.base.scsr_target_id]
                    activations.extend(internal_rows(on, tir, config.SCSR.EPS))
                if '3' in args.experiments:
                    effects.extend(layerwise_rows(on, off, batch_records, config.SCSR.EPS))
            if '2' in args.experiments:
                write_csv(output / 'scsr_internal_activation.csv', activations)
                manifest['internal_group_summary'] = grouped_means(
                    activations, ('r_generated', 'r_cast', 'r_effective', 'changed_ratio'))
            if '3' in args.experiments:
                write_csv(output / 'scsr_layerwise_feature_effect.csv', effects)
                manifest['layer_group_summary'] = grouped_means(
                    effects, ('abs_diff', 'rel_diff', 'cos_sim', 'changed_ratio'), ('layer',))
            del activations, effects, on, off
            save_manifest()

        if '4' in args.experiments:
            progress('Experiment 4: fixed eval-state CE/WRT gradients, no parameter update')
            gradient_rows, losses = [], []
            loss_scale = args.gradient_loss_scale
            if loss_scale is None:
                loss_scale = 4096.0 if args.amp else 1.0
            for i, (groups, labels) in enumerate(fixed_batches):
                images = load_gradient_images(groups, config, device)
                target = torch.tensor(labels, dtype=torch.long, device=device)
                rows, loss_rows = gradient_probe(model, images, target, i + 1, device,
                                                 args.amp, loss_scale)
                gradient_rows.extend(rows)
                losses.extend(loss_rows)
            write_csv(output / 'scsr_gradient_probe.csv', gradient_rows)
            manifest['gradient_loss_summary'] = losses
            manifest['gradient_available_rows'] = sum(row['gradient_available'] for row in gradient_rows)
            manifest['gradient_nonfinite_rows'] = sum(row.get('finite_gradient') is False for row in gradient_rows)
            save_manifest()

        manifest['status'] = ('complete_with_nonfinite_gradient'
                              if manifest.get('gradient_nonfinite_rows', 0) else 'complete')
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        if model is not None and before is not None:
            after = state_digest(model)
            manifest['model_state_sha256_after'] = after
            manifest['state_unchanged'] = before == after
            if before != after:
                manifest['status'] = 'failed_model_state_changed'
                manifest['parameters_modified'] = None
                manifest['bn_buffers_modified'] = None
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    if not manifest.get('state_unchanged', False):
        raise RuntimeError('Diagnostic changed parameters or persistent buffers')
    progress('Completed; model parameters/BN buffers unchanged. Output: ' + str(output))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/SCSR.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--uad-checkpoint', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiments', default='1,2,3,4', help='Subset of 1,2,3,4')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--distance-chunk', type=int, default=64)
    parser.add_argument('--probe-samples-per-modality', type=int, default=32)
    parser.add_argument('--gradient-batches', type=int, default=2)
    parser.add_argument('--gradient-identities', type=int, default=4)
    parser.add_argument('--gradient-loss-scale', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--amp', action='store_true', help='Match training AMP arithmetic; default matches original test.py FP32')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.experiments = [value.strip() for value in args.experiments.split(',')]
    if not args.experiments or any(value not in ('1', '2', '3', '4') for value in args.experiments):
        parser.error('--experiments must be a subset of 1,2,3,4')
    config = cfg.clone()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    # Diagnostics load a full trained checkpoint, not a second ImageNet model.
    config.MODEL.PRETRAIN_CHOICE = 'none'
    config.MODEL.DIST_TRAIN = False
    config.freeze()
    run(args, config)


if __name__ == '__main__':
    main()
