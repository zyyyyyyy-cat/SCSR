"""One command: fixed-checkpoint V4.1 activations, gradients and full rankings."""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config import cfg
from model import make_model
from utils.scsr_v41_checkpoint import load_v41
from diagnostics.core import (Capture, amp_context, diagnostic_session, gradient_probe,
                              internal_rows, json_safe, layerwise_rows, state_digest, write_csv)
from diagnostics.data import (evaluation_entries, image_loader, load_dataset, load_gradient_images,
                              metadata, order_digest, select_gradient_batches, select_queries)
from diagnostics.retrieval import evaluate_on_off, paired_features


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def run(args, config):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Diagnostics use python on one GPU, not torchrun')
    if not config.SCSR_V41.ENABLED or config.MODEL.SIE_CAMERA or config.MODEL.SIE_VIEW:
        raise ValueError('Use the V4.1 UAD config with SCSR enabled and original SIE disabled')
    if (config.TEST.NECK_FEAT != 'before' or config.TEST.FEAT_NORM != 'yes' or config.TEST.RE_RANKING
            or config.TEST.TOP_K_EVAL != 0 or list(config.INPUT.SIZE_TEST) != [256, 128]):
        raise ValueError('Require original pre-BN CLS, L2, no reranking/Top-K, 256x128')
    if config.MODEL.METRIC_LOSS_TYPE != 'wrt' or config.MODEL.IF_LABELSMOOTH != 'off':
        raise ValueError('Gradient diagnostic requires UAD CE(no smoothing) and WRT')
    if min(args.batch_size, args.distance_chunk, args.probe_samples_per_modality) < 1 or args.workers < 0:
        raise ValueError('Batch/chunk/sample counts must be positive; workers nonnegative')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh diagnostic output directory')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = dict(version='SCSR-V4.1', status='running', args=vars(args), config_yaml=config.dump(),
                    checkpoint_sha256=file_digest(args.checkpoint),
                    checkpoint_metadata_sha256=file_digest(str(args.checkpoint) + '.json'),
                    environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda),
                    protocol=dict(feature='pre-BN CLS', normalization='L2', reranking=False, top_k=0,
                                  camera_exclusion='ALL gallery images from query camera',
                                  camera_platform='0..4 Ground, 5..6 Aerial', amp=args.amp),
                    gradient_semantics='eval-mode CE/WRT separately; no optimizer or GPD prototype update',
                    gpd_not_reconstructed='UAD model checkpoints do not store the prototype bank')

    def save_manifest():
        (output / 'diagnostic_manifest.json').write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False,
                                                                   allow_nan=False), encoding='utf-8')

    def progress(message):
        print(message, flush=True)

    model, before = None, None
    save_manifest()
    try:
        dataset = load_dataset(config)
        modalities = list(config.DATASETS.MODALITIES)
        model = make_model(config, dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids)
        # Gradient probes require the saved classifier, unlike ordinary retrieval.
        load_v41(model, args.checkpoint, include_classifier=True)
        model.to(device).eval()
        before = state_digest(model)
        manifest['state_before_sha256'] = before
        selected = select_queries(dataset, modalities, args.probe_samples_per_modality, args.seed)
        records = [metadata(entry, modalities) for entry in selected]
        internal, layers = [], []
        for images, indices in image_loader(selected, config, args.batch_size, args.workers):
            images = images.to(device)
            batch_records = [records[int(i)] for i in indices]
            ids = torch.tensor([record['modal_id'] for record in batch_records], device=device, dtype=torch.long)
            captures = {}
            with torch.no_grad(), amp_context(device, args.amp):
                for state in ('ON', 'OFF'):
                    cap = Capture()
                    with diagnostic_session(model, state == 'ON', cap):
                        model(images, mode=1, modal_ids=ids)
                    captures[state] = cap
            internal += internal_rows(captures['ON'], [r for r in batch_records if r['modality_name'] == 'TIR'], config.SCSR_V41.EPS)
            batch_layers = layerwise_rows(captures['ON'], captures['OFF'], batch_records, config.SCSR_V41.EPS)
            if any(row['abs_diff'] != 0 for row in batch_layers if row['modality_name'] != 'TIR'):
                raise AssertionError('ON/OFF unexpectedly changed RGB/NIR features')
            layers += batch_layers
        write_csv(output / 'scsr_internal_activation.csv', internal)
        write_csv(output / 'scsr_layerwise_feature_effect.csv', layers)
        gradients, losses = [], []
        fixed = select_gradient_batches(dataset, modalities, 2, args.gradient_identities, config.DATALOADER.NUM_INSTANCE, args.seed)
        for index, (groups, targets) in enumerate(fixed):
            images = load_gradient_images(groups, config, device)
            rows, values = gradient_probe(model, images, torch.tensor(targets, device=device), index, device,
                                          args.amp, args.loss_scale)
            gradients += rows
            losses += values
            progress('CE/WRT gradient batch {} complete'.format(index + 1))
        write_csv(output / 'scsr_gradient_probe.csv', gradients)
        manifest['gradient_losses'] = losses
        manifest['selected_queries'] = records
        manifest['gradient_batches'] = [[[metadata(entry, modalities) for entry in group] for group in groups]
                                        for groups, _ in fixed]
        if not args.skip_retrieval:
            queries, galleries = evaluation_entries(dataset, modalities)
            qrecords, grecords = [metadata(e, modalities) for e in queries], [metadata(e, modalities) for e in galleries]
            manifest['query_order_sha256'], manifest['gallery_order_sha256'] = order_digest(qrecords), order_digest(grecords)
            manifest['query_count'], manifest['gallery_count'] = len(queries), len(galleries)
            qfeatures = paired_features(model, image_loader(queries, config, args.batch_size, args.workers), qrecords, device, args.amp, progress)
            gfeatures = paired_features(model, image_loader(galleries, config, args.batch_size, args.workers), grecords, device, args.amp, progress)
            metrics, comparisons, summary = evaluate_on_off(qfeatures, gfeatures, qrecords, grecords, args.distance_chunk, progress)
            write_csv(output / 'scsr_on_off_metrics.csv', metrics)
            write_csv(output / 'scsr_on_off_query_comparison.csv', comparisons)
            manifest['retrieval_summary'] = summary
        manifest['state_after_sha256'] = state_digest(model)
        manifest['parameters_and_buffers_unchanged'] = manifest['state_after_sha256'] == before
        if not manifest['parameters_and_buffers_unchanged']:
            raise AssertionError('Diagnostic mutated model parameters or BN buffers')
        manifest['status'] = 'complete'
    except Exception as error:
        manifest['status'], manifest['error'] = 'failed', repr(error)
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/SCSR_V41.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--skip-retrieval', action='store_true', help='Only activation/layer/CE-WRT probes; omit full-gallery ranking')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--distance-chunk', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--probe-samples-per-modality', type=int, default=20)
    parser.add_argument('--gradient-identities', type=int, default=4)
    parser.add_argument('--loss-scale', type=float, default=1.0, help='Explicit AMP diagnostic loss scale; gradients are divided back')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = cfg.clone()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.MODEL.PRETRAIN_CHOICE = 'none'
    config.MODEL.DIST_TRAIN = False
    config.freeze()
    run(args, config)
