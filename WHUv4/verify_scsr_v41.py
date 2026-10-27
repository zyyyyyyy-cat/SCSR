"""Post-training V4.1 mechanism verification. Never imported by train.py."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from model import make_model
from utils.scsr_v41_checkpoint import load_torch, load_v41, signature
from diagnostics.core import amp_context, diagnostic_session, json_safe, state_digest, write_csv
from diagnostics.data import (evaluation_entries, image_loader, load_dataset, load_gradient_images,
                              metadata, order_digest, select_gradient_batches)
from diagnostics.v41_initial import file_hash, parameter_change_rows, restore_initial_control
from diagnostics.v41_mechanisms import (capture_batch, fixed_donor_pairs, fixed_tir_samples,
                                       gate_statistics, input_dependence_rows, paired_feature_rows,
                                       paired_gate_rows, parameter_intervention)
from diagnostics.v41_ranking import CSVStream, evaluate_four_combinations
from diagnostics.v41_gradients import final_gradient_probe


ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def validate_protocol(config, queries, galleries):
    if (list(config.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal'] or
            not config.SCSR_V41.ENABLED or config.MODEL.SIE_CAMERA or config.MODEL.SIE_VIEW or
            config.TEST.NECK_FEAT != 'before' or config.TEST.FEAT_NORM != 'yes' or
            config.TEST.RE_RANKING or config.TEST.TOP_K_EVAL != 0 or
            list(config.INPUT.SIZE_TEST) != [256, 128]):
        raise ValueError('Use original WHU-MARS-1000 RGB/IR/Thermal, pre-BN/L2, no SIE/rerank/Top-K/TTA, 256x128')
    if (len(queries), len(galleries)) != (6405, 93609):
        raise ValueError('Expected Query=6405 / Gallery=93609; got {}/{}'.format(len(queries), len(galleries)))


def extraction_identity(config, checkpoint, qrecords, grecords, args):
    return dict(version='v41-mechanism-cache-1', checkpoint_sha256=file_hash(checkpoint),
                checkpoint_metadata_sha256=file_hash(str(checkpoint) + '.json'),
                query_order_sha256=order_digest(qrecords), gallery_order_sha256=order_digest(grecords),
                query_count=len(qrecords), gallery_count=len(grecords), model=signature(config),
                pixel_mean=list(config.INPUT.PIXEL_MEAN), pixel_std=list(config.INPUT.PIXEL_STD),
                test_size=list(config.INPUT.SIZE_TEST), neck_feat=config.TEST.NECK_FEAT,
                amp=args.amp, tf32=False, extraction_batch_size=args.batch_size, torch=str(torch.__version__),
                backend=dict(device_type=torch.device(args.device).type, cuda=torch.version.cuda,
                             cudnn=torch.backends.cudnn.version(),
                             gpu=torch.cuda.get_device_name(torch.device(args.device))
                             if torch.device(args.device).type == 'cuda' else None),
                feature='L2 normalized FP32 cache of 768D pre-BN CLS',
                feature_source_hashes={name: file_hash(ROOT / name) for name in
                                       ('model/make_model.py', 'model/backbones/vit_pytorch.py', 'model/scsr_v41.py',
                                        'diagnostics/data.py')})


def extract_paired(model, entries, records, config, args, device, progress):
    features = {'ON': [], 'OFF': []}
    for batch, (images, indices) in enumerate(image_loader(entries, config, args.batch_size, args.workers)):
        images = images.to(device)
        ids = torch.tensor([records[int(i)]['modal_id'] for i in indices], device=device, dtype=torch.long)
        outputs = {}
        with torch.no_grad(), amp_context(device, args.amp):
            for state in ('ON', 'OFF'):
                with diagnostic_session(model, state == 'ON'):
                    value = model(images, mode=1, modal_ids=ids)
                if value.shape != (len(images), 768) or not value.isfinite().all():
                    raise ValueError('Nonfinite / wrong-shaped pre-BN feature in ' + state)
                if not (value.float().norm(dim=1) > 1e-12).all():
                    raise ValueError('Zero pre-BN feature cannot form a unit L2 retrieval vector in ' + state)
                outputs[state] = value.detach().float().cpu()
            non_tir = ids.detach().cpu() != 3
            if not torch.equal(outputs['ON'][non_tir], outputs['OFF'][non_tir]):
                raise AssertionError('ON/OFF unexpectedly changed RGB/NIR features')
        for state in features:
            features[state].append(outputs[state])
        if (batch + 1) % 20 == 0:
            progress('Paired extraction batch {}'.format(batch + 1))
    return {state: F.normalize(torch.cat(parts), dim=1) for state, parts in features.items()}


def feature_cache(model, queries, galleries, qrecords, grecords, config, args, output, device, identity, progress):
    path = Path(args.feature_cache) if args.feature_cache else output / 'v41_feature_cache.pth'
    reused = path.is_file()
    if reused:
        cache = load_torch(path)
        if cache.get('identity') != identity:
            raise ValueError('Feature cache provenance/precision/order/config mismatch: ' + str(path))
        qfeatures, gfeatures = cache['query'], cache['gallery']
        for group, count in ((qfeatures, len(queries)), (gfeatures, len(galleries))):
            if set(group) != {'ON', 'OFF'}:
                raise ValueError('Feature cache must contain ON and OFF')
            for value in group.values():
                if value.shape != (count, 768) or value.dtype != torch.float32 or not value.isfinite().all():
                    raise ValueError('Invalid feature cache shape/dtype/finite values')
                if not torch.allclose(value.norm(dim=1), torch.ones(count), atol=1e-5, rtol=1e-5):
                    raise ValueError('Cached features are not L2 normalized')
        for features, records in ((qfeatures, qrecords), (gfeatures, grecords)):
            non_tir = torch.tensor([r['modal_id'] != 3 for r in records])
            if not torch.equal(features['ON'][non_tir], features['OFF'][non_tir]):
                raise ValueError('Cached RGB/NIR features differ between ON/OFF')
        progress('Reusing verified paired feature cache')
    else:
        qfeatures = extract_paired(model, queries, qrecords, config, args, device, progress)
        gfeatures = extract_paired(model, galleries, grecords, config, args, device, progress)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(identity=identity, query=qfeatures, gallery=gfeatures), path)
    return qfeatures, gfeatures, dict(path=str(path.resolve()), reused=reused, sha256=file_hash(path), identity=identity)


def run_mechanisms(model, dataset, config, args, output, device, checkpoint_metadata, requested, progress, identity):
    entries = fixed_tir_samples(dataset, args.tir_samples, args.seed)
    records = [metadata(entry, list(config.DATASETS.MODALITIES)) for entry in entries]
    same_pairs, cross_pairs = fixed_donor_pairs(records), fixed_donor_pairs(records, True)
    cohort = dict(sample_count=len(records), selection='fixed seeded identity round-robin, half Ground/half Aerial; no correctness selection',
                  records=records, order_sha256=order_digest(records), same_platform_donor_indices=same_pairs,
                  cross_platform_donor_indices=cross_pairs, seed=args.seed)
    save_json(output / 'v41_tir_fixed_samples.json', cohort)
    trained = {key: value.detach().cpu().clone() for key, value in model.base.scsr_v41.state_dict().items()}
    initial, initial_manifest = None, dict(status='not_requested')
    if 2 in requested:
        parent = Path(args.checkpoint).parent
        records_dir = args.initial_records_dir or str(parent)
        trace = args.training_diagnostics or str(parent / 'scsr_v41_training_diagnostics.jsonl')
        initial, initial_manifest = restore_initial_control(config, model, checkpoint_metadata, records_dir, trace, args.initial_state)
        initial_manifest.update(checkpoint_sha256=identity['checkpoint_sha256'], fixed_sample_order_sha256=cohort['order_sha256'],
                                source_sha256={name: file_hash(ROOT / name) for name in
                                               ('model/make_model.py', 'model/backbones/vit_pytorch.py', 'model/scsr_v41.py')},
                                torch=str(torch.__version__), precision='AMP' if args.amp else 'FP32',
                                groups=dict(A='trained U / trained condition', B='initial U / initial condition',
                                            C='initial U / trained condition', D='trained U / initial condition'),
                                semantics='all groups keep the same final epoch-120 backbone; no independent initialization-model baseline')
        save_json(output / 'v41_initial_reconstruction_manifest.json', initial_manifest)
        if initial is None:
            progress('Experiment 2 unavailable: {}; other experiments continue'.format(initial_manifest.get('reason')))
            for name in ('v41_initial_vs_trained_parameters.csv', 'v41_initial_vs_trained_gate.csv', 'v41_initial_vs_trained_features.csv'):
                write_csv(output / name, [dict(status='skipped', reason=initial_manifest.get('reason'))])
        else:
            write_csv(output / 'v41_initial_vs_trained_parameters.csv', parameter_change_rows(initial, trained, initial_manifest))
            torch.save(dict(state_dict=initial, provenance=initial_manifest), output / 'v41_initial_control.pth')
            torch.save(dict(provenance=initial_manifest,
                            parameter_differences={key: trained[key].double() - initial[key].double() for key in initial}),
                       output / 'v41_initial_parameter_deltas.pth')
    if initial is None and 3 not in requested:
        return dict(cohort_sha256=cohort['order_sha256'], initial_control=initial_manifest)
    states = {}
    if initial is not None:
        states['B'] = initial
        states['C'] = dict(trained, U=initial['U'])
        states['D'] = dict(initial, U=trained['U'])
    gate_rows, feature_rows, vparts, mparts = [], [], [], []
    for batch, (images, indices) in enumerate(image_loader(entries, config, args.batch_size, args.workers)):
        images = images.to(device); indices = [int(i) for i in indices]
        batch_records = [records[i] for i in indices]
        normal = capture_batch(model, images, device, args.amp)
        vparts.append(normal.internal['v']); mparts.append(normal.internal['m'])
        if initial is not None:
            gate_rows += gate_statistics(normal, batch_records, 'A', indices)
            for group, state in states.items():
                with parameter_intervention(model.base.scsr_v41, state):
                    other = capture_batch(model, images, device, args.amp)
                if not torch.equal(other.layers['block7_norm1'], normal.layers['block7_norm1']):
                    raise AssertionError('Parameter intervention changed the fixed upstream backbone')
                gate_rows += gate_statistics(other, batch_records, group, indices)
                gate_rows += paired_gate_rows(normal, other, batch_records, 'A', group, indices)
                feature_rows += paired_feature_rows(normal, other, batch_records, 'A', group, indices)
        progress('Mechanism reference/initial batch {}'.format(batch + 1))
    vbank, mbank = torch.cat(vparts), torch.cat(mparts)
    del vparts, mparts
    if initial is not None:
        for rows in (gate_rows, feature_rows):
            for row in rows:
                row['control_type'] = initial_manifest['control_type']
                row['exact_original_initialization'] = initial_manifest['exact_original_initialization']
        write_csv(output / 'v41_initial_vs_trained_gate.csv', gate_rows)
        write_csv(output / 'v41_initial_vs_trained_features.csv', feature_rows)
    if 3 in requested:
        condition_rows, band_rows = [], []
        kinds = ('spatial_mean_v', 'donor_same_platform_v', 'donor_cross_platform_v', 'zero_ll_condition',
                 'disable_lh', 'disable_hl', 'disable_hh', 'disable_all')
        for batch, (images, indices) in enumerate(image_loader(entries, config, args.batch_size, args.workers)):
            images = images.to(device); indices = [int(i) for i in indices]
            batch_records = [records[i] for i in indices]
            normal = capture_batch(model, images, device, args.amp)
            if not torch.equal(normal.internal['v'], vbank[indices]):
                raise AssertionError('Condition-bank pass changed input order or model behavior')
            # The untouched production path and normal intervention must agree exactly.
            mirror = capture_batch(model, images, device, args.amp, kind='normal')
            if any(not torch.equal(normal.layers[key], mirror.layers[key]) for key in normal.layers):
                raise AssertionError('Diagnostic mirror is not equivalent to production SCSR')
            for kind in kinds:
                pairs = same_pairs if kind == 'donor_same_platform_v' else cross_pairs
                donor_v = vbank[[pairs[index] for index in indices]] if kind.startswith('donor_') else None
                other = capture_batch(model, images, device, args.amp, kind=kind, donor_v=donor_v)
                if not torch.equal(other.internal['ll'], normal.internal['ll']) or any(
                        not torch.equal(other.internal[key], normal.internal[key]) for key in ('lh', 'hl', 'hh')):
                    raise AssertionError('Condition/band intervention changed original frequency content')
                if kind.startswith('disable_') and not torch.equal(other.internal['m'], normal.internal['m']):
                    raise AssertionError('Band intervention changed generated gate/condition')
                if not torch.equal(other.internal['condition'][:, 16:], mirror.internal['condition'][:, 16:]):
                    raise AssertionError('Intervention changed signed HF projections or RMS conditioning')
                if kind == 'zero_ll_condition' and other.internal['condition'][:, :16].count_nonzero():
                    raise AssertionError('LL condition was not zeroed')
                rows = gate_statistics(other, batch_records, kind, indices)
                rows += paired_gate_rows(normal, other, batch_records, 'normal', kind, indices)
                rows += [dict(row, row_type='feature_comparison') for row in
                         paired_feature_rows(normal, other, batch_records, 'normal', kind, indices)]
                for row in rows:
                    local = indices.index(row['sample_index'])
                    donor = pairs[row['sample_index']] if kind.startswith('donor_') else None
                    row.update(donor_index=donor, donor_pid=records[donor]['pid'] if donor is not None else '',
                               donor_path=records[donor]['path'] if donor is not None else '',
                               donor_platform=records[donor]['platform'] if donor is not None else '',
                               current_input_platform=batch_records[local]['platform'])
                (band_rows if kind.startswith('disable_') else condition_rows).extend(rows)
            progress('Condition/band intervention batch {}'.format(batch + 1))
        write_csv(output / 'v41_condition_intervention.csv', condition_rows)
        write_csv(output / 'v41_frequency_band_intervention.csv', band_rows)
        stream = CSVStream(output / 'v41_gate_input_dependence.csv')
        try:
            for row in input_dependence_rows(mbank, records, same_pairs, cross_pairs):
                stream.append(row)
        finally:
            stream.close()
        save_json(output / 'v41_condition_manifest.json', dict(identity=identity, cohort=cohort,
                  interventions=list(kinds), donor_replacement='V only, original LL/HF preserved',
                  zero_ll_condition='zero W_L(LL), original LL kept for IDWT',
                  disable_band='zero applied M for that direction, generated M and original HF unchanged',
                  disable_all_identity_shortcut='exact original Patch writeback to avoid Haar roundoff',
                  gain_order='descending signed M (equivalent to descending 1+M); tie-averaged Spearman',
                  normal_mirror_exactly_matches_production=True))
    return dict(cohort_sha256=cohort['order_sha256'], initial_control=initial_manifest)


def run(args, config):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run verification with python on one GPU, not torchrun')
    requested = {int(value) for value in args.experiments.split(',')}
    if not requested or requested - {1, 2, 3, 4, 5}:
        raise ValueError('--experiments must be a comma-separated subset of 1,2,3,4,5')
    if min(args.batch_size, args.distance_chunk, args.tir_samples) < 1 or args.workers < 0:
        raise ValueError('Positive batch/chunk/sample counts and nonnegative workers are required')
    if 5 in requested and (args.gradient_identities < 2 or not np.isfinite(args.loss_scale) or args.loss_scale <= 0):
        raise ValueError('Gradient probe needs at least two identities and a finite positive loss scale')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file() or not Path(str(checkpoint) + '.json').is_file():
        raise ValueError('Provide the epoch-120 .pth together with its matching .pth.json sidecar')
    checkpoint_metadata = json.loads(Path(str(checkpoint) + '.json').read_text(encoding='utf-8'))
    if checkpoint_metadata.get('epoch') != 120:
        raise ValueError('This protocol requires checkpoint metadata epoch=120, not filename inference')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh output directory to keep paired experiments separate')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    elif args.amp:
        raise ValueError('--amp requires CUDA')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = dict(status='running', model_version='SCSR-V4.1', requested_experiments=sorted(requested),
                    args=vars(args), config_yaml=config.dump(), checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=file_hash(checkpoint), checkpoint_metadata_sha256=file_hash(str(checkpoint) + '.json'),
                    environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device)),
                    protocol=dict(feature='768D pre-BN CLS, then L2 normalization',
                                  distance='original UAD squared Euclidean, FP32 CPU chunks / torch.argsort',
                                  camera_filter='exclude ALL query-camera Gallery images', reranking=False, TTA=False,
                                  top_k=0, amp=args.amp, tf32=False, modalities={1: 'RGB', 2: 'NIR', 3: 'TIR'},
                                  platform='camera 0..4 Ground; 5..6 Aerial', seed=args.seed),
                    combinations=dict(A='Query OFF / Gallery OFF', B='Query ON / Gallery ON',
                                      C='Query ON / Gallery OFF', D='Query OFF / Gallery ON'),
                    source_sha256={str(path.relative_to(ROOT)): file_hash(path) for path in
                                   [Path(__file__).resolve()] +
                                   sorted((ROOT / 'diagnostics').glob('v41_*.py'))})
    model, before, flags = None, None, None
    def progress(message):
        print(message, flush=True)
    def persist():
        save_json(output / 'diagnostic_manifest.json', manifest)
    persist()
    try:
        dataset = load_dataset(config)
        queries, galleries = evaluation_entries(dataset, list(config.DATASETS.MODALITIES))
        validate_protocol(config, queries, galleries)
        qrecords = [metadata(entry, list(config.DATASETS.MODALITIES)) for entry in queries]
        grecords = [metadata(entry, list(config.DATASETS.MODALITIES)) for entry in galleries]
        identity = extraction_identity(config, checkpoint, qrecords, grecords, args)
        manifest['data_identity'] = identity
        model = make_model(config, dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids)
        load_v41(model, checkpoint, include_classifier=True)
        model.to(device).eval()
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        before = state_digest(model); manifest['state_before_sha256'] = before; persist()
        if requested & {1, 4}:
            qfeatures, gfeatures, cache_info = feature_cache(model, queries, galleries, qrecords, grecords,
                                                           config, args, output, device, identity, progress)
            manifest['feature_cache'] = cache_info; persist()
            summary = evaluate_four_combinations(qfeatures, gfeatures, qrecords, grecords, output,
                                                 args.distance_chunk, include_identity=4 in requested, progress=progress)
            manifest['retrieval_summary'] = summary
            save_json(output / 'v41_on_off_manifest.json', dict(identity=identity, cache=cache_info,
                      combinations=manifest['combinations'], summary=summary, complete_gallery=True, top_k=0,
                      margin='nearest negative squared distance minus nearest positive squared distance',
                      fixed_anchors='A and B selected neighbors both measured/ranked in both states'))
            del qfeatures, gfeatures
        if requested & {2, 3}:
            manifest['mechanisms'] = run_mechanisms(model, dataset, config, args, output, device,
                                                   checkpoint_metadata, requested, progress, identity)
        if 5 in requested:
            if config.MODEL.METRIC_LOSS_TYPE != 'wrt' or config.MODEL.IF_LABELSMOOTH != 'off':
                raise ValueError('Final gradient probe requires the supplied CE(no smoothing)/WRT protocol')
            fixed = select_gradient_batches(dataset, list(config.DATASETS.MODALITIES), args.gradient_batches,
                                            args.gradient_identities, config.DATALOADER.NUM_INSTANCE, args.seed)
            rows, losses = [], []
            try:
                for name, p in model.named_parameters():
                    p.requires_grad_(flags[name])
                for batch, (groups, target) in enumerate(fixed):
                    images = load_gradient_images(groups, config, device)
                    values, summary = final_gradient_probe(model, images, torch.tensor(target, device=device), batch,
                                                           device, args.amp, args.loss_scale, args.proca_gradient)
                    rows += values; losses += summary
                    progress('Final CE/WRT gradient batch {}'.format(batch + 1))
            finally:
                for p in model.parameters():
                    p.requires_grad_(False)
            write_csv(output / 'v41_final_gradient_probe.csv', rows)
            manifest['gradient_probe'] = dict(losses=losses, gpd='not evaluated: original checkpoints contain no prototype bank',
                                              raw_unweighted_single_loss_gradients=True,
                                              hypothetical_single_loss_clip_not_joint_training_clip=True,
                                              batches=[[[metadata(entry, list(config.DATASETS.MODALITIES)) for entry in group]
                                                        for group in groups] for groups, _ in fixed])
        after = state_digest(model)
        manifest.update(state_after_sha256=after, parameters_and_bn_buffers_unchanged=after == before,
                        parameter_grad_buffers_unchanged=all(p.grad is None for p in model.parameters()))
        if after != before or not manifest['parameter_grad_buffers_unchanged']:
            raise AssertionError('Verification mutated parameters/BN buffers or populated .grad')
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        if model is not None and before is not None:
            manifest['state_finally_sha256'] = state_digest(model)
            manifest['finally_parameters_and_bn_buffers_unchanged'] = manifest['state_finally_sha256'] == before
            if flags is not None:
                for name, p in model.named_parameters():
                    p.requires_grad_(flags[name])
        persist()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/SCSR_V41.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--experiments', default='1,2,3,4', help='Comma separated, default four core experiments; optional 5')
    parser.add_argument('--feature-cache', help='Reuse/create paired cache after strict provenance validation')
    parser.add_argument('--initial-state', help='Optional original calibrated initial SCSR/full-model snapshot')
    parser.add_argument('--initial-records-dir', help='Default: checkpoint directory containing rank initialization JSONs')
    parser.add_argument('--training-diagnostics', help='Default: checkpoint directory/scsr_v41_training_diagnostics.jsonl')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--distance-chunk', type=int, default=16)
    parser.add_argument('--tir-samples', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gradient-batches', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--gradient-identities', type=int, default=4)
    parser.add_argument('--proca-gradient', action='store_true')
    parser.add_argument('--loss-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    config = cfg.clone()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.MODEL.PRETRAIN_CHOICE = 'none'
    config.MODEL.DIST_TRAIN = False
    config.freeze()
    run(args, config)
