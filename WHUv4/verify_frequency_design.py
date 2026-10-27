"""Evidence-only WHU-MARS frequency diagnostics for frozen UAD/SCSR checkpoints."""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from model import make_model
from utils.scsr_v41_checkpoint import load_torch, load_v41
from diagnostics.core import amp_context, json_safe, write_csv
from diagnostics.data import image_loader, load_dataset, load_gradient_images, select_gradient_batches
from diagnostics.frequency_core import (BANDS, LAYERS, FrequencyIntervention, LayerCapture,
                                        calibration_alpha, cosine, decompose, descriptor,
                                        no_parameter_mutation, pair_auc, relative_l2,
                                        paired_distances, reconstruct_v41_condition, squared_distance, tensor_sha256)
from diagnostics.frequency_pairs import (digest, fixed_train_split, matched_pairs, record,
                                         relation_pairs, required_scene_pairs)
from diagnostics.frequency_evaluation import (compare_margins, compare_queries, evaluate_features,
                                              normalized_features)
from diagnostics.v41_mechanisms import capture_batch
from loss.wrt_loss import TripletLoss_WRT
from processor.processor import PCA


ROOT = Path(__file__).resolve().parent
DESCRIPTORS = ('signed_spatial', 'channel_rms', 'energy', 'energy_ratio')


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_uad(model, path):
    raw = load_torch(path)
    if isinstance(raw, dict) and 'state_dict' in raw and isinstance(raw['state_dict'], dict):
        raw = raw['state_dict']
    state = {key[7:] if key.startswith('module.') else key: value for key, value in raw.items()}
    if any('scsr_v41' in key for key in state):
        raise ValueError('The UAD checkpoint unexpectedly contains SCSR parameters')
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('UAD checkpoint/model mismatch; missing={}, extra={}'.format(
            sorted(set(expected) - set(state)), sorted(set(state) - set(expected))))
    model.load_state_dict(state, strict=True)


def validate_cfg(config):
    if (config.DATASETS.NAMES != 'WHU-MARS' or list(config.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal'] or
            config.DATASETS.DATASET_DIR != 'WHU-MARS' or config.DATASETS.FILTER_GD or
            list(config.INPUT.SIZE_TEST) != [256, 128] or list(config.MODEL.STRIDE_SIZE) != [16, 16] or
            config.MODEL.SIE_CAMERA or config.MODEL.SIE_VIEW or config.TEST.NECK_FEAT != 'before' or
            config.TEST.FEAT_NORM != 'yes' or config.TEST.RE_RANKING or config.TEST.TOP_K_EVAL != 0):
        raise ValueError('Frequency diagnostics require full WHU-MARS-1000 and the frozen 256x128 pre-BN/L2 protocol')


def extract_profiles(model, entries, config, args, layers=LAYERS):
    stores = {layer: {band: {kind: [] for kind in DESCRIPTORS} for band in BANDS} for layer in layers}
    final = []
    with torch.no_grad():
        for images, indices in image_loader(entries, config, args.batch_size, args.workers):
            images = images.to(args.device)
            ids = torch.tensor([entries[int(i)][3] for i in indices], device=args.device, dtype=torch.long)
            with LayerCapture(model, layers) as capture, amp_context(torch.device(args.device), args.amp):
                cls = model(images, mode=1, modal_ids=ids)
            final.append(cls.detach().float().cpu())
            for layer in layers:
                with torch.no_grad():
                    bands = decompose(capture.values[layer])
                    for band, value in zip(BANDS, bands):
                        for kind in DESCRIPTORS:
                            item = descriptor(value, kind, bands).cpu()
                            stores[layer][band][kind].append(item.half() if kind == 'signed_spatial' else item)
    return ({layer: {band: {kind: torch.cat(parts) for kind, parts in kinds.items()}
                            for band, kinds in bands.items()} for layer, bands in stores.items()},
            normalized_features(torch.cat(final)))


def pair_rows(profiles, records, positives, negatives, final_features=None):
    groups = relation_pairs(records, positives, negatives)
    summary, distributions = [], []
    for layer in LAYERS:
        for band in BANDS:
            for kind in DESCRIPTORS:
                values = profiles[layer][band][kind]
                for (qm, gm, relation), pairs in sorted(groups.items()):
                    pos, neg = pairs['positive'], pairs['negative']
                    pd = paired_distances(values, pos).numpy()
                    nd = paired_distances(values, neg).numpy()
                    correspondence = ''
                    if final_features is not None and len(pd) + len(nd) > 1:
                        cd = torch.cat((paired_distances(final_features, pos), paired_distances(final_features, neg))).numpy()
                        fd = np.concatenate((pd, nd))
                        if fd.std() > 0 and cd.std() > 0:
                            correspondence = float(np.corrcoef(fd, cd)[0, 1])
                    pids = {records[a]['pid'] for a, _ in pos}
                    row = dict(layer=layer, band=band, descriptor_type=kind, query_modality=qm,
                               gallery_modality=gm, platform_relation=relation,
                               positive_count=len(pd), negative_count=len(nd), valid_identity_count=len(pids),
                               positive_distance=float(pd.mean()) if len(pd) else '',
                               positive_distance_std=float(pd.std()) if len(pd) else '',
                               negative_distance=float(nd.mean()) if len(nd) else '',
                               negative_distance_std=float(nd.std()) if len(nd) else '',
                               identity_gap=float(nd.mean() - pd.mean()) if len(pd) and len(nd) else '',
                               pair_auc=pair_auc(pd, nd), final_cls_distance_correlation=correspondence)
                    summary.append(row)
                    for label, array in (('positive', pd), ('negative', nd)):
                        if len(array):
                            distributions.append(dict(layer=layer, band=band, descriptor_type=kind,
                                query_modality=qm, gallery_modality=gm, platform_relation=relation, pair_type=label,
                                count=len(array), mean=float(array.mean()), std=float(array.std()),
                                minimum=float(array.min()), q25=float(np.quantile(array, .25)),
                                median=float(np.median(array)), q75=float(np.quantile(array, .75)),
                                maximum=float(array.max())))
    return summary, distributions


def scene_rows(profiles, records, positives, negatives):
    groups = relation_pairs(records, positives, negatives)
    rows, camera = [], []
    for layer in LAYERS:
        for band in BANDS:
            for kind in DESCRIPTORS:
                values = profiles[layer][band][kind]
                for key, pairs in sorted(groups.items()):
                    pos, neg = pairs['positive'], pairs['negative']
                    for label, selected in (('same_identity', pos), ('different_identity', neg)):
                        distances = paired_distances(values, selected)
                        rows.append(dict(layer=layer, band=band, descriptor_type=kind,
                            query_modality=key[0], gallery_modality=key[1], relation=key[2], pair_type=label,
                            pair_count=len(selected), valid_identity_count=len({records[a]['pid'] for a, _ in selected}),
                            mean_distance=float(distances.mean()) if len(distances) else '',
                            std_distance=float(distances.std(unbiased=False)) if len(distances) else ''))
                # Energy/camera/platform statistics are descriptive and explicitly not identity evidence.
                if kind in ('energy', 'energy_ratio'):
                    for modality in ('RGB', 'NIR', 'TIR'):
                        for camid in range(7):
                            indices = [i for i, r in enumerate(records) if r['modality'] == modality and r['camid'] == camid]
                            if indices:
                                x = values[indices]
                                camera.append(dict(layer=layer, band=band, descriptor_type=kind, modality=modality,
                                    camid=camid, platform='G' if camid < 5 else 'A', sample_count=len(indices),
                                    identity_count=len({records[i]['pid'] for i in indices}), mean=float(x.mean()),
                                    std=float(x.std(unbiased=False))))
    return rows, camera


def run_v41_scene(v41, entries, records, config, args):
    if v41 is None:
        return [dict(status='not_run', reason='--v41-checkpoint was not supplied')]
    vectors = defaultdict(list); meta = []
    for images, indices in image_loader(entries, config, args.batch_size, args.workers):
        ids = torch.tensor([entries[int(i)][3] for i in indices], device=args.device, dtype=torch.long)
        tir_local = [j for j, value in enumerate(ids.tolist()) if value == 3]
        if not tir_local:
            continue
        tir_images = images[tir_local].to(args.device)
        cap = capture_batch(v41, tir_images, torch.device(args.device), args.amp)
        condition, rebuilt_v = reconstruct_v41_condition(v41.base.scsr_v41, cap.internal, verify_v=True)
        vectors['condition'].append(condition.flatten(1))
        vectors['v'].append(cap.internal['v'].flatten(1).float().cpu())
        vectors['m'].append(cap.internal['m'].flatten(1).float().cpu())
        meta.extend(records[int(indices[j])] for j in tir_local)
    rows = []
    for key, parts in vectors.items():
        value = F.normalize(torch.cat(parts), dim=1)
        by = defaultdict(list)
        for i, row in enumerate(meta):
            by[(row['platform'], row['camid'])].append(i)
        for (platform, camid), indices in sorted(by.items()):
            x = value[indices]
            rows.append(dict(status='ok', component=key, platform=platform, camid=camid,
                             sample_count=len(indices), identity_count=len({meta[i]['pid'] for i in indices}),
                             normalized_feature_rms=float(x.square().mean().sqrt())))
        pos, neg = matched_pairs(meta, min(args.max_pairs, 2048), args.seed)
        pd = squared_distance(value[[a for a, _ in pos]], value[[b for _, b in pos]])
        nd = squared_distance(value[[a for a, _ in neg]], value[[b for _, b in neg]])
        rows.append(dict(status='ok', component=key, platform='all', camid='all', sample_count=len(meta),
                         identity_count=len({r['pid'] for r in meta}), positive_distance=float(pd.mean()),
                         negative_distance=float(nd.mean()), identity_gap=float(nd.mean() - pd.mean()),
                         pair_auc=pair_auc(pd.numpy(), nd.numpy())))
    return rows


def parse_directions(text):
    result = []
    if not text:
        return result
    for item in text.split(','):
        fields = item.strip().split(':')
        if len(fields) not in (2, 3) or fields[0].upper() not in BANDS or fields[1] not in ('+', '-'):
            raise ValueError('Directions use BAND:+[:budget], for example LH:+:0.01')
        result.append((fields[0].upper(), 1 if fields[1] == '+' else -1,
                       float(fields[2]) if len(fields) == 3 else None))
    return result


def calibration(model, entries, config, args):
    tir = [entry for entry in entries if entry[3] == 3][:args.calibration_images]
    if not tir:
        raise ValueError('No TIR calibration images in fixed train split')
    output = {}
    for images, indices in image_loader(tir, config, args.batch_size, args.workers):
        ids = torch.full((len(images),), 3, device=args.device, dtype=torch.long)
        with torch.no_grad(), LayerCapture(model, LAYERS) as capture:
            model(images.to(args.device), mode=1, modal_ids=ids)
        for layer in LAYERS:
            output.setdefault(layer, []).append(capture.values[layer].detach())
    tokens = {layer: torch.cat(parts) for layer, parts in output.items()}
    rows, lookup = [], {}
    for layer in LAYERS:
        for band in BANDS:
            for target in args.budgets:
                cap = args.ll_alpha_cap if band == 'LL' else args.hf_alpha_cap
                alpha, raw = calibration_alpha(tokens[layer], band, target, cap)
                lookup[(layer, band, target)] = alpha
                rows.append(dict(layer=layer, band=band, target_r_attn=target, alpha=alpha,
                                 unconstrained_alpha=raw, alpha_cap=cap, capped=raw > cap,
                                 calibration_images=len(tir)))
    return lookup, rows


def extract_intervention(model, entries, records, config, args, layer, band, sign, alpha, control, seed):
    features, r_attn, attention_change, cls_change, cls_cosine = [], [], [], [], []
    for images, indices in image_loader(entries, config, args.batch_size, args.workers):
        images = images.to(args.device)
        ids = torch.tensor([entries[int(i)][3] for i in indices], device=args.device, dtype=torch.long)
        attention = {}
        block = model.base.blocks[layer - 1]
        def save_attention(name):
            def hook(module, inputs, output):
                attention[name] = output.detach()
            return hook
        handle = block.attn.register_forward_hook(save_attention('baseline'))
        with torch.no_grad(), LayerCapture(model, (layer,)):
            base = model(images, mode=1, modal_ids=ids)
        handle.remove()
        handle = block.attn.register_forward_hook(save_attention('changed'))
        intervention = FrequencyIntervention(model, layer, band, sign, alpha, ids, control, seed, collect=True)
        try:
            with torch.no_grad(), intervention, LayerCapture(model, (layer,)) as changed_cap:
                changed = model(images, mode=1, modal_ids=ids)
        finally:
            handle.remove()
        mask = ids == 3
        if mask.any():
            r_attn.extend(intervention.records)
            cls_change.extend(relative_l2(base[mask], changed[mask]).cpu().tolist())
            cls_cosine.extend(cosine(base[mask], changed[mask]).cpu().tolist())
            attention_change.extend(relative_l2(attention['baseline'][mask], attention['changed'][mask]).cpu().tolist())
        features.append(normalized_features(changed))
    return torch.cat(features), dict(actual_r_attn_mean=float(np.mean(r_attn)) if r_attn else 0.,
        actual_r_attn_std=float(np.std(r_attn)) if r_attn else 0., tir_count=len(r_attn),
        attention_output_relative_l2_mean=float(np.mean(attention_change)) if attention_change else 0.,
        final_cls_relative_l2_mean=float(np.mean(cls_change)) if cls_change else 0.,
        final_cls_cosine_mean=float(np.mean(cls_cosine)) if cls_cosine else 1.)


def formal_confirmation(model, entries, records, query_count, config, args, alpha_lookup, directions):
    qi, gi = list(range(query_count)), list(range(query_count, len(entries)))
    baseline, diag = extract_intervention(model, entries, records, config, args, 7, None, 1, 0., 'none', args.seed)
    protocols = [('ALL', 'ALL', qi, gi),
                 ('TIR', 'ALL', [i for i in qi if records[i]['modality'] == 'TIR'], gi),
                 ('ALL', 'TIR', qi, [i for i in gi if records[i]['modality'] == 'TIR']),
                 ('TIR', 'TIR', [i for i in qi if records[i]['modality'] == 'TIR'],
                  [i for i in gi if records[i]['modality'] == 'TIR'])]
    metrics = []
    base_queries = base_margins = None
    for qdomain, gdomain, pqi, pgi in protocols:
        metric, query_rows, margin_rows = evaluate_features(baseline, records, pqi, pgi, 'A_original',
                                                            'official_test_confirmation')
        metrics.append(dict(metric, query_domain=qdomain, gallery_domain=gdomain, tir_side='none', **diag))
        if qdomain == 'ALL' and gdomain == 'ALL':
            base_queries, base_margins = query_rows, margin_rows
    changes, margins = [], []
    for band, sign, budget in directions:
        if budget is None:
            raise ValueError('Every --confirm-interventions item must include its frozen training-selected budget')
        alpha = alpha_lookup[(7, band, budget)]
        changed, cdiag = extract_intervention(model, entries, records, config, args, 7, band, sign, alpha,
                                              'frequency', args.seed)
        for side in ('query', 'gallery', 'both'):
            hybrid = baseline.clone()
            if side in ('query', 'both'):
                hybrid[:query_count] = changed[:query_count]
            if side in ('gallery', 'both'):
                hybrid[query_count:] = changed[query_count:]
            name = '{}_{}_{}_{}'.format(band, 'enhance' if sign > 0 else 'suppress', budget, side)
            query_rows = margin_rows = None
            for qdomain, gdomain, pqi, pgi in protocols:
                metric, current_queries, current_margins = evaluate_features(hybrid, records, pqi, pgi, name,
                                                                             'official_test_confirmation')
                metrics.append(dict(metric, query_domain=qdomain, gallery_domain=gdomain, band=band,
                                    direction=sign, alpha=alpha, target_r_attn=budget, tir_side=side, **cdiag))
                if qdomain == 'ALL' and gdomain == 'ALL':
                    query_rows, margin_rows = current_queries, current_margins
            for row in compare_queries(base_queries, query_rows):
                row['tir_side'] = side; changes.append(row)
            for row in compare_margins(base_margins, margin_rows):
                row['tir_side'] = side; margins.append(row)
    return metrics, changes, margins


def intervention_suite(model, entries, records, qi, gi, config, args, alpha_lookup, output, layer, directions, scope):
    baseline, base_diag = extract_intervention(model, entries, records, config, args, layer, None, 1, 0., 'none', args.seed)
    base_metric, base_queries, base_margins = evaluate_features(baseline, records, qi, gi, 'A_original', scope)
    metrics, changes, margins, controls = [dict(base_metric, **base_diag)], [], [], [dict(group='A_original', **base_diag)]
    for band, sign, explicit_budget in directions:
        budgets = (explicit_budget,) if explicit_budget is not None else args.budgets
        for budget in budgets:
            alpha = alpha_lookup[(layer, band, budget)]
            name = '{}_{}_{}_{}'.format(band, 'enhance' if sign > 0 else 'suppress', layer, budget)
            feature, diag = extract_intervention(model, entries, records, config, args, layer, band, sign, alpha,
                                                 'frequency', args.seed)
            metric, queries, mr = evaluate_features(feature, records, qi, gi, name, scope)
            metrics.append(dict(metric, band=band, direction=sign, alpha=alpha, target_r_attn=budget, **diag))
            changes.extend(compare_queries(base_queries, queries)); margins.extend(compare_margins(base_margins, mr))
            controls.append(dict(group=name, control='frequency', band=band, direction=sign, alpha=alpha,
                                 target_r_attn=budget, **diag))
            if layer == 7:
                for control in ('random_frequency', 'spatial', 'channel'):
                    for seed in args.control_seeds:
                        cname = '{}_{}_seed{}'.format(name, control, seed)
                        f, d = extract_intervention(model, entries, records, config, args, layer, band, sign, alpha, control, seed)
                        cm, cq, cr = evaluate_features(f, records, qi, gi, cname, scope)
                        metrics.append(dict(cm, band=band, direction=sign, alpha=alpha, target_r_attn=budget,
                                            control=control, control_seed=seed, **d))
                        changes.extend(compare_queries(base_queries, cq)); margins.extend(compare_margins(base_margins, cr))
                        controls.append(dict(group=cname, control=control, band=band, direction=sign, alpha=alpha,
                                             target_r_attn=budget, control_seed=seed, **d))
    return metrics, changes, margins, controls


def gradient_probe(model, dataset, config, args):
    loss_rows, alignment_rows, margin_rows = [], [], []
    batches = select_gradient_batches(dataset, list(config.DATASETS.MODALITIES), args.gradient_batches,
                                      args.gradient_identities, config.DATALOADER.NUM_INSTANCE, args.seed)
    for batch_id, (groups, targets) in enumerate(batches):
        images = load_gradient_images(groups, config, args.device)
        target = torch.tensor(targets, device=args.device)
        labels = target.repeat(len(groups))
        gradients = {}
        for loss_name in ('CE', 'WRT', 'ProCA'):
            with torch.enable_grad(), LayerCapture(model, (7,), keep_grad=True) as capture:
                score, cls, _ = model(images, label=target, mode=0)
                if loss_name == 'CE':
                    loss = F.cross_entropy(score, labels)
                elif loss_name == 'WRT':
                    loss = TripletLoss_WRT()(cls, labels)[0]
                else:
                    loss = PCA(cls, target, len(groups), config.DATALOADER.NUM_INSTANCE)[0]
            z = capture.values[7]
            grad = torch.autograd.grad(loss, z, retain_graph=False)[0].detach().float()
            gradients[loss_name] = grad
            bands, gbands = decompose(z.detach()), decompose(grad)
            for band, value, gvalue in zip(BANDS, bands, gbands):
                norm, gnorm = float(value.norm()), float(gvalue.norm())
                for direction in (-1, 1):
                    delta = direction * args.gradient_alpha * value
                    derivative = float((gvalue * delta).sum())
                    loss_rows.append(dict(batch=batch_id, loss=loss_name, band=band,
                        feature_norm=norm, gradient_norm=gnorm, gradient_rms=float(gvalue.square().mean().sqrt()),
                        normalized_gradient_norm=gnorm / max(norm, 1e-12), direction=direction,
                        alpha=args.gradient_alpha, directional_derivative=derivative,
                        normalized_directional_derivative=derivative / max(float(delta.norm()), 1e-12),
                        optimizer_step=False, gpd='not evaluated: checkpoint contains no prototype bank'))
        for band_index, band in enumerate(BANDS):
            for left, right in (('CE', 'WRT'), ('CE', 'ProCA'), ('WRT', 'ProCA')):
                a, b = decompose(gradients[left])[band_index].flatten(), decompose(gradients[right])[band_index].flatten()
                alignment_rows.append(dict(batch=batch_id, band=band, loss_a=left, loss_b=right,
                                           cosine=float(F.cosine_similarity(a, b, dim=0))))
        # Fixed hard positives/negatives inside the frozen baseline batch.
        feat = F.normalize(cls.detach().float(), dim=1)
        distances = torch.cdist(feat, feat).square()
        margins = []
        for i in range(len(feat)):
            pos = (labels == labels[i]); pos[i] = False
            neg = labels != labels[i]
            if pos.any() and neg.any():
                p = int(torch.argmax(distances[i].masked_fill(~pos, -1)))
                n = int(torch.argmin(distances[i].masked_fill(~neg, float('inf'))))
                margins.append(distances[i, n] - distances[i, p])
        if margins:
            with torch.enable_grad(), LayerCapture(model, (7,), keep_grad=True) as cap:
                _, live, _ = model(images, label=target, mode=0)
                live = F.normalize(live.float(), dim=1); d = torch.cdist(live, live).square()
                fixed = []
                for i in range(len(live)):
                    pos = labels == labels[i]; pos[i] = False; neg = labels != labels[i]
                    if pos.any() and neg.any():
                        p = int(torch.argmax(distances[i].masked_fill(~pos, -1)))
                        n = int(torch.argmin(distances[i].masked_fill(~neg, float('inf'))))
                        fixed.append(d[i, n] - d[i, p])
                margin = torch.stack(fixed).mean()
            mg = torch.autograd.grad(margin, cap.values[7])[0].detach().float()
            for band, value, gvalue in zip(BANDS, decompose(cap.values[7].detach()), decompose(mg)):
                for direction in (-1, 1):
                    delta = direction * args.gradient_alpha * value
                    margin_rows.append(dict(batch=batch_id, band=band, direction=direction,
                        alpha=args.gradient_alpha, fixed_triplet_count=len(fixed), baseline_margin=float(margin.detach()),
                        directional_derivative=float((gvalue * delta).sum()),
                        normalized_directional_derivative=float((gvalue * delta).sum() / delta.norm().clamp_min(1e-12))))
    return loss_rows, alignment_rows, margin_rows


def run(args, config):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run diagnostics with python on one GPU, not torchrun')
    requested = {int(x) for x in args.experiments.split(',')}
    if not requested or requested - set(range(1, 7)):
        raise ValueError('--experiments must be a comma-separated subset of 1..6')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    validate_cfg(config)
    if not Path(args.uad_checkpoint).is_file():
        raise ValueError('Missing UAD checkpoint')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
        torch.cuda.set_device(device); torch.cuda.manual_seed_all(args.seed)
    elif args.amp:
        raise ValueError('--amp requires CUDA')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    manifest = dict(status='running', requested_experiments=sorted(requested), args=vars(args),
        config_yaml=config.dump(), protocol=dict(precision='AMP' if args.amp else 'FP32', haar='FP32',
        selection='train identities only; official test labels never select configurations', optimizer_step=False),
        checkpoints=dict(uad=dict(path=str(Path(args.uad_checkpoint).resolve()), sha256=file_hash(args.uad_checkpoint))),
        source_sha256={str(path.relative_to(ROOT)): file_hash(path) for path in
            [Path(__file__).resolve()] + sorted((ROOT / 'diagnostics').glob('frequency_*.py'))})
    save_json(output / 'diagnostic_manifest.json', manifest)
    model = v41 = None
    try:
        dataset = load_dataset(config)
        entries, records, qi, gi, split = fixed_train_split(dataset, list(config.DATASETS.MODALITIES),
            args.train_queries, args.gallery_per_identity, args.seed)
        positives, negatives = matched_pairs(records, args.max_pairs, args.seed)
        split.update(records=records, query_indices=qi, gallery_indices=gi,
                     positive_pairs=positives, negative_pairs=negatives)
        save_json(output / 'frequency_fixed_train_split.json', split)
        uad_cfg = config.clone(); uad_cfg.defrost(); uad_cfg.SCSR_V41.ENABLED = False
        uad_cfg.MODEL.PRETRAIN_CHOICE = 'none'; uad_cfg.MODEL.DIST_TRAIN = False; uad_cfg.freeze()
        model = make_model(uad_cfg, dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids)
        load_uad(model, args.uad_checkpoint); model.to(device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        profiles = final = None
        with no_parameter_mutation(model) as state_before:
            manifest['uad_state_before_sha256'] = state_before
            if requested & {1, 3, 4}:
                profiles, final = extract_profiles(model, entries, config, args)
                identity_rows, distributions = pair_rows(profiles, records, positives, negatives, final)
                write_csv(output / 'frequency_identity_discriminability.csv', identity_rows if 1 in requested else [dict(status='not_run')])
                write_csv(output / 'frequency_distance_distributions.csv', distributions if 1 in requested else [dict(status='not_run')])
                scene_positive, scene_negative = required_scene_pairs(records, args.seed)
                scene, camera = scene_rows(profiles, records, scene_positive, scene_negative)
                write_csv(output / 'frequency_scene_stability.csv', scene if 3 in requested else [dict(status='not_run')])
                write_csv(output / 'frequency_cross_modality_pairs.csv',
                          [r for r in scene if r['query_modality'] != r['gallery_modality']] if 3 in requested else [dict(status='not_run')])
                write_csv(output / 'frequency_camera_platform_statistics.csv', camera if 3 in requested else [dict(status='not_run')])
                write_csv(output / 'frequency_layer_profile.csv', identity_rows if 4 in requested else [dict(status='not_run')])
            alpha_lookup, calibration_rows = calibration(model, entries, config, args)
            write_csv(output / 'frequency_intervention_calibration.csv', calibration_rows)
            if 2 in requested:
                directions = [(b, s, None) for b in BANDS for s in (1, -1)]
                metrics, changes, margins, controls = intervention_suite(model, entries, records, qi, gi, config,
                    args, alpha_lookup, output, 7, directions, 'train_identity_selection')
                write_csv(output / 'frequency_intervention_metrics.csv', metrics)
                write_csv(output / 'frequency_intervention_query_changes.csv', changes)
                write_csv(output / 'frequency_intervention_margin.csv', margins)
                write_csv(output / 'frequency_intervention_controls.csv', controls)
            else:
                for name in ('frequency_intervention_metrics.csv', 'frequency_intervention_query_changes.csv',
                             'frequency_intervention_margin.csv', 'frequency_intervention_controls.csv'):
                    write_csv(output / name, [dict(status='not_run')])
            if 4 in requested:
                lm, lq, lmargin, lc = [], [], [], []
                for layer in LAYERS:
                    result = intervention_suite(model, entries, records, qi, gi, config, args, alpha_lookup, output,
                                                layer, parse_directions(args.layer_directions), 'train_layer_comparison')
                    lm += result[0]; lq += result[1]; lmargin += result[2]; lc += result[3]
                write_csv(output / 'frequency_layer_intervention.csv', lm)
                write_csv(output / 'frequency_layer_identity_sensitivity.csv', lmargin)
            else:
                write_csv(output / 'frequency_layer_intervention.csv', [dict(status='not_run')])
                write_csv(output / 'frequency_layer_identity_sensitivity.csv', [dict(status='not_run')])
            if 5 in requested:
                for p in model.parameters(): p.requires_grad_(True)
                try:
                    losses, alignments, sensitivities = gradient_probe(model, dataset, config, args)
                finally:
                    for p in model.parameters(): p.requires_grad_(False); p.grad = None
                write_csv(output / 'frequency_loss_gradient.csv', losses)
                write_csv(output / 'frequency_gradient_alignment.csv', alignments)
                write_csv(output / 'frequency_margin_sensitivity.csv', sensitivities)
            else:
                for name in ('frequency_loss_gradient.csv', 'frequency_gradient_alignment.csv', 'frequency_margin_sensitivity.csv'):
                    write_csv(output / name, [dict(status='not_run')])
            if 6 in requested:
                controls = parse_directions(args.control_directions)
                result = intervention_suite(model, entries, records, qi, gi, config, args, alpha_lookup, output,
                                            7, controls, 'frequency_vs_spatial')
                rows = ([dict(row, row_type='metric') for row in result[0]] +
                        [dict(row, row_type='query_change') for row in result[1]] +
                        [dict(row, row_type='margin_change') for row in result[2]])
                write_csv(output / 'frequency_vs_spatial_control.csv', rows)
            else:
                write_csv(output / 'frequency_vs_spatial_control.csv', [dict(status='not_run')])
            # Formal test confirmation is explicit and fixed before looking at these outputs.
            confirmations = parse_directions(args.confirm_interventions)
            if confirmations:
                test_queries = [e for name in config.DATASETS.MODALITIES for e in dataset.query.get(name, [])]
                test_galleries = [e for name in config.DATASETS.MODALITIES for e in dataset.gallery.get(name, [])]
                test_entries = test_queries + test_galleries
                test_records = [record(e, list(config.DATASETS.MODALITIES)) for e in test_entries]
                nq = len(test_queries)
                if (nq, len(test_entries) - nq) != (6405, 93609):
                    raise ValueError('Formal confirmation requires official 6405/93609 evaluation split')
                result = formal_confirmation(model, test_entries, test_records, nq, config, args,
                                             alpha_lookup, confirmations)
                write_csv(output / 'frequency_official_confirmation_metrics.csv', result[0])
                write_csv(output / 'frequency_official_confirmation_query_changes.csv', result[1])
                write_csv(output / 'frequency_official_confirmation_margin.csv', result[2])
            else:
                write_csv(output / 'frequency_official_confirmation_metrics.csv',
                          [dict(status='not_run', reason='no preselected --confirm-interventions')])
                write_csv(output / 'frequency_official_confirmation_query_changes.csv', [dict(status='not_run')])
                write_csv(output / 'frequency_official_confirmation_margin.csv', [dict(status='not_run')])
        if args.v41_checkpoint and 3 in requested:
            vcfg = config.clone(); vcfg.defrost(); vcfg.SCSR_V41.ENABLED = True
            vcfg.MODEL.PRETRAIN_CHOICE = 'none'; vcfg.MODEL.DIST_TRAIN = False; vcfg.freeze()
            v41 = make_model(vcfg, dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids)
            load_v41(v41, args.v41_checkpoint, include_classifier=True); v41.to(device).eval()
            for p in v41.parameters(): p.requires_grad_(False)
            with no_parameter_mutation(v41):
                vrows = run_v41_scene(v41, entries, records, config, args)
            manifest['checkpoints']['v41'] = dict(path=str(Path(args.v41_checkpoint).resolve()),
                                                  sha256=file_hash(args.v41_checkpoint))
        else:
            vrows = [dict(status='not_run', reason='experiment 3 or --v41-checkpoint absent')]
        write_csv(output / 'v41_condition_identity_scene_statistics.csv', vrows)
        experiment_outputs = {
            1: ['frequency_identity_discriminability.csv', 'frequency_distance_distributions.csv'],
            2: ['frequency_intervention_calibration.csv', 'frequency_intervention_metrics.csv',
                'frequency_intervention_query_changes.csv', 'frequency_intervention_margin.csv',
                'frequency_intervention_controls.csv', 'frequency_official_confirmation_metrics.csv',
                'frequency_official_confirmation_query_changes.csv', 'frequency_official_confirmation_margin.csv'],
            3: ['frequency_scene_stability.csv', 'frequency_cross_modality_pairs.csv',
                'frequency_camera_platform_statistics.csv', 'v41_condition_identity_scene_statistics.csv'],
            4: ['frequency_layer_profile.csv', 'frequency_layer_intervention.csv',
                'frequency_layer_identity_sensitivity.csv'],
            5: ['frequency_loss_gradient.csv', 'frequency_gradient_alignment.csv',
                'frequency_margin_sensitivity.csv'],
            6: ['frequency_vs_spatial_control.csv']}
        for number, names in experiment_outputs.items():
            save_json(output / 'frequency_experiment_{}_manifest.json'.format(number), dict(
                experiment=number, status='complete' if number in requested else 'not_run',
                outputs=names, uad_checkpoint_sha256=manifest['checkpoints']['uad']['sha256'],
                fixed_train_split_sha256=split['order_sha256'], seed=args.seed,
                precision='AMP' if args.amp else 'FP32', haar_precision='FP32',
                optimizer_step=False, automatic_test_selection=False,
                formal_test_confirmation=bool(confirmations) if number == 2 else 'not_applicable'))
        manifest.update(status='complete', fixed_train_split=split,
                        uad_state_after_sha256=tensor_sha256(model.state_dict()),
                        parameters_and_buffers_unchanged=tensor_sha256(model.state_dict()) == manifest['uad_state_before_sha256'],
                        parameter_grad_buffers_empty=all(p.grad is None for p in model.parameters()),
                        automatic_test_selection=False)
        if not manifest['parameters_and_buffers_unchanged'] or not manifest['parameter_grad_buffers_empty']:
            raise AssertionError('Diagnostics changed UAD state or parameter gradients')
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        if model is not None:
            manifest['uad_state_finally_sha256'] = tensor_sha256(model.state_dict())
            if 'uad_state_before_sha256' in manifest:
                manifest['finally_parameters_and_buffers_unchanged'] = manifest['uad_state_finally_sha256'] == manifest['uad_state_before_sha256']
        save_json(output / 'diagnostic_manifest.json', manifest)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config_file', default='configs/SCSR_V41.yml')
    p.add_argument('--uad-checkpoint', required=True)
    p.add_argument('--v41-checkpoint')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--experiments', default='1,2,3,4,5', help='Core 1..5 by default; add 6 for auxiliary controls')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--train-queries', type=int, default=64)
    p.add_argument('--gallery-per-identity', type=int, default=6)
    p.add_argument('--max-pairs', type=int, default=4096)
    p.add_argument('--calibration-images', type=int, default=64)
    p.add_argument('--budgets', type=float, nargs='+', default=(0.005, 0.01))
    p.add_argument('--hf-alpha-cap', type=float, default=.2)
    p.add_argument('--ll-alpha-cap', type=float, default=.1)
    p.add_argument('--control-seeds', type=int, nargs='+', default=(101, 211, 307))
    p.add_argument('--layer-directions', default='LH:+:0.005,HL:+:0.005,HH:+:0.005')
    p.add_argument('--control-directions', default='LH:+:0.005,HL:+:0.005,HH:+:0.005')
    p.add_argument('--confirm-interventions', default='',
                   help='Preselected after train diagnostics, e.g. LH:+:0.01,HH:-:0.005; never auto-selected')
    p.add_argument('--gradient-batches', type=int, choices=(2, 3, 4), default=2)
    p.add_argument('--gradient-identities', type=int, default=4)
    p.add_argument('--gradient-alpha', type=float, default=.01)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('opts', nargs=argparse.REMAINDER)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    config = cfg.clone(); config.merge_from_file(args.config_file); config.merge_from_list(args.opts)
    config.defrost(); config.MODEL.PRETRAIN_CHOICE = 'none'; config.MODEL.DIST_TRAIN = False; config.freeze()
    run(args, config)

