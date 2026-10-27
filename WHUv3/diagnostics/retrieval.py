"""Streaming ON/OFF ranking, using the existing distance and camera protocol."""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import iter_euclidean_distance_chunks
from .core import amp_context, diagnostic_session, difference_stats


def paired_features(model, loader, records, device, use_amp=False, progress=None):
    features = {'ON': [], 'OFF': []}
    for batch, (images, indices) in enumerate(loader):
        images = images.to(device)
        mods = torch.tensor([records[int(i)]['modal_id'] for i in indices], dtype=torch.long, device=device)
        with torch.no_grad(), amp_context(device, use_amp):
            for state in ('ON', 'OFF'):
                with diagnostic_session(model, state == 'ON'):
                    # Same tensor, same model/BN/checkpoint, sensor IDs distinct from mode.
                    output = model(images, mode=1, modal_ids=mods)
                if not output.isfinite().all():
                    raise ValueError('Nonfinite retrieval features in ' + state)
                features[state].append(output.detach().float().cpu())
        if progress is not None and (batch + 1) % 20 == 0:
            progress('Feature extraction batch {}'.format(batch + 1))
    return {state: F.normalize(torch.cat(parts), dim=1) for state, parts in features.items()}


def query_metrics(distances, query, gallery_records, gallery_arrays=None, ordered=None):
    if gallery_arrays is None:
        gallery_arrays = (np.asarray([row['camid'] for row in gallery_records]),
                          np.asarray([row['pid'] for row in gallery_records]))
    cams, pids = gallery_arrays
    # Match compute_indices_chunked: torch.argsort on CPU FP32, full gallery.
    if ordered is None:
        ordered = torch.argsort(distances, dim=0).cpu().numpy()
    ordered = ordered[cams[ordered] != query['camid']]
    positives = pids[ordered] == query['pid']
    relevant = np.flatnonzero(positives)
    if relevant.size == 0:
        return None
    ranks = relevant + 1
    cumulative = np.arange(1, len(ranks) + 1, dtype=np.float64)
    top = int(ordered[0])
    result = dict(rank1=float(ranks[0] <= 1), rank5=float(ranks[0] <= 5),
                  rank10=float(ranks[0] <= 10), best_positive_rank=int(ranks[0]),
                  AP=float(np.mean(cumulative / ranks)), INP=float(len(ranks) / ranks[-1]),
                  top_gallery_index=top, top_gallery_pid=int(pids[top]),
                  top_gallery_path=gallery_records[top]['path'])
    result['ranking_sha256'] = hashlib.sha256(ordered.astype(np.int64).tobytes()).hexdigest()
    return result


def protocols(queries, galleries):
    def indices(records, predicate):
        return np.asarray([i for i, record in enumerate(records) if predicate(record)], dtype=np.int64)
    result = [('overall', 'ALL', 'ALL', np.arange(len(queries)), np.arange(len(galleries)))]
    modalities = sorted(set(row['modal_id'] for row in queries) | set(row['modal_id'] for row in galleries))
    names = {row['modal_id']: row['modality_name'] for row in queries + galleries}
    for qm in modalities:
        for gm in modalities:
            result.append(('modality_pair', names[qm], names[gm],
                           indices(queries, lambda r, qm=qm: r['modal_id'] == qm),
                           indices(galleries, lambda r, gm=gm: r['modal_id'] == gm)))
    for qp in ('G', 'A'):
        for gp in ('G', 'A'):
            result.append(('tir_platform', 'TIR-' + qp, 'TIR-' + gp,
                           indices(queries, lambda r, qp=qp: r['modality_name'] == 'TIR' and r['platform'] == qp),
                           indices(galleries, lambda r, gp=gp: r['modality_name'] == 'TIR' and r['platform'] == gp)))
    return result


def evaluate_on_off(qfeatures, gfeatures, queries, galleries, chunk_size=64, progress=None):
    metrics, comparisons = [], []
    categories = dict(both_correct=0, on_correct_off_wrong=0, on_wrong_off_correct=0, both_wrong=0)
    summary = dict(valid_queries=0, invalid_queries=0, ranking_changed=0,
                   rank1_gallery_changed=0, rank1_identity_changed=0)
    for protocol, qdomain, gdomain, qi, gi in protocols(queries, galleries):
        gallery = [galleries[int(i)] for i in gi]
        arrays = (np.asarray([row['camid'] for row in gallery]),
                  np.asarray([row['pid'] for row in gallery]))
        sums = {state: dict(rank1=0., rank5=0., rank10=0., AP=0., INP=0.) for state in ('ON', 'OFF')}
        valid = 0
        if len(qi) and len(gi):
            chunks = {state: iter_euclidean_distance_chunks(
                qfeatures[state].index_select(0, torch.from_numpy(qi).long()),
                gfeatures[state].index_select(0, torch.from_numpy(gi).long()), chunk_size)
                      for state in ('ON', 'OFF')}
            offset = 0
            for distances_on, distances_off in zip(chunks['ON'], chunks['OFF']):
                orders_on = torch.argsort(distances_on, dim=1).cpu().numpy()
                orders_off = torch.argsort(distances_off, dim=1).cpu().numpy()
                for local in range(distances_on.shape[0]):
                    qidx = int(qi[offset + local])
                    query = queries[qidx]
                    on = query_metrics(distances_on[local], query, gallery, arrays, orders_on[local])
                    off = query_metrics(distances_off[local], query, gallery, arrays, orders_off[local])
                    if (on is None) != (off is None):
                        raise AssertionError('ON/OFF valid query sets differ')
                    if on is not None:
                        valid += 1
                        for state, stats in (('ON', on), ('OFF', off)):
                            for key in sums[state]:
                                sums[state][key] += stats[key]
                    if protocol == 'overall':
                        row = dict(query_index=qidx, **query, valid=on is not None)
                        row.update(difference_stats(qfeatures['ON'][qidx], qfeatures['OFF'][qidx]))
                        if on is None:
                            summary['invalid_queries'] += 1
                        else:
                            summary['valid_queries'] += 1
                            for state, stats in (('on', on), ('off', off)):
                                row.update({state + '_' + key: value for key, value in stats.items()})
                            row['ranking_changed'] = on['ranking_sha256'] != off['ranking_sha256']
                            row['rank1_gallery_changed'] = on['top_gallery_index'] != off['top_gallery_index']
                            row['rank1_identity_changed'] = on['top_gallery_pid'] != off['top_gallery_pid']
                            for key in ('ranking_changed', 'rank1_gallery_changed', 'rank1_identity_changed'):
                                summary[key] += int(row[key])
                            category = ('both_correct' if on['rank1'] and off['rank1'] else
                                        'on_correct_off_wrong' if on['rank1'] else
                                        'on_wrong_off_correct' if off['rank1'] else 'both_wrong')
                            row['correctness_category'] = category
                            categories[category] += 1
                        comparisons.append(row)
                offset += distances_on.shape[0]
                if progress and protocol == 'overall':
                    progress('Overall ON/OFF ranking queries {}/{}'.format(offset, len(qi)))
        for state in ('ON', 'OFF'):
            row = dict(state=state, protocol=protocol, query_domain=qdomain, gallery_domain=gdomain,
                       query_count=len(qi), gallery_count=len(gi), valid_query_count=valid,
                       status='ok' if valid else 'empty_or_no_valid_query')
            row.update({output: sums[state][key] / valid if valid else ''
                        for output, key in (('mAP', 'AP'), ('Rank-1', 'rank1'), ('Rank-5', 'rank5'),
                                            ('Rank-10', 'rank10'), ('mINP', 'INP'))})
            metrics.append(row)
        if progress:
            progress('{} {} -> {}: valid queries {}'.format(protocol, qdomain, gdomain, valid))
    summary.update(categories)
    return metrics, comparisons, summary
