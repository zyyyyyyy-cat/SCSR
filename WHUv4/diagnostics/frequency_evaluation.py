"""Retrieval and pair statistics shared by frequency diagnostics."""
import numpy as np
import torch
import torch.nn.functional as F

from .core import write_csv
from .retrieval import iter_euclidean_distance_chunks
from .v41_ranking import query_metrics


def evaluate_features(features, records, query_indices, gallery_indices, group, scope, chunk_size=16):
    q = features[torch.tensor(query_indices, dtype=torch.long)]
    g = features[torch.tensor(gallery_indices, dtype=torch.long)]
    gallery = [records[i] for i in gallery_indices]
    arrays = (np.asarray([row['camid'] for row in gallery]), np.asarray([row['pid'] for row in gallery]))
    sums = dict(AP=0., rank1=0., rank5=0., rank10=0., INP=0.)
    valid, query_rows, margin_rows = 0, [], []
    offset = 0
    for distances in iter_euclidean_distance_chunks(q, g, chunk_size):
        orders = torch.argsort(distances, dim=1).numpy()
        for local in range(len(distances)):
            index = query_indices[offset + local]
            stats = query_metrics(distances[local], records[index], gallery, arrays, orders[local])
            row = dict(scope=scope, group=group, query_index=index, query_path=records[index]['path'],
                       pid=records[index]['pid'], camid=records[index]['camid'], modality=records[index]['modality'],
                       platform=records[index]['platform'], valid=stats is not None)
            if stats is None:
                row['status'] = 'no_positive_after_camera_filter'
            else:
                valid += 1; row['status'] = 'ok'
                for key in sums:
                    sums[key] += stats[key]
                row.update(AP=stats['AP'], rank1_correct=bool(stats['rank1']), rank5=stats['rank5'], rank10=stats['rank10'],
                           first_positive_rank=stats['best_positive_rank'], rank1_gallery_index=gallery_indices[stats['top_gallery_index']],
                           rank1_gallery_path=stats['top_gallery_path'], ranking_sha256=stats['ranking_sha256'])
                same = np.asarray([x['pid'] == records[index]['pid'] and x['camid'] != records[index]['camid'] for x in gallery])
                different = np.asarray([x['pid'] != records[index]['pid'] and x['camid'] != records[index]['camid']
                                        for x in gallery])
                positive = int(torch.argmin(distances[local].masked_fill(torch.from_numpy(~same), float('inf'))))
                negative = int(torch.argmin(distances[local].masked_fill(torch.from_numpy(~different), float('inf'))))
                margin_rows.append(dict(scope=scope, group=group, query_index=index,
                    positive_gallery_index=gallery_indices[positive], negative_gallery_index=gallery_indices[negative],
                    positive_distance=float(distances[local, positive]), negative_distance=float(distances[local, negative]),
                    margin=float(distances[local, negative] - distances[local, positive])))
            query_rows.append(row)
        offset += len(distances)
    if offset != len(query_indices):
        raise AssertionError('Chunked retrieval did not cover every Query')
    metric = dict(scope=scope, group=group, query_count=len(query_indices), gallery_count=len(gallery_indices),
                  valid_query_count=valid, status='ok' if valid else 'no_valid_query')
    for output, key in (('mAP', 'AP'), ('Rank-1', 'rank1'), ('Rank-5', 'rank5'), ('Rank-10', 'rank10'), ('mINP', 'INP')):
        metric[output] = sums[key] / valid if valid else ''
    return metric, query_rows, margin_rows


def compare_queries(baseline, changed):
    base = {row['query_index']: row for row in baseline}
    rows = []
    for row in changed:
        old = base[row['query_index']]
        result = dict(row)
        if row['valid'] and old['valid']:
            result.update(delta_AP=row['AP'] - old['AP'],
                          first_positive_rank_delta=row['first_positive_rank'] - old['first_positive_rank'],
                          rank1_transition='{}->{}'.format(int(old['rank1_correct']), int(row['rank1_correct'])),
                          rank1_gallery_changed=old['rank1_gallery_index'] != row['rank1_gallery_index'])
        rows.append(result)
    return rows


def compare_margins(baseline, changed):
    base = {row['query_index']: row for row in baseline}
    rows = []
    for row in changed:
        old = base.get(row['query_index'])
        if old is not None:
            rows.append(dict(row, baseline_positive_distance=old['positive_distance'],
                             baseline_negative_distance=old['negative_distance'], baseline_margin=old['margin'],
                             delta_margin=row['margin'] - old['margin']))
    return rows


def normalized_features(value):
    return F.normalize(value.detach().float().cpu(), dim=1)
