"""Four feature combinations and identity margins, with full-gallery ranking."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.metrics import iter_euclidean_distance_chunks
from .retrieval import query_metrics


COMBINATIONS = {'A': ('OFF', 'OFF'), 'B': ('ON', 'ON'), 'C': ('ON', 'OFF'), 'D': ('OFF', 'ON')}
TRANSITIONS = ('correct_to_correct', 'correct_to_wrong', 'wrong_to_correct', 'wrong_to_wrong')


class CSVStream:
    def __init__(self, path, fields=None):
        self.path, self.fields = Path(path), fields
        self.stream = self.path.open('w', encoding='utf-8', newline='')
        self.writer = None
        if fields is not None:
            self.writer = csv.DictWriter(self.stream, fieldnames=fields)
            self.writer.writeheader()
        self.count = 0

    def append(self, row):
        if self.writer is None:
            self.fields = list(row)
            self.writer = csv.DictWriter(self.stream, fieldnames=self.fields)
            self.writer.writeheader()
        if set(row) - set(self.fields):
            raise ValueError('CSV schema changed: ' + ','.join(set(row) - set(self.fields)))
        self.writer.writerow({key: format(float(value), '.12e') if isinstance(value, (float, np.floating)) else
                              '' if value is None else value for key, value in row.items()})
        self.count += 1

    def close(self):
        self.stream.close()


def full_protocols(queries, galleries):
    result = [('overall', 'ALL', 'ALL', np.arange(len(queries)), np.arange(len(galleries)))]
    names = {1: 'RGB', 2: 'NIR', 3: 'TIR'}
    def indices(records, modality, platform=None):
        return np.asarray([i for i, record in enumerate(records) if record['modal_id'] == modality and
                           (platform is None or record['platform'] == platform)], dtype=np.int64)
    for qm in (1, 2, 3):
        for gm in (1, 2, 3):
            result.append(('modality_pair', names[qm], names[gm], indices(queries, qm), indices(galleries, gm)))
    for qm in (1, 2, 3):
        for qp in ('G', 'A'):
            for gm in (1, 2, 3):
                for gp in ('G', 'A'):
                    result.append(('modality_platform', names[qm] + '-' + qp, names[gm] + '-' + gp,
                                   indices(queries, qm, qp), indices(galleries, gm, gp)))
    return result


def transition(a, b):
    return ('correct_to_correct' if a and b else 'correct_to_wrong' if a else
            'wrong_to_correct' if b else 'wrong_to_wrong')


IDENTITY_FIELDS = ['protocol', 'query_domain', 'gallery_domain', 'query_index', 'query_path', 'pid', 'camid',
                   'modality', 'platform', 'gallery_count', 'status', 'transition', 'distance_kind',
                   'delta_margin', 'delta_AP', 'delta_top10_correct', 'first_positive_rank_delta',
                   'rank1_gallery_changed', 'rank1_identity_changed', 'positive_neighbor_changed', 'negative_neighbor_changed']
for _state in ('A', 'B'):
    IDENTITY_FIELDS += [key + '_' + _state for key in ('d_pos', 'd_neg', 'margin', 'top10_correct', 'first_positive_rank', 'AP', 'rank1_correct')]
    for _kind in ('pos', 'neg'):
        IDENTITY_FIELDS += ['nearest_{}_{}_{}'.format(_kind, _state, key) for key in
                            ('gallery_index', 'pid', 'camid', 'modality', 'platform', 'path', 'distance', 'rank')]
        IDENTITY_FIELDS += ['anchor_{}_{}_{}'.format(_state, _kind, key) for key in
                            ('distance_A', 'distance_B', 'rank_A', 'rank_B', 'delta_distance')]


QUERY_FIELDS = ['query_index', 'query_path', 'pid', 'camid', 'modality', 'platform', 'valid', 'status',
                'rank1_pid_A', 'rank1_pid_B', 'rank1_gallery_index_A', 'rank1_gallery_index_B',
                'rank1_gallery_path_A', 'rank1_gallery_path_B', 'rank1_correct_A', 'rank1_correct_B',
                'first_positive_rank_A', 'first_positive_rank_B', 'AP_A', 'AP_B', 'delta_AP',
                'rank1_changed', 'rank1_identity_changed', 'ranking_changed', 'first_positive_rank_delta', 'transition']


def identity_row(query_index, query, gallery, global_gallery_indices, distances, ordered, stats, protocol, qdomain, gdomain):
    row = dict.fromkeys(IDENTITY_FIELDS)
    row.update(protocol=protocol, query_domain=qdomain, gallery_domain=gdomain, query_index=query_index,
               query_path=query['path'], pid=query['pid'], camid=query['camid'], modality=query['modality_name'],
               platform=query['platform'], gallery_count=len(gallery), distance_kind='squared_euclidean_of_L2_features',
               status='ok' if stats['A'] is not None else 'no_positive_after_camera_filter')
    arrays = {key: np.asarray([r[key] for r in gallery]) for key in ('camid', 'pid')}
    filtered = {state: order[arrays['camid'][order] != query['camid']] for state, order in ordered.items()}
    inverses, selected = {}, {}
    for state in ('A', 'B'):
        order = filtered[state]
        inverse = np.zeros(len(gallery), dtype=np.int32)
        inverse[order] = np.arange(1, len(order) + 1)
        inverses[state] = inverse
        positive = arrays['pid'][order] == query['pid']
        selected[state] = {'pos': int(order[np.flatnonzero(positive)[0]]) if positive.any() else None,
                           'neg': int(order[np.flatnonzero(~positive)[0]]) if (~positive).any() else None}
        row['top10_correct_' + state] = int(positive[:10].sum())
        if stats[state] is not None:
            row.update({key + '_' + state: value for key, value in
                        (('first_positive_rank', stats[state]['best_positive_rank']), ('AP', stats[state]['AP']),
                         ('rank1_correct', bool(stats[state]['rank1'])))})
        for kind, index in selected[state].items():
            if index is None:
                continue
            record = gallery[index]
            values = dict(gallery_index=int(global_gallery_indices[index]), pid=record['pid'], camid=record['camid'],
                          modality=record['modality_name'], platform=record['platform'], path=record['path'],
                          distance=float(distances[state][index]), rank=int(inverse[index]))
            row.update({'nearest_{}_{}_{}'.format(kind, state, key): value for key, value in values.items()})
            row['d_' + kind + '_' + state] = values['distance']
        if row['d_pos_' + state] is not None and row['d_neg_' + state] is not None:
            row['margin_' + state] = row['d_neg_' + state] - row['d_pos_' + state]
    for state in ('A', 'B'):
        for kind, index in selected[state].items():
            if index is not None:
                da, db = float(distances['A'][index]), float(distances['B'][index])
                row.update({'anchor_{}_{}_{}'.format(state, kind, key): value for key, value in
                            (('distance_A', da), ('distance_B', db), ('rank_A', int(inverses['A'][index])),
                             ('rank_B', int(inverses['B'][index])), ('delta_distance', db - da))})
    if stats['A'] is not None:
        a, b = stats['A'], stats['B']
        row.update(transition=transition(a['rank1'], b['rank1']), delta_AP=b['AP'] - a['AP'],
                   first_positive_rank_delta=b['best_positive_rank'] - a['best_positive_rank'],
                   delta_top10_correct=row['top10_correct_B'] - row['top10_correct_A'],
                   rank1_gallery_changed=a['top_gallery_index'] != b['top_gallery_index'],
                   rank1_identity_changed=a['top_gallery_pid'] != b['top_gallery_pid'],
                   positive_neighbor_changed=selected['A']['pos'] != selected['B']['pos'],
                   negative_neighbor_changed=selected['A']['neg'] != selected['B']['neg'])
        if row['margin_A'] is not None and row['margin_B'] is not None:
            row['delta_margin'] = row['margin_B'] - row['margin_A']
    return row


def evaluate_four_combinations(qfeatures, gfeatures, queries, galleries, output, chunk_size=32,
                               include_identity=True, progress=None):
    output = Path(output)
    metrics, platform_metrics, summaries, representatives = [], [], [], []
    query_writer = CSVStream(output / 'v41_on_off_query_comparison.csv', QUERY_FIELDS)
    identity_writer = CSVStream(output / 'v41_identity_distance_changes.csv', IDENTITY_FIELDS) if include_identity else None
    negative_writer = CSVStream(output / 'v41_hard_negative_transitions.csv', IDENTITY_FIELDS) if include_identity else None
    overall = dict(valid_queries=0, invalid_queries=0, rank1_gallery_changed=0,
                   rank1_identity_changed=0, ranking_changed=0, **{key: 0 for key in TRANSITIONS})
    try:
        for protocol, qdomain, gdomain, qi, gi in full_protocols(queries, galleries):
            gallery = [galleries[int(index)] for index in gi]
            arrays = (np.asarray([r['camid'] for r in gallery]), np.asarray([r['pid'] for r in gallery]))
            sums = {state: {key: 0. for key in ('rank1', 'rank5', 'rank10', 'AP', 'INP')} for state in COMBINATIONS}
            transitions = {key: [] for key in TRANSITIONS}
            valid = 0
            if len(qi) and len(gi):
                qindex, gindex = torch.from_numpy(qi).long(), torch.from_numpy(gi).long()
                q_has_tir = any(queries[int(i)]['modal_id'] == 3 for i in qi)
                g_has_tir = any(galleries[int(i)]['modal_id'] == 3 for i in gi)
                keys = {state: (qs if q_has_tir else 'OFF', gs if g_has_tir else 'OFF')
                        for state, (qs, gs) in COMBINATIONS.items()}
                query_subsets = {state: qfeatures[state].index_select(0, qindex) for state in {key[0] for key in keys.values()}}
                gallery_subsets = {state: gfeatures[state].index_select(0, gindex) for state in {key[1] for key in keys.values()}}
                chunks = {key: iter_euclidean_distance_chunks(query_subsets[key[0]], gallery_subsets[key[1]], chunk_size)
                          for key in dict.fromkeys(keys.values())}
                offset = 0
                for packed in zip(*chunks.values()):
                    blocks = dict(zip(chunks, packed))
                    orders = {key: torch.argsort(value, dim=1).cpu().numpy() for key, value in blocks.items()}
                    for local in range(packed[0].shape[0]):
                        index = int(qi[offset + local]); query = queries[index]
                        stats = {state: query_metrics(blocks[key][local], query, gallery, arrays, orders[key][local])
                                 for state, key in keys.items()}
                        if len({value is not None for value in stats.values()}) != 1:
                            raise AssertionError('Feature interventions changed valid Query identities')
                        if stats['A'] is not None:
                            valid += 1
                            for state, values in stats.items():
                                for key in sums[state]:
                                    sums[state][key] += values[key]
                        if protocol == 'overall':
                            row = dict.fromkeys(QUERY_FIELDS)
                            row.update(query_index=index, query_path=query['path'], pid=query['pid'], camid=query['camid'],
                                       modality=query['modality_name'], platform=query['platform'], valid=stats['A'] is not None,
                                       status='ok' if stats['A'] is not None else 'no_positive_after_camera_filter')
                            if stats['A'] is None:
                                overall['invalid_queries'] += 1
                            else:
                                a, b = stats['A'], stats['B']; overall['valid_queries'] += 1
                                for state, values in (('A', a), ('B', b)):
                                    row.update({key + '_' + state: value for key, value in
                                                (('rank1_pid', values['top_gallery_pid']),
                                                 ('rank1_gallery_index', int(gi[values['top_gallery_index']])),
                                                 ('rank1_gallery_path', values['top_gallery_path']), ('rank1_correct', bool(values['rank1'])),
                                                 ('first_positive_rank', values['best_positive_rank']), ('AP', values['AP']))})
                                row.update(delta_AP=b['AP'] - a['AP'], rank1_changed=a['top_gallery_index'] != b['top_gallery_index'],
                                           rank1_identity_changed=a['top_gallery_pid'] != b['top_gallery_pid'],
                                           ranking_changed=a['ranking_sha256'] != b['ranking_sha256'],
                                           first_positive_rank_delta=b['best_positive_rank'] - a['best_positive_rank'],
                                           transition=transition(a['rank1'], b['rank1']))
                                overall[row['transition']] += 1
                                overall['rank1_gallery_changed'] += int(row['rank1_changed'])
                                overall['rank1_identity_changed'] += int(row['rank1_identity_changed'])
                                overall['ranking_changed'] += int(row['ranking_changed'])
                            query_writer.append(row)
                        if include_identity:
                            row = identity_row(index, query, gallery, gi,
                                               {state: blocks[keys[state]][local] for state in ('A', 'B')},
                                               {state: orders[keys[state]][local] for state in ('A', 'B')}, stats,
                                               protocol, qdomain, gdomain)
                            identity_writer.append(row)
                            if row['status'] == 'ok':
                                transitions[row['transition']].append({key: row[key] for key in
                                                                      ('delta_margin', 'delta_AP', 'delta_top10_correct',
                                                                       'first_positive_rank_delta', 'negative_neighbor_changed')})
                                if row['negative_neighbor_changed'] or row['rank1_gallery_changed']:
                                    negative_writer.append(row)
                                if len([r for r in representatives if r['protocol'] == protocol and r['query_domain'] == qdomain
                                        and r['gallery_domain'] == gdomain and r['transition'] == row['transition']]) < 3:
                                    representatives.append(row)
                    offset += packed[0].shape[0]
                    if progress and protocol == 'overall':
                        progress('Four-combination full ranking: {}/{} Queries'.format(offset, len(qi)))
                if offset != len(qi):
                    raise AssertionError('Distance iterator did not cover all selected Queries')
            for state in COMBINATIONS:
                row = dict(group=state, query_feature=COMBINATIONS[state][0], gallery_feature=COMBINATIONS[state][1],
                           protocol=protocol, query_domain=qdomain, gallery_domain=gdomain,
                           query_count=len(qi), gallery_count=len(gi), valid_query_count=valid,
                           status='ok' if valid else 'empty_or_no_valid_query')
                row.update({output_key: sums[state][key] / valid if valid else '' for output_key, key in
                            (('mAP', 'AP'), ('Rank-1', 'rank1'), ('Rank-5', 'rank5'), ('Rank-10', 'rank10'), ('mINP', 'INP'))})
                (platform_metrics if protocol == 'modality_platform' else metrics).append(row)
            if include_identity:
                for category, values in transitions.items():
                    row = dict(protocol=protocol, query_domain=qdomain, gallery_domain=gdomain, transition=category,
                               query_count=len(qi), valid_query_count=valid, transition_count=len(values))
                    for key in ('delta_margin', 'delta_AP', 'delta_top10_correct', 'first_positive_rank_delta', 'negative_neighbor_changed'):
                        finite = [float(value[key]) for value in values if value[key] is not None]
                        row[key + '_mean'] = float(np.mean(finite)) if finite else ''
                    summaries.append(row)
            if progress:
                progress('{} {} -> {}: valid Queries {}'.format(protocol, qdomain, gdomain, valid))
    finally:
        query_writer.close()
        if identity_writer:
            identity_writer.close(); negative_writer.close()
    from .core import write_csv
    write_csv(output / 'v41_on_off_metrics.csv', metrics)
    write_csv(output / 'v41_on_off_modality_platform.csv', platform_metrics)
    if include_identity:
        write_csv(output / 'v41_query_transition_summary.csv', summaries)
        write_csv(output / 'v41_tir_platform_summary.csv', [row for row in summaries if row['protocol'] == 'modality_platform'
                                                         and ('TIR' in row['query_domain'] or 'TIR' in row['gallery_domain'])])
        (output / 'v41_representative_cases.json').write_text(json.dumps(dict(
            selection='first three valid Queries by fixed query_index in every protocol/transition, not best performance',
            cases=representatives), indent=2, ensure_ascii=False), encoding='utf-8')
    return overall
