"""Deterministic train-identity split and condition-matched frequency pairs."""
import hashlib
import json
import random
from collections import defaultdict


def record(entry, modalities):
    path, pid, camid, modal_id = entry
    name = {'IR': 'NIR', 'Thermal': 'TIR'}.get(modalities[modal_id - 1], modalities[modal_id - 1])
    return dict(path=path, pid=int(pid), camid=int(camid), modal_id=int(modal_id),
                modality=name, platform='G' if camid < 5 else 'A')


def digest(items):
    h = hashlib.sha256()
    for item in items:
        h.update((json.dumps(item, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
    return h.hexdigest()


def fixed_train_split(dataset, modalities, max_queries=384, gallery_per_identity=12, seed=1234):
    entries = [entry for name in modalities for entry in dataset.train.get(name, [])]
    rows = [record(entry, modalities) for entry in entries]
    by_pid = defaultdict(list)
    for index, row in enumerate(rows):
        by_pid[row['pid']].append(index)
    eligible = []
    for pid in sorted(by_pid):
        pool = by_pid[pid]
        for index in pool:
            if any(rows[j]['camid'] != rows[index]['camid'] for j in pool):
                eligible.append(index); break
    rng = random.Random(seed); rng.shuffle(eligible)
    query_indices = sorted(eligible[:min(max_queries, len(eligible))])
    gallery_indices = []
    for qi in query_indices:
        pool = list(by_pid[rows[qi]['pid']]); rng.shuffle(pool)
        positives = [j for j in pool if rows[j]['camid'] != rows[qi]['camid']]
        others = [j for j in pool if j not in positives and j != qi]
        ordered = positives + others
        stratified = []
        for modality in ('RGB', 'NIR', 'TIR'):
            for platform in ('G', 'A'):
                candidate = next((j for j in ordered if rows[j]['modality'] == modality and
                                  rows[j]['platform'] == platform and j not in stratified), None)
                if candidate is not None:
                    stratified.append(candidate)
        stratified.extend(j for j in ordered if j not in stratified)
        gallery_indices.extend(stratified[:gallery_per_identity])
    # Add one deterministic condition-matched negative identity per query.
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[(row['modality'], row['camid'], row['platform'])].append(index)
    for qi in query_indices:
        q = rows[qi]
        candidates = [j for j in buckets[(q['modality'], q['camid'], q['platform'])] if rows[j]['pid'] != q['pid']]
        if candidates:
            gallery_indices.append(candidates[rng.randrange(len(candidates))])
    gallery_indices = sorted(set(gallery_indices) - set(query_indices))
    if not query_indices or not gallery_indices:
        raise ValueError('Could not build the fixed train-identity diagnostic split')
    # Every query must retain a different-camera positive in Gallery.
    for qi in query_indices:
        if not any(rows[gi]['pid'] == rows[qi]['pid'] and rows[gi]['camid'] != rows[qi]['camid'] for gi in gallery_indices):
            raise AssertionError('Fixed train split contains an invalid Query')
    selected = sorted(set(query_indices + gallery_indices))
    remap = {old: new for new, old in enumerate(selected)}
    selected_rows = [rows[i] for i in selected]
    return ([entries[i] for i in selected], selected_rows,
            [remap[i] for i in query_indices], [remap[i] for i in gallery_indices],
            dict(seed=seed, query_count=len(query_indices), gallery_count=len(gallery_indices),
                 entry_count=len(selected), identity_count=len({rows[i]['pid'] for i in selected}),
                 order_sha256=digest(selected_rows)))


def matched_pairs(records, max_pairs=4096, seed=1234):
    by_pid, by_condition = defaultdict(list), defaultdict(list)
    for index, row in enumerate(records):
        by_pid[row['pid']].append(index)
        by_condition[(row['modality'], row['camid'], row['platform'])].append(index)
    rng = random.Random(seed); positives, negatives = [], []
    anchors = list(range(len(records))); rng.shuffle(anchors)
    for anchor in anchors:
        row = records[anchor]
        same = [i for i in by_pid[row['pid']] if i != anchor]
        different_camera = [i for i in same if records[i]['camid'] != row['camid']]
        if not same:
            continue
        positive = rng.choice(different_camera or same)
        positive_row = records[positive]
        negative_pool = [i for i in by_condition[(positive_row['modality'], positive_row['camid'], positive_row['platform'])]
                         if records[i]['pid'] != row['pid']]
        if not negative_pool:
            negative_pool = [i for i, other in enumerate(records)
                             if other['pid'] != row['pid'] and other['modality'] == positive_row['modality']
                             and other['platform'] == positive_row['platform']]
        if not negative_pool:
            continue
        positives.append((anchor, positive)); negatives.append((anchor, rng.choice(negative_pool)))
        if len(positives) >= max_pairs:
            break
    if not positives:
        raise ValueError('No valid matched identity pairs')
    return positives, negatives


def relation(a, b):
    if a['modality'] != b['modality']:
        return '{}->{}'.format(a['modality'], b['modality'])
    if a['platform'] != b['platform']:
        return 'cross_platform'
    if a['camid'] != b['camid']:
        return 'same_platform_cross_camera'
    return 'same_camera'


def relation_pairs(records, positive_pairs, negative_pairs):
    groups = defaultdict(lambda: {'positive': [], 'negative': []})
    if len(positive_pairs) != len(negative_pairs):
        raise ValueError('Positive and matched-negative pair lists must align')
    for positive, negative in zip(positive_pairs, negative_pairs):
        if positive[0] != negative[0]:
            raise ValueError('Each negative control must share its positive anchor')
        a, b = records[positive[0]], records[positive[1]]
        key = (a['modality'], b['modality'], relation(a, b))
        groups[key]['positive'].append(positive)
        groups[key]['negative'].append(negative)
    return groups


def required_scene_pairs(records, seed=1234):
    """Construct only real same-ID pairs for the four requested scene relations."""
    by_pid = defaultdict(list)
    for index, row in enumerate(records):
        by_pid[row['pid']].append(index)
    rng = random.Random(seed); positives = []
    for pid in sorted(by_pid):
        pool = by_pid[pid]
        tir = [i for i in pool if records[i]['modality'] == 'TIR']
        for platform in ('G', 'A'):
            candidates = [i for i in tir if records[i]['platform'] == platform]
            pair = next(((a, b) for a in candidates for b in candidates if records[a]['camid'] != records[b]['camid']), None)
            if pair is not None: positives.append(pair)
        pair = next(((a, b) for a in tir for b in tir if records[a]['platform'] != records[b]['platform']), None)
        if pair is not None: positives.append(pair)
        for modality in ('RGB', 'NIR'):
            left = [i for i in pool if records[i]['modality'] == modality]
            pair = next(((a, b) for a in left for b in tir), None)
            if pair is not None: positives.append(pair)
    negatives = []
    for anchor, positive in positives:
        target = records[positive]
        candidates = [i for i, row in enumerate(records) if row['pid'] != records[anchor]['pid']
                      and row['modality'] == target['modality'] and row['camid'] == target['camid']]
        if not candidates:
            candidates = [i for i, row in enumerate(records) if row['pid'] != records[anchor]['pid']
                          and row['modality'] == target['modality'] and row['platform'] == target['platform']]
        if not candidates:
            raise ValueError('Cannot form condition-matched negative for a required scene pair')
        negatives.append((anchor, candidates[rng.randrange(len(candidates))]))
    return positives, negatives
