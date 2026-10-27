"""Fixed WHU-MARS data selections without training augmentation or DDP."""
import hashlib
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from datasets.bases import read_image
from datasets.whu_mars import WHU_MARS


def metadata(entry, modalities):
    path, pid, camid, modal_id = entry
    name = {'IR': 'NIR', 'Thermal': 'TIR'}.get(modalities[modal_id - 1], modalities[modal_id - 1])
    return dict(path=path, pid=int(pid), camid=int(camid), modal_id=int(modal_id),
                modality_name=name, platform='G' if camid < 5 else 'A')


class FixedImages(Dataset):
    def __init__(self, entries, transform):
        self.entries, self.transform = list(entries), transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.transform(read_image(self.entries[index][0])), index


def validation_transform(cfg):
    # Exactly the val_transforms operations in make_dataloader.py.
    return T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                      T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])


def load_dataset(cfg):
    if cfg.DATASETS.NAMES != 'WHU-MARS':
        raise ValueError('This diagnostic entrypoint implements the WHU-MARS protocol only')
    if cfg.DATASETS.DATASET_DIR != 'WHU-MARS' or cfg.DATASETS.FILTER_GD:
        raise ValueError('The four requested experiments require full WHU-MARS-1000, not GD/2337')
    dataset = WHU_MARS(root=cfg.DATASETS.ROOT_DIR, modalities=list(cfg.DATASETS.MODALITIES),
                       dataset_dir=cfg.DATASETS.DATASET_DIR, filter_gd=False)
    if dataset.num_train_pids != 500 or dataset.num_query_pids != 500 or dataset.num_gallery_pids != 500:
        raise ValueError('Expected WHU-MARS-1000 with 500 training/test identities; check dataset split')
    entries = [entry for split in (dataset.train, dataset.query, dataset.gallery)
               for name in cfg.DATASETS.MODALITIES for entry in split.get(name, [])]
    if any(entry[2] < 0 or entry[2] > 6 for entry in entries):
        raise ValueError('Platform inference requires official zero-based cameras 0..6')
    return dataset


def evaluation_entries(dataset, modalities):
    # Equivalent to R1_mAP_eval.split_all on a single process: concatenate the
    # queries across configured modalities, then concatenate the galleries.
    queries = [entry for name in modalities for entry in dataset.query.get(name, [])]
    galleries = [entry for name in modalities for entry in dataset.gallery.get(name, [])]
    return queries, galleries


def select_queries(dataset, modalities, count_per_modality, seed):
    rng, selected = random.Random(seed), []
    for name in modalities:
        pools = [[entry for entry in dataset.query.get(name, []) if (entry[2] < 5) == ground]
                 for ground in (True, False)]
        if any(not pool for pool in pools):
            raise ValueError('Representative query selection needs Ground and Aerial for ' + name)
        for pool in pools:
            rng.shuffle(pool)
        chosen = []
        for index in range(max(len(pool) for pool in pools)):
            for pool in pools:
                if index < len(pool) and len(chosen) < count_per_modality:
                    chosen.append(pool[index])
        if len(chosen) != count_per_modality:
            raise ValueError('Not enough representative queries for ' + name)
        selected.extend(chosen)
    return selected


def select_gradient_batches(dataset, modalities, num_batches, identities, instances, seed):
    if num_batches < 2 or identities < 2 or instances < 1:
        raise ValueError('Gradient probe needs at least two batches, two identities, and positive K')
    by_modality = {}
    for name in modalities:
        by_modality[name] = {}
        for entry in dataset.train.get(name, []):
            by_modality[name].setdefault(entry[1], []).append(entry)
    common = sorted(set.intersection(*(set(by_modality[name]) for name in modalities)))
    if len(common) < num_batches * identities:
        raise ValueError('Not enough shared identities for disjoint fixed gradient batches')
    rng = random.Random(seed)
    rng.shuffle(common)
    batches = []
    for batch in range(num_batches):
        ids = common[batch * identities:(batch + 1) * identities]
        groups = []
        for name in modalities:
            samples = []
            for pid in ids:
                pool = by_modality[name][pid]
                samples.extend(rng.sample(pool, instances) if len(pool) >= instances
                               else [rng.choice(pool) for _ in range(instances)])
            groups.append(samples)
        targets = [pid for pid in ids for _ in range(instances)]
        batches.append((groups, targets))
    return batches


def image_loader(entries, cfg, batch_size, workers):
    return DataLoader(FixedImages(entries, validation_transform(cfg)), batch_size=batch_size,
                      shuffle=False, num_workers=workers, pin_memory=True)


def order_digest(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
    return digest.hexdigest()


def load_gradient_images(groups, cfg, device):
    transform = validation_transform(cfg)
    return [torch.stack([transform(read_image(entry[0])) for entry in group]).to(device)
            for group in groups]
