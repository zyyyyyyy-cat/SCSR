"""Numerical/protocol/state regression checks for the four diagnostics."""
import tempfile
import unittest
import csv
import json
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from PIL import Image

from config import cfg
from model.backbones.vit_pytorch import TransReID
from diagnostics.core import (Capture, diagnostic_session, gradient_probe, internal_rows,
                              layerwise_rows, load_full_checkpoint, state_digest, write_csv)
from diagnostics.data import metadata, select_gradient_batches, select_queries
from diagnostics.retrieval import evaluate_on_off, paired_features, query_metrics
from utils.metrics import compute_indices_chunked, eval_func

torch.set_num_threads(2)


def make_toy_model():
    c = cfg.SCSR.clone()
    c.ENABLED = True
    c.IN_DIM = 32
    c.SPECTRAL_DIM = 8
    c.CONDITION_DIM = 4

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = TransReID(img_size=(32, 32), patch_size=8, stride_size=8,
                                 embed_dim=32, depth=12, num_heads=4, num_classes=0,
                                 mlp_ratio=0.5, scsr_cfg=c, scsr_target_id=3, num_modalities=3)
            self.bottleneck = nn.BatchNorm1d(32)
            self.classifier = nn.Linear(32, 4, bias=False)

        def forward(self, x, mode=0, modal_ids=None):
            if mode == 0:
                modal_ids = torch.cat([torch.full((images.shape[0],), i + 1, dtype=torch.long)
                                       for i, images in enumerate(x)])
                x = torch.cat(x)
            feat = self.base(x, modal_label=modal_ids)
            neck = self.bottleneck(feat)
            return (self.classifier(neck), feat, neck) if mode == 0 else feat

    return ToyModel().eval()


def records():
    return [dict(path='image_{}'.format(i), pid=i % 2, camid=i % 7, modal_id=i % 3 + 1,
                 modality_name=('RGB', 'NIR', 'TIR')[i % 3], platform='G' if i % 7 < 5 else 'A')
            for i in range(6)]


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_capture_matches_actual_output_and_cleans_up(self):
        model = make_toy_model()
        nn.init.normal_(model.base.scsr.condition[-1].weight, std=0.05)
        before = state_digest(model)
        images, mods = torch.randn(6, 3, 32, 32), torch.tensor([1, 2, 3, 1, 2, 3])
        on, off = Capture(), Capture()
        with torch.no_grad():
            with diagnostic_session(model, True, on):
                on_feat = model(images, mode=1, modal_ids=mods)
            with diagnostic_session(model, False, off):
                off_feat = model(images, mode=1, modal_ids=mods)
        self.assertTrue(torch.equal(on.layers['final_cls'], on_feat))
        self.assertTrue(torch.equal(off.layers['final_cls'], off_feat))
        self.assertTrue(torch.equal(on.layers['block_6'], off.layers['block_6']))
        self.assertTrue(torch.equal(on_feat[[0, 1, 3, 4]], off_feat[[0, 1, 3, 4]]))
        self.assertFalse(torch.equal(on_feat[[2, 5]], off_feat[[2, 5]]))
        torch.testing.assert_close(on.internal['updated'], on.layers['post_scsr'][[2, 5], 1:], atol=0, rtol=0)
        rows = internal_rows(on, [records()[2], records()[5]], 1e-6)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['r_effective'] > 0 for row in rows))
        effects = layerwise_rows(on, off, records(), 1e-6)
        self.assertTrue(any(row['layer'] == 'block_12_cls' for row in effects))
        self.assertEqual(before, state_digest(model))
        self.assertIsNone(model.base.diagnostic_callback)
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with diagnostic_session(model, False, Capture()):
                raise RuntimeError('test failure')
        self.assertTrue(model.base.scsr_runtime_enabled)
        self.assertIsNone(model.base.scsr.diagnostic_callback)
        model.train()
        with self.assertRaises(RuntimeError):
            with diagnostic_session(model, False):
                pass

    def test_rounding_is_measured_after_actual_addition(self):
        model = make_toy_model()
        with torch.no_grad():
            model.base.scsr.condition[-1].weight.fill_(1e-9)
        capture = Capture()
        with torch.no_grad(), diagnostic_session(model, True, capture):
            model(torch.randn(1, 3, 32, 32), mode=1, modal_ids=[3])
        row = internal_rows(capture, [records()[2]], 1e-6)[0]
        self.assertGreater(row['r_generated'], 0)
        self.assertEqual(row['r_effective'], 0)
        self.assertEqual(row['changed_ratio'], 0)

    def test_gradients_without_step_or_bn_update(self):
        model = make_toy_model()
        nn.init.normal_(model.base.scsr.condition[-1].weight, std=0.05)
        before = state_digest(model)
        rows = []
        with mock.patch.object(torch.optim.SGD, 'step', side_effect=AssertionError('No optimizer step')):
            for batch in (1, 2):
                batch_rows, losses = gradient_probe(model,
                    [torch.randn(4, 3, 32, 32) for _ in range(3)], torch.tensor([0, 0, 1, 1]),
                    batch, torch.device('cpu'))
                rows.extend(batch_rows)
                self.assertEqual({loss['loss_type'] for loss in losses}, {'CE', 'WRT'})
        self.assertEqual(before, state_digest(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(row['finite_gradient'] for row in rows))
        for loss_type in ('CE', 'WRT'):
            qrows = [row for row in rows if row['component'] == 'q' and row['loss_type'] == loss_type]
            self.assertTrue(all(row['grad_norm'] > 0 for row in qrows))

    def test_protocol_matches_original_and_noop_on_off(self):
        qf, gf = torch.randn(9, 32), torch.randn(60, 32)
        qf, gf = nn.functional.normalize(qf, dim=1), nn.functional.normalize(gf, dim=1)
        queries = [dict(path='q{}'.format(i), pid=i % 3, camid=i % 7, modal_id=i % 3 + 1,
                        modality_name=('RGB', 'NIR', 'TIR')[i % 3], platform='G' if i % 7 < 5 else 'A')
                   for i in range(9)]
        galleries = [dict(path='g{}'.format(i), pid=i % 3, camid=(i + 1) % 7, modal_id=i % 3 + 1,
                          modality_name=('RGB', 'NIR', 'TIR')[i % 3], platform='G' if (i + 1) % 7 < 5 else 'A')
                     for i in range(60)]
        metrics, comparisons, summary = evaluate_on_off({'ON': qf, 'OFF': qf.clone()},
            {'ON': gf, 'OFF': gf.clone()}, queries, galleries, chunk_size=3)
        indices = compute_indices_chunked(qf, gf, q_chunk_size=3)
        cmc, ap, inp = eval_func(indices, np.array([r['pid'] for r in queries]),
            np.array([r['pid'] for r in galleries]), np.array([r['camid'] for r in queries]),
            np.array([r['camid'] for r in galleries]), max_rank=10, return_minp=True)
        overall = metrics[0]
        self.assertAlmostEqual(overall['mAP'], ap)
        self.assertAlmostEqual(overall['mINP'], inp)
        for rank in (1, 5, 10):
            self.assertAlmostEqual(overall['Rank-' + str(rank)], float(cmc[rank - 1]), places=6)
        self.assertEqual(summary['ranking_changed'], 0)
        self.assertEqual(summary['on_correct_off_wrong'], 0)
        self.assertEqual(summary['on_wrong_off_correct'], 0)
        self.assertTrue(all(not row['ranking_changed'] for row in comparisons))
        invalid = dict(queries[0], pid=999)
        self.assertIsNone(query_metrics(torch.arange(60).float(), invalid, galleries))

    def test_fixed_selections_and_full_classifier_load(self):
        modalities = ['RGB', 'IR', 'Thermal']
        dataset = type('Dataset', (), {})()
        dataset.query = {name: [('q_{}_{}'.format(m, i), i % 3, i % 7, m + 1) for i in range(30)]
                         for m, name in enumerate(modalities)}
        dataset.train = {name: [('t_{}_{}_{}'.format(m, pid, i), pid, i % 7, m + 1)
                               for pid in range(8) for i in range(4)]
                         for m, name in enumerate(modalities)}
        a = select_queries(dataset, modalities, 8, 1234)
        self.assertEqual(a, select_queries(dataset, modalities, 8, 1234))
        self.assertEqual(len(a), 24)
        self.assertEqual({metadata(e, modalities)['platform'] for e in a}, {'G', 'A'})
        batches = select_gradient_batches(dataset, modalities, 2, 4, 4, 1234)
        for groups, labels in batches:
            self.assertTrue(all([e[1] for e in group] == labels for group in groups))
        # Exercise the real strict diagnostic loader, including a DDP-prefixed classifier.
        from test_scsr import experiment, wrapper
        config = experiment()
        original = wrapper(config).eval()
        with torch.no_grad():
            original.classifier.weight.fill_(0.123)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'scsr.pth'
            torch.save({'module.' + k: v for k, v in original.state_dict().items()}, checkpoint)
            Path(str(checkpoint) + '.yml').write_text(config.dump(), encoding='utf-8')
            restored = wrapper(config).eval()
            load_full_checkpoint(restored, checkpoint)
            self.assertTrue(torch.equal(original.classifier.weight, restored.classifier.weight))
            self.assertEqual(state_digest(original), state_digest(restored))
            csv_path = Path(directory) / 'tiny.csv'
            write_csv(csv_path, [dict(value=1e-12)])
            self.assertIn('1.000000000000e-12', csv_path.read_text())

    def test_six_artifact_pipeline_on_synthetic_images(self):
        import diagnose_scsr
        from model.make_model import build_transformer
        from test_scsr import experiment

        def factory(**kwargs):
            return TransReID(depth=12, mlp_ratio=0.125, qkv_bias=True, **kwargs)

        def construct(config, classes, cameras, views):
            return build_transformer(classes, cameras, views, config, {'vit_base_in': factory})

        c = experiment()
        c.DATALOADER.NUM_INSTANCE = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = SimpleNamespace(train={}, query={}, gallery={}, num_train_pids=4,
                                      num_train_cams=7, num_train_vids=3)
            for modal_id, name in enumerate(c.DATASETS.MODALITIES, 1):
                for split, pids, cameras in (('train', [0, 0, 1, 1, 2, 2, 3, 3], [0, 5] * 4),
                                             ('query', [0, 1], [0, 5]), ('gallery', [0, 1], [1, 6])):
                    entries = []
                    for i, (pid, cam) in enumerate(zip(pids, cameras)):
                        path = root / '{}_{}_{}.jpg'.format(split, modal_id, i)
                        rng = np.random.RandomState(modal_id * 100 + i)
                        Image.fromarray(rng.randint(0, 255, (32, 16, 3)).astype(np.uint8)).save(path)
                        entries.append((str(path), pid, cam, modal_id))
                    getattr(dataset, split)[name] = entries
            model = construct(c, 4, 7, 3)
            nn.init.normal_(model.base.scsr.condition[-1].weight, std=0.03)
            checkpoint = root / 'scsr_120.pth'
            torch.save(model.state_dict(), checkpoint)
            Path(str(checkpoint) + '.yml').write_text(c.dump(), encoding='utf-8')
            del model
            args = SimpleNamespace(checkpoint=str(checkpoint), uad_checkpoint='',
                output_dir=str(root / 'report'), device='cpu', seed=1234, amp=False,
                experiments=['1', '2', '3', '4'], batch_size=2, distance_chunk=2,
                probe_samples_per_modality=2, gradient_batches=2, gradient_identities=2,
                gradient_loss_scale=None, workers=0)
            with mock.patch.object(diagnose_scsr, 'load_dataset', return_value=dataset), \
                    mock.patch.object(diagnose_scsr, 'make_model', side_effect=construct), \
                    mock.patch.object(torch.optim.SGD, 'step', side_effect=AssertionError('No optimizer step')):
                result = diagnose_scsr.run(args, c)
            self.assertTrue(result['state_unchanged'])
            self.assertEqual(result['status'], 'complete')
            output = root / 'report'
            names = ['diagnostic_manifest.json', 'scsr_on_off_metrics.csv',
                     'scsr_on_off_query_comparison.csv', 'scsr_internal_activation.csv',
                     'scsr_layerwise_feature_effect.csv', 'scsr_gradient_probe.csv']
            self.assertTrue(all((output / name).is_file() for name in names))
            self.assertFalse(list(output.glob('*.pth')))
            manifest = json.loads((output / names[0]).read_text(encoding='utf-8'))
            self.assertTrue(manifest['classifier_restored'])
            with open(output / names[1], newline='', encoding='utf-8') as stream:
                metric_rows = list(csv.DictReader(stream))
            self.assertEqual(len(metric_rows), 28)  # overall + 9 modalities + 4 TIR platforms, ON/OFF
            self.assertEqual(manifest['gradient_nonfinite_rows'], 0)


if __name__ == '__main__':
    unittest.main()
