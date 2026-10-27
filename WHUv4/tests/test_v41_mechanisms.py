"""Focused post-training intervention/ranking/provenance tests; no real data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import cfg
from model import make_model
from model.scsr_v41 import SCSRV41
from diagnostics.core import Capture, diagnostic_session, state_digest
from diagnostics.v41_mechanisms import (mechanism_intervention, parameter_intervention, fixed_tir_samples,
                                       fixed_donor_pairs, gain_order_stats, rankdata, capture_batch)
from diagnostics.v41_initial import restore_initial_control, tensor_state_hash
from diagnostics.v41_ranking import evaluate_four_combinations
from diagnostics.v41_gradients import final_gradient_probe
from verify_scsr_v41 import build_parser, feature_cache, run_mechanisms, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


class V41MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.cfg = cfg.clone(); cls.cfg.merge_from_file(str(ROOT / 'configs/SCSR_V41.yml'))
        cls.cfg.MODEL.PRETRAIN_CHOICE = 'none'
        torch.manual_seed(1234)
        cls.model = make_model(cls.cfg, 4, 0, 0).eval()
        cls.images = torch.randn(3, 3, 256, 128)

    def test_01_normal_mirror_and_content_preservation(self):
        module = SCSRV41(self.cfg.SCSR_V41, self.cfg.DATASETS.MODALITIES).eval()
        tokens = torch.randn(5, 129, 768); ids = torch.tensor([3, 1, 3, 2, 3])
        before = tensor_state_hash(module.state_dict())
        cap = Capture(); module.diagnostic_callback = cap.spectral
        original = module(tokens, ids).detach()
        ref = cap.internal
        for kind in ('normal', 'zero_ll_condition', 'spatial_mean_v', 'disable_lh', 'disable_hl', 'disable_hh', 'disable_all'):
            with torch.no_grad(), mechanism_intervention(module, kind):
                result = module(tokens, ids)
            data = cap.internal
            self.assertTrue(torch.equal(result[ids != 3], tokens[ids != 3]))
            self.assertTrue(torch.equal(result[:, 0], tokens[:, 0]))
            for key in ('ll', 'lh', 'hl', 'hh'):
                self.assertTrue(torch.equal(data[key], ref[key]), key)
            if kind == 'normal':
                self.assertTrue(torch.equal(result, original))
            elif kind.startswith('disable_'):
                self.assertTrue(torch.equal(data['m'], ref['m']))
                self.assertTrue(torch.equal(data['v'], ref['v']))
                if kind == 'disable_all':
                    self.assertTrue(torch.equal(result, tokens))
                else:
                    j = ('disable_lh', 'disable_hl', 'disable_hh').index(kind)
                    self.assertEqual(float(data['applied_m'][:, j].abs().max()), 0)
            elif kind == 'spatial_mean_v':
                self.assertTrue(torch.equal(data['v'], ref['v'].mean((-2, -1), keepdim=True).expand_as(ref['v'])))
        self.assertNotIn('forward', module.__dict__)
        self.assertEqual(tensor_state_hash(module.state_dict()), before)

    def test_02_donor_v_only_and_restore_on_error(self):
        module = self.model.base.scsr_v41
        normal = capture_batch(self.model, self.images, torch.device('cpu'))
        donors = normal.internal['v'][[1, 2, 0]]
        before = state_digest(self.model)
        other = capture_batch(self.model, self.images, torch.device('cpu'), kind='donor_same_platform_v', donor_v=donors)
        self.assertTrue(torch.equal(other.internal['v'], donors))
        self.assertTrue(torch.equal(other.internal['own_v'], normal.internal['v']))
        for key in ('ll', 'lh', 'hl', 'hh'):
            self.assertTrue(torch.equal(other.internal[key], normal.internal[key]))
        with self.assertRaises(RuntimeError):
            with mechanism_intervention(module, 'zero_ll_condition'):
                raise RuntimeError('injected failure')
        self.assertNotIn('forward', module.__dict__)
        changed = {key: value.detach().clone() + 1 for key, value in module.state_dict().items()}
        with self.assertRaises(RuntimeError):
            with parameter_intervention(module, changed):
                raise RuntimeError('injected failure')
        self.assertEqual(state_digest(self.model), before)

    def test_03_all_disabled_equals_complete_off(self):
        with torch.no_grad():
            cap = capture_batch(self.model, self.images, torch.device('cpu'), kind='disable_all')
            with diagnostic_session(self.model, False):
                off = self.model(self.images, mode=1, modal_ids=torch.full((3,), 3, dtype=torch.long))
        self.assertTrue(torch.equal(cap.layers['final_cls'], off))

    def test_04_fixed_sampling_pairs_and_gain_ranking(self):
        class Dataset:
            query = {'Thermal': [('q{}_{}'.format(pid, cam), pid, cam, 3) for pid in range(6) for cam in (0, 5)]}
            gallery = {'Thermal': [('g{}_{}'.format(pid, cam), pid, cam, 3) for pid in range(6) for cam in (1, 6)]}
        a, b = fixed_tir_samples(Dataset(), 12, 1234), fixed_tir_samples(Dataset(), 12, 1234)
        self.assertEqual(a, b); self.assertEqual(len({row[0] for row in a}), 12)
        records = [dict(path=p, pid=pid, camid=cam, platform='G' if cam < 5 else 'A') for p, pid, cam, _ in a]
        self.assertEqual(sum(r['platform'] == 'G' for r in records), 6)
        for cross in (True, False):
            pairs = fixed_donor_pairs(records, cross)
            for i, j in enumerate(pairs):
                self.assertNotEqual(records[i]['pid'], records[j]['pid'])
                self.assertEqual(records[i]['platform'] != records[j]['platform'], cross)
        np.testing.assert_equal(rankdata([3, 1, 1, 2]), [3, .5, .5, 2])
        self.assertEqual(gain_order_stats(torch.arange(768.), torch.arange(768.))['gain_order_spearman'], 1)
        self.assertAlmostEqual(gain_order_stats(torch.arange(768.), -torch.arange(768.))['gain_order_spearman'], -1)
        self.assertTrue(gain_order_stats(torch.zeros(768), torch.ones(768))['gain_order_degenerate'])

    def test_05_four_combinations_full_ap_and_fixed_neighbors(self):
        queries = [dict(path='q', pid=1, camid=0, modal_id=3, modality_name='TIR', platform='G')]
        gallery = [dict(path=name, pid=pid, camid=cam, modal_id=3, modality_name='TIR', platform='G' if cam < 5 else 'A')
                   for name, pid, cam in [('excluded', 1, 0), ('positive', 1, 1), ('negative_A', 2, 5), ('negative_B', 3, 6)]]
        q = {'OFF': torch.tensor([[1., 0.]]), 'ON': torch.tensor([[0., 1.]])}
        g = {'OFF': torch.tensor([[1., 0.], [.6, .8], [1., 0.], [0., 1.]]),
             'ON': F.normalize(torch.tensor([[1., 0.], [0., 1.], [.8, .6], [.1, .995]]), dim=1)}
        with tempfile.TemporaryDirectory() as directory:
            summary = evaluate_four_combinations(q, g, queries, gallery, directory, 1, True)
            self.assertEqual(summary['wrong_to_correct'], 1)
            with (Path(directory) / 'v41_on_off_metrics.csv').open(encoding='utf-8') as stream:
                metrics = list(csv.DictReader(stream))
            overall = {row['group']: row for row in metrics if row['protocol'] == 'overall'}
            self.assertEqual(float(overall['A']['mAP']), .5)
            self.assertEqual(float(overall['B']['mAP']), 1.)
            self.assertEqual(float(overall['C']['mAP']), .5)
            self.assertAlmostEqual(float(overall['D']['mAP']), 1. / 3., places=12)
            with (Path(directory) / 'v41_identity_distance_changes.csv').open(encoding='utf-8') as stream:
                identity = list(csv.DictReader(stream))
            row = next(r for r in identity if r['protocol'] == 'overall')
            self.assertEqual(row['nearest_neg_A_path'], 'negative_A')
            self.assertEqual(row['nearest_neg_B_path'], 'negative_B')
            self.assertAlmostEqual(float(row['anchor_A_neg_distance_A']), 0)
            self.assertAlmostEqual(float(row['anchor_A_neg_distance_B']), .8, places=5)
            self.assertGreater(float(row['delta_margin']), .8)
            self.assertEqual(row['nearest_pos_A_path'], 'positive')
            self.assertEqual(row['anchor_A_pos_rank_A'], '2')
            self.assertEqual(row['anchor_A_pos_rank_B'], '1')

    def test_06_initial_reconstruction_provenance(self):
        files = list(ROOT.rglob('scsr_v41_init_rank0.json'))
        if not files:
            self.skipTest('Original initialization JSON not available in this workspace')
        record = json.loads(files[0].read_text(encoding='utf-8'))
        initial, manifest = restore_initial_control(self.cfg, self.model, {'calibration': record}, str(files[0].parent),
                                                   str(files[0].parent / 'scsr_v41_training_diagnostics.jsonl'))
        self.assertIsNotNone(initial, manifest)
        self.assertEqual(manifest['control_type'], 'reconstructed_initial_control')
        self.assertFalse(manifest['exact_original_initialization'])
        self.assertTrue(all(row['matches'] for row in manifest['norm_checks']))
        self.assertAlmostEqual(float(initial['output.weight'].std(unbiased=False)), record['measured_output_std'], places=7)
        broken = dict(record, initialization_seed=999)
        state, report = restore_initial_control(self.cfg, self.model, {'calibration': broken})
        self.assertIsNone(state)
        self.assertEqual(report['status'], 'unavailable')

    def test_07_gradient_probe_cosines_without_mutation(self):
        before = state_digest(self.model)
        images = [torch.randn(8, 3, 256, 128) for _ in range(3)]
        target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        rows, summaries = final_gradient_probe(self.model, images, target, 0, torch.device('cpu'), include_proca=True)
        parameter_rows = [row for row in rows if row['kind'] == 'parameter']
        self.assertEqual(len(parameter_rows), 21)
        for row in parameter_rows:
            self.assertTrue(row['finite_gradient'])
            self.assertGreater(row['grad_norm'], 0)
            self.assertTrue(-1.000001 <= row['gradient_cosine_CE_WRT'] <= 1.000001)
            self.assertLessEqual(row['post_clip_grad_norm'], row['grad_norm'])
        for row in rows:
            if row['kind'] == 'activation':
                self.assertFalse(row['clipping_applies'])
                self.assertEqual(row['post_clip_grad_norm'], row['grad_norm'])
        self.assertEqual(state_digest(self.model), before)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not row['gpd_included'] for row in summaries))

    def test_08_cache_rejects_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.pth'
            values = F.normalize(torch.randn(2, 768), dim=1)
            torch.save(dict(identity={'key': 'original'}, query={'ON': values, 'OFF': values},
                            gallery={'ON': values, 'OFF': values}), path)
            args = build_parser().parse_args(['--checkpoint', 'unused', '--output-dir', directory, '--feature-cache', str(path)])
            records = [dict(modal_id=3), dict(modal_id=3)]
            with self.assertRaises(ValueError):
                feature_cache(self.model, [0, 1], [0, 1], records, records, self.cfg, args, Path(directory),
                              torch.device('cpu'), {'key': 'different'}, lambda message: None)
            q, g, info = feature_cache(self.model, [0, 1], [0, 1], records, records, self.cfg, args, Path(directory),
                                      torch.device('cpu'), {'key': 'original'}, lambda message: None)
            self.assertTrue(info['reused']); self.assertTrue(torch.equal(q['ON'], values))

    def test_09_protocol_guards_and_training_rejection(self):
        with self.assertRaises(ValueError):
            validate_protocol(self.cfg, [0], [0])
        module = SCSRV41(self.cfg.SCSR_V41, self.cfg.DATASETS.MODALITIES).train()
        with self.assertRaises(ValueError):
            with mechanism_intervention(module):
                pass
        with self.assertRaises(RuntimeError):
            with parameter_intervention(module, module.state_dict()):
                pass

    def test_10_small_image_cohort_all_mechanism_outputs(self):
        files = list(ROOT.rglob('scsr_v41_init_rank0.json'))
        if not files:
            self.skipTest('Initialization record not available')
        calibration = json.loads(files[0].read_text(encoding='utf-8'))
        before = state_digest(self.model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for pid in (0, 1, 2):
                for camera in (0, 5):
                    path = root / '{}_{}.jpg'.format(pid, camera)
                    Image.new('RGB', (128, 256), (40 + pid * 60, 80 + camera, 120)).save(path)
                    entries.append((str(path), pid, camera, 3))
            class Dataset:
                query = {'Thermal': entries}
                gallery = {'Thermal': []}
            output = root / 'outputs'; output.mkdir()
            args = build_parser().parse_args(['--checkpoint', str(root / 'unused.pth'), '--output-dir', str(output),
                                             '--tir-samples', '4', '--batch-size', '2', '--workers', '0',
                                             '--initial-records-dir', str(files[0].parent)])
            report = run_mechanisms(self.model, Dataset(), self.cfg, args, output, torch.device('cpu'),
                                    {'calibration': calibration}, {2, 3}, lambda text: None, {'checkpoint_sha256': 'unit_fixture'})
            self.assertEqual(report['initial_control']['status'], 'ready')
            for filename in ('v41_initial_vs_trained_parameters.csv', 'v41_initial_vs_trained_gate.csv',
                             'v41_initial_vs_trained_features.csv', 'v41_condition_intervention.csv',
                             'v41_frequency_band_intervention.csv', 'v41_gate_input_dependence.csv',
                             'v41_condition_manifest.json', 'v41_initial_reconstruction_manifest.json'):
                self.assertTrue((output / filename).is_file(), filename)
            with (output / 'v41_frequency_band_intervention.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row['group'] == 'disable_all' and row['row_type'] == 'statistics':
                    for direction in ('lh', 'hl', 'hh'):
                        self.assertEqual(float(row[direction + '_increment_norm']), 0)
        self.assertEqual(state_digest(self.model), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
