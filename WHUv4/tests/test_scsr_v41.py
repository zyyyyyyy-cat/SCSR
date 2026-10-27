"""CPU correctness/compatibility tests; no WHU-MARS or ImageNet files needed."""
import ast
import json
import logging
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import cfg
from model import make_model
from model.backbones.vit_pytorch import AttentionInputBlock, Block
from model.scsr_v41 import SCSRV41, haar_dwt, haar_idwt
from solver.make_optimizer import make_optimizer
from processor.processor import PCA, GPD, PrototypeBank
from loss.wrt_loss import TripletLoss_WRT
from utils.scsr_v41_runtime import (Capture, diagnostic_session, fixed_calibration_batch,
                                    preserve_rng, evaluation_mode, short_on_off, initial_scale_factor)
from utils.scsr_v41_checkpoint import load_v41, save_metadata
from diagnostics.core import gradient_probe, state_digest
from diagnostics.retrieval import evaluate_on_off
from utils.scsr_v41_monitor import TrainingMonitor


ROOT = Path(__file__).resolve().parents[1]


class V41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.cfg = cfg.clone()
        cls.cfg.merge_from_file(str(ROOT / 'configs/SCSR_V41.yml'))
        cls.cfg.MODEL.PRETRAIN_CHOICE = 'none'
        torch.manual_seed(1234)
        cls.model = make_model(cls.cfg, 4, 0, 0).eval()
        baseline_cfg = cls.cfg.clone()
        baseline_cfg.SCSR_V41.ENABLED = False
        original_vit = {'__name__': 'original_uad_vit'}
        original_wrapper = {'__name__': 'original_uad_wrapper', '__package__': 'model'}
        fixture = ROOT / 'tests/fixtures'
        exec(compile((fixture / 'uad_vit_original.py').read_text(encoding='utf-8'), 'original_uad_vit', 'exec'), original_vit)
        exec(compile((fixture / 'uad_make_model_original.py').read_text(encoding='utf-8'), 'original_uad_wrapper', 'exec'), original_wrapper)
        torch.manual_seed(1234)
        cls.baseline = original_wrapper['build_transformer'](4, 0, 0, baseline_cfg,
                                                            {'vit_base_in': original_vit['vit_base_in']}).eval()
        cls.images = [torch.randn(8, 3, 256, 128) for _ in range(3)]
        cls.target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    def test_01_haar_direction_energy_and_inverse(self):
        x = torch.randn(3, 768, 16, 8, dtype=torch.float64)
        bands = haar_dwt(x)
        torch.testing.assert_close(haar_idwt(*bands), x, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(sum(b.square().sum() for b in bands)), float(x.square().sum()), places=8)
        tile = torch.tensor([[[[1., 2.], [3., 4.]]]])
        self.assertEqual([float(b) for b in haar_dwt(tile)], [5., -2., -1., 0.])
        x.requires_grad_()
        haar_idwt(*haar_dwt(x)).square().sum().backward()
        torch.testing.assert_close(x.grad, 2 * x, rtol=1e-12, atol=1e-12)

    def test_02_original_code_and_seed_equivalence(self):
        # Compare against the archived BEFORE-edit original, not the new OFF path alone.
        for key, value in self.baseline.state_dict().items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]), key + ' seed drift')
        self.assertEqual([i for i, b in enumerate(self.model.base.blocks) if isinstance(b, AttentionInputBlock)], [6])
        ids = torch.tensor([1, 2, 3, 3, 1, 2, 3, 3])
        with torch.no_grad(), diagnostic_session(self.model, False):
            actual = self.model(self.images[0], mode=1, modal_ids=ids)
            expected = self.baseline(self.images[0], mode=1)
            self.assertTrue(torch.equal(expected, actual))

    def test_03_modalities_bounds_dtype(self):
        module = SCSRV41(self.cfg.SCSR_V41, ('Thermal', 'RGB', 'IR'))
        self.assertEqual(module.target_id, 1)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 40032)
        module.collect_stats = True
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            z = torch.randn(4, 129, 768).to(dtype)
            ids = torch.tensor([3, 1, 2, 1])
            cap = Capture()
            module.diagnostic_callback = cap.spectral
            result = module(z, ids)
            self.assertEqual(result.dtype, z.dtype)
            self.assertTrue(torch.equal(result[:, 0], z[:, 0]))
            self.assertTrue(torch.equal(result[ids != 1], z[ids != 1]))
            self.assertGreater(float((result[ids == 1].float() - z[ids == 1].float()).detach().abs().max()), 0)
            p = cap.internal['z_tir'].float().transpose(1, 2).reshape(-1, 768, 16, 8)
            updated = cap.internal['updated_fp32'].transpose(1, 2).reshape_as(p)
            ratio = (updated - p).flatten(1).norm(dim=1) / p.flatten(1).norm(dim=1)
            self.assertTrue((ratio <= .200001).all())
            self.assertLessEqual(float(cap.internal['m'].abs().max()), .200001)
            torch.testing.assert_close(haar_dwt(updated)[0], haar_dwt(p)[0], rtol=1e-5, atol=5e-7)
        with self.assertRaises(ValueError):
            module(torch.randn(1, 129, 768), torch.tensor([4]))
        with self.assertRaises(ValueError):
            module(torch.randn(1, 129, 768), None)

    def test_04_block_residual_and_runtime_guard(self):
        block = AttentionInputBlock(8, 2, drop_path=0).eval()
        x = torch.randn(2, 3, 8)
        z = block.norm1(x)
        y = x + block.attn(z * 2)
        expected = y + block.mlp(block.norm2(y))
        self.assertTrue(torch.equal(block(x, lambda value, ids: value * 2), expected))
        self.assertTrue(torch.equal(block(x), Block.forward(block, x)))
        with evaluation_mode(self.model):
            result = short_on_off(self.model, self.images, self.cfg.DATASETS.MODALITIES, use_amp=False)
        self.assertGreater(result['block7_attention_output_tir_relative'], 1e-6)
        self.assertGreater(result['final_cls_tir_relative'], 1e-7)
        self.model.train()
        self.model.base.scsr_runtime_enabled = False
        try:
            with self.assertRaises(RuntimeError):
                self.model(self.images)
        finally:
            self.model.base.scsr_runtime_enabled = True
            self.model.eval()

    def test_05_optimizer_and_fixed_batch_rng(self):
        center = nn.Linear(2, 2)
        optimizer, _ = make_optimizer(self.cfg, self.model, center)
        group = [g for g in optimizer.param_groups if g.get('name') == 'SCSR-V4.1']
        self.assertEqual(len(group), 1)
        self.assertEqual(group[0]['lr'], .008)
        self.assertEqual(group[0]['weight_decay'], 0)
        self.assertEqual(len(group[0]['params']), 7)
        classifier = next(g for g in optimizer.param_groups if g['params'][0] is self.model.classifier.weight)
        self.assertEqual(classifier['weight_decay'], 1e-4)
        class Dataset:
            modalities = ['RGB', 'IR', 'Thermal']
            dataset = {name: [('fake/{}_{}.jpg'.format(name, p), p, 0, m + 1) for p in range(20)]
                       for m, name in enumerate(modalities)}
            def __getitem__(self, index):
                random.random(); np.random.rand()
                return [torch.rand(3, 2, 2) for _ in range(3)], self.dataset['RGB'][index[0]][1], [0, 0, 0]
        class Loader:
            dataset = Dataset()
            def __iter__(self):
                raise AssertionError('Must not consume the training loader/sampler')
        random.seed(12); np.random.seed(12); torch.manual_seed(12)
        initial = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with preserve_rng():
            images, meta = fixed_calibration_batch(self.cfg, Loader(), 0, 1)
        self.assertEqual(random.getstate(), initial[0])
        np.testing.assert_equal(np.random.get_state(), initial[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), initial[2]))
        self.assertEqual([len(g) for g in images], [64, 64, 64])
        self.assertEqual(meta['images'], 192)
        ddp_cfg = self.cfg.clone(); ddp_cfg.SOLVER.IMS_PER_BATCH = 256
        with preserve_rng():
            _, meta2 = fixed_calibration_batch(ddp_cfg, Loader(), 0, 4)
        self.assertEqual(meta['sample_paths_sha256'], meta2['sample_paths_sha256'])

    def test_06_separate_identity_gradient_probe_no_mutation(self):
        before = state_digest(self.model)
        rows, summaries = gradient_probe(self.model, self.images, self.target, 0, torch.device('cpu'))
        parameters = [r for r in rows if r['kind'] == 'parameter']
        self.assertEqual(len(parameters), 14)
        for row in parameters:
            self.assertTrue(row['finite_gradient'], row['component'])
            self.assertGreater(row['grad_norm'], 0, row['component'])
        self.assertTrue(all(np.isfinite(row['loss']) for row in summaries))
        self.assertEqual(state_digest(self.model), before)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_07_joint_uad_loss_actual_sgd_update(self):
        optimizer, _ = make_optimizer(self.cfg, self.model, nn.Linear(2, 2))
        bank = PrototypeBank(4, 768, momentum=.2, device='cpu')
        self.model.train()
        changes = []
        for step in range(2):
            old = {key: p.detach().clone() for key, p in self.model.base.scsr_v41.named_parameters()}
            optimizer.zero_grad()
            score, feat, neck = self.model(self.images, label=self.target)
            ce = F.cross_entropy(score, self.target.repeat(3))
            wrt = TripletLoss_WRT()(feat, self.target.repeat(3))[0]
            pca, ids, centers = PCA(feat, self.target, 3, group_size=self.cfg.DATALOADER.NUM_INSTANCE)
            gpd = GPD(feat, self.target, bank, ids, centers, 3, tau=.03)
            loss = ce + wrt + .01 * pca + gpd
            self.assertTrue(all(value.isfinite() for value in (ce, wrt, pca, gpd)))
            loss.backward()
            for key, param in self.model.base.scsr_v41.named_parameters():
                self.assertIsNotNone(param.grad, key)
                self.assertTrue(param.grad.isfinite().all(), key)
                self.assertGreater(float(param.grad.norm()), 0, key)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optimizer.step()
            bank.update(ids, centers, ddp=False)
            actual = {key: float((p.detach() - old[key]).norm() / old[key].norm())
                      for key, p in self.model.base.scsr_v41.named_parameters()}
            for key, value in actual.items():
                self.assertGreater(value, 0, key + ' actual update is zero')
            changes.append(actual)
        self.model.eval()
        print('Synthetic joint CE/WRT/ProCA/GPD actual relative SGD updates:', json.dumps(changes))

    def test_08_strict_checkpoint_and_version(self):
        module = self.model.base.scsr_v41
        module.calibration = dict(verified=True, test_fixture=True)
        module.calibrated = True
        ids = torch.tensor([1, 2, 3, 3, 1, 2, 3, 3])
        with torch.no_grad():
            expected = self.model(self.images[0], mode=1, modal_ids=ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'transformer_120.pth'
            torch.save(self.model.state_dict(), path)
            save_metadata(self.model, path, 120, 2, 0)
            with torch.no_grad():
                module.U.add_(1)
            load_v41(self.model, path, include_classifier=True)
            with torch.no_grad():
                actual = self.model(self.images[0], mode=1, modal_ids=ids)
            self.assertTrue(torch.equal(expected, actual))
            sidecar = Path(str(path) + '.json')
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            metadata['version'] = 'SCSR-V3'
            sidecar.write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_v41(self.model, path)

    def test_09_ranking_protocol_and_old_syntax(self):
        queries = [dict(path='q', pid=1, camid=0, modal_id=3, modality_name='TIR', platform='G')]
        galleries = [dict(path='samecam', pid=1, camid=0, modal_id=3, modality_name='TIR', platform='G'),
                     dict(path='pos', pid=1, camid=5, modal_id=3, modality_name='TIR', platform='A'),
                     dict(path='neg', pid=2, camid=6, modal_id=3, modality_name='TIR', platform='A')]
        q = {'ON': torch.tensor([[1., 0.]]), 'OFF': torch.tensor([[1., 0.]])}
        g = {'ON': torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
             'OFF': torch.tensor([[1., 0.], [1., 0.], [0., 1.]])}
        metrics, comparisons, summary = evaluate_on_off(q, g, queries, galleries, 1)
        self.assertEqual(summary['valid_queries'], 1)
        self.assertEqual(summary['ranking_changed'], 0)
        self.assertEqual(comparisons[0]['on_top_gallery_path'], 'pos')
        self.assertEqual(comparisons[0]['on_AP'], 1)
        for path in ROOT.rglob('*.py'):
            if '__pycache__' not in path.parts:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=8)
        text = (ROOT / 'processor/processor.py').read_text(encoding='utf-8')
        self.assertNotIn('register_step_post_hook', text)

    def test_10_amp_overflow_diagnostic_serialization(self):
        # AMP overflow must remain inspectable instead of failing JSON writing.
        with tempfile.TemporaryDirectory() as directory:
            config = self.cfg.clone(); config.OUTPUT_DIR = directory
            monitor = TrainingMonitor(config, self.model, logging.getLogger('test'))
            module = self.model.base.scsr_v41
            module.last_stats = dict(v_rms=.07, t_rms=.07, r_attn_mean=.01)
            monitor.snapshot()
            original_grad = module.U.grad
            try:
                module.U.grad = torch.full_like(module.U, float('inf'))
                monitor.gradients(); monitor.gradients(post_clip=True)
                # >200 avoids paired CUDA probes in this CPU-only writer test.
                monitor.record(1, 1, 202, 201, 1, False, 65536, 32768, self.images)
                record = json.loads(Path(monitor.path).read_text(encoding='utf-8'))
                self.assertFalse(record['parameters']['U']['grad_pre_clip']['finite'])
                self.assertEqual(record['parameters']['U']['grad_pre_clip']['norm'], 'inf')
                self.assertEqual(record['parameters']['U']['changed_fraction'], 0)
            finally:
                module.U.grad = original_grad

    def test_11_one_time_initial_scale_factor(self):
        self.assertEqual(initial_scale_factor(dict(v_rms=.07, t_rms=.08)), 1.)
        for values in (dict(v_rms=.02, t_rms=.03), dict(v_rms=.4, t_rms=.35)):
            scale = initial_scale_factor(values)
            for value in values.values():
                self.assertGreaterEqual(value * scale, .05 - 1e-8)
                self.assertLessEqual(value * scale, .10 + 1e-8)
        for values in (dict(v_rms=.01, t_rms=.2), dict(v_rms=0., t_rms=.1), dict(v_rms=float('nan'), t_rms=.1)):
            with self.assertRaises(RuntimeError):
                initial_scale_factor(values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
