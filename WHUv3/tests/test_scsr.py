"""Dataset-free numerical/integration checks. Run from WHU:
python -m unittest discover -s tests -p test_scsr.py -v
"""
import tempfile
import unittest
import zipfile
import inspect
import logging
from types import SimpleNamespace
from pathlib import Path
from contextlib import nullcontext
from unittest import mock

import torch
from torch import nn

from config import cfg
from model.scsr import SCSR, haar_dwt, haar_idwt, smooth_bounded_residual, _disable_autocast
from model.backbones.vit_pytorch import TransReID
from model.make_model import build_transformer
from solver.make_optimizer import make_optimizer
from processor.processor import PCA, GPD, PrototypeBank
import processor.processor as processor
from loss.wrt_loss import TripletLoss_WRT

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def experiment(enabled=True):
    c = cfg.clone()
    c.merge_from_file(str(ROOT / 'configs' / 'SCSR.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'none'
    c.MODEL.DROP_PATH = 0.0
    c.SCSR.ENABLED = enabled
    return c


def small_factory(**kwargs):
    # Same image grid and embedding channels, less FFN capacity for CPU tests.
    return TransReID(depth=8, mlp_ratio=0.125, qkv_bias=True, **kwargs)


def wrapper(c):
    return build_transformer(4, 0, 0, c, {'vit_base_in': small_factory})


class SCSRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_haar_inverse_energy_signed_orientation(self):
        x = torch.randn(2, 3, 16, 8, dtype=torch.float64, requires_grad=True)
        bands = haar_dwt(x)
        torch.testing.assert_close(haar_idwt(*bands), x, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(sum(t.square().sum() for t in bands), x.square().sum())
        self.assertTrue(torch.autograd.gradcheck(lambda z: haar_idwt(*haar_dwt(z)), (x,)))
        # Known signed convention: horizontal edge=-2, vertical edge=-1.
        cell = torch.tensor([[[[1., 2.], [3., 4.]]]])
        self.assertEqual([b.item() for b in haar_dwt(cell)], [5., -2., -1., 0.])
        self.assertLess(bands[1].min().item(), 0)
        with self.assertRaises(ValueError):
            haar_dwt(torch.randn(1, 1, 15, 8))

    def test_parameters_identity_and_gradient_startup(self):
        m = SCSR()
        self.assertEqual(sum(p.numel() for p in m.parameters()), 383616)
        p = torch.randn(3, 128, 768, requires_grad=True)
        target = torch.randn_like(p)
        self.assertTrue(torch.equal(m(p), p))
        optimizer = torch.optim.SGD(m.parameters(), lr=0.008)
        for step in range(4):
            optimizer.zero_grad()
            loss = (m(p) - target).square().mean()
            loss.backward()
            self.assertGreater(m.condition[-1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(v.grad is not None and v.grad.isfinite().all() for v in m.parameters()))
            if step == 0:
                self.assertEqual(m.down.weight.grad.abs().sum().item(), 0)
            else:
                for parameter in m.parameters():
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(m(p), p))

    def test_zero_ll_bound_and_amp_dtype(self):
        m = SCSR()
        nn.init.normal_(m.condition[-1].weight, std=0.1)
        # Exercise large raw updates in the smooth asymptotic regime.
        with torch.no_grad():
            m.up.weight.mul_(1000)
        p = torch.randn(3, 128, 768)
        raw = []
        handle = m.up.register_forward_hook(lambda mod, args, out: raw.append(out.detach()))
        delta = m(p) - p
        handle.remove()
        expected = smooth_bounded_residual(p, raw[0])
        torch.testing.assert_close(delta, expected, atol=3e-7, rtol=1e-4)
        ll = haar_dwt(delta.transpose(1, 2).reshape(3, 768, 16, 8))[0]
        self.assertLess(ll.abs().max().item(), 1e-6)
        ratios = delta.flatten(1).norm(dim=1) / p.flatten(1).norm(dim=1)
        self.assertTrue((ratios < 0.2).all())
        self.assertTrue(torch.equal(m(torch.zeros_like(p)), torch.zeros_like(p)))
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            cpu_amp = getattr(getattr(torch, 'cpu', None), 'amp', None)
            autocast = getattr(cpu_amp, 'autocast', None)
            context = autocast(dtype=torch.bfloat16) if autocast is not None else nullcontext()
            with context:
                result = m(p.to(dtype))
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(result.isfinite().all())

    def test_frozen_smooth_formula_ratios_and_radial_gradients(self):
        # Four samples with different norms verify per-sample (not batch) scaling.
        p = torch.tensor([1., 2., 3., 4.], dtype=torch.float64).view(4, 1, 1)
        t = torch.tensor([0.1, 0.5, 1., 2.], dtype=torch.float64).view(4, 1, 1)
        delta = (t * p).requires_grad_()
        residual = smooth_bounded_residual(p, delta)
        ratio = (residual / p).flatten()
        expected = (0.2 * t / (1 + t * p / (p + 1e-6))).flatten()
        torch.testing.assert_close(ratio, expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(ratio, torch.tensor(
            [0.018181818, 0.066666667, 0.1, 0.133333333], dtype=torch.float64),
            atol=4e-8, rtol=0)
        grad = torch.autograd.grad(residual.sum(), delta)[0]
        torch.testing.assert_close(grad, 0.2 / (1 + delta.detach() / (p + 1e-6)).square())
        self.assertTrue((grad > 0).all())  # t=2 must not hard-saturate.
        zero = torch.zeros_like(p, requires_grad=True)
        r0 = smooth_bounded_residual(p, zero)
        torch.testing.assert_close(torch.autograd.grad(r0.sum(), zero)[0], torch.full_like(p, 0.2))
        self.assertTrue(torch.autograd.gradcheck(
            lambda d: smooth_bounded_residual(p, d), (delta,)))

    def test_original_backbone_and_initialized_full_vit_equivalence(self):
        # Compare directly with the archived, unmodified UAD backbone.
        with zipfile.ZipFile(ROOT / 'docs' / 'uad_before_scsr.zip') as archive:
            source = archive.read('vit_pytorch.py').decode('utf-8')
        namespace = {'__name__': 'original_uad'}
        exec(compile(source, 'original_vit_pytorch.py', 'exec'), namespace)
        torch.manual_seed(1234)
        original = namespace['TransReID'](img_size=(256, 128), num_heads=12,
                                          qkv_bias=True).eval()
        torch.manual_seed(1234)
        baseline = TransReID(img_size=(256, 128), num_heads=12, qkv_bias=True).eval()
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, baseline.state_dict()[key]), key)
        c = experiment()
        torch.manual_seed(1234)
        enhanced = TransReID(img_size=(256, 128), num_heads=12, qkv_bias=True,
                            scsr_cfg=c.SCSR, scsr_target_id=3, num_modalities=3).eval()
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, enhanced.state_dict()[key]), key)
        images = torch.randn(3, 3, 256, 128)
        with torch.no_grad():
            expected = original(images)
            self.assertTrue(torch.equal(baseline(images), expected))
            self.assertTrue(torch.equal(enhanced(images, modal_label=[1, 2, 3]), expected))
            nn.init.normal_(enhanced.scsr.condition[-1].weight, std=0.1)
            actual = enhanced(images, modal_label=[1, 2, 3])
        self.assertTrue(torch.equal(actual[:2], expected[:2]))
        self.assertFalse(torch.equal(actual[2], expected[2]))

    def test_l6_route_cls_and_batch_order(self):
        c = experiment()
        m = wrapper(c)
        nn.init.normal_(m.base.scsr.condition[-1].weight, std=0.1)
        tokens = torch.randn(5, 129, 768, requires_grad=True)
        ids = torch.tensor([3, 1, 2, 3, 1])
        updated = m.base._route_scsr(tokens, ids)
        self.assertTrue(torch.equal(updated[:, 0], tokens[:, 0]))
        self.assertTrue(torch.equal(updated[[1, 2, 4]], tokens[[1, 2, 4]]))
        torch.testing.assert_close(updated[[0, 3], 1:], m.base.scsr(tokens[[0, 3], 1:]))
        updated.sum().backward()
        self.assertTrue(tokens.grad.isfinite().all())
        events = []
        handles = [block.register_forward_hook(
            lambda mod, inp, out, i=i: events.append(i)) for i, block in enumerate(m.base.blocks, 1)]
        handles.append(m.base.scsr.register_forward_hook(lambda *args: events.append('SCSR')))
        with mock.patch.object(m.base, 'forward', wraps=m.base.forward) as forward:
            m([torch.randn(2, 3, 256, 128) for _ in range(3)])
            seen = [forward.call_args[1]['modal_label'].tolist()]
        self.assertEqual(events, [1, 2, 3, 4, 5, 6, 'SCSR', 7, 8])
        self.assertEqual(seen, [[1, 1, 2, 2, 3, 3]])
        for handle in handles:
            handle.remove()

    def test_no_unified_autocast_dependency(self):
        # Simulate old torch without unified autocast or CPU AMP namespaces.
        m = SCSR()
        p = torch.randn(2, 128, 768)
        with mock.patch.object(torch, 'autocast', None, create=True), \
                mock.patch.object(torch, 'cpu', None, create=True):
            self.assertTrue(torch.equal(m(p), p))
        with mock.patch.object(torch.cuda.amp, 'autocast', return_value=nullcontext()) as autocast:
            with _disable_autocast('cuda'):
                pass
            autocast.assert_called_once_with(enabled=False)

    def test_training_checkpoint_validation_with_legacy_optimizer(self):
        # Execute the production loop on CPU. Only device placement and CUDA
        # services are adapted; loss/step/checkpoint/validation code is retained.
        class LegacySGD(torch.optim.SGD):
            def __getattribute__(self, name):
                if name == 'register_step_post_hook':
                    raise AttributeError('Legacy SGD has no optimizer hook API')
                return super().__getattribute__(name)

        c = experiment()
        c.SOLVER.MAX_EPOCHS = 1
        c.SOLVER.CHECKPOINT_PERIOD = 1
        c.SOLVER.EVAL_PERIOD = 1
        c.SOLVER.LOG_PERIOD = 1
        m = wrapper(c)
        center = nn.Linear(1, 1)
        optimizer = LegacySGD(m.parameters(), lr=0.008, momentum=0.9)
        center_optimizer = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = SimpleNamespace(step=lambda epoch: None, _get_lr=lambda epoch: [0.008])
        target = torch.tensor([0] * 4 + [1] * 4)
        batches = [([torch.randn(8, 3, 256, 128) for _ in range(3)], target,
                    [torch.zeros(8, dtype=torch.long) for _ in range(3)]) for _ in range(2)]
        train_loader = list(batches)  # do_train's single-GPU speed log uses this property.
        class Loader(list):
            batch_size = 8
        train_loader = Loader(train_loader)
        val_loaders = []
        for modality in range(1, 4):
            pids = tuple([0, 1] + [0, 1] * 5)
            cams = tuple([0, 0] + [1] * 10)
            val_loaders.append([(torch.randn(12, 3, 256, 128), pids, cams,
                                 torch.tensor(cams), (modality,) * 12, ('synthetic',) * 12)])

        def loss_fn(score, features, labels):
            ce = nn.functional.cross_entropy(score, labels)
            wrt = TripletLoss_WRT()(features, labels)[0]
            return ce + wrt, ce, wrt

        source = inspect.getsource(processor.do_train)
        source = source.replace("device = 'cuda'", "device = 'cpu'")
        source = source.replace('model.to(local_rank)', 'model.to(device)')
        namespace = dict(processor.__dict__)
        namespace['amp'] = SimpleNamespace(
            GradScaler=lambda: torch.cuda.amp.GradScaler(enabled=False),
            autocast=lambda **kwargs: nullcontext())
        exec(compile(source, 'cpu_adapted_production_loop', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, 'synchronize'), \
                mock.patch.object(torch.cuda, 'empty_cache'), \
                mock.patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                mock.patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
            c.OUTPUT_DIR = directory
            m.experiment_cfg = c.clone()
            with self.assertLogs('transreid.train', level=logging.INFO) as captured:
                namespace['do_train'](c, m, center, train_loader, val_loaders,
                                      optimizer, center_optimizer, scheduler, loss_fn, [2, 2, 2], 0)
            output = '\n'.join(captured.output)
            self.assertIn('epoch optimizer updates=2', output)
            self.assertIn('cumulative AMP-skipped updates=0', output)
            self.assertIn('Validation Results', output)
            path = Path(directory) / 'transformer_1.pth'
            self.assertTrue(path.is_file())
            self.assertTrue(Path(str(path) + '.yml').is_file())
            restored = wrapper(c).eval()
            restored.load_param(path)
            with torch.no_grad():
                images = val_loaders[2][0][0][:2]
                self.assertTrue(torch.equal(m(images, mode=1, modal_ids=[3, 3]),
                                            restored(images, mode=1, modal_ids=[3, 3])))

    def test_mapping_invalid_ids_and_inference_mode(self):
        c = experiment()
        c.DATASETS.MODALITIES = ['Thermal', 'RGB', 'IR']
        m = wrapper(c).eval()
        self.assertEqual(m.base.scsr_target_id, 1)
        images = torch.randn(2, 3, 256, 128)
        for ids in (None, [1], [0, 4], [1.0, 2.0], [True, False]):
            with self.assertRaises(ValueError):
                m(images, mode=99, modal_ids=ids)
        with torch.no_grad():
            self.assertEqual(m(images, mode=99, modal_ids=[1, 2]).shape, (2, 768))
        c.SCSR.TARGET_MODALITY = 'missing'
        with self.assertRaises(ValueError):
            wrapper(c)

    def test_192_image_feature_contract(self):
        m = wrapper(experiment()).eval()
        observed = []
        handle = m.base.scsr.register_forward_pre_hook(
            lambda mod, args: observed.append(tuple(args[0].shape)))
        with torch.no_grad():
            score, feature, neck = m([torch.randn(64, 3, 256, 128) for _ in range(3)])
        handle.remove()
        self.assertEqual(feature.shape, (192, 768))
        self.assertEqual(neck.shape, (192, 768))
        self.assertEqual(score.shape, (192, 4))
        self.assertEqual(observed, [(64, 128, 768)])

    def test_uad_losses_reach_scsr_and_optimizer_policy(self):
        c = experiment()
        m = wrapper(c)
        optimizer, _ = make_optimizer(c, m, nn.Linear(1, 1))
        groups = {id(g['params'][0]): g for g in optimizer.param_groups}
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            expected = 0.008 if 'base.scsr.' in name or 'bias' not in name else 0.016
            self.assertEqual(groups[id(p)]['lr'], expected, name)
        imgs = [torch.randn(8, 3, 256, 128) for _ in range(3)]
        ids = torch.tensor([0] * 4 + [1] * 4)
        scores, feat, neck = m(imgs)
        self.assertEqual(feat.shape, (24, 768))
        pca, center_ids, centers = PCA(feat, ids, 3, 4)
        bank = PrototypeBank(4, 768, device='cpu')
        losses = [nn.functional.cross_entropy(scores, ids.repeat(3)),
                  TripletLoss_WRT()(feat, ids.repeat(3))[0], pca,
                  GPD(feat, ids, bank, center_ids, centers, 3)]
        for loss in losses:
            grad = torch.autograd.grad(loss, m.base.scsr.condition[-1].weight, retain_graph=True)[0]
            self.assertTrue(grad.isfinite().all())
            self.assertGreater(grad.abs().sum().item(), 0)
        (losses[0] + losses[1] + 0.01 * losses[2] + losses[3]).backward()
        optimizer.step()
        bank.update(center_ids, centers, ddp=False)
        torch.testing.assert_close(bank.prototypes[center_ids], centers)

    def test_checkpoint_roundtrip_and_mismatch(self):
        c = experiment()
        m = wrapper(c).eval()
        nn.init.normal_(m.base.scsr.condition[-1].weight, std=0.1)
        images = torch.randn(3, 3, 256, 128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(m.state_dict(), path)
            path.with_suffix('.pth.yml').write_text(c.dump(), encoding='utf-8')
            restored = wrapper(c).eval()
            restored.load_param(path)
            with torch.no_grad():
                self.assertTrue(torch.equal(m(images, mode=1, modal_ids=[3, 1, 2]),
                                            restored(images, mode=1, modal_ids=[3, 1, 2])))
            with self.assertRaises(ValueError):
                wrapper(experiment(False)).load_param(path)
            c.SCSR.RESIDUAL_SCALE = 0.1
            with self.assertRaises(ValueError):
                wrapper(c).load_param(path)
            torch.save({k: v for k, v in m.state_dict().items() if 'scsr' not in k}, path)
            with self.assertRaises(ValueError):
                restored.load_param(path)


if __name__ == '__main__':
    unittest.main()
