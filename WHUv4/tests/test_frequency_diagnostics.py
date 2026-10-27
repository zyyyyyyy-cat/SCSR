import sys
from pathlib import Path
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.frequency_core import (FrequencyIntervention, LayerCapture, calibration_alpha,
                                        decompose, descriptor, pair_auc, reconstruct_v41_condition)
from diagnostics.core import Capture
from config import cfg
from model.scsr_v41 import SCSRV41
from diagnostics.frequency_pairs import matched_pairs, relation_pairs, required_scene_pairs
from diagnostics.frequency_evaluation import evaluate_features
from model.scsr_v41 import haar_dwt, haar_idwt


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.norm1 = nn.Identity(); self.attn = nn.Identity()
    def forward(self, x):
        return x + self.attn(self.norm1(x))


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.ModuleList([TinyBlock() for _ in range(12)])
    def forward(self, x):
        for block in self.blocks: x = block(x)
        return x[:, 0]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.base = TinyBase()
    def forward(self, x, mode=1, modal_ids=None):
        return self.base(x)


class FrequencyDiagnosticTests(unittest.TestCase):
    def test_haar_roundtrip_and_descriptors(self):
        x = torch.randn(3, 768, 16, 8)
        bands = haar_dwt(x)
        self.assertTrue(torch.allclose(haar_idwt(*bands), x, atol=5e-7, rtol=5e-7))
        tokens = torch.cat((torch.randn(3, 1, 768), x.flatten(2).transpose(1, 2)), 1)
        recovered = decompose(tokens)
        for kind, width in (('signed_spatial', 24576), ('channel_rms', 768), ('energy', 1), ('energy_ratio', 1)):
            self.assertEqual(descriptor(recovered[1], kind, recovered).shape, (3, width))

    def test_intervention_only_tir_and_hook_removed(self):
        model = TinyModel(); tokens = torch.randn(3, 129, 768); ids = torch.tensor([1, 2, 3])
        baseline = model(tokens, modal_ids=ids)
        with FrequencyIntervention(model, 7, 'LH', 1, .1, ids, collect=True) as intervention:
            with LayerCapture(model, (7,)) as capture:
                changed = model(tokens, modal_ids=ids)
        self.assertTrue(torch.equal(changed[:2], baseline[:2]))
        self.assertTrue(torch.equal(model(tokens, modal_ids=ids), baseline))
        self.assertEqual(len(intervention.records), 1)
        self.assertFalse(torch.equal(capture.values[7][2, 1:], tokens[2, 1:] * 64))

    def test_calibration_cap_and_pair_auc(self):
        tokens = torch.randn(8, 129, 768)
        alpha, raw = calibration_alpha(tokens, 'HH', .01, .2)
        self.assertAlmostEqual(alpha, min(raw, .2))
        self.assertAlmostEqual(pair_auc([.1, .2], [.8, .9]), 1.)
        self.assertAlmostEqual(pair_auc([.5], [.5]), .5)

    def test_aligned_condition_matched_pairs_and_scene_relations(self):
        records = []
        for pid in range(3):
            for modality, mid in (('RGB', 1), ('NIR', 2), ('TIR', 3)):
                for cam in (0, 1, 5, 6):
                    records.append(dict(path='{}_{}_{}'.format(pid, modality, cam), pid=pid, camid=cam,
                                        modal_id=mid, modality=modality, platform='G' if cam < 5 else 'A'))
        pos, neg = matched_pairs(records, 20, 4)
        groups = relation_pairs(records, pos, neg)
        self.assertEqual(sum(len(v['positive']) for v in groups.values()), len(pos))
        self.assertTrue(all(a[0] == b[0] for a, b in zip(pos, neg)))
        sp, sn = required_scene_pairs(records, 4)
        relations = {records[a]['modality'] + ':' + records[b]['modality'] for a, b in sp}
        self.assertTrue({'RGB:TIR', 'NIR:TIR', 'TIR:TIR'} <= relations)
        self.assertEqual(len(sp), len(sn))

    def test_retrieval_camera_filter(self):
        records = [
            dict(path='q', pid=0, camid=0, modality='TIR', platform='G'),
            dict(path='same_cam', pid=0, camid=0, modality='TIR', platform='G'),
            dict(path='positive', pid=0, camid=1, modality='TIR', platform='G'),
            dict(path='negative', pid=1, camid=1, modality='TIR', platform='G')]
        features = torch.tensor([[1., 0.], [1., 0.], [.9, .1], [0., 1.]])
        metric, queries, margins = evaluate_features(features, records, [0], [1, 2, 3], 'x', 'train')
        self.assertEqual(metric['valid_query_count'], 1)
        self.assertTrue(queries[0]['rank1_correct'])
        self.assertGreater(margins[0]['margin'], 0)

    def test_production_v41_condition_reconstruction(self):
        config = cfg.clone(); config.merge_from_file(str(ROOT / 'configs/SCSR_V41.yml'))
        module = SCSRV41(config.SCSR_V41, config.DATASETS.MODALITIES).eval()
        tokens = torch.randn(3, 129, 768); ids = torch.tensor([3, 1, 3])
        capture = Capture(); module.diagnostic_callback = capture.spectral
        with torch.no_grad():
            module(tokens, ids)
        self.assertNotIn('condition', capture.internal)
        condition, rebuilt = reconstruct_v41_condition(module, capture.internal)
        self.assertEqual(condition.shape, (2, 31, 8, 4))
        self.assertTrue(torch.allclose(rebuilt, capture.internal['v'], atol=2e-6, rtol=2e-5))


if __name__ == '__main__':
    unittest.main()
