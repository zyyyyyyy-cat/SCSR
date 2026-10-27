"""Two-process CPU/Gloo smoke test; no dataset or CUDA required.

Run from WHU: python tests/check_scsr_ddp.py
Checks the existing find_unused_parameters=True policy, including a rank
without TIR. This does not replace a CUDA/NCCL production test.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from config import cfg
from model.backbones.vit_pytorch import TransReID


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(1234)
        c = cfg.SCSR.clone()
        c.ENABLED = True
        c.IN_DIM = 32
        c.SPECTRAL_DIM = 8
        c.CONDITION_DIM = 4
        model = TransReID(img_size=(32, 32), patch_size=8, stride_size=8,
                         embed_dim=32, num_heads=4, depth=8, num_classes=0,
                         scsr_cfg=c, scsr_target_id=3, num_modalities=3)
        model = DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.008)
        for step in range(3):
            optimizer.zero_grad()
            torch.manual_seed(100 + rank + step)
            # First two passes resemble PKM; third deliberately has no TIR on rank 0.
            ids = [1, 2, 3] if step < 2 or rank == 1 else [1, 2, 1]
            result = model(torch.randn(3, 3, 32, 32), modal_label=ids)
            (result * torch.randn_like(result)).mean().backward()
            gradient = model.module.scsr.condition[-1].weight.grad
            assert gradient is not None and gradient.isfinite().all()
            assert gradient.abs().sum() > 0
            gradients = [torch.empty_like(gradient) for _ in range(2)]
            dist.all_gather(gradients, gradient)
            torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
            optimizer.step()
        weight = model.module.scsr.condition[-1].weight.detach()
        weights = [torch.empty_like(weight) for _ in range(2)]
        dist.all_gather(weights, weight)
        torch.testing.assert_close(weights[0], weights[1], atol=0, rtol=0)
        if rank == 0:
            print('PASS: two-rank Gloo SCSR gradients/weights agree, including no-TIR rank')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = (Path(directory) / 'rendezvous').as_uri()
        mp.spawn(worker, args=(rendezvous,), nprocs=2, join=True)
