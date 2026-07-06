
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.datasets_pairs import MultiTaskDataset
from loss.losses import fftLoss
from networks.ddl_arch import DDLRestorationNet
from networks.diffusion_reg import (
    GradientOrthogonalLoss,
    NoiseSchedule,
    ParameterRegularizer,
    compute_denoising_loss,
    compute_pretrain_loss,
)
from networks.moe_adapter import attach_moe_adapters


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"[OK] {name}")


class FakePairDataset(torch.utils.data.Dataset):
    def __init__(self, size, shape=(3, 32, 32)):
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        gt = torch.rand(*self.shape)
        lq = torch.clamp(gt + 0.05 * torch.randn_like(gt), 0, 1)
        return lq, gt, f"fake_{index:04d}.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DDLRestorationNet(
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
    ).to(device)

    x = torch.rand(2, 3, 64, 64, device=device)
    residual = net(x, time=torch.tensor([4, 47], device=device))
    check("residual shape", residual.shape == x.shape)

    attach_moe_adapters(net, num_experts=4, T=50)
    residual_moe = net(x, time=torch.tensor([4, 47], device=device))
    check("MoE forward shape", residual_moe.shape == x.shape)

    schedule = NoiseSchedule(max_sigma=10, T=50, schedule="linear", eps=0.005, device=device)
    loss_pre = compute_pretrain_loss(net, x, schedule)
    check("pre-training loss finite", torch.isfinite(loss_pre).item())

    gt = torch.rand_like(x)
    restored = x - net(x, 47)
    loss_content = F.l1_loss(restored, gt) + 0.1 * fftLoss().to(device)(restored, gt)
    loss_gen = compute_denoising_loss(net, gt, schedule)
    loss_orthog = GradientOrthogonalLoss.compute(net, loss_gen, loss_content)
    loss_reg = ParameterRegularizer(net).loss(net)
    total = loss_content + loss_orthog + loss_reg
    total.backward()
    check("full DDL loss backward", any(p.grad is not None for p in net.parameters()))

    ds = MultiTaskDataset([FakePairDataset(3), FakePairDataset(2)], [4, 47])
    check("multi-task dataset length", len(ds) == 5)
    check("first task timestep", ds[0][3] == 4)
    check("second task timestep", ds[4][3] == 47)

    print("DDL smoke test passed.")


if __name__ == "__main__":
    main()

