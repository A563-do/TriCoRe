"""Small 3D residual refinement head.

Input (B,4,D,H,W): run1, run2, 5NAC condition, pairwise-disagreement map.
Output (B,1,D,H,W): refined prediction in [0,1].

Key design: the output conv is ZERO-initialized and the base is a learned
per-voxel blend of the two runs (mix conv zero-init -> sigmoid=0.5), so at
initialization the model is exactly the ensemble mean of the two runs.
Training can only move it away from that point if it reduces L1 loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineHead3D(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.mix = nn.Conv3d(2, 1, 1)
        self.inp = nn.Conv3d(4, ch, 3, padding=1)
        self.body = nn.ModuleList(
            [nn.Conv3d(ch, ch, 3, padding=d, dilation=d)
             for d in (1, 2, 4, 8, 4, 2, 1)])
        self.out = nn.Conv3d(ch, 1, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):  # x: (B,4,D,H,W)
        w = torch.sigmoid(self.mix(x[:, :2]))
        base = w * x[:, 0:1] + (1.0 - w) * x[:, 1:2]
        h = F.relu(self.inp(x))
        for conv in self.body:
            h = h + F.relu(conv(h))
        return torch.clamp(base + self.out(h), 0.0, 1.0)


@torch.no_grad()
def infer_volume(model, x, device, slab=96, overlap=24):
    """Full-volume inference; falls back to overlapping z-slabs if needed.

    x: torch tensor (1,4,D,H,W). Returns (D,H,W) numpy float32 in [0,1].
    """
    model.eval()
    D = x.shape[2]
    if D <= slab:
        return model(x.to(device))[0, 0].clamp(0, 1).cpu().numpy()
    step = slab - overlap
    starts = list(range(0, D - slab + 1, step)) or [0]
    if starts[-1] != D - slab:
        starts.append(D - slab)
    out = torch.zeros(1, 1, *x.shape[2:], dtype=torch.float32)
    acc = torch.zeros(D, dtype=torch.float32)
    for z in starts:
        out[:, :, z:z + slab] += model(x[:, :, z:z + slab].to(device)).cpu()
        acc[z:z + slab] += 1.0
    return (out[0, 0] / acc.view(-1, 1, 1)).clamp(0, 1).numpy()
