"""Train the TriCoRe-R four-channel refinement head.

Inputs per case (see ``data.py``): [p_800, p_600, 5NAC, agreement-map].

Protocol (frozen before test36 evaluation, identical to the paper):
  * val30 five-fold CV: default_rng(0).permutation(30), 5 groups of 6,
    3000 Adam steps per fold (lr 1e-4, cosine), seed = fold index,
    64^3 patches, batch size 2, foreground sampling probability 0.75,
    foreground-weighted L1 (weight 2.5 inside the mask, 1 outside);
  * every CV prediction comes from a case held out of that fold's training;
  * after CV, one final head is trained on ALL 30 val cases (4000 steps,
    seed 100) and saved for ``apply_refine_head.py`` on test36.

The refinement head is zero-initialized so that at step 0 it outputs the
plain average of the two trajectories.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_cache, case_ids  # noqa: E402
from refine_head import RefineHead3D, infer_volume  # noqa: E402


class Patchset:
    """Foreground-biased 64^3 patch sampler over cached case arrays."""

    def __init__(self, cases, patch=64, fg_prob=0.75, seed=0,
                 zero_uncertainty=False):
        self.patch = patch
        self.fg_prob = fg_prob
        self.zero_uncertainty = zero_uncertainty
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.fg_idx = []
        for c in cases:
            self.data.append({k: np.load(str(v), mmap_mode="r")
                              for k, v in c.items()})
            self.fg_idx.append(np.argwhere(np.load(str(c["mask"])) > 0.5))

    def _sample_one(self):
        half = self.patch // 2
        for _ in range(50):
            i = int(self.rng.integers(len(self.data)))
            d = self.data[i]
            shape = d["run1"].shape
            if self.rng.random() < self.fg_prob and len(self.fg_idx[i]):
                cz, cy, cx = self.fg_idx[i][
                    self.rng.integers(len(self.fg_idx[i]))]
            else:
                cz = int(self.rng.integers(half, shape[0] - half))
                cy = int(self.rng.integers(half, shape[1] - half))
                cx = int(self.rng.integers(half, shape[2] - half))
            z0, y0, x0 = int(cz) - half, int(cy) - half, int(cx) - half
            z1, y1, x1 = z0 + self.patch, y0 + self.patch, x0 + self.patch
            if 0 <= z0 and 0 <= y0 and 0 <= x0 and \
               z1 <= shape[0] and y1 <= shape[1] and x1 <= shape[2]:
                break
        else:  # fallback: center patch
            d = self.data[0]
            shape = d["run1"].shape
            z0 = max(0, (shape[0] - self.patch) // 2)
            y0 = max(0, (shape[1] - self.patch) // 2)
            x0 = max(0, (shape[2] - self.patch) // 2)
            z1, y1, x1 = z0 + self.patch, y0 + self.patch, x0 + self.patch
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        unc = (np.zeros_like(d["unc"][sl]) if self.zero_uncertainty
               else np.asarray(d["unc"][sl]))
        x = np.stack([np.asarray(d["run1"][sl]), np.asarray(d["run2"][sl]),
                      np.asarray(d["cond"][sl]), unc]).astype(np.float32)
        y = np.asarray(d["ref"][sl], dtype=np.float32)
        w = 1.0 + 1.5 * np.asarray(d["mask"][sl], dtype=np.float32)
        return x, y, w

    def sample_batch(self, bs):
        xs, ys, ws = zip(*(self._sample_one() for _ in range(bs)))
        return (torch.from_numpy(np.stack(xs)),
                torch.from_numpy(np.stack(ys))[:, None],
                torch.from_numpy(np.stack(ws))[:, None])


def train_head(model, cases, device, steps, seed, zero_uncertainty=False):
    sampler = Patchset(cases, seed=seed, zero_uncertainty=zero_uncertainty)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for step in range(1, steps + 1):
        x, y, w = sampler.sample_batch(2)
        x, y, w = x.to(device), y.to(device), w.to(device)
        loss = (w * (model(x) - y).abs()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step == 1 or step % 200 == 0:
            print(f"    step {step:>5d}/{steps}  wL1={loss.item():.5f}",
                  flush=True)


@torch.no_grad()
def eval_case(model, cid, cache, out_root, device,
              zero_uncertainty=False):
    import nibabel as nib
    d = {k: np.load(str(v)) for k, v in cache.items()}
    unc = np.zeros_like(d["unc"]) if zero_uncertainty else d["unc"]
    x = torch.from_numpy(np.stack(
        [d["run1"], d["run2"], d["cond"], unc])[None])
    refined = infer_volume(model, x, device)
    case_out = Path(out_root) / f"case_{cid}"
    case_out.mkdir(parents=True, exist_ok=True)
    out_path = case_out / f"{cid}_refined.nii"
    nib.save(nib.Nifti1Image(refined.astype(np.float32), np.eye(4)),
             str(out_path))
    print(f"  [cv] {cid}: saved {out_path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-for-cv", default="val")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "tricore_refine_head_cache")
    ap.add_argument("--out-root", type=Path,
                    default=Path("outputs/head"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--final-steps", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--zero-uncertainty", action="store_true",
                    help="ablation: replace the agreement-map channel with 0")
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing final_head.pt / CV predictions")
    a = ap.parse_args()

    device = torch.device(
        a.device if a.device == "cuda" and torch.cuda.is_available()
        else "cpu")
    a.out_root.mkdir(parents=True, exist_ok=True)

    ids = case_ids("val")
    print(f"[head] val cases: {len(ids)}  device={device}")
    caches = {cid: build_cache("val", cid, a.cache_dir) for cid in ids}

    cv_root = a.out_root / "val_cv"
    if a.resume and (a.out_root / "final_head.pt").exists() and \
            all((cv_root / f"case_{cid}" / f"{cid}_refined.nii").exists()
                for cid in ids):
        print("[head] resume: all CV outputs and final head exist; "
              "skipping CV.")
    else:
        perm = np.random.default_rng(0).permutation(len(ids))
        fold_ids = np.array_split(perm, a.folds)
        for k, fold in enumerate(fold_ids):
            held = [ids[i] for i in fold]
            train = [caches[c] for c in ids if c not in held]
            print(f"\n[head] fold {k + 1}/{a.folds}: hold out {held}",
                  flush=True)
            model = RefineHead3D().to(device)
            train_head(model, train, device, a.steps, seed=k,
                       zero_uncertainty=a.zero_uncertainty)
            for cid in held:
                eval_case(model, cid, caches[cid], cv_root,
                          device, zero_uncertainty=a.zero_uncertainty)

    summary = {
        "folds": a.folds, "steps": a.steps,
        "zero_uncertainty": a.zero_uncertainty,
    }
    print("\n====== TriCoRe-R val30 5-fold CV ======")
    print(f"  folds={a.folds}  steps={a.steps}  "
          f"zero_uncertainty={a.zero_uncertainty}")

    final_path = a.out_root / "final_head.pt"
    if a.resume and final_path.exists():
        print(f"[head] keeping existing final head -> {final_path}")
    else:
        print(f"\n[head] training final head on all {len(ids)} val cases ...",
              flush=True)
        final = RefineHead3D().to(device)
        train_head(final, [caches[c] for c in ids], device,
                   a.final_steps, seed=100,
                   zero_uncertainty=a.zero_uncertainty)
        torch.save(final.state_dict(), final_path)
        print(f"[head] saved final head -> {final_path}")

    (a.out_root / "val_cv_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
