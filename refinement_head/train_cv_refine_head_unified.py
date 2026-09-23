"""Unified refinement-head comparison and head-only baselines.

Reviewer isolation experiments (Exp 1 + Exp 2):

  Exp 1 (unified head, single trajectory): the SAME RefineHead3D protocol is
  trained on inputs built from four fusion-rule outputs (all fused at t=800,
  frozen teacher):
      madm           archived "mean" fusion (per-step equal-weight views)
      xstart_mean    20260912 same-protocol equal-weight clean-state fusion
      xstart_median  20260912 voxelwise median
      xstart_fixed   20260912 self-inclusive fixed-tau softmax
      loo            archived xstart_agreement (LOO reliability fusion)
  For every arm: run1 = run2 = fused_t800, cond = normalized 5NAC, unc =
  pairwise axis disagreement map (identical across arms, per-axis predictions
  are the same), ref/mask as usual. Arms therefore differ ONLY in the fused
  input volume -> any post-head difference isolates the fusion rule itself.

  Exp 2 (head-only baselines): the head cannot bypass the diffusion model if
  it fails to reconstruct AC-SDPET from non-diffusion inputs alone:
      cgan      run1=run2=normalized refined cGAN prior,   unc=zeros
      nac       run1=run2=normalized 5NAC,                 unc=zeros
      shuffle   run1=run2=voxel-shuffled LOO t800 output,  unc=zeros

Protocol is byte-for-byte the archived main head: 5-fold CV on val30 (3000
steps/fold, seed=k, 64^3 patch, fg_prob 0.75, bs 2, Adam 1e-4, cosine), final
head on all 30 val cases (4000 steps, seed=100), then applied to test36.

Self-contained: no dependency on the lost data_v2/common modules.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader_scripts.load_pet_2_5D import get_mask, normalize_ac  # noqa: E402
from refine_head import RefineHead3D, infer_volume  # noqa: E402

ARMS = ("madm", "xstart_mean", "xstart_median", "xstart_fixed", "loo",
        "cgan", "nac", "shuffle")
EXP1_ARMS = ("madm", "xstart_mean", "xstart_median", "xstart_fixed", "loo")

# Fused-output roots per (arm, split) for the t=800 single-trajectory inputs.
# Layout produced by run_five_case_comparison.py via scripts/20_run_loo_ablation.sh:
#   outputs/ablation/<setting>/<fusion_mode>/case_<cid>/adj8_models_xyz/
#       noise_2_priort_40_comb/<cid>_pred.nii
ABLATION_ROOT = {
    "val": "outputs/ablation/val30_t800_xstart",
    "test": "outputs/ablation/test36_t800_xstart",
}
FUSED_ROOTS = {
    "val": {
        "madm": f"{ABLATION_ROOT['val']}/mean",
        "xstart_mean": f"{ABLATION_ROOT['val']}/xstart_mean",
        "xstart_median": f"{ABLATION_ROOT['val']}/xstart_median",
        "xstart_fixed": f"{ABLATION_ROOT['val']}/xstart_fixed",
        "loo": f"{ABLATION_ROOT['val']}/xstart_agreement",
    },
    "test": {
        "madm": f"{ABLATION_ROOT['test']}/mean",
        "xstart_mean": f"{ABLATION_ROOT['test']}/xstart_mean",
        "xstart_median": f"{ABLATION_ROOT['test']}/xstart_median",
        "xstart_fixed": f"{ABLATION_ROOT['test']}/xstart_fixed",
        "loo": f"{ABLATION_ROOT['test']}/xstart_agreement",
    },
}
# Per-axis clean-state predictions used for the shared disagreement map.
# Produced by the main sampling script (scripts/04_sample_tricore.sh) with
# --save_axis_xstarts true.
AXIS_ROOTS = {
    "val": "outputs/samples/val30_t800_loo"
           "/adj8_models_xyz/noise_2_priort_40_axis_xstarts",
    "test": "outputs/samples/test36_t800_loo"
            "/adj8_models_xyz/noise_2_priort_40_axis_xstarts",
}
PRIOR_ROOTS = {
    "val": "outputs/priors/refined/val",
    "test": "outputs/priors/refined/test",
}


def case_ids(split: str) -> list[str]:
    n = 30 if split == "val" else 36
    return [f"{i:04d}" for i in range(n)]


def ref_path(split, cid):
    return ROOT / "data" / "udpet" / split / "100AC" / f"{cid}_100_AC.nii"


def nac_path(split, cid):
    return ROOT / "data" / "udpet" / split / "5NAC" / f"{cid}_5_NAC.nii"


def fused_pred(arm, split, cid):
    if arm == "cgan":
        return ROOT / PRIOR_ROOTS[split] / f"{cid}umap_pred.nii"
    if arm == "nac":
        return nac_path(split, cid)
    if arm == "shuffle":
        return fused_pred("loo", split, cid)
    return ROOT / FUSED_ROOTS[split][arm] / f"case_{cid}" \
        / "adj8_models_xyz" / "noise_2_priort_40_comb" / f"{cid}_pred.nii"


def disagreement_map(split, cid):
    """Shared pairwise disagreement of the three per-axis clean predictions."""
    xs = {}
    for a in ("x", "y", "z"):
        p = ROOT / AXIS_ROOTS[split] / f"{cid}_{a}_xstart.nii"
        xs[a] = nib.load(str(p)).get_fdata(dtype=np.float32)
    d = (np.abs(xs["x"] - xs["y"]) + np.abs(xs["y"] - xs["z"])
         + np.abs(xs["x"] - xs["z"])) / 3.0
    return d.astype(np.float32)


def build_cache(arm, split, cid, cache_dir):
    """Return dict of (B-less) arrays for one case; cache to disk on first use."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {c: cache_dir / f"{arm}_{split}_{cid}_{c}.npy"
             for c in ("run1", "run2", "cond", "unc", "ref", "mask")}
    if all(f.exists() for f in files.values()):
        return files

    nac = nib.load(str(nac_path(split, cid))).get_fdata(dtype=np.float32)
    mask = get_mask(nac).astype(np.float32)
    ref = normalize_ac(
        nib.load(str(ref_path(split, cid))).get_fdata(dtype=np.float32),
        mask).astype(np.float32)
    cond = normalize_ac(nac, mask).astype(np.float32)

    p = fused_pred(arm, split, cid)
    if not p.exists():
        raise FileNotFoundError(f"{arm} {split} {cid}: {p}")
    run = nib.load(str(p)).get_fdata(dtype=np.float32)
    if run.shape != ref.shape:
        raise ValueError(f"shape mismatch {p}: {run.shape} vs ref {ref.shape}")

    if arm in EXP1_ARMS:
        run1 = run2 = run
        unc = disagreement_map(split, cid)
    else:
        run1 = run2 = run
        unc = np.zeros_like(run)
    if arm == "shuffle":
        rng = np.random.default_rng(20260915 + int(cid))
        flat = run1.ravel()
        idx = rng.permutation(flat.size)
        run1 = run2 = flat[idx].reshape(run.shape).astype(np.float32)

    arrays = {"run1": run1.astype(np.float32), "run2": run2.astype(np.float32),
              "cond": cond, "unc": unc, "ref": ref, "mask": mask}
    for c, arr in arrays.items():
        np.save(files[c], arr)
    return files


class Patchset:
    """Foreground-biased patch sampler (identical to the archived head)."""

    def __init__(self, cases, patch=64, fg_prob=0.75, seed=0):
        self.patch = patch
        self.fg_prob = fg_prob
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.fg_idx = []
        for c in cases:
            self.data.append({k: np.load(str(v), mmap_mode="r")
                              for k, v in c.items()})
            m = np.load(str(c["mask"]))
            self.fg_idx.append(np.argwhere(m > 0.5))

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
        else:
            d = self.data[0]
            shape = d["run1"].shape
            z0 = max(0, (shape[0] - self.patch) // 2)
            y0 = max(0, (shape[1] - self.patch) // 2)
            x0 = max(0, (shape[2] - self.patch) // 2)
            z1, y1, x1 = z0 + self.patch, y0 + self.patch, x0 + self.patch
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        x = np.stack([np.asarray(d["run1"][sl]), np.asarray(d["run2"][sl]),
                      np.asarray(d["cond"][sl]), np.asarray(d["unc"][sl])]
                     ).astype(np.float32)
        y = np.asarray(d["ref"][sl], dtype=np.float32)
        w = 1.0 + 1.5 * np.asarray(d["mask"][sl], dtype=np.float32)
        return x, y, w

    def sample_batch(self, bs):
        xs, ys, ws = zip(*(self._sample_one() for _ in range(bs)))
        return (torch.from_numpy(np.stack(xs)),
                torch.from_numpy(np.stack(ys))[:, None],
                torch.from_numpy(np.stack(ws))[:, None])


def train_head(model, cases, device, steps, seed):
    sampler = Patchset(cases, seed=seed)
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
def eval_case(model, split, cid, cache, device, out_dir):
    d = {k: np.load(str(v)) for k, v in cache.items()}
    x = torch.from_numpy(np.stack(
        [d["run1"], d["run2"], d["cond"], d["unc"]])[None])
    refined = infer_volume(model, x, device)
    case_dir = Path(out_dir) / f"case_{cid}"
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / f"{cid}_refined.nii"
    nib.save(nib.Nifti1Image(refined.astype(np.float32), np.eye(4)), str(out))
    print(f"  [{split}] {cid}: saved {out.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "refine_head_unified_cache")
    ap.add_argument("--out-root", type=Path,
                    default=Path("outputs/head_unified"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--final-steps", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    device = torch.device(
        a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    out = a.out_root / a.arm
    out.mkdir(parents=True, exist_ok=True)
    print(f"[unified] arm={a.arm} device={device} out={out}", flush=True)

    def cv_path(cid):
        return out / "val_cv" / f"case_{cid}" / f"{cid}_refined.nii"

    ids = case_ids("val")
    caches = {cid: build_cache(a.arm, "val", cid, a.cache_dir) for cid in ids}

    completed = [cid for cid in ids if cv_path(cid).exists()]
    if a.resume and len(completed) == len(ids):
        print(f"[unified] resume: all {len(ids)} val_cv outputs exist; "
              "skipping CV.", flush=True)
    else:
        perm = np.random.default_rng(0).permutation(len(ids))
        fold_ids = np.array_split(perm, a.folds)
        for k, fold in enumerate(fold_ids):
            held = [ids[i] for i in fold]
            train = [caches[c] for c in ids if c not in held]
            print(f"\n[unified] fold {k + 1}/{a.folds}: hold out {held}", flush=True)
            model = RefineHead3D().to(device)
            train_head(model, train, device, a.steps, seed=k)
            for cid in held:
                eval_case(model, "val", cid, caches[cid], device,
                          out / "val_cv")

    summary = {
        "arm": a.arm,
        "folds": a.folds, "steps": a.steps,
    }
    print("\n====== unified head val 5-fold CV ======")
    print(f"  {a.arm}: folds={a.folds}  steps={a.steps}", flush=True)

    # final head on all 30 val cases -> test36 application
    final_path = out / "final_head.pt"
    if a.resume and final_path.exists():
        print("[unified] resume: keeping existing final head.", flush=True)
        model = RefineHead3D().to(device)
        model.load_state_dict(torch.load(final_path, map_location=device))
    else:
        print(f"\n[unified] training final head on all {len(ids)} val cases ...",
              flush=True)
        final = RefineHead3D().to(device)
        train_head(final, [caches[c] for c in ids], device, a.final_steps, seed=100)
        torch.save(final.state_dict(), final_path)
        model = final
    model.eval()

    for cid in case_ids("test"):
        cache = build_cache(a.arm, "test", cid, a.cache_dir)
        eval_case(model, "test", cid, cache, device, out / "test36")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n====== {a.arm} test36 (final head) ======")
    print(f"[unified] outputs -> {out}")


if __name__ == "__main__":
    main()
