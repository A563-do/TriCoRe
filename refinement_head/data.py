"""Data layer for the four-channel TriCoRe-R refinement head.

For every case it builds (and caches as ``.npy``) the six arrays consumed by
``RefineHead3D``:

  run1 : p_800, LOO-fused trajectory starting at original t=800
         (sparse index 40 of the 50-step respaced axis)
  run2 : p_600, LOO-fused trajectory starting at original t=600
         (sparse index 30)
  cond : y, the normalized 5% NAC-LDPET condition
  unc  : a(v) = exp(-d_pair(v) / 0.01), the pairwise axis-agreement map
         built from the aligned per-axis clean predictions (paper Eq. 7-8)
  ref  : normalized 100AC reference (training/evaluation only)
  mask : foreground mask derived from 5% NAC

The trajectory volumes and per-axis clean predictions are the outputs of
``sample_3D.py`` (run with ``--save_axis_xstarts true``):

  <run-root>/adj8_models_xyz/noise_2_priort_<sparse_index>_comb/
      <cid>_pred.nii
  <run-root>/adj8_models_xyz/noise_2_priort_<sparse_index>_axis_xstarts/
      <cid>_{x,y,z}_xstart.nii

All metrics and supervision live in the same tanh(x/5) normalized space used
everywhere else in the project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader_scripts.load_pet_2_5D import get_mask, normalize_ac  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "udpet"
T800_ROOTS = {
    "val": REPO_ROOT / "outputs" / "samples" / "val30_t800_loo",
    "test": REPO_ROOT / "outputs" / "samples" / "test36_t800_loo",
}
T600_ROOTS = {
    "val": REPO_ROOT / "outputs" / "samples" / "val30_t600_loo",
    "test": REPO_ROOT / "outputs" / "samples" / "test36_t600_loo",
}
AXES = ("x", "y", "z")
CHANNELS = ("run1", "run2", "cond", "unc", "ref", "mask")
AGREEMENT_TAU = 0.01


def case_ids(split: str) -> list[str]:
    """Zero-padded case ids: 30 validation cases or 36 test cases."""
    n = 30 if split == "val" else 36
    return [f"{i:04d}" for i in range(n)]


def list_case_ids(split: str, data_root: Path = DATA_ROOT) -> list[str]:
    """List case ids actually present under ``data_root/<split>/5NAC``."""
    nac_dir = Path(data_root) / split / "5NAC"
    return sorted(p.name.split("_")[0] for p in nac_dir.glob("*_5_NAC.nii"))


def case_paths(split: str, cid: str, data_root: Path = DATA_ROOT) -> dict:
    base = Path(data_root) / split
    return {
        "nac": base / "5NAC" / f"{cid}_5_NAC.nii",
        "ref": base / "100AC" / f"{cid}_100_AC.nii",
    }


def _pred_path(run_root: Path, cid: str, sparse_index: int) -> Path:
    return (Path(run_root) / "adj8_models_xyz"
            / f"noise_2_priort_{sparse_index}_comb" / f"{cid}_pred.nii")


def _axis_path(run_root: Path, cid: str, sparse_index: int, axis: str) -> Path:
    return (Path(run_root) / "adj8_models_xyz"
            / f"noise_2_priort_{sparse_index}_axis_xstarts"
            / f"{cid}_{axis}_xstart.nii")


def agreement_map(run_root: Path, cid: str, sparse_index: int) -> np.ndarray:
    """a(v) = exp(-d_pair(v)/tau), Eq. 7-8, from aligned per-axis predictions."""
    xs = [
        nib.load(str(_axis_path(run_root, cid, sparse_index, a))).get_fdata(
            dtype=np.float32)
        for a in AXES
    ]
    d_pair = (np.abs(xs[0] - xs[1]) + np.abs(xs[1] - xs[2])
              + np.abs(xs[0] - xs[2])) / 3.0
    return np.exp(-d_pair / AGREEMENT_TAU).astype(np.float32)


def build_cache(split, cid, cache_dir, t800_root=None, t600_root=None,
                data_root=DATA_ROOT):
    """Return ``{channel: npy_path}`` for one case; build on first use."""
    t800_root = Path(t800_root or T800_ROOTS[split])
    t600_root = Path(t600_root or T600_ROOTS[split])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {c: cache_dir / f"{split}_{cid}_{c}.npy" for c in CHANNELS}
    if all(f.exists() for f in files.values()):
        return files

    run1 = nib.load(str(_pred_path(t800_root, cid, 40))).get_fdata(
        dtype=np.float32)
    run2 = nib.load(str(_pred_path(t600_root, cid, 30))).get_fdata(
        dtype=np.float32)
    unc = agreement_map(t800_root, cid, 40)

    p = case_paths(split, cid, data_root)
    nac = nib.load(str(p["nac"])).get_fdata(dtype=np.float32)
    mask = get_mask(nac).astype(np.float32)
    cond = normalize_ac(nac, mask).astype(np.float32)
    ref = normalize_ac(
        nib.load(str(p["ref"])).get_fdata(dtype=np.float32), mask
    ).astype(np.float32)

    if not (run1.shape == run2.shape == cond.shape == ref.shape):
        raise ValueError(
            f"{split} {cid}: shape mismatch run1={run1.shape} "
            f"run2={run2.shape} cond={cond.shape} ref={ref.shape}")

    arrays = {"run1": run1, "run2": run2, "cond": cond, "unc": unc,
              "ref": ref, "mask": mask}
    for c, arr in arrays.items():
        np.save(files[c], arr.astype(np.float32))
    return files
