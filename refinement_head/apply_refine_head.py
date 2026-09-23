"""Apply the trained TriCoRe-R refinement head to a held-out split.

Loads ``outputs/head/final_head.pt`` (produced by ``train_refine_head.py``),
runs full-volume inference for every test case on the four cached channels
[p_800, p_600, 5NAC, agreement] and saves refined NIfTI predictions.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--head", type=Path, default=Path("outputs/head/final_head.pt"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "tricore_refine_head_cache")
    ap.add_argument("--out-root", type=Path,
                    default=Path("outputs/head/test36_refined"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--zero-uncertainty", action="store_true",
                    help="ablation: replace the agreement-map channel with 0")
    a = ap.parse_args()

    device = torch.device(
        a.device if a.device == "cuda" and torch.cuda.is_available()
        else "cpu")
    model = RefineHead3D().to(device)
    model.load_state_dict(torch.load(a.head, map_location=device))
    model.eval()

    a.out_root.mkdir(parents=True, exist_ok=True)
    n_cases = 0
    for cid in case_ids(a.split):
        cache = build_cache(a.split, cid, a.cache_dir)
        d = {k: np.load(str(v)) for k, v in cache.items()}
        unc = np.zeros_like(d["unc"]) if a.zero_uncertainty else d["unc"]
        x = torch.from_numpy(np.stack(
            [d["run1"], d["run2"], d["cond"], unc])[None])
        refined = infer_volume(model, x, device)

        import nibabel as nib
        case_out = a.out_root / f"case_{cid}"
        case_out.mkdir(parents=True, exist_ok=True)
        out_path = case_out / f"{cid}_refined.nii"
        nib.save(nib.Nifti1Image(refined.astype(np.float32), np.eye(4)),
                 str(out_path))
        n_cases += 1
        print(f"[{a.split}] {cid}: saved {out_path.name}", flush=True)

    summary = {
        "split": a.split, "n_cases": n_cases,
        "head": str(a.head),
    }
    (a.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("\n====== TriCoRe-R {} summary ======".format(a.split))
    print(f"  refined {n_cases} cases")
    print(f"[head] predictions -> {a.out_root}")


if __name__ == "__main__":
    main()
