"""Paired teacher/student sampling benchmark with a strict timing protocol.

Why this script exists
----------------------
The archived 2.58x number came from four hand-selected cases whose teacher
wall times drifted heavily (per-case ratios 1.73x-3.72x), no peak-memory was
recorded, and the GPU2 teacher timings were missing. This benchmark fixes the
protocol:

  * SAME CASES  - teacher (DDIM200) and student (DDIM50) run on every case;
  * SAME GPU    - one process, CUDA_VISIBLE_DEVICES pinned to one device, and
                  the script refuses (or warns) when other compute processes
                  are resident;
  * SAME BOUNDARY- only the sampling region is timed: prior q_sample plus the
                  full p_sample_loop. Model loading and NIfTI I/O are timed
                  separately and excluded;
  * SYNCHRONIZED- torch.cuda.synchronize() brackets every timed region;
  * MEMORY      - torch.cuda.max_memory_{allocated,reserved} reset per case;
  * WARM-UP     - the first case is an untimed-in-aggregate warm-up (cuDNN
                  autotune / allocator caches); it is still written to the CSV
                  with is_warmup=1;
  * ORDER       - teacher/student order alternates per case to balance thermal
                  and cache drift;
  * STEPS       - posterior updates actually executed are derived from the
                  respaced schedule and recorded.

Outputs (default outputs/ablation/timing_distill_paired/):
  manifest.json            environment, presets, cases, seed, command line
  timing_case_metrics.csv  per case x model timing and memory rows
  summary.json             paired ratios (per-case, median, mean, geometric)

The script never trains anything and never reads 100AC; it only measures the
sampling cost of the teacher and the distilled student.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch as th

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader_scripts.load_pet_2_5D import LoadTestData
from guided_diffusion import dist_util
from guided_diffusion.respace import space_timesteps
from guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
import sample_3D


PRESETS = {
    # Teacher: frozen tri-axis diffusion backbone, 200 retained steps, original prior
    # start t=600 maps to sparse index 120 (121 posterior updates).
    "teacher_ddim200": {
        "model_root": "weights/axis_models",
        "timestep_respacing": "ddim200",
        "prior_original_t": 600,
    },
    # Student: progressive-distillation checkpoint, 50 retained steps, same
    # original prior start t=600 maps to sparse index 30 (31 updates).
    "student_ddim50": {
        "model_root": "weights/student_ddim50",
        "timestep_respacing": "ddim50",
        "prior_original_t": 600,
    },
}

CSV_FIELDS = [
    "model", "case_id", "order", "is_warmup", "gpu_shared",
    "timestep_respacing", "prior_start_t_effective", "steps_executed",
    "sampling_seconds", "prep_seconds",
    "peak_alloc_mb", "peak_reserved_mb",
]


def build_args(preset: dict, case_id: str, prior_root: str, split: str, seed: int):
    defaults = dict(
        clip_denoised=True,
        batch_size=64,
        use_ddim=False,
        out_channels=1,
        model_root=preset["model_root"],
        model_axis=["x", "y", "z"],
        prior_start_t=200,
        load_adj=8,
        avg_start_number=2,
        sample_num=1,
        save_root="benchmark_unused",
        load_prior_root=prior_root,
        data_root="data/udpet",
        split=split,
        save_single=False,
        max_cases=None,
        case_id=case_id,
        use_prior=True,
        allow_no_prior_start=False,
        allow_shared_prior_root=True,
        final_prior_weight=0.0,
        fusion_mode="xstart_agreement",
        fusion_temperature=0.01,
        save_fusion_stats=False,
        save_axis_xstarts=False,
        save_axis_final=False,
        seed=seed,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults["timestep_respacing"] = preset["timestep_respacing"]
    defaults["in_channels"] = defaults["load_adj"] * 2 + 1 + defaults["out_channels"]
    defaults["model_axis"] = defaults.pop("model_axis")
    return argparse.Namespace(**defaults)


def effective_prior_index(respacing: str, original_t: int) -> int:
    retained = sorted(space_timesteps(1000, respacing))
    if original_t not in retained:
        raise ValueError(
            f"original prior t={original_t} not retained by {respacing}; "
            f"nearest={min(retained, key=lambda u: abs(u - original_t))}"
        )
    return retained.index(original_t)


def load_preset_models(args, device):
    models = []
    diffusion = None
    for axis in args.model_axis:
        model_path = os.path.join(args.model_root, f"model_{axis}.pt")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(th.load(model_path, map_location="cpu"))
        model.to(device)
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        model.requires_grad_(False)
        models.append(model)
    return models, diffusion


def gpu_compute_pids(gpu=None):
    """PIDs of CUDA compute processes on the physical GPU (best effort)."""
    cmd = ["nvidia-smi"]
    if gpu is not None:
        cmd += ["-i", str(gpu)]
    cmd += ["--query-compute-apps=pid", "--format=csv,noheader"]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return None
    return {int(x) for x in out.split() if x.strip().isdigit()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--prior-root", default="outputs/priors/refined/val")
    ap.add_argument("--cases", nargs="+",
                    default=[f"{i:04d}" for i in range(30)])
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--output-root", type=Path,
                    default=Path("outputs/ablation/timing_distill_paired"))
    ap.add_argument("--require-exclusive", action="store_true",
                    help="abort if another CUDA process is resident")
    ap.add_argument("--restart", action="store_true",
                    help="do not resume; overwrite existing CSV")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not th.cuda.is_available():
        raise SystemExit("CUDA unavailable; this benchmark must run on one GPU")
    device = th.cuda.current_device()
    own_pid = os.getpid()
    dist_util.setup_dist()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "timing_case_metrics.csv"
    done = set()
    write_header = True
    if csv_path.exists() and not args.restart:
        with csv_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["model"], r["case_id"]))
        write_header = False

    # Precompute sparse indices once for the manifest / step counts.
    steps_executed = {
        name: effective_prior_index(p["timestep_respacing"], p["prior_original_t"]) + 1
        for name, p in PRESETS.items()
    }

    env = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "torch": th.__version__,
        "cuda_runtime": th.version.cuda,
        "gpu_index_env": args.gpu,
        "gpu_name": th.cuda.get_device_name(0),
        "gpu_total_memory_mb": round(th.cuda.get_device_properties(0).total_memory / 1e6, 1),
        "split": args.split,
        "prior_root": args.prior_root,
        "cases": args.cases,
        "seed": args.seed,
        "per_case_seed": "seed + dataset_index, reset before every timed run",
        "timing_boundary": "cuda-synchronized wall time of q_sample + p_sample_loop; "
                           "model load and NIfTI I/O excluded",
        "presets": PRESETS,
        "steps_executed": steps_executed,
        "argv": sys.argv,
    }
    try:
        env["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        env["driver"] = None
    (args.output_root / "manifest.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: env[k] for k in ("gpu_name", "torch", "cuda_runtime",
                                          "steps_executed")}, indent=2))

    # Load both presets once; reuse across all cases.
    loaded = {}
    for name, preset in PRESETS.items():
        sargs = build_args(preset, args.cases[0], args.prior_root, args.split, args.seed)
        print(f"loading {name}: {preset['model_root']}")
        loaded[name] = (*load_preset_models(sargs, device), sargs)

    test_dir = os.path.join("data/udpet", args.split)
    test_input = LoadTestData(root_dir=test_dir, load_adj=8)
    name_by_case = {test_input.get_name(i).rstrip("_"): i
                    for i in range(len(test_input))}

    def run_case(model_name: str, case_id: str, order: int, is_warmup: bool):
        models, diffusion, sargs = loaded[model_name]
        preset = PRESETS[model_name]
        eff_t = effective_prior_index(preset["timestep_respacing"],
                                      preset["prior_original_t"])
        if case_id not in name_by_case:
            raise ValueError(f"case {case_id} not found under {test_dir}")
        idx = name_by_case[case_id]
        test_input.idx = idx
        shape = (sargs.image_size, sargs.image_size, test_input.get_zsize())

        # --- untimed preparation (NIfTI I/O) ---
        t0 = time.perf_counter()
        patient_id = test_input.get_name(idx)
        prior_path = os.path.join(args.prior_root, f"{patient_id}umap_pred.nii")
        if not os.path.exists(prior_path):
            alt = os.path.join(args.prior_root, f"{patient_id.rstrip('_')}umap_pred.nii")
            prior_path = alt if os.path.exists(alt) else None
        prior_numpy = nib.load(prior_path).get_fdata() if prior_path else None
        if prior_numpy is not None and prior_numpy.shape != shape:
            prior_numpy = sample_3D.resize_xy_to_shape(prior_numpy, shape)
        prep_seconds = time.perf_counter() - t0

        # Same RNG state for teacher and student on the same case, so the
        # paired runs draw the same initial noise stream (per-case seed =
        # global seed + case ordinal).
        case_seed = args.seed + idx
        np.random.seed(case_seed)
        th.manual_seed(case_seed)
        th.cuda.manual_seed_all(case_seed)

        th.cuda.synchronize()
        th.cuda.reset_peak_memory_stats(device)

        # --- timed region: prior q_sample + full reverse sampling loop ---
        t_start = time.perf_counter()
        prior = th.zeros(shape, device=device)
        if prior_numpy is not None:
            prior[:, :, :test_input.get_original_z()] = th.from_numpy(
                prior_numpy[:, :, :test_input.get_original_z()].astype(np.float32)
            ).to(device)
        noisy_priors = []
        for _ in range(sargs.avg_start_number):
            if eff_t < diffusion.num_timesteps:
                noisy_priors.append(
                    diffusion.q_sample(prior, th.tensor(eff_t, device=device)))
            else:
                noisy_priors.append(th.randn(shape, device=device))
        # The sampled volume is discarded: only its wall time is measured.
        diffusion.p_sample_loop(
            models,
            sargs.model_axis,
            test_input,
            shape,
            sargs.batch_size,
            eff_t,
            noise=noisy_priors,
            clip_denoised=sargs.clip_denoised,
            model_kwargs={},
            fusion_mode=sargs.fusion_mode,
            fusion_temperature=sargs.fusion_temperature,
            fusion_weight_callback=None,
            axis_xstart_callback=None,
            axis_final_callback=None,
        )
        th.cuda.synchronize()
        sampling_seconds = time.perf_counter() - t_start
        peak_alloc_mb = th.cuda.max_memory_allocated(device) / 1e6
        peak_reserved_mb = th.cuda.max_memory_reserved(device) / 1e6

        pids = gpu_compute_pids(args.gpu)
        gpu_shared = None if pids is None else bool(pids - {own_pid})
        if args.require_exclusive and gpu_shared:
            raise SystemExit(
                f"other CUDA PIDs resident on GPU: {sorted(pids - {own_pid})}; "
                "free the GPU or rerun without --require-exclusive")

        return {
            "model": model_name,
            "case_id": case_id,
            "order": order,
            "is_warmup": int(is_warmup),
            "gpu_shared": gpu_shared,
            "timestep_respacing": preset["timestep_respacing"],
            "prior_start_t_effective": eff_t,
            "steps_executed": steps_executed[model_name],
            "sampling_seconds": round(sampling_seconds, 3),
            "prep_seconds": round(prep_seconds, 3),
            "peak_alloc_mb": round(peak_alloc_mb, 1),
            "peak_reserved_mb": round(peak_reserved_mb, 1),
        }

    csv_fh = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    rows = []
    for i, case_id in enumerate(args.cases):
        is_warmup = i == 0
        order = ["teacher_ddim200", "student_ddim50"] if i % 2 == 0 else \
                ["student_ddim50", "teacher_ddim200"]
        for k, model_name in enumerate(order):
            if (model_name, case_id) in done:
                print(f"skip done {model_name} {case_id}")
                continue
            print(f"\n[{i+1}/{len(args.cases)}] {model_name} case {case_id} "
                  f"(warmup={is_warmup})")
            row = run_case(model_name, case_id, order=k, is_warmup=is_warmup)
            writer.writerow(row)
            csv_fh.flush()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    csv_fh.close()

    # ---- aggregate from the CSV so resumed rows are included ----
    all_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    timed = [r for r in all_rows if r["is_warmup"] == "0"]

    def stats(model, field, cast=float):
        vals = [cast(r[field]) for r in timed if r["model"] == model and r[field] != ""]
        if not vals:
            return None
        vals_sorted = sorted(vals)
        return {"n": len(vals), "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals_sorted)),
                "min": vals_sorted[0], "max": vals_sorted[-1]}

    by_model = {}
    for model in PRESETS:
        by_model[model] = {
            "sampling_seconds": stats(model, "sampling_seconds"),
            "peak_alloc_mb": stats(model, "peak_alloc_mb"),
        }

    trows = {r["case_id"]: r for r in timed if r["model"] == "teacher_ddim200"}
    srows = {r["case_id"]: r for r in timed if r["model"] == "student_ddim50"}
    paired = []
    for cid in sorted(set(trows) & set(srows)):
        tt = float(trows[cid]["sampling_seconds"])
        ss = float(srows[cid]["sampling_seconds"])
        paired.append({
            "case_id": cid,
            "teacher_s": round(tt, 3),
            "student_s": round(ss, 3),
            "speedup_per_case": round(tt / ss, 3),
        })
    ratios = [p["speedup_per_case"] for p in paired]
    summary = {"by_model": by_model, "paired": paired}
    if ratios:
        t_mean = by_model["teacher_ddim200"]["sampling_seconds"]["mean"]
        s_mean = by_model["student_ddim50"]["sampling_seconds"]["mean"]
        summary["speedup"] = {
            "ratio_of_means": round(t_mean / s_mean, 3),
            "ratio_of_medians": round(
                by_model["teacher_ddim200"]["sampling_seconds"]["median"]
                / by_model["student_ddim50"]["sampling_seconds"]["median"], 3),
            "geomean_per_case": round(float(np.exp(np.log(ratios).mean())), 3),
            "per_case_min": min(ratios),
            "per_case_max": max(ratios),
        }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n===== summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
