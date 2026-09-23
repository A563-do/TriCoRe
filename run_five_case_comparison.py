"""Run a fixed five-case MADM comparison without modifying sample_3D.py."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from guided_diffusion.axis_fusion import XSTART_FUSION_MODES
from guided_diffusion.respace import space_timesteps


DEFAULT_CASES = ("0000", "0001", "0002", "0003", "0004")


def prediction_path(root, case_id, prior_start_t):
    output = root / "adj8_models_xyz" / f"noise_2_priort_{prior_start_t}_comb"
    candidates = (output / f"{case_id}_pred.nii", output / f"{case_id}pred.nii")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Prediction for case {case_id} was not found under {output}")


def run_one(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env=env
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description="Compare MADM fusion modes on fixed validation cases.")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--prior-start-t", type=int, default=800)
    parser.add_argument(
        "--fusion-temperature",
        type=float,
        default=0.01,
        help="softmax temperature for xstart reliability fusion modes "
             "(reviewer #4: sensitivity scan passes different values here)",
    )
    parser.add_argument(
        "--timestep-respacing",
        default="",
        help="optional DDIM respacing, e.g. ddim50; prior-start-t is mapped from the original 1000-step index",
    )
    parser.add_argument("--prior-root", default="outputs/priors/refined/val")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-case predictions in an existing output root",
    )
    parser.add_argument(
        "--allow-shared-prior-root",
        action="store_true",
        help="allow a prior root without a split subdirectory (only use for a verified split-specific root)",
    )
    parser.add_argument("--skip-mean", action="store_true")
    parser.add_argument("--skip-xstart", action="store_true")
    parser.add_argument(
        "--include-adaptive-xstart",
        action="store_true",
        help="also evaluate the timestep-adaptive xstart fusion innovation",
    )
    parser.add_argument(
        "--include-median",
        action="store_true",
        help="also evaluate deterministic voxelwise median fusion",
    )
    parser.add_argument(
        "--include-xstart-mean",
        action="store_true",
        help="same-protocol arm: equal-weight fusion of clean x_start predictions "
             "(shared posterior path, identical to the LOO arm)",
    )
    parser.add_argument(
        "--include-xstart-median",
        action="store_true",
        help="same-protocol arm: voxelwise median of clean x_start predictions "
             "(shared posterior path, identical to the LOO arm)",
    )
    parser.add_argument(
        "--include-xstart-fixed",
        action="store_true",
        help="same-protocol fixed-weight arm: fixed-temperature softmax against "
             "the self-inclusive three-view consensus (no leave-one-out)",
    )
    args = parser.parse_args()
    if (
        args.skip_mean
        and args.skip_xstart
        and not args.include_adaptive_xstart
        and not args.include_median
        and not args.include_xstart_mean
        and not args.include_xstart_median
        and not args.include_xstart_fixed
    ):
        raise ValueError("At least one fusion method must be enabled")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.output_root or Path(f"outputs/ablation/compare_{args.split}_t{args.prior_start_t}_{tag}")
    methods = []
    if not args.skip_mean:
        methods.append(("mean", "mean"))
    if not args.skip_xstart:
        methods.append(("xstart_agreement", "xstart_agreement"))
    if args.include_adaptive_xstart:
        methods.append(("adaptive_xstart", "adaptive_xstart"))
    if args.include_median:
        methods.append(("median", "median"))
    if args.include_xstart_mean:
        methods.append(("xstart_mean", "xstart_mean"))
    if args.include_xstart_median:
        methods.append(("xstart_median", "xstart_median"))
    if args.include_xstart_fixed:
        methods.append(("xstart_fixed", "xstart_fixed"))
    effective_prior_start_t = args.prior_start_t
    if args.timestep_respacing:
        if not args.timestep_respacing.startswith("ddim"):
            raise ValueError("--timestep-respacing must be empty or like ddim50")
        retained = sorted(space_timesteps(1000, args.timestep_respacing))
        if args.prior_start_t not in retained:
            nearest = min(retained, key=lambda value: abs(value - args.prior_start_t))
            raise ValueError(
                f"original prior-start-t={args.prior_start_t} is not retained by "
                f"{args.timestep_respacing}; nearest retained timestep is {nearest}"
            )
        effective_prior_start_t = retained.index(args.prior_start_t)
    manifest = {
        "cases": args.cases,
        "split": args.split,
        "prior_start_t": args.prior_start_t,
        "effective_prior_start_t": effective_prior_start_t,
        "timestep_respacing": args.timestep_respacing,
        "prior_root": args.prior_root,
        "seed": args.seed,
        "gpu": args.gpu,
        "fusion_temperature": args.fusion_temperature,
        "methods": [x[0] for x in methods],
    }
    manifest_path = root / "manifest.json"
    if root.exists():
        if not args.resume:
            raise FileExistsError(f"output root already exists: {root}; pass --resume to reuse completed cases")
        if not manifest_path.exists():
            raise FileNotFoundError(f"cannot resume without manifest: {manifest_path}")
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("cases", "split", "prior_start_t", "effective_prior_start_t", "timestep_respacing", "prior_root", "seed", "methods"):
            if recorded.get(key) != manifest[key]:
                raise ValueError(f"resume manifest mismatch for {key}: {recorded.get(key)!r} != {manifest[key]!r}")
        if recorded.get("fusion_temperature") is not None and recorded["fusion_temperature"] != manifest["fusion_temperature"]:
            raise ValueError(
                "resume manifest mismatch for fusion_temperature: "
                f"{recorded['fusion_temperature']!r} != {manifest['fusion_temperature']!r}"
            )
    else:
        root.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    summary = {}
    for method_name, fusion_mode in methods:
        inference_seconds = []
        for case_id in args.cases:
            case_root = root / method_name / f"case_{case_id}"
            try:
                prediction_path(case_root, case_id, effective_prior_start_t)
                print(f"\n=== {method_name} case {case_id}: reusing completed prediction ===")
            except FileNotFoundError:
                command = [sys.executable, "sample_3D.py", "--model_root", "weights/axis_models", "--model_axis", "x", "y", "z", "--use_prior", "true", "--load_prior_root", args.prior_root, "--prior_start_t", str(effective_prior_start_t), "--data_root", "data/udpet", "--split", args.split, "--case_id", case_id, "--sample_num", "1", "--avg_start_number", "2", "--save_single", "false", "--save_fusion_stats", "false", "--fusion_mode", fusion_mode, "--fusion_temperature", str(args.fusion_temperature if fusion_mode in XSTART_FUSION_MODES else 0.05), "--seed", str(args.seed), "--save_root", str(case_root)]
                if args.timestep_respacing:
                    command.extend(("--timestep_respacing", args.timestep_respacing))
                if args.allow_shared_prior_root:
                    command.extend(("--allow_shared_prior_root", "true"))
                print(f"\n=== {method_name} case {case_id} ===")
                seconds = run_one(command, case_root / "run.log", env)
                inference_seconds.append(seconds)
                print(f"{method_name} {case_id}: sampled in {seconds:.1f}s")
        summary[method_name] = {
            "count": len(args.cases),
            "inference_seconds_mean": (
                sum(inference_seconds) / len(inference_seconds)
                if inference_seconds else None),
        }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"All outputs: {root}")


if __name__ == "__main__":
    main()
