#!/usr/bin/env python3
"""TriCoRe end-to-end reproduction driver.

Single Python entry point that shells out to the released modules with the
frozen paper configuration, replacing the per-step shell scripts.

Stages
  01  train the three orthogonal 2.5-D diffusion backbones (x, y, z)
  02  train the cGAN initial prior and generate priors for train/val/test
  03  train the 3-D residual prior corrector and emit refined val/test priors
  04  core TriCoRe sampling: leave-one-out clean-state fusion at t=800, t=600
  05  train the four-channel refinement head (5-fold CV, then the final head)
  06  apply the final head to test36
  20  fusion-rule ablation arms
  30  progressive distillation 200 -> 100 -> 50 + timing benchmark
  all run every stage in the listed order

Examples
  python reproduce.py --list
  python reproduce.py --stage 04 --gpu 1
  python reproduce.py --stage all --data-root data/udpet --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Numeric stage ids in execution order for ``--stage all``.
ALL_ORDER = ("01", "02", "03", "04", "05", "06", "20", "30")

STAGE_TITLES = {
    "01": "train the three orthogonal 2.5-D diffusion backbones",
    "02": "train the cGAN prior generator and generate priors for all splits",
    "03": "train the 3-D residual prior corrector and refine val/test priors",
    "04": "core TriCoRe sampling: LOO fusion at t=800 and t=600",
    "05": "train the four-channel refinement head (CV + final)",
    "06": "apply the final head to test36",
    "20": "fusion-rule ablations",
    "30": "progressive distillation and the paired timing benchmark",
}


def cuda_env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def banner(text: str) -> None:
    print(f"\n===== {text} =====", flush=True)


def run(cfg: argparse.Namespace, cmd: list, *, gpu: bool = False,
        check: bool = True) -> None:
    """Run one command from the repository root."""
    print(("[gpu %s] $ " % cfg.gpu if gpu else "$ ")
          + " ".join(str(part) for part in cmd), flush=True)
    if cfg.dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT,
                   env=cuda_env(cfg.gpu) if gpu else None, check=check)


def copy_once(cfg: argparse.Namespace, src: Path, dst: Path) -> None:
    """Mirror ``cp -n``: only place the checkpoint when the target is absent."""
    print(f"$ cp -n {src.relative_to(REPO_ROOT)} {dst.relative_to(REPO_ROOT)}",
          flush=True)
    if cfg.dry_run or dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"missing checkpoint: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def case_ids(count: int) -> list:
    return [f"{index:04d}" for index in range(count)]


# --------------------------------------------------------------------------
# 01 - axis diffusion backbones
# --------------------------------------------------------------------------
def stage_01(cfg: argparse.Namespace) -> None:
    for axis in ("x", "y", "z"):
        banner(f"01 axis {axis}: 2.5-D diffusion backbone")
        run(cfg, [cfg.python, "-u", "train.py",
                  "--train_axis", axis,
                  "--data_root", cfg.data_root,
                  "--logdir", f"{cfg.weights}/axis_{axis}",
                  "--lr_anneal_steps", cfg.steps,
                  "--save_only_best", "True"], gpu=True)

    # sample_3D.py expects weights/axis_models/model_{x,y,z}.pt
    for axis in ("x", "y", "z"):
        copy_once(cfg,
                  REPO_ROOT / cfg.weights / f"axis_{axis}" / "best_model.pt",
                  REPO_ROOT / cfg.weights / "axis_models" / f"model_{axis}.pt")


# --------------------------------------------------------------------------
# 02 - cGAN initial prior
# --------------------------------------------------------------------------
def stage_02(cfg: argparse.Namespace) -> None:
    banner("02 train the cGAN prior generator")
    run(cfg, [cfg.python, "-u", "train_cgan_prior.py",
              "--mode", "train",
              "--data_root", cfg.data_root,
              "--model_save_dir", cfg.cgan_weights], gpu=True)

    for split in ("train", "val", "test"):
        banner(f"02 generate cGAN priors: {split}")
        run(cfg, [cfg.python, "-u", "train_cgan_prior.py",
                  "--mode", "infer",
                  "--infer_split", split,
                  "--data_root", cfg.data_root,
                  "--model_save_dir", cfg.cgan_weights,
                  "--prior_save_dir", f"{cfg.prior_root}/cgan/{split}"],
            gpu=True)


# --------------------------------------------------------------------------
# 03 - residual prior correction
# --------------------------------------------------------------------------
def stage_03(cfg: argparse.Namespace) -> None:
    banner("03 train the 3-D residual prior corrector")
    run(cfg, [cfg.python, "-u", "prior_correction/refine_prior.py",
              "--mode", "train",
              "--data-root", cfg.data_root,
              "--prior-root", f"{cfg.prior_root}/cgan/train",
              "--val-prior-root", f"{cfg.prior_root}/cgan/val",
              "--output-dir", cfg.corrector_weights], gpu=True)

    for split in ("val", "test"):
        banner(f"03 generate refined priors: {split}")
        run(cfg, [cfg.python, "-u", "prior_correction/refine_prior.py",
                  "--mode", "infer",
                  "--split", split,
                  "--data-root", cfg.data_root,
                  "--prior-root", f"{cfg.prior_root}/cgan/{split}",
                  "--checkpoint", f"{cfg.corrector_weights}/best_model.pt",
                  "--output-dir", f"{cfg.prior_root}/refined/{split}"],
            gpu=True)


# --------------------------------------------------------------------------
# 04 - core TriCoRe sampling
# --------------------------------------------------------------------------
def stage_04(cfg: argparse.Namespace) -> None:
    # Direct sampling takes the *sparse* start index: t=800 -> 40, t=600 -> 30
    # under the 50-step respaced schedule.
    for split, count in (("val", 30), ("test", 36)):
        for t_original, sparse_index in ((800, 40), (600, 30)):
            name = f"{split}{count}_t{t_original}_loo"
            banner(f"04 {name}: LOO fusion, t={t_original} "
                   f"(sparse index {sparse_index})")
            run(cfg, [cfg.python, "-u", "sample_3D.py",
                      "--model_root", f"{cfg.weights}/axis_models",
                      "--model_axis", "x", "y", "z",
                      "--use_prior", "true",
                      "--allow_shared_prior_root", "true",
                      "--load_prior_root", f"{cfg.prior_root}/refined/{split}",
                      "--prior_start_t", str(sparse_index),
                      "--timestep_respacing", "ddim50",
                      "--data_root", cfg.data_root,
                      "--split", split,
                      "--sample_num", "1",
                      "--avg_start_number", "2",
                      "--save_single", "false",
                      "--save_fusion_stats", "false",
                      "--fusion_mode", "xstart_agreement",
                      "--fusion_temperature", "0.01",
                      "--save_axis_xstarts", "true",
                      "--seed", "20260819",
                      "--save_root", f"{cfg.sample_root}/{name}"], gpu=True)


# --------------------------------------------------------------------------
# 05 - refinement head
# --------------------------------------------------------------------------
def stage_05(cfg: argparse.Namespace) -> None:
    banner("05 refinement head: 5-fold CV then the final head")
    run(cfg, [cfg.python, "-u", "refinement_head/train_refine_head.py",
              "--out-root", f"{cfg.out_root}/head"], gpu=True)


# --------------------------------------------------------------------------
# 06 - apply
# --------------------------------------------------------------------------
def stage_06(cfg: argparse.Namespace) -> None:
    banner("06 TriCoRe-R: apply the final head to test36")
    run(cfg, [cfg.python, "-u", "refinement_head/apply_refine_head.py",
              "--split", "test",
              "--head", f"{cfg.out_root}/head/final_head.pt",
              "--out-root", f"{cfg.out_root}/head/test36_refined"], gpu=True)
    print(f"\nTriCoRe-R refined test36 volumes: "
          f"{cfg.out_root}/head/test36_refined")


# --------------------------------------------------------------------------
# 20 - ablations
# --------------------------------------------------------------------------
ABLATION_COMMON = ["--timestep-respacing", "ddim50",
                   "--fusion-temperature", "0.01",
                   "--include-median",
                   "--include-xstart-mean",
                   "--include-xstart-median",
                   "--include-xstart-fixed",
                   "--allow-shared-prior-root"]


def stage_20(cfg: argparse.Namespace) -> None:
    ablation = f"{cfg.out_root}/ablation"
    settings = (("val", 30, 600, "val30_t600_xstart"),
                ("val", 30, 800, "val30_t800_xstart"),
                ("test", 36, 800, "test36_t800_xstart"))
    for split, count, t_original, name in settings:
        banner(f"20 fusion arms on {split}{count} at t={t_original}")
        run(cfg, [cfg.python, "-u", "run_five_case_comparison.py",
                  "--gpu", str(cfg.gpu),
                  *ABLATION_COMMON,
                  "--split", split,
                  "--prior-start-t", str(t_original),
                  "--prior-root", f"{cfg.prior_root}/refined/{split}",
                  "--cases", *case_ids(count),
                  "--output-root", f"{ablation}/{name}"])


# --------------------------------------------------------------------------
# 30 - progressive distillation
# --------------------------------------------------------------------------
def stage_30(cfg: argparse.Namespace) -> None:
    distillation = cfg.distill_root

    def train_stage(axis, teacher_steps, student_steps, teacher, out):
        banner(f"30 axis {axis}: {teacher_steps} -> {student_steps} steps")
        run(cfg, [cfg.python, "-u", "-m", "distillation.train_stage",
                  "--axis", axis,
                  "--teacher-checkpoint", teacher,
                  "--data-root", cfg.data_root,
                  "--teacher-steps", str(teacher_steps),
                  "--student-steps", str(student_steps),
                  "--train-steps", str(cfg.distill_steps),
                  "--output-root", out], gpu=True)

    for axis in ("x", "y", "z"):
        train_stage(axis, 200, 100,
                    f"{cfg.weights}/axis_models/model_{axis}.pt",
                    f"{distillation}/paper_student_{axis}_100_from_200")
    for axis in ("x", "y", "z"):
        train_stage(axis, 100, 50,
                    f"{distillation}/paper_student_{axis}_100_from_200/"
                    f"best_model.pt",
                    f"{distillation}/paper_student_{axis}_50_from_100")

    banner(f"30 assemble the student model root ({cfg.student_root})")
    run(cfg, [cfg.python, "distillation/assemble_model_root.py",
              "--x", f"{distillation}/paper_student_x_50_from_100/best_model.pt",
              "--y", f"{distillation}/paper_student_y_50_from_100/best_model.pt",
              "--z", f"{distillation}/paper_student_z_50_from_100/best_model.pt",
              "--student-steps", "50",
              "--output-root", cfg.student_root])

    banner("30 paired timing benchmark (teacher ddim200 vs student ddim50)")
    run(cfg, [cfg.python, "-u", "ablation/benchmark_distill_timing.py",
              "--split", "val",
              "--prior-root", f"{cfg.prior_root}/refined/val",
              "--output-root", f"{cfg.out_root}/ablation/"
                               f"timing_distill_paired"], gpu=True)


STAGES = {
    "01": stage_01,
    "02": stage_02,
    "03": stage_03,
    "04": stage_04,
    "05": stage_05,
    "06": stage_06,
    "20": stage_20,
    "30": stage_30,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reproduce.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=(*STAGES, "all"),
                        help="stage id to run, or 'all' for the full pipeline")
    parser.add_argument("--list", action="store_true",
                        help="print the stage table and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without executing them")
    parser.add_argument("--python", default=os.environ.get("PYTHON")
                        or sys.executable,
                        help="interpreter used for every stage "
                             "(default: the one running this file)")
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"),
                        help="CUDA_VISIBLE_DEVICES for GPU stages")
    parser.add_argument("--data-root",
                        default=os.environ.get("DATA_ROOT", "data/udpet"),
                        help="dataset root holding train/val/test splits")
    parser.add_argument("--weights", default="weights",
                        help="root for the axis backbones and the corrector")
    parser.add_argument("--cgan-weights", default="weights/cgan_prior",
                        help="cGAN prior generator checkpoints")
    parser.add_argument("--corrector-weights",
                        default="weights/prior_corrector",
                        help="residual prior corrector checkpoints")
    parser.add_argument("--prior-root", default="outputs/priors",
                        help="root for cgan/ and refined/ priors")
    parser.add_argument("--sample-root", default="outputs/samples",
                        help="root for core TriCoRe sampling runs")
    parser.add_argument("--out-root", default="outputs",
                        help="root for head and ablation outputs")
    parser.add_argument("--distill-root", default="weights/distillation",
                        help="root for progressive-distillation checkpoints")
    parser.add_argument("--student-root", default="weights/student_ddim50",
                        help="assembled 50-step student model root")
    parser.add_argument("--steps", type=int, default=200000,
                        help="LR annealing steps for each axis backbone")
    parser.add_argument("--distill-steps", type=int, default=10000,
                        help="training steps per distillation stage")
    return parser


def print_stages() -> None:
    print("Stages (run order for --stage all: " + " ".join(ALL_ORDER) + ")\n")
    for stage, title in STAGE_TITLES.items():
        print(f"  {stage}  {title}")
    print("\nStages 01-06 reproduce the paper's main results; "
          "20 and 30 reproduce the ablation and efficiency tables.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list or not args.stage:
        print_stages()
        if not args.stage:
            parser.print_usage()
            print("\nSpecify --stage, or --list to inspect the pipeline.")
        return

    stages = ALL_ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        banner(f"stage {stage}: {STAGE_TITLES[stage]}")
        STAGES[stage](args)
    banner("done")


if __name__ == "__main__":
    main()
