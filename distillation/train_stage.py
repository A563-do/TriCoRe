"""Train one strict two-step-to-one-step progressive distillation stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader_scripts.load_pet_2_5D import LoadPetSlices
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .distill_transition import (
    stage_transition_indices,
    student_transition_loss,
    teacher_two_step_target,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default="data/udpet")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-steps", type=int, required=True)
    parser.add_argument("--student-steps", type=int, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def load_model(defaults, checkpoint, device):
    model, diffusion = create_model_and_diffusion(**defaults)
    state = dist_util.load_state_dict(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    return model, diffusion


def main():
    args = build_parser().parse_args()
    if args.teacher_steps != 2 * args.student_steps:
        raise ValueError("teacher-steps must equal 2 * student-steps")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-volume student training")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dist_util.setup_dist()
    defaults = model_and_diffusion_defaults()
    defaults["in_channels"] = 8 * 2 + 1 + defaults["out_channels"]
    starts, middles, ends = stage_transition_indices(
        args.diffusion_steps, args.teacher_steps, args.student_steps
    )
    teacher, diffusion = load_model(defaults, args.teacher_checkpoint, device)
    teacher.eval().requires_grad_(False)
    student, _ = create_model_and_diffusion(**defaults)
    student.load_state_dict(teacher.state_dict())
    student.to(device).train()
    dataset = LoadPetSlices(
        root_dir=os.path.join(args.data_root, "train"), axis=args.axis,
        load_adj=8, max_samples=args.max_samples,
        out_size=defaults["image_size"], seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1.0e-5)
    args.output_root.mkdir(parents=True)
    history = []
    best = float("inf")
    start_tensor = torch.tensor(starts, device=device, dtype=torch.long)
    middle_tensor = torch.tensor(middles, device=device, dtype=torch.long)
    end_tensor = torch.tensor(ends, device=device, dtype=torch.long)
    for step in range(1, args.train_steps + 1):
        try:
            target, cond = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            target, cond = next(iterator)
        target, cond = target.to(device).float(), cond.to(device).float()
        index = torch.randint(0, len(starts), (target.shape[0],), device=device)
        t_start, t_middle, t_end = start_tensor[index], middle_tensor[index], end_tensor[index]
        noisy = diffusion.q_sample(target, t_start, noise=torch.randn_like(target))
        teacher_target = teacher_two_step_target(
            diffusion, teacher, noisy, cond, t_start, t_middle, t_end
        )
        loss, _ = student_transition_loss(
            diffusion, student, noisy, cond, t_start, t_end, teacher_target
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append({"step": step, "loss": value})
        if value < best:
            best = value
            # Write atomically so a full disk cannot leave a misleading partial
            # best_model.pt that a later stage might attempt to load.
            checkpoint = args.output_root / "best_model.pt"
            temporary = args.output_root / "best_model.pt.tmp"
            torch.save(
                {"state_dict": student.state_dict(), "axis": args.axis,
                 "teacher_steps": args.teacher_steps, "student_steps": args.student_steps,
                 "diffusion_steps": args.diffusion_steps, "best_loss": best},
                temporary,
            )
            temporary.replace(checkpoint)
        if step == 1 or step % args.log_interval == 0:
            print(json.dumps(history[-1]), flush=True)
    (args.output_root / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({
        "method": "strict_progressive_two_to_one_ddim_distillation",
        "axis": args.axis, "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_steps": args.teacher_steps, "student_steps": args.student_steps,
        "diffusion_steps": args.diffusion_steps, "train_steps": args.train_steps,
        "seed": args.seed, "target_usage": "teacher_transition_only",
        "target_100AC_used_by_inference": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
