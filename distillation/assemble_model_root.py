"""Export three stage checkpoints as pure sampler-ready state dictionaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--y", type=Path, required=True)
    parser.add_argument("--z", type=Path, required=True)
    parser.add_argument("--student-steps", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    sources = {axis: path for axis, path in (("x", args.x), ("y", args.y), ("z", args.z))}
    for axis, root in sources.items():
        state = torch.load(root / "best_model.pt", map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        torch.save(state, args.output_root / f"model_{axis}.pt")
    (args.output_root / "manifest.json").write_text(json.dumps({
        "method": "strict_progressive_two_to_one_ddim_distillation",
        "student_steps": args.student_steps,
        "sources": {axis: str(path / "best_model.pt") for axis, path in sources.items()},
        "sampler_contract": "pure state_dict model_x/model_y/model_z",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
