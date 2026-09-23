"""Train and apply a small NAC-guided residual correction to cGAN priors.

The corrector is deliberately separate from the cGAN and diffusion models. It
expects one prior per case and learns only a bounded residual:

    refined_prior = prior + max_residual * tanh(corrector(nac, prior))

Training requires train-split priors. Validation and test priors must never be
used as training inputs when their 100AC files are later used for reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader_scripts.load_pet_2_5D import get_mask, normalize_ac


def prior_path(root: Path, case_id: str) -> Path:
    candidates = (root / f"{case_id}umap_pred.nii", root / f"{case_id}_umap_pred.nii")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing prior for case {case_id}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def load_volume(path: Path) -> np.ndarray:
    value = nib.load(str(path)).get_fdata(dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError(f"Invalid volume {path}: shape={value.shape}")
    return value.astype(np.float32)


def load_normalized(path: Path, mask: np.ndarray) -> np.ndarray:
    return normalize_ac(load_volume(path), mask).astype(np.float32)


class PriorRefinementDataset(Dataset):
    """Pair NAC, an existing cGAN prior, and 100AC for one split."""

    def __init__(self, data_root: Path, prior_root: Path, split: str):
        self.data_root = Path(data_root)
        self.prior_root = Path(prior_root)
        self.split = split
        nac_root = self.data_root / split / "5NAC"
        ac_root = self.data_root / split / "100AC"
        self.case_ids = sorted(path.name.replace("_5_NAC.nii", "") for path in nac_root.glob("*_5_NAC.nii"))
        if not self.case_ids:
            raise FileNotFoundError(f"No 5NAC files found under {nac_root}")
        missing = []
        for case_id in self.case_ids:
            if not (ac_root / f"{case_id}_100_AC.nii").exists():
                missing.append(str(ac_root / f"{case_id}_100_AC.nii"))
            try:
                prior_path(self.prior_root, case_id)
            except FileNotFoundError:
                missing.append(f"prior:{case_id}")
        if missing:
            raise FileNotFoundError(
                f"Split {split!r} is not ready for prior refinement; missing "
                + ", ".join(missing[:12])
                + (" ..." if len(missing) > 12 else "")
            )

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        nac_path = self.data_root / self.split / "5NAC" / f"{case_id}_5_NAC.nii"
        ac_path = self.data_root / self.split / "100AC" / f"{case_id}_100_AC.nii"
        nac_raw = nib.load(str(nac_path)).get_fdata(dtype=np.float32)
        mask = get_mask(nac_raw)
        nac = normalize_ac(nac_raw, mask).astype(np.float32)
        ac = load_normalized(ac_path, mask)
        # cGAN priors are already in the normalized model space. Re-normalizing
        # them by their own mean would change the scale seen by the diffuser.
        prior = load_volume(prior_path(self.prior_root, case_id))
        if nac.shape != ac.shape or nac.shape != prior.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: nac={nac.shape}, ac={ac.shape}, prior={prior.shape}"
            )
        return {
            "nac": torch.from_numpy(nac[None]),
            "prior": torch.from_numpy(prior[None]),
            "target": torch.from_numpy(ac[None]),
            "case_id": case_id,
        }


class ResidualCorrector(nn.Module):
    """Small full-volume 3-D corrector with a bounded residual output."""

    def __init__(self, channels: int = 16, max_residual: float = 0.25):
        super().__init__()
        self.max_residual = float(max_residual)
        self.features = nn.Sequential(
            nn.Conv3d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, 1, 3, padding=1),
        )

    def forward(self, nac: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        residual = self.max_residual * torch.tanh(self.features(torch.cat([nac, prior], dim=1)))
        return prior + residual


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = pred.new_zeros(())
    for dim in range(2, 5):
        pred_diff = pred.diff(dim=dim)
        target_diff = target.diff(dim=dim)
        loss = loss + torch.nn.functional.l1_loss(pred_diff, target_diff)
    return loss / 3.0


def masked_l1(pred: torch.Tensor, target: torch.Tensor, nac: torch.Tensor) -> torch.Tensor:
    mask = (nac > 0.05).float()
    error = (pred - target).abs()
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def run_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    val_prior_root = getattr(args, "val_prior_root", None) or args.prior_root
    train_data = PriorRefinementDataset(Path(args.data_root), Path(args.prior_root), "train")
    val_data = PriorRefinementDataset(Path(args.data_root), Path(val_prior_root), "val")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = ResidualCorrector(args.channels, args.max_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            nac = batch["nac"].to(device)
            prior = batch["prior"].to(device)
            target = batch["target"].to(device)
            refined = model(nac, prior)
            loss = masked_l1(refined, target, nac) + args.lambda_grad * gradient_loss(refined, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * nac.shape[0]
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nac = batch["nac"].to(device)
                refined = model(nac, batch["prior"].to(device))
                val_loss += (masked_l1(refined, batch["target"].to(device), nac) + args.lambda_grad * gradient_loss(refined, batch["target"].to(device))).item()
        val_loss /= len(val_loader.dataset)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "config": vars(args), "best_val_loss": best}, output_dir / "best_model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"best_val_loss={best:.8f}; checkpoint={output_dir / 'best_model.pt'}")


def run_infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = ResidualCorrector(config.get("channels", args.channels), config.get("max_residual", args.max_residual)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    data_root = Path(args.data_root)
    prior_root = Path(args.prior_root)
    split = args.split
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    case_ids = sorted(path.name.replace("_5_NAC.nii", "") for path in (data_root / split / "5NAC").glob("*_5_NAC.nii"))
    with torch.no_grad():
        for case_id in case_ids:
            nac_path = data_root / split / "5NAC" / f"{case_id}_5_NAC.nii"
            nac_image = nib.load(str(nac_path))
            nac_raw = nac_image.get_fdata(dtype=np.float32)
            mask = get_mask(nac_raw)
            nac = load_normalized(nac_path, mask)
            prior = load_volume(prior_path(prior_root, case_id))
            if nac.shape != prior.shape:
                raise ValueError(f"Shape mismatch for {case_id}: nac={nac.shape}, prior={prior.shape}")
            refined = model(torch.from_numpy(nac[None, None]).to(device), torch.from_numpy(prior[None, None]).to(device))
            refined = refined.squeeze().cpu().numpy().astype(np.float32)
            refined = np.maximum(refined, 0.0)
            nib.save(nib.Nifti1Image(refined, nac_image.affine, nac_image.header), str(out_root / f"{case_id}umap_pred.nii"))
    print(f"saved refined priors to {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--data-root", default="data/udpet")
    parser.add_argument("--prior-root", default="outputs/priors/cgan")
    parser.add_argument("--val-prior-root", default=None,
                        help="val-split prior root for train mode; "
                             "defaults to --prior-root (flat shared layout)")
    parser.add_argument("--output-dir", default="weights/prior_corrector")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--max-residual", type=float, default=0.25)
    parser.add_argument("--lambda-grad", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.checkpoint is None:
        parser.error("--checkpoint is required with --mode infer")
    else:
        run_infer(args)


if __name__ == "__main__":
    main()
