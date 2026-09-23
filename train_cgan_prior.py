"""
Structure-aware cGAN prior training for TriCoRe.

This script keeps the original paper-style cGAN + L1 prior objective, and adds
optional structure-preserving losses to reduce over-smoothing:
  - 3D gradient L1 loss
  - 3D Laplacian edge loss

Default lambda_grad=0 and lambda_log=0 preserve the original behavior.
"""
import argparse
import os
import random
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader_scripts.load_pet_2_5D import get_mask, normalize_ac


class PETDataset(Dataset):
    def __init__(self, data_root, split="train"):
        self.data_root = data_root
        self.split = split
        self.nac_dir = os.path.join(data_root, split, "5NAC")
        self.ac_dir = os.path.join(data_root, split, "100AC")
        self.file_list = sorted([f for f in os.listdir(self.nac_dir) if f.endswith(".nii")])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        nac_path = os.path.join(self.nac_dir, fname)
        ac_path = os.path.join(self.ac_dir, fname.replace("_5_NAC", "_100_AC"))

        nac = nib.load(nac_path).get_fdata().astype(np.float32)
        ac = nib.load(ac_path).get_fdata().astype(np.float32)

        mask = get_mask(nac)
        nac = normalize_ac(nac, mask)
        ac = normalize_ac(ac, mask)

        return torch.from_numpy(nac).unsqueeze(0), torch.from_numpy(ac).unsqueeze(0), fname


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.inc = conv_block(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), conv_block(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), conv_block(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), conv_block(128, 256))

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv1 = conv_block(64, 32)

        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        return torch.tanh(self.outc(x))


class Discriminator3D(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        def disc_block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv3d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_channels, 64, norm=False),
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512, stride=1),
            nn.Conv3d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def gradient_l1_loss(pred, target):
    loss = pred.new_tensor(0.0)
    count = 0
    for dim in (2, 3, 4):
        pred_diff = pred.diff(dim=dim)
        target_diff = target.diff(dim=dim)
        loss = loss + F.l1_loss(pred_diff, target_diff)
        count += 1
    return loss / count


def laplacian_3d(x):
    kernel = x.new_zeros((1, 1, 3, 3, 3))
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    return F.conv3d(x, kernel, padding=1)


def laplacian_l1_loss(pred, target):
    return F.l1_loss(laplacian_3d(pred), laplacian_3d(target))


def generator_losses(fake, ac, pred_fake_g, criterion_gan, criterion_l1, args):
    loss_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
    loss_l1 = criterion_l1(fake, ac)

    loss_grad = fake.new_tensor(0.0)
    if args.lambda_grad > 0:
        loss_grad = gradient_l1_loss(fake, ac)

    loss_log = fake.new_tensor(0.0)
    if args.lambda_log > 0:
        loss_log = laplacian_l1_loss(fake, ac)

    total = (
        loss_gan
        + args.lambda_l1 * loss_l1
        + args.lambda_grad * loss_grad
        + args.lambda_log * loss_log
    )
    return total, loss_gan, loss_l1, loss_grad, loss_log


def validation_score(val_l1, val_grad, val_log, args):
    if args.val_objective == "l1":
        return val_l1
    if args.val_objective == "edge":
        return val_grad + val_log
    return val_l1 + args.lambda_grad * val_grad + args.lambda_log * val_log


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(
        "loss weights: "
        f"lambda_l1={args.lambda_l1}, "
        f"lambda_grad={args.lambda_grad}, "
        f"lambda_log={args.lambda_log}, "
        f"val_objective={args.val_objective}"
    )

    train_dataset = PETDataset(args.data_root, split="train")
    val_dataset = PETDataset(args.data_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    net_g = UNet3D().to(device)
    net_d = Discriminator3D(in_channels=2).to(device)

    criterion_l1 = nn.L1Loss()
    criterion_gan = nn.MSELoss()

    optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_score = float("inf")
    best_epoch = -1
    patience = 0
    os.makedirs(args.model_save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        net_g.train()
        epoch_l1 = 0.0
        epoch_gan = 0.0
        epoch_grad = 0.0
        epoch_log = 0.0

        for nac, ac, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            nac = nac.to(device, non_blocking=True)
            ac = ac.to(device, non_blocking=True)

            optimizer_d.zero_grad(set_to_none=True)
            fake = net_g(nac)
            pred_real = net_d(torch.cat([nac, ac], dim=1))
            pred_fake = net_d(torch.cat([nac, fake.detach()], dim=1))
            loss_d = 0.5 * (
                criterion_gan(pred_real, torch.ones_like(pred_real))
                + criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            pred_fake_g = net_d(torch.cat([nac, fake], dim=1))
            loss_g, loss_gan, loss_l1, loss_grad, loss_log = generator_losses(
                fake, ac, pred_fake_g, criterion_gan, criterion_l1, args
            )
            loss_g.backward()
            optimizer_g.step()

            batch_size = nac.size(0)
            epoch_l1 += loss_l1.item() * batch_size
            epoch_gan += loss_gan.item() * batch_size
            epoch_grad += loss_grad.item() * batch_size
            epoch_log += loss_log.item() * batch_size

        n_train = len(train_loader.dataset)
        train_l1 = epoch_l1 / n_train
        train_gan = epoch_gan / n_train
        train_grad = epoch_grad / n_train
        train_log = epoch_log / n_train

        net_g.eval()
        val_l1_sum = 0.0
        val_grad_sum = 0.0
        val_log_sum = 0.0
        with torch.no_grad():
            for nac, ac, _ in val_loader:
                nac = nac.to(device, non_blocking=True)
                ac = ac.to(device, non_blocking=True)
                pred = net_g(nac)
                batch_size = nac.size(0)
                val_l1_sum += criterion_l1(pred, ac).item() * batch_size
                val_grad_sum += gradient_l1_loss(pred, ac).item() * batch_size
                val_log_sum += laplacian_l1_loss(pred, ac).item() * batch_size

        n_val = len(val_loader.dataset)
        val_l1 = val_l1_sum / n_val
        val_grad = val_grad_sum / n_val
        val_log = val_log_sum / n_val
        val_score = validation_score(val_l1, val_grad, val_log, args)

        print(
            f"Epoch {epoch + 1}: "
            f"train_l1={train_l1:.6f}, train_gan={train_gan:.6f}, "
            f"train_grad={train_grad:.6f}, train_log={train_log:.6f}, "
            f"val_l1={val_l1:.6f}, val_grad={val_grad:.6f}, "
            f"val_log={val_log:.6f}, val_score={val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience = 0
            save_path = os.path.join(args.model_save_dir, "best_prior_model.pt")
            torch.save(
                {
                    "model": net_g.state_dict(),
                    "epoch": best_epoch,
                    "best_score": best_score,
                    "val_l1": val_l1,
                    "val_grad": val_grad,
                    "val_log": val_log,
                    "args": vars(args),
                },
                save_path,
            )
            torch.save(net_g.state_dict(), os.path.join(args.model_save_dir, "best_prior_model_state_dict.pt"))
            print(f"saved best model: epoch={best_epoch}, score={best_score:.6f}")
        else:
            patience += 1
            if patience >= args.max_patience:
                print(f"early stop: no validation improvement for {args.max_patience} epochs")
                break

        if args.save_last:
            torch.save(net_g.state_dict(), os.path.join(args.model_save_dir, "last_prior_model_state_dict.pt"))

    print(f"training done: best_epoch={best_epoch}, best_score={best_score:.6f}")


def load_generator_for_infer(args, device):
    model = UNet3D().to(device)
    ckpt_path = os.path.join(args.model_save_dir, "best_prior_model.pt")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator_for_infer(args, device)

    test_nac_dir = os.path.join(args.data_root, args.infer_split, "5NAC")
    file_list = sorted([f for f in os.listdir(test_nac_dir) if f.endswith(".nii")])
    os.makedirs(args.prior_save_dir, exist_ok=True)

    with torch.no_grad():
        for fname in tqdm(file_list, desc=f"infer prior [{args.infer_split}]"):
            nac_path = os.path.join(test_nac_dir, fname)
            nac_img = nib.load(nac_path)
            nac = nac_img.get_fdata().astype(np.float32)

            mask = get_mask(nac)
            nac = normalize_ac(nac, mask)

            nac_tensor = torch.from_numpy(nac).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(nac_tensor)
            pred = pred.squeeze(0).squeeze(0).cpu().numpy()
            pred[pred < 0] = 0

            patient_id = fname.replace("_5_NAC.nii", "")
            save_path = os.path.join(args.prior_save_dir, f"{patient_id}umap_pred.nii")
            nib.save(nib.Nifti1Image(pred, nac_img.affine, nac_img.header), save_path)

    print(f"prior inference done: {args.prior_save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/udpet")
    parser.add_argument("--model_save_dir", type=str, default="weights/cgan_prior")
    parser.add_argument("--prior_save_dir", type=str, default="outputs/priors/cgan")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_grad", type=float, default=0.0)
    parser.add_argument("--lambda_log", type=float, default=0.0)
    parser.add_argument("--val_objective", choices=["l1", "edge", "combined"], default="combined")
    parser.add_argument("--max_patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    parser.add_argument("--infer_split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode in ("train", "both"):
        train(args)
    if args.mode in ("infer", "both"):
        infer(args)


if __name__ == "__main__":
    main()
