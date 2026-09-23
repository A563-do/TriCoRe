# TriCoRe: Cross-axis Consistency Diffusion Reconstruction for Low-dose PET

Official implementation of **TriCoRe** — a cross-axis consistency-guided
adaptive diffusion pipeline for reconstructing 100%-count AC PET from 5%-count
non-attenuation-corrected low-dose PET (5% NAC-LDPET).

TriCoRe runs three orthogonal 2.5-D conditional diffusion models, fuses their
per-step clean-state predictions with a **leave-one-out (LOO) agreement
weight**, integrates **two sampling trajectories** started at different
diffusion timesteps, and applies a lightweight **four-channel 3-D refinement
head**. Progressive DDIM distillation is included as an independent
sampling-efficiency experiment.

## Method overview

1. **Prior construction.** A structure-aware cGAN produces an initial prior
   `r`; a small 3-D residual corrector predicts a bounded residual
   (`alpha = 0.25`) to form the refined prior `r_ref = r + alpha·tanh(C([y,r]))`.
2. **Three-axis 2.5-D diffusion.** Three identical U-Nets (conditioned on 17
   adjacent slices, `load_adj=8`) run along the x/y/z axes with cosine noise
   schedule and 1000 training timesteps.
3. **Cross-axis LOO consistency fusion.** At every reverse step, each axis is
   scored against the mean of the *other two* axes:
   `w_i = softmax(-|x̂_i − c_i| / tau)` with `tau = 0.01`. The weighted clean
   state feeds one shared stochastic DDPM posterior step (50-step respaced
   DDPM).
4. **Dual trajectories.** Two frozen-model trajectories start at original
   `t = 800` (sparse index 40, 41 updates) and `t = 600` (sparse index 30,
   31 updates) with the pre-registered seed `20260819`.
5. **Four-channel refinement head.** Inputs `[p_800, p_600, 5NAC, a(v)]`,
   where `a(v) = exp(-d_pair(v)/0.01)` is the pairwise axis-agreement map.
   The head learns a per-voxel blend of the two trajectories plus a dilated
   residual branch; both output layers are zero-initialized, so training
   starts at the plain ensemble mean.

## Repository layout

```
guided_diffusion/      Core diffusion library (2.5-D UNet, Gaussian diffusion,
                       DDIM respacing) and axis_fusion.py (LOO/mean/median/
                       fixed clean-state fusion rules)
dataloader_scripts/    2.5-D slice datasets, mask and tanh(x/5) normalization
train.py               Train one x/y/z 2.5-D diffusion backbone
sample_3D.py           Three-axis LOO-fused sampling, dual trajectories,
                       per-axis x_start export
run_five_case_comparison.py  Per-case orchestration for fusion-rule ablations
train_cgan_prior.py    Structure-aware cGAN initial prior
prior_correction/      3-D residual prior corrector (refined prior)
refinement_head/       RefineHead3D, self-contained CV/final trainer,
                       test application, and reviewer-isolation variants
ablation/              Fusion-rule arms and teacher/student timing benchmark
distillation/          Progressive DDIM distillation (200 -> 50 steps)
reproduce.py           Single entry point driving every stage end to end
```

## Setup

```bash
pip install -r requirements.txt   # Python 3.10, torch 2.x + CUDA
```

## Data preparation

The UDPET dataset (MICCAI Ultra-low-dose PET Challenge, collected by the
Department of Nuclear Medicine, University of Bern; long-axis FOV whole-body
PET systems — Siemens Biograph Vision Quadra and United Imaging uEXPLORER)
is **not** distributed with this repository. We use the DRF-20 subset
(5%-count NAC) paired with full-dose AC images, resampled to `128^3`
isotropic at 1 mm, and split into 140 / 30 / 36 train / val / test volumes.

Place the data as:

```
data/udpet/
├── train/{5NAC,100AC}/<cid>_{5_NAC,100_AC}.nii
├── val/  {5NAC,100AC}/...          # val cases
└── test/ {5NAC,100AC}/...          # test cases
```

All training uses the mask-normalized space: build the foreground
mask from 5% NAC, divide by its in-mask mean, and compress with
`tanh(x/5)` (`dataloader_scripts/load_pet_2_5D.py:get_mask/normalize_ac`).

Trained weights are also not released. The directory convention used by the
reproducer is `weights/axis_models/model_{x,y,z}.pt`, `weights/cgan_prior/`,
`weights/prior_corrector/`.

## Reproducing TriCoRe-R

Everything is driven by `reproduce.py`, which runs from the repository root
and shells out to the released modules with the frozen configuration:

```bash
python reproduce.py --list          # stage table and run order
python reproduce.py --stage all --dry-run   # print every command first
```

The paper's main results come from stages 01-06, in order:

```bash
python reproduce.py --stage 01   # x, y, z backbones
python reproduce.py --stage 02   # cGAN priors for all splits
python reproduce.py --stage 03   # residual prior corrector
python reproduce.py --stage 04   # LOO fusion, t800 + t600
python reproduce.py --stage 05   # 5-fold CV + final head
python reproduce.py --stage 06   # refined test36 volumes
```

Frozen configuration: seed `20260819`, `tau = 0.01`, 50-step respaced
stochastic DDPM, `load_adj = 8`; refinement head trained with 64³ patches,
batch size 2, foreground sampling probability 0.75, Adam `1e-4` + cosine
schedule — 3000 steps/fold (seed = fold index) on val30 five-fold CV, then
4000 steps (seed 100) on all 30 val cases for the final head applied to
test36.

## Ablations and efficiency

```bash
python reproduce.py --stage 30   # progressive DDIM 200 -> 50 + timing
python reproduce.py --stage 20   # mean/median/fixed/LOO fusion arms
```

Every stage takes `--gpu`, `--data-root` and the other output roots as flags
(`GPU`, `DATA_ROOT` and `PYTHON` environment variables are honoured as
defaults), and `--python` selects the interpreter used for all stages.

Outputs of every run land under `outputs/` (git-ignored): sampled volumes and
per-axis predictions (`outputs/samples/`), priors (`outputs/priors/`),
refinement heads and predictions (`outputs/head*/`), and ablation outputs
(`outputs/ablation/`).

## Notes

- Inference never reads test-set GT; the cGAN, prior corrector and refinement
  head are trained on train/val only.
- `refinement_head/data.py` reconstructs the paper-described head data layer
  (`a(v) = exp(-d_pair/0.01)` from aligned per-axis clean predictions) in a
  self-contained form.

## License

Released under the MIT License (see [LICENSE](LICENSE)).
