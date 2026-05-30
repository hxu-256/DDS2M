# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

DDS2M (ICCV 2023) is a **self-supervised** method for hyperspectral image (HSI) / multispectral image (MSI) restoration. It requires **no training data** — the prior comes entirely from (1) a diffusion process with a schedule learned implicitly and (2) untrained deep image prior networks. The three tasks are:

- **Denoising** (`main_denoising.py`, `--deg denoising0.1` for σ=0.1)
- **Completion / Inpainting** (`main_completion.py`, `--deg completion10` / `completion20` / `completion30`)
- **Super-Resolution** (`main_sisr.py`, `--deg sisr_bicubic4`)

## Environment

Python 3.7.10 + CUDA 11.3.

```bash
pip install -r requirements.txt
# torch==1.12.1+cu113 — install from PyPI with the cu113 index if needed
```

The code hard-codes `torch.cuda.set_device(0)` and requires a GPU.

## Running Experiments

```bash
# Denoising (default: σ=0.1, rank=10, start_point=1000, timesteps=2000)
python main_denoising.py

# Change noise level or rank:
python main_denoising.py --deg denoising0.2 --rank 6

# Completion at 10%/20%/30% missing pixels:
python main_completion.py --deg completion10

# 4× bicubic super-resolution:
python main_sisr.py --deg sisr_bicubic4
```

Config files for each task live in `configs/msi_denoising.yml`, `configs/msi_completion.yml`, `configs/msi_sisr.yml`. Key knobs: `diffusion.beta_start/end`, `model.iter_number` (DIP iterations per diffusion step), `model.lr`.

## Data

Input data must be placed at `./exp/datasets/ood_msi/<filename>.mat` (configured via `data.root` and `data.filename` in the YAML). The `.mat` file is expected to contain:

- `img_clean` — ground truth HSI/MSI, shape `(H, W, C)` e.g. `(256, 256, 32)`
- `mask_10`, `mask_20`, `mask_30` — binary masks for completion tasks (same shape)

The denoising task adds noise programmatically from `img_clean`; it does not need a separate noisy input.

## Output

Results are saved to `./results/<run_name>/`. The run name encodes all hyperparameters (deg, filename, rank, eta, beta schedule, start_point, timesteps, iter_number, lr). Each result directory contains:

- `x_<timesteps>.mat` — a `.mat` file with keys: `y_0` (degraded input), `x_recon` (final reconstruction), `img_clean` (ground truth), `psnr`, `x_best` (reconstruction at best PSNR), `psnr_best`
- `<run_name>.log` — per-iteration PSNR log

## Metric

PSNR is computed per-band and averaged across all spectral bands (`runners/com_psnr.py`). It is logged every diffusion step and also stored in the output `.mat`. To compare results, load `x_best` and `img_clean` from the output `.mat` and compute PSNR with `runners/com_psnr.quality()`.

## Architecture: How the Pieces Fit Together

```
main_denoising.py
  └─ runners/diffusion.py : Diffusion.sample()
       ├─ runners/VS2M.py : VS2M  (the spatio-spectral deep image prior model)
       │    ├─ rank × skip networks  (models/skip.py) — spatial prior, one per rank component
       │    └─ rank × FCN networks   (models/fcn.py)  — spectral prior, one per rank component
       │    Output = low-rank product: spatial_out @ spectral_out  →  (H×W, C)
       └─ functions/denoising.py : efficient_generalized_steps()
            ├─ functions/svd_replacement.py : H_functions subclasses
            │    (Denoising / Inpainting / SRConv — encode the degradation operator via SVD)
            └─ per diffusion step: calls VS2M.optimize() to update DIP weights, then
               applies the DDRM-style posterior update in SVD space
```

**Key algorithm** (`functions/denoising.py`):

1. Reverse diffusion runs from `start_point` down to 0 (earlier steps use `x0_t = xt` — DIP is not updated yet).
2. At each active step, `VS2M.optimize()` runs `iter_number` gradient steps on the spatial+spectral networks to fit the current noisy diffusion sample `xt`.
3. The loss is `MSE(x0_hat * sqrt(alpha_t), xt) + beta * TV`.
4. The predicted clean image `x0_t` is plugged into the DDRM posterior update (Eq. 13), which conditions on the observations `y_0` via the SVD of the degradation operator `H`.

**VS2M low-rank decomposition**: the model factorizes the HSI as a sum of `rank` outer products — each spatial map (from a skip/U-Net DIP) dotted with a spectral signature (from an FCN). This is the "Spatio-Spectral" in DDS2M.
