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

## BCARS experimental-data runs (status 2026-06-12)

This repo is being used to denoise **VST-whitened experimental BCARS cubes** (not the stock MSI demos). Big-picture context lives in `../note.md` and `../AGENTS.md`; this section is the DDS2M-specific summary.

**Data**: `../2025_Celegans/data/20251111/preprocessed_vst_nomedian/mat_whittaker_p0.05/` (made by `h5tomat_detrend.py`, TT=6 → side divisible by 64). Per-spectrum **symmetric Whittaker** baseline (λ=1e5, dispersive-safe) + **robust_symmetric** norm: `s=max(|p0.05|,|p99.95|)`, `[-s,s]→[0,1]`, so the dispersive VST-zero maps to 0.5 and the cube is exactly **zero-centered (mean 0.500)** — ideal for the `2X-1` transform. `channels=685`.

**How the runner consumes it** (`runners/diffusion.py`): for `--deg denoising<σ>` it loads `y_0_real` (the real noisy obs) when present and uses it directly instead of synthesizing noise (diffusion.py:145-170). `sigma_0` is **doubled** internally for the [-1,1] range (diffusion.py:156). σ is data-driven and matches the flags: σ̂ ≈ 0.0735 glycerol / 0.1293 bead → runs use `--deg denoising0.07 / 0.13` (worm 0.11–0.12). Launch via `run_all_denoising.sh` (4 cubes / 2 GPUs, per-cube `configs/msi_denoising_<name>.yml`). Defaults: rank 10, **beta 0 (no TV)**, iter_number 1, lr 5e-4, start_point 1000, timesteps 2000.

**Symptom**: DDS2M comes out **noisier / wrong-shaped on bead & C.elegans** (glycerol is OK) while S2DIP/`dip_baseline.py` is qualitatively fine. Leading causes, in order:

1. **No valid metric / stopping**: every exp `.mat` has `img_clean = all zeros` (CONFIRMED) → logged PSNR (~6 dB) is vs zeros, so `x_best`/`psnr_best` are NOISE-selected and there is no real early stop. **Fix first**: build a glycerol pseudo-GT (FOV-mean broadcast into `img_clean`) so PSNR is real, calibrate a fixed step budget, then apply to bead/worm with no-reference residual checks.
2. **Peak clipping/clamp**: `CLIP_TO_01=True` in preprocessing flat-tops the sparse bead/worm Raman peaks (≈0.1% of pixels, ~4× over p99.95) at the input, and `inverse_data_transform` ends with `clamp(X,0,1)` (`utils/data_utils.py:33`, `rescaled:true`) so any value >1 is destroyed at the **output** too. Glycerol peaks aren't sparse (overshoot ~1.4×) → survives. Regenerate bead/worm with `CLIP_TO_01=False` AND widen the symmetric range `s` to enclose the peak (then pass the correspondingly smaller σ).
3. **No regularizer**: `beta=0` → no TV (S2DIP has spectral-TV + early stop). Try `--beta 1e-2` and/or `iter_number 3–5`; sweep `rank 6/10/15`. Only after (1).

**Watch out**: `glycerol_pad128` was zero-padded 64→128; zero pad → −1 after `2X-1` skews global stats (log shows mean ~3.16) — pad with 0.5 or use native 64×64 instead. Inspect outputs in `visualize_results_exp.ipynb`.
