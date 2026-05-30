"""Estimate the white-noise standard deviation of the noisy BCARS cube.

Assumption (per request): the noise is spatially homogeneous and white
(i.e. a single global sigma, uncorrelated across pixels/bands). We therefore
estimate one global sigma with robust high-pass estimators that isolate noise
from the underlying signal structure, and aggregate across bands by median.

The data lives in [0, 1]. main_denoising.py / runners/diffusion.py internally
rescale to [-1, 1] and double sigma (`args.sigma_0 = 2 * args.sigma_0`), so the
value printed here (in [0,1] units) is exactly what you pass as
    python main_denoising.py --deg denoising<sigma>
"""

import numpy as np
import scipy.io as sio

MAT_PATH = '/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/DDS2M/exp/datasets/ood_msi/bcars_denoising.mat'

_raw_mat = sio.loadmat(MAT_PATH)
# Noisy observation cube, shape (H, W, C), values in [0, 1].
Z_noisy = np.asarray(_raw_mat['y_0_real'], dtype=np.float64)
H, W, C = Z_noisy.shape
print(f"noisy cube y_0_real: shape={Z_noisy.shape}, "
      f"min={Z_noisy.min():.4g}, max={Z_noisy.max():.4g}")

MAD_SCALE = 0.6744897501960817  # Phi^-1(0.75); makes MAD a consistent sigma estimator


def sigma_mad_haar(cube):
    """Donoho robust estimator: MAD of the finest-scale Haar diagonal (HH)
    wavelet detail, computed per band then aggregated by median over bands.
    Diagonal detail strongly suppresses smooth signal, leaving mostly noise."""
    a = cube[0:H - H % 2:2, 0:W - W % 2:2, :]
    b = cube[0:H - H % 2:2, 1:W - W % 2:2, :]
    c = cube[1:H - H % 2:2, 0:W - W % 2:2, :]
    d = cube[1:H - H % 2:2, 1:W - W % 2:2, :]
    hh = (a - b - c + d) / 2.0  # normalized so noise std is preserved
    per_band = np.median(np.abs(hh), axis=(0, 1)) / MAD_SCALE
    return float(np.median(per_band)), per_band


def sigma_laplacian(cube):
    """Immerkaer (1996) fast noise estimator using a Laplacian-difference mask,
    which cancels locally-linear signal. Per band, aggregated by median."""
    # interior second-difference response of mask [[1,-2,1],[-2,4,-2],[1,-2,1]]
    L = (cube[0:-2, 0:-2, :] + cube[0:-2, 2:, :] + cube[2:, 0:-2, :] + cube[2:, 2:, :]
         - 2 * (cube[0:-2, 1:-1, :] + cube[2:, 1:-1, :]
                + cube[1:-1, 0:-2, :] + cube[1:-1, 2:, :])
         + 4 * cube[1:-1, 1:-1, :])
    scale = np.sqrt(np.pi / 2.0) / 6.0
    per_band = scale * np.mean(np.abs(L), axis=(0, 1))
    return float(np.median(per_band)), per_band


def sigma_finite_diff(cube, axis):
    """Neighbor finite-difference estimator: sigma = sqrt(var(diff)/2).
    Biased high by true signal gradients; reported as an upper sanity bound."""
    dz = np.diff(cube, axis=axis)
    return float(np.sqrt(dz.var() / 2.0))


sigma_mad, mad_bands = sigma_mad_haar(Z_noisy)
sigma_lap, lap_bands = sigma_laplacian(Z_noisy)
sigma_spatial = sigma_finite_diff(Z_noisy, axis=0)
sigma_spectral = sigma_finite_diff(Z_noisy, axis=2)

print("\n--- white-noise sigma estimates (in [0,1] data units) ---")
print(f"MAD Haar-diagonal (robust):   {sigma_mad:.5f}   "
      f"[per-band spread: {mad_bands.min():.5f}-{mad_bands.max():.5f}]")
print(f"Laplacian (Immerkaer):        {sigma_lap:.5f}")
print(f"spatial finite-diff (upper):  {sigma_spatial:.5f}")
print(f"spectral finite-diff (upper): {sigma_spectral:.5f}")

# The MAD Haar-diagonal estimator is the most robust to signal structure under a
# white-noise assumption, so use it as the recommended value.
sigma_hat = sigma_mad
print(f"\n=> recommended sigma = {sigma_hat:.5f}")
print(f"=> run: python main_denoising.py --deg denoising{sigma_hat:.4f}")
