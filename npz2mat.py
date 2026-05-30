import numpy as np
import scipy.io
import os

npz_path = (
    "/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/2025_Celegans/0506/generated/"
    "bcars_batched_fit_2026-05-07_17-26-16_preprocessed_medfilter_05072025_NR_dhs-3-gfp_AD_05_PROCESS_202657_14_45_24_362743_PROCESS_202657_14_58_5_69804_tolerance=1e-05_with_Znoisy.npz"
)

d = np.load(npz_path)
print("Keys:", list(d.files))
for k in d.files:
    arr = d[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

raw = d['Z_noisy'].astype(np.float64)                                               # (320, 500, 695)
nrb = d['nrb_amp_normed']                                                           # (695,)
gt  = (d['bcars_noise_free'] - nrb**2) / (2 * nrb)                                 # (320, 500, 695)

print(f"\nraw range: [{raw.min():.5f}, {raw.max():.5f}]")
print(f"gt  range: [{gt.min():.5f},  {gt.max():.5f}]")

# Shared min-max normalization so raw and gt live on the same [0,1] scale
global_min = float(min(raw.min(), gt.min()))
global_max = float(max(raw.max(), gt.max()))
raw_norm = ((raw - global_min) / (global_max - global_min)).astype(np.float32)
gt_norm  = ((gt  - global_min) / (global_max - global_min)).astype(np.float32)

# Center-crop nx=500 → 320 so spatial dims are square (smaller dimension = 320)
cx = raw_norm.shape[1] // 2  # 250
raw_crop = raw_norm[:, cx-160:cx+160, :]   # (320, 320, 695)
gt_crop  = gt_norm [:, cx-160:cx+160, :]

print(f"\nOutput shape: {gt_crop.shape}")
print(f"raw_crop range: [{raw_crop.min():.5f}, {raw_crop.max():.5f}]")
print(f"gt_crop  range: [{gt_crop.min():.5f}, {gt_crop.max():.5f}]")

out_dir = "./exp/datasets/ood_msi"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "bcars_denoising.mat")

scipy.io.savemat(out_path, {
    'img_clean': gt_crop,
    'y_0_real':  raw_crop,
    'norm_min':  np.array([[global_min]]),
    'norm_max':  np.array([[global_max]]),
    'mask_10':   np.ones_like(gt_crop, dtype=np.float32),
    'mask_20':   np.ones_like(gt_crop, dtype=np.float32),
    'mask_30':   np.ones_like(gt_crop, dtype=np.float32),
})
print(f"\nSaved to {out_path}")
