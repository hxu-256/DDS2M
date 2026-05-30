# DDS2M Hyperparameter Guide

## `--deg` — degradation type

Selects which `H_functions` subclass is built in `runners/diffusion.py:113-148`.

| Value | What it does |
|---|---|
| `denoising0.1` | Identity operator (H = I). The suffix sets σ (0.1 = moderate noise). No pixels removed. |
| `completion10` | Randomly masks 10% of pixels. `10/20/30` pick `mask_10/20/30` from the `.mat`. |
| `sisr_bicubic4` | Bicubic downsampling by factor 4. |

---

## `--rank` (default 10)

Controls the **low-rank factorization** in `VS2M` (`runners/VS2M.py:66-79`).

The model represents the HSI as a sum of `rank` outer products:

```
HSI ≈ Σᵢ  spatial_net_i(noise_input)  ⊗  spectral_net_i(noise_input)
          (256×256, 1)                        (1, 32)
```

- **Higher rank** → more expressive, can capture more spectral variation, but slower and slightly more prone to fitting noise.
- **Lower rank** (e.g. 6) → stronger implicit regularization, faster, but may under-fit complex scenes.
- Typical values: 6–15.

---

## `--beta` (default 0)

Weight of the **total variation (TV) loss** in `VS2M._optimization_closure` (`runners/VS2M.py:166`):

```python
total_loss = mse_loss + beta * tv_loss
```

- `beta = 0`: no spatial smoothness penalty — pure data fidelity.
- `beta > 0` (e.g. 0.01): encourages piecewise-smooth reconstructions. Helpful for very noisy inputs.
- Has diminishing returns; too large blurs edges.

---

## `--sigma_0` / the suffix in `--deg denoising<σ>`

The **noise level on the observations** (`runners/diffusion.py:151`):

```python
args.sigma_0 = 2 * args.sigma_0   # scaled to [-1,1] range
y_0 = H(x) + sigma_0 * randn()
```

- Also controls which diffusion singular values are "before" vs "after" the noise floor in the DDRM posterior update — the split at `singulars * sigma_next > sigma_0` (`functions/denoising.py:86-88`).
- For denoising: set it to the actual noise σ of your data.

---

## `--eta` / `--etaB` (both default 1)

Stochasticity knobs from DDRM (`functions/denoising.py:91-107`):

| Name | Controls | Effect |
|---|---|---|
| `etaA` (= `--eta`) | **"after" components** — singular values below σ₀ (already cleaner than noise) | η=1 → fully stochastic; η=0 → deterministic DDIM-style |
| `etaB` | **"before" components** — singular values above σ₀ (noisier than observation) | η=1 → heavily guided by y₀; lower η → more model prior |
| `etaC` (= `--eta`) | **null-space components** — not observed at all | Same as etaA |

Setting both to 1 (default) is the standard DDRM setting. Reducing them makes the process more deterministic and slightly sharper but less diverse.

---

## `--timesteps` and `--start_point`

| Arg | Default | Role |
|---|---|---|
| `--timesteps` | 2000 | How many reverse diffusion steps to visit |
| `--start_point` | 1000 | Step index at which DIP network updates begin |

```python
skip = num_diffusion_timesteps // timesteps   # subsampling factor
seq  = range(0, num_diffusion_timesteps, skip)
# DIP weights only updated for iii >= start_point
```

- `timesteps = 2000` with `num_diffusion_timesteps = 2000` → skip=1, every step visited.
- Reducing `start_point` → fewer "free-running" steps before optimization begins → faster but starts from a less noisy point.
- Increasing `start_point` → more pure diffusion warm-up before DIP engages.

---

## Beta schedule  (`diffusion.beta_start`, `beta_end`, `beta_schedule` in YAML)

Defined in the config YAML, consumed in `runners/diffusion.py:22-51`. Controls the **noise schedule** α_t:

```
alpha_t = cumprod(1 - beta_t)    # signal retention at timestep t
```

| Setting | Effect |
|---|---|
| `beta_start=0.0001`, `beta_end=0.005`, `schedule=linear` (default) | Gentle schedule; most signal retained until late. Good for spectrally correlated MSI. |
| Larger `beta_end` | Faster noise injection → weaker diffusion prior, less constraining. |
| `schedule=quad` | Front-loaded schedule; more aggressive early noise. |

---

## `model.iter_number`  (YAML)

Number of **gradient steps per diffusion timestep** inside `VS2M.optimize()`.

- Set to `0` in the YAML means no DIP update at that step.
- In the active phase (`iii >= start_point`), more iterations = better fit to current `xt` but more compute.
- Typical range: 1–5 steps per diffusion step.

---

## TL;DR — where to start for denoising experiments

The most impactful knobs are:

1. `--deg denoising<σ>` — match σ to your actual noise level
2. `--rank` — sweep 6 / 10 / 15 to trade expressiveness vs. regularization
3. `--start_point` — lower = faster; higher = more diffusion warm-up

Keep `--eta 1 --etaB 1 --beta 0` at defaults until you have a baseline.
