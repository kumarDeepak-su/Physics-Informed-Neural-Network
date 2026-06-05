# ADMM-Guided Physics-Informed Deep Learning for 2D Acoustic Impedance Inversion

Code, trained models, results, and manuscript for:

> **ADMM-Guided Physics-Informed Deep Learning for Two-Dimensional Acoustic
> Impedance Inversion with Reweighted ℓ₁ Sparse Regularization**
> Deepak Kumar¹, Jayant Nath Tripathi²
> ¹ Institute of Geophysics, Polish Academy of Sciences, Warsaw, Poland
> ² University of Allahabad, Prayagraj, India
> Corresponding author: `deepak.kumar@igf.edu.pl`

This repository accompanies the manuscript and reproduces every figure and
table in it.

---

## Overview

Post-stack acoustic impedance inversion is ill-posed: seismic data are
band-limited and noisy. A reweighted ℓ₁ ADMM inversion (after He et al., 2022)
recovers sparse impedance boundaries trace by trace, but ignores lateral
geological continuity. This work uses that ADMM estimate as a physics prior for
a 2D convolutional refinement network trained with a differentiable
wavelet-convolution physics layer, reweighted ℓ₁ sparsity, model-proximity,
lateral-smoothness, and (benchmark-only) supervised impedance/gradient terms.

Three networks are compared — a 2D U-Net, a Hybrid ResCNN, and an Attention
ResUNet — against two classical baselines:

* **RW-ℓ₁ (trace-wise)** — reweighted ℓ₁ ADMM, each trace independent.
* **2D-coupled** — the same reweighted ℓ₁ formulation augmented with a lateral
  total-variation penalty, solved by consensus ADMM (a *stronger*,
  spatially-coupled classical comparator).

Validation spans a purpose-built controlled synthetic section and two
geologically distinct Marmousi-2 crops, plus robustness studies (input-noise
sweep, wavelet mismatch) and a five-seed confidence-interval analysis.

---

## Repository layout

```
.
├── 2D_ADMM_PINN_ResCNN_Attention.py   # main module: forward model, ADMM solvers,
│                                       # networks, training, prediction, metrics
├── run_2d_tv_baseline.py              # 2D-coupled (TV + RW-ℓ₁) classical baseline
├── run_noise_sweep.py                 # noise sweep, single deployed model (test-time)
├── run_noise_sweep_matched.py         # noise sweep, networks retrained per SNR level
├── run_wavelet_mismatch.py            # wrong-wavelet robustness (freq + phase error)
├── run_multiseed_ci.py                # 5-seed confidence intervals (scratch + transfer)
├── generate_*.py                      # helper scripts for derivation/architecture figures
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.x, NumPy, SciPy, Matplotlib (exact versions in
`requirements.txt`). Runs on CPU; automatically uses Apple-Silicon `mps` or CUDA
if available.

---

## Reproducing the results

All runner scripts auto-discover the crops under `ADDM_PINN_RESULTS/` and read
the saved `2D_results.npz` (true model, clean seismic, wavelet) so they run
without the raw Marmousi-2 SEG-Y files.

```bash
# Stronger 2D-coupled classical baseline (lateral TV + reweighted ℓ₁)
python run_2d_tv_baseline.py

# Noise-robustness sweep — single deployed model applied across SNR levels
python run_noise_sweep.py

# Noise-robustness sweep — networks retrained at each SNR level (matched protocol)
python run_noise_sweep_matched.py

# Wavelet-mismatch robustness — wrong wavelet, no retraining
python run_wavelet_mismatch.py

# Five-seed confidence intervals (from random init; add --transfer for warm-start)
python run_multiseed_ci.py
python run_multiseed_ci.py --transfer
```

Each script prints per-run metrics and writes JSON + PNG into its own
`*_RESULTS/` folder. The checkpoints needed for the test-time experiments are
included under `ADDM_PINN_RESULTS/`.

### Building the manuscript

```bash
cd manuscript
latexmk -pdf MANUSCRIPT_CORRECTED.tex
```

---

## Data

The controlled synthetic section is generated in code. The Marmousi-2 crops are
derived from the public Marmousi-2 elastic model (Martin et al., 2006). The raw
SEG-Y `vp`/`density` files are **not** redistributed here; the cropped impedance,
seismic, and wavelet used in every experiment are stored in each crop's
`2D_results.npz`, which is sufficient to reproduce all results.

---

## Citation

If you use this code, please cite the manuscript (update once published):

```bibtex
@article{kumar_admm_pinn_2026,
  title   = {ADMM-Guided Physics-Informed Deep Learning for Two-Dimensional
             Acoustic Impedance Inversion with Reweighted L1 Sparse Regularization},
  author  = {Kumar, Deepak and Tripathi, Jayant Nath},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

Released under the MIT License (see `LICENSE`).
