# Physics-Informed-Neural-Network

Seismic inversion using ADMM-guided physics-informed neural networks.

This repository contains the Python code for the 2D ADMM-guided physics-informed neural inversion experiment.

Main files:

- 2D_ADMM_PINN_ResCNN_Attention.py

- Main 2D synthetic acoustic impedance inversion code. It compares reweighted L1 ADMM, ResCNN, physics-informed 2D U-Net, and Attention ResUNet.

- marmousi_fault_crop_400x300/

- Reproducibility archive for the fault-focused cropped Marmousi transfer benchmark. It contains the executed Python script, `2D_results.npz`, and `2D_results_metadata.json`.

The manuscript and PDFs are intentionally not included here.

Basic usage

Run the 2D experiment:

```bash
python 2D_ADMM_PINN_ResCNN_Attention.py
```

Run the uploaded Marmousi benchmark script:

```bash
python marmousi_fault_crop_400x300/02_2D_Laptop_ADMM_PINN_ResCNN_Attention.py \
  --benchmark marmousi-crop \
  --run-tag marmousi_fault_crop_400x300 \
  --epochs 80 \
  --device cpu
```

The code expects the Python scientific stack used in the local seismic-ml environment: NumPy, SciPy, Matplotlib, and PyTorch.
