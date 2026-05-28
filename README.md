# Physics-Informed-Neuaral-Network
Seismic Inversion using PINN and CNN
2D ADMM-PINN Code

This repository contains the Python code for the 2D ADMM-guided physics-informed neural inversion experiment.

Files:

- 2D_ADMM_PINN_ResCNN_Attention.py

- Main 2D synthetic acoustic impedance inversion code. It compares reweighted L1 ADMM, ResCNN, physics-informed 2D U-Net, and Attention ResUNet.

The manuscript package,  and PDFs are intentionally not included here.

Basic usage

Run the 2D experiment:

python 02_2D_ADMM_PINN_ResCNN_Attention.py

The code expects the Python scientific stack used in the local seismic-ml environment: NumPy, SciPy, Matplotlib, and PyTorch.
