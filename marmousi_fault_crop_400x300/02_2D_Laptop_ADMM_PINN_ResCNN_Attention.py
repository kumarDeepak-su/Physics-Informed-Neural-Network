#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
  Seismic Acoustic Impedance Inversion — 2D Synthetic Section
  Part 1 – Reweighted L1 sparse inversion   (ADMM, trace-by-trace)
  Part 2 – Physics-Informed 2D U-Net        (patch-based, reweighted L1 loss)
  Part 3 – Side-by-side comparison figures

  Relationship to the 1D code
  ───────────────────────────
  This 2D code extends the 1D single-trace study to a full 2D section.
  Every algorithm, equation, parameter name and loss-function structure
  is kept IDENTICAL to the 1D version so the paper presents a clean
  1D → 2D progression:

    1D  Classical :  single-trace ADMM
    2D  Classical :  same ADMM applied trace-by-trace across the section

    1D  Neural    :  PINN / CoordMLP  (single trace, no dataset)
    2D  Neural    :  Physics-Informed 2D U-Net  (patch-based)
                     • SAME physics loss (conv forward model per trace via Conv2d)
                     • SAME reweighted L1 sparsity on reflectivity
                     • ADDS cross-trace lateral coherence via (3,7) Conv2d kernels

  Consistency checklist (matches 1D code exactly)
  ────────────────────────────────────────────────
  ✓ ADMM parameters   : MU, ALPHA, LAMBDA_  (same names)
  ✓ Forward model      : D_half = 0.5·D  (linearised Born approximation)
  ✓ Reflectivity       : exact r = (Z[i+1]-Z[i])/(Z[i+1]+Z[i]) in neural
  ✓ Reweighted L1      : M = 1/(|r|+ε)  in both ADMM and neural
  ✓ Loss structure     : W_PHYSICS·MSE + W_SPARSE·RW-L1 + w_init·MSE_init
  ✓ Annealing          : w_init linearly decays over WARMUP_EPOCHS
  ✓ Metrics            : SNR = 20·log10(‖Z‖/‖ΔZ‖),  RMSE = √mean(ΔZ²)

  Key fixes applied
  ─────────────────
  v1→v2  D_half = 0.5·D used consistently in ADMM forward model.
  v2→v3  Terminal anchor in compute_patch_corners prevents coverage gap.
  v3→v4  ADMM parameters re-calibrated: MU=5e-6, ALPHA=5e-5, LAMBDA_=1e-4.

  Author : Deepak Kumar
  Date   : 2026-03-18
==============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import json
import os
import platform
import struct
import sys
import time

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d, gaussian_filter

matplotlib.rcParams.update({
    'font.size'      : 11,
    'axes.titlesize' : 12,
    'axes.labelsize' : 11,
    'figure.dpi'     : 100,
})

# ==============================================================================
#  GLOBAL CONFIGURATION
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── 2D model geometry ────────────────────────────────────────────────────────
NX         = 400       # compact 2D benchmark: number of traces (lateral)
NT         = 300       # compact 2D benchmark: time samples per trace
DT         = 0.001     # sampling interval (s)
F0         = 30.0      # dominant wavelet frequency (Hz)
NOISE_SNR  = 8.0       # harder noisy case for testing neural robustness
SMOOTH_SIGMA = 12.0    # Gaussian sigma for initial-model smoothing

# Nominal layer boundaries before faulting and lens insertion
LAYER_TOPS = [0, 60, 125, 200, NT]

# ── Classical ADMM parameters ────────────────────────────────────────────────
# Same calibration rule as 1D code: LAMBDA_ ≈ 20×MU, ALPHA ≈ 10×MU
MU          = 5e-6      # L1 sparsity weight on reflectivity
ALPHA       = 5e-5      # Tikhonov weight (proximity to initial model L0)
LAMBDA_     = 1e-4      # ADMM augmented-Lagrangian parameter
EPSILON_CL  = 1e-4      # reweighting stability floor
MAX_ITER_CL = 200       # maximum ADMM iterations
TOL_CL      = 1e-6      # relative convergence threshold

# ── U-Net architecture & training ────────────────────────────────────────────
PATCH_NX    = 64        # patch width (traces)
PATCH_NT    = 128       # patch height (time samples)
STRIDE_NX   = 32        # patch stride in x direction
STRIDE_NT   = 64        # patch stride in t direction

EPOCHS      = 80        # compact benchmark; increase to 300+ for larger runs
BATCH_SIZE  = 8         # smaller CPU batches for low-resource runs
LR          = 5e-4      # initial learning rate
LR_MIN      = 1e-6      # cosine annealing minimum LR
WARMUP_EPOCHS = 25      # epochs to anneal w_init for compact benchmark

# ── U-Net loss weights ───────────────────────────────────────────────────────
W_PHYSICS    = 3.0      # seismic data-fit MSE (1.0→3.0: physics must dominate)
W_SPARSE     = 0.01     # reweighted L1 on reflectivity (0.02→0.01)
W_INIT_START = 0.5      # initial-model proximity weight at epoch 0
W_INIT_END   = 0.01     # initial-model proximity weight after warm-up
W_LATERAL    = 0.005    # lateral smoothness weight (0.003→0.005)
W_SUPERVISED = 2.0      # synthetic benchmark only: impedance target loss
W_GRAD       = 0.6      # gradient-domain loss to sharpen thin beds/fault edges
UNET_BASE_CH = 16        # CPU-friendly; keep same in Marmousi when loading this checkpoint
USE_HE_ADMM_AS_NEURAL_INITIAL = True  # refine the He et al. RW-L1 ADMM result
RUN_ATTENTION = False     # server preset enables Attention ResUNet ablation
RUN_TAG = "laptop"      # retained for compatibility with archived outputs
BENCHMARK = "synthetic"

# ── Reweighting schedule ─────────────────────────────────────────────────────
REWEIGHT_EVERY = 20     # update reweighting every N epochs
EPSILON_RW     = 1e-4   # reweighting stability floor

# ── Figures ──────────────────────────────────────────────────────────────────
SAVE_DPI = 600

# ── Saved outputs ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(SCRIPT_DIR, "ADDM_PINN_RESULTS")
RESULTS_NPZ = os.path.join(RESULTS_DIR, "2D_results.npz")
RESULTS_JSON = os.path.join(RESULTS_DIR, "2D_results_metadata.json")
UNET_CLEAN_CKPT = os.path.join(RESULTS_DIR, "2D_UNet_clean_pretrain.pt")
UNET_NOISY_CKPT = os.path.join(RESULTS_DIR, "2D_UNet_noisy_pretrain.pt")
RESCNN_CLEAN_CKPT = os.path.join(RESULTS_DIR, "2D_ResCNN_clean_pretrain.pt")
RESCNN_NOISY_CKPT = os.path.join(RESULTS_DIR, "2D_ResCNN_noisy_pretrain.pt")
ATTN_CLEAN_CKPT = os.path.join(RESULTS_DIR, "2D_AttentionResUNet_clean_pretrain.pt")
ATTN_NOISY_CKPT = os.path.join(RESULTS_DIR, "2D_AttentionResUNet_noisy_pretrain.pt")


# ==============================================================================
#  SECTION 1 – HELPER FUNCTIONS
# ==============================================================================

def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def result_path(filename: str) -> str:
    ensure_results_dir()
    return os.path.join(RESULTS_DIR, filename)


def refresh_output_paths() -> None:
    global RESULTS_NPZ, RESULTS_JSON, UNET_CLEAN_CKPT, UNET_NOISY_CKPT
    global RESCNN_CLEAN_CKPT, RESCNN_NOISY_CKPT, ATTN_CLEAN_CKPT, ATTN_NOISY_CKPT
    RESULTS_NPZ = os.path.join(RESULTS_DIR, "2D_results.npz")
    RESULTS_JSON = os.path.join(RESULTS_DIR, "2D_results_metadata.json")
    UNET_CLEAN_CKPT = os.path.join(RESULTS_DIR, "2D_UNet_clean_pretrain.pt")
    UNET_NOISY_CKPT = os.path.join(RESULTS_DIR, "2D_UNet_noisy_pretrain.pt")
    RESCNN_CLEAN_CKPT = os.path.join(RESULTS_DIR, "2D_ResCNN_clean_pretrain.pt")
    RESCNN_NOISY_CKPT = os.path.join(RESULTS_DIR, "2D_ResCNN_noisy_pretrain.pt")
    ATTN_CLEAN_CKPT = os.path.join(RESULTS_DIR, "2D_AttentionResUNet_clean_pretrain.pt")
    ATTN_NOISY_CKPT = os.path.join(RESULTS_DIR, "2D_AttentionResUNet_noisy_pretrain.pt")


def json_ready(value):
    """Convert NumPy/PyTorch scalar values to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return value


def file_record(path: str) -> dict:
    """Small file record for reproducibility metadata."""
    exists = os.path.exists(path)
    record = {"path": path, "exists": exists}
    if exists:
        record.update(size_bytes=os.path.getsize(path),
                      modified_time=time.strftime(
                          "%Y-%m-%dT%H:%M:%S%z",
                          time.localtime(os.path.getmtime(path))))
    return record


def save_metadata_json(args, metrics: dict, timings: dict, marmousi_crop,
                       dt_used: float) -> None:
    """Write a compact reproducibility record next to the numerical NPZ."""
    transfer_dir = (os.path.abspath(args.transfer_dir)
                    if args.transfer_dir else default_transfer_dir())
    metadata = {
        "description": (
            "Reproducibility metadata for the ADMM-guided physics-informed "
            "2D acoustic-impedance benchmark."),
        "command": " ".join(sys.argv),
        "benchmark": args.benchmark,
        "run_tag": RUN_TAG,
        "results_dir": RESULTS_DIR,
        "software": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "device": {
            "requested": args.device,
            "used": str(DEVICE),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "geometry": {
            "nx": NX,
            "nt": NT,
            "dt_seconds": float(dt_used),
            "wavelet_frequency_hz": F0,
            "noise_snr_db": NOISE_SNR,
            "smooth_sigma": SMOOTH_SIGMA,
        },
        "patch_training": {
            "patch_nx": PATCH_NX,
            "patch_nt": PATCH_NT,
            "stride_nx": STRIDE_NX,
            "stride_nt": STRIDE_NT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "learning_rate_min": LR_MIN,
            "warmup_epochs": WARMUP_EPOCHS,
            "base_channels": UNET_BASE_CH,
            "run_attention": RUN_ATTENTION,
            "transfer_learning_enabled": not args.no_transfer,
            "transfer_dir": transfer_dir,
        },
        "loss_weights": {
            "physics": W_PHYSICS,
            "sparse": W_SPARSE,
            "init_start": W_INIT_START,
            "init_end": W_INIT_END,
            "lateral": W_LATERAL,
            "supervised": W_SUPERVISED,
            "gradient": W_GRAD,
            "supervised_benchmark_enabled": not args.no_supervised_benchmark,
        },
        "admm_parameters": {
            "mu": MU,
            "alpha": ALPHA,
            "lambda": LAMBDA_,
            "epsilon": EPSILON_CL,
            "max_iter": MAX_ITER_CL,
            "tolerance": TOL_CL,
        },
        "marmousi": None,
        "metrics": metrics,
        "timings_seconds": timings,
        "outputs": {
            "npz": file_record(RESULTS_NPZ),
            "json": {"path": RESULTS_JSON},
            "figures": [
                file_record(result_path(name)) for name in (
                    "Marmousi_Crop_Location.png",
                    "2D_Fig1_Model_Data.png",
                    "2D_Fig8_Metrics.png",
                    "2D_Fig9_Architecture_Comparison.png",
                )
            ],
        },
    }

    if marmousi_crop is not None:
        base = os.path.abspath(args.marmousi_dir) if args.marmousi_dir else project_root()
        vp_path = (os.path.abspath(args.marmousi_vp)
                   if args.marmousi_vp else os.path.join(base, "vp.segy"))
        rho_path = (os.path.abspath(args.marmousi_density)
                    if args.marmousi_density else os.path.join(base, "density.segy"))
        x0, x1, t0, t1 = [int(v) for v in marmousi_crop]
        metadata["marmousi"] = {
            "vp_file": file_record(vp_path),
            "density_file": file_record(rho_path),
            "crop_trace_start": x0,
            "crop_trace_end_exclusive": x1,
            "crop_sample_start": t0,
            "crop_sample_end_exclusive": t1,
            "crop_shape": [x1 - x0, t1 - t0],
        }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(json_ready(metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  Saved reproducibility metadata: {RESULTS_JSON}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADMM-guided physics-informed 2D seismic impedance inversion")
    parser.add_argument("--preset", choices=["laptop", "server"], default="laptop",
                        help="Use compact benchmark defaults or larger server/HPC defaults.")
    parser.add_argument("--benchmark", choices=["synthetic", "marmousi-crop"],
                        default="synthetic",
                        help="Run the controlled synthetic section or a cropped Marmousi benchmark.")
    parser.add_argument("--marmousi-dir", default=None,
                        help="Folder containing vp.segy and density.segy. Defaults to the project root.")
    parser.add_argument("--marmousi-vp", default=None,
                        help="Optional path to Marmousi Vp SEG-Y.")
    parser.add_argument("--marmousi-density", default=None,
                        help="Optional path to Marmousi density SEG-Y.")
    parser.add_argument("--marmousi-trace-start", type=int, default=8100,
                        help="First trace of the Marmousi crop.")
    parser.add_argument("--marmousi-sample-start", type=int, default=1500,
                        help="First sample of the Marmousi crop.")
    parser.add_argument("--transfer-dir", default=None,
                        help="Folder containing synthetic pretraining checkpoints.")
    parser.add_argument("--no-transfer", action="store_true",
                        help="Disable synthetic-to-Marmousi transfer initialization.")
    parser.add_argument("--no-supervised-benchmark", action="store_true",
                        help="Do not use known true impedance in the neural loss.")
    parser.add_argument("--results-dir", default=None,
                        help="Output directory. Defaults to ADDM_PINN_RESULTS/<run-tag> for server.")
    parser.add_argument("--run-tag", default=None,
                        help="Subfolder/tag for outputs, e.g. server_1200x600.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-ch", type=int, default=None)
    parser.add_argument("--patch-nx", type=int, default=None)
    parser.add_argument("--patch-nt", type=int, default=None)
    parser.add_argument("--stride-nx", type=int, default=None)
    parser.add_argument("--stride-nt", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--noise-snr", type=float, default=None)
    parser.add_argument("--with-attention", action="store_true",
                        help="Train Attention ResUNet in addition to U-Net and ResCNN.")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable Attention ResUNet even in server preset.")
    return parser.parse_args()


def configure_runtime(args) -> None:
    global DEVICE, NX, NT, LAYER_TOPS, PATCH_NX, PATCH_NT, STRIDE_NX, STRIDE_NT
    global EPOCHS, BATCH_SIZE, LR, WARMUP_EPOCHS, REWEIGHT_EVERY, UNET_BASE_CH
    global NOISE_SNR, RESULTS_DIR, RUN_ATTENTION, RUN_TAG, BENCHMARK

    BENCHMARK = args.benchmark
    RUN_TAG = args.run_tag or args.preset

    if args.device == "cuda":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "mps":
        DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "cpu":
        DEVICE = torch.device("cpu")

    if args.preset == "server":
        NX = 1200
        NT = 600
        PATCH_NX = 128
        PATCH_NT = 256
        STRIDE_NX = 64
        STRIDE_NT = 128
        EPOCHS = 300
        BATCH_SIZE = 8
        LR = 3e-4
        WARMUP_EPOCHS = 80
        REWEIGHT_EVERY = 30
        UNET_BASE_CH = 32
        RUN_ATTENTION = True
        if args.run_tag is None:
            RUN_TAG = f"server_{NX}x{NT}_b{UNET_BASE_CH}"
    else:
        RUN_ATTENTION = False

    if args.nx is not None:
        NX = args.nx
    if args.nt is not None:
        NT = args.nt
    if args.patch_nx is not None:
        PATCH_NX = args.patch_nx
    if args.patch_nt is not None:
        PATCH_NT = args.patch_nt
    if args.stride_nx is not None:
        STRIDE_NX = args.stride_nx
    if args.stride_nt is not None:
        STRIDE_NT = args.stride_nt
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.base_ch is not None:
        UNET_BASE_CH = args.base_ch
    if args.lr is not None:
        LR = args.lr
    if args.noise_snr is not None:
        NOISE_SNR = args.noise_snr
    if args.with_attention:
        RUN_ATTENTION = True
    if args.no_attention:
        RUN_ATTENTION = False

    WARMUP_EPOCHS = min(WARMUP_EPOCHS, max(EPOCHS // 3, 1))
    LAYER_TOPS = [0, int(0.20 * NT), int(0.42 * NT), int(0.67 * NT), NT]

    if args.run_tag is None and args.benchmark == "marmousi-crop":
        RUN_TAG = f"marmousi_crop_{args.preset}_{NX}x{NT}"

    if args.results_dir is not None:
        RESULTS_DIR = os.path.abspath(args.results_dir)
    elif args.preset == "server" or args.run_tag is not None or args.benchmark != "synthetic":
        RESULTS_DIR = os.path.join(SCRIPT_DIR, "ADDM_PINN_RESULTS", RUN_TAG)

    refresh_output_paths()


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convolution_matrix(w: np.ndarray, n: int) -> np.ndarray:
    """Build (n+nw-1) × n Toeplitz convolution matrix for wavelet w."""
    nw     = len(w)
    n_rows = n + nw - 1
    col    = np.zeros(n_rows); col[:nw] = w
    row    = np.zeros(n);      row[0]   = w[0]
    return toeplitz(col, row)


def difference_matrix(m: int) -> np.ndarray:
    """First-difference matrix D, shape (m-1, m).  Born operator = 0.5*D."""
    D = np.zeros((m - 1, m))
    idx = np.arange(m - 1)
    D[idx, idx]     = -1.0
    D[idx, idx + 1] =  1.0
    return D


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft-thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def ricker_wavelet(f0: float, dt: float, half_duration: float = 0.06) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet — identical to 1D code."""
    t = np.arange(-half_duration, half_duration, dt)
    w = (1.0 - 2.0 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)
    return w / np.max(np.abs(w))


def add_noise(S: np.ndarray, target_snr: float, seed: int = 42) -> np.ndarray:
    """Add white Gaussian noise to achieve a given amplitude SNR."""
    rng       = np.random.default_rng(seed)
    sig_rms   = np.linalg.norm(S) / np.sqrt(S.size)
    noise_std = sig_rms / target_snr
    return S + rng.normal(0.0, noise_std, S.shape)


def compute_metrics(Z_pred: np.ndarray, Z_true: np.ndarray) -> tuple:
    """SNR (dB) and RMSE.  Same formula as 1D code."""
    err  = Z_pred - Z_true
    snr  = 20.0 * np.log10(np.linalg.norm(Z_true) / (np.linalg.norm(err) + 1e-12))
    rmse = np.sqrt(np.mean(err ** 2))
    return float(snr), float(rmse)


def build_layered_model(nx: int, nt: int, dt: float,
                        layer_tops: list) -> tuple:
    """
    Build a complex 2D impedance model with dipping layers, a normal fault,
    thin beds, and an oval impedance body between layers.

    This is intentionally harder than the original four horizontal layers so
    trace-by-trace He et al. RW-L1 ADMM shows its limits and the 2D neural
    refinement has meaningful lateral/geometric information to exploit.
    """
    x = np.arange(nx, dtype=float)
    t = np.arange(nt, dtype=float)
    X, T = np.meshgrid(x, t, indexing='ij')

    x_norm = x / max(nx - 1, 1)
    fault_x = 0.53 * nx
    throw = 28.0 / (1.0 + np.exp(-(x - fault_x) / 4.0))

    b1 = layer_tops[1] + 12.0 * x_norm + 7.0 * np.sin(2 * np.pi * x_norm) + 0.35 * throw
    b2 = layer_tops[2] + 25.0 * x_norm + 10.0 * np.sin(2 * np.pi * (x_norm + 0.15)) + 0.70 * throw
    b3 = layer_tops[3] + 18.0 * x_norm + 9.0 * np.sin(2 * np.pi * (x_norm + 0.35)) + 1.00 * throw
    b1 = np.clip(b1, 35, nt - 120)
    b2 = np.maximum(b2, b1 + 35)
    b3 = np.maximum(b3, b2 + 45)
    b3 = np.clip(b3, 160, nt - 25)

    Z = np.zeros((nx, nt), dtype=float)
    layer_values = np.array([8200.0, 9700.0, 11100.0, 12600.0])
    for i in range(nx):
        i1, i2, i3 = int(round(b1[i])), int(round(b2[i])), int(round(b3[i]))
        Z[i, :i1] = layer_values[0]
        Z[i, i1:i2] = layer_values[1]
        Z[i, i2:i3] = layer_values[2]
        Z[i, i3:] = layer_values[3]

    # Thin beds below the second interface; deliberately below wavelet tuning.
    for amp, offset, thick in [(650.0, 16, 4), (-520.0, 27, 3), (420.0, 39, 3)]:
        for i in range(nx):
            center = int(round(b2[i] + offset + 4.0 * np.sin(2 * np.pi * x_norm[i] * 1.7)))
            top = max(0, center - thick // 2)
            bot = min(nt, center + thick // 2 + 1)
            Z[i, top:bot] += amp

    # Oval body placed between layers, crossing the faulted stratigraphy.
    oval = (((X - 0.63 * nx) / (0.16 * nx)) ** 2 +
            ((T - 0.56 * nt) / (0.105 * nt)) ** 2) <= 1.0
    Z[oval] = 7600.0

    # A compact high-impedance inclusion on the upthrown side.
    hard_body = (((X - 0.35 * nx) / (0.075 * nx)) ** 2 +
                 ((T - 0.70 * nt) / (0.07 * nt)) ** 2) <= 1.0
    Z[hard_body] = 13300.0

    # Gentle lateral compaction trend plus deterministic texture.
    Z += 250.0 * x_norm[:, None]
    Z += 90.0 * np.sin(2 * np.pi * X / nx * 3.0) * np.exp(-T / (0.8 * nt))
    Z = np.clip(Z, 7200.0, 13600.0)

    return Z, x


def ibm32_to_float(raw: bytes) -> np.ndarray:
    """Convert big-endian IBM 32-bit floating point samples to float32."""
    u = np.frombuffer(raw, dtype=">u4")
    sign = np.where((u >> 31) & 1, -1.0, 1.0)
    exponent = ((u >> 24) & 0x7f).astype(np.int32)
    fraction = (u & 0x00ffffff).astype(np.float64)
    out = sign * (fraction / float(0x01000000)) * np.power(16.0, exponent - 64)
    out[u == 0] = 0.0
    return out.astype(np.float32)


def read_segy_simple(filepath: str) -> tuple:
    """
    Minimal SEG-Y reader for local Marmousi files.

    Supports IBM float format code 1 and IEEE float format code 5, with
    fixed-length traces and standard 3200-byte text + 400-byte binary header.
    """
    with open(filepath, "rb") as f:
        f.seek(3200)
        binary_header = f.read(400)
        if len(binary_header) != 400:
            raise ValueError(f"Invalid SEG-Y header in {filepath}")

        dt_us = struct.unpack(">H", binary_header[16:18])[0]
        ns = struct.unpack(">H", binary_header[20:22])[0]
        fmt = struct.unpack(">H", binary_header[24:26])[0]
        if ns <= 0:
            raise ValueError(f"SEG-Y sample count is invalid in {filepath}")
        if fmt not in (1, 5):
            raise ValueError(f"Unsupported SEG-Y sample format code {fmt} in {filepath}")

        trace_bytes = 240 + ns * 4
        f.seek(0, os.SEEK_END)
        size = f.tell()
        payload = size - 3600
        if payload <= 0 or payload % trace_bytes != 0:
            raise ValueError(f"SEG-Y trace size does not match header in {filepath}")
        ntrace = payload // trace_bytes

        data = np.empty((ntrace, ns), dtype=np.float32)
        f.seek(3600)
        for itr in range(ntrace):
            f.seek(240, os.SEEK_CUR)
            raw = f.read(ns * 4)
            if len(raw) != ns * 4:
                raise ValueError(f"Unexpected end of file in {filepath} at trace {itr}")
            if fmt == 1:
                data[itr] = ibm32_to_float(raw)
            else:
                data[itr] = np.frombuffer(raw, dtype=">f4").astype(np.float32)

    dt = dt_us / 1e6 if dt_us > 0 else DT
    return data, float(dt)


def project_root() -> str:
    return os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def load_marmousi_crop(args, nx: int, nt: int) -> tuple:
    """Load Vp and density SEG-Y files, crop them, and return acoustic impedance."""
    base = os.path.abspath(args.marmousi_dir) if args.marmousi_dir else project_root()
    vp_path = os.path.abspath(args.marmousi_vp) if args.marmousi_vp else os.path.join(base, "vp.segy")
    rho_path = (os.path.abspath(args.marmousi_density)
                if args.marmousi_density else os.path.join(base, "density.segy"))

    if not os.path.exists(vp_path):
        raise FileNotFoundError(f"Marmousi Vp file not found: {vp_path}")
    if not os.path.exists(rho_path):
        raise FileNotFoundError(f"Marmousi density file not found: {rho_path}")

    print(f"  Reading Marmousi Vp: {vp_path}")
    vp, dt = read_segy_simple(vp_path)
    print(f"  Reading Marmousi density: {rho_path}")
    rho, dt_rho = read_segy_simple(rho_path)
    if vp.shape != rho.shape:
        raise ValueError(f"Vp and density shapes differ: {vp.shape} vs {rho.shape}")
    if abs(dt - dt_rho) > 1e-9:
        print(f"  Warning: Vp dt={dt:g}s differs from density dt={dt_rho:g}s; using Vp dt.")

    x0 = args.marmousi_trace_start
    t0 = args.marmousi_sample_start
    x1 = x0 + nx
    t1 = t0 + nt
    if x0 < 0 or t0 < 0 or x1 > vp.shape[0] or t1 > vp.shape[1]:
        raise ValueError(
            f"Marmousi crop [{x0}:{x1}, {t0}:{t1}] exceeds model shape {vp.shape}")

    Z_full = vp * rho
    figure_marmousi_crop_overview(Z_full, (x0, x1, t0, t1))

    vp_crop = vp[x0:x1, t0:t1].astype(np.float64)
    rho_crop = rho[x0:x1, t0:t1].astype(np.float64)
    Z = vp_crop * rho_crop
    x_coord = np.arange(x0, x1, dtype=float)
    return Z, x_coord, dt, (x0, x1, t0, t1)


def default_transfer_dir() -> str:
    return os.path.join(project_root(), "ADDM_PINN_RESULTS")


def transfer_checkpoint_path(model_name: str, data_label: str, args) -> str:
    if args.benchmark != "marmousi-crop" or args.no_transfer:
        return None
    transfer_dir = os.path.abspath(args.transfer_dir) if args.transfer_dir else default_transfer_dir()
    names = {
        "U-Net": "2D_UNet",
        "ResCNN": "2D_ResCNN",
        "AttentionResUNet": "2D_AttentionResUNet",
    }
    prefix = names[model_name]
    return os.path.join(transfer_dir, f"{prefix}_{data_label}_pretrain.pt")


def load_compatible_checkpoint(net: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        print(f"  Transfer checkpoint not found: {checkpoint_path}")
        return
    # Local project checkpoints include NumPy metadata, so PyTorch >=2.6 needs
    # weights_only=False. These checkpoints are generated by this project.
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    current = net.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(compatible)
    net.load_state_dict(current)
    print(f"  Loaded transfer checkpoint: {checkpoint_path}")
    print(f"  Compatible tensors loaded: {len(compatible)}/{len(current)}")


# ==============================================================================
#  SECTION 2 – PATCH UTILITIES
# ==============================================================================

def compute_patch_corners(nx: int, nt: int,
                          patch_nx: int, patch_nt: int,
                          stride_nx: int, stride_nt: int) -> list:
    """
    Compute all (x0, t0) patch top-left corners with terminal anchors
    to guarantee full spatial coverage.
    """
    x_positions = list(range(0, nx - patch_nx + 1, stride_nx))
    t_positions = list(range(0, nt - patch_nt + 1, stride_nt))

    # Terminal anchors — ensure last samples are covered
    if len(x_positions) == 0 or x_positions[-1] < nx - patch_nx:
        x_positions.append(nx - patch_nx)
    if len(t_positions) == 0 or t_positions[-1] < nt - patch_nt:
        t_positions.append(nt - patch_nt)

    corners = [(x0, t0) for x0 in x_positions for t0 in t_positions]
    return corners


def extract_patches_at_corners(data: np.ndarray, corners: list,
                               patch_nx: int, patch_nt: int) -> np.ndarray:
    """Extract 2D patches from data[nx, nt] at given corner positions."""
    patches = np.zeros((len(corners), patch_nx, patch_nt))
    for k, (x0, t0) in enumerate(corners):
        patches[k] = data[x0:x0 + patch_nx, t0:t0 + patch_nt]
    return patches


def reconstruct_from_patches(patches: np.ndarray, corners: list,
                             nx: int, nt: int,
                             patch_nx: int, patch_nt: int) -> np.ndarray:
    """Overlap-add reconstruction from patches back to full section."""
    recon  = np.zeros((nx, nt))
    counts = np.zeros((nx, nt))
    for k, (x0, t0) in enumerate(corners):
        recon[x0:x0 + patch_nx, t0:t0 + patch_nt]  += patches[k]
        counts[x0:x0 + patch_nx, t0:t0 + patch_nt] += 1.0
    counts = np.maximum(counts, 1.0)
    return recon / counts


# ==============================================================================
#  SECTION 3 – CLASSICAL INVERSIONS  (ADMM)
# ==============================================================================
#
#  Solves:  min_L  ½ ‖W · D_half · L − S‖²  +  α/2 ‖L − L₀‖²
#                                              +  μ ‖M · D_half · L‖₁
#
#  where  D_half = 0.5 · D   (linearised Born reflectivity operator)
#         M = 1/(|r|+ε)      (reweighting diagonal for adaptive sparsity)
#

def admm_l1_single_trace(S, w, L0, mu, alpha, lambda_,
                          epsilon=1e-4, max_iter=200, tol=1e-6,
                          reweight=True):
    """
    ADMM L1 / Reweighted L1 inversion for a single trace.

    Uses D_half = 0.5 * D consistently (same as 1D code).

    Parameters
    ----------
    S         : observed seismic trace (length m + nw - 2)
    w         : wavelet (length nw)
    L0        : initial log-impedance (length m)
    mu        : L1 sparsity weight
    alpha     : Tikhonov (proximity) weight
    lambda_   : ADMM penalty parameter
    epsilon   : reweighting floor
    max_iter  : max iterations
    tol       : convergence tolerance
    reweight  : if True → reweighted L1; if False → standard L1 (M=1)

    Returns
    -------
    Z_inv : inverted impedance = exp(L)
    convergence : list of relative changes
    """
    m  = len(L0)
    nw = len(w)

    # Forward operators
    D      = difference_matrix(m)
    D_half = 0.5 * D                      # ← KEY: linearised Born operator
    W_mat  = convolution_matrix(w, m - 1)

    # Pre-compute fixed matrices
    WDh     = W_mat @ D_half              # forward: seismic = W @ D_half @ L
    WDhtWDh = WDh.T @ WDh
    WDhtS   = WDh.T @ S
    alphaI  = alpha * np.eye(m)

    # Initialise ADMM variables
    L = L0.copy()
    R = np.zeros(m - 1)                   # auxiliary (split variable)
    C = np.zeros(m - 1)                   # scaled dual variable
    M = np.ones(m - 1)                    # reweighting diagonal

    convergence = []

    for it in range(max_iter):
        L_old = L.copy()

        # L-update: solve linear system
        MDh       = M[:, None] * D_half
        MtM_part  = lambda_ * MDh.T @ MDh
        A         = WDhtWDh + alphaI + MtM_part
        b         = WDhtS + alpha * L0 + lambda_ * D_half.T @ (M * (R - C))
        L         = np.linalg.solve(A, b)

        # R-update: soft-thresholding
        y   = D_half @ L                  # reflectivity estimate
        tmp = M * y + C
        R   = soft_threshold(tmp, mu / lambda_)

        # C-update: dual ascent
        C = C + (M * y - R)

        # Reweight M (adaptive sparsity)
        if reweight:
            M = 1.0 / (np.abs(y) + epsilon)

        # Convergence check
        rel_diff = np.linalg.norm(L - L_old) / (np.linalg.norm(L_old) + 1e-12)
        convergence.append(rel_diff)
        if rel_diff < tol:
            break

    return np.exp(L), convergence


# ==============================================================================
#  SECTION 4 – PHYSICS-INFORMED 2D U-NET
# ==============================================================================

class ConvBlock2D(nn.Module):
    """Two Conv2d layers with asymmetric kernels + BatchNorm + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, kx: int = 3, kt: int = 7):
        super().__init__()
        pad_x, pad_t = kx // 2, kt // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (kx, kt), padding=(pad_x, pad_t)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, (kx, kt), padding=(pad_x, pad_t)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """
    Full 2D Physics-Informed U-Net for acoustic impedance inversion.

    The network sees two channels per patch:
        1. observed seismic patch, normalised per patch
        2. current impedance prior L0, standardised per patch

    Unlike the earlier temporal-only variant, this U-Net downsamples and
    upsamples both trace and time axes, so the bottleneck learns true 2D
    lateral/depth context. The prediction remains residual:

        L_pred = L0 + scale * residual + bias
    """
    def __init__(self, patch_nx: int, patch_nt: int,
                 base_ch: int = UNET_BASE_CH, in_ch: int = 2):
        super().__init__()
        self.patch_nx = patch_nx
        self.patch_nt = patch_nt

        self.enc1 = ConvBlock2D(in_ch, base_ch)
        self.enc2 = ConvBlock2D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock2D(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock2D(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4,
                                      kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2,
                                      kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch,
                                      kernel_size=2, stride=2)

        self.dec3 = ConvBlock2D(base_ch * 8, base_ch * 4)
        self.dec2 = ConvBlock2D(base_ch * 4, base_ch * 2)
        self.dec1 = ConvBlock2D(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

        self.out_scale = nn.Parameter(torch.tensor(0.1))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _normalise_seismic(x: torch.Tensor) -> torch.Tensor:
        amp = torch.amax(torch.abs(x), dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return x / amp

    @staticmethod
    def _standardise_model(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, seismic, L0=None):
        if L0 is None:
            x = seismic
        else:
            x = torch.cat([
                self._normalise_seismic(seismic),
                self._standardise_model(L0),
            ], dim=1)

        _, _, nx, nt = x.shape
        pad_x = (8 - nx % 8) % 8
        pad_t = (8 - nt % 8) % 8
        if pad_x > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_x), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self._match_and_cat(self.up3(b), e3)
        d3 = self.dec3(d3)
        d2 = self._match_and_cat(self.up2(d3), e2)
        d2 = self.dec2(d2)
        d1 = self._match_and_cat(self.up1(d2), e1)
        d1 = self.dec1(d1)

        residual = self.head(d1)
        residual = residual[:, :, :self.patch_nx, :self.patch_nt]

        if L0 is not None:
            return L0 + self.out_scale * residual + self.out_bias
        return residual

    @staticmethod
    def _match_and_cat(upsampled, skip):
        diff_x = skip.shape[2] - upsampled.shape[2]
        diff_t = skip.shape[3] - upsampled.shape[3]
        if diff_x > 0 or diff_t > 0:
            upsampled = F.pad(upsampled, (0, max(0, diff_t), 0, max(0, diff_x)))
        upsampled = upsampled[:, :, :skip.shape[2], :skip.shape[3]]
        return torch.cat([upsampled, skip], dim=1)


class AttentionGate2D(nn.Module):
    """Attention gate for U-Net skip connections."""
    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        self.theta = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        alpha = self.psi(self.act(self.theta(skip) + self.phi(gate)))
        return skip * alpha


class AttentionResUNet2D(nn.Module):
    """
    Attention ResUNet for ADMM-guided seismic impedance refinement.

    Attention gates suppress irrelevant skip features and emphasize faults,
    lenses, and discontinuities. Residual refinement keeps the prediction tied
    to the He et al. RW-L1 ADMM prior while allowing high-resolution correction.
    """
    def __init__(self, patch_nx: int, patch_nt: int,
                 base_ch: int = UNET_BASE_CH, in_ch: int = 2):
        super().__init__()
        self.patch_nx = patch_nx
        self.patch_nt = patch_nt

        self.enc1 = ConvBlock2D(in_ch, base_ch)
        self.enc2 = ConvBlock2D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock2D(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock2D(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate2D(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec3 = ConvBlock2D(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate2D(base_ch * 2, base_ch * 2, base_ch)
        self.dec2 = ConvBlock2D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.att1 = AttentionGate2D(base_ch, base_ch, max(base_ch // 2, 1))
        self.dec1 = ConvBlock2D(base_ch * 2, base_ch)

        self.refine = ResidualBlock2D(base_ch, dilation=1)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.out_scale = nn.Parameter(torch.tensor(0.1))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _normalise_seismic(x: torch.Tensor) -> torch.Tensor:
        amp = torch.amax(torch.abs(x), dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return x / amp

    @staticmethod
    def _standardise_model(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, seismic, L0=None):
        if L0 is None:
            x = seismic
        else:
            x = torch.cat([
                self._normalise_seismic(seismic),
                self._standardise_model(L0),
            ], dim=1)

        _, _, nx, nt = x.shape
        pad_x = (8 - nx % 8) % 8
        pad_t = (8 - nt % 8) % 8
        if pad_x > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_x), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        u3 = self.up3(b)
        s3 = self.att3(e3, u3)
        d3 = self.dec3(self._match_and_cat(u3, s3))

        u2 = self.up2(d3)
        s2 = self.att2(e2, u2)
        d2 = self.dec2(self._match_and_cat(u2, s2))

        u1 = self.up1(d2)
        s1 = self.att1(e1, u1)
        d1 = self.dec1(self._match_and_cat(u1, s1))
        d1 = self.refine(d1)

        residual = self.head(d1)
        residual = residual[:, :, :self.patch_nx, :self.patch_nt]
        if L0 is not None:
            return L0 + self.out_scale * residual + self.out_bias
        return residual

    @staticmethod
    def _match_and_cat(upsampled, skip):
        diff_x = skip.shape[2] - upsampled.shape[2]
        diff_t = skip.shape[3] - upsampled.shape[3]
        if diff_x > 0 or diff_t > 0:
            upsampled = F.pad(upsampled, (0, max(0, diff_t), 0, max(0, diff_x)))
        upsampled = upsampled[:, :, :skip.shape[2], :skip.shape[3]]
        return torch.cat([upsampled, skip], dim=1)


class ResidualBlock2D(nn.Module):
    """Dilated residual block for full 2D seismic/impedance context."""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad_x = dilation
        pad_t = 3 * dilation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 7),
                      padding=(pad_x, pad_t), dilation=(dilation, dilation)),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 7),
                      padding=(pad_x, pad_t), dilation=(dilation, dilation)),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class ResCNN2D(nn.Module):
    """
    Dilated residual CNN refinement baseline.

    This is a strong CNN baseline for comparison with the U-Net. It keeps full
    resolution throughout, uses dilated residual blocks for a wider receptive
    field, and predicts a residual correction to the He et al. impedance prior.
    """
    def __init__(self, patch_nx: int, patch_nt: int,
                 base_ch: int = UNET_BASE_CH, in_ch: int = 2):
        super().__init__()
        self.patch_nx = patch_nx
        self.patch_nt = patch_nt
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock2D(base_ch, dilation=1),
            ResidualBlock2D(base_ch, dilation=2),
            ResidualBlock2D(base_ch, dilation=4),
            ResidualBlock2D(base_ch, dilation=2),
            ResidualBlock2D(base_ch, dilation=1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(3, 7), padding=(1, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )
        self.out_scale = nn.Parameter(torch.tensor(0.1))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _normalise_seismic(x: torch.Tensor) -> torch.Tensor:
        amp = torch.amax(torch.abs(x), dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return x / amp

    @staticmethod
    def _standardise_model(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, seismic, L0=None):
        if L0 is None:
            x = seismic
        else:
            x = torch.cat([
                self._normalise_seismic(seismic),
                self._standardise_model(L0),
            ], dim=1)
        residual = self.head(self.blocks(self.stem(x)))
        residual = residual[:, :, :self.patch_nx, :self.patch_nt]
        if L0 is not None:
            return L0 + self.out_scale * residual + self.out_bias
        return residual


class PhysicsLayer2D(nn.Module):
    """
    Differentiable zero-phase 2D seismic forward operator.

    L -> exact reflectivity -> wavelet convolution along time. The full
    convolution is centre-trimmed to the impedance patch length so the loss
    compares against the displayed/trimmed seismic section.
    """
    def __init__(self, wavelet: np.ndarray):
        super().__init__()
        nw = len(wavelet)
        weight = torch.tensor(wavelet[::-1].copy(), dtype=torch.float32)
        weight = weight.view(1, 1, 1, nw)
        self.register_buffer('weight', weight)
        self.pad_t = nw - 1
        self.half_pad = (nw - 1) // 2

    def forward(self, L):
        nt_imp = L.shape[-1]
        Z = torch.exp(L)
        r = (Z[:, :, :, 1:] - Z[:, :, :, :-1]) / \
            (Z[:, :, :, 1:] + Z[:, :, :, :-1] + 1e-8)
        S_full = F.conv2d(r, self.weight, padding=(0, self.pad_t))
        S_pred = S_full[..., self.half_pad:self.half_pad + nt_imp]
        return S_pred, r


# ==============================================================================
#  SECTION 5 – U-NET TRAINING
# ==============================================================================

def train_model_2d(S_patches, L_init_patches, wavelet,
                   patch_nx, patch_nt, label="", checkpoint_path=None,
                   L_true_patches=None, model_class=UNet2D,
                   model_name="U-Net", transfer_checkpoint=None):
    """
    Train a physics-informed 2D neural impedance model on patches.

    Parameters
    ----------
    S_patches      : (N, patch_nx, patch_nt) zero-phase seismic patches
    L_init_patches : (N, patch_nx, patch_nt)  initial log-impedance patches
    L_true_patches : optional synthetic target log-impedance patches
    wavelet        : 1D wavelet array
    patch_nx, patch_nt : patch dimensions

    Returns
    -------
    net          : trained neural model
    loss_history : list of per-epoch losses
    """
    # Convert to tensors: (N, 1, patch_nx, patch_nt)
    S_t    = torch.tensor(S_patches, dtype=torch.float32).unsqueeze(1)
    L0_t   = torch.tensor(L_init_patches, dtype=torch.float32).unsqueeze(1)
    if L_true_patches is not None:
        Lt_t = torch.tensor(L_true_patches, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(S_t, L0_t, Lt_t)
    else:
        dataset = TensorDataset(S_t, L0_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         pin_memory=(DEVICE.type == "cuda"))

    net     = model_class(patch_nx, patch_nt, base_ch=UNET_BASE_CH).to(DEVICE)
    load_compatible_checkpoint(net, transfer_checkpoint)
    physics = PhysicsLayer2D(wavelet).to(DEVICE)
    optim   = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS,
                                                          eta_min=LR_MIN)

    # Reweighting tensor: (1, 1, 1, patch_nt-1)
    rw_weights = torch.ones(1, 1, 1, patch_nt - 1, device=DEVICE)

    loss_history = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        net.train()
        epoch_loss, n_batch = 0.0, 0

        # Linear annealing: w_init(epoch) = W_INIT_START -
        # (W_INIT_START - W_INIT_END) * epoch / WARMUP_EPOCHS.
        if epoch < WARMUP_EPOCHS:
            w_init = W_INIT_START - (W_INIT_START - W_INIT_END) * epoch / WARMUP_EPOCHS
        else:
            w_init = W_INIT_END

        for batch in loader:
            if L_true_patches is not None:
                S_batch, L0_batch, Lt_batch = batch
                Lt_batch = Lt_batch.to(DEVICE)
            else:
                S_batch, L0_batch = batch
                Lt_batch = None
            S_batch  = S_batch.to(DEVICE)
            L0_batch = L0_batch.to(DEVICE)

            L_pred = net(S_batch, L0_batch)
            S_pred, r_pred = physics(L_pred)

            # Losses
            loss_phys   = F.mse_loss(S_pred, S_batch)
            rw_cur = rw_weights
            if rw_cur.shape[-2:] != r_pred.shape[-2:]:
                rw_cur = torch.ones_like(r_pred)
            loss_sparse = torch.mean(rw_cur * torch.abs(r_pred))
            loss_init   = F.mse_loss(L_pred, L0_batch)

            if Lt_batch is not None:
                loss_supervised = F.mse_loss(L_pred, Lt_batch)
                grad_pred_t = L_pred[:, :, :, 1:] - L_pred[:, :, :, :-1]
                grad_true_t = Lt_batch[:, :, :, 1:] - Lt_batch[:, :, :, :-1]
                grad_pred_x = L_pred[:, :, 1:, :] - L_pred[:, :, :-1, :]
                grad_true_x = Lt_batch[:, :, 1:, :] - Lt_batch[:, :, :-1, :]
                loss_grad = (F.l1_loss(grad_pred_t, grad_true_t) +
                             F.l1_loss(grad_pred_x, grad_true_x))
            else:
                loss_supervised = torch.tensor(0.0, device=DEVICE)
                loss_grad = torch.tensor(0.0, device=DEVICE)

            # Lateral smoothness (cross-trace coherence)
            if L_pred.shape[2] > 1:
                loss_lat = torch.mean(torch.abs(L_pred[:, :, 1:, :] -
                                                L_pred[:, :, :-1, :]))
            else:
                loss_lat = torch.tensor(0.0, device=DEVICE)

            loss = (W_PHYSICS * loss_phys +
                    W_SPARSE  * loss_sparse +
                    w_init    * loss_init +
                    W_LATERAL * loss_lat +
                    W_SUPERVISED * loss_supervised +
                    W_GRAD * loss_grad)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += loss.item()
            n_batch += 1

        sched.step()
        avg_loss = epoch_loss / max(n_batch, 1)
        loss_history.append(avg_loss)

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"  [{label}] Epoch {epoch:4d}/{EPOCHS} | "
                  f"loss={avg_loss:.6f} | lr={sched.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.1f}s")

        # Update reweighting
        if (epoch + 1) % REWEIGHT_EVERY == 0 and epoch < EPOCHS - 1:
            net.eval()
            all_r = []
            with torch.no_grad():
                for batch in loader:
                    S_b, L0_b = batch[0], batch[1]
                    S_b  = S_b.to(DEVICE)
                    L0_b = L0_b.to(DEVICE)
                    _, r_p = physics(net(S_b, L0_b))
                    all_r.append(r_p.cpu())
            all_r = torch.cat(all_r, dim=0)
            mean_abs_r = torch.mean(torch.abs(all_r), dim=0, keepdim=True)
            rw_weights = (1.0 / (mean_abs_r + EPSILON_RW)).to(DEVICE)
            rw_weights = rw_weights / rw_weights.mean()

    if checkpoint_path is not None:
        torch.save({
            "model": model_name,
            "model_state_dict": net.state_dict(),
            "patch_nx": patch_nx,
            "patch_nt": patch_nt,
            "base_ch": UNET_BASE_CH,
            "wavelet": wavelet,
            "loss_history": loss_history,
            "config": {
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LR": LR,
                "W_PHYSICS": W_PHYSICS,
                "W_SPARSE": W_SPARSE,
                "W_INIT_START": W_INIT_START,
                "W_INIT_END": W_INIT_END,
                "W_LATERAL": W_LATERAL,
                "W_SUPERVISED": W_SUPERVISED,
                "W_GRAD": W_GRAD,
                "USE_HE_ADMM_AS_NEURAL_INITIAL": USE_HE_ADMM_AS_NEURAL_INITIAL,
                "model_name": model_name,
            },
        }, checkpoint_path)
        print(f"  [{label}] Saved {model_name} checkpoint: {checkpoint_path}")

    return net, loss_history


def train_unet_2d(*args, **kwargs):
    return train_model_2d(*args, model_class=UNet2D, model_name="U-Net", **kwargs)


def predict_full_section(net, S_full_trim, L_init_full,
                         corners, patch_nx, patch_nt, nx, nt):
    """
    Run trained U-Net on all patches and reconstruct full section.

    The seismic patch fed to the net matches the zero-phase trimmed patches
    used during training.
    """
    n_seismic = S_full_trim.shape[1]
    net.eval()

    L_patches = np.zeros((len(corners), patch_nx, patch_nt))

    with torch.no_grad():
        for k, (x0, t0) in enumerate(corners):
            s_end = min(t0 + patch_nt, n_seismic)
            s_len = s_end - t0
            s_patch = np.zeros((patch_nx, patch_nt))
            s_patch[:, :s_len] = S_full_trim[x0:x0 + patch_nx, t0:s_end]

            l0_patch = L_init_full[x0:x0 + patch_nx, t0:t0 + patch_nt]

            S_t  = torch.tensor(s_patch, dtype=torch.float32
                                ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            L0_t = torch.tensor(l0_patch, dtype=torch.float32
                                ).unsqueeze(0).unsqueeze(0).to(DEVICE)

            L_pred = net(S_t, L0_t)
            L_patches[k] = L_pred.squeeze().cpu().numpy()

    # Reconstruct via overlap-add then exponentiate
    L_full = reconstruct_from_patches(L_patches, corners, nx, nt,
                                       patch_nx, patch_nt)
    return np.exp(L_full)


# ==============================================================================
#  SECTION 6 – FIGURE GENERATION
# ==============================================================================

def figure_marmousi_crop_overview(Z_full, crop):
    """Show the selected crop on the full Marmousi acoustic-impedance model."""
    if BENCHMARK != "marmousi-crop":
        return

    x0, x1, t0, t1 = crop
    crop_nx = x1 - x0
    crop_nt = t1 - t0
    rx0 = max(0, x0 - max(crop_nx, 500))
    rx1 = min(Z_full.shape[0], x1 + max(crop_nx, 500))
    rt0 = max(0, t0 - max(crop_nt, 400))
    rt1 = min(Z_full.shape[1], t1 + max(crop_nt, 400))

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    vmin = np.percentile(Z_full, 1) / 1e4
    vmax = np.percentile(Z_full, 99) / 1e4

    full_view = Z_full[::8, ::2]
    im = axes[0].imshow(full_view.T / 1e4, aspect='auto', cmap='turbo',
                        extent=[0, Z_full.shape[0] - 1,
                                Z_full.shape[1] - 1, 0],
                        vmin=vmin, vmax=vmax)
    region_rect = plt.Rectangle((rx0, rt0), rx1 - rx0, rt1 - rt0, fill=False,
                                edgecolor='yellow', linewidth=2.0)
    crop_rect = plt.Rectangle((x0, t0), crop_nx, crop_nt, fill=False,
                              edgecolor='white', linewidth=2.4)
    axes[0].add_patch(region_rect)
    axes[0].add_patch(crop_rect)
    axes[0].set_title('(a) Full Marmousi AI model')
    axes[0].set_xlabel('Trace')
    axes[0].set_ylabel('Sample')
    plt.colorbar(im, ax=axes[0], label='AI (x10^4 Pa s/m)')

    region_ai = Z_full[rx0:rx1, rt0:rt1]
    im = axes[1].imshow(region_ai.T / 1e4, aspect='auto', cmap='turbo',
                        extent=[rx0, rx1 - 1, rt1 - 1, rt0],
                        vmin=vmin, vmax=vmax)
    rect = plt.Rectangle((x0, t0), crop_nx, crop_nt, fill=False,
                         edgecolor='white', linewidth=2.4)
    axes[1].add_patch(rect)
    axes[1].set_title('(b) Regional fault-zone zoom')
    axes[1].set_xlabel('Trace')
    axes[1].set_ylabel('Sample')
    plt.colorbar(im, ax=axes[1], label='AI (x10^4 Pa s/m)')

    crop_ai = Z_full[x0:x1, t0:t1]
    im = axes[2].imshow(crop_ai.T / 1e4, aspect='auto', cmap='turbo',
                        extent=[x0, x1 - 1, t1 - 1, t0],
                        vmin=vmin, vmax=vmax)
    axes[2].set_title('(c) 400 x 300 benchmark crop')
    axes[2].set_xlabel('Trace')
    axes[2].set_ylabel('Sample')
    plt.colorbar(im, ax=axes[2], label='AI (x10^4 Pa s/m)')

    plt.suptitle('Marmousi Crop Location and Faulted Benchmark Window',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('Marmousi_Crop_Location.png'), dpi=SAVE_DPI,
                bbox_inches='tight')
    plt.close(fig)


def figure1_model_and_data(Z_true, Z_init, S_clean_trim, S_noisy_trim,
                           r_true, wavelet, time_ms, x_coord):
    """Fig 1: True model, initial model, wavelet, seismic data, reflectivity."""
    if BENCHMARK == "marmousi-crop":
        figure_marmousi_model_and_data(Z_true, Z_init, S_clean_trim,
                                       S_noisy_trim, r_true, time_ms, x_coord)
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]

    # (a) Wavelet
    nw = len(wavelet)
    tw = np.arange(nw) * DT * 1000 - (nw // 2) * DT * 1000
    axes[0, 0].plot(tw, wavelet, 'b', lw=2)
    axes[0, 0].set_title(f'(a) Ricker Wavelet (f₀ = {F0:.0f} Hz)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', lw=0.5)

    # (b) True impedance
    im = axes[0, 1].imshow(Z_true.T / 1e4, aspect='auto', cmap='jet',
                            extent=ext, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('(b) True Impedance Model')
    axes[0, 1].set_xlabel('Trace')
    axes[0, 1].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[0, 1], label='AI (×10⁴ Pa·s/m)')

    # (c) Initial model (smoothed)
    im = axes[0, 2].imshow(Z_init.T / 1e4, aspect='auto', cmap='jet',
                            extent=ext, vmin=vmin, vmax=vmax)
    snr_init, rmse_init = compute_metrics(Z_init, Z_true)
    axes[0, 2].set_title(f'(c) Initial Model (smoothed)\n'
                          f'SNR={snr_init:.1f} dB, RMSE={rmse_init:.0f}')
    axes[0, 2].set_xlabel('Trace')
    axes[0, 2].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[0, 2], label='AI (×10⁴ Pa·s/m)')

    # (d) Clean seismic
    s_vmax = np.percentile(np.abs(S_clean_trim), 99)
    im = axes[1, 0].imshow(S_clean_trim.T, aspect='auto', cmap='gray',
                            extent=ext, vmin=-s_vmax, vmax=s_vmax)
    axes[1, 0].set_title('(d) Clean Seismic Data')
    axes[1, 0].set_xlabel('Trace')
    axes[1, 0].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[1, 0], label='Amplitude')

    # (e) Noisy seismic
    im = axes[1, 1].imshow(S_noisy_trim.T, aspect='auto', cmap='gray',
                            extent=ext, vmin=-s_vmax, vmax=s_vmax)
    axes[1, 1].set_title(f'(e) Noisy Seismic (SNR ≈ {NOISE_SNR:.0f})')
    axes[1, 1].set_xlabel('Trace')
    axes[1, 1].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[1, 1], label='Amplitude')

    # (f) True reflectivity
    r_vmax = np.percentile(np.abs(r_true), 99)
    im = axes[1, 2].imshow(r_true.T, aspect='auto', cmap='gray',
                            extent=ext, vmin=-r_vmax, vmax=r_vmax)
    axes[1, 2].set_title('(f) True Reflectivity r = (Z₊−Z₋)/(Z₊+Z₋)')
    axes[1, 2].set_xlabel('Trace')
    axes[1, 2].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[1, 2], label='RC')

    plt.suptitle('Fig 1 – Synthetic Model, Wavelet & Seismic Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig1_Model_Data.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure_marmousi_model_and_data(Z_true, Z_init, S_clean_trim, S_noisy_trim,
                                   r_true, time_ms, x_coord):
    """Marmousi model/data summary without repeating the shared wavelet panel."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))

    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]

    im = axes[0].imshow(Z_true.T / 1e4, aspect='auto', cmap='jet',
                        extent=ext, vmin=vmin, vmax=vmax)
    axes[0].set_title('(a) True AI crop')
    axes[0].set_xlabel('Trace')
    axes[0].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[0], label='AI (x10^4 Pa s/m)')

    im = axes[1].imshow(Z_init.T / 1e4, aspect='auto', cmap='jet',
                        extent=ext, vmin=vmin, vmax=vmax)
    snr_init, rmse_init = compute_metrics(Z_init, Z_true)
    axes[1].set_title(f'(b) Initial model\nSNR={snr_init:.1f} dB, RMSE={rmse_init:.0f}')
    axes[1].set_xlabel('Trace')
    axes[1].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[1], label='AI (x10^4 Pa s/m)')

    s_vmax = np.percentile(np.abs(S_clean_trim), 99)
    im = axes[2].imshow(S_clean_trim.T, aspect='auto', cmap='gray',
                        extent=ext, vmin=-s_vmax, vmax=s_vmax)
    axes[2].set_title('(c) Clean seismic')
    axes[2].set_xlabel('Trace')
    axes[2].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[2], label='Amplitude')

    im = axes[3].imshow(S_noisy_trim.T, aspect='auto', cmap='gray',
                        extent=ext, vmin=-s_vmax, vmax=s_vmax)
    axes[3].set_title(f'(d) Noisy seismic (SNR={NOISE_SNR:.0f})')
    axes[3].set_xlabel('Trace')
    axes[3].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[3], label='Amplitude')

    r_vmax = np.percentile(np.abs(r_true), 99)
    im = axes[4].imshow(r_true.T, aspect='auto', cmap='gray',
                        extent=ext, vmin=-r_vmax, vmax=r_vmax)
    axes[4].set_title('(e) True reflectivity')
    axes[4].set_xlabel('Trace')
    axes[4].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[4], label='RC')

    plt.suptitle('Marmousi Faulted Crop: Model and Simulated Seismic Data',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig1_Model_Data.png'), dpi=SAVE_DPI,
                bbox_inches='tight')
    plt.close(fig)


def figure2_clean_inversion(Z_true, Z_init, Z_rw, Z_unet,
                             time_ms, x_coord):
    """Fig 2: Clean data inversion – True, Initial, RW-L1, U-Net + errors."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]
    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4

    snr_init, rmse_init = compute_metrics(Z_init, Z_true)
    snr_rw, rmse_rw     = compute_metrics(Z_rw, Z_true)
    snr_un, rmse_un     = compute_metrics(Z_unet, Z_true)

    panels_top = [
        (Z_true,  '(a) True Model'),
        (Z_init,  f'(b) Initial\nSNR={snr_init:.1f}, RMSE={rmse_init:.0f}'),
        (Z_rw,    f'(c) RW-L1\nSNR={snr_rw:.1f}, RMSE={rmse_rw:.0f}'),
        (Z_unet,  f'(d) U-Net\nSNR={snr_un:.1f}, RMSE={rmse_un:.0f}'),
    ]
    for j, (d, t) in enumerate(panels_top):
        im = axes[0, j].imshow(d.T / 1e4, aspect='auto', cmap='jet',
                                extent=ext, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(t)
        axes[0, j].set_xlabel('Trace')
        axes[0, j].set_ylabel('Time (ms)')
        plt.colorbar(im, ax=axes[0, j])

    # Error panels
    err_init = np.abs(Z_init - Z_true)
    err_rw   = np.abs(Z_rw - Z_true)
    err_unet = np.abs(Z_unet - Z_true)
    err_vmax = max(np.percentile(err_init, 99),
                   np.percentile(err_rw, 99),
                   np.percentile(err_unet, 99)) / 1e4

    # (e) True reflectivity in error row
    r_true = np.zeros((Z_true.shape[0], Z_true.shape[1] - 1))
    for i in range(Z_true.shape[0]):
        Z = Z_true[i, :]
        r_true[i, :] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
    r_vmax = np.percentile(np.abs(r_true), 99)
    im = axes[1, 0].imshow(r_true.T, aspect='auto', cmap='gray',
                            extent=ext, vmin=-r_vmax, vmax=r_vmax)
    axes[1, 0].set_title('(e) True Reflectivity')
    axes[1, 0].set_xlabel('Trace')
    axes[1, 0].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[1, 0])

    panels_err = [
        (err_init, '(f) Error – Initial'),
        (err_rw,   '(g) Error – RW-L1'),
        (err_unet, '(h) Error – U-Net'),
    ]
    for j, (d, t) in enumerate(panels_err):
        im = axes[1, j + 1].imshow(d.T / 1e4, aspect='auto', cmap='hot',
                                    extent=ext, vmin=0, vmax=err_vmax)
        axes[1, j + 1].set_title(t)
        axes[1, j + 1].set_xlabel('Trace')
        axes[1, j + 1].set_ylabel('Time (ms)')
        plt.colorbar(im, ax=axes[1, j + 1])

    plt.suptitle('Fig 2 – Clean Data Inversion: True → Initial → RW-L1 → U-Net',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig2_Clean_Inversion.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure3_noisy_inversion(Z_true, Z_init, Z_rw, Z_unet,
                              S_noisy_trim, time_ms, x_coord):
    """Fig 3: Noisy data inversion – same layout as Fig 2."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]
    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4

    snr_init, rmse_init = compute_metrics(Z_init, Z_true)
    snr_rw, rmse_rw     = compute_metrics(Z_rw, Z_true)
    snr_un, rmse_un     = compute_metrics(Z_unet, Z_true)

    panels_top = [
        (Z_true,  '(a) True Model'),
        (Z_init,  f'(b) Initial\nSNR={snr_init:.1f}, RMSE={rmse_init:.0f}'),
        (Z_rw,    f'(c) RW-L1 (noisy)\nSNR={snr_rw:.1f}, RMSE={rmse_rw:.0f}'),
        (Z_unet,  f'(d) U-Net (noisy)\nSNR={snr_un:.1f}, RMSE={rmse_un:.0f}'),
    ]
    for j, (d, t) in enumerate(panels_top):
        im = axes[0, j].imshow(d.T / 1e4, aspect='auto', cmap='jet',
                                extent=ext, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(t)
        axes[0, j].set_xlabel('Trace')
        axes[0, j].set_ylabel('Time (ms)')
        plt.colorbar(im, ax=axes[0, j])

    err_init = np.abs(Z_init - Z_true)
    err_rw   = np.abs(Z_rw - Z_true)
    err_unet = np.abs(Z_unet - Z_true)
    err_vmax = max(np.percentile(err_init, 99),
                   np.percentile(err_rw, 99),
                   np.percentile(err_unet, 99)) / 1e4

    # (e) Noisy seismic input
    s_vmax = np.percentile(np.abs(S_noisy_trim), 99)
    im = axes[1, 0].imshow(S_noisy_trim.T, aspect='auto', cmap='gray',
                            extent=ext, vmin=-s_vmax, vmax=s_vmax)
    axes[1, 0].set_title(f'(e) Noisy Seismic Input (SNR ≈ {NOISE_SNR:.0f})')
    axes[1, 0].set_xlabel('Trace')
    axes[1, 0].set_ylabel('Time (ms)')
    plt.colorbar(im, ax=axes[1, 0])

    panels_err = [
        (err_init, '(f) Error – Initial'),
        (err_rw,   '(g) Error – RW-L1'),
        (err_unet, '(h) Error – U-Net'),
    ]
    for j, (d, t) in enumerate(panels_err):
        im = axes[1, j + 1].imshow(d.T / 1e4, aspect='auto', cmap='hot',
                                    extent=ext, vmin=0, vmax=err_vmax)
        axes[1, j + 1].set_title(t)
        axes[1, j + 1].set_xlabel('Trace')
        axes[1, j + 1].set_ylabel('Time (ms)')
        plt.colorbar(im, ax=axes[1, j + 1])

    plt.suptitle('Fig 3 – Noisy Data Inversion: True → Initial → RW-L1 → U-Net',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig3_Noisy_Inversion.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure4_single_traces(Z_true, Z_init, Z_rw_clean, Z_rw_noisy,
                           Z_unet_clean, Z_unet_noisy, time_ms):
    """Fig 4: Single trace comparison (3 traces, clean vs noisy)."""
    trace_indices = [NX // 4, NX // 2, 3 * NX // 4]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, idx in enumerate(trace_indices):
        # Clean
        ax = axes[0, col]
        ax.plot(time_ms, Z_true[idx] / 1e4, 'k', lw=2.5, label='True')
        ax.plot(time_ms, Z_init[idx] / 1e4, 'g--', lw=1.8, label='Initial')
        ax.plot(time_ms, Z_rw_clean[idx] / 1e4, 'b--', lw=1.5, label='RW-L1')
        ax.plot(time_ms, Z_unet_clean[idx] / 1e4, 'r', lw=1.5, label='U-Net')
        ax.set_title(f'Clean – Trace {idx + 1}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('AI (×10⁴ Pa·s/m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Noisy
        ax = axes[1, col]
        ax.plot(time_ms, Z_true[idx] / 1e4, 'k', lw=2.5, label='True')
        ax.plot(time_ms, Z_init[idx] / 1e4, 'g--', lw=1.8, label='Initial')
        ax.plot(time_ms, Z_rw_noisy[idx] / 1e4, 'b--', lw=1.5, label='RW-L1')
        ax.plot(time_ms, Z_unet_noisy[idx] / 1e4, 'r', lw=1.5, label='U-Net')
        ax.set_title(f'Noisy – Trace {idx + 1}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('AI (×10⁴ Pa·s/m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Fig 4 – Single Trace Comparison (Clean vs Noisy)', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig4_Traces.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure5_reflectivity(Z_true, Z_rw_clean, Z_rw_noisy,
                          Z_unet_clean, Z_unet_noisy, time_ms):
    """Fig 5: Reflection coefficient recovery (single traces)."""
    trace_indices = [NX // 4, NX // 2, 3 * NX // 4]
    rc_axis = time_ms[:-1] + 0.5 * DT * 1000  # midpoints

    def exact_rc(Z):
        return (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, idx in enumerate(trace_indices):
        rc_true = exact_rc(Z_true[idx])

        # Clean
        ax = axes[0, col]
        ax.stem(rc_axis, rc_true, linefmt='k-', markerfmt='ko', basefmt='k-',
                label='True')
        ax.plot(rc_axis, 0.5 * np.diff(np.log(Z_rw_clean[idx])),
                'b--', lw=1.5, label='RW-L1')
        ax.plot(rc_axis, exact_rc(Z_unet_clean[idx]),
                'r', lw=1.5, label='U-Net (exact)')
        ax.set_title(f'Clean – Trace {idx + 1}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('RC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Noisy
        ax = axes[1, col]
        ax.stem(rc_axis, rc_true, linefmt='k-', markerfmt='ko', basefmt='k-',
                label='True')
        ax.plot(rc_axis, 0.5 * np.diff(np.log(Z_rw_noisy[idx])),
                'b--', lw=1.5, label='RW-L1')
        ax.plot(rc_axis, exact_rc(Z_unet_noisy[idx]),
                'r', lw=1.5, label='U-Net (exact)')
        ax.set_title(f'Noisy – Trace {idx + 1}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('RC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Fig 5 – Reflection Coefficient Recovery', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig5_Reflectivity.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure6_data_fit(S_clean, S_noisy, Z_rw_clean, Z_rw_noisy,
                      Z_unet_clean, Z_unet_noisy, wavelet, time_ms):
    """Fig 6: Forward model fit to observed seismic (single traces)."""
    trace_idx = NX // 2
    nw = len(wavelet)
    W_mat = convolution_matrix(wavelet, NT - 1)

    def forward_model(Z):
        rc = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
        return W_mat @ rc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trim observed seismic to match forward model output
    half_pad = (nw - 1) // 2
    s_len = NT + nw - 2
    s_axis = np.arange(s_len) * DT * 1000

    # Clean
    ax = axes[0]
    ax.plot(s_axis, S_clean[trace_idx, :s_len], 'k', lw=2, label='Observed')
    ax.plot(s_axis, forward_model(Z_rw_clean[trace_idx]),
            'b--', lw=1.5, label='RW-L1 fit')
    ax.plot(s_axis, forward_model(Z_unet_clean[trace_idx]),
            'r', lw=1.5, label='U-Net fit')
    ax.set_title(f'(a) Seismic Data Fit (Clean) – Trace {trace_idx + 1}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Noisy
    ax = axes[1]
    ax.plot(s_axis, S_noisy[trace_idx, :s_len], 'k', lw=2, label='Observed')
    ax.plot(s_axis, forward_model(Z_rw_noisy[trace_idx]),
            'b--', lw=1.5, label='RW-L1 fit')
    ax.plot(s_axis, forward_model(Z_unet_noisy[trace_idx]),
            'r', lw=1.5, label='U-Net fit')
    ax.set_title(f'(b) Seismic Data Fit (Noisy) – Trace {trace_idx + 1}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig 6 – Forward Model Fit to Observed Seismic', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig6_DataFit.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure7_convergence(loss_clean, loss_noisy):
    """Fig 7: U-Net training loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(loss_clean, 'r', lw=1.5)
    ax1.set_title('(a) U-Net Training Loss (Clean)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(loss_noisy, 'r', lw=1.5)
    ax2.set_title('(b) U-Net Training Loss (Noisy)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Fig 7 – Convergence Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig7_Convergence.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure8_metrics(Z_true, Z_init, Z_rw_clean, Z_rw_noisy,
                     Z_rescnn_clean, Z_rescnn_noisy,
                     Z_unet_clean, Z_unet_noisy):
    """Fig 8: Quantitative comparison bar chart for ADMM, ResCNN, and U-Net."""
    snr_init, rmse_init = compute_metrics(Z_init, Z_true)
    snr_rw_c, rmse_rw_c = compute_metrics(Z_rw_clean, Z_true)
    snr_rw_n, rmse_rw_n = compute_metrics(Z_rw_noisy, Z_true)
    snr_rc_c, rmse_rc_c = compute_metrics(Z_rescnn_clean, Z_true)
    snr_rc_n, rmse_rc_n = compute_metrics(Z_rescnn_noisy, Z_true)
    snr_un_c, rmse_un_c = compute_metrics(Z_unet_clean, Z_true)
    snr_un_n, rmse_un_n = compute_metrics(Z_unet_noisy, Z_true)

    methods = ['Initial', 'RW-L1\n(clean)', 'ResCNN\n(clean)', 'U-Net\n(clean)',
               'RW-L1\n(noisy)', 'ResCNN\n(noisy)', 'U-Net\n(noisy)']
    snrs = [snr_init, snr_rw_c, snr_rc_c, snr_un_c, snr_rw_n, snr_rc_n, snr_un_n]
    rmses = [rmse_init, rmse_rw_c, rmse_rc_c, rmse_un_c, rmse_rw_n, rmse_rc_n, rmse_un_n]
    colors = ['#4CAF50', '#2196F3', '#9C27B0', '#F44336',
              '#64B5F6', '#CE93D8', '#EF9A9A']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))

    bars = ax1.bar(methods, snrs, color=colors, edgecolor='k', linewidth=0.8)
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('(a) Signal-to-Noise Ratio')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', labelrotation=20)
    for bar, val in zip(bars, snrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    bars = ax2.bar(methods, rmses, color=colors, edgecolor='k', linewidth=0.8)
    ax2.set_ylabel('RMSE (Pa.s/m)')
    ax2.set_title('(b) Root Mean Squared Error')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', labelrotation=20)
    for bar, val in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    plt.suptitle('Fig 8 - Quantitative Comparison: He RW-L1 ADMM vs ResCNN vs U-Net',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig8_Metrics.png'), dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure9_architecture_comparison(Z_true, Z_rw_clean, Z_rw_noisy,
                                     Z_rescnn_clean, Z_rescnn_noisy,
                                     Z_unet_clean, Z_unet_noisy,
                                     time_ms, x_coord):
    """Fig 9: Visual architecture comparison on clean and noisy inversions."""
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]
    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4

    panels = [
        (Z_true, 'True model'),
        (Z_rw_clean, 'He RW-L1 ADMM clean'),
        (Z_rescnn_clean, 'Hybrid ResCNN clean'),
        (Z_unet_clean, 'Hybrid U-Net clean'),
        (Z_true, 'True model'),
        (Z_rw_noisy, 'He RW-L1 ADMM noisy'),
        (Z_rescnn_noisy, 'Hybrid ResCNN noisy'),
        (Z_unet_noisy, 'Hybrid U-Net noisy'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for ax, (data, title) in zip(axes.ravel(), panels):
        snr, rmse = compute_metrics(data, Z_true)
        im = ax.imshow(data.T / 1e4, aspect='auto', cmap='jet',
                       extent=ext, vmin=vmin, vmax=vmax)
        if 'True' in title:
            ax.set_title(title)
        else:
            ax.set_title(f'{title}\nSNR={snr:.2f} dB, RMSE={rmse:.1f}')
        ax.set_xlabel('Trace')
        ax.set_ylabel('Time (ms)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Fig 9 - Architecture Comparison on Complex Fault/Lens Model', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig9_Architecture_Comparison.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure10_attention_comparison(Z_true, Z_init, Z_rw_clean, Z_rw_noisy,
                                  Z_rescnn_clean, Z_rescnn_noisy,
                                  Z_unet_clean, Z_unet_noisy,
                                  Z_attn_clean, Z_attn_noisy,
                                  time_ms, x_coord):
    """Server figure: ADMM vs ResCNN vs U-Net vs Attention ResUNet."""
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]
    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4
    rows = [
        ('Clean', [Z_true, Z_init, Z_rw_clean, Z_rescnn_clean, Z_unet_clean, Z_attn_clean]),
        ('Noisy', [Z_true, Z_init, Z_rw_noisy, Z_rescnn_noisy, Z_unet_noisy, Z_attn_noisy]),
    ]
    labels = ['True', 'Initial', 'He RW-L1 ADMM', 'ResCNN', 'U-Net', 'Attention ResUNet']
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    for i, (row, data_list) in enumerate(rows):
        for j, (label, data) in enumerate(zip(labels, data_list)):
            ax = axes[i, j]
            im = ax.imshow(data.T / 1e4, aspect='auto', cmap='jet',
                           extent=ext, vmin=vmin, vmax=vmax)
            if label == 'True':
                title = f'{row}: {label}'
            else:
                snr, rmse = compute_metrics(data, Z_true)
                title = f'{row}: {label}\nSNR={snr:.2f} dB, RMSE={rmse:.1f}'
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Trace')
            ax.set_ylabel('Time (ms)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('Fig 10 - Server Advanced Comparison: ADMM-Guided Neural Refinement', fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig10_Server_AttentionResUNet_Comparison.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


def figure16_structural_zoom_profiles(Z_true, Z_rw_noisy, Z_unet_noisy,
                                       time_ms, x_coord):
    """Structural diagnostic: fault and thin-bed zooms plus profile extracts."""
    fault_x = slice(int(0.41 * NX), int(0.64 * NX))
    fault_t = slice(int(0.42 * NT), int(0.78 * NT))
    thin_x = slice(int(0.10 * NX), int(0.45 * NX))
    thin_t = slice(int(0.45 * NT), int(0.68 * NT))
    fault_profile_t = int(0.62 * NT)
    thin_trace_x = int(0.30 * NX)

    vmin = Z_true.min() / 1e4
    vmax = Z_true.max() / 1e4

    def ext(xs, ts):
        return [x_coord[xs.start], x_coord[xs.stop - 1],
                time_ms[ts.stop - 1], time_ms[ts.start]]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])

    image_panels = [
        (Z_true, fault_x, fault_t, '(a) True fault window'),
        (Z_rw_noisy, fault_x, fault_t, '(b) ADMM noisy fault window'),
        (Z_unet_noisy, fault_x, fault_t, '(c) U-Net noisy fault window'),
        (np.abs(Z_unet_noisy - Z_rw_noisy), fault_x, fault_t,
         '(d) |U-Net - ADMM| near fault'),
    ]

    for i, (data, xs, ts, title) in enumerate(image_panels):
        ax = fig.add_subplot(gs[0, i])
        if i == 3:
            vmax_err = np.percentile(data[xs, ts], 99) / 1e4
            im = ax.imshow(data[xs, ts].T / 1e4, aspect='auto', cmap='magma',
                           extent=ext(xs, ts), vmin=0, vmax=vmax_err)
        else:
            im = ax.imshow(data[xs, ts].T / 1e4, aspect='auto', cmap='jet',
                           extent=ext(xs, ts), vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Trace')
        ax.set_ylabel('Time (ms)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='AI (x10^4 Pa.s/m)')

    ax = fig.add_subplot(gs[1, 0:2])
    xvals = x_coord[fault_x]
    t_idx = fault_profile_t
    ax.plot(xvals, Z_true[fault_x, t_idx] / 1e4, 'k', lw=2.3, label='True')
    ax.plot(xvals, Z_rw_noisy[fault_x, t_idx] / 1e4, 'b--', lw=1.6,
            label='ADMM noisy')
    ax.plot(xvals, Z_unet_noisy[fault_x, t_idx] / 1e4, 'r', lw=1.8,
            label='U-Net noisy')
    ax.set_title(f'(e) Lateral profile across fault at {time_ms[t_idx]:.0f} ms')
    ax.set_xlabel('Trace')
    ax.set_ylabel('AI (x10^4 Pa.s/m)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[1, 2:4])
    tvals = time_ms[thin_t]
    x_idx = thin_trace_x
    ax.plot(tvals, Z_true[x_idx, thin_t] / 1e4, 'k', lw=2.3, label='True')
    ax.plot(tvals, Z_rw_noisy[x_idx, thin_t] / 1e4, 'b--', lw=1.6,
            label='ADMM noisy')
    ax.plot(tvals, Z_unet_noisy[x_idx, thin_t] / 1e4, 'r', lw=1.8,
            label='U-Net noisy')
    ax.set_title(f'(f) Vertical trace through thin beds at trace {x_idx + 1}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('AI (x10^4 Pa.s/m)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle('Fig 16 - Structural Diagnostics for Fault and Thin Beds',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(result_path('2D_Fig16_Structural_Zoom_Profiles.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":

    args = parse_args()
    configure_runtime(args)
    set_seed(SEED)
    ensure_results_dir()

    print(f"Using device: {DEVICE}")
    print(f"Run tag: {RUN_TAG}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Model geometry: NX={NX}, NT={NT}, DT={DT}")
    print(f"Patch size: {PATCH_NX}x{PATCH_NT}, stride: {STRIDE_NX}x{STRIDE_NT}")
    print(f"Training: epochs={EPOCHS}, batch={BATCH_SIZE}, base_ch={UNET_BASE_CH}, attention={RUN_ATTENTION}")

    if args.benchmark == "marmousi-crop":
        print("\n[1/6] Loading cropped Marmousi acoustic-impedance benchmark...")
        Z_true, x_coord, dt_used, marmousi_crop = load_marmousi_crop(args, NX, NT)
        print(f"  Marmousi crop: traces {marmousi_crop[0]}:{marmousi_crop[1]}, "
              f"samples {marmousi_crop[2]}:{marmousi_crop[3]}, dt={dt_used*1000:.3f} ms")
    else:
        print("\n[1/6] Building complex 2D fault/lens model...")
        Z_true, x_coord = build_layered_model(NX, NT, DT, LAYER_TOPS)
        dt_used = DT
        marmousi_crop = None

    L_true = np.log(Z_true)
    time_ms = np.arange(NT) * dt_used * 1000

    wavelet = ricker_wavelet(F0, dt_used)
    nw = len(wavelet)
    print(f"  Wavelet: {nw} samples, f0={F0} Hz")

    r_true = np.zeros((NX, NT - 1))
    S_clean = np.zeros((NX, NT + nw - 2))
    W_mat = convolution_matrix(wavelet, NT - 1)

    for i in range(NX):
        Z = Z_true[i, :]
        r_true[i, :] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
        S_clean[i, :] = W_mat @ r_true[i, :]

    S_noisy = add_noise(S_clean, NOISE_SNR)
    print(f"  Noise added: target SNR = {NOISE_SNR}")

    Z_init = gaussian_filter(Z_true, sigma=(8.0, SMOOTH_SIGMA))
    L_init = np.log(Z_init)

    snr_init, rmse_init = compute_metrics(Z_init, Z_true)
    print(f"  Initial model: SNR = {snr_init:.2f} dB, RMSE = {rmse_init:.2f}")

    half_pad = (nw - 1) // 2
    S_clean_trim = S_clean[:, half_pad:half_pad + NT]
    S_noisy_trim = S_noisy[:, half_pad:half_pad + NT]

    print("\n" + "=" * 60)
    print("  PART 1: Reweighted L1 sparse inversion (ADMM)")
    print("=" * 60)

    Z_rw_clean = np.zeros_like(Z_true)
    Z_rw_noisy = np.zeros_like(Z_true)

    print("\n  Inverting clean data...")
    t0 = time.time()
    for i in range(NX):
        if i % 100 == 0:
            print(f"    trace {i}/{NX}, elapsed {time.time()-t0:.1f}s")
        Z_rw_clean[i, :], _ = admm_l1_single_trace(
            S_clean[i, :], wavelet, L_init[i, :],
            MU, ALPHA, LAMBDA_, EPSILON_CL, MAX_ITER_CL, TOL_CL,
            reweight=True)
    rw_clean_time = time.time() - t0

    print("\n  Inverting noisy data...")
    t0 = time.time()
    for i in range(NX):
        if i % 100 == 0:
            print(f"    trace {i}/{NX}, elapsed {time.time()-t0:.1f}s")
        Z_rw_noisy[i, :], _ = admm_l1_single_trace(
            S_noisy[i, :], wavelet, L_init[i, :],
            MU, ALPHA, LAMBDA_, EPSILON_CL, MAX_ITER_CL, TOL_CL,
            reweight=True)
    rw_noisy_time = time.time() - t0

    snr_rw_c, rmse_rw_c = compute_metrics(Z_rw_clean, Z_true)
    snr_rw_n, rmse_rw_n = compute_metrics(Z_rw_noisy, Z_true)

    print(f"\n  RW-L1 (clean): SNR = {snr_rw_c:.2f} dB | RMSE = {rmse_rw_c:.2f} | Time = {rw_clean_time:.1f}s")
    print(f"  RW-L1 (noisy): SNR = {snr_rw_n:.2f} dB | RMSE = {rmse_rw_n:.2f} | Time = {rw_noisy_time:.1f}s")

    print("\n" + "=" * 60)
    print("  PART 2: Physics-Informed neural inversions")
    print("=" * 60)

    corners = compute_patch_corners(NX, NT, PATCH_NX, PATCH_NT,
                                    STRIDE_NX, STRIDE_NT)
    print(f"\n  Patches: {len(corners)} patches at {PATCH_NX}x{PATCH_NT}")

    if USE_HE_ADMM_AS_NEURAL_INITIAL:
        print("  Neural prior: He et al. reweighted L1 ADMM output")
        L_neural_init_clean = np.log(Z_rw_clean + 1e-12)
        L_neural_init_noisy = np.log(Z_rw_noisy + 1e-12)
    else:
        print("  Neural prior: smoothed initial model")
        L_neural_init_clean = L_init
        L_neural_init_noisy = L_init

    S_clean_patches = extract_patches_at_corners(S_clean_trim, corners,
                                                 PATCH_NX, PATCH_NT)
    S_noisy_patches = extract_patches_at_corners(S_noisy_trim, corners,
                                                 PATCH_NX, PATCH_NT)
    L_init_patches_clean = extract_patches_at_corners(L_neural_init_clean, corners,
                                                      PATCH_NX, PATCH_NT)
    L_init_patches_noisy = extract_patches_at_corners(L_neural_init_noisy, corners,
                                                      PATCH_NX, PATCH_NT)
    L_true_patches = extract_patches_at_corners(L_true, corners, PATCH_NX, PATCH_NT)
    if args.no_supervised_benchmark:
        L_true_patches_for_loss = None
        print("  Neural loss: physics + He-style RW-L1 + prior/lateral terms (no true-impedance supervision)")
    else:
        L_true_patches_for_loss = L_true_patches
        if args.benchmark == "marmousi-crop":
            print("  Neural loss: supervised Marmousi benchmark loss enabled (known true AI)")
        else:
            print("  Neural loss: physics + He-style RW-L1 + supervised impedance/gradient sharpening")

    print("\n  Training U-Net on CLEAN data...")
    t0 = time.time()
    net_clean, loss_clean = train_unet_2d(
        S_clean_patches, L_init_patches_clean, wavelet,
        PATCH_NX, PATCH_NT, label="unet-clean",
        checkpoint_path=UNET_CLEAN_CKPT,
        L_true_patches=L_true_patches_for_loss,
        transfer_checkpoint=transfer_checkpoint_path("U-Net", "clean", args))
    unet_clean_time = time.time() - t0
    Z_unet_clean = predict_full_section(
        net_clean, S_clean_trim, L_neural_init_clean,
        corners, PATCH_NX, PATCH_NT, NX, NT)

    print("\n  Training U-Net on NOISY data...")
    t0 = time.time()
    net_noisy, loss_noisy = train_unet_2d(
        S_noisy_patches, L_init_patches_noisy, wavelet,
        PATCH_NX, PATCH_NT, label="unet-noisy",
        checkpoint_path=UNET_NOISY_CKPT,
        L_true_patches=L_true_patches_for_loss,
        transfer_checkpoint=transfer_checkpoint_path("U-Net", "noisy", args))
    unet_noisy_time = time.time() - t0
    Z_unet_noisy = predict_full_section(
        net_noisy, S_noisy_trim, L_neural_init_noisy,
        corners, PATCH_NX, PATCH_NT, NX, NT)

    print("\n  Training ResCNN on CLEAN data...")
    t0 = time.time()
    net_rescnn_clean, loss_rescnn_clean = train_model_2d(
        S_clean_patches, L_init_patches_clean, wavelet,
        PATCH_NX, PATCH_NT, label="rescnn-clean",
        checkpoint_path=RESCNN_CLEAN_CKPT,
        L_true_patches=L_true_patches_for_loss,
        model_class=ResCNN2D,
        model_name="ResCNN",
        transfer_checkpoint=transfer_checkpoint_path("ResCNN", "clean", args))
    rescnn_clean_time = time.time() - t0
    Z_rescnn_clean = predict_full_section(
        net_rescnn_clean, S_clean_trim, L_neural_init_clean,
        corners, PATCH_NX, PATCH_NT, NX, NT)

    print("\n  Training ResCNN on NOISY data...")
    t0 = time.time()
    net_rescnn_noisy, loss_rescnn_noisy = train_model_2d(
        S_noisy_patches, L_init_patches_noisy, wavelet,
        PATCH_NX, PATCH_NT, label="rescnn-noisy",
        checkpoint_path=RESCNN_NOISY_CKPT,
        L_true_patches=L_true_patches_for_loss,
        model_class=ResCNN2D,
        model_name="ResCNN",
        transfer_checkpoint=transfer_checkpoint_path("ResCNN", "noisy", args))
    rescnn_noisy_time = time.time() - t0
    Z_rescnn_noisy = predict_full_section(
        net_rescnn_noisy, S_noisy_trim, L_neural_init_noisy,
        corners, PATCH_NX, PATCH_NT, NX, NT)

    Z_attn_clean = None
    Z_attn_noisy = None
    loss_attn_clean = []
    loss_attn_noisy = []
    attn_clean_time = 0.0
    attn_noisy_time = 0.0
    if RUN_ATTENTION:
        print("\n  Training Attention ResUNet on CLEAN data...")
        t0 = time.time()
        net_attn_clean, loss_attn_clean = train_model_2d(
            S_clean_patches, L_init_patches_clean, wavelet,
            PATCH_NX, PATCH_NT, label="attn-clean",
            checkpoint_path=ATTN_CLEAN_CKPT,
            L_true_patches=L_true_patches_for_loss,
            model_class=AttentionResUNet2D,
            model_name="AttentionResUNet",
            transfer_checkpoint=transfer_checkpoint_path("AttentionResUNet", "clean", args))
        attn_clean_time = time.time() - t0
        Z_attn_clean = predict_full_section(
            net_attn_clean, S_clean_trim, L_neural_init_clean,
            corners, PATCH_NX, PATCH_NT, NX, NT)

        print("\n  Training Attention ResUNet on NOISY data...")
        t0 = time.time()
        net_attn_noisy, loss_attn_noisy = train_model_2d(
            S_noisy_patches, L_init_patches_noisy, wavelet,
            PATCH_NX, PATCH_NT, label="attn-noisy",
            checkpoint_path=ATTN_NOISY_CKPT,
            L_true_patches=L_true_patches_for_loss,
            model_class=AttentionResUNet2D,
            model_name="AttentionResUNet",
            transfer_checkpoint=transfer_checkpoint_path("AttentionResUNet", "noisy", args))
        attn_noisy_time = time.time() - t0
        Z_attn_noisy = predict_full_section(
            net_attn_noisy, S_noisy_trim, L_neural_init_noisy,
            corners, PATCH_NX, PATCH_NT, NX, NT)

    snr_un_c, rmse_un_c = compute_metrics(Z_unet_clean, Z_true)
    snr_un_n, rmse_un_n = compute_metrics(Z_unet_noisy, Z_true)
    snr_rc_c, rmse_rc_c = compute_metrics(Z_rescnn_clean, Z_true)
    snr_rc_n, rmse_rc_n = compute_metrics(Z_rescnn_noisy, Z_true)

    print(f"\n  ResCNN (clean): SNR = {snr_rc_c:.2f} dB | RMSE = {rmse_rc_c:.2f} | Time = {rescnn_clean_time:.1f}s")
    print(f"  U-Net  (clean): SNR = {snr_un_c:.2f} dB | RMSE = {rmse_un_c:.2f} | Time = {unet_clean_time:.1f}s")
    print(f"  ResCNN (noisy): SNR = {snr_rc_n:.2f} dB | RMSE = {rmse_rc_n:.2f} | Time = {rescnn_noisy_time:.1f}s")
    print(f"  U-Net  (noisy): SNR = {snr_un_n:.2f} dB | RMSE = {rmse_un_n:.2f} | Time = {unet_noisy_time:.1f}s")
    if RUN_ATTENTION:
        snr_att_c, rmse_att_c = compute_metrics(Z_attn_clean, Z_true)
        snr_att_n, rmse_att_n = compute_metrics(Z_attn_noisy, Z_true)
        print(f"  Attention ResUNet (clean): SNR = {snr_att_c:.2f} dB | RMSE = {rmse_att_c:.2f} | Time = {attn_clean_time:.1f}s")
        print(f"  Attention ResUNet (noisy): SNR = {snr_att_n:.2f} dB | RMSE = {rmse_att_n:.2f} | Time = {attn_noisy_time:.1f}s")

    print("\n" + "=" * 60)
    print("  PART 3: Generating comparison figures")
    print("=" * 60)

    figure1_model_and_data(Z_true, Z_init, S_clean_trim, S_noisy_trim,
                           r_true, wavelet, time_ms, x_coord)
    figure2_clean_inversion(Z_true, Z_init, Z_rw_clean, Z_unet_clean,
                            time_ms, x_coord)
    figure3_noisy_inversion(Z_true, Z_init, Z_rw_noisy, Z_unet_noisy,
                            S_noisy_trim, time_ms, x_coord)
    figure4_single_traces(Z_true, Z_init, Z_rw_clean, Z_rw_noisy,
                          Z_unet_clean, Z_unet_noisy, time_ms)
    figure5_reflectivity(Z_true, Z_rw_clean, Z_rw_noisy,
                         Z_unet_clean, Z_unet_noisy, time_ms)
    figure6_data_fit(S_clean, S_noisy, Z_rw_clean, Z_rw_noisy,
                     Z_unet_clean, Z_unet_noisy, wavelet, time_ms)
    figure7_convergence(loss_clean, loss_noisy)
    figure8_metrics(Z_true, Z_init, Z_rw_clean, Z_rw_noisy,
                    Z_rescnn_clean, Z_rescnn_noisy,
                    Z_unet_clean, Z_unet_noisy)
    figure9_architecture_comparison(
        Z_true, Z_rw_clean, Z_rw_noisy,
        Z_rescnn_clean, Z_rescnn_noisy,
        Z_unet_clean, Z_unet_noisy, time_ms, x_coord)
    if RUN_ATTENTION:
        figure10_attention_comparison(
            Z_true, Z_init, Z_rw_clean, Z_rw_noisy,
            Z_rescnn_clean, Z_rescnn_noisy,
            Z_unet_clean, Z_unet_noisy,
            Z_attn_clean, Z_attn_noisy, time_ms, x_coord)
    figure16_structural_zoom_profiles(
        Z_true, Z_rw_noisy, Z_unet_noisy, time_ms, x_coord)

    result_payload = dict(
             Z_true=Z_true, Z_init=Z_init,
             Z_rw_clean=Z_rw_clean, Z_rw_noisy=Z_rw_noisy,
             Z_rescnn_clean=Z_rescnn_clean, Z_rescnn_noisy=Z_rescnn_noisy,
             Z_unet_clean=Z_unet_clean, Z_unet_noisy=Z_unet_noisy,
             S_clean=S_clean, S_noisy=S_noisy, S_clean_trim=S_clean_trim,
             S_noisy_trim=S_noisy_trim, wavelet=wavelet,
             L_neural_init_clean=L_neural_init_clean,
             L_neural_init_noisy=L_neural_init_noisy,
             model_type=('marmousi_crop' if args.benchmark == "marmousi-crop"
                         else 'complex_fault_oval_body'),
             benchmark=args.benchmark,
             dt=dt_used,
             preset=args.preset,
             run_tag=RUN_TAG,
             loss_clean=np.asarray(loss_clean), loss_noisy=np.asarray(loss_noisy),
             loss_rescnn_clean=np.asarray(loss_rescnn_clean),
             loss_rescnn_noisy=np.asarray(loss_rescnn_noisy))
    if RUN_ATTENTION:
        result_payload.update(
            Z_attn_clean=Z_attn_clean, Z_attn_noisy=Z_attn_noisy,
            loss_attn_clean=np.asarray(loss_attn_clean),
            loss_attn_noisy=np.asarray(loss_attn_noisy),
            attn_clean_time=attn_clean_time,
            attn_noisy_time=attn_noisy_time)
    if marmousi_crop is not None:
        result_payload.update(marmousi_crop=np.asarray(marmousi_crop))
    np.savez(RESULTS_NPZ, **result_payload)
    print(f"  Saved numerical results: {RESULTS_NPZ}")

    metrics = {
        "initial_model": {"snr_db": snr_init, "rmse": rmse_init},
        "admm_clean": {"snr_db": snr_rw_c, "rmse": rmse_rw_c},
        "admm_noisy": {"snr_db": snr_rw_n, "rmse": rmse_rw_n},
        "rescnn_clean": {"snr_db": snr_rc_c, "rmse": rmse_rc_c},
        "rescnn_noisy": {"snr_db": snr_rc_n, "rmse": rmse_rc_n},
        "unet_clean": {"snr_db": snr_un_c, "rmse": rmse_un_c},
        "unet_noisy": {"snr_db": snr_un_n, "rmse": rmse_un_n},
    }
    timings = {
        "admm_clean": rw_clean_time,
        "admm_noisy": rw_noisy_time,
        "rescnn_clean": rescnn_clean_time,
        "rescnn_noisy": rescnn_noisy_time,
        "unet_clean": unet_clean_time,
        "unet_noisy": unet_noisy_time,
    }
    if RUN_ATTENTION:
        metrics.update(
            attention_resunet_clean={"snr_db": snr_att_c, "rmse": rmse_att_c},
            attention_resunet_noisy={"snr_db": snr_att_n, "rmse": rmse_att_n})
        timings.update(attention_resunet_clean=attn_clean_time,
                       attention_resunet_noisy=attn_noisy_time)
    save_metadata_json(args, metrics, timings, marmousi_crop, dt_used)

    print("\n" + "=" * 75)
    print(f"{'Method':<28} {'Data':<8} {'SNR (dB)':<12} {'RMSE':<12} {'Time (s)':<10}")
    print("-" * 75)
    print(f"{'Initial model':<28} {'-':<8} {snr_init:<12.2f} {rmse_init:<12.2f} {'-':<10}")
    print(f"{'Reweighted L1 (ADMM)':<28} {'clean':<8} {snr_rw_c:<12.2f} {rmse_rw_c:<12.2f} {rw_clean_time:<10.1f}")
    print(f"{'Hybrid ResCNN':<28} {'clean':<8} {snr_rc_c:<12.2f} {rmse_rc_c:<12.2f} {rescnn_clean_time:<10.1f}")
    print(f"{'Physics-Informed 2D U-Net':<28} {'clean':<8} {snr_un_c:<12.2f} {rmse_un_c:<12.2f} {unet_clean_time:<10.1f}")
    print(f"{'Reweighted L1 (ADMM)':<28} {'noisy':<8} {snr_rw_n:<12.2f} {rmse_rw_n:<12.2f} {rw_noisy_time:<10.1f}")
    print(f"{'Hybrid ResCNN':<28} {'noisy':<8} {snr_rc_n:<12.2f} {rmse_rc_n:<12.2f} {rescnn_noisy_time:<10.1f}")
    print(f"{'Physics-Informed 2D U-Net':<28} {'noisy':<8} {snr_un_n:<12.2f} {rmse_un_n:<12.2f} {unet_noisy_time:<10.1f}")
    if RUN_ATTENTION:
        print(f"{'Attention ResUNet':<28} {'clean':<8} {snr_att_c:<12.2f} {rmse_att_c:<12.2f} {attn_clean_time:<10.1f}")
        print(f"{'Attention ResUNet':<28} {'noisy':<8} {snr_att_n:<12.2f} {rmse_att_n:<12.2f} {attn_noisy_time:<10.1f}")
    print("=" * 75)

    print("\nAll figures saved. Done.")
