#!/usr/bin/env python3
"""
Multi-seed confidence intervals for the physics-informed networks.

The headline Marmousi-2 metrics are single-seed (seed 42). To quantify training
stochasticity, this script retrains both networks from random initialization
(no transfer warm-start) at several seeds on each crop's noisy data, and reports
the mean +/- standard deviation of the reconstruction SNR/RMSE.

This is a deliberately conservative reproducibility check: it isolates the
variance of the learned refinement under random weight init, data shuffling, and
dropout, without relying on any pre-trained checkpoint. The classical baselines
are deterministic and are listed once for reference.

Outputs go to code/MULTISEED_CI_RESULTS/<tag>/.
"""

import os
import json
import time
import argparse
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "2D_ADMM_PINN_ResCNN_Attention.py")
RESULTS_ROOT = os.path.join(HERE, "ADDM_PINN_RESULTS")
OUT_ROOT = os.path.join(HERE, "MULTISEED_CI_RESULTS")

SEEDS = [42, 123, 7, 2024, 2025]


def load_main_module():
    spec = importlib.util.spec_from_file_location("admm_pinn_2d", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_crops():
    found = []
    for sub in sorted(os.listdir(RESULTS_ROOT)):
        cand = os.path.join(RESULTS_ROOT, sub, "2D_results.npz")
        if os.path.isfile(cand):
            found.append((sub, os.path.join(RESULTS_ROOT, sub)))
    return found


def classical_trace_wise(mod, S, wavelet, L_init, NX):
    Z = np.zeros((NX, L_init.shape[1]))
    for i in range(NX):
        Z[i, :], _ = mod.admm_l1_single_trace(
            S[i, :], wavelet, L_init[i, :],
            mod.MU, mod.ALPHA, mod.LAMBDA_, mod.EPSILON_CL,
            mod.MAX_ITER_CL, mod.TOL_CL, reweight=True)
    return Z


def train_once(mod, model_class, model_name, seed, wavelet,
               S_patches, L0_patches, Lt_patches, S_trim, L_prior,
               corners, NX, NT, transfer_ckpt):
    mod.set_seed(seed)
    t0 = time.time()
    net, _ = mod.train_model_2d(
        S_patches, L0_patches, wavelet, mod.PATCH_NX, mod.PATCH_NT,
        label=f"{model_name}-s{seed}", checkpoint_path=None,
        L_true_patches=Lt_patches, model_class=model_class,
        model_name=model_name, transfer_checkpoint=transfer_ckpt)
    dt = time.time() - t0
    Z = mod.predict_full_section(net, S_trim, L_prior, corners,
                                 mod.PATCH_NX, mod.PATCH_NT, NX, NT)
    return Z, dt


def process_crop(mod, tag, crop_dir, seeds, transfer):
    print(f"\n{'='*70}\n  Multi-seed CI: {tag}  (transfer={transfer})\n{'='*70}", flush=True)
    npz = np.load(os.path.join(crop_dir, "2D_results.npz"), allow_pickle=True)
    Z_true = npz["Z_true"]; Z_init = npz["Z_init"]; S_clean = npz["S_clean"]
    wavelet = npz["wavelet"]
    L_init = np.log(Z_init); L_true = np.log(Z_true)
    NX, NT = Z_true.shape
    nw = len(wavelet); half_pad = (nw - 1) // 2

    # noisy data at the nominal training SNR, classical prior computed once
    S_noisy = mod.add_noise(S_clean, mod.NOISE_SNR)
    S_trim = S_noisy[:, half_pad:half_pad + NT]
    Z_rw = classical_trace_wise(mod, S_noisy, wavelet, L_init, NX)
    L_prior = np.log(Z_rw + 1e-12)

    corners = mod.compute_patch_corners(NX, NT, mod.PATCH_NX, mod.PATCH_NT,
                                        mod.STRIDE_NX, mod.STRIDE_NT)
    S_patches = mod.extract_patches_at_corners(S_trim, corners, mod.PATCH_NX, mod.PATCH_NT)
    L0_patches = mod.extract_patches_at_corners(L_prior, corners, mod.PATCH_NX, mod.PATCH_NT)
    Lt_patches = mod.extract_patches_at_corners(L_true, corners, mod.PATCH_NX, mod.PATCH_NT)

    M = lambda Z: mod.compute_metrics(Z, Z_true)
    rw_m = M(Z_rw)
    print(f"  classical RW-L1 (noisy): SNR={rw_m[0]:.2f} dB  RMSE={rw_m[1]:.1f}", flush=True)

    transfer_paths = {
        "U-Net": os.path.join(crop_dir, "2D_UNet_noisy_pretrain.pt"),
        "ResCNN": os.path.join(crop_dir, "2D_ResCNN_noisy_pretrain.pt"),
    }

    per_model = {}
    for model_class, model_name in [(mod.UNet2D, "U-Net"), (mod.ResCNN2D, "ResCNN")]:
        tckpt = transfer_paths[model_name] if transfer else None
        snrs, rmses = [], []
        for s in seeds:
            Z, dt = train_once(mod, model_class, model_name, s, wavelet,
                               S_patches, L0_patches, Lt_patches, S_trim,
                               L_prior, corners, NX, NT, tckpt)
            m = M(Z)
            snrs.append(m[0]); rmses.append(m[1])
            print(f"    [{model_name} seed={s:4d}] SNR={m[0]:6.2f} dB  "
                  f"RMSE={m[1]:8.1f}  ({dt:.0f}s)", flush=True)
        per_model[model_name] = {
            "seeds": seeds,
            "snr": snrs, "rmse": rmses,
            "snr_mean": float(np.mean(snrs)), "snr_std": float(np.std(snrs, ddof=1)),
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses, ddof=1)),
        }
        pm = per_model[model_name]
        print(f"    >>> {model_name}: SNR={pm['snr_mean']:.2f}+/-{pm['snr_std']:.2f} dB  "
              f"RMSE={pm['rmse_mean']:.1f}+/-{pm['rmse_std']:.1f}", flush=True)

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    result = {"benchmark": tag, "transfer_init": transfer, "seeds": seeds,
              "data": "noisy", "snr_ratio": mod.NOISE_SNR,
              "rw_classical": list(rw_m), "models": per_model}
    fname = "multiseed_ci_transfer.json" if transfer else "multiseed_ci_scratch.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_dir}", flush=True)
    return tag, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--transfer", action="store_true",
                    help="Warm-start from the deployed per-crop checkpoint instead of random init.")
    ap.add_argument("--only", default=None, help="Process only this crop tag.")
    args = ap.parse_args()

    mod = load_main_module()
    if not hasattr(mod, "DEVICE") or mod.DEVICE is None:
        import torch
        mod.DEVICE = torch.device("cpu")
    os.makedirs(OUT_ROOT, exist_ok=True)
    crops = find_crops()
    if args.only:
        crops = [(t, d) for t, d in crops if t == args.only]
    print(f"Found crops: {[t for t, _ in crops]}  seeds={args.seeds}  "
          f"transfer={args.transfer}  DEVICE={mod.DEVICE}", flush=True)
    for tag, crop_dir in crops:
        process_crop(mod, tag, crop_dir, args.seeds, args.transfer)
    print(f"\nAll done. Output root: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
