#!/usr/bin/env python3
"""
Noise-robustness sweep (test-time generalization).

For each Marmousi-2 crop, regenerate the seismic data at several input-noise
levels and, *without retraining*, evaluate four methods at each level:
  - trace-wise reweighted-L1 ADMM (He et al.)
  - 2D-coupled ADMM (lateral TV + reweighted-L1, consensus ADMM)
  - physics-informed 2D U-Net   (noisy-trained checkpoint, applied as-is)
  - Hybrid ResCNN               (noisy-trained checkpoint, applied as-is)

The networks are trained once (on the SNR=8 case) and applied across the whole
sweep, so the curve measures how a single deployed model degrades as the input
SNR drops -- a test-time robustness / generalization study, not a re-fit.

Outputs go to code/NOISE_SWEEP_RESULTS/<tag>/.
"""

import os
import sys
import json
import time
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "2D_ADMM_PINN_ResCNN_Attention.py")
RESULTS_ROOT = os.path.join(HERE, "ADDM_PINN_RESULTS")
OUT_ROOT = os.path.join(HERE, "NOISE_SWEEP_RESULTS")

# amplitude-SNR ratios (same units as add_noise / NOISE_SNR=8 in the main code)
SNR_RATIOS = [2.0, 4.0, 8.0, 16.0, 32.0]


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


def load_net(mod, cls, ckpt_path, patch_nx, patch_nt):
    net = cls(patch_nx, patch_nt)
    mod.load_compatible_checkpoint(net, ckpt_path)
    net.to(mod.DEVICE)
    net.eval()
    return net


def process_crop(mod, tag, crop_dir):
    print(f"\n{'='*70}\n  Noise sweep: {tag}\n{'='*70}")
    npz = np.load(os.path.join(crop_dir, "2D_results.npz"), allow_pickle=True)
    Z_true = npz["Z_true"]
    Z_init = npz["Z_init"]
    S_clean = npz["S_clean"]
    wavelet = npz["wavelet"]
    dt = float(npz["dt"])
    L_init = np.log(Z_init)
    NX, NT = Z_true.shape
    nw = len(wavelet)
    half_pad = (nw - 1) // 2

    # 2D-coupled params identical to the manuscript baseline
    mu_t, alpha, lambda_t = mod.MU, mod.ALPHA, mod.LAMBDA_
    eps = mod.EPSILON_CL
    mu_x, rho = 100.0 * mod.MU, 40.0 * mod.ALPHA

    # neural prior fed to the net is log of the trace-wise RW-L1 result
    corners = mod.compute_patch_corners(NX, NT, mod.PATCH_NX, mod.PATCH_NT,
                                        mod.STRIDE_NX, mod.STRIDE_NT)
    net_unet = load_net(mod, mod.UNet2D,
                        os.path.join(crop_dir, "2D_UNet_noisy_pretrain.pt"),
                        mod.PATCH_NX, mod.PATCH_NT)
    net_rescnn = load_net(mod, mod.ResCNN2D,
                          os.path.join(crop_dir, "2D_ResCNN_noisy_pretrain.pt"),
                          mod.PATCH_NX, mod.PATCH_NT)

    def M(Z):
        return mod.compute_metrics(Z, Z_true)

    rows = []
    for ratio in SNR_RATIOS:
        snr_db = 20.0 * np.log10(ratio)
        print(f"\n  --- input SNR ratio={ratio:g} ({snr_db:.1f} dB) ---")
        S_noisy = mod.add_noise(S_clean, ratio)
        S_noisy_trim = S_noisy[:, half_pad:half_pad + NT]

        t0 = time.time()
        Z_rw = classical_trace_wise(mod, S_noisy, wavelet, L_init, NX)
        print(f"    trace-wise RW-L1 done in {time.time()-t0:.1f}s")

        t0 = time.time()
        Z_tv, _ = mod.admm_l1_2d_tv(
            S_noisy, wavelet, L_init, mu_t, alpha, lambda_t,
            mu_x=mu_x, rho=rho, epsilon=eps,
            max_iter=40, tol=1e-6, vert_inner=10, reweight=True)
        print(f"    2D-coupled done in {time.time()-t0:.1f}s")

        L_prior = np.log(Z_rw + 1e-12)
        Z_unet = mod.predict_full_section(net_unet, S_noisy_trim, L_prior,
                                          corners, mod.PATCH_NX, mod.PATCH_NT, NX, NT)
        Z_rescnn = mod.predict_full_section(net_rescnn, S_noisy_trim, L_prior,
                                            corners, mod.PATCH_NX, mod.PATCH_NT, NX, NT)

        m = {"rw": M(Z_rw), "tv": M(Z_tv), "unet": M(Z_unet), "rescnn": M(Z_rescnn)}
        for k, name in [("rw", "RW-L1 (1D)"), ("tv", "2D-coupled"),
                        ("unet", "U-Net"), ("rescnn", "ResCNN")]:
            print(f"      {name:12s} SNR={m[k][0]:6.2f} dB  RMSE={m[k][1]:8.1f}")
        rows.append({"ratio": ratio, "snr_db_in": snr_db,
                     **{k: list(v) for k, v in m.items()}})

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "noise_sweep.json"), "w") as f:
        json.dump({"benchmark": tag, "snr_ratios": SNR_RATIOS, "rows": rows}, f, indent=2)

    sweep_figure(rows, tag, os.path.join(out_dir, "Noise_Sweep_Curve.png"))
    print(f"  Saved to {out_dir}")
    return tag, rows


def sweep_figure(rows, tag, out_path):
    x = [r["snr_db_in"] for r in rows]
    series = [("rw", "RW-L1 (trace-wise)", "#7570b3", "--o"),
              ("tv", "2D-coupled (TV+RW-L1)", "#1b9e77", "-.s"),
              ("rescnn", "Hybrid ResCNN", "#d95f02", "-^"),
              ("unet", "Physics-informed U-Net", "#2c7fb8", "-D")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for key, lbl, col, sty in series:
        ax1.plot(x, [r[key][0] for r in rows], sty, color=col, lw=1.8, label=lbl)
        ax2.plot(x, [r[key][1] for r in rows], sty, color=col, lw=1.8, label=lbl)
    ax1.set_xlabel("Input SNR (dB)"); ax1.set_ylabel("Reconstruction SNR (dB)")
    ax1.set_title("(a) Output SNR vs input SNR"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("Input SNR (dB)"); ax2.set_ylabel("Reconstruction RMSE")
    ax2.set_title("(b) Output RMSE vs input SNR"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f"Noise-robustness sweep (single trained model, applied across noise) — {tag}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    mod = load_main_module()
    if not hasattr(mod, "DEVICE") or mod.DEVICE is None:
        import torch
        mod.DEVICE = torch.device("cpu")
    os.makedirs(OUT_ROOT, exist_ok=True)
    crops = find_crops()
    print(f"Found crops: {[t for t, _ in crops]}")
    for tag, crop_dir in crops:
        process_crop(mod, tag, crop_dir)
    print(f"\nAll done. Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
