#!/usr/bin/env python3
"""
Noise-robustness sweep with MATCHED retraining.

Unlike run_noise_sweep.py (which applies a single SNR=8 model across all noise
levels), this script fine-tunes both networks *at each noise level* so that the
comparison is matched: every method is given data at the same input SNR it is
evaluated on.

For each Marmousi-2 crop and each input-noise level:
  - regenerate noisy seismic at that level
  - recompute the trace-wise reweighted-L1 prior (also the neural input prior)
  - fine-tune U-Net and ResCNN for EPOCHS epochs, warm-started from the
    published per-crop checkpoint, with the supervised Marmousi loss
  - evaluate all four methods (RW-L1, 2D-coupled, U-Net, ResCNN)

The trained SNR=8 (ratio 8) level reuses the published cached result, and the
classical RW-L1 / 2D-coupled metrics are read from the test-time sweep JSON
(they do not depend on training), so only the off-nominal neural models are
retrained.

Outputs go to code/NOISE_SWEEP_MATCHED_RESULTS/<tag>/.
"""

import os
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
SWEEP_ROOT = os.path.join(HERE, "NOISE_SWEEP_RESULTS")        # test-time sweep (classical reuse)
OUT_ROOT = os.path.join(HERE, "NOISE_SWEEP_MATCHED_RESULTS")

SNR_RATIOS = [2.0, 4.0, 8.0, 16.0, 32.0]
TRAINED_RATIO = 8.0     # published model; reuse cached neural result here


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


def retrain_and_predict(mod, model_class, transfer_ckpt, S_trim, L_prior,
                        L_true, corners, NX, NT, label):
    patch_nx, patch_nt = mod.PATCH_NX, mod.PATCH_NT
    S_patches = mod.extract_patches_at_corners(S_trim, corners, patch_nx, patch_nt)
    L0_patches = mod.extract_patches_at_corners(L_prior, corners, patch_nx, patch_nt)
    Lt_patches = mod.extract_patches_at_corners(L_true, corners, patch_nx, patch_nt)
    t0 = time.time()
    net, _ = mod.train_model_2d(
        S_patches, L0_patches, mod_wavelet, patch_nx, patch_nt,
        label=label, checkpoint_path=None, L_true_patches=Lt_patches,
        model_class=model_class, model_name=("U-Net" if model_class is mod.UNet2D else "ResCNN"),
        transfer_checkpoint=transfer_ckpt)
    dt_train = time.time() - t0
    Z = mod.predict_full_section(net, S_trim, L_prior, corners, patch_nx, patch_nt, NX, NT)
    print(f"      [{label}] trained in {dt_train:.0f}s")
    return Z, dt_train


def process_crop(mod, tag, crop_dir):
    global mod_wavelet
    print(f"\n{'='*70}\n  MATCHED noise sweep: {tag}\n{'='*70}", flush=True)
    npz = np.load(os.path.join(crop_dir, "2D_results.npz"), allow_pickle=True)
    Z_true = npz["Z_true"]; Z_init = npz["Z_init"]; S_clean = npz["S_clean"]
    wavelet = npz["wavelet"]; mod_wavelet = wavelet
    L_init = np.log(Z_init); L_true = np.log(Z_true)
    NX, NT = Z_true.shape
    nw = len(wavelet); half_pad = (nw - 1) // 2

    mu_t, alpha, lambda_t = mod.MU, mod.ALPHA, mod.LAMBDA_
    eps = mod.EPSILON_CL
    mu_x, rho = 100.0 * mod.MU, 40.0 * mod.ALPHA
    corners = mod.compute_patch_corners(NX, NT, mod.PATCH_NX, mod.PATCH_NT,
                                        mod.STRIDE_NX, mod.STRIDE_NT)

    # classical metrics reused from the test-time sweep
    sweep = json.load(open(os.path.join(SWEEP_ROOT, tag, "noise_sweep.json")))
    classical = {round(r["ratio"], 6): r for r in sweep["rows"]}

    unet_ckpt = os.path.join(crop_dir, "2D_UNet_noisy_pretrain.pt")
    rescnn_ckpt = os.path.join(crop_dir, "2D_ResCNN_noisy_pretrain.pt")
    M = lambda Z: mod.compute_metrics(Z, Z_true)

    rows = []
    for ratio in SNR_RATIOS:
        snr_db = 20.0 * np.log10(ratio)
        c = classical[round(ratio, 6)]
        rw_m, tv_m = c["rw"], c["tv"]
        print(f"\n  --- input SNR ratio={ratio:g} ({snr_db:.1f} dB) ---", flush=True)

        if abs(ratio - TRAINED_RATIO) < 1e-9:
            unet_m = M(npz["Z_unet_noisy"]); rescnn_m = M(npz["Z_rescnn_noisy"])
            print("    (reusing published cached neural result)", flush=True)
        else:
            S_noisy = mod.add_noise(S_clean, ratio)
            S_trim = S_noisy[:, half_pad:half_pad + NT]
            Z_rw = classical_trace_wise(mod, S_noisy, wavelet, L_init, NX)
            L_prior = np.log(Z_rw + 1e-12)
            Z_unet, _ = retrain_and_predict(mod, mod.UNet2D, unet_ckpt, S_trim,
                                            L_prior, L_true, corners, NX, NT,
                                            f"unet-r{ratio:g}")
            Z_rescnn, _ = retrain_and_predict(mod, mod.ResCNN2D, rescnn_ckpt, S_trim,
                                              L_prior, L_true, corners, NX, NT,
                                              f"rescnn-r{ratio:g}")
            unet_m = M(Z_unet); rescnn_m = M(Z_rescnn)

        for k, name, m in [("rw", "RW-L1 (1D)", rw_m), ("tv", "2D-coupled", tv_m),
                           ("rescnn", "ResCNN", rescnn_m), ("unet", "U-Net", unet_m)]:
            print(f"      {name:12s} SNR={m[0]:6.2f} dB  RMSE={m[1]:8.1f}", flush=True)
        rows.append({"ratio": ratio, "snr_db_in": snr_db,
                     "rw": list(rw_m), "tv": list(tv_m),
                     "rescnn": list(rescnn_m), "unet": list(unet_m)})

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "noise_sweep_matched.json"), "w") as f:
        json.dump({"benchmark": tag, "protocol": "matched-retrain",
                   "snr_ratios": SNR_RATIOS, "rows": rows}, f, indent=2)
    sweep_figure(rows, tag, os.path.join(out_dir, "Noise_Sweep_Matched_Curve.png"))
    print(f"  Saved to {out_dir}", flush=True)
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
    fig.suptitle(f"Noise-robustness sweep (networks retrained at each level) — {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


mod_wavelet = None


def main():
    mod = load_main_module()
    if not hasattr(mod, "DEVICE") or mod.DEVICE is None:
        import torch
        mod.DEVICE = torch.device("cpu")
    os.makedirs(OUT_ROOT, exist_ok=True)
    crops = find_crops()
    print(f"Found crops: {[t for t, _ in crops]}  DEVICE={mod.DEVICE}", flush=True)
    for tag, crop_dir in crops:
        process_crop(mod, tag, crop_dir)
    print(f"\nAll done. Output root: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
