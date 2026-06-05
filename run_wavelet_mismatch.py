#!/usr/bin/env python3
"""
Wavelet-mismatch robustness test (test-time generalization).

Synthetic benchmarks hand the inversion the *true* wavelet; field data never
does.  Here the clean seismic data are generated with the true Ricker wavelet,
but every inversion is run with a *wrong* wavelet, and -- without retraining --
we measure how each method degrades as the wavelet error grows:

  - trace-wise reweighted-L1 ADMM (He et al.)
  - 2D-coupled ADMM (lateral TV + reweighted-L1, consensus ADMM)
  - physics-informed 2D U-Net   (clean-trained checkpoint, applied as-is)
  - Hybrid ResCNN               (clean-trained checkpoint, applied as-is)

Two error axes are swept independently on clean data (so wavelet error is not
confounded with noise):
  (1) peak-frequency error  f0' = f0 * (1 + delta)
  (2) constant-phase rotation of the true wavelet by phi degrees

Each perturbed wavelet is rescaled to the true wavelet's L2 norm so the test
isolates shape/phase error from a trivial amplitude mismatch.

Outputs go to code/WAVELET_MISMATCH_RESULTS/<tag>/.
"""

import os
import json
import time
import importlib.util

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "2D_ADMM_PINN_ResCNN_Attention.py")
RESULTS_ROOT = os.path.join(HERE, "ADDM_PINN_RESULTS")
OUT_ROOT = os.path.join(HERE, "WAVELET_MISMATCH_RESULTS")

FREQ_DELTAS = [-0.30, -0.15, 0.0, 0.15, 0.30]      # fractional peak-freq error
PHASE_DEGS = [0.0, 30.0, 60.0, 90.0]               # constant phase rotation


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


def match_norm(w, ref):
    n = np.linalg.norm(w)
    return w * (np.linalg.norm(ref) / n) if n > 0 else w


def phase_rotate(w, deg):
    a = hilbert(w)                      # analytic signal w + i H(w)
    return np.real(a * np.exp(1j * np.deg2rad(deg)))


def classical_trace_wise(mod, S, wavelet, L_init, NX):
    Z = np.zeros((NX, L_init.shape[1]))
    for i in range(NX):
        Z[i, :], _ = mod.admm_l1_single_trace(
            S[i, :], wavelet, L_init[i, :],
            mod.MU, mod.ALPHA, mod.LAMBDA_, mod.EPSILON_CL,
            mod.MAX_ITER_CL, mod.TOL_CL, reweight=True)
    return Z


def load_net(mod, cls, ckpt_path, pnx, pnt):
    net = cls(pnx, pnt)
    mod.load_compatible_checkpoint(net, ckpt_path)
    net.to(mod.DEVICE); net.eval()
    return net


def evaluate(mod, S_clean, wav_wrong, Z_true, Z_init, L_init, NX, NT,
             corners, net_unet, net_rescnn, half_pad):
    S_trim = S_clean[:, half_pad:half_pad + NT]
    mu_t, alpha, lambda_t = mod.MU, mod.ALPHA, mod.LAMBDA_
    eps = mod.EPSILON_CL
    mu_x, rho = 100.0 * mod.MU, 40.0 * mod.ALPHA
    Z_rw = classical_trace_wise(mod, S_clean, wav_wrong, L_init, NX)
    Z_tv, _ = mod.admm_l1_2d_tv(S_clean, wav_wrong, L_init, mu_t, alpha, lambda_t,
                                mu_x=mu_x, rho=rho, epsilon=eps,
                                max_iter=40, tol=1e-6, vert_inner=10, reweight=True)
    L_prior = np.log(Z_rw + 1e-12)
    Z_unet = mod.predict_full_section(net_unet, S_trim, L_prior, corners,
                                      mod.PATCH_NX, mod.PATCH_NT, NX, NT)
    Z_rescnn = mod.predict_full_section(net_rescnn, S_trim, L_prior, corners,
                                        mod.PATCH_NX, mod.PATCH_NT, NX, NT)
    M = lambda Z: mod.compute_metrics(Z, Z_true)
    return {"rw": M(Z_rw), "tv": M(Z_tv), "unet": M(Z_unet), "rescnn": M(Z_rescnn)}


def process_crop(mod, tag, crop_dir):
    print(f"\n{'='*70}\n  Wavelet mismatch: {tag}\n{'='*70}")
    npz = np.load(os.path.join(crop_dir, "2D_results.npz"), allow_pickle=True)
    Z_true = npz["Z_true"]; Z_init = npz["Z_init"]
    S_clean = npz["S_clean"]; wav_true = npz["wavelet"]; dt = float(npz["dt"])
    L_init = np.log(Z_init)
    NX, NT = Z_true.shape
    nw = len(wav_true); half_pad = (nw - 1) // 2
    f0 = float(mod.F0)

    corners = mod.compute_patch_corners(NX, NT, mod.PATCH_NX, mod.PATCH_NT,
                                        mod.STRIDE_NX, mod.STRIDE_NT)
    net_unet = load_net(mod, mod.UNet2D, os.path.join(crop_dir, "2D_UNet_clean_pretrain.pt"),
                        mod.PATCH_NX, mod.PATCH_NT)
    net_rescnn = load_net(mod, mod.ResCNN2D, os.path.join(crop_dir, "2D_ResCNN_clean_pretrain.pt"),
                          mod.PATCH_NX, mod.PATCH_NT)

    def run_point(wav_wrong):
        return evaluate(mod, S_clean, wav_wrong, Z_true, Z_init, L_init, NX, NT,
                        corners, net_unet, net_rescnn, half_pad)

    freq_rows = []
    for d in FREQ_DELTAS:
        f0p = f0 * (1.0 + d)
        wav = match_norm(mod.ricker_wavelet(f0p, dt), wav_true)
        t0 = time.time()
        m = run_point(wav)
        print(f"  freq delta={d:+.2f} (f0={f0p:.1f}Hz)  "
              f"RW={m['rw'][0]:.2f} TV={m['tv'][0]:.2f} "
              f"ResCNN={m['rescnn'][0]:.2f} UNet={m['unet'][0]:.2f}  [{time.time()-t0:.0f}s]")
        freq_rows.append({"delta": d, "f0": f0p, **{k: list(v) for k, v in m.items()}})

    phase_rows = []
    for ph in PHASE_DEGS:
        wav = match_norm(phase_rotate(wav_true, ph), wav_true)
        t0 = time.time()
        m = run_point(wav)
        print(f"  phase={ph:+.0f} deg  RW={m['rw'][0]:.2f} TV={m['tv'][0]:.2f} "
              f"ResCNN={m['rescnn'][0]:.2f} UNet={m['unet'][0]:.2f}  [{time.time()-t0:.0f}s]")
        phase_rows.append({"phase_deg": ph, **{k: list(v) for k, v in m.items()}})

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "wavelet_mismatch.json"), "w") as f:
        json.dump({"benchmark": tag, "f0_true": f0,
                   "freq_rows": freq_rows, "phase_rows": phase_rows}, f, indent=2)
    mismatch_figure(freq_rows, phase_rows, tag, f0,
                    os.path.join(out_dir, "Wavelet_Mismatch_Curve.png"))
    print(f"  Saved to {out_dir}")
    return tag, freq_rows, phase_rows


def mismatch_figure(freq_rows, phase_rows, tag, f0, out_path):
    series = [("rw", "RW-L1 (trace-wise)", "#7570b3", "--o"),
              ("tv", "2D-coupled (TV+RW-L1)", "#1b9e77", "-.s"),
              ("rescnn", "Hybrid ResCNN", "#d95f02", "-^"),
              ("unet", "Physics-informed U-Net", "#2c7fb8", "-D")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    xf = [100.0 * r["delta"] for r in freq_rows]
    for key, lbl, col, sty in series:
        ax1.plot(xf, [r[key][0] for r in freq_rows], sty, color=col, lw=1.8, label=lbl)
    ax1.set_xlabel(f"Peak-frequency error (%)  [true f0={f0:.0f} Hz]")
    ax1.set_ylabel("Reconstruction SNR (dB)")
    ax1.set_title("(a) Frequency-mismatch robustness"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    xp = [r["phase_deg"] for r in phase_rows]
    for key, lbl, col, sty in series:
        ax2.plot(xp, [r[key][0] for r in phase_rows], sty, color=col, lw=1.8, label=lbl)
    ax2.set_xlabel("Constant phase rotation (degrees)")
    ax2.set_ylabel("Reconstruction SNR (dB)")
    ax2.set_title("(b) Phase-mismatch robustness"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f"Wavelet-mismatch robustness on clean data (true-wavelet data, wrong-wavelet inversion) — {tag}",
                 fontsize=11)
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
