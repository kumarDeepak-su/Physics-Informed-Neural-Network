#!/usr/bin/env python3
"""
2D spatially-coupled classical baseline comparison.

Addresses the reviewer concern that the neural network is only compared against
a *trace-wise* He et al. solver, so its advantage could be dismissed as "just
2D spatial coupling".  Here we add a genuine 2D-coupled classical baseline
(`admm_l1_2d_tv`: vertical reweighted-L1 + lateral edge-preserving TV, solved
matrix-free by conjugate gradient) and show the physics-informed networks still
win.

The script reuses the *already trained* neural results stored in the existing
2D_results.npz files (no 20-minute retrain): it loads S_clean / S_noisy /
Z_init / Z_true / wavelet, recomputes the trace-wise and the new 2D-coupled
classical inversions, and renders side-by-side comparisons + metrics into a NEW
folder:  code/2D_TV_BASELINE_RESULTS/<benchmark>/.
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
OUT_ROOT = os.path.join(HERE, "2D_TV_BASELINE_RESULTS")


def load_main_module():
    """Import the (non-identifier-named) main module by file path."""
    spec = importlib.util.spec_from_file_location("admm_pinn_2d", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_npz_files():
    root = os.path.join(HERE, "ADDM_PINN_RESULTS")
    found = []
    for sub in sorted(os.listdir(root)):
        cand = os.path.join(root, sub, "2D_results.npz")
        if os.path.isfile(cand):
            found.append((sub, cand))
    return found


def panel(ax, Z, title, vmin, vmax, dt):
    nx, nt = Z.shape
    im = ax.imshow(Z.T, aspect="auto", cmap="viridis",
                   extent=[0, nx, nt * dt, 0], vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (s)")
    return im


def section_figure(Z_true, Z_init, Z_rw, Z_tv, Z_unet, Z_rescnn,
                   metrics, dt, tag, out_path):
    vmin, vmax = np.percentile(Z_true, [1, 99])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.14, hspace=0.06, wspace=0.06)
    items = [
        (Z_true,   "(a) True impedance",            None),
        (Z_init,   "(b) Initial model",             "init"),
        (Z_rw,     "(c) Classical RW-L1 (trace-wise)", "rw"),
        (Z_tv,     "(d) Classical 2D-coupled (TV+RW-L1)", "tv"),
        (Z_unet,   "(e) PINN U-Net2D",              "unet"),
        (Z_rescnn, "(f) PINN ResCNN2D",             "rescnn"),
    ]
    for ax, (Z, ttl, key) in zip(axes.ravel(), items):
        if key is not None and key in metrics:
            ttl = f"{ttl}\nSNR={metrics[key][0]:.2f} dB | RMSE={metrics[key][1]:.0f}"
        im = panel(ax, Z, ttl, vmin, vmax, dt)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Impedance")
    fig.suptitle(f"2D classical-vs-PINN comparison — {tag}", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def metrics_bar_figure(m_clean, m_noisy, out_path, tag):
    labels = ["Initial", "RW-L1\n(trace-wise)", "2D-coupled\n(TV+RW-L1)",
              "PINN\nU-Net2D", "PINN\nResCNN2D"]
    keys = ["init", "rw", "tv", "unet", "rescnn"]
    snr_c = [m_clean[k][0] for k in keys]
    snr_n = [m_noisy[k][0] for k in keys]
    rmse_c = [m_clean[k][1] for k in keys]
    rmse_n = [m_noisy[k][1] for k in keys]

    x = np.arange(len(labels))
    width = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(x - width / 2, snr_c, width, label="Clean", color="#2c7fb8")
    ax1.bar(x + width / 2, snr_n, width, label="Noisy", color="#e6550d")
    ax1.set_ylabel("SNR (dB)  — higher is better")
    ax1.set_title("(a) Reconstruction SNR")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for xi, (vc, vn) in enumerate(zip(snr_c, snr_n)):
        ax1.text(xi - width / 2, vc, f"{vc:.1f}", ha="center", va="bottom", fontsize=8)
        ax1.text(xi + width / 2, vn, f"{vn:.1f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x - width / 2, rmse_c, width, label="Clean", color="#2c7fb8")
    ax2.bar(x + width / 2, rmse_n, width, label="Noisy", color="#e6550d")
    ax2.set_ylabel("RMSE  — lower is better")
    ax2.set_title("(b) Reconstruction RMSE")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Classical (1D & 2D) vs physics-informed networks — {tag}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def trace_figure(Z_true, Z_rw, Z_tv, Z_unet, dt, tag, out_path):
    nx, nt = Z_true.shape
    t = np.arange(nt) * dt
    cols = [int(0.30 * nx), int(0.55 * nx), int(0.78 * nx)]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 6), sharey=True)
    for ax, c in zip(axes, cols):
        ax.plot(Z_true[c], t, "k", lw=2.0, label="True")
        ax.plot(Z_rw[c], t, "--", color="#7570b3", lw=1.3, label="RW-L1 (1D)")
        ax.plot(Z_tv[c], t, "-.", color="#1b9e77", lw=1.3, label="2D-coupled")
        ax.plot(Z_unet[c], t, color="#d95f02", lw=1.3, label="PINN U-Net")
        ax.invert_yaxis()
        ax.set_title(f"Trace {c}")
        ax.set_xlabel("Impedance")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Time (s)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Representative vertical profiles — {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process(mod, tag, npz_path):
    print(f"\n{'='*70}\n  Benchmark: {tag}\n{'='*70}")
    d = np.load(npz_path, allow_pickle=True)
    Z_true = d["Z_true"]; Z_init = d["Z_init"]
    S_clean = d["S_clean"]; S_noisy = d["S_noisy"]
    wavelet = d["wavelet"]; dt = float(d["dt"])
    Z_rw_clean = d["Z_rw_clean"]; Z_rw_noisy = d["Z_rw_noisy"]
    Z_unet_clean = d["Z_unet_clean"]; Z_unet_noisy = d["Z_unet_noisy"]
    Z_rescnn_clean = d["Z_rescnn_clean"]; Z_rescnn_noisy = d["Z_rescnn_noisy"]
    L_init = np.log(Z_init)

    # classical 2D-coupled baseline parameters: same vertical reweighted-L1 as
    # the trace-wise He et al. solver (mu_t, alpha, lambda_t), plus a lateral
    # total-variation coupling (mu_x) solved by consensus ADMM.
    mu_t = mod.MU
    alpha = mod.ALPHA
    lambda_t = mod.LAMBDA_
    eps = mod.EPSILON_CL
    mu_x = 100.0 * mod.MU        # lateral TV weight (consensus coupling)
    rho = 40.0 * mod.ALPHA       # consensus penalty
    max_iter, vert_inner = 40, 10

    print("  Running 2D-coupled classical baseline (clean)...")
    t0 = time.time()
    Z_tv_clean, conv_c = mod.admm_l1_2d_tv(
        S_clean, wavelet, L_init, mu_t, alpha, lambda_t,
        mu_x=mu_x, rho=rho, epsilon=eps,
        max_iter=max_iter, tol=1e-6, vert_inner=vert_inner, reweight=True)
    tv_clean_time = time.time() - t0
    print(f"    done in {tv_clean_time:.1f}s, {len(conv_c)} outer iters")

    print("  Running 2D-coupled classical baseline (noisy)...")
    t0 = time.time()
    Z_tv_noisy, conv_n = mod.admm_l1_2d_tv(
        S_noisy, wavelet, L_init, mu_t, alpha, lambda_t,
        mu_x=mu_x, rho=rho, epsilon=eps,
        max_iter=max_iter, tol=1e-6, vert_inner=vert_inner, reweight=True)
    tv_noisy_time = time.time() - t0
    print(f"    done in {tv_noisy_time:.1f}s, {len(conv_n)} outer iters")

    def M(Z):
        return mod.compute_metrics(Z, Z_true)

    m_clean = {
        "init": M(Z_init), "rw": M(Z_rw_clean), "tv": M(Z_tv_clean),
        "unet": M(Z_unet_clean), "rescnn": M(Z_rescnn_clean),
    }
    m_noisy = {
        "init": M(Z_init), "rw": M(Z_rw_noisy), "tv": M(Z_tv_noisy),
        "unet": M(Z_unet_noisy), "rescnn": M(Z_rescnn_noisy),
    }

    print("\n  --- SNR (dB) / RMSE ---")
    for name, key in [("Initial model", "init"), ("RW-L1 (trace-wise)", "rw"),
                      ("2D-coupled (TV+RW-L1)", "tv"), ("PINN U-Net2D", "unet"),
                      ("PINN ResCNN2D", "rescnn")]:
        print(f"    {name:24s} clean: {m_clean[key][0]:6.2f} dB / {m_clean[key][1]:8.1f}"
              f"   noisy: {m_noisy[key][0]:6.2f} dB / {m_noisy[key][1]:8.1f}")

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    section_figure(Z_true, Z_init, Z_rw_clean, Z_tv_clean,
                   Z_unet_clean, Z_rescnn_clean, m_clean, dt,
                   f"{tag} — CLEAN",
                   os.path.join(out_dir, "Fig_A_sections_clean.png"))
    section_figure(Z_true, Z_init, Z_rw_noisy, Z_tv_noisy,
                   Z_unet_noisy, Z_rescnn_noisy, m_noisy, dt,
                   f"{tag} — NOISY",
                   os.path.join(out_dir, "Fig_B_sections_noisy.png"))
    metrics_bar_figure(m_clean, m_noisy,
                       os.path.join(out_dir, "Fig_C_metrics_bar.png"), tag)
    trace_figure(Z_true, Z_rw_noisy, Z_tv_noisy, Z_unet_noisy, dt,
                 f"{tag} — NOISY", os.path.join(out_dir, "Fig_D_traces_noisy.png"))

    summary = {
        "benchmark": tag,
        "source_npz": npz_path,
        "classical_2d_params": {
            "mu_t": mu_t, "alpha": alpha, "lambda_t": lambda_t,
            "mu_x": mu_x, "rho": rho, "epsilon": eps,
            "max_iter": max_iter, "vert_inner": vert_inner, "reweight": True,
        },
        "runtime_sec": {"tv_clean": tv_clean_time, "tv_noisy": tv_noisy_time},
        "metrics_clean_snr_rmse": {k: list(v) for k, v in m_clean.items()},
        "metrics_noisy_snr_rmse": {k: list(v) for k, v in m_noisy.items()},
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join(out_dir, "Z_tv_baseline.npz"),
        Z_tv_clean=Z_tv_clean, Z_tv_noisy=Z_tv_noisy,
        conv_clean=np.asarray(conv_c), conv_noisy=np.asarray(conv_n))

    print(f"  Saved figures + metrics to: {out_dir}")
    return tag, m_clean, m_noisy


def main():
    mod = load_main_module()
    os.makedirs(OUT_ROOT, exist_ok=True)
    files = find_npz_files()
    if not files:
        print("No 2D_results.npz files found under ADDM_PINN_RESULTS/.")
        sys.exit(1)
    print(f"Found {len(files)} benchmark result file(s): "
          f"{[t for t, _ in files]}")
    for tag, path in files:
        process(mod, tag, path)
    print(f"\nAll done. Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
