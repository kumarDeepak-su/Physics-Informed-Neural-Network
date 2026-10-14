#!/usr/bin/env python3
"""Regenerate all script-A manuscript figures from cached results with bold-Times
fonts + larger axis ticks/labels/colorbars. No retraining, no science change.

Outputs (with manuscript basenames) into:
  /tmp/fonts_syn/   (synthetic benchmark)
  /tmp/fonts_fault/ (Marmousi fault crop)
"""
import os, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(HERE, "2D_ADMM_PINN_ResCNN_Attention.py")

# Build a PATCHED COPY of the canonical script (canonical stays untouched) with
# Unicode sub/superscripts -> mathtext, so Times New Roman renders them via STIX.
with open(MOD_PATH, "r") as f:
    _src = f.read()
_src = _src.replace("f₀", "$f_0$")                       # f0
_src = _src.replace("×10⁴", r"$\times 10^{4}$")     # x10^4
_src = _src.replace("(Z₊−Z₋)/(Z₊+Z₋)",
                    r"$(Z_+ - Z_-)/(Z_+ + Z_-)$")             # reflectivity formula
PATCHED = "/tmp/_fontmod.py"
with open(PATCHED, "w") as f:
    f.write(_src)

spec = importlib.util.spec_from_file_location("admm_pinn_fontmod", PATCHED)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import matplotlib.pyplot as plt

# ── Bold font override (default sans family, NOT Times) ───────────────────────
plt.rcParams.update({
    "font.weight": "bold",
    "mathtext.default": "bf",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Remove in-image figure headings ("Fig 1 - ...") to avoid clashing with the
# manuscript's own figure numbers/captions. Per-panel titles are kept.
import matplotlib.figure as _mf
_mf.Figure.suptitle = lambda self, *a, **k: None

compute_metrics = mod.compute_metrics
ROOT = HERE                       # fault cache under HERE/ADDM_PINN_RESULTS
MANU_ROOT = os.path.dirname(HERE) # synthetic cache under <manuscript>/results

def rtrue(Z):
    nx, nt = Z.shape
    r = np.zeros((nx, nt - 1))
    for i in range(nx):
        z = Z[i]; r[i] = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    return r

# ============================================================================
#  SYNTHETIC SET  (results/2D_ADMM_PINN.npz)
# ============================================================================
def do_synthetic():
    out = "/tmp/fonts_syn"; os.makedirs(out, exist_ok=True)
    mod.RESULTS_DIR = out
    d = np.load(os.path.join(MANU_ROOT, "results", "2D_ADMM_PINN.npz"), allow_pickle=True)
    Z_true = d["Z_true"]; Z_init = d["Z_init"]
    Z_rw_c = d["Z_rw_clean"]; Z_rw_n = d["Z_rw_noisy"]
    Z_rc_c = d["Z_rescnn_clean"]; Z_rc_n = d["Z_rescnn_noisy"]
    Z_un_c = d["Z_unet_clean"]; Z_un_n = d["Z_unet_noisy"]
    Z_at_c = d["Z_attn_clean"]; Z_at_n = d["Z_attn_noisy"]
    S_clean_trim = d["S_clean_trim"]; S_noisy_trim = d["S_noisy_trim"]
    wavelet = d["wavelet"]

    mod.BENCHMARK = "synthetic"
    NX, NT = Z_true.shape
    mod.NX = NX; mod.NT = NT; mod.DT = 0.001; mod.F0 = 30.0; mod.NOISE_SNR = 8.0
    x_coord = np.arange(NX, dtype=float)
    time_ms = np.arange(NT) * mod.DT * 1000.0
    r_true = rtrue(Z_true)

    mod.figure1_model_and_data(Z_true, Z_init, S_clean_trim, S_noisy_trim,
                               r_true, wavelet, time_ms, x_coord)          # 2D_Fig1_Model_Data
    mod.figure2_clean_inversion(Z_true, Z_init, Z_rw_c, Z_un_c,
                                time_ms, x_coord)                          # 2D_Fig2_Clean_Inversion
    mod.figure3_noisy_inversion(Z_true, Z_init, Z_rw_n, Z_un_n,
                                S_noisy_trim, time_ms, x_coord)            # 2D_Fig3_Noisy_Inversion
    mod.figure16_structural_zoom_profiles(Z_true, Z_rw_n, Z_un_n,
                                          time_ms, x_coord)                # 2D_Fig16_Structural

    # ── Fig12 (advanced attention 2x6): replicate published title/format ──────
    ext = [x_coord[0], x_coord[-1], time_ms[-1], time_ms[0]]
    vmin = Z_true.min() / 1e4; vmax = Z_true.max() / 1e4
    rows = [("Clean", [Z_true, Z_init, Z_rw_c, Z_rc_c, Z_un_c, Z_at_c]),
            ("Noisy", [Z_true, Z_init, Z_rw_n, Z_rc_n, Z_un_n, Z_at_n])]
    labels = ["True", "Initial", "He RW-L1 ADMM", "ResCNN", "U-Net", "Attention ResUNet"]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    for i, (row, dl) in enumerate(rows):
        for j, (label, data) in enumerate(zip(labels, dl)):
            ax = axes[i, j]
            im = ax.imshow(data.T / 1e4, aspect="auto", cmap="jet",
                           extent=ext, vmin=vmin, vmax=vmax)
            if label == "True":
                ax.set_title(f"{row}: {label}")
            else:
                snr, rmse = compute_metrics(data, Z_true)
                ax.set_title(f"{row}: {label}\nSNR={snr:.2f}, RMSE={rmse:.1f}")
            ax.set_xlabel("Trace"); ax.set_ylabel("Time (ms)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Advanced Comparison: ADMM-Guided ResCNN, U-Net, and Attention ResUNet")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "2D_Fig12_Advanced_AttentionResUNet_Comparison.png"),
                dpi=mod.SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("synthetic done ->", out)

# ============================================================================
#  FAULT CROP SET  (ADDM_PINN_RESULTS/marmousi_fault_crop_400x300/2D_results.npz)
# ============================================================================
def do_fault():
    out = "/tmp/fonts_fault"; os.makedirs(out, exist_ok=True)
    mod.RESULTS_DIR = out
    cache = os.path.join(ROOT, "ADDM_PINN_RESULTS",
                         "marmousi_fault_crop_400x300", "2D_results.npz")
    d = np.load(cache, allow_pickle=True)
    Z_true = d["Z_true"]; Z_init = d["Z_init"]
    Z_rw_c = d["Z_rw_clean"]; Z_rw_n = d["Z_rw_noisy"]
    Z_rc_c = d["Z_rescnn_clean"]; Z_rc_n = d["Z_rescnn_noisy"]
    Z_un_c = d["Z_unet_clean"]; Z_un_n = d["Z_unet_noisy"]
    S_clean_trim = d["S_clean_trim"]; S_noisy_trim = d["S_noisy_trim"]
    crop = d["marmousi_crop"]
    x0, x1 = int(crop[0]), int(crop[1])

    mod.BENCHMARK = "marmousi-crop"
    NX, NT = Z_true.shape
    mod.NX = NX; mod.NT = NT; mod.DT = float(d["dt"]); mod.NOISE_SNR = 8.0
    x_coord = np.arange(x0, x1, dtype=float)
    time_ms = np.arange(NT) * mod.DT * 1000.0
    r_true = rtrue(Z_true)

    # Crop_Location (needs full Marmousi AI volume from SEG-Y)
    seg_dir = os.environ.get("MARMOUSI_DIR", os.path.join(HERE, "..", "data", "marmousi2"))
    vp, _ = mod.read_segy_simple(os.path.join(seg_dir, "vp.segy"))
    rho, _ = mod.read_segy_simple(os.path.join(seg_dir, "density.segy"))
    Z_full = vp * rho
    mod.figure_marmousi_crop_overview(Z_full, (int(crop[0]), int(crop[1]),
                                               int(crop[2]), int(crop[3])))  # Marmousi_Crop_Location

    mod.figure_marmousi_model_and_data(Z_true, Z_init, S_clean_trim, S_noisy_trim,
                                       r_true, time_ms, x_coord)              # 2D_Fig1_Model_Data
    mod.figure3_noisy_inversion(Z_true, Z_init, Z_rw_n, Z_un_n,
                                S_noisy_trim, time_ms, x_coord)               # 2D_Fig3_Noisy_Inversion
    mod.figure4_single_traces(Z_true, Z_init, Z_rw_c, Z_rw_n,
                              Z_un_c, Z_un_n, time_ms)                        # 2D_Fig4_Traces
    mod.figure8_metrics(Z_true, Z_init, Z_rw_c, Z_rw_n,
                        Z_rc_c, Z_rc_n, Z_un_c, Z_un_n)                       # 2D_Fig8_Metrics
    mod.figure9_architecture_comparison(Z_true, Z_rw_c, Z_rw_n,
                                        Z_rc_c, Z_rc_n, Z_un_c, Z_un_n,
                                        time_ms, x_coord)                     # 2D_Fig9_Architecture
    print("fault done ->", out)


if __name__ == "__main__":
    do_synthetic()
    do_fault()
    print("\nALL SCRIPT-A FIGURES REGENERATED")
