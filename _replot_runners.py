#!/usr/bin/env python3
"""Regenerate the two runner-script figures from CACHED results with bold-Times
fonts + larger tick/label/title/legend sizes. No recompute, no retrain.

  TV baseline   -> Fig17 (sections noisy), Fig18 (metrics bar)   -> /tmp/fonts_tv
  Wavelet mism. -> Fig23 (fault crop), Fig24 (second crop)       -> /tmp/fonts_wav

The canonical runner sources stay untouched; PATCHED COPIES are written to /tmp
with (a) hardcoded small fontsizes bumped up and (b) Unicode em-dashes -> hyphens
so the regenerated titles match the manuscript's no-em-dash style.
"""
import os, json, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
TV_SRC  = os.path.join(HERE, "run_2d_tv_baseline.py")
WAV_SRC = os.path.join(HERE, "run_wavelet_mismatch.py")
NS_SRC  = os.path.join(HERE, "run_noise_sweep_matched.py")


def patch_and_import(src_path, mod_name, replacements):
    with open(src_path, "r") as f:
        s = f.read()
    for a, b in replacements:
        s = s.replace(a, b)
    patched = f"/tmp/_{mod_name}.py"
    with open(patched, "w") as f:
        f.write(s)
    spec = importlib.util.spec_from_file_location(mod_name, patched)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# bump hardcoded fontsizes + drop em-dashes (U+2014) for both runners
COMMON = [("—", "-")]
TV_REPL = COMMON + [
    ("fontsize=10)", "fontsize=17)"),   # panel titles
    ("fontsize=14)", "fontsize=19)"),   # sections suptitle
    ("fontsize=13)", "fontsize=19)"),   # metrics-bar suptitle
    ("fontsize=9)",  "fontsize=13)"),   # xticklabels
    ("fontsize=8)",  "fontsize=11)"),   # bar value labels
]
WAV_REPL = COMMON + [
    ("fontsize=8)",  "fontsize=13)"),   # legends
    ("fontsize=11)", "fontsize=17)"),   # suptitle
]
NS_REPL = COMMON + [
    ("fontsize=8)",  "fontsize=13)"),   # legends
    ("fontsize=12)", "fontsize=17)"),   # suptitle
]

tv  = patch_and_import(TV_SRC,  "tvmod",  TV_REPL)
wav = patch_and_import(WAV_SRC, "wavmod", WAV_REPL)
ns  = patch_and_import(NS_SRC,  "nsmod",  NS_REPL)

import matplotlib.pyplot as plt

# bold override (default sans family, NOT Times) - identical to script-A driver
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

# Remove in-image figure headings (suptitles) to avoid clashing with the
# manuscript's own figure numbers/captions. Per-panel "(a)/(b)" titles kept.
import matplotlib.figure as _mf
_mf.Figure.suptitle = lambda self, *a, **k: None

FAULT = "marmousi_fault_crop_400x300"
SECOND = "marmousi_crop_400x300"


def do_tv():
    out = "/tmp/fonts_tv"; os.makedirs(out, exist_ok=True)
    tag = FAULT
    npz = np.load(os.path.join(HERE, "ADDM_PINN_RESULTS", tag, "2D_results.npz"),
                  allow_pickle=True)
    Z_true = npz["Z_true"]; Z_init = npz["Z_init"]; dt = float(npz["dt"])
    Z_rw_n = npz["Z_rw_noisy"]; Z_un_n = npz["Z_unet_noisy"]; Z_rc_n = npz["Z_rescnn_noisy"]
    tvz = np.load(os.path.join(HERE, "2D_TV_BASELINE_RESULTS", tag, "Z_tv_baseline.npz"))
    Z_tv_n = tvz["Z_tv_noisy"]
    with open(os.path.join(HERE, "2D_TV_BASELINE_RESULTS", tag, "metrics_summary.json")) as f:
        summ = json.load(f)
    m_noisy = {k: tuple(v) for k, v in summ["metrics_noisy_snr_rmse"].items()}
    m_clean = {k: tuple(v) for k, v in summ["metrics_clean_snr_rmse"].items()}

    # Fig17 = sections (noisy)
    tv.section_figure(Z_true, Z_init, Z_rw_n, Z_tv_n, Z_un_n, Z_rc_n,
                      m_noisy, dt, f"{tag} - NOISY",
                      os.path.join(out, "2D_Fig17_2D_Coupled_Baseline_Sections.png"))
    # Fig18 = metrics bar
    tv.metrics_bar_figure(m_clean, m_noisy,
                          os.path.join(out, "2D_Fig18_2D_Coupled_Baseline_Metrics.png"), tag)

    # Fig19 = SECOND-crop section (noisy) -- same section_figure as Fig17
    tag2 = SECOND
    n2 = np.load(os.path.join(HERE, "ADDM_PINN_RESULTS", tag2, "2D_results.npz"),
                 allow_pickle=True)
    dt2 = float(n2["dt"])
    tv2 = np.load(os.path.join(HERE, "2D_TV_BASELINE_RESULTS", tag2, "Z_tv_baseline.npz"))
    with open(os.path.join(HERE, "2D_TV_BASELINE_RESULTS", tag2, "metrics_summary.json")) as f:
        s2 = json.load(f)
    m2 = {k: tuple(v) for k, v in s2["metrics_noisy_snr_rmse"].items()}
    tv.section_figure(n2["Z_true"], n2["Z_init"], n2["Z_rw_noisy"], tv2["Z_tv_noisy"],
                      n2["Z_unet_noisy"], n2["Z_rescnn_noisy"], m2, dt2,
                      f"{tag2} - NOISY",
                      os.path.join(out, "2D_Fig19_SecondCrop_Noisy_Comparison.png"))
    print("TV done ->", out)


def do_wav():
    out = "/tmp/fonts_wav"; os.makedirs(out, exist_ok=True)
    for tag, fname in [(FAULT, "2D_Fig23_WaveletMismatch_FaultCrop.png"),
                       (SECOND, "2D_Fig24_WaveletMismatch_SecondCrop.png")]:
        with open(os.path.join(HERE, "WAVELET_MISMATCH_RESULTS", tag,
                               "wavelet_mismatch.json")) as f:
            j = json.load(f)
        wav.mismatch_figure(j["freq_rows"], j["phase_rows"], tag,
                            float(j["f0_true"]), os.path.join(out, fname))
    print("wavelet done ->", out)


def do_noisesweep():
    out = "/tmp/fonts_ns"; os.makedirs(out, exist_ok=True)
    for tag, fname in [(FAULT, "2D_Fig21_NoiseSweep_Matched_FaultCrop.png"),
                       (SECOND, "2D_Fig22_NoiseSweep_Matched_SecondCrop.png")]:
        with open(os.path.join(HERE, "NOISE_SWEEP_MATCHED_RESULTS", tag,
                               "noise_sweep_matched.json")) as f:
            j = json.load(f)
        ns.sweep_figure(j["rows"], tag, os.path.join(out, fname))
    print("noise-sweep done ->", out)


if __name__ == "__main__":
    do_tv()
    do_wav()
    do_noisesweep()
    print("\nRUNNER FIGURES REGENERATED")
