#!/usr/bin/env python3
"""
Manuscript figures of the Poseidon field example, drawn from the prepared data
and the sealed outputs of poseidon_runner.py (no recomputation).

  Fig_poseidon_map.png       the five wells and their lines on the survey grid
  Fig_poseidon_sections.png  one evaluation line (default: the inline through
                             Proteus 1): data, starting model, trace-wise and
                             2-D coupled inversions, the two networks (seed
                             mean), with the withheld well log inserted at the
                             well trace
  Fig_poseidon_wells.png     blind-well panels: the withheld log against every
                             method at the well trace, one panel per well
  Fig_poseidon_scores.png    pooled scores per method with the per-well spread
  figures.json               the numbers behind the figures

usage: python figures_poseidon.py --prepared DIR --run RUN --out OUT [--section-line Proteus_1_inline]
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.weight": "bold", "mathtext.default": "bf", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12, "axes.labelweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# the canvases are 15 to 21 inches wide and are printed at a two-column width, so the fonts are
# set for that reduction
BIG_CANVAS = {"font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15,
              "ytick.labelsize": 15, "legend.fontsize": 15}

WELLS = ["Proteus 1", "Pharos 1", "Poseidon 2", "Kronos 1", "Torosa 1"]
METHODS = [("B1_start", "Starting model", "0.5", ":"), ("M1_admm", "Trace-wise RW-$\\ell_1$ ADMM", "#7570b3", "--"),
           ("M2_tv", "2-D coupled ADMM", "#1b9e77", "-."), ("M3_unet_tv", "2D U-Net, 2-D input", "#d95f02", "-"),
           ("M4_rescnn_tv", "ResCNN, 2-D input", "#e7298a", "-")]


def tag(w):
    return w.replace(" ", "_")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def seed_mean_section(case_dir, line, network, inp="tv"):
    files = sorted(glob.glob(os.path.join(case_dir, "phaseA", f"net_{network}_{inp}_s*", line + ".npy")))
    if not files:
        return None, 0
    return np.mean(np.stack([np.load(f) for f in files]), axis=0), len(files)


def fig_map(prepared, out):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for w in WELLS:
        for kind, col in (("inline", "#1b9e77"), ("crossline", "#d95f02")):
            L = np.load(os.path.join(prepared, "lines", f"{tag(w)}_{kind}.npz"))
            ax.plot(L["cdp_x"] / 1000, L["cdp_y"] / 1000, color=col, lw=1.6, label=kind if w == WELLS[0] else None)
        d = np.load(os.path.join(prepared, "wells", f"{tag(w)}.npz"))
        ax.plot(float(d["cdp_x"]) / 1000, float(d["cdp_y"]) / 1000, "ko", ms=7)
        ax.annotate(w, (float(d["cdp_x"]) / 1000, float(d["cdp_y"]) / 1000), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Easting (km, MGA51)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Poseidon 3D: tied wells and the inversion lines")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "Fig_poseidon_map.png"), dpi=200)
    plt.close(fig)


def fig_sections(prepared, run, out, line):
    well = line.rsplit("_", 1)[0].replace("_", " ")
    case_dir = os.path.join(run, "cases", tag(well))
    Lz = np.load(os.path.join(case_dir, "phaseA", "lines", line + ".npz"))
    d = np.load(os.path.join(prepared, "wells", f"{tag(well)}.npz"))
    wt, t0, dt, nt = int(Lz["well_trace"]), float(Lz["t0"]), float(Lz["dt"]), int(Lz["nt"])
    nx = Lz["S"].shape[0]
    t = t0 + dt * np.arange(nt)
    panels = [("S", "(a) Near-stack data", None), ("L0", "(b) Starting model (flat, calibration wells)", "B1_start"),
              ("L_rw", "(c) Trace-wise RW-$\\ell_1$ ADMM", "M1_admm"), ("L_tv", "(d) 2-D coupled ADMM", "M2_tv")]
    nets = [("unet", "(e) 2D U-Net, 2-D input"), ("rescnn", "(f) ResCNN, 2-D input")]
    secs = {k: Lz[k].astype(np.float64) for k, _, _ in panels}
    for net, title in nets:
        sec, n = seed_mean_section(case_dir, line, net)
        if sec is not None:
            secs[net] = sec
            panels.append((net, f"{title} ({n} seeds)", None))
    lnz = d["lnz"]
    rows = np.flatnonzero(np.isfinite(lnz))
    zs = [np.exp(secs[k]) / 1e6 for k, _, _ in panels if k != "S"]
    vmin, vmax = np.percentile(np.concatenate([z[:, rows[0]:rows[-1]].ravel() for z in zs]), [2, 98])
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), layout="constrained")
    for ax, (k, title, _) in zip(axes.ravel(), panels):
        if k == "S":
            v = np.percentile(np.abs(secs["S"]), 99)
            im = ax.imshow(secs["S"].T, aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=[0, nx, t[-1] + dt, t[0]])
        else:
            Z = np.exp(secs[k]) / 1e6
            Zi = Z.copy()
            Zi[wt - 3:wt + 4, rows] = (np.exp(lnz[rows]) / 1e6)[None, :]          # the withheld log inserted at the well
            im = ax.imshow(Zi.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, extent=[0, nx, t[-1] + dt, t[0]])
        ax.axvline(wt, color="w" if k != "S" else "r", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("TWT (s)")
        ax.set_ylim(t[rows[-1]] + 0.15, max(t[0], t[rows[0]] - 0.15))
    fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), shrink=0.6, pad=0.01, label="Impedance (10$^6$ kg m$^{-2}$ s$^{-1}$)")
    fig.savefig(os.path.join(out, "Fig_poseidon_sections.png"), dpi=200)
    plt.close(fig)


def fig_wells(prepared, run, out, kind="inline"):
    fig, axes = plt.subplots(1, len(WELLS), figsize=(3.3 * len(WELLS), 8), sharey=False)
    numbers = {}
    for ax, w in zip(axes, WELLS):
        case_dir = os.path.join(run, "cases", tag(w))
        line = f"{tag(w)}_{kind}"
        path = os.path.join(case_dir, "phaseA", "lines", line + ".npz")
        if not os.path.exists(path):
            ax.set_title(f"{w}\n(not run)")
            continue
        Lz = np.load(path)
        d = np.load(os.path.join(prepared, "wells", f"{tag(w)}.npz"))
        wt, t0, dt, nt = int(Lz["well_trace"]), float(Lz["t0"]), float(Lz["dt"]), int(Lz["nt"])
        t = t0 + dt * np.arange(nt)
        lnz, tie = d["lnz"], d["tie_rows"]
        r = np.flatnonzero(tie)
        sl = slice(r[0], r[-1] + 1)
        ax.plot(np.exp(lnz[sl]) / 1e6, t[sl], "k", lw=1.4, label="withheld log")
        cols = {"B1_start": Lz["L0"][wt], "M1_admm": Lz["L_rw"][wt], "M2_tv": Lz["L_tv"][wt]}
        for net, key in (("unet", "M3_unet_tv"), ("rescnn", "M4_rescnn_tv")):
            sec, n = seed_mean_section(case_dir, line, net)
            if sec is not None:
                cols[key] = sec[wt]
        for key, name, col, ls in METHODS:
            if key in cols:
                ax.plot(np.exp(cols[key][sl]) / 1e6, t[sl], color=col, ls=ls, lw=1.2, label=name)
        ax.set_ylim(t[r[-1]], t[r[0]])
        ax.set_xlabel("AI (10$^6$ kg m$^{-2}$ s$^{-1}$)")
        ax.set_title(w)
        ax.grid(alpha=0.3)
        sc = os.path.join(case_dir, "phaseB", "scores.json")
        if os.path.exists(sc):
            m = read_json(sc)["lines"].get(line, {}).get("methods", {})
            numbers[line] = {k: (v["r_envelope"]["mean"] if isinstance(v.get("r_envelope"), dict) else v.get("r_envelope")) for k, v in m.items()}
    axes[0].set_ylabel("TWT (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(out, "Fig_poseidon_wells.png"), dpi=200)
    plt.close(fig)
    return numbers


def fig_scores(run, out):
    path = os.path.join(run, "report", "pooled.json")
    if not os.path.exists(path):
        return None
    pooled = read_json(path)["pooled"]
    metrics = [("r_envelope", "Envelope correlation"), ("r_band", "Band-limited correlation"), ("r_full", "Full-band correlation"),
               ("rel_rmse", "Relative RMSE of ln Z")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.2), layout="constrained")
    keys = [k for k, *_ in METHODS if k in pooled]
    for ax, (m, title) in zip(axes, metrics):
        for i, key in enumerate(keys):
            per = pooled[key]["per_line"]
            vals = [v[m] for v in per.values() if v[m] is not None and not (isinstance(v[m], float) and np.isnan(v[m]))]
            name, col = next((n, c) for k, n, c, _ in METHODS if k == key)
            ax.plot([i] * len(vals), vals, "o", color=col, alpha=0.45, ms=6)
            ax.plot([i - 0.25, i + 0.25], [pooled[key][m]] * 2, color=col, lw=3)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([next(n for k, n, *_ in METHODS if k == key).replace(", ", ",\n") for key in keys], rotation=60, ha="right", fontsize=13)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "Fig_poseidon_scores.png"), dpi=200)
    plt.close(fig)
    return {k: {m: pooled[k][m] for m, _ in metrics} for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--section-line", default="Proteus_1_inline")
    ap.add_argument("--all-sections", action="store_true", help="one section figure per line (supplement)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update(BIG_CANVAS)
    fig_map(args.prepared, args.out)
    rec = {"section_line": args.section_line}
    if os.path.exists(os.path.join(args.run, "cases", tag(args.section_line.rsplit("_", 1)[0].replace("_", " ")), "phaseA", "lines", args.section_line + ".npz")):
        fig_sections(args.prepared, args.run, args.out, args.section_line)
    if args.all_sections:
        for w in WELLS:
            for kind in ("inline", "crossline"):
                line = f"{tag(w)}_{kind}"
                if os.path.exists(os.path.join(args.run, "cases", tag(w), "phaseA", "lines", line + ".npz")):
                    fig_sections(args.prepared, args.run, args.out, line)
                    os.replace(os.path.join(args.out, "Fig_poseidon_sections.png"), os.path.join(args.out, f"Fig_poseidon_sections_{line}.png"))
        if os.path.exists(os.path.join(args.out, f"Fig_poseidon_sections_{args.section_line}.png")):
            import shutil
            shutil.copy(os.path.join(args.out, f"Fig_poseidon_sections_{args.section_line}.png"), os.path.join(args.out, "Fig_poseidon_sections.png"))
    rec["wells_r_envelope"] = fig_wells(args.prepared, args.run, args.out)
    rec["pooled"] = fig_scores(args.run, args.out)
    with open(os.path.join(args.out, "figures.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print("written to", args.out)


if __name__ == "__main__":
    main()
