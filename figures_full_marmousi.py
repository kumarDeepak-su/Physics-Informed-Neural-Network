#!/usr/bin/env python3
"""
Manuscript figures for the complete-Marmousi-2 section, drawn from the stored
outputs of run_marmousi_full.py (no recomputation, no retraining).

  Fig_full_sections.png   true model, starting model, trace-wise RW-L1 prior,
                          2-D coupled baseline, U-Net and ResCNN (sparse wells)
                          on the noisy data, with the two crop windows and the
                          well positions marked
  Fig_full_zoom.png       the same six panels inside the fault-zone window and
                          the second window
  Fig_full_errors.png     absolute error of the four inversions on one scale
  Fig_starting_model.png  whole-model SNR of every method against the SNR of
                          the starting model (one run folder per sweep point;
                          the five-well networks as one series per input prior)
  Fig_label_count.png     whole-model and unlabelled-column SNR against the
                          number of simulated wells (headline run)
  figures.json            every number written into the figures

usage: python figures_full_marmousi.py --run RUN --out OUT [--sweep RUN ...]
                                       [--data noisy] [--decimate 4 2]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams.update({
    "font.weight": "bold", "mathtext.default": "bf", "font.size": 13,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CLASSICAL = [("initial_model", "Starting model"), ("admm_{d}", "Trace-wise RW-$\\ell_1$ ADMM"),
             ("tv2d_{d}", "2-D coupled ADMM (TV + RW-$\\ell_1$)")]
NET_NAMES = {"unet": "2D U-Net", "rescnn": "ResCNN"}
# font sizes of the section figures, whose 16-inch canvases are printed at a two-column width
BIG_CANVAS = {"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16, "xtick.labelsize": 14,
              "ytick.labelsize": 14, "legend.fontsize": 14}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def snr_db(Z, Z_true):
    return 20.0 * np.log10(np.linalg.norm(Z_true) / (np.linalg.norm(Z - Z_true) + 1e-12))


def wells_label(network_keys, data):
    """The five-well variant shown in the section figures: the 2-D-prior variant when it exists,
    else the trace-wise-prior one ('wells5' in the current runs, 'wells' in the first run)."""
    for cand in ("wells5+tv", "wells5", "wells"):
        if any(k == f"unet_{data}_{cand}" or k == f"rescnn_{data}_{cand}" for k in network_keys):
            return cand
    raise KeyError("no five-well variant in " + ", ".join(sorted(network_keys)))


def figure_run_tag(run, metrics, network, data, label):
    """The seed shown in the section figures: the report's choice if it matches, else the
    seed whose whole-model SNR is closest to the seed mean."""
    want = f"{network}_{data}_{label}_all_"
    fr = metrics.get("figure_runs", {}).get(network)
    if fr and fr.startswith(want):
        return fr
    recs = []
    for rec in glob.glob(os.path.join(run, "train", want + "s*", "train_record.json")):
        r = read_json(rec)
        recs.append((r["metrics"]["whole"]["snr_db"], r["tag"]))
    if not recs:
        raise FileNotFoundError(want + "*")
    mean = np.mean([s for s, _ in recs])
    return min(recs, key=lambda x: abs(x[0] - mean))[1]


def load_sections(run, data, label, metrics):
    P = os.path.join(run, "prepared")
    out = {"true": np.load(os.path.join(P, "Z_true.npy")),
           "init": np.load(os.path.join(P, "Z_init.npy")),
           "prior": np.load(os.path.join(P, f"Z_rw_{data}.npy")),
           "tv": np.load(os.path.join(run, "baseline", f"Z_tv_{data}.npy"))}
    tags = {}
    for net in ("unet", "rescnn"):
        tag = figure_run_tag(run, metrics, net, data, label)
        out[net] = np.load(os.path.join(run, "train", tag, "Z_pred.npy"))
        tags[net] = tag
    return out, tags


def panel(ax, Z, title, vmin, vmax, dt, x0=0, t0=0, decimate=(1, 1)):
    dx, dtt = decimate
    Zd = Z[::dx, ::dtt]
    nx, nt = Z.shape
    im = ax.imshow(Zd.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[x0, x0 + nx, (t0 + nt) * dt, t0 * dt], interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Pseudo-time (s)")
    return im


def fig_sections(S, tags, info, regions, wells, dt, out, data, decimate):
    Zt = S["true"]
    vmin, vmax = np.percentile(Zt, [1, 99])
    order = [("true", "(a) True impedance"), ("init", "(b) Starting model"),
             ("prior", "(c) Trace-wise RW-$\\ell_1$ ADMM"), ("tv", "(d) 2-D coupled ADMM"),
             ("unet", "(e) 2D U-Net, five wells, 2-D prior"), ("rescnn", "(f) ResCNN, five wells, 2-D prior")]
    numbers = {}
    fig, axes = plt.subplots(3, 2, figsize=(16, 12.5), layout="constrained")
    for ax, (key, title) in zip(axes.ravel(), order):
        Z = S[key]
        if key != "true":
            s = snr_db(Z, Zt)
            numbers[key] = s
            title = f"{title}: SNR {s:.2f} dB"
        im = panel(ax, Z, title, vmin, vmax, dt, decimate=decimate)
        for name, letter in (("crop_fault", "A"), ("crop_second", "B")):
            x0, x1, t0, t1 = regions[name]
            ax.add_patch(Rectangle((x0, t0 * dt), x1 - x0, (t1 - t0) * dt, fill=False,
                                   edgecolor="white", linewidth=1.2))
            ax.text(x0, t0 * dt - 0.04, letter, color="white", fontsize=15, fontweight="bold",
                    ha="left", va="bottom", path_effects=[patheffects.withStroke(linewidth=2.5, foreground="black")])
        if key in ("unet", "rescnn"):
            for w in wells:
                ax.axvline(w, color="white", linestyle="--", linewidth=0.8)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("Impedance (m/s $\\cdot$ g/cm$^3$)")
    fig.savefig(os.path.join(out, "Fig_full_sections.png"), dpi=200)
    plt.close(fig)
    return numbers


def fig_zoom(S, regions, wells, dt, out):
    Zt = S["true"]
    order = [("true", "True"), ("init", "Starting model"), ("prior", "RW-$\\ell_1$ ADMM"),
             ("tv", "2-D coupled"), ("unet", "U-Net, wells"), ("rescnn", "ResCNN, wells")]
    numbers = {}
    fig, axes = plt.subplots(2, 6, figsize=(16, 6.9), layout="constrained")
    for row, name in enumerate(("crop_fault", "crop_second")):
        x0, x1, t0, t1 = regions[name]
        Zw = Zt[x0:x1, t0:t1]
        vmin, vmax = np.percentile(Zw, [1, 99])
        for col, (key, title) in enumerate(order):
            ax = axes[row, col]
            Z = S[key][x0:x1, t0:t1]
            lab = f"({'AB'[row]}{col + 1}) {title}"
            if key != "true":
                s = snr_db(Z, Zw)
                numbers[f"{name}/{key}"] = s
                lab += f"\nSNR {s:.2f} dB"
            im = panel(ax, Z, lab, vmin, vmax, dt, x0=x0, t0=t0)
            ax.title.set_fontsize(14)
            if col > 0:
                ax.set_ylabel("")
            for w in wells:
                if x0 <= w < x1 and key in ("unet", "rescnn"):
                    ax.axvline(w, color="white", linestyle="--", linewidth=0.8)
        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.9, pad=0.01, label="Impedance")
    fig.savefig(os.path.join(out, "Fig_full_zoom.png"), dpi=200)
    plt.close(fig)
    return numbers


def fig_errors(S, regions, dt, out, decimate):
    Zt = S["true"]
    order = [("prior", "(a) Trace-wise RW-$\\ell_1$ ADMM"), ("tv", "(b) 2-D coupled ADMM"),
             ("unet", "(c) 2D U-Net, five wells"), ("rescnn", "(d) ResCNN, five wells")]
    errs = {k: np.abs(S[k] - Zt) for k, _ in order}
    vmax = np.percentile(np.concatenate([e[::16, ::8].ravel() for e in errs.values()]), 99)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.4), layout="constrained")
    dx, dtt = decimate
    for ax, (key, title) in zip(axes.ravel(), order):
        E = errs[key][::dx, ::dtt]
        nx, nt = Zt.shape
        im = ax.imshow(E.T, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       extent=[0, nx, nt * dt, 0], interpolation="nearest")
        ax.set_title(f"{title}: RMSE {np.sqrt(np.mean(errs[key] ** 2)):.0f}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Pseudo-time (s)")
        for name in ("crop_fault", "crop_second"):
            x0, x1, t0, t1 = regions[name]
            ax.add_patch(Rectangle((x0, t0 * dt), x1 - x0, (t1 - t0) * dt, fill=False,
                                   edgecolor="white", linewidth=1.2))
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01, label="Absolute impedance error")
    fig.savefig(os.path.join(out, "Fig_full_errors.png"), dpi=200)
    plt.close(fig)


def net_snr(networks, key, region="whole"):
    e = networks.get(key)
    if e is None:
        return None, None
    v = e["snr_db"][region]
    if isinstance(v, (list, tuple)):
        return float(v[0]), (float(v[1]) if v[1] is not None else 0.0)
    return float(v), 0.0


def sweep_points(runs):
    pts = []
    for run in runs:
        mpath = os.path.join(run, "report", "metrics.json")
        if not os.path.exists(mpath):
            print(f"  sweep: no report in {run}, skipped")
            continue
        m = read_json(mpath)
        prep = read_json(os.path.join(run, "prepared", "prepared.json"))
        sm = prep.get("starting_model") or {"kind": "gaussian", "sigma_x_traces": prep["smooth_sigma"][0],
                                            "sigma_t_samples": prep["smooth_sigma"][1]}
        nets = m["groups"][0]["networks"] if m["groups"] else {}
        p = {"run": os.path.basename(run.rstrip("/")), "starting_model": sm,
             "start": m["classical"]["initial_model"]["whole"]["snr_db"]}
        for d in ("clean", "noisy"):
            p[f"prior_{d}"] = m["classical"][f"admm_{d}"]["whole"]["snr_db"]
            p[f"tv_{d}"] = m["classical"].get(f"tv2d_{d}", {}).get("whole", {}).get("snr_db")
            for net in ("unet", "rescnn"):
                # the two five-well variants are kept apart: trace-wise input ('wells5', or 'wells'
                # in the first run) and 2-D coupled input ('wells5+tv'); a run that lacks one of
                # them leaves that point empty instead of borrowing the other
                rw = next((f"{net}_{d}_{c}" for c in ("wells5", "wells") if f"{net}_{d}_{c}" in nets), None)
                tv = f"{net}_{d}_wells5+tv" if f"{net}_{d}_wells5+tv" in nets else None
                p[f"{net}_{d}_wells"] = net_snr(nets, rw)[0] if rw else None
                p[f"{net}_{d}_wells_tv"] = net_snr(nets, tv)[0] if tv else None
                p[f"{net}_{d}_none"] = net_snr(nets, f"{net}_{d}_none")[0]
                p[f"{net}_{d}_dense"] = net_snr(nets, f"{net}_{d}_dense")[0]
        pts.append(p)
    pts.sort(key=lambda p: p["start"])
    return pts


def sm_text(sm):
    if sm["kind"] == "gaussian":
        return f"{sm['sigma_x_traces']:g}$\\times${sm['sigma_t_samples']:g}"
    n = len(sm.get("well_traces") or []) or sm.get("n_wells", "")
    return f"{n} wells"


def fig_starting_model(pts, out):
    series = [("prior", "Trace-wise RW-$\\ell_1$ ADMM", "s", "#7570b3", "--"),
              ("tv", "2-D coupled ADMM", "D", "#1b9e77", "-."),
              ("unet_{d}_none", "2D U-Net, label-free, trace-wise input", "^", "#d95f02", ":"),
              ("rescnn_{d}_none", "ResCNN, label-free, trace-wise input", "v", "#e7298a", ":"),
              ("unet_{d}_wells", "2D U-Net, five wells, trace-wise input", "o", "#d95f02", "--"),
              ("rescnn_{d}_wells", "ResCNN, five wells, trace-wise input", "o", "#e7298a", "--"),
              ("unet_{d}_wells_tv", "2D U-Net, five wells, 2-D input", "s", "#d95f02", "-"),
              ("rescnn_{d}_wells_tv", "ResCNN, five wells, 2-D input", "s", "#e7298a", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout="constrained")
    x = [p["start"] for p in pts]
    for ax, d, letter in zip(axes, ("noisy", "clean"), "ab"):
        for key, name, mk, col, ls in series:
            k = key.format(d=d) if "{d}" in key else f"{key}_{d}"
            y = [p.get(k) for p in pts]
            xs = [xi for xi, yi in zip(x, y) if yi is not None]
            ys = [yi for yi in y if yi is not None]
            if ys:
                ax.plot(xs, ys, marker=mk, color=col, linestyle=ls, linewidth=1.6, markersize=7, label=name)
        lo, hi = min(x) - 1, max(x) + 1
        ax.plot([lo, hi], [lo, hi], color="0.5", linewidth=0.8, label="Output = starting model")
        ax.set_xlabel("Starting-model SNR (dB)")
        ax.set_ylabel("Whole-model SNR (dB)")
        ax.set_title(f"({letter}) {d.capitalize()} data")
        ax.grid(alpha=0.3)
        for p in pts:
            ax.annotate(sm_text(p["starting_model"]), (p["start"], lo + 0.4), fontsize=8, rotation=90,
                        ha="center", va="bottom", color="0.35")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.savefig(os.path.join(out, "Fig_starting_model.png"), dpi=200)
    plt.close(fig)


def label_count_points(run, metrics):
    """Rows: (network, data, n_wells or 0/'dense', whole SNR mean, std, unlabelled SNR mean, std, seeds)."""
    nets = metrics["groups"][0]["networks"]
    rows = []
    for key, e in nets.items():
        net, d, sup = key.split("_", 2)
        variant = "+tv" if sup.endswith("+tv") else ""
        base = sup[:-3] if variant else sup
        if "+" in base:
            continue
        if base == "none":
            n = 0
        elif base == "dense":
            n = "dense"
        else:
            mm = re.fullmatch(r"wells(\d*)", base)
            if not mm:
                continue
            n = int(mm.group(1)) if mm.group(1) else 5
        whole = net_snr(nets, key)
        unl = []
        for s in e["seeds"]:
            rec = os.path.join(run, "train", f"{key}_all_s{s}", "train_record.json")
            if os.path.exists(rec):
                r = read_json(rec)
                u = r["metrics"].get("unlabelled")
                if u:
                    unl.append(u["snr_db"])
        rows.append({"network": net, "data": d, "variant": variant, "n_wells": n, "whole": whole,
                     "unlabelled": (float(np.mean(unl)), float(np.std(unl, ddof=1)) if len(unl) > 1 else 0.0) if unl else None,
                     "seeds": e["seeds"]})
    return rows


def fig_label_count(rows, classical, out):
    """Whole-model SNR against the number of simulated wells, one series per network and input
    prior (trace-wise RW-L1 result or 2-D coupled result), with the classical references: the
    trace-wise and 2-D coupled inversions (lines) and, where present, the 2-D coupled inversion
    conditioned on the same wells (markers)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    cols = {"unet": "#d95f02", "rescnn": "#e7298a"}
    styles = {"": ("--", "o", "trace-wise prior"), "+tv": ("-", "s", "2-D prior")}
    for ax, d, letter in zip(axes, ("noisy", "clean"), "ab"):
        sub = [r for r in rows if r["data"] == d]
        ns = sorted({r["n_wells"] for r in sub if r["n_wells"] != "dense"})
        has_dense = any(r["n_wells"] == "dense" for r in sub)
        xs = list(range(len(ns))) + ([len(ns)] if has_dense else [])
        cats = ns + (["dense"] if has_dense else [])
        labels = [str(n) for n in ns] + (["dense"] if has_dense else [])
        for net in ("unet", "rescnn"):
            for variant, (ls, mk, vname) in styles.items():
                pts = []
                for xi, n in zip(xs, cats):
                    r = next((r for r in sub if r["network"] == net and r["n_wells"] == n
                              and r["variant"] == variant), None)
                    if r is not None:
                        pts.append((xi, r["whole"][0], r["whole"][1]))
                if pts:
                    ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                                color=cols[net], linestyle=ls, marker=mk, capsize=3, linewidth=1.6,
                                markersize=6, label=f"{NET_NAMES[net]}, {vname}")
        for key, name, col, ls in (("admm", "Trace-wise RW-$\\ell_1$ ADMM", "#7570b3", "--"),
                                   ("tv2d", "2-D coupled ADMM", "#1b9e77", "-.")):
            v = classical.get(f"{key}_{d}", {}).get("whole", {}).get("snr_db")
            if v is not None:
                ax.axhline(v, color=col, linestyle=ls, linewidth=1.4, label=name)
        comp = []
        for key, e in classical.items():
            mm = re.fullmatch(rf"tv2d_wells(\d+)_{d}", key)
            if mm and int(mm.group(1)) in ns:
                comp.append((xs[ns.index(int(mm.group(1)))], e["whole"]["snr_db"]))
        if comp:
            comp.sort()
            ax.plot([c[0] for c in comp], [c[1] for c in comp], color="#1b9e77", linestyle="none",
                    marker="D", markersize=8, label="2-D coupled ADMM with the same wells")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Number of simulated wells (0 = label-free)")
        ax.set_ylabel("Whole-model SNR (dB)")
        ax.set_title(f"({letter}) {d.capitalize()} data")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8.5)
    fig.savefig(os.path.join(out, "Fig_label_count.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="headline run folder (prepared/, baseline/, train/, report/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", nargs="*", default=[], help="run folders of the starting-model sweep (the headline run is added)")
    ap.add_argument("--data", choices=["clean", "noisy"], default="noisy")
    ap.add_argument("--decimate", nargs=2, type=int, default=[4, 2], metavar=("TRACES", "SAMPLES"),
                    help="display decimation of the whole-model panels (metrics are not affected)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metrics = read_json(os.path.join(args.run, "report", "metrics.json"))
    prep = read_json(os.path.join(args.run, "prepared", "prepared.json"))
    dt = float(prep["dt"])
    regions = metrics["region_definitions"]
    nets = metrics["groups"][0]["networks"]
    label = wells_label(nets.keys(), args.data)
    S, tags = load_sections(args.run, args.data, label, metrics)
    wells = read_json(os.path.join(args.run, "train", tags["unet"], "train_record.json")).get("well_traces") or []
    record = {"run": args.run, "data": args.data, "wells_label": label, "figure_tags": tags, "well_traces": wells,
              "decimation": args.decimate}
    print(f"sections from {tags} (wells at {wells})", flush=True)
    with plt.rc_context(BIG_CANVAS):
        record["sections"] = fig_sections(S, tags, prep, regions, wells, dt, args.out, args.data, tuple(args.decimate))
        record["zoom"] = fig_zoom(S, regions, wells, dt, args.out)
        fig_errors(S, regions, dt, args.out, tuple(args.decimate))
    del S

    sweep = list(dict.fromkeys(list(args.sweep) + [args.run]))
    pts = sweep_points(sweep)
    if len(pts) >= 2:
        fig_starting_model(pts, args.out)
        record["starting_model"] = pts
        print(f"starting-model figure from {len(pts)} runs", flush=True)
    rows = label_count_points(args.run, metrics)
    if len({r["n_wells"] for r in rows}) >= 3:
        fig_label_count(rows, metrics["classical"], args.out)
        record["label_count"] = rows
        print(f"label-count figure from {len(rows)} rows", flush=True)
    with open(os.path.join(args.out, "figures.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print("written to", args.out)


if __name__ == "__main__":
    main()
