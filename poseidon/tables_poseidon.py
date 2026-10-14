#!/usr/bin/env python3
"""
LaTeX rows of the field-example tables from the sealed scores of
poseidon_runner.py (report/pooled.json and cases/*/phaseB/scores.json).

  pooled table   one row per method: envelope, band-limited (primary and
                 own-lag sensitivity), low-pass, relative RMSE and full-band
                 scores, row-weighted over the ten evaluation lines, with the
                 median spatial-null p of the envelope score;
  per-well table one row per well and method (mean of the two lines).

usage: python tables_poseidon.py --run RUN [--out rows.tex]
"""

import argparse
import glob
import json
import os

import numpy as np

WELLS = ["Proteus 1", "Pharos 1", "Poseidon 2", "Kronos 1", "Torosa 1"]
METHODS = [("B1_start", "Starting model (flat)"), ("M1_admm", "Trace-wise RW-$\\ell_1$ ADMM"), ("M2_tv", "2D-coupled ADMM"),
           ("M3_unet_tv", "2D U-Net, 2-D input, four wells"), ("M4_rescnn_tv", "ResCNN, 2-D input, four wells"),
           ("M5_ensemble_tv", "Ensemble of the two networks, 2-D input (derived)"),
           ("M3_unet_rw", "2D U-Net, trace-wise input, four wells"), ("M4_rescnn_rw", "ResCNN, trace-wise input, four wells")]


def f3(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pooled = json.load(open(os.path.join(args.run, "report", "pooled.json")))["pooled"]
    lines = ["% ---- pooled table: Method & envelope & envelope (VSP lag) & band & band (VSP lag) & band (own lag) & low & rel. RMSE & rel. RMSE (VSP lag) & full & p env (median)"]
    for key, name in METHODS:
        if key not in pooled:
            continue
        p = pooled[key]
        lines.append(f"{name} & {f3(p['r_envelope'])} & {f3(p.get('r_envelope_vsp'))} & {f3(p['r_band'])} & {f3(p.get('r_band_vsp'))} & {f3(p.get('r_band_ownlag'))} & "
                     f"{f3(p['r_low'])} & {f3(p['rel_rmse'])} & {f3(p.get('rel_rmse_vsp'))} & {f3(p['r_full'])} & {p['p_env_median']:.2f} \\\\")
    lines += ["", "% ---- per-well table (mean of the two lines): Well & rows & lag & VSP lag & method: envelope / band / band (VSP lag) / rel. RMSE (VSP lag)"]
    per = {}
    for sc in sorted(glob.glob(os.path.join(args.run, "cases", "*", "phaseB", "scores.json"))):
        d = json.load(open(sc))
        well = d["case"]
        for line, ent in d["lines"].items():
            for key, m in ent["methods"].items():
                vals = {k: (m[k]["mean"] if isinstance(m[k], dict) else m[k]) for k in ("r_envelope", "r_band", "rel_rmse", "r_full")}
                mv = lambda x: x["mean"] if isinstance(x, dict) else x      # networks: mean over seeds; single sections: the value
                vals["r_band_ownlag"] = mv(m.get("own_lag", {}).get("r_band"))
                vals["r_band_vsp"] = mv(m.get("vsp_lag", {}).get("r_band"))
                vals["rel_rmse_vsp"] = mv(m.get("vsp_lag", {}).get("rel_rmse"))
                per.setdefault(well, {}).setdefault(key, []).append(vals)
        per[well]["_rows"] = d["rows"][1] - d["rows"][0]
        per[well]["_lag"] = d["lag_declared_ms"]
        per[well]["_vsp"] = d.get("vsp_alignment_ms")
    for well in WELLS:
        if well not in per:
            continue
        cells = []
        for key, _ in METHODS[:6]:
            v = per[well].get(key)
            if not v:
                cells.append("--")
                continue
            mean = lambda k: float(np.mean([x[k] for x in v if x[k] is not None])) if any(x[k] is not None for x in v) else None
            cells.append(f"{f3(mean('r_envelope'))} / {f3(mean('r_band'))} / {f3(mean('r_band_vsp'))} / {f3(mean('rel_rmse_vsp'))}")
        vsp = per[well]["_vsp"]
        lines.append(f"{well} & {per[well]['_rows']} & {per[well]['_lag']:+.1f} & {vsp:+.1f} & " + " & ".join(cells) + " \\\\")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
