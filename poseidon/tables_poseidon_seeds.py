#!/usr/bin/env python3
"""
LaTeX rows of the per-seed field statistics (Supplementary Material): for each
withheld well and each network variant, the mean and the sample standard
deviation over the five training seeds of the envelope correlation, the
band-limited correlation and the relative RMSE under the primary alignment,
averaged over the inline and the crossline through the well (the per-line
values are in cases/*/phaseB/scores.json).

usage: python tables_poseidon_seeds.py --run RUN [--out rows.tex]
"""

import argparse
import glob
import json
import os

import numpy as np

WELLS = ["Proteus 1", "Pharos 1", "Poseidon 2", "Kronos 1", "Torosa 1"]
NETS = [("M3_unet_tv", "2D U-Net, 2-D input"), ("M4_rescnn_tv", "ResCNN, 2-D input"),
        ("M3_unet_rw", "2D U-Net, trace-wise input"), ("M4_rescnn_rw", "ResCNN, trace-wise input")]
METRICS = [("r_envelope", "envelope"), ("r_band", "band"), ("rel_rmse", "rel. RMSE")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    per = {}
    for sc in sorted(glob.glob(os.path.join(args.run, "cases", "*", "phaseB", "scores.json"))):
        d = json.load(open(sc))
        for line, ent in d["lines"].items():
            for key, m in ent["methods"].items():
                for met, _ in METRICS:
                    v = m.get(met)
                    if isinstance(v, dict):
                        per.setdefault((d["case"], key, met), []).append((v["mean"], v["std"]))
    lines = ["% ---- per-seed field statistics: Well & network & envelope (mean +- std over seeds) & band & rel. RMSE; each entry averaged over the two lines"]
    spread = {}
    for well in WELLS:
        for key, name in NETS:
            cells = []
            for met, _ in METRICS:
                vals = per.get((well, key, met))
                if not vals:
                    cells.append("--")
                    continue
                mean = float(np.mean([v[0] for v in vals]))
                std = float(np.mean([v[1] for v in vals]))
                spread.setdefault((key, met), []).append(std)
                cells.append(f"${mean:.2f} \\pm {std:.2f}$")
            lines.append(f"{well} & {name} & " + " & ".join(cells) + " \\\\")
    lines.append("% seed-std range per network and metric (averaged over the two lines of each well):")
    for (key, met), v in spread.items():
        lines.append(f"%   {key} {met}: {min(v):.3f} to {max(v):.3f}")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
