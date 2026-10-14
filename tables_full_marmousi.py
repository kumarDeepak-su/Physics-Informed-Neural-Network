#!/usr/bin/env python3
"""
LaTeX table rows for the complete-Marmousi-2 section, generated from the
report of run_marmousi_full.py (report/metrics.json and the training records)
so that no number is transcribed by hand.

  label-count table   whole-model SNR against the number of simulated wells,
                      clean / noisy, one row per network and input prior, with
                      the classical references and the well-conditioned
                      comparator;
  region table        whole model, fault-zone window and second window for the
                      main variants, clean / noisy;
  generalisation      whole-model and held-out SNR of the subset-trained
                      networks.

usage: python tables_full_marmousi.py --run RUN [--out tables.tex]
"""

import argparse
import json
import os
import re


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def snr(entry, region="whole"):
    v = entry["snr_db"][region]
    if isinstance(v, (list, tuple)):
        return float(v[0]), (float(v[1]) if v[1] is not None else None), len(entry.get("seeds", []))
    return float(v), None, 1


def fmt(mean, std=None, pm=True):
    if mean is None:
        return "--"
    if std is None or not pm:
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def cell(nets, key_clean, key_noisy, region="whole", pm=False):
    parts = []
    for key in (key_clean, key_noisy):
        e = nets.get(key)
        if e is None:
            parts.append("--")
        else:
            m, s, n = snr(e, region)
            parts.append(fmt(m, s if n > 1 else None, pm))
    return " / ".join(parts)


def parse_key(key):
    net, data, sup = key.split("_", 2)
    variant = "tv" if sup.endswith("+tv") else "rw"
    base = sup[:-3] if variant == "tv" else sup
    if "+" in base:
        return None
    if base == "none":
        n = 0
    elif base == "dense":
        n = "dense"
    else:
        mm = re.fullmatch(r"wells(\d*)", base)
        if not mm:
            return None
        n = int(mm.group(1)) if mm.group(1) else 5
    return net, data, variant, n


def label_count_table(m):
    nets = {}
    for g in m["groups"]:
        if g["train_set"] == "all":
            nets = g["networks"]
    cls = m["classical"]
    counts = sorted({p[3] for p in map(parse_key, nets) if p and p[3] != "dense"})
    cols = counts + ["dense"]
    head = " & ".join(["Method, input"] + [("label-free" if c == 0 else f"{c} wells" if c != "dense" else "dense") for c in cols])
    lines = [f"% columns: {head}"]

    def classical_row(name, key):
        c = cls.get(f"{key}_clean", {}).get("whole", {}).get("snr_db")
        n = cls.get(f"{key}_noisy", {}).get("whole", {}).get("snr_db")
        return f"{name} & " + " & ".join([f"{c:.2f} / {n:.2f}" if c is not None and n is not None else "--"] + ["" for _ in cols[1:]]) + " \\\\"

    lines.append(classical_row("Trace-wise RW-$\\ell_1$ ADMM", "admm"))
    lines.append(classical_row("2-D coupled ADMM", "tv2d"))
    comp = []
    for c in cols:
        if c == 0:      # no well labels: the constraint term is absent, the comparator is the plain 2-D baseline
            cc = cls.get("tv2d_clean", {}).get("whole", {}).get("snr_db")
            nn = cls.get("tv2d_noisy", {}).get("whole", {}).get("snr_db")
            comp.append(f"{cc:.2f} / {nn:.2f}$^\\dagger$" if cc is not None and nn is not None else "--")
            continue
        if c == "dense":  # no densely constrained classical run is made
            comp.append("NR$^\\ddagger$")
            continue
        cc = cls.get(f"tv2d_wells{c}_clean", {}).get("whole", {}).get("snr_db")
        nn = cls.get(f"tv2d_wells{c}_noisy", {}).get("whole", {}).get("snr_db")
        comp.append(f"{cc:.2f} / {nn:.2f}" if cc is not None and nn is not None else "--")
    lines.append("2-D coupled ADMM with well constraints & " + " & ".join(comp) + " \\\\")
    lines.append("\\midrule")
    names = {"unet": "2D U-Net", "rescnn": "ResCNN"}
    variants = {"rw": "trace-wise prior", "tv": "2-D prior"}
    for net in ("unet", "rescnn"):
        for var, vname in variants.items():
            cells = []
            for c in cols:
                sup = "none" if c == 0 else "dense" if c == "dense" else f"wells{c}"
                suffix = "+tv" if var == "tv" else ""
                k_c, k_n = f"{net}_clean_{sup}{suffix}", f"{net}_noisy_{sup}{suffix}"
                if k_c not in nets and k_n not in nets and c == 5:
                    k_c, k_n = f"{net}_clean_wells{suffix}", f"{net}_noisy_wells{suffix}"
                cells.append(cell(nets, k_c, k_n))
            if all(x in ("--", "-- / --") for x in cells):
                continue
            lines.append(f"{names[net]}, {vname} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def region_table(m, regions=("whole", "crop_fault", "crop_second")):
    nets = {}
    for g in m["groups"]:
        if g["train_set"] == "all":
            nets = g["networks"]
    cls = m["classical"]
    lines = ["% columns: Method & " + " & ".join(regions) + " (clean / noisy)"]

    def crow(name, key):
        cells = []
        for r in regions:
            c = cls.get(f"{key}_clean", {}).get(r, {}).get("snr_db")
            n = cls.get(f"{key}_noisy", {}).get(r, {}).get("snr_db")
            cells.append(f"{c:.2f} / {n:.2f}" if c is not None and n is not None else "--")
        return f"{name} & " + " & ".join(cells) + " \\\\"

    init = [f"{cls['initial_model'][r]['snr_db']:.2f}" for r in regions]
    lines.append("Starting model & " + " & ".join(init) + " \\\\")
    lines.append(crow("Trace-wise RW-$\\ell_1$ ADMM", "admm"))
    lines.append(crow("2-D coupled ADMM", "tv2d"))
    if "tv2d_wells5_clean" in cls:
        lines.append(crow("2-D coupled ADMM with five wells", "tv2d_wells5"))
    lines.append("\\midrule")
    names = {"unet": "2D U-Net", "rescnn": "ResCNN"}
    for net in ("unet", "rescnn"):
        for sup, sname in (("none", "label-free"), ("wells5", "five wells"), ("dense", "dense labels")):
            for var, vname in (("", "trace-wise prior"), ("+tv", "2-D prior")):
                k_c, k_n = f"{net}_clean_{sup}{var}", f"{net}_noisy_{sup}{var}"
                if k_c not in nets and sup == "wells5":
                    k_c, k_n = f"{net}_clean_wells{var}", f"{net}_noisy_wells{var}"
                if k_c not in nets and k_n not in nets:
                    continue
                cells = [cell(nets, k_c, k_n, r) for r in regions]
                lines.append(f"{names[net]}, {sname}, {vname} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def generalisation_table(m):
    g = next((g for g in m["groups"] if g["train_set"] == "subset"), None)
    if g is None:
        return "% no subset group"
    nets = g["networks"]
    lines = [f"% subset: {g['n_patches']} patches, {100 * g['held_out_fraction']:.1f} % of samples held out",
             "% columns: Method & whole (clean / noisy) & held out (clean / noisy)"]
    names = {"unet": "2D U-Net", "rescnn": "ResCNN"}
    for net in ("unet", "rescnn"):
        for sup, sname in (("none", "label-free"), ("wells5", "five wells"), ("dense", "dense labels")):
            for var, vname in (("", "trace-wise prior"), ("+tv", "2-D prior")):
                k_c, k_n = f"{net}_clean_{sup}{var}", f"{net}_noisy_{sup}{var}"
                if k_c not in nets and k_n not in nets:
                    continue
                lines.append(f"{names[net]}, {sname}, {vname} & {cell(nets, k_c, k_n, 'whole')} & "
                             f"{cell(nets, k_c, k_n, 'held_out')} \\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m = read_json(os.path.join(args.run, "report", "metrics.json"))
    text = "\n".join(["% ---- label-count table", label_count_table(m), "",
                      "% ---- region table", region_table(m), "",
                      "% ---- generalisation", generalisation_table(m), ""])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
