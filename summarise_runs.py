#!/usr/bin/env python3
"""
Combined tables over several run folders of run_marmousi_full.py.

  starting-model table   one row per run folder: the starting model, its
                         SNR, the classical results and every network variant
                         (whole-model SNR, mean +/- std over seeds);
  label-count table      within one run folder: the pseudo-well variant
                         against the number of wells, next to the label-free
                         and dense rows.

usage: python summarise_runs.py --runs DIR [DIR ...] [--region whole] [--out summary.md]
"""

import argparse
import json
import os


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def starting_model_text(prep):
    sm = prep.get("starting_model")
    if sm is None:
        sx, st = prep.get("smooth_sigma", ["?", "?"])
        return f"Gaussian {sx:g} x {st:g}"
    if sm["kind"] == "gaussian":
        return (f"Gaussian {sm['sigma_x_traces']:g} x {sm['sigma_t_samples']:g} samples, "
                f"cut-off {sm['cutoff_hz']:.1f} Hz")
    return (f"{len(sm['well_traces'])} wells, low-pass {sm['lowpass_sigma_samples']:g} samples "
            f"({sm['cutoff_hz']:.1f} Hz)")


def fmt(stat):
    if stat is None or stat[0] is None:
        return ""
    m, sd = stat
    return f"{m:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")


def load_run(folder):
    report = read_json(os.path.join(folder, "report", "metrics.json"))
    prep = read_json(os.path.join(folder, "prepared", "prepared.json"))
    return report, prep


def starting_model_table(runs, region):
    labels = []
    for folder, (report, prep) in runs.items():
        for g in report["groups"]:
            for key in g["networks"]:
                if key not in labels:
                    labels.append(key)
    labels.sort()
    head = ["run", "starting model", "start SNR", "ADMM clean", "ADMM noisy", "2-D TV clean",
            "2-D TV noisy"] + labels
    lines = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
    for folder, (report, prep) in runs.items():
        cl = report["classical"]

        def c(name):
            v = cl.get(name, {}).get(region)
            return f"{v['snr_db']:.2f}" if v else ""
        row = [os.path.basename(folder.rstrip("/")), starting_model_text(prep), c("initial_model"),
               c("admm_clean"), c("admm_noisy"), c("tv2d_clean"), c("tv2d_noisy")]
        nets = {}
        for g in report["groups"]:
            for key, entry in g["networks"].items():
                nets[key] = fmt(entry["snr_db"].get(region))
        row += [nets.get(k, "") for k in labels]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def label_count_table(folder, report, region):
    rows = []
    for g in report["groups"]:
        for key, entry in g["networks"].items():
            net, data, sup = key.split("_", 2)
            if sup.startswith("wells"):
                digits = "".join(ch for ch in sup[5:].split("+")[0] if ch.isdigit())
                n = int(digits) if digits else 5   # records before the count was in the label: five wells
                order = (1, n)
            elif sup == "none":
                order = (0, 0)
            else:
                order = (2, 0)
            rows.append((order, net, data, sup, len(entry["seeds"]), fmt(entry["snr_db"].get(region))))
    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    lines = [f"Run `{os.path.basename(folder.rstrip('/'))}`, region `{region}`", "",
             "| network | data | supervision | seeds | SNR (dB) |", "|---|---|---|---:|---:|"]
    for _, net, data, sup, n, v in rows:
        lines.append(f"| {net} | {data} | {sup} | {n} | {v} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--region", default="whole")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = {}
    for folder in args.runs:
        try:
            runs[folder] = load_run(folder)
        except FileNotFoundError as e:
            print(f"skipping {folder}: {e}")
    text = [f"# Starting-model table, region `{args.region}`, whole-model SNR in dB", "",
            starting_model_table(runs, args.region), ""]
    for folder, (report, prep) in runs.items():
        if any(k.split("_", 2)[2].startswith("wells") for g in report["groups"] for k in g["networks"]):
            text += ["## Label count", "", label_count_table(folder, report, args.region), ""]
    text = "\n".join(text)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
