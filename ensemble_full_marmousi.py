#!/usr/bin/env python3
"""
Ensemble rows for the complete-Marmousi-2 report: the mean over seeds of one
network (seed ensemble) and the mean over both networks and all seeds
(architecture ensemble), scored on the whole model and in the two windows,
from the stored predictions of run_marmousi_full.py. Nothing is retrained.

usage: python ensemble_full_marmousi.py --run RUN [--labels wells5+tv wells5 none dense ...]
"""
import argparse, glob, json, os
import numpy as np

def snr(Z, Zt, m=None):
    if m is not None:
        Z, Zt = Z[m], Zt[m]
    return 20 * np.log10(np.linalg.norm(Zt) / np.linalg.norm(Z - Zt))

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
ap.add_argument("--labels", nargs="*", default=["wells5+tv", "wells5", "none+tv", "none", "dense+tv", "dense", "wells2+tv", "wells10+tv", "wells20+tv"])
a = ap.parse_args()
m = json.load(open(os.path.join(a.run, "report", "metrics.json")))
regions = m["region_definitions"]
Zt = np.load(os.path.join(a.run, "prepared", "Z_true.npy"), mmap_mode="r")
masks = {}
for name in ("crop_fault", "crop_second"):
    x0, x1, t0, t1 = regions[name]; mk = np.zeros(Zt.shape, bool); mk[x0:x1, t0:t1] = True; masks[name] = mk
Zt = np.asarray(Zt)
out = {}
for data in ("clean", "noisy"):
    for lab in a.labels:
        per_net = {}
        for net in ("unet", "rescnn"):
            files = sorted(glob.glob(os.path.join(a.run, "train", f"{net}_{data}_{lab}_all_s*", "Z_pred.npy")))
            if not files:
                continue
            acc = None
            for f in files:
                Z = np.load(f).astype(np.float64)
                acc = Z if acc is None else acc + Z
            per_net[net] = (acc / len(files), len(files))
        if not per_net:
            continue
        rows = {}
        for net, (Zs, n) in per_net.items():
            rows[f"{net}_seed_ensemble"] = dict(seeds=n, whole=snr(Zs, Zt), crop_fault=snr(Zs, Zt, masks["crop_fault"]), crop_second=snr(Zs, Zt, masks["crop_second"]))
        if len(per_net) == 2:
            Za = 0.5 * (per_net["unet"][0] + per_net["rescnn"][0])
            rows["architecture_ensemble"] = dict(seeds=sum(n for _, n in per_net.values()), whole=snr(Za, Zt), crop_fault=snr(Za, Zt, masks["crop_fault"]), crop_second=snr(Za, Zt, masks["crop_second"]))
        out[f"{data}_{lab}"] = rows
        print(data, lab, {k: (round(v["whole"], 2), round(v["crop_fault"], 2), round(v["crop_second"], 2)) for k, v in rows.items()}, flush=True)
json.dump(out, open(os.path.join(a.run, "report", "ensemble.json"), "w"), indent=2)
print("written", os.path.join(a.run, "report", "ensemble.json"))
