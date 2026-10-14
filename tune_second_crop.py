"""
Tuning of the two new settings of the practical variants on the SECOND crop only
(4500:4900 x 1000:1300, epsilon = 1e-3 data of run crop_second_e3):
  network input prior   : trace-wise RW-L1 (paper) or 2-D coupled ADMM/TV result
  low-band anchor       : weight W in {0, 0.3, 1, 3}, low-pass sigma in {12, 24} samples
U-Net, seed 42, 80 epochs, synthetic pretraining; label-free and two simulated wells
(crop traces 60 and 340); clean and noisy data.
Selection rule, fixed before running: the configuration with the highest mean SNR over the
four cases (clean/noisy x label-free/wells); the wells-only mean is reported as well.
The fault crop and the full model are not touched here.
"""
import os, sys, time, json, itertools
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, CODE)
import run_marmousi_full as rmf

# run folder of the second crop: first argument, else the laptop mirror
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CODE, "..", "7_athena", "results", "runs", "crop_second_e3")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
WELLS = [60, 340]
GRID = [("rw", 0.0, 12.0), ("tv", 0.0, 12.0)] + [(p, w, s) for p in ("rw", "tv") for w in (0.3, 1.0, 3.0) for s in (12.0, 24.0)]


def main():
    mod = rmf.load_main_module()
    mod.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    info = json.load(open(os.path.join(RUN, "prepared", "prepared.json")))
    nw, nt = info["nw"], info["nt"]
    P = {k: np.load(os.path.join(RUN, "prepared", k + ".npy")) for k in ("Z_true", "S_clean", "S_noisy", "Z_rw_clean", "Z_rw_noisy", "wavelet")}
    TV = {d: np.load(os.path.join(RUN, "baseline", f"Z_tv_{d}.npy")) for d in ("clean", "noisy")}
    Z = P["Z_true"]; nx = Z.shape[0]; L_true = np.log(Z); w = P["wavelet"]
    corners = mod.compute_patch_corners(nx, nt, mod.PATCH_NX, mod.PATCH_NT, mod.STRIDE_NX, mod.STRIDE_NT)
    mask = np.zeros((nx, nt)); mask[WELLS, :] = 1.0
    Lt_p = mod.extract_patches_at_corners(L_true, corners, mod.PATCH_NX, mod.PATCH_NT)
    M_p = mod.extract_patches_at_corners(mask, corners, mod.PATCH_NX, mod.PATCH_NT)
    snr = lambda a, b: 20 * np.log10(np.linalg.norm(b) / np.linalg.norm(a - b))
    rows = []
    os.makedirs(OUT, exist_ok=True)
    out_json = os.path.join(OUT, "tuning_second_crop.json")
    for prior, W, sig in GRID:
        for data in ("clean", "noisy"):
            S_trim = rmf.trim(P[f"S_{data}"], nw, nt)
            Zp = P[f"Z_rw_{data}"] if prior == "rw" else TV[data]
            L_prior = np.log(Zp + 1e-12)
            S_p = mod.extract_patches_at_corners(S_trim, corners, mod.PATCH_NX, mod.PATCH_NT)
            L0_p = mod.extract_patches_at_corners(L_prior, corners, mod.PATCH_NX, mod.PATCH_NT)
            for sup in ("none", "wells"):
                mod.W_LOWBAND, mod.LOWBAND_SIGMA = W, sig
                mod.set_seed(42); t = time.time()
                net, _ = mod.train_model_2d(S_p, L0_p, w, mod.PATCH_NX, mod.PATCH_NT, label=f"{prior}-lb{W:g}-{sup}",
                                            L_true_patches=None if sup == "none" else Lt_p,
                                            label_mask_patches=None if sup == "none" else M_p,
                                            model_class=mod.UNet2D, model_name="U-Net",
                                            transfer_checkpoint=os.path.join(CODE, "SYNTHETIC_PRETRAIN", f"2D_UNet_{data}_pretrain.pt"))
                Zn = rmf.predict_section_batched(mod, net, S_trim, L_prior, corners, nx, nt)
                r = dict(prior=prior, lowband=W, sigma=sig, data=data, sup=sup, prior_snr=snr(Zp, Z), snr=snr(Zn, Z), sec=time.time() - t)
                rows.append(r)
                print(f"prior {prior} lb {W:<3g} s {sig:<2g} {data:5s} {sup:5s}: prior {r['prior_snr']:.2f} -> {r['snr']:.2f} dB  [{r['sec']:.0f} s]", flush=True)
                json.dump(rows, open(out_json, "w"), indent=1)
    # ---- selection
    print("\nconfiguration scores (mean SNR over clean/noisy x none/wells; wells-only mean):")
    best = None
    lines = ["| prior | W_lowband | sigma | clean none | clean wells | noisy none | noisy wells | mean | wells mean |", "|---|---|---|---|---|---|---|---|---|"]
    for prior, W, sig in GRID:
        sel = [r for r in rows if r["prior"] == prior and r["lowband"] == W and r["sigma"] == sig]
        get = lambda d, s: next(r["snr"] for r in sel if r["data"] == d and r["sup"] == s)
        mean = np.mean([r["snr"] for r in sel]); wm = np.mean([r["snr"] for r in sel if r["sup"] == "wells"])
        lines.append(f"| {prior} | {W:g} | {sig:g} | {get('clean','none'):.2f} | {get('clean','wells'):.2f} | {get('noisy','none'):.2f} | {get('noisy','wells'):.2f} | {mean:.2f} | {wm:.2f} |")
        print(f"  prior {prior} lb {W:<3g} s {sig:<2g}: mean {mean:.2f}  wells {wm:.2f}")
        if best is None or mean > best[0]:
            best = (mean, prior, W, sig)
    print(f"\nSELECTED (rule: highest mean): prior {best[1]}, W_lowband {best[2]:g}, sigma {best[3]:g} (mean {best[0]:.2f} dB)")
    prior_rows = {d: (snr(P[f"Z_rw_{d}"], Z), snr(TV[d], Z)) for d in ("clean", "noisy")}
    with open(os.path.join(OUT, "tuning_second_crop.md"), "w") as f:
        f.write("# Tuning on the second crop (U-Net, seed 42, epsilon 1e-3)\n\n" + __doc__ + "\n")
        f.write(f"Priors: RW-L1 clean {prior_rows['clean'][0]:.2f} / noisy {prior_rows['noisy'][0]:.2f}; 2-D TV clean {prior_rows['clean'][1]:.2f} / noisy {prior_rows['noisy'][1]:.2f}.\n\n")
        f.write("\n".join(lines) + f"\n\nSelected: prior {best[1]}, W_lowband {best[2]:g}, sigma {best[3]:g} (mean {best[0]:.2f} dB).\n")


if __name__ == "__main__":
    main()
