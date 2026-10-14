#!/usr/bin/env python3
"""
Bounded diagnosis of the label-free objective on one crop (run folder of
run_marmousi_full.py: prepared/ and baseline/ must exist).

  A  zero-step reference: the U-Net's output before any training on the
     target (random weights, and the synthetic pretraining checkpoint),
     scored against the prior it was given;
  B  direct optimisation: the log-impedance section itself is optimised under
     the same label-free objective (physics + sparsity + prior anchor +
     lateral term) with the same optimiser, schedule and reweighting as the
     network; if this also degrades the prior the objective is the cause,
     if not the network/optimisation is;
  C  mechanism log of the network training: the weighted loss terms, the
     gradient norm, the SNR of the full prediction and the low-frequency
     drift from the prior after every epoch;
  D  one principal correction and one control, both in B and C: the low-band
     anchor (weight 1, sigma 12) as the correction, a constant full-band
     anchor (w_init fixed at 0.5) as the control.

usage: python diagnose_objective.py --run <folder> --out <folder> [--data noisy]
                                    [--prior rw|tv] [--network unet] [--epochs 80]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_marmousi_full as rmf  # noqa: E402


def snr(a, b):
    return 20 * np.log10(np.linalg.norm(b) / np.linalg.norm(a - b))


def lowband_drift(L_pred, L_prior, sigma):
    """RMS of the time low-pass of (prediction - prior): the drift below the band."""
    from scipy.ndimage import gaussian_filter1d
    d = gaussian_filter1d(L_pred - L_prior, sigma, axis=1, mode="nearest")
    return float(np.sqrt(np.mean(d ** 2)))


def direct_optimisation(mod, S_trim, L_prior, wavelet, epochs, steps_per_epoch, lowband, fixed_anchor):
    """
    Optimise the log-impedance L directly under the label-free objective of
    train_model_2d (same weights, Adam, cosine schedule, w_init anneal,
    reweighting every REWEIGHT_EVERY epochs), on the whole section.
    """
    import torch.nn.functional as F
    dev = mod.DEVICE
    S_t = torch.tensor(S_trim, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    L0_t = torch.tensor(L_prior, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    L = L0_t.clone().requires_grad_(True)
    physics = mod.PhysicsLayer2D(wavelet).to(dev)
    optim = torch.optim.Adam([L], lr=mod.LR, weight_decay=0.0)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=mod.LR_MIN)
    rw = torch.ones(1, 1, 1, L_prior.shape[1] - 1, device=dev)
    kernel = mod.lowpass_kernel(mod.LOWBAND_SIGMA, dev) if lowband > 0 else None
    history = []
    for epoch in range(epochs):
        if fixed_anchor is not None:
            w_init = fixed_anchor
        elif epoch < mod.WARMUP_EPOCHS:
            w_init = mod.W_INIT_START - (mod.W_INIT_START - mod.W_INIT_END) * epoch / mod.WARMUP_EPOCHS
        else:
            w_init = mod.W_INIT_END
        for _ in range(steps_per_epoch):
            S_pred, r_pred = physics(L)
            loss_phys = F.mse_loss(S_pred, S_t)
            loss_sparse = torch.mean(rw * torch.abs(r_pred))
            loss_init = F.mse_loss(L, L0_t)
            loss_lat = torch.mean(torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :]))
            loss_low = mod.lowband_mse(L - L0_t, kernel) if kernel is not None else torch.tensor(0.0, device=dev)
            loss = (mod.W_PHYSICS * loss_phys + mod.W_SPARSE * loss_sparse + w_init * loss_init
                    + mod.W_LATERAL * loss_lat + lowband * loss_low)
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()
        if (epoch + 1) % mod.REWEIGHT_EVERY == 0 and epoch < epochs - 1:
            with torch.no_grad():
                _, r_p = physics(L)
                mean_abs = torch.mean(torch.abs(r_p), dim=(0, 1, 2), keepdim=True)
                rw = 1.0 / (mean_abs + mod.EPSILON_RW)
                rw = rw / rw.mean()
        with torch.no_grad():
            Lnp = L[0, 0].cpu().numpy().astype(np.float64)
        history.append(dict(epoch=epoch, physics=float(mod.W_PHYSICS * loss_phys), sparse=float(mod.W_SPARSE * loss_sparse),
                            init=float(w_init * loss_init), lateral=float(mod.W_LATERAL * loss_lat),
                            lowband=float(lowband * loss_low), snr=None, L=Lnp))
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", choices=["clean", "noisy"], default="noisy")
    p.add_argument("--prior", choices=["rw", "tv"], default="rw")
    p.add_argument("--network", choices=["unet", "rescnn"], default="unet")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--transfer-dir", default=os.path.join(HERE, "SYNTHETIC_PRETRAIN"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mod = rmf.load_main_module()
    mod.DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    info, P = rmf.load_prepared(args.run)
    nx, nt, nw = info["nx"], info["nt"], info["nw"]
    Z_true = np.asarray(P["Z_true"])
    S_trim = rmf.trim(P[f"S_{args.data}"], nw, nt)
    wavelet = np.asarray(P["wavelet"])
    if args.prior == "tv":
        Zp = np.load(os.path.join(args.run, "baseline", f"Z_tv_{args.data}.npy"))
    else:
        Zp = np.asarray(P[f"Z_rw_{args.data}"])
    L_prior = np.log(Zp + 1e-12)
    corners = mod.compute_patch_corners(nx, nt, mod.PATCH_NX, mod.PATCH_NT, mod.STRIDE_NX, mod.STRIDE_NT)
    S_p = mod.extract_patches_at_corners(S_trim, corners, mod.PATCH_NX, mod.PATCH_NT)
    L0_p = mod.extract_patches_at_corners(L_prior, corners, mod.PATCH_NX, mod.PATCH_NT)
    class_name, model_name, prefix = rmf.NETWORKS[args.network]
    ckpt = os.path.join(args.transfer_dir, f"{prefix}_{args.data}_pretrain.pt")
    mod.EPOCHS = args.epochs
    result = dict(run=args.run, data=args.data, prior=args.prior, network=args.network,
                  prior_snr=snr(Zp, Z_true), epochs=args.epochs)
    print(f"prior ({args.prior}, {args.data}): {result['prior_snr']:.2f} dB", flush=True)

    # ---- A: zero-step reference
    mod.set_seed(args.seed)
    net = getattr(mod, class_name)(mod.PATCH_NX, mod.PATCH_NT, base_ch=mod.UNET_BASE_CH).to(mod.DEVICE)
    Z0 = rmf.predict_section_batched(mod, net, S_trim, L_prior, corners, nx, nt)
    result["A_zero_step_random_weights"] = snr(Z0, Z_true)
    mod.load_compatible_checkpoint(net, ckpt)
    Z0 = rmf.predict_section_batched(mod, net, S_trim, L_prior, corners, nx, nt)
    result["A_zero_step_pretrained"] = snr(Z0, Z_true)
    result["A_zero_step_pretrained_drift"] = lowband_drift(np.log(Z0), L_prior, 12.0)
    print(f"A  zero-step output: random weights {result['A_zero_step_random_weights']:.2f} dB, "
          f"synthetic checkpoint {result['A_zero_step_pretrained']:.2f} dB "
          f"(low-band drift {result['A_zero_step_pretrained_drift']:.4f})", flush=True)

    # ---- B: direct optimisation of the section under the same objective
    steps_per_epoch = max(1, int(np.ceil(len(corners) / mod.BATCH_SIZE)))
    result["B_direct"] = {}
    for name, lowband, fixed in (("objective_as_trained", 0.0, None),
                                 ("control_constant_anchor_0.5", 0.0, 0.5),
                                 ("correction_lowband_1", 1.0, None)):
        mod.set_seed(args.seed)
        t = time.time()
        hist = direct_optimisation(mod, S_trim, L_prior, wavelet, args.epochs, steps_per_epoch, lowband, fixed)
        curve = [snr(np.exp(h["L"]), Z_true) for h in hist]
        drift = [lowband_drift(h["L"], L_prior, 12.0) for h in hist]
        result["B_direct"][name] = dict(final_snr=curve[-1], best_snr=max(curve), best_epoch=int(np.argmax(curve)),
                                        snr_curve=curve, lowband_drift=drift,
                                        terms_final={k: hist[-1][k] for k in ("physics", "sparse", "init", "lateral", "lowband")},
                                        seconds=time.time() - t)
        print(f"B  direct optimisation, {name}: final {curve[-1]:.2f} dB (best {max(curve):.2f} at epoch "
              f"{int(np.argmax(curve))}), low-band drift {drift[-1]:.4f}  [{time.time() - t:.0f} s]", flush=True)

    # ---- C/D: network training with the mechanism log
    result["C_network"] = {}
    for name, lowband, w_end in (("objective_as_trained", 0.0, None),
                                 ("control_constant_anchor_0.5", 0.0, 0.5),
                                 ("correction_lowband_1", 1.0, None)):
        mod.W_LOWBAND, mod.LOWBAND_SIGMA = lowband, 12.0
        start_save, end_save = mod.W_INIT_START, mod.W_INIT_END
        if w_end is not None:
            mod.W_INIT_START = mod.W_INIT_END = w_end
        log = []

        def monitor(epoch, terms, net_, log=log):
            Zp_ = rmf.predict_section_batched(mod, net_, S_trim, L_prior, corners, nx, nt)
            log.append(dict(epoch=epoch, snr=snr(Zp_, Z_true), drift=lowband_drift(np.log(Zp_), L_prior, 12.0), **terms))

        mod.set_seed(args.seed)
        t = time.time()
        net, _ = mod.train_model_2d(S_p, L0_p, wavelet, mod.PATCH_NX, mod.PATCH_NT, label=name,
                                    model_class=getattr(mod, class_name), model_name=model_name,
                                    transfer_checkpoint=ckpt, monitor=monitor)
        mod.W_INIT_START, mod.W_INIT_END = start_save, end_save
        curve = [e["snr"] for e in log]
        result["C_network"][name] = dict(final_snr=curve[-1], best_snr=max(curve), best_epoch=int(np.argmax(curve)),
                                         log=log, seconds=time.time() - t)
        e = log[-1]
        print(f"C  network, {name}: final {curve[-1]:.2f} dB (best {max(curve):.2f} at epoch {int(np.argmax(curve))}); "
              f"last epoch terms physics {e['physics']:.2e} sparse {e['sparse']:.2e} init {e['init']:.2e} "
              f"lateral {e['lateral']:.2e} lowband {e['lowband']:.2e}, grad norm {e['grad_norm']:.3f}, "
              f"low-band drift {e['drift']:.4f}  [{time.time() - t:.0f} s]", flush=True)

    tag = f"{os.path.basename(args.run.rstrip('/'))}_{args.data}_{args.prior}_{args.network}"
    rmf.write_json(os.path.join(args.out, f"diagnosis_{tag}.json"), result)
    print(f"written {os.path.join(args.out, f'diagnosis_{tag}.json')}")


if __name__ == "__main__":
    main()
