#!/usr/bin/env python3
"""
Checks that the banded solvers and the batched predictor of the full-model
runner give the published crop results, then runs the whole runner on a small
slice of the model.

  T1  admm_banded.admm_l1_traces      vs published Z_rw_clean / Z_rw_noisy
  T2  admm_banded.admm_l1_2d_tv_banded vs published Z_tv_clean / Z_tv_noisy
  T3  predict_section_batched          vs predict_full_section (same checkpoint)
  T4  admm_banded.forward_model        vs the per-trace convolution loop
  T5  prepare / train / baseline / report on a 600-trace slice

usage: python test_marmousi_full.py [--crop marmousi_fault_crop_400x300]
                                    [--marmousi-dir ../data/marmousi2] [--skip-t5]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import admm_banded  # noqa: E402
import run_marmousi_full as rmf  # noqa: E402


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def verdict(name, value, limit):
    ok = value < limit
    print(f"  {name:58s} {value:9.2e}  (limit {limit:.0e})  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_t5(args):
    """Runner stages on a 600-trace slice; returns True when every output exists."""
    ok = True
    print("T5  runner stages on a 600-trace slice (2 epochs; subset and all-patch training sets)")
    out = os.path.join(HERE, "MARMOUSI_FULL_RESULTS", "test_slice_8000_8600")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    py = sys.executable
    script = os.path.join(HERE, "run_marmousi_full.py")
    common = ["--out", out]
    steps = [
        ["prepare", "--marmousi-dir", args.marmousi_dir, "--trace-range", "8000:8600"]
        + (["--n-jobs", str(args.n_jobs)] if args.n_jobs else []),
        ["sweep", "--networks", "unet", "rescnn", "--datas", "noisy", "--supervisions", "dense", "none",
         "--seeds", "42", "--epochs", "2", "--train-set", "subset", "--n-random", "40",
         "--transfer-dir", args.transfer_dir,
         "--per-gpu", "2", "--with-baseline", "--baseline-jobs", str(args.n_jobs or 4)],
        ["sweep", "--networks", "unet", "--datas", "noisy", "--supervisions", "none", "wells",
         "--seeds", "42", "--epochs", "2", "--train-set", "all", "--transfer-dir", args.transfer_dir],
        ["report"],
    ]
    t = time.time()
    for step in steps:
        cmd = [py, script, step[0]] + common + step[1:]
        print("      " + " ".join(os.path.basename(c) if c.startswith("/") else c for c in cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            ok = False
            break
    else:
        expected = ["prepared/prepared.json", "prepared/Z_rw_noisy.npy",
                    "train/unet_noisy_dense_subset_s42/Z_pred.npy",
                    "train/rescnn_noisy_none_subset_s42/train_record.json",
                    "train/unet_noisy_none_all_s42/Z_pred.npy",
                    "train/unet_noisy_wells5_all_s42/train_record.json",
                    "baseline/Z_tv_noisy.npy", "report/metrics.md",
                    "report/Fig_full_sections.png", "report/Fig_zoom_fault.png"]
        missing = [e for e in expected if not os.path.exists(os.path.join(out, e))]
        print(f"      {time.time() - t:.0f} s; missing outputs: {missing or 'none'}")
        ok &= not missing
        if not missing:
            rec = rmf.read_json(os.path.join(out, "train/unet_noisy_dense_subset_s42/train_record.json"))
            print(f"      unet noisy dense: {rec['n_patches']} patches, "
                  f"{100 * rec['held_out_fraction']:.1f} % held out, "
                  f"{rec['ms_per_patch_step']:.2f} ms per patch and step on {rec['device']}")
    print(f"      outputs kept in {out}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop", default="marmousi_fault_crop_400x300")
    p.add_argument("--marmousi-dir", default=os.path.join(HERE, "..", "data", "marmousi2"))
    p.add_argument("--transfer-dir", default=os.path.join(HERE, "SYNTHETIC_PRETRAIN"),
                   help="synthetic pretraining checkpoints (copies of MANUSCRIPT_IEEE_TGRS/results)")
    p.add_argument("--skip-t5", action="store_true")
    p.add_argument("--only-t5", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    args = p.parse_args()
    if args.only_t5:
        ok = run_t5(args)
        print("ALL PASSED" if ok else "SOME CHECKS FAILED")
        sys.exit(0 if ok else 1)

    mod = rmf.load_main_module()
    mod.DEVICE = torch.device("cpu")
    crop_dir = os.path.join(HERE, "ADDM_PINN_RESULTS", args.crop)
    d = np.load(os.path.join(crop_dir, "2D_results.npz"), allow_pickle=True)
    Z_true, Z_init, wavelet = d["Z_true"], d["Z_init"], d["wavelet"]
    S_clean, S_noisy = d["S_clean"], d["S_noisy"]
    L_init = np.log(Z_init)
    nx, nt = Z_true.shape
    ok = True

    print("T1  trace-wise reweighted-L1 ADMM, banded and parallel vs published")
    for label, S, Z_ref in (("clean", S_clean, d["Z_rw_clean"]), ("noisy", S_noisy, d["Z_rw_noisy"])):
        t = time.time()
        Z, iters = admm_banded.admm_l1_traces(
            S, wavelet, L_init, mod.MU, mod.ALPHA, mod.LAMBDA_, mod.EPSILON_CL,
            mod.MAX_ITER_CL, mod.TOL_CL, reweight=True, n_jobs=args.n_jobs)
        dt_ = time.time() - t
        ok &= verdict(f"{label}: max relative difference to published Z_rw", rel(Z, Z_ref), 1e-6)
        print(f"      {dt_:.1f} s, {iters.mean():.1f} iterations per trace (max {iters.max()})")
    # The dense LU solve of the original and the banded Cholesky solve differ
    # by rounding only, but the reweighting floor (epsilon = 1e-4, M up to
    # 1e4) makes the system matrix conditioned at about 1e8, so two exact
    # solvers agree to about 1e-8 per solve and the soft-thresholding then
    # amplifies that over the iterations.  Three checks: the assembled band
    # against the dense matrix of the reference (exact algebra), a fixed small
    # number of iterations, and the full convergence rule.
    worst_fixed, worst_full, worst_band = 0.0, 0.0, 0.0
    ab, WDh = admm_banded.banded_operators(wavelet, nt)
    kb = ab.shape[0] - 1
    rng = np.random.default_rng(0)
    for i in (0, nx // 2, nx - 1):
        M = 1.0 / (np.abs(rng.normal(0.0, 0.02, nt - 1)) + mod.EPSILON_CL)
        D_half = 0.5 * mod.difference_matrix(nt)
        A_dense = (WDh.T @ WDh + mod.ALPHA * np.eye(nt)
                   + mod.LAMBDA_ * (M[:, None] * D_half).T @ (M[:, None] * D_half))
        band = ab.copy()
        band[kb, :] += mod.ALPHA
        band[kb, :-1] += 0.25 * mod.LAMBDA_ * M * M
        band[kb, 1:] += 0.25 * mod.LAMBDA_ * M * M
        band[kb - 1, 1:] -= 0.25 * mod.LAMBDA_ * M * M
        dense_band = np.zeros_like(band)
        for off in range(kb + 1):
            dense_band[kb - off, off:] = np.diagonal(A_dense, off)
        worst_band = max(worst_band, rel(band, dense_band))
        if nt > kb + 1:
            worst_band = max(worst_band, float(np.abs(np.diagonal(A_dense, kb + 1)).max()
                                               / np.abs(A_dense).max()))
    ok &= verdict("band assembly vs dense system matrix of the reference", worst_band, 1e-13)
    for i in (0, nx // 2, nx - 1):
        Z_i, conv = mod.admm_l1_single_trace(S_noisy[i], wavelet, L_init[i], mod.MU, mod.ALPHA,
                                             mod.LAMBDA_, mod.EPSILON_CL, mod.MAX_ITER_CL,
                                             mod.TOL_CL, reweight=True)
        worst_full = max(worst_full, rel(Z[i], Z_i))
        Z_5, _ = mod.admm_l1_single_trace(S_noisy[i], wavelet, L_init[i], mod.MU, mod.ALPHA,
                                          mod.LAMBDA_, mod.EPSILON_CL, 5, 0.0, reweight=True)
        L_5, _, _, _, _ = admm_banded._trace_iterations(
            S_noisy[i] @ WDh, L_init[i], ab, mod.MU, mod.ALPHA, mod.LAMBDA_, mod.EPSILON_CL,
            5, 0.0, True, np.zeros(nt - 1), np.zeros(nt - 1), np.ones(nt - 1), 5, False)
        worst_fixed = max(worst_fixed, rel(np.exp(L_5), Z_5))
    ok &= verdict("noisy: 5 fixed iterations vs admm_l1_single_trace (dense)", worst_fixed, 1e-6)
    ok &= verdict("noisy: converged, three traces vs admm_l1_single_trace", worst_full, 1e-6)
    m_new = mod.compute_metrics(Z, Z_true)
    m_ref = mod.compute_metrics(d["Z_rw_noisy"], Z_true)
    ok &= verdict("noisy: |SNR difference| to published (dB)", abs(m_new[0] - m_ref[0]), 1e-5)

    print("T2  2-D coupled RW-L1 + TV baseline, banded and parallel vs published")
    tv_path = os.path.join(HERE, "2D_TV_BASELINE_RESULTS", args.crop, "Z_tv_baseline.npz")
    tv = np.load(tv_path)
    for label, S in (("clean", S_clean), ("noisy", S_noisy)):
        t = time.time()
        Z, conv = admm_banded.admm_l1_2d_tv_banded(
            S, wavelet, L_init, mod.MU, mod.ALPHA, mod.LAMBDA_, mu_x=100.0 * mod.MU,
            rho=40.0 * mod.ALPHA, epsilon=mod.EPSILON_CL, max_iter=40, tol=1e-6,
            vert_inner=10, reweight=True, n_jobs=args.n_jobs)
        dt_ = time.time() - t
        ok &= verdict(f"{label}: max relative difference to published Z_tv", rel(Z, tv[f"Z_tv_{label}"]), 1e-5)
        ok &= verdict(f"{label}: outer residual history vs published",
                      float(np.max(np.abs(np.asarray(conv) - tv[f"conv_{label}"]))), 1e-8)
        m_new = mod.compute_metrics(Z, Z_true)
        m_ref = mod.compute_metrics(tv[f"Z_tv_{label}"], Z_true)
        ok &= verdict(f"{label}: |SNR difference| to published (dB)", abs(m_new[0] - m_ref[0]), 1e-5)
        print(f"      {dt_:.1f} s, {len(conv)} outer iterations")

    print("T3  batched predictor vs predict_full_section")
    net = mod.UNet2D(mod.PATCH_NX, mod.PATCH_NT, base_ch=16)
    ckpt = torch.load(os.path.join(crop_dir, "2D_UNet_noisy_pretrain.pt"),
                      map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    corners = mod.compute_patch_corners(nx, nt, mod.PATCH_NX, mod.PATCH_NT,
                                        mod.STRIDE_NX, mod.STRIDE_NT)
    S_trim = d["S_noisy_trim"]
    L_prior = np.log(d["Z_rw_noisy"] + 1e-12)
    Z_a = mod.predict_full_section(net, S_trim, L_prior, corners, mod.PATCH_NX, mod.PATCH_NT, nx, nt)
    Z_b = rmf.predict_section_batched(mod, net, S_trim, L_prior, corners, nx, nt, batch=7)
    ok &= verdict("batched (batch 7) vs one patch at a time", rel(Z_b, Z_a), 1e-5)
    ok &= verdict("batched prediction vs published Z_unet_noisy", rel(Z_b, d["Z_unet_noisy"]), 1e-5)

    print("T4  vectorised forward model vs per-trace loop")
    L_true = np.log(Z_true)
    W = mod.convolution_matrix(wavelet, nt - 1)
    S_loop = np.zeros((nx, nt + len(wavelet) - 2))
    for i in range(nx):
        Z = Z_true[i]
        S_loop[i] = W @ ((Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1]))
    ok &= verdict("forward_model vs loop", rel(admm_banded.forward_model(L_true, wavelet), S_loop), 1e-12)
    ok &= verdict("forward_model vs published S_clean", rel(admm_banded.forward_model(L_true, wavelet), S_clean), 1e-12)

    print("T6  sparse-label supervision terms")
    g = torch.Generator().manual_seed(0)
    Lp = torch.randn(4, 1, 8, 16, generator=g)
    Lt = torch.randn(4, 1, 8, 16, generator=g)
    s_dense, g_dense = mod.supervised_terms(Lp, Lt)
    s_full, g_full = mod.supervised_terms(Lp, Lt, torch.ones_like(Lp))
    g_vert = torch.nn.functional.l1_loss(Lp[..., 1:] - Lp[..., :-1], Lt[..., 1:] - Lt[..., :-1])
    ok &= verdict("all-ones mask: impedance term equals the dense MSE", float(abs(s_full - s_dense)), 1e-6)
    ok &= verdict("all-ones mask: gradient term equals the vertical L1 term", float(abs(g_full - g_vert)), 1e-6)
    s_zero, g_zero = mod.supervised_terms(Lp, Lt, torch.zeros_like(Lp))
    ok &= verdict("empty mask: both terms are zero", float(abs(s_zero) + abs(g_zero)), 1e-12)
    m = torch.zeros_like(Lp); m[:, :, 3, :] = 1.0
    s_col, g_col = mod.supervised_terms(Lp, Lt, m)
    ref = torch.mean((Lp[:, :, 3, :] - Lt[:, :, 3, :]) ** 2)
    ok &= verdict("one labelled column: impedance term is its own MSE", float(abs(s_col - ref)), 1e-6)

    print("T7  low-band anchor")
    k = mod.lowpass_kernel(12.0, torch.device("cpu"))
    ok &= verdict("kernel sums to one", float(abs(k.sum() - 1.0)), 1e-6)
    x = torch.zeros(2, 1, 8, 128)
    ok &= verdict("zero difference gives zero", float(mod.lowband_mse(x, k)), 1e-12)
    ok &= verdict("constant offset 0.3 gives 0.09", float(abs(mod.lowband_mse(x + 0.3, k) - 0.09)), 1e-6)
    t = torch.arange(128, dtype=torch.float32)
    hf = torch.sin(2 * torch.pi * t / 4).view(1, 1, 1, -1).expand(2, 1, 8, 128)   # period 4 samples
    ok &= verdict("high-frequency difference is almost invisible", float(mod.lowband_mse(hf, k)), 1e-3)
    ok &= verdict("W_LOWBAND default is 0 (published loss unchanged)", float(mod.W_LOWBAND), 1e-12)

    print("T8  well penalty in the classical solvers")
    wells = [60, 340]
    logs = [np.log(Z_true[c]) for c in wells]
    Z_a, _ = admm_banded.admm_l1_traces(S_noisy[:64], wavelet, L_init[:64], mod.MU, mod.ALPHA, mod.LAMBDA_,
                                        mod.EPSILON_CL, mod.MAX_ITER_CL, mod.TOL_CL, True, n_jobs=4)
    Z_b, _ = admm_banded.admm_l1_traces(S_noisy[:64], wavelet, L_init[:64], mod.MU, mod.ALPHA, mod.LAMBDA_,
                                        mod.EPSILON_CL, mod.MAX_ITER_CL, mod.TOL_CL, True, n_jobs=4,
                                        well_cols=[60], well_logs=[logs[0]], well_weight=0.0)
    ok &= verdict("weight 0: trace-wise result unchanged", rel(Z_b, Z_a), 1e-12)
    Z_c, _ = admm_banded.admm_l1_traces(S_noisy[:64], wavelet, L_init[:64], mod.MU, mod.ALPHA, mod.LAMBDA_,
                                        mod.EPSILON_CL, mod.MAX_ITER_CL, mod.TOL_CL, True, n_jobs=4,
                                        well_cols=[60], well_logs=[logs[0]], well_weight=1e4)
    ok &= verdict("weight 1e4: well trace equals its log", rel(Z_c[60], Z_true[60]), 1e-3)
    ok &= verdict("weight 1e4: a far trace is unchanged", rel(Z_c[0], Z_a[0]), 1e-12)
    Z_d, _ = admm_banded.admm_l1_2d_tv_banded(S_noisy, wavelet, L_init, mod.MU, mod.ALPHA, mod.LAMBDA_,
                                              mu_x=100.0 * mod.MU, rho=40.0 * mod.ALPHA, epsilon=mod.EPSILON_CL,
                                              max_iter=3, tol=0.0, vert_inner=10, n_jobs=4,
                                              well_cols=wells, well_logs=logs, well_weight=0.0)
    Z_e, _ = admm_banded.admm_l1_2d_tv_banded(S_noisy, wavelet, L_init, mod.MU, mod.ALPHA, mod.LAMBDA_,
                                              mu_x=100.0 * mod.MU, rho=40.0 * mod.ALPHA, epsilon=mod.EPSILON_CL,
                                              max_iter=3, tol=0.0, vert_inner=10, n_jobs=4)
    ok &= verdict("2-D solver, weight 0: unchanged", rel(Z_d, Z_e), 1e-12)
    Z_f, _ = admm_banded.admm_l1_2d_tv_banded(S_noisy, wavelet, L_init, mod.MU, mod.ALPHA, mod.LAMBDA_,
                                              mu_x=100.0 * mod.MU, rho=40.0 * mod.ALPHA, epsilon=mod.EPSILON_CL,
                                              max_iter=3, tol=0.0, vert_inner=10, n_jobs=4,
                                              well_cols=wells, well_logs=logs, well_weight=1e4)
    # the 2-D solver returns the consensus average of the penalised vertical
    # variable and the laterally smoothed one, so the wells are not pinned
    # exactly; the penalty must cut the well-trace error by at least half
    ratio = max(rel(Z_f[c], Z_true[c]) / rel(Z_e[c], Z_true[c]) for c in wells)
    ok &= verdict("2-D solver, weight 1e4: well-trace error at most half of the unpenalised", ratio, 0.5)
    far = [c for c in range(nx) if min(abs(c - w) for w in wells) > 100]
    ok &= verdict("2-D solver, weight 1e4: traces far from the wells barely change",
                  rel(Z_f[far], Z_e[far]), 5e-2)

    if not args.skip_t5:
        ok &= run_t5(args)

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
