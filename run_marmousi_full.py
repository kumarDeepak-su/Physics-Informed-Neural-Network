#!/usr/bin/env python3
"""
Complete Marmousi-2 benchmark (13 601 traces x 2 801 samples) for the
ADMM-guided physics-informed inversion, organised in stages so that the work
spreads over all CPU cores and GPUs of a compute node:

  prepare   read the model, forward-model clean and noisy data, build the
            smooth starting model and the trace-wise reweighted-L1 ADMM prior
            for both data sets (banded solver, all cores);
  train     train one network on one data set with one supervision mode and
            one seed, predict the whole model (batched on the GPU) and score it
            on the whole model and on the samples no training patch covers;
  baseline  the spatially coupled RW-L1 + lateral-TV classical baseline
            (banded solver, all cores);
  sweep     run many `train` jobs concurrently over the visible GPUs
            (several per GPU), optionally with the baseline alongside;
  report    metrics tables per region and figures from the saved sections.

The networks, losses, optimiser, schedule, reweighting and patch geometry are
those of 2D_ADMM_PINN_ResCNN_Attention.py, imported unchanged.  The classical
solvers always work on the complete model (every trace at full length, the
whole section for the 2-D baseline) and every network predicts the complete
model.  The training set is every patch of the model (`--train-set all`, the
default: the 64 x 128 patch grid with stride 32 x 64 covers the whole model)
or, for a held-out test, the patches of the two published crop windows plus
`--n-random` random patches drawn with a fixed seed (`--train-set subset`).

Declared rules
  noise      white Gaussian noise with standard deviation RMS(S) / SNR, where
             the RMS is taken over the whole section (`--noise-reference whole`,
             the rule of add_noise in the main code) or over one of the crop
             windows so that the local SNR there equals the crop experiments;
             the noise seed is 42 unless `--noise-seed` says otherwise;
  seed       set_seed(seed) is called immediately before each training, as in
             run_multiseed_ci.py (the main script seeds once and trains four
             networks in sequence, so its seed 42 is not the same RNG state);
  held out   every sample that no training patch covers;
  regions    the whole model, the crop windows, thirds in time, thirds in
             trace; the index ranges are written to report/metrics.json;
  wells      `--supervision wells` gives the true log-impedance on a few trace
             columns only (pseudo-wells); the supervised and vertical-gradient
             terms are averaged over the labelled samples (supervised_terms in
             the main script).  Default positions: the centres of five equal
             lateral bins of the loaded range (for the paper's 400-trace crops
             this is overridden by `--well-traces`); the positions are
             recorded in train_record.json;
  precision  full FP32 on the GPU unless `--tf32` is given.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "2D_ADMM_PINN_ResCNN_Attention.py")
sys.path.insert(0, HERE)
import admm_banded  # noqa: E402

# crop windows of the paper (trace start, trace end, sample start, sample end)
CROP_WINDOWS = {
    "fault": (8100, 8500, 1500, 1800),
    "second": (4500, 4900, 1000, 1300),
}
NETWORKS = {
    "unet": ("UNet2D", "U-Net", "2D_UNet"),
    "rescnn": ("ResCNN2D", "ResCNN", "2D_ResCNN"),
    "attention": ("AttentionResUNet2D", "AttentionResUNet", "2D_AttentionResUNet"),
}
PREPARED_ARRAYS = ("Z_true", "Z_init", "S_clean", "S_noisy", "Z_rw_clean",
                   "Z_rw_noisy", "wavelet")


# ------------------------------------------------------------------ helpers

def load_main_module():
    spec = importlib.util.spec_from_file_location("admm_pinn_2d", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_range(text, n):
    """'a:b' -> (a, b) clipped to [0, n]; None -> (0, n)."""
    if not text:
        return 0, n
    a, b = text.split(":")
    a = int(a) if a else 0
    b = int(b) if b else n
    return max(0, a), min(n, b)


def software_info():
    import matplotlib
    import scipy
    import torch
    info = dict(python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                torch=torch.__version__, platform=platform.platform(),
                host=platform.node())
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def masked_metrics(Z_pred, Z_true, mask=None):
    """SNR (dB) and RMSE of the samples selected by mask (all if None)."""
    if mask is None:
        err = Z_pred - Z_true
        ref = Z_true
    else:
        if not mask.any():
            return None
        err = Z_pred[mask] - Z_true[mask]
        ref = Z_true[mask]
    snr = 20.0 * np.log10(np.linalg.norm(ref) / (np.linalg.norm(err) + 1e-12))
    rmse = np.sqrt(np.mean(err ** 2))
    return {"snr_db": float(snr), "rmse": float(rmse), "samples": int(ref.size)}


def coverage_mask(corners, nx, nt, patch_nx, patch_nt):
    mask = np.zeros((nx, nt), dtype=bool)
    for x0, t0 in corners:
        mask[x0:x0 + patch_nx, t0:t0 + patch_nt] = True
    return mask


def region_definitions(nx, nt, offset, crops):
    """
    Named sample windows (trace start, trace end, sample start, sample end) in
    full-model coordinates: the whole loaded range, the crop windows that lie
    inside it, thirds in time and thirds in trace.  `offset` is the (trace,
    sample) origin of the loaded range.
    """
    ox, ot = offset
    defs = {"whole": (ox, ox + nx, ot, ot + nt)}
    for name, (x0, x1, t0, t1) in crops.items():
        if x0 >= ox and x1 <= ox + nx and t0 >= ot and t1 <= ot + nt:
            defs[f"crop_{name}"] = (x0, x1, t0, t1)
    thirds_t = np.linspace(0, nt, 4).astype(int)
    for name, k in zip(("shallow", "middle", "deep"), range(3)):
        defs[name] = (ox, ox + nx, ot + int(thirds_t[k]), ot + int(thirds_t[k + 1]))
    thirds_x = np.linspace(0, nx, 4).astype(int)
    for name, k in zip(("left", "centre", "right"), range(3)):
        defs[name] = (ox + int(thirds_x[k]), ox + int(thirds_x[k + 1]), ot, ot + nt)
    return defs


def region_masks(nx, nt, offset, crops):
    """Boolean masks of the regions of region_definitions."""
    ox, ot = offset
    masks = {}
    for name, (x0, x1, t0, t1) in region_definitions(nx, nt, offset, crops).items():
        m = np.zeros((nx, nt), dtype=bool)
        m[x0 - ox:x1 - ox, t0 - ot:t1 - ot] = True
        masks[name] = m
    return masks


def gaussian_cutoff_hz(sigma_samples, dt):
    """Frequency at which a Gaussian of width sigma (samples) has amplitude 0.5."""
    return float(np.sqrt(np.log(2.0)) / (np.sqrt(2.0) * np.pi * sigma_samples * dt))


def well_starting_model(L_true, cols, sigma_t):
    """
    Log-impedance starting model from the true log-impedance at the trace
    columns `cols`: each column is low-passed in time with a Gaussian of
    width sigma_t (samples) and the columns are interpolated linearly across
    traces (held constant beyond the outer columns).
    """
    from scipy.ndimage import gaussian_filter1d
    order = np.argsort(cols)
    cols = [cols[k] for k in order]
    logs = np.stack([gaussian_filter1d(np.asarray(L_true[c], dtype=np.float64), sigma_t)
                     for c in cols])
    x = np.arange(L_true.shape[0])
    out = np.empty(L_true.shape)
    for t in range(L_true.shape[1]):
        out[:, t] = np.interp(x, cols, logs[:, t])
    return out


def load_prepared(out, arrays=PREPARED_ARRAYS, mmap=True):
    d = os.path.join(out, "prepared")
    info = read_json(os.path.join(d, "prepared.json"))
    data = {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r" if mmap else None)
            for k in arrays}
    return info, data


def trim(S, nw, nt):
    half_pad = (nw - 1) // 2
    return np.asarray(S[:, half_pad:half_pad + nt])


# ------------------------------------------------------------------ prepare

def stage_prepare(args):
    mod = load_main_module()
    out = os.path.join(args.out, "prepared")
    if os.path.exists(os.path.join(out, "prepared.json")) and not args.overwrite:
        raise SystemExit(f"{out} exists; use --overwrite to rebuild it (this invalidates "
                         "every training and baseline result in the run folder)")
    os.makedirs(out, exist_ok=True)
    t_all = time.time()

    vp_path = os.path.join(args.marmousi_dir, "vp.segy")
    rho_path = os.path.join(args.marmousi_dir, "density.segy")
    print(f"  reading {vp_path}", flush=True)
    vp, dt = mod.read_segy_simple(vp_path)
    print(f"  reading {rho_path}", flush=True)
    rho, dt_rho = mod.read_segy_simple(rho_path)
    if vp.shape != rho.shape or abs(dt - dt_rho) > 1e-9:
        raise ValueError("vp and density files do not match")
    x0, x1 = parse_range(args.trace_range, vp.shape[0])
    t0, t1 = parse_range(args.sample_range, vp.shape[1])
    Z_true = (vp[x0:x1, t0:t1].astype(np.float64) * rho[x0:x1, t0:t1].astype(np.float64))
    del vp, rho
    nx, nt = Z_true.shape
    print(f"  model {nx} x {nt} samples, dt = {dt * 1e3:.3f} ms, "
          f"traces {x0}:{x1}, samples {t0}:{t1}", flush=True)

    wavelet = mod.ricker_wavelet(args.f0, dt)
    nw = len(wavelet)
    L_true = np.log(Z_true)

    t = time.time()
    S_clean = admm_banded.forward_model(L_true, wavelet)
    # noise: RMS(S) / SNR, the RMS over the whole section (as add_noise in the
    # main code) or over one crop window; the draw is the same as add_noise
    windows = region_definitions(nx, nt, (x0, t0), CROP_WINDOWS)
    if args.noise_reference == "whole":
        sig_rms = np.linalg.norm(S_clean) / np.sqrt(S_clean.size)
    else:
        key = f"crop_{args.noise_reference}"
        if key not in windows:
            raise SystemExit(f"crop window '{args.noise_reference}' is outside the loaded range")
        a, b, c, d = windows[key]
        ref = trim(S_clean, nw, nt)[a - x0:b - x0, c - t0:d - t0]
        sig_rms = np.linalg.norm(ref) / np.sqrt(ref.size)
    noise_std = sig_rms / args.noise_snr
    rng = np.random.default_rng(args.noise_seed)
    S_noisy = S_clean + rng.normal(0.0, noise_std, S_clean.shape)
    t_forward = time.time() - t
    S_trim_c = trim(S_clean, nw, nt)
    S_trim_n = trim(S_noisy, nw, nt)
    local_snr = {}
    for name, (a, b, c, d) in windows.items():
        sig = S_trim_c[a - x0:b - x0, c - t0:d - t0]
        noi = (S_trim_n - S_trim_c)[a - x0:b - x0, c - t0:d - t0]
        local_snr[name] = float(np.linalg.norm(sig) / np.linalg.norm(noi))
    # starting model: Gaussian smoothing of the true impedance (the rule of
    # the paper) or a well-derived model: the true log-impedance at a few
    # trace columns, low-passed in time and interpolated linearly between the
    # wells (constant beyond the outer wells); both carry only frequencies
    # below the stated cut-off
    if args.starting_model == "wells":
        wells = args.well_traces or default_well_traces(nx, (x0, t0))
        cols = [int(w) - x0 for w in wells]
        if min(cols) < 0 or max(cols) >= nx:
            raise SystemExit(f"well traces {wells} are outside the loaded range")
        Z_init = np.exp(well_starting_model(L_true, cols, args.well_lowpass))
        starting_model = dict(kind="wells", well_traces=[int(w) for w in wells],
                              lowpass_sigma_samples=args.well_lowpass,
                              cutoff_hz=gaussian_cutoff_hz(args.well_lowpass, dt))
    else:
        Z_init = gaussian_filter(Z_true, sigma=(args.smooth_sigma_x, args.smooth_sigma_t))
        starting_model = dict(kind="gaussian", sigma_x_traces=args.smooth_sigma_x,
                              sigma_t_samples=args.smooth_sigma_t,
                              cutoff_hz=gaussian_cutoff_hz(args.smooth_sigma_t, dt))
    L_init = np.log(Z_init)
    print(f"  starting model: {starting_model}", flush=True)
    print(f"  forward model + noise: {t_forward:.1f} s; wavelet {nw} samples; noise std "
          f"{noise_std:.4g} ({args.noise_reference} reference), local SNR ratio in the crop "
          f"windows {', '.join(f'{k[5:]} {v:.1f}' for k, v in local_snr.items() if k.startswith('crop_'))}",
          flush=True)

    n_jobs = args.n_jobs or os.cpu_count()
    timings = {"forward_and_noise": t_forward}
    iterations = {}
    results = {}
    for label, S in (("clean", S_clean), ("noisy", S_noisy)):
        t = time.time()
        Z_rw, iters = admm_banded.admm_l1_traces(
            S, wavelet, L_init, mod.MU, mod.ALPHA, mod.LAMBDA_, args.epsilon,
            mod.MAX_ITER_CL, mod.TOL_CL, reweight=True, n_jobs=n_jobs,
            chunk=args.chunk)
        timings[f"admm_{label}"] = time.time() - t
        iterations[label] = {"mean": float(iters.mean()), "max": int(iters.max()),
                             "at_limit": int((iters >= mod.MAX_ITER_CL).sum())}
        results[label] = (Z_rw, iters)
        m = masked_metrics(Z_rw, Z_true)
        print(f"  ADMM {label}: {timings[f'admm_{label}']:.1f} s on {n_jobs} workers, "
              f"{iters.mean():.1f} iterations per trace, SNR {m['snr_db']:.2f} dB", flush=True)

    arrays = dict(Z_true=Z_true, Z_init=Z_init, S_clean=S_clean, S_noisy=S_noisy,
                  Z_rw_clean=results["clean"][0], Z_rw_noisy=results["noisy"][0],
                  wavelet=wavelet, admm_iters_clean=results["clean"][1],
                  admm_iters_noisy=results["noisy"][1])
    for k, v in arrays.items():
        np.save(os.path.join(out, k + ".npy"), v)

    regions = region_masks(nx, nt, (x0, t0), CROP_WINDOWS)
    metrics = {name: {reg: masked_metrics(Z, Z_true, m) for reg, m in regions.items()}
               for name, Z in (("initial_model", Z_init),
                               ("admm_clean", results["clean"][0]),
                               ("admm_noisy", results["noisy"][0]))}
    warnings = []
    for label in ("clean", "noisy"):
        for reg in regions:
            if metrics[f"admm_{label}"][reg]["snr_db"] < metrics["initial_model"][reg]["snr_db"]:
                warnings.append(f"ADMM prior ({label}) is below the starting model in region "
                                f"'{reg}': {metrics[f'admm_{label}'][reg]['snr_db']:.2f} dB vs "
                                f"{metrics['initial_model'][reg]['snr_db']:.2f} dB")
    for w_ in warnings:
        print("  WARNING: " + w_, flush=True)
    info = dict(
        nx=nx, nt=nt, nw=nw, dt=dt, f0=args.f0, noise_snr=args.noise_snr,
        noise_reference=args.noise_reference, noise_seed=args.noise_seed,
        noise_std=float(noise_std), local_snr_ratio=local_snr,
        smooth_sigma=[args.smooth_sigma_x, args.smooth_sigma_t],
        starting_model=starting_model,
        trace_range=[x0, x1], sample_range=[t0, t1],
        outputs={k + ".npy": sha256(os.path.join(out, k + ".npy")) for k in arrays},
        admm=dict(mu=mod.MU, alpha=mod.ALPHA, lambda_=mod.LAMBDA_,
                  epsilon=args.epsilon, max_iter=mod.MAX_ITER_CL, tol=mod.TOL_CL,
                  n_jobs=n_jobs, chunk=args.chunk, iterations=iterations),
        inputs={os.path.basename(p): {"path": p, "sha256": sha256(p)}
                for p in (vp_path, rho_path)},
        metrics=metrics, warnings=warnings, timings=timings,
        total_seconds=time.time() - t_all,
        software=software_info(), command=" ".join(sys.argv))
    write_json(os.path.join(out, "prepared.json"), info)
    print(f"  prepared in {info['total_seconds']:.1f} s -> {out}", flush=True)


# -------------------------------------------------------------------- train

def training_corners(mod, train_set, nx, nt, offset, n_random, seed, crops):
    px, pt, sx, st = mod.PATCH_NX, mod.PATCH_NT, mod.STRIDE_NX, mod.STRIDE_NT
    if train_set == "all":
        return mod.compute_patch_corners(nx, nt, px, pt, sx, st), []
    corners, used = [], []
    ox, ot = offset
    for name, (x0, x1, t0, t1) in crops.items():
        a, b, c, d = x0 - ox, x1 - ox, t0 - ot, t1 - ot
        if a < 0 or b > nx or c < 0 or d > nt:
            continue
        used.append(name)
        for cx, ct in mod.compute_patch_corners(b - a, d - c, px, pt, sx, st):
            corners.append((a + cx, c + ct))
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, nx - px + 1, n_random)
    ts = rng.integers(0, nt - pt + 1, n_random)
    corners.extend(zip(xs.tolist(), ts.tolist()))
    return corners, used


def predict_section_batched(mod, net, S_trim, L_prior, corners, nx, nt, batch=256):
    """Same result as mod.predict_full_section, with patches run in batches."""
    import torch
    px, pt = mod.PATCH_NX, mod.PATCH_NT
    net.eval()
    L_patches = np.zeros((len(corners), px, pt), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(corners), batch):
            cs = corners[start:start + batch]
            s = np.stack([S_trim[x0:x0 + px, t0:t0 + pt] for x0, t0 in cs])
            l0 = np.stack([L_prior[x0:x0 + px, t0:t0 + pt] for x0, t0 in cs])
            S_t = torch.tensor(s, dtype=torch.float32).unsqueeze(1).to(mod.DEVICE)
            L0_t = torch.tensor(l0, dtype=torch.float32).unsqueeze(1).to(mod.DEVICE)
            L_patches[start:start + len(cs)] = net(S_t, L0_t)[:, 0].cpu().numpy()
    L_full = mod.reconstruct_from_patches(L_patches, corners, nx, nt, px, pt)
    return np.exp(L_full)


def default_well_traces(nx, offset, n_wells=5):
    """Centres of n_wells equal lateral bins of the loaded range (full coordinates)."""
    edges = np.linspace(0, nx, n_wells + 1)
    return [int(offset[0] + (edges[k] + edges[k + 1]) // 2) for k in range(n_wells)]


def well_mask(nx, nt, offset, well_traces):
    mask = np.zeros((nx, nt))
    for x in well_traces:
        j = int(x) - offset[0]
        if j < 0 or j >= nx:
            raise SystemExit(f"well trace {x} is outside the loaded range")
        mask[j, :] = 1.0
    return mask


def supervision_label(supervision, n_wells, prior="rw", lowband=0.0, lowband_sigma=12.0):
    """
    dense | none | wells<n>, followed by the variant suffixes: "+tv" when the
    network input prior is the 2-D coupled result, "+lb<w>" (and "s<sigma>"
    when it is not 12) when the low-band anchor is used.  Used in tags,
    records and tables.
    """
    label = f"wells{int(n_wells)}" if supervision == "wells" else supervision
    if prior == "tv":
        label += "+tv"
    if lowband > 0.0:
        label += f"+lb{lowband:g}" + (f"s{lowband_sigma:g}" if lowband_sigma != 12.0 else "")
    return label


def make_tag(network, data, supervision, n_wells, train_set, seed, prior="rw", lowband=0.0,
             lowband_sigma=12.0):
    return (f"{network}_{data}_{supervision_label(supervision, n_wells, prior, lowband, lowband_sigma)}"
            f"_{train_set}_s{seed}")


def run_tag(args):
    n_wells = len(args.well_traces) if args.well_traces else args.n_wells
    return args.tag or make_tag(args.network, args.data, args.supervision, n_wells,
                                args.train_set, args.seed, args.prior, args.lowband,
                                args.lowband_sigma)


def configure_torch(mod, args):
    import torch
    if args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device
    mod.DEVICE = torch.device(dev)
    if args.threads:
        torch.set_num_threads(args.threads)
    if dev == "cuda":
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    return dev


def stage_train(args):
    import torch
    mod = load_main_module()
    tag = run_tag(args)
    out = os.path.join(args.out, "train", tag)
    if os.path.exists(os.path.join(out, "Z_pred.npy")) and not args.overwrite:
        print(f"  {tag}: results exist, skipping (use --overwrite to redo)")
        return
    os.makedirs(out, exist_ok=True)
    t_all = time.time()
    dev = configure_torch(mod, args)

    info, P = load_prepared(args.out)
    nx, nt, nw = info["nx"], info["nt"], info["nw"]
    offset = (info["trace_range"][0], info["sample_range"][0])
    Z_true = np.asarray(P["Z_true"])
    L_true = np.log(Z_true)
    S_trim = trim(P[f"S_{args.data}"], nw, nt)
    if args.prior == "tv":
        tv_path = os.path.join(args.out, "baseline", f"Z_tv_{args.data}.npy")
        if not os.path.exists(tv_path):
            raise SystemExit(f"--prior tv needs the baseline first: {tv_path} is missing")
        L_prior = np.log(np.load(tv_path) + 1e-12)
    else:
        L_prior = np.log(np.asarray(P[f"Z_rw_{args.data}"]) + 1e-12)
    wavelet = np.asarray(P["wavelet"])

    mod.W_LOWBAND = args.lowband
    mod.LOWBAND_SIGMA = args.lowband_sigma
    mod.EPOCHS = args.epochs
    mod.BATCH_SIZE = args.batch_size
    mod.WARMUP_EPOCHS = min(mod.WARMUP_EPOCHS, max(args.epochs // 3, 1))
    if args.lr is not None:
        mod.LR = args.lr
    if args.base_ch is not None:
        mod.UNET_BASE_CH = args.base_ch

    corners, crops_used = training_corners(
        mod, args.train_set, nx, nt, offset, args.n_random, args.train_set_seed, CROP_WINDOWS)
    covered = coverage_mask(corners, nx, nt, mod.PATCH_NX, mod.PATCH_NT)
    held_out = ~covered
    print(f"  {tag}: {len(corners)} training patches on {dev}; "
          f"{100 * held_out.mean():.1f} % of the samples held out", flush=True)

    S_patches = mod.extract_patches_at_corners(S_trim, corners, mod.PATCH_NX, mod.PATCH_NT)
    L0_patches = mod.extract_patches_at_corners(L_prior, corners, mod.PATCH_NX, mod.PATCH_NT)
    Lt_patches, mask_patches, wells = None, None, None
    if args.supervision == "dense":
        Lt_patches = mod.extract_patches_at_corners(L_true, corners, mod.PATCH_NX, mod.PATCH_NT)
    elif args.supervision == "wells":
        wells = args.well_traces or default_well_traces(nx, offset, args.n_wells)
        mask = well_mask(nx, nt, offset, wells)
        Lt_patches = mod.extract_patches_at_corners(L_true, corners, mod.PATCH_NX, mod.PATCH_NT)
        mask_patches = mod.extract_patches_at_corners(mask, corners, mod.PATCH_NX, mod.PATCH_NT)
        n_lab = int(sum(1 for m in mask_patches if m.any()))
        print(f"  {tag}: pseudo-wells at traces {wells}; {n_lab} of {len(corners)} patches "
              f"carry labels", flush=True)

    class_name, model_name, prefix = NETWORKS[args.network]
    if args.no_transfer:
        transfer = None
    else:
        if not args.transfer_dir:
            raise SystemExit("give --transfer-dir (synthetic checkpoints) or --no-transfer")
        transfer = os.path.join(args.transfer_dir, f"{prefix}_{args.data}_pretrain.pt")
        if not os.path.exists(transfer):
            raise SystemExit(f"transfer checkpoint not found: {transfer}")

    mod.set_seed(args.seed)
    t = time.time()
    net, loss_history = mod.train_model_2d(
        S_patches, L0_patches, wavelet, mod.PATCH_NX, mod.PATCH_NT, label=tag,
        checkpoint_path=os.path.join(out, "checkpoint.pt"), L_true_patches=Lt_patches,
        model_class=getattr(mod, class_name), model_name=model_name,
        transfer_checkpoint=transfer, label_mask_patches=mask_patches)
    t_train = time.time() - t

    t = time.time()
    grid = mod.compute_patch_corners(nx, nt, mod.PATCH_NX, mod.PATCH_NT,
                                     mod.STRIDE_NX, mod.STRIDE_NT)
    Z_pred = predict_section_batched(mod, net, S_trim, L_prior, grid, nx, nt,
                                     batch=args.predict_batch)
    t_predict = time.time() - t
    np.save(os.path.join(out, "Z_pred.npy"), Z_pred.astype(np.float32))
    np.save(os.path.join(out, "loss_history.npy"), np.asarray(loss_history))
    outputs = {f: sha256(os.path.join(out, f)) for f in
               ("Z_pred.npy", "loss_history.npy", "checkpoint.pt")}

    metrics = {"whole": masked_metrics(Z_pred, Z_true),
               "held_out": masked_metrics(Z_pred, Z_true, held_out),
               "prior_whole": masked_metrics(np.exp(L_prior), Z_true),
               "prior_held_out": masked_metrics(np.exp(L_prior), Z_true, held_out)}
    if wells:
        # the score outside the labelled columns: what the labels did not see
        unlabelled = np.ones((nx, nt), dtype=bool)
        unlabelled[[int(w) - offset[0] for w in wells], :] = False
        metrics["unlabelled"] = masked_metrics(Z_pred, Z_true, unlabelled)
    peak = None
    if dev == "cuda":
        peak = int(torch.cuda.max_memory_allocated() // (1 << 20))
    record = dict(
        tag=tag, network=args.network, model_name=model_name, data=args.data,
        supervision=args.supervision,
        supervision_label=supervision_label(args.supervision, len(wells) if wells else 0,
                                            args.prior, args.lowband, args.lowband_sigma),
        prior=args.prior, lowband=args.lowband, lowband_sigma=args.lowband_sigma,
        n_wells=len(wells) if wells else 0, seed=args.seed, train_set=args.train_set,
        well_traces=wells, n_random=args.n_random, train_set_seed=args.train_set_seed,
        crops_used=crops_used, n_patches=len(corners),
        corners_sha256=hashlib.sha256(np.asarray(corners, dtype=np.int64).tobytes()).hexdigest(),
        corners=corners, held_out_fraction=float(held_out.mean()),
        epochs=mod.EPOCHS, batch_size=mod.BATCH_SIZE, lr=mod.LR, base_ch=mod.UNET_BASE_CH,
        warmup_epochs=mod.WARMUP_EPOCHS, transfer_checkpoint=transfer,
        transfer_sha256=sha256(transfer) if transfer else None,
        device=dev, tf32=bool(args.tf32), cudnn_benchmark=bool(args.cudnn_benchmark),
        threads=args.threads, gpu_peak_memory_mib=peak,
        timings={"train": t_train, "predict": t_predict, "total": time.time() - t_all},
        ms_per_patch_step=1000.0 * t_train / (len(corners) * mod.EPOCHS),
        metrics=metrics, outputs=outputs, prepared_json=sha256(
            os.path.join(args.out, "prepared", "prepared.json")),
        software=software_info(), command=" ".join(sys.argv),
        slurm={k: os.environ[k] for k in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
               if k in os.environ})
    write_json(os.path.join(out, "train_record.json"), record)
    print(f"  {tag}: train {t_train:.0f} s ({record['ms_per_patch_step']:.2f} ms per patch "
          f"and step), predict {t_predict:.1f} s, SNR whole {metrics['whole']['snr_db']:.2f} dB"
          + (f", held out {metrics['held_out']['snr_db']:.2f} dB" if metrics['held_out'] else ""),
          flush=True)


# ----------------------------------------------------------------- baseline

def stage_baseline(args):
    """
    2-D coupled RW-L1 + TV baseline.  With --wells the solver also gets the
    true log-impedance at the pseudo-well traces as a quadratic penalty of
    weight --well-weight (the information-matched classical comparator of the
    network's wells variant); its outputs go to baseline_wells<n>/.
    """
    mod = load_main_module()
    info, P = load_prepared(args.out)
    nx, nt = info["nx"], info["nt"]
    offset = (info["trace_range"][0], info["sample_range"][0])
    Z_true = np.asarray(P["Z_true"])
    L_init = np.log(np.asarray(P["Z_init"]))
    wavelet = np.asarray(P["wavelet"])
    wells, cols, logs = None, None, None
    if args.wells:
        wells = args.well_traces or default_well_traces(nx, offset, args.n_wells)
        cols = [int(w) - offset[0] for w in wells]
        if min(cols) < 0 or max(cols) >= nx:
            raise SystemExit(f"well traces {wells} are outside the loaded range")
        logs = [np.log(Z_true[c]) for c in cols]
        out = os.path.join(args.out, f"baseline_wells{len(wells)}")
    else:
        out = os.path.join(args.out, "baseline")
    os.makedirs(out, exist_ok=True)
    epsilon = info["admm"]["epsilon"] if args.epsilon is None else args.epsilon
    params = dict(mu_t=mod.MU, alpha=mod.ALPHA, lambda_t=mod.LAMBDA_,
                  mu_x=args.mu_x_factor * mod.MU, rho=args.rho_factor * mod.ALPHA,
                  epsilon=epsilon, max_iter=args.max_iter, tol=args.tol,
                  vert_inner=args.vert_inner, reweight=True,
                  well_traces=wells, well_weight=args.well_weight if wells else 0.0)
    n_jobs = args.n_jobs or os.cpu_count()
    record = dict(params=params, n_jobs=n_jobs, chunk=args.chunk, timings={},
                  outer_iterations={}, metrics={}, software=software_info(),
                  command=" ".join(sys.argv))
    unlabelled = None
    if cols:
        unlabelled = np.ones((nx, nt), dtype=bool)
        unlabelled[cols, :] = False
    for label in ("clean", "noisy"):
        S = np.asarray(P[f"S_{label}"])
        t = time.time()
        Z_tv, conv = admm_banded.admm_l1_2d_tv_banded(
            S, wavelet, L_init, params["mu_t"], params["alpha"], params["lambda_t"],
            mu_x=params["mu_x"], rho=params["rho"], epsilon=params["epsilon"],
            max_iter=params["max_iter"], tol=params["tol"],
            vert_inner=params["vert_inner"], reweight=True, n_jobs=n_jobs,
            chunk=args.chunk, verbose=args.verbose,
            well_cols=cols, well_logs=logs, well_weight=params["well_weight"])
        record["timings"][label] = time.time() - t
        record["outer_iterations"][label] = len(conv)
        record["metrics"][label] = masked_metrics(Z_tv, Z_true)
        if unlabelled is not None:
            record["metrics"][label + "_unlabelled"] = masked_metrics(Z_tv, Z_true, unlabelled)
        np.save(os.path.join(out, f"Z_tv_{label}.npy"), Z_tv)
        np.save(os.path.join(out, f"convergence_{label}.npy"), np.asarray(conv))
        print(f"  2D-TV{' + wells' if wells else ''} {label}: {record['timings'][label]:.0f} s, "
              f"{len(conv)} outer iterations, SNR {record['metrics'][label]['snr_db']:.2f} dB",
              flush=True)
    write_json(os.path.join(out, "baseline.json"), record)


# -------------------------------------------------------------------- sweep

def stage_sweep(args):
    """Run train jobs concurrently: `--per-gpu` processes on each visible GPU."""
    if not args.no_transfer and not args.transfer_dir:
        raise SystemExit("give --transfer-dir (synthetic checkpoints) or --no-transfer")
    if args.gpus:
        gpus = [str(g) for g in range(args.gpus)]
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[:args.gpus]
    else:
        gpus = [None]
    jobs = [(n, d, s, seed) for n in args.networks for d in args.datas
            for s in args.supervisions for seed in args.seeds]
    n_wells = len(args.well_traces) if args.well_traces else args.n_wells
    if not args.overwrite:
        finished = []
        for j in jobs:
            folder = os.path.join(args.out, "train",
                                  make_tag(j[0], j[1], j[2], n_wells, args.train_set, j[3],
                                           args.prior, args.lowband, args.lowband_sigma))
            if not os.path.exists(os.path.join(folder, "Z_pred.npy")):
                continue
            rec = read_json(os.path.join(folder, "train_record.json"))
            same = {"epochs": args.epochs, "batch_size": args.batch_size,
                    "n_random": args.n_random, "train_set_seed": args.train_set_seed}
            differ = {k: (rec.get(k), v) for k, v in same.items() if rec.get(k) != v}
            if differ:
                raise SystemExit(f"{folder} was trained with different settings {differ}; "
                                 "use another --out folder or --overwrite")
            finished.append(j)
        if finished:
            print(f"  {len(finished)} trainings already finished, kept", flush=True)
            jobs = [j for j in jobs if j not in finished]
    passthrough = ["--out", args.out, "--train-set", args.train_set,
                   "--n-random", str(args.n_random), "--train-set-seed", str(args.train_set_seed),
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--predict-batch", str(args.predict_batch), "--device", args.device]
    if args.threads:
        passthrough += ["--threads", str(args.threads)]
    if args.lr is not None:
        passthrough += ["--lr", str(args.lr)]
    passthrough += ["--n-wells", str(args.n_wells), "--prior", args.prior,
                    "--lowband", str(args.lowband), "--lowband-sigma", str(args.lowband_sigma)]
    if args.prior == "tv":
        for d in args.datas:
            if not os.path.exists(os.path.join(args.out, "baseline", f"Z_tv_{d}.npy")):
                raise SystemExit("--prior tv needs the baseline stage to have run first")
    if args.well_traces:
        passthrough += ["--well-traces"] + [str(w) for w in args.well_traces]
    if args.base_ch is not None:
        passthrough += ["--base-ch", str(args.base_ch)]
    passthrough += ["--no-transfer"] if args.no_transfer else ["--transfer-dir", args.transfer_dir]
    if args.tf32:
        passthrough.append("--tf32")
    if args.cudnn_benchmark:
        passthrough.append("--cudnn-benchmark")
    if args.overwrite:
        passthrough.append("--overwrite")

    slots = [(g, k) for g in gpus for k in range(args.per_gpu)]
    running = {}
    failed, done = [], []
    log_dir = os.path.join(args.out, "train")
    os.makedirs(log_dir, exist_ok=True)

    baseline_proc = None
    if args.with_baseline:
        cmd = [sys.executable, __file__, "baseline", "--out", args.out,
               "--n-jobs", str(args.baseline_jobs)]
        os.makedirs(os.path.join(args.out, "baseline"), exist_ok=True)
        blog = open(os.path.join(args.out, "baseline", "stdout.log"), "w")
        baseline_proc = subprocess.Popen(cmd, stdout=blog, stderr=subprocess.STDOUT)
        print(f"  baseline started with {args.baseline_jobs} workers", flush=True)

    t0 = time.time()
    print(f"  {len(jobs)} training jobs over {len(slots)} slots", flush=True)
    while jobs or running:
        free = [s for s in slots if s not in running.values()]
        while jobs and free:
            n, d, s, seed = jobs.pop(0)
            slot = free.pop(0)
            tag = make_tag(n, d, s, n_wells, args.train_set, seed, args.prior, args.lowband,
                           args.lowband_sigma)
            env = dict(os.environ)
            if slot[0] is not None:
                env["CUDA_VISIBLE_DEVICES"] = slot[0]
            if args.threads:
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    env[var] = str(args.threads)
            cmd = [sys.executable, "-u", __file__, "train", "--network", n, "--data", d,
                   "--supervision", s, "--seed", str(seed)] + passthrough
            os.makedirs(os.path.join(log_dir, tag), exist_ok=True)
            log = open(os.path.join(log_dir, tag, "stdout.log"), "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[proc] = slot
            print(f"  started {tag} on gpu {slot[0]} slot {slot[1]}", flush=True)
        for proc in list(running):
            code = proc.poll()
            if code is None:
                continue
            (done if code == 0 else failed).append(proc.args[3:11])
            del running[proc]
            print(f"  finished ({'ok' if code == 0 else 'FAILED'}) {' '.join(proc.args[3:11])}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
        time.sleep(3)
    baseline_failed = False
    if baseline_proc is not None:
        code = baseline_proc.wait()
        baseline_failed = code != 0
        print(f"  baseline finished ({'ok' if code == 0 else 'FAILED'}) [{time.time() - t0:.0f} s]",
              flush=True)
    print(f"  sweep done: {len(done)} ok, {len(failed)} failed, {time.time() - t0:.0f} s", flush=True)
    if failed or baseline_failed:
        raise SystemExit(1)


# ------------------------------------------------------------------- report

def stage_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    mod = load_main_module()
    out = os.path.join(args.out, "report")
    os.makedirs(out, exist_ok=True)
    info, P = load_prepared(args.out)
    nx, nt, dt = info["nx"], info["nt"], info["dt"]
    offset = (info["trace_range"][0], info["sample_range"][0])
    Z_true = np.asarray(P["Z_true"])
    regions = region_masks(nx, nt, offset, CROP_WINDOWS)

    sections = {"initial_model": np.asarray(P["Z_init"]),
                "admm_clean": np.asarray(P["Z_rw_clean"]),
                "admm_noisy": np.asarray(P["Z_rw_noisy"])}
    bdir = os.path.join(args.out, "baseline")
    for label in ("clean", "noisy"):
        f = os.path.join(bdir, f"Z_tv_{label}.npy")
        if os.path.exists(f):
            sections[f"tv2d_{label}"] = np.load(f)
    for wdir in sorted(glob.glob(os.path.join(args.out, "baseline_wells*"))):
        n = os.path.basename(wdir)[len("baseline_wells"):]
        for label in ("clean", "noisy"):
            f = os.path.join(wdir, f"Z_tv_{label}.npy")
            if os.path.exists(f):
                sections[f"tv2d_wells{n}_{label}"] = np.load(f)

    runs = []
    tdir = os.path.join(args.out, "train")
    if os.path.isdir(tdir):
        for tag in sorted(os.listdir(tdir)):
            rec = os.path.join(tdir, tag, "train_record.json")
            if os.path.exists(rec):
                runs.append(read_json(rec))
    groups = {}
    for r in runs:
        key = (r["train_set"], r["n_random"], r["train_set_seed"])
        groups.setdefault(key, []).append(r)

    table = {"prepared": {k: v for k, v in info.items() if k in
                          ("nx", "nt", "dt", "nw", "noise_snr", "noise_reference", "noise_seed",
                           "noise_std", "local_snr_ratio", "trace_range", "sample_range")},
             "region_definitions": region_definitions(nx, nt, offset, CROP_WINDOWS),
             "regions": {k: int(v.sum()) for k, v in regions.items()},
             "classical": {}, "groups": [], "figure_runs": {}}
    for name, Z in sections.items():
        table["classical"][name] = {reg: masked_metrics(Z, Z_true, m)
                                    for reg, m in regions.items()}

    lines = ["# Complete Marmousi-2 model: metrics", "",
             f"Model {nx} x {nt} samples at {dt * 1e3:.3f} ms, noise SNR ratio "
             f"{info['noise_snr']}, traces {info['trace_range']}, samples {info['sample_range']}.", "",
             "## Classical methods, SNR (dB) per region", "",
             "| Method | " + " | ".join(regions) + " |", "|---|" + "---:|" * len(regions)]
    for name, per in table["classical"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{per[r]['snr_db']:.2f}" if per[r] else "" for r in regions) + " |")

    for key, group in sorted(groups.items()):
        train_set, n_random, ts_seed = key
        corners = [tuple(c) for c in group[0]["corners"]]
        held = ~coverage_mask(corners, nx, nt, mod.PATCH_NX, mod.PATCH_NT)
        g = {"train_set": train_set, "n_random": n_random, "train_set_seed": ts_seed,
             "n_patches": group[0]["n_patches"], "held_out_fraction": float(held.mean()),
             "classical_held_out": {name: masked_metrics(Z, Z_true, held)
                                    for name, Z in sections.items()},
             "networks": {}}
        lines += ["", f"## Networks, training set `{train_set}` ({group[0]['n_patches']} patches, "
                      f"{100 * held.mean():.1f} % of samples held out)", "",
                  "| Network | Data | Supervision | Seeds | SNR whole | SNR unlabelled | SNR held out | "
                  "prior whole | prior held out | " + " | ".join(
                      r for r in regions if r != "whole") + " | ms/patch-step |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|" + "---:|" * (len(regions) - 1) + "---:|"]
        combos = {}
        for r in group:
            combos.setdefault((r["network"], r["data"],
                               r.get("supervision_label", r["supervision"])), []).append(r)
        for (n, d, s), rs in sorted(combos.items()):
            per_seed = []
            for r in rs:
                Z = np.load(os.path.join(tdir, r["tag"], "Z_pred.npy")).astype(np.float64)
                per_seed.append({reg: masked_metrics(Z, Z_true, m) for reg, m in regions.items()}
                                | {"held_out": masked_metrics(Z, Z_true, held)})
            def stat(reg):
                v = [p[reg]["snr_db"] for p in per_seed if p[reg]]
                if not v:
                    return (None, None)
                return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None)
            unl = [r["metrics"]["unlabelled"]["snr_db"] for r in rs
                   if r["metrics"].get("unlabelled")]
            entry = {"seeds": [r["seed"] for r in rs],
                     "snr_db": {reg: stat(reg) for reg in list(regions) + ["held_out"]},
                     "snr_db_unlabelled": ((float(np.mean(unl)), float(np.std(unl, ddof=1)) if len(unl) > 1 else None)
                                           if unl else (None, None)),
                     "rmse_whole": float(np.mean([p["whole"]["rmse"] for p in per_seed])),
                     "ms_per_patch_step": float(np.mean([r["ms_per_patch_step"] for r in rs])),
                     "train_seconds": float(np.mean([r["timings"]["train"] for r in rs]))}
            g["networks"][f"{n}_{d}_{s}"] = entry
            prior = g["classical_held_out"][f"admm_{d}"]
            prior_whole = table["classical"][f"admm_{d}"]["whole"]

            def fmt(reg):
                m, sd = entry["snr_db"][reg]
                if m is None:
                    return ""
                return f"{m:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")
            prior_text = f"{prior['snr_db']:.2f}" if prior else ""
            um, usd = entry["snr_db_unlabelled"]
            unl_text = "" if um is None else f"{um:.2f}" + (f" ± {usd:.2f}" if usd is not None else "")
            lines.append(
                f"| {n} | {d} | {s} | {len(rs)} | {fmt('whole')} | {unl_text} | {fmt('held_out')} | "
                f"{prior_whole['snr_db']:.2f} | {prior_text} | "
                + " | ".join(fmt(r) for r in regions if r != "whole")
                + f" | {entry['ms_per_patch_step']:.2f} |")
        table["groups"].append(g)

    write_json(os.path.join(out, "metrics.json"), table)
    with open(os.path.join(out, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # ---- figures: sections, errors, zooms, training coverage ----------------
    def choose(name_list):
        for name in name_list:
            if name in sections:
                return name, sections[name]
        return None, None

    panels = [("True impedance", Z_true), ("Starting model", sections["initial_model"]),
              ("RW-L1 ADMM (noisy)", sections["admm_noisy"])]
    name, Z = choose(["tv2d_noisy"])
    if Z is not None:
        panels.append(("2-D coupled TV (noisy)", Z))
    if runs:
        # figure panels: first group, noisy data, pseudo-well labels before
        # label-free before dense supervision, the smallest seed; the chosen
        # runs are recorded
        first = sorted(groups)[0]
        for n in ("unet", "rescnn", "attention"):
            cands = [r for r in groups[first] if r["network"] == n and r["data"] == "noisy"]
            if not cands:
                continue
            order = {"wells": 0, "none": 1, "dense": 2}
            r = sorted(cands, key=lambda r: (order.get(r["supervision"], 3),
                                             -r.get("n_wells", 0), r["seed"]))[0]
            table["figure_runs"][n] = r["tag"]
            panels.append((f"{r['model_name']} (noisy, {r.get('supervision_label', r['supervision'])})",
                           np.load(os.path.join(tdir, r["tag"], "Z_pred.npy")).astype(np.float64)))
        write_json(os.path.join(out, "metrics.json"), table)
    vmin, vmax = np.percentile(Z_true, [1, 99])
    x0, t0 = offset
    extent = [x0, x0 + nx, (t0 + nt) * dt, t0 * dt]

    def draw_boxes(ax):
        for (a, b, c, d) in CROP_WINDOWS.values():
            ax.add_patch(Rectangle((a, c * dt), b - a, (d - c) * dt, fill=False,
                                   edgecolor="white", linewidth=0.8))

    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.2 * nrow), layout="constrained")
    for ax, (title, Z) in zip(axes.ravel(), panels):
        im = ax.imshow(Z.T.astype(np.float32), aspect="auto", cmap="viridis",
                       extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest")
        m = masked_metrics(Z, Z_true)
        ax.set_title(f"{title}: SNR {m['snr_db']:.2f} dB", fontsize=10)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time (s)")
        draw_boxes(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Impedance")
    fig.savefig(os.path.join(out, "Fig_full_sections.png"), dpi=args.dpi)
    plt.close(fig)

    err_panels = [(t, np.abs(Z - Z_true)) for t, Z in panels[1:]]
    emax = np.percentile(np.abs(sections["admm_noisy"] - Z_true), 99)
    nrow = int(np.ceil(len(err_panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.2 * nrow), layout="constrained")
    for ax, (title, E) in zip(axes.ravel(), err_panels):
        im = ax.imshow(E.T.astype(np.float32), aspect="auto", cmap="magma", extent=extent,
                       vmin=0, vmax=emax, interpolation="nearest")
        ax.set_title(f"|error| {title}", fontsize=10)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time (s)")
        draw_boxes(ax)
    for ax in axes.ravel()[len(err_panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="|error|")
    fig.savefig(os.path.join(out, "Fig_full_errors.png"), dpi=args.dpi)
    plt.close(fig)

    for cname, (a, b, c, d) in CROP_WINDOWS.items():
        a, b, c, d = a - x0, b - x0, c - t0, d - t0
        if a < 0 or b > nx or c < 0 or d > nt:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.5),
                                 layout="constrained")
        for ax, (title, Z) in zip(np.atleast_1d(axes), panels):
            crop = Z[a:b, c:d]
            m = masked_metrics(crop, Z_true[a:b, c:d])
            ax.imshow(crop.T.astype(np.float32), aspect="auto", cmap="viridis",
                      extent=[a + x0, b + x0, (d + t0) * dt, (c + t0) * dt], vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\n{m['snr_db']:.2f} dB", fontsize=9)
            ax.set_xlabel("Trace")
        np.atleast_1d(axes)[0].set_ylabel("Time (s)")
        fig.savefig(os.path.join(out, f"Fig_zoom_{cname}.png"), dpi=args.dpi)
        plt.close(fig)

    for key, group in groups.items():
        r = group[0]
        corners = [tuple(c) for c in r["corners"]]
        cov = coverage_mask(corners, nx, nt, mod.PATCH_NX, mod.PATCH_NT)
        fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
        # vmin/vmax fixed: a mask that is True everywhere would otherwise be
        # drawn as a constant (white) image
        ax.imshow(cov.T.astype(np.float32), aspect="auto", cmap="gray_r", extent=extent,
                  interpolation="nearest", vmin=0.0, vmax=1.0)
        if cov.all():
            # every sample is covered: show the patch tiling instead, one
            # patch outline in every tenth row and column of the grid
            xs = sorted({c[0] for c in corners})[::10]
            ts = sorted({c[1] for c in corners})[::10]
            for cx in xs:
                for ct in ts:
                    ax.add_patch(Rectangle((cx + x0, (ct + t0) * dt), mod.PATCH_NX,
                                           mod.PATCH_NT * dt, fill=False,
                                           edgecolor="orange", linewidth=0.5))
            ax.set_title(f"Training patches ({r['train_set']}, {r['n_patches']} patches of "
                         f"{mod.PATCH_NX} x {mod.PATCH_NT}, stride {mod.STRIDE_NX} x "
                         f"{mod.STRIDE_NT}): every sample covered; every tenth patch "
                         "outlined", fontsize=10)
        else:
            ax.set_title(f"Training patches ({r['train_set']}, {r['n_patches']} patches); "
                         f"white = held out ({100 * (1 - cov.mean()):.1f} %)", fontsize=10)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time (s)")
        draw_boxes(ax)
        name = (f"Fig_coverage_{key[0]}.png" if key[0] == "all"
                else f"Fig_coverage_{key[0]}_{key[1]}_{key[2]}.png")
        fig.savefig(os.path.join(out, name), dpi=args.dpi)
        plt.close(fig)
    print(f"  report written to {out}", flush=True)


# --------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def add_out(q):
        q.add_argument("--out", required=True, help="run folder")

    q = sub.add_parser("prepare", help="model, data, starting model, ADMM prior")
    add_out(q)
    q.add_argument("--marmousi-dir", required=True, help="folder with vp.segy and density.segy")
    q.add_argument("--trace-range", default=None, help="a:b, default all traces")
    q.add_argument("--sample-range", default=None, help="a:b, default all samples")
    q.add_argument("--f0", type=float, default=30.0)
    q.add_argument("--noise-snr", type=float, default=8.0, help="amplitude SNR ratio")
    q.add_argument("--noise-reference", choices=["whole"] + list(CROP_WINDOWS), default="whole",
                   help="window whose RMS defines the noise level")
    q.add_argument("--noise-seed", type=int, default=42)
    q.add_argument("--starting-model", choices=["gaussian", "wells"], default="gaussian",
                   help="gaussian: smoothed true impedance (the paper); wells: low-passed "
                        "true log-impedance at a few traces, interpolated between them")
    q.add_argument("--smooth-sigma-x", type=float, default=8.0, help="traces (gaussian)")
    q.add_argument("--smooth-sigma-t", type=float, default=12.0, help="samples (gaussian)")
    q.add_argument("--well-traces", nargs="+", type=int, default=None,
                   help="well trace indices for --starting-model wells (full-model coordinates); "
                        "default: centres of five equal lateral bins")
    q.add_argument("--well-lowpass", type=float, default=48.0,
                   help="Gaussian width in samples of the time low-pass of the well logs")
    q.add_argument("--epsilon", type=float, default=1e-4,
                   help="reweighting floor of the trace-wise ADMM prior (1e-4 in the paper)")
    q.add_argument("--n-jobs", type=int, default=None, help="worker processes (default all cores)")
    q.add_argument("--chunk", type=int, default=64, help="traces per worker task")
    q.add_argument("--overwrite", action="store_true")

    q = sub.add_parser("train", help="one network, one data set, one seed")
    add_out(q)
    add_train_args(q)
    q.add_argument("--network", choices=list(NETWORKS), required=True)
    q.add_argument("--data", choices=["clean", "noisy"], required=True)
    q.add_argument("--supervision", choices=["dense", "none", "wells"], required=True)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--tag", default=None)

    q = sub.add_parser("baseline", help="2-D coupled RW-L1 + TV classical baseline")
    add_out(q)
    q.add_argument("--n-jobs", type=int, default=None)
    q.add_argument("--chunk", type=int, default=64)
    q.add_argument("--mu-x-factor", type=float, default=100.0, help="mu_x = factor * MU")
    q.add_argument("--rho-factor", type=float, default=40.0, help="rho = factor * ALPHA")
    q.add_argument("--epsilon", type=float, default=None,
                   help="reweighting floor (default: the value used by prepare)")
    q.add_argument("--wells", action="store_true",
                   help="well-conditioned comparator: quadratic penalty towards the true "
                        "log-impedance at the pseudo-well traces (outputs in baseline_wells<n>/)")
    q.add_argument("--n-wells", type=int, default=5)
    q.add_argument("--well-traces", nargs="+", type=int, default=None)
    q.add_argument("--well-weight", type=float, default=1.0,
                   help="weight of the well penalty (ALPHA is 5e-5; 1.0 pins the well traces)")
    q.add_argument("--max-iter", type=int, default=40)
    q.add_argument("--vert-inner", type=int, default=10)
    q.add_argument("--tol", type=float, default=1e-6)
    q.add_argument("--verbose", action="store_true")

    q = sub.add_parser("sweep", help="many train jobs concurrently over the GPUs")
    add_out(q)
    add_train_args(q)
    q.add_argument("--networks", nargs="+", default=["unet", "rescnn"], choices=list(NETWORKS))
    q.add_argument("--datas", nargs="+", default=["clean", "noisy"], choices=["clean", "noisy"])
    q.add_argument("--supervisions", nargs="+", default=["dense", "none", "wells"],
                   choices=["dense", "none", "wells"])
    q.add_argument("--seeds", nargs="+", type=int, default=[42])
    q.add_argument("--gpus", type=int, default=0, help="GPUs to use (0 = no CUDA, one slot group)")
    q.add_argument("--per-gpu", type=int, default=1, help="concurrent trainings per GPU")
    q.add_argument("--with-baseline", action="store_true", help="run the baseline alongside")
    q.add_argument("--baseline-jobs", type=int, default=16)

    q = sub.add_parser("report", help="tables and figures")
    add_out(q)
    q.add_argument("--dpi", type=int, default=200)
    return p


def add_train_args(q):
    q.add_argument("--train-set", choices=["all", "subset"], default="all",
                   help="all: every patch of the model; subset: the crop windows + random patches")
    q.add_argument("--n-random", type=int, default=2000, help="random patches added to the crops")
    q.add_argument("--train-set-seed", type=int, default=1, help="seed of the random patch positions")
    q.add_argument("--epochs", type=int, default=80)
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--lr", type=float, default=None)
    q.add_argument("--base-ch", type=int, default=None)
    q.add_argument("--prior", choices=["rw", "tv"], default="rw",
                   help="network input prior: trace-wise RW-L1 result (paper) or the 2-D coupled result")
    q.add_argument("--lowband", type=float, default=0.0,
                   help="weight of the low-band anchor to the prior (0 = off, as in the paper)")
    q.add_argument("--lowband-sigma", type=float, default=12.0,
                   help="Gaussian width in samples of the low-pass of that anchor")
    q.add_argument("--n-wells", type=int, default=5,
                   help="number of pseudo-wells for --supervision wells, placed at the centres "
                        "of equal lateral bins (ignored when --well-traces is given)")
    q.add_argument("--well-traces", nargs="+", type=int, default=None,
                   help="pseudo-well trace indices (full-model coordinates) for --supervision wells")
    q.add_argument("--transfer-dir", default=None, help="folder with the synthetic pretrain checkpoints")
    q.add_argument("--no-transfer", action="store_true")
    q.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    q.add_argument("--threads", type=int, default=None, help="CPU threads per training process")
    q.add_argument("--predict-batch", type=int, default=256)
    q.add_argument("--tf32", action="store_true", help="allow TF32 on Ampere GPUs (off = full FP32)")
    q.add_argument("--cudnn-benchmark", action="store_true")
    q.add_argument("--overwrite", action="store_true")


def main():
    args = build_parser().parse_args()
    stages = {"prepare": stage_prepare, "train": stage_train, "baseline": stage_baseline,
              "sweep": stage_sweep, "report": stage_report}
    print(f"[{args.stage}] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    stages[args.stage](args)


if __name__ == "__main__":
    main()
