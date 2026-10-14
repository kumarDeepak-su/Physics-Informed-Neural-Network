#!/usr/bin/env python3
"""
Poseidon 3D field example: the runner of protocol version 1
(8_field_poseidon/protocol/PROTOCOL_v1.md).

Phase A, per leave-one-well-out case (the evaluation well's log is never read):
  prior   gain and flat starting model from the calibration wells, trace-wise
          RW-L1 ADMM (M1) and 2-D coupled ADMM (M2) on every line of the case;
  train   one network (U-Net or ResCNN) with one seed on the patches of all ten
          lines, sparse labels on the calibration lines only, prediction of
          every line;
  seal    hashes of every phase-A output.
Phase B, once per case:
  score   the evaluation well's log is read, every method is scored at the
          well trace of the two evaluation lines (section 7 of the protocol).
Then
  report  pooled tables over the cases and the figures.

usage: poseidon_runner.py prior  --prepared DIR --out RUN --case "Proteus 1" [--n-jobs N]
       poseidon_runner.py train  --prepared DIR --out RUN --case "Proteus 1" --network unet --seed 42
                                 [--input tv|rw] [--transfer-dir DIR] [--device cuda] [--epochs 80]
       poseidon_runner.py seal   --prepared DIR --out RUN --case "Proteus 1"
       poseidon_runner.py score  --prepared DIR --out RUN --case "Proteus 1"
       poseidon_runner.py report --prepared DIR --out RUN
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
import admm_banded  # noqa: E402
import run_marmousi_full as rmf  # noqa: E402

WELLS = ["Proteus 1", "Pharos 1", "Poseidon 2", "Kronos 1", "Torosa 1"]     # the case order of the protocol
KINDS = ("inline", "crossline")
def protocol_path(prepared):
    """PROTOCOL_v1.md next to the prepared folder (8_field_poseidon/ locally, field/poseidon/ on the cluster)."""
    return os.path.join(os.path.dirname(os.path.abspath(prepared)), "protocol", "PROTOCOL_v1.md")
EPSILON = 1e-3
BAND_HZ = (12.1, 58.3)
LOWPASS_HZ = 12.0
NULL_MIN_TRACES = 80
METHOD_NAMES = {"B1_start": "Starting model", "M1_admm": "Trace-wise RW-L1 ADMM", "M2_tv": "2-D coupled ADMM",
                "M3_unet_tv": "2D U-Net, 2-D input", "M4_rescnn_tv": "ResCNN, 2-D input",
                "M5_ensemble_tv": "Ensemble U-Net + ResCNN, 2-D input (derived)",
                "M3_unet_rw": "2D U-Net, trace-wise input", "M4_rescnn_rw": "ResCNN, trace-wise input"}


# ----------------------------------------------------------------- records
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tag(well):
    return well.replace(" ", "_")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def software():
    info = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform())
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


class Ledger:
    """Every file read in a phase, with its hash; a must-not-read list that refuses the evaluation well."""

    def __init__(self, forbidden=()):
        self.forbidden = [os.path.abspath(p) for p in forbidden]
        self.read = {}

    def load(self, path):
        ap = os.path.abspath(path)
        if ap in self.forbidden:
            raise SystemExit(f"REFUSED: {path} is on the must-not-read list of this phase")
        self.read[os.path.relpath(ap)] = sha256(ap)
        return np.load(ap, allow_pickle=False)


# ----------------------------------------------------------------- data of a case
def case_wells(evaluation):
    return [w for w in WELLS if w != evaluation]


def line_path(prepared, well, kind):
    return os.path.join(prepared, "lines", f"{tag(well)}_{kind}.npz")


def well_path(prepared, well):
    return os.path.join(prepared, "wells", f"{tag(well)}.npz")


def synthetic_trace(lnz, w, c):
    Z = np.exp(lnz)
    r = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
    r = np.concatenate([r, [0.0]])
    return np.convolve(r, w)[c:c + lnz.size]


def well_gain(ledger, prepared, well, wavelet, half):
    """data RMS / synthetic RMS over the tie rows of a calibration well, inline through the well, 5 x 5 mean."""
    d = ledger.load(well_path(prepared, well))
    L = ledger.load(line_path(prepared, well, "inline"))
    nt = int(L["nt"])
    S = L["S_ext"][:, half:half + nt].astype(np.float64)
    wt = int(L["well_trace"])
    tr = S[max(0, wt - 2):wt + 3].mean(axis=0)
    lnz, mask, tie = d["lnz"], d["log_rows"], d["tie_rows"]
    fill = np.where(np.isfinite(lnz), lnz, np.interp(np.arange(nt), np.flatnonzero(mask), lnz[mask]))
    syn = synthetic_trace(fill, wavelet, half)
    return float(np.sqrt(np.mean(tr[tie] ** 2)) / np.sqrt(np.mean(syn[tie] ** 2)))


def flat_starting_model(ledger, prepared, wells, nt):
    """Mean of the calibration wells' low-passed log-impedance per sample, end values held outside."""
    acc = np.zeros(nt)
    cnt = np.zeros(nt)
    for w in wells:
        lp = ledger.load(well_path(prepared, w))["lnz_lowpass"]
        ok = np.isfinite(lp)
        acc[ok] += lp[ok]
        cnt[ok] += 1
    prof = np.full(nt, np.nan)
    prof[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    idx = np.flatnonzero(cnt > 0)
    prof[:idx[0]] = prof[idx[0]]
    prof[idx[-1] + 1:] = prof[idx[-1]]
    gaps = np.isnan(prof)
    if gaps.any():
        prof[gaps] = np.interp(np.flatnonzero(gaps), np.flatnonzero(~gaps), prof[~gaps])
    return prof, (int(idx[0]), int(idx[-1] + 1))


def weight_scale(prepared):
    return float(read_json(os.path.join(prepared, "survey_wavelet.json"))["s_n"])


def load_wavelet(ledger, prepared):
    w = ledger.load(os.path.join(prepared, "wavelets", "survey_unit_peak.npy"))
    return -np.asarray(w, dtype=np.float64)          # declared polarity


# ----------------------------------------------------------------- phase A: prior
def stage_prior(args):
    mod = rmf.load_main_module()
    case = args.case
    cal = case_wells(case)
    out = os.path.join(args.out, "cases", tag(case), "phaseA")
    os.makedirs(os.path.join(out, "lines"), exist_ok=True)
    ledger = Ledger(forbidden=[well_path(args.prepared, case)])
    w = load_wavelet(ledger, args.prepared)
    half = (w.size - 1) // 2
    s_n = weight_scale(args.prepared)
    gains = {c: well_gain(ledger, args.prepared, c, w, half) for c in cal}
    g = float(np.exp(np.mean([np.log(v) for v in gains.values()])))
    nt = int(np.load(line_path(args.prepared, cal[0], "inline"))["nt"])
    prof, lfm_rows = flat_starting_model(ledger, args.prepared, cal, nt)
    mu, alpha, lam = mod.MU * s_n, mod.ALPHA * s_n, mod.LAMBDA_ * s_n
    params = dict(mu=mu, alpha=alpha, lambda_=lam, epsilon=EPSILON, max_iter=mod.MAX_ITER_CL, tol=mod.TOL_CL,
                  tv=dict(mu_x=100.0 * mu, rho=40.0 * alpha, max_iter=40, vert_inner=10, tol=1e-6), s_n=s_n)
    n_jobs = args.n_jobs or os.cpu_count()
    record = dict(case=case, calibration_wells=cal, gains=gains, gain_case=g, params=params,
                  starting_model_rows=lfm_rows, wavelet="survey_unit_peak.npy (negated)", n_jobs=n_jobs,
                  protocol_sha256=sha256(protocol_path(args.prepared)), software=software(), lines={}, started_utc=now())
    np.save(os.path.join(out, "starting_model_profile.npy"), prof)
    print(f"case {case}: calibration {cal}, gain {g:.4g} (wells {', '.join(f'{v:.3g}' for v in gains.values())}), "
          f"s_n {s_n:.4f}, LFM rows {lfm_rows}", flush=True)
    for well in WELLS:
        for kind in KINDS:
            if args.only and f"{tag(well)}_{kind}" != args.only:
                continue
            L = ledger.load(line_path(args.prepared, well, kind))
            S_ext = L["S_ext"].astype(np.float64) / g
            nx = S_ext.shape[0]
            L0 = np.repeat(prof[None, :], nx, axis=0)
            t = time.time()
            Z_rw, iters = admm_banded.admm_l1_traces(S_ext, w, L0, mu, alpha, lam, epsilon=EPSILON,
                                                     max_iter=mod.MAX_ITER_CL, tol=mod.TOL_CL, reweight=True,
                                                     n_jobs=n_jobs, chunk=64)
            t_rw = time.time() - t
            t = time.time()
            Z_tv, conv = admm_banded.admm_l1_2d_tv_banded(S_ext, w, L0, mu, alpha, lam, mu_x=params["tv"]["mu_x"],
                                                          rho=params["tv"]["rho"], epsilon=EPSILON,
                                                          max_iter=params["tv"]["max_iter"], tol=params["tv"]["tol"],
                                                          vert_inner=params["tv"]["vert_inner"], reweight=True,
                                                          n_jobs=n_jobs, chunk=64)
            t_tv = time.time() - t
            name = f"{tag(well)}_{kind}"
            path = os.path.join(out, "lines", name + ".npz")
            np.savez(path, S=S_ext[:, half:half + nt].astype(np.float32), S_ext=S_ext.astype(np.float32),
                     L0=L0.astype(np.float32), L_rw=np.log(Z_rw).astype(np.float32), L_tv=np.log(Z_tv).astype(np.float32),
                     well_trace=int(L["well_trace"]), t0=float(L["t0"]), dt=float(L["dt"]), nt=nt, half_pad=half,
                     well=well, kind=kind, evaluation=(well == case))
            record["lines"][name] = dict(well=well, kind=kind, evaluation=(well == case), nx=int(nx),
                                         admm_seconds=t_rw, admm_iterations_mean=float(iters.mean()),
                                         admm_at_limit=int((iters >= mod.MAX_ITER_CL).sum()), tv_seconds=t_tv,
                                         tv_outer_iterations=len(conv) if hasattr(conv, "__len__") else None,
                                         sha256=sha256(path))
            print(f"  {name:24s} nx {nx}: ADMM {t_rw:5.1f} s ({iters.mean():.0f} it), TV {t_tv:5.1f} s", flush=True)
    rec_path = os.path.join(out, "prior_record.json")
    if args.only and os.path.exists(rec_path):                      # debug runs accumulate their lines
        previous = read_json(rec_path)
        previous["lines"].update(record["lines"])
        record["lines"] = previous["lines"]
    record["ledger"] = ledger.read
    record["finished_utc"] = now()
    write_json(rec_path, record)
    print("prior stage written to", out)


# ----------------------------------------------------------------- phase A: train
def label_field(ledger, prepared, well, kind, nx, nt):
    """Sparse label of a calibration line: ln Z of its well on the well column over the log rows."""
    d = ledger.load(well_path(prepared, well))
    L = np.load(line_path(prepared, well, kind))
    wt = int(L["well_trace"])
    Lt = np.zeros((nx, nt), dtype=np.float64)
    M = np.zeros((nx, nt), dtype=np.float64)
    rows = d["log_rows"]
    Lt[wt, rows] = d["lnz"][rows]
    M[wt, rows] = 1.0
    return Lt, M


def stage_train(args):
    import torch
    mod = rmf.load_main_module()
    mod.DEVICE = torch.device(args.device if args.device != "auto" else
                              ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    case = args.case
    cal = case_wells(case)
    out_a = os.path.join(args.out, "cases", tag(case), "phaseA")
    rec_prior = read_json(os.path.join(out_a, "prior_record.json"))
    ledger = Ledger(forbidden=[well_path(args.prepared, case)])
    w = load_wavelet(ledger, args.prepared)
    s_n = rec_prior["params"]["s_n"]
    key = "L_tv" if args.input == "tv" else "L_rw"
    px, pt, sx, st = mod.PATCH_NX, mod.PATCH_NT, mod.STRIDE_NX, mod.STRIDE_NT
    S_p, L0_p, Lt_p, M_p, per_line = [], [], [], [], {}
    for well in WELLS:
        for kind in KINDS:
            name = f"{tag(well)}_{kind}"
            if name not in rec_prior["lines"]:
                continue
            Lz = ledger.load(os.path.join(out_a, "lines", name + ".npz"))
            S, Lp = Lz["S"].astype(np.float64), Lz[key].astype(np.float64)
            nx, nt = S.shape
            corners = mod.compute_patch_corners(nx, nt, px, pt, sx, st)
            S_p.append(mod.extract_patches_at_corners(S, corners, px, pt))
            L0_p.append(mod.extract_patches_at_corners(Lp, corners, px, pt))
            if well in cal:
                Lt, M = label_field(ledger, args.prepared, well, kind, nx, nt)
            else:
                Lt, M = np.zeros((nx, nt)), np.zeros((nx, nt))
            Lt_p.append(mod.extract_patches_at_corners(Lt, corners, px, pt))
            M_p.append(mod.extract_patches_at_corners(M, corners, px, pt))
            per_line[name] = dict(n_patches=len(corners), labelled_patches=int((M_p[-1].sum(axis=(1, 2)) > 0).sum()),
                                  corners=corners, S=S, Lp=Lp, nx=nx, nt=nt)
    S_p, L0_p, Lt_p, M_p = (np.concatenate(a, axis=0) for a in (S_p, L0_p, Lt_p, M_p))
    class_name, model_name, prefix = rmf.NETWORKS[args.network]
    ckpt = os.path.join(args.transfer_dir, f"{prefix}_noisy_pretrain.pt") if args.transfer_dir else None
    if ckpt and not os.path.exists(ckpt):
        raise SystemExit(f"transfer checkpoint missing: {ckpt}")
    mod.W_PHYSICS = 3.0 / s_n
    mod.EPOCHS = args.epochs
    mod.set_seed(args.seed)
    label = f"{args.network}_{args.input}_s{args.seed}"
    out = os.path.join(out_a, f"net_{label}")
    os.makedirs(out, exist_ok=True)
    t = time.time()
    net, history = mod.train_model_2d(S_p, L0_p, w, px, pt, label=label,
                                      checkpoint_path=os.path.join(out, "checkpoint.pt"), L_true_patches=Lt_p,
                                      model_class=getattr(mod, class_name), model_name=model_name,
                                      transfer_checkpoint=ckpt, label_mask_patches=M_p)
    t_train = time.time() - t
    outputs = {}
    for name, info in per_line.items():
        Z = rmf.predict_section_batched(mod, net, info["S"], info["Lp"], info["corners"], info["nx"], info["nt"])
        path = os.path.join(out, name + ".npy")
        np.save(path, np.log(Z).astype(np.float32))
        outputs[name + ".npy"] = sha256(path)
    np.save(os.path.join(out, "loss_history.npy"), np.asarray(history))
    record = dict(case=case, network=args.network, input=args.input, seed=args.seed, epochs=args.epochs,
                  n_patches=int(S_p.shape[0]), labelled_patches=int((M_p.sum(axis=(1, 2)) > 0).sum()),
                  labelled_samples=int(M_p.sum()), lines={k: {kk: v[kk] for kk in ("n_patches", "labelled_patches")}
                                                          for k, v in per_line.items()},
                  w_physics=mod.W_PHYSICS, s_n=s_n, transfer_checkpoint=ckpt,
                  transfer_sha256=sha256(ckpt) if ckpt else None, device=str(mod.DEVICE), train_seconds=t_train,
                  outputs=outputs, ledger=ledger.read, protocol_sha256=sha256(protocol_path(args.prepared)), software=software(),
                  finished_utc=now())
    write_json(os.path.join(out, "train_record.json"), record)
    print(f"{label}: {S_p.shape[0]} patches ({record['labelled_patches']} labelled, {record['labelled_samples']} "
          f"labelled samples), {t_train:.0f} s on {mod.DEVICE}; written to {out}", flush=True)


def stage_seal(args):
    out_a = os.path.join(args.out, "cases", tag(args.case), "phaseA")
    files = sorted(glob.glob(os.path.join(out_a, "**", "*"), recursive=True))
    seal = {os.path.relpath(f, out_a): sha256(f) for f in files if os.path.isfile(f) and not f.endswith("phaseA_sealed.json")}
    write_json(os.path.join(out_a, "phaseA_sealed.json"), dict(case=args.case, sealed_utc=now(), files=seal))
    print(f"sealed {len(seal)} phase-A files of {args.case}")


# ----------------------------------------------------------------- phase B: score
def bandpass(x, dt, lo, hi):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [lo * 2 * dt, hi * 2 * dt], btype="band")
    return filtfilt(b, a, x)


def lowpass(x, dt, fc):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, fc * 2 * dt, btype="low")
    return filtfilt(b, a, x)


def envelope(x):
    from scipy.signal import hilbert
    return np.abs(hilbert(x))


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fractional_shift(x, shift_samples):
    """Shift a series by a fractional number of samples (positive = later) with a Fourier phase ramp."""
    n = x.size
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * shift_samples), n=n)


def metrics_at(target, pred, L0col, rows, dt):
    """Section 7: metrics of one predicted column against the target on the tie rows."""
    r0, r1 = rows
    tg, pr, l0 = target[r0:r1], pred[r0:r1], L0col[r0:r1]
    tb, pb = bandpass(tg, dt, *BAND_HZ), bandpass(pr, dt, *BAND_HZ)
    return dict(r_band=corr(tb, pb), r_envelope=corr(envelope(tb), envelope(pb)),
                r_low=corr(lowpass(tg - l0, dt, LOWPASS_HZ), lowpass(pr - l0, dt, LOWPASS_HZ)),
                rel_rmse=float(np.sqrt(np.mean((pr - tg) ** 2)) / np.std(tg)), r_full=corr(tg, pr), rows=int(r1 - r0))


def stage_score(args):
    case = args.case
    out_a = os.path.join(args.out, "cases", tag(case), "phaseA")
    out_b = os.path.join(args.out, "cases", tag(case), "phaseB")
    os.makedirs(out_b, exist_ok=True)
    seal = read_json(os.path.join(out_a, "phaseA_sealed.json"))
    for rel, h in seal["files"].items():
        if sha256(os.path.join(out_a, rel)) != h:
            raise SystemExit(f"REFUSED: phase-A file changed after sealing: {rel}")
    d = np.load(well_path(args.prepared, case))
    lnz, tie = d["lnz"], d["tie_rows"]
    idx = np.flatnonzero(tie)
    rows = (int(idx[0]), int(idx[-1] + 1))
    target = np.where(np.isfinite(lnz), lnz, np.interp(np.arange(lnz.size), np.flatnonzero(np.isfinite(lnz)), lnz[np.isfinite(lnz)]))
    lag_ms = float(d["tie_lag_subsample_ms"])
    vsp_path = os.path.join(args.prepared, "vsp_alignment.json")
    vsp_ms = float(read_json(vsp_path)["wells"][case]["shift_ms"]) if os.path.exists(vsp_path) else None
    scores = dict(case=case, rows=rows, lag_declared_ms=lag_ms, lines={}, finished_utc=now(),
                  phaseA_seal_sha256=sha256(os.path.join(out_a, "phaseA_sealed.json")),
                  own_lag_sensitivity="target shifted by the well's own verification lag (in-sample; declared sensitivity, section 3)",
                  vsp_alignment_ms=vsp_ms, vsp_alignment="target shifted by the VSP-derived shift (corridor stack against the surface trace; addendum 2)")
    nets = sorted(glob.glob(os.path.join(out_a, "net_*")))
    for kind in KINDS:
        name = f"{tag(case)}_{kind}"
        Lz = np.load(os.path.join(out_a, "lines", name + ".npz"))
        wt, dt = int(Lz["well_trace"]), float(Lz["dt"])
        preds = {"B1_start": Lz["L0"].astype(np.float64), "M1_admm": Lz["L_rw"].astype(np.float64),
                 "M2_tv": Lz["L_tv"].astype(np.float64)}
        for nd in nets:
            rec = read_json(os.path.join(nd, "train_record.json"))
            key = f"{'M3_unet' if rec['network'] == 'unet' else 'M4_rescnn'}_{rec['input']}"
            preds.setdefault(key + "__seeds", []).append(np.load(os.path.join(nd, name + ".npy")).astype(np.float64))
        L0col = Lz["L0"][wt].astype(np.float64)
        target_lag = fractional_shift(target, lag_ms / 1000.0 / dt)
        target_vsp = fractional_shift(target, vsp_ms / 1000.0 / dt) if vsp_ms is not None else None
        def with_vsp(sec_col, rec):
            if target_vsp is not None:
                rec["vsp_lag"] = metrics_at(target_vsp, sec_col, L0col, rows, dt)
            return rec
        entry = {}
        for key, val in list(preds.items()):
            if key.endswith("__seeds"):
                base = key[:-7]
                per_seed = [metrics_at(target, sec[wt], L0col, rows, dt) for sec in val]
                agg = {m: dict(mean=float(np.mean([p[m] for p in per_seed])), std=float(np.std([p[m] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
                       for m in ("r_band", "r_envelope", "r_low", "rel_rmse", "r_full")}
                mean_sec = np.mean(np.stack(val), axis=0)
                null = spatial_null(target, mean_sec, L0col, rows, dt, wt)
                def seed_agg(tgt):
                    # the same aggregation as the primary columns: score every seed, then mean and std over the seeds
                    ps = [metrics_at(tgt, sec[wt], L0col, rows, dt) for sec in val]
                    return {m: dict(mean=float(np.mean([p[m] for p in ps])), std=float(np.std([p[m] for p in ps], ddof=1)) if len(ps) > 1 else 0.0)
                            for m in ("r_band", "r_envelope", "r_low", "rel_rmse", "r_full")}
                rec = dict(seeds=len(per_seed), per_seed=per_seed, **agg, seed_mean_section=metrics_at(target, mean_sec[wt], L0col, rows, dt), null=null,
                           own_lag=seed_agg(target_lag), seed_mean_section_own_lag=metrics_at(target_lag, mean_sec[wt], L0col, rows, dt),
                           aggregation="every alignment column is the mean of the per-seed scores (standardized 2026-10-12); the seed-mean-section scores are kept for the record")
                if target_vsp is not None:
                    rec["vsp_lag"] = seed_agg(target_vsp)
                    rec["seed_mean_section_vsp_lag"] = metrics_at(target_vsp, mean_sec[wt], L0col, rows, dt)
                entry[base] = rec
            else:
                m = metrics_at(target, val[wt], L0col, rows, dt)
                entry[key] = with_vsp(val[wt], dict(**m, null=spatial_null(target, val, L0col, rows, dt, wt), own_lag=metrics_at(target_lag, val[wt], L0col, rows, dt)))
        if "M3_unet_tv" in entry and "M4_rescnn_tv" in entry:
            ens = 0.5 * (np.mean(np.stack(preds["M3_unet_tv__seeds"]), axis=0) + np.mean(np.stack(preds["M4_rescnn_tv__seeds"]), axis=0))
            entry["M5_ensemble_tv"] = with_vsp(ens[wt], dict(**metrics_at(target, ens[wt], L0col, rows, dt), null=spatial_null(target, ens, L0col, rows, dt, wt),
                                                              own_lag=metrics_at(target_lag, ens[wt], L0col, rows, dt),
                                                              note="derived from the sealed seed-mean sections of M3 and M4 (protocol v1 addendum), no new inversion"))
        scores["lines"][name] = dict(well_trace=wt, methods=entry)
        print(f"{name}: " + "; ".join(f"{k} band {v.get('r_band', v.get('r_band', {}))}" for k, v in entry.items() if not isinstance(v.get("r_band"), dict)), flush=True)
    write_json(os.path.join(out_b, "scores.json"), scores)
    print("scores written to", out_b)


def spatial_null(target, section, L0col, rows, dt, wt):
    """Metrics of the prediction at every trace at least NULL_MIN_TRACES from the well: p = fraction >= well."""
    nx = section.shape[0]
    far = [x for x in range(nx) if abs(x - wt) >= NULL_MIN_TRACES]
    here = metrics_at(target, section[wt], L0col, rows, dt)
    vals = {m: [] for m in ("r_band", "r_envelope", "r_full")}
    for x in far:
        m = metrics_at(target, section[x], L0col, rows, dt)
        for k in vals:
            vals[k].append(m[k])
    out = {}
    for k, v in vals.items():
        v = np.asarray(v)
        out[k] = dict(p=float(np.mean(v >= here[k])), q95=float(np.nanquantile(v, 0.95)), mean=float(np.nanmean(v)), n=len(far))
    return out


# ----------------------------------------------------------------- report
def stage_report(args):
    cases = sorted(glob.glob(os.path.join(args.out, "cases", "*", "phaseB", "scores.json")))
    table = {}
    for path in cases:
        sc = read_json(path)
        for line, ent in sc["lines"].items():
            for key, m in ent["methods"].items():
                row = table.setdefault(key, [])
                val = {k: (m[k]["mean"] if isinstance(m[k], dict) else m[k]) for k in ("r_band", "r_envelope", "r_low", "rel_rmse", "r_full")}
                val["rows"] = m["rows"] if "rows" in m else m["seed_mean_section"]["rows"]
                val["line"] = line
                val["p_band"] = m["null"]["r_band"]["p"]
                val["p_env"] = m["null"]["r_envelope"]["p"]
                mv = lambda x: x["mean"] if isinstance(x, dict) else x      # networks: mean over seeds; single sections: the value
                val["r_band_ownlag"] = mv(m.get("own_lag", {}).get("r_band"))
                val["r_envelope_ownlag"] = mv(m.get("own_lag", {}).get("r_envelope"))
                val["r_band_vsp"] = mv(m.get("vsp_lag", {}).get("r_band"))
                val["r_envelope_vsp"] = mv(m.get("vsp_lag", {}).get("r_envelope"))
                val["rel_rmse_vsp"] = mv(m.get("vsp_lag", {}).get("rel_rmse"))
                row.append(val)
    lines = ["# Poseidon field example: pooled scores (protocol v1)", "",
             "Row-weighted means over the evaluation lines (two per withheld well); networks: seed means.", "",
             "| Method | lines | r band | r envelope | r low | rel. RMSE | r full | p(band) median | p(env) median |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pooled = {}
    for key in METHOD_NAMES:
        rows = table.get(key)
        if not rows:
            continue
        wts = np.array([r["rows"] for r in rows], float)
        agg = {m: float(np.sum(wts * np.array([r[m] for r in rows])) / wts.sum()) for m in ("r_band", "r_envelope", "r_low", "rel_rmse", "r_full")}
        for m in ("r_band_ownlag", "r_envelope_ownlag", "r_band_vsp", "r_envelope_vsp", "rel_rmse_vsp"):
            vals = [r[m] for r in rows if r.get(m) is not None]
            agg[m] = float(np.sum(wts[:len(vals)] * np.array(vals)) / wts[:len(vals)].sum()) if vals else None
        agg["p_band_median"] = float(np.median([r["p_band"] for r in rows]))
        agg["p_env_median"] = float(np.median([r["p_env"] for r in rows]))
        agg["spread"] = {m: [float(min(r[m] for r in rows)), float(max(r[m] for r in rows))] for m in ("r_band", "r_envelope", "r_full")}
        pooled[key] = dict(n_lines=len(rows), **agg, per_line={r["line"]: {k: r.get(k) for k in ("r_band", "r_envelope", "r_low", "rel_rmse", "r_full", "p_band", "p_env", "r_band_ownlag", "r_envelope_ownlag", "r_band_vsp", "r_envelope_vsp", "rel_rmse_vsp")} for r in rows})
        f3 = lambda v: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
        lines.append(f"| {METHOD_NAMES[key]} | {len(rows)} | {f3(agg['r_band'])} | {f3(agg['r_envelope'])} | {f3(agg['r_low'])} | "
                     f"{agg['rel_rmse']:.3f} | {agg['r_full']:.3f} | {agg['p_band_median']:.3f} | {agg['p_env_median']:.3f} |")
    lines += ["", "## Per line", "", "| Line | Method | r band | r envelope | r low | rel. RMSE | r full | p(band) | p(env) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, rows in table.items():
        for r in rows:
            f3 = lambda v: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
            lines.append(f"| {r['line']} | {METHOD_NAMES.get(key, key)} | {f3(r['r_band'])} | {f3(r['r_envelope'])} | {f3(r['r_low'])} | "
                         f"{f3(r['rel_rmse'])} | {f3(r['r_full'])} | {f3(r['p_band'])} | {f3(r['p_env'])} |")
    os.makedirs(os.path.join(args.out, "report"), exist_ok=True)
    write_json(os.path.join(args.out, "report", "pooled.json"), dict(pooled=pooled, cases=len(cases), finished_utc=now()))
    with open(os.path.join(args.out, "report", "pooled.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print("report written to", os.path.join(args.out, "report"))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("prior", "train", "seal", "score", "report"):
        q = sub.add_parser(name)
        q.add_argument("--prepared", required=True)
        q.add_argument("--out", required=True)
        if name != "report":
            q.add_argument("--case", required=True, choices=WELLS)
        if name == "prior":
            q.add_argument("--n-jobs", type=int, default=None)
            q.add_argument("--only", default=None, help="debug: one line name, e.g. Proteus_1_inline")
        if name == "train":
            q.add_argument("--network", choices=["unet", "rescnn"], required=True)
            q.add_argument("--seed", type=int, default=42)
            q.add_argument("--input", choices=["tv", "rw"], default="tv")
            q.add_argument("--transfer-dir", default=os.path.join(CODE, "SYNTHETIC_PRETRAIN"))
            q.add_argument("--device", default="auto")
            q.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    {"prior": stage_prior, "train": stage_train, "seal": stage_seal, "score": stage_score, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
