#!/usr/bin/env python3
"""
Poseidon 3D field data: the inversion lines, the well logs in time, the
per-block wavelets and the calibration quantities, prepared once and hashed.

For every tied well (Kronos 1, Pharos 1, Poseidon 2, Proteus 1, Torosa 1) the
script writes
  lines/<well>_inline.npz     the inline through the well, every crossline of
                              the extracted block (512 traces), time window
                              [t0, t1) extended by the wavelet half-length on
                              both sides (S_ext), the well trace index and the
                              bin coordinates;
  lines/<well>_crossline.npz  the crossline through the well (384 traces);
  wells/<well>.npz            log-impedance in two-way time on the 4 ms axis of
                              the window (NaN outside the log rows), the
                              low-passed log for the starting model, the data
                              and synthetic RMS over the log rows at the well
                              (gain calibration) and the declared tie lag;
  wavelets/<well>.npy         the block's zero-phase statistical wavelet
                              (verification item v05, unit peak, 129 taps,
                              centre 64) with the declared polarity applied;
  manifest.json               every input and output with its SHA-256;
  PREPARE_RECORD.md           the facts in words.

Everything comes from the verification evidence (v04 time-depth sources, v05
wavelets, v06 tie lags) and the raw files it points to; nothing is fitted here.
The log-impedance rendering and the forward convention are those of the
verification pass (vlib / v06): r[q] = (Z[q+1]-Z[q])/(Z[q+1]+Z[q]) at sample q,
synthetic = convolve(r, w)[c:c+N] with w[c] at zero lag, data compared with
the negated wavelet (SEG reverse polarity, trough = impedance increase).

usage: python prepare_poseidon.py --out DIR [--t0 1.5 --t1 4.0] [--lowpass-sigma 16]
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
VERIFY = os.path.join(MANUSCRIPT, "DATA_POST_STACK", "POSEIDON_FIELD_INVERSION", "verify")
sys.path.insert(0, os.path.join(VERIFY, "scripts"))
import vlib as V  # noqa: E402

TIED_WELLS = ["Kronos 1", "Pharos 1", "Poseidon 2", "Proteus 1", "Torosa 1"]
HALF = 64          # wavelet half-length in samples (129 taps, centre 64)
NW = 129


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def impedance_in_time(well, v04):
    """AI(t) [kg m-2 s-1] on the delivered 1 ms time axis, restricted to the joint density + sonic window.
    Same construction as verification item v06."""
    d = os.path.join(V.WELLS_DIR, V.well_tag(well))
    w4 = v04["wells"][well]
    if w4.get("time_domain_logs"):
        path = os.path.join(V.RAW, w4["time_domain_logs"])
        c, _, _ = V.las_curves(path)
        t = c["TIME"]
        dtc = c.get("DTCO", c.get("DT"))
        rho = c["RHOB"]
        ok = np.isfinite(t) & np.isfinite(dtc) & np.isfinite(rho) & (rho > 1.5)
        vel = 1e6 / dtc[ok] * 0.3048
        return t[ok], rho[ok] * 1000.0 * vel, path
    if well == "Torosa 1":
        path = os.path.join(d, "VSP", "TZV_DEPTH_CS_CALIBRATED_LOGS.LAS")
        c, _, _ = V.las_curves(path)
        md = c["MD"]
        t = c["TIME"] / 1000.0
        dtc = c["DTC_CS"]
        rho = c["RHO_CS"]
        ok = np.isfinite(t) & np.isfinite(dtc) & np.isfinite(rho) & (md >= 3577.0) & (md <= 4666.0)
        vel = 1e6 / dtc[ok] * 0.3048
        return t[ok], rho[ok] * 1000.0 * vel, path
    raise SystemExit(f"{well}: no delivered impedance in time")


def render_log(t_log, ai_log, t_axis, dt):
    """Log-impedance on the seismic axis: mean of ln(AI) over [t - dt/2, t + dt/2), rows with at least half of
    the bin covered; NaN elsewhere."""
    lnai = np.log(ai_log)
    out = np.full(t_axis.size, np.nan)
    order = np.argsort(t_log)
    t_log, lnai = t_log[order], lnai[order]
    step = np.median(np.diff(t_log))
    need = max(1, int(round(0.5 * dt / step)))
    for j, t in enumerate(t_axis):
        a, b = np.searchsorted(t_log, t - dt / 2), np.searchsorted(t_log, t + dt / 2)
        if b - a >= need:
            out[j] = lnai[a:b].mean()
    return out


def synthetic(lnz, w, c):
    """Zero-phase synthetic of a log-impedance trace (the verification convention)."""
    Z = np.exp(lnz)
    r = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
    r = np.concatenate([r, [0.0]])
    return np.convolve(r, w)[c:c + lnz.size]


def contiguous_rows(mask):
    """Largest contiguous run of True rows (the log window on the seismic axis)."""
    best, cur = (0, 0), None
    for j, m in enumerate(mask):
        if m and cur is None:
            cur = j
        if (not m or j == mask.size - 1) and cur is not None:
            end = j + 1 if m else j
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=1.5, help="window start, s")
    ap.add_argument("--t1", type=float, default=4.0, help="window end, s (exclusive)")
    ap.add_argument("--lowpass-sigma", type=float, default=16.0,
                    help="Gaussian width (samples) of the starting-model low-pass; 16 samples at 4 ms = 2.9 Hz")
    ap.add_argument("--edge-s", type=float, default=0.064, help="wavelet-edge exclusion of the tie rows, s")
    args = ap.parse_args()
    for sub in ("lines", "wells", "wavelets", "figures"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    from scipy.ndimage import gaussian_filter1d

    v04 = read_json(os.path.join(V.EVID, "v04_time_depth.json"))
    v05 = read_json(os.path.join(V.EVID, "v05_seismic_band.json"))
    v06 = read_json(os.path.join(V.EVID, "v06_ties.json"))
    rec = read_json(os.path.join(V.SEISMIC, "near_stack_blocks_record.json"))
    dt = float(V.NOMINAL_DT)
    k0 = int(round(args.t0 / dt))
    nt = int(round((args.t1 - args.t0) / dt))
    t_axis = args.t0 + dt * np.arange(nt)
    inputs = {os.path.relpath(p, V.RAW): sha256(p) for p in
              [os.path.join(V.SEISMIC, "near_stack_blocks_record.json")]}
    for name in ("v04_time_depth", "v05_seismic_band", "v06_ties"):
        inputs[f"verify/evidence/{name}.json"] = sha256(os.path.join(V.EVID, name + ".json"))
    outputs, facts = {}, {}

    for well in TIED_WELLS:
        tag = V.well_tag(well)
        block, hdr, blk, _ = V.seismic_block(well)
        il_idx = int(blk["il"] - blk["il_range"][0])
        xl_idx = int(blk["xl"] - blk["xl_range"][0])
        assert block.shape[0] == 384 and block.shape[1] == 512 and block.shape[2] == 1501
        a, b = k0 - HALF, k0 + nt + HALF - 1
        assert a >= 0 and b <= block.shape[2], "window plus wavelet margins outside the block"
        for kind, S_full, wt, il, xl, x, y in (
                ("inline", np.asarray(block[il_idx, :, :]), xl_idx, hdr["inline"][il_idx, :], hdr["crossline"][il_idx, :],
                 hdr["cdp_x"][il_idx, :], hdr["cdp_y"][il_idx, :]),
                ("crossline", np.asarray(block[:, xl_idx, :]), il_idx, hdr["inline"][:, xl_idx], hdr["crossline"][:, xl_idx],
                 hdr["cdp_x"][:, xl_idx], hdr["cdp_y"][:, xl_idx])):
            S_ext = np.ascontiguousarray(S_full[:, a:b], dtype=np.float32)      # nt + 2*HALF - 1 samples
            path = os.path.join(args.out, "lines", f"{tag}_{kind}.npz")
            np.savez(path, S_ext=S_ext, well_trace=wt, t0=args.t0, dt=dt, nt=nt, half_pad=HALF,
                     inline=il.astype(np.int32), crossline=xl.astype(np.int32), cdp_x=x, cdp_y=y,
                     block_file=blk["file"], well=well, kind=kind)
            outputs[os.path.relpath(path, args.out)] = sha256(path)
        inputs[f"seismic/{blk['file']}"] = blk["sha256"]
        inputs[f"seismic/{blk['file'].replace('.npy', '_headers.npz')}"] = sha256(
            os.path.join(V.SEISMIC, blk["file"].replace(".npy", "_headers.npz")))

        # wavelet of the block with the declared polarity
        w_unit = np.asarray(v05["blocks"][well]["wavelet_zero_phase_unit_peak"], dtype=np.float64)
        c = int(v05["blocks"][well]["wavelet_centre_index"])
        assert w_unit.size == NW and c == HALF
        w_used = -w_unit
        wpath = os.path.join(args.out, "wavelets", f"{tag}.npy")
        np.save(wpath, w_used)
        outputs[os.path.relpath(wpath, args.out)] = sha256(wpath)

        # log-impedance in time on the window axis
        t_log, ai_log, src = impedance_in_time(well, v04)
        inputs[os.path.relpath(src, V.RAW)] = sha256(src)
        lnz = render_log(t_log, ai_log, t_axis, dt)
        rows = contiguous_rows(np.isfinite(lnz))
        mask = np.zeros(nt, bool)
        mask[rows[0]:rows[1]] = True
        lnz_win = np.where(mask, lnz, np.nan)
        # low-passed log for the flat starting model (inside the log rows only)
        lp = np.full(nt, np.nan)
        seg = lnz[rows[0]:rows[1]]
        seg = np.where(np.isfinite(seg), seg, np.interp(np.arange(seg.size), np.flatnonzero(np.isfinite(seg)),
                                                        seg[np.isfinite(seg)]))
        lp[rows[0]:rows[1]] = gaussian_filter1d(seg, args.lowpass_sigma, mode="nearest")
        # tie quantities at the well: the inline through the well, 5 x 5 bin mean trace, log rows minus the
        # wavelet edge, synthetic from the rendered log with the block wavelet (declared polarity)
        tr = np.asarray(block[il_idx - 2:il_idx + 3, xl_idx - 2:xl_idx + 3, k0:k0 + nt], dtype=np.float64).mean(axis=(0, 1))
        lnz_fill = np.where(mask, lnz_win, np.nan)
        lnz_fill = np.where(np.isfinite(lnz_fill), lnz_fill, np.interp(np.arange(nt), np.flatnonzero(mask), lnz_win[mask]))
        syn = synthetic(lnz_fill, w_used, c)
        edge = int(round(args.edge_s / dt))
        tie = np.zeros(nt, bool)
        tie[rows[0] + edge:rows[1] - edge] = True
        r0 = float(np.corrcoef(syn[tie], tr[tie])[0, 1])
        data_rms = float(np.sqrt(np.mean(tr[tie] ** 2)))
        syn_rms = float(np.sqrt(np.mean(syn[tie] ** 2)))
        lag = v06["wells"][well]["best_lag_subsample_ms"]
        wpath = os.path.join(args.out, "wells", f"{tag}.npz")
        np.savez(wpath, t_axis=t_axis, lnz=lnz_win, log_rows=mask, tie_rows=tie, lnz_lowpass=lp,
                 data_rms_tie=data_rms, syn_rms_tie=syn_rms, gain_well=data_rms / syn_rms, r_zero_lag=r0,
                 tie_lag_subsample_ms=float(lag), well=well, source=os.path.relpath(src, V.RAW),
                 inline=int(blk["il"]), crossline=int(blk["xl"]), cdp_x=float(blk["x"]), cdp_y=float(blk["y"]))
        outputs[os.path.relpath(wpath, args.out)] = sha256(wpath)
        facts[well] = dict(block=blk["file"], inline=int(blk["il"]), crossline=int(blk["xl"]),
                           inline_line_traces=512, crossline_line_traces=384, well_trace_inline=xl_idx,
                           well_trace_crossline=il_idx, log_rows=[int(rows[0]), int(rows[1])],
                           log_window_s=[float(t_axis[rows[0]]), float(t_axis[rows[1] - 1] + dt)],
                           tie_rows=int(tie.sum()), r_zero_lag=r0, data_rms_tie=data_rms, syn_rms_tie=syn_rms,
                           gain_well=data_rms / syn_rms, tie_lag_subsample_ms=float(lag), log_source=os.path.relpath(src, V.RAW),
                           ai_range_kg_m2_s=[float(np.exp(np.nanmin(lnz_win))), float(np.exp(np.nanmax(lnz_win)))])
        print(f"{well:12s} il {blk['il']} xl {blk['xl']}  log rows {rows[0]}-{rows[1]} "
              f"({t_axis[rows[0]]:.3f}-{t_axis[rows[1] - 1] + dt:.3f} s, {tie.sum()} tie rows)  r0 {r0:+.3f}  "
              f"gain {data_rms / syn_rms:.4g}  lag {lag:+.1f} ms", flush=True)

        # quick-look figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        S_il = np.asarray(block[il_idx, :, k0:k0 + nt])
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[3, 1]))
        v = np.percentile(np.abs(S_il), 99)
        axes[0].imshow(S_il.T, aspect="auto", cmap="gray", vmin=-v, vmax=v,
                       extent=[0, S_il.shape[0], t_axis[-1] + dt, t_axis[0]])
        axes[0].axvline(xl_idx, color="r", lw=0.8)
        axes[0].set_title(f"{well}: inline {blk['il']} through the well (crossline {blk['xl']} at the red line)")
        axes[0].set_xlabel("trace (crossline index)")
        axes[0].set_ylabel("TWT (s)")
        axes[1].plot(np.exp(lnz_win) / 1e6, t_axis, "k", lw=1.0, label="log AI")
        axes[1].plot(np.exp(lp) / 1e6, t_axis, "r", lw=1.5, label="low-pass")
        axes[1].set_ylim(t_axis[-1] + dt, t_axis[0])
        axes[1].set_xlabel("AI (10$^6$ kg m$^{-2}$ s$^{-1}$)")
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "figures", f"{tag}_quicklook.png"), dpi=120)
        plt.close(fig)

    manifest = dict(item="Poseidon field data prepared for the TGRS revision (lines, logs, wavelets, calibration)",
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    raw_root=V.RAW, window_s=[args.t0, args.t1], dt_s=dt, nt=nt, half_pad=HALF, wavelet_taps=NW,
                    polarity="data compared with the negated unit-peak zero-phase wavelet (SEG reverse, declared)",
                    lowpass_sigma_samples=args.lowpass_sigma,
                    lowpass_cutoff_hz=float(np.sqrt(np.log(2.0)) / (np.sqrt(2.0) * np.pi * args.lowpass_sigma * dt)),
                    edge_exclusion_s=args.edge_s, wells=facts, inputs_sha256=inputs, outputs_sha256=outputs,
                    script_sha256=sha256(os.path.abspath(__file__)))
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    lines = ["# Poseidon field data: preparation record", "",
             f"Window {args.t0}-{args.t1} s at {dt * 1000:.0f} ms ({nt} samples), lines extended by {HALF} samples on each side "
             f"for the wavelet ({NW} taps, centre {HALF}); starting-model low-pass sigma {args.lowpass_sigma:g} samples "
             f"({manifest['lowpass_cutoff_hz']:.2f} Hz half amplitude).", "",
             "| Well | Inline | Crossline | Log window (s) | Tie rows | r at zero lag | Gain (data / synthetic) | Declared lag (ms) |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for w, f_ in facts.items():
        lines.append(f"| {w} | {f_['inline']} | {f_['crossline']} | {f_['log_window_s'][0]:.3f}-{f_['log_window_s'][1]:.3f} | "
                     f"{f_['tie_rows']} | {f_['r_zero_lag']:+.3f} | {f_['gain_well']:.4g} | {f_['tie_lag_subsample_ms']:+.1f} |")
    with open(os.path.join(args.out, "PREPARE_RECORD.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("written to", args.out)


if __name__ == "__main__":
    main()
