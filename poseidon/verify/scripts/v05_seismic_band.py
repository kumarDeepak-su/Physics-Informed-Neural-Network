import sys; sys.dont_write_bytecode = True
"""v05 - the near-stack data band around every extracted well block: amplitude spectrum, half-amplitude band
(f_lo, f_hi), peak frequency, the zero-phase statistical wavelet from the autocorrelation, and the F3/IBDP
timing gate S_gate = c_gate / f_hi. Item (5). No well log is opened: this is seismic-only and identical for
calibration and evaluation wells, so it cannot leak. Phase is NOT estimated here (v06 does that at calibration
wells only).
    ~/.venvs/deepfwi/bin/python v05_seismic_band.py [--t0 1.5 --t1 4.0 --half 12]
"""
import argparse
import glob
import json
import os

import numpy as np

import vlib as V

ap = argparse.ArgumentParser(); ap.add_argument("--t0", type=float, default=1.5); ap.add_argument("--t1", type=float, default=4.0)
ap.add_argument("--half", type=int, default=12); a = ap.parse_args()
rec = json.load(open(os.path.join(V.SEISMIC, "near_stack_blocks_record.json")))
dt = rec["time_ms"][2] / 1000.0; t_axis = np.arange(rec["time_ms"][0], rec["time_ms"][1] + 1e-9, rec["time_ms"][2]) / 1000.0
k0, k1 = int(round(a.t0 / dt)), int(round(a.t1 / dt))
out = {}; inputs = []
for b in sorted(rec["blocks"], key=lambda x: x["well"]):
    p = os.path.join(V.SEISMIC, b["file"])
    if not os.path.exists(p):
        continue
    inputs.append(p)
    data, hdr, blk, _ = V.seismic_block(b["well"])
    i = b["il"] - b["il_range"][0]; j = b["xl"] - b["xl_range"][0]
    sub = np.asarray(data[max(0, i - a.half):i + a.half + 1, max(0, j - a.half):j + a.half + 1, k0:k1], dtype=np.float64)
    tr = sub.reshape(-1, sub.shape[-1]); tr = tr - tr.mean(axis=1, keepdims=True)
    win = np.hanning(tr.shape[1]); n = 4096
    spec = np.sqrt(np.mean(np.abs(np.fft.rfft(tr * win, n=n)) ** 2, axis=0)); f = np.fft.rfftfreq(n, dt)
    spec_n = spec / spec.max(); pk = float(f[np.argmax(spec_n)])
    above = np.where(spec_n >= 0.5)[0]; f_lo, f_hi = float(f[above[0]]), float(f[above[-1]])
    # zero-phase statistical wavelet: inverse FFT of the mean amplitude spectrum, 128 ms half-length, tapered
    w = np.fft.irfft(spec, n=n); w = np.concatenate([w[-64:], w[:65]]); w = w * np.hanning(w.size); w = w / np.abs(w).max()
    rms = float(np.sqrt(np.mean(tr ** 2)))
    gate = {m: (c / f_hi * 1000.0 if c > 0 else 0.0) for m, c in V.GATE_C.items()}
    out[b["well"]] = dict(file=b["file"], sha256=b["sha256"], traces=int(tr.shape[0]), window_s=[a.t0, a.t1], rms=rms,
                          peak_hz=pk, f_lo_half_amp_hz=f_lo, f_hi_half_amp_hz=f_hi, bandwidth_hz=f_hi - f_lo,
                          s_gate_ms=gate, wavelet_zero_phase_unit_peak=w.tolist(), wavelet_dt_s=dt, wavelet_centre_index=64,
                          spectrum_hz=f[:600].tolist(), spectrum_norm=spec_n[:600].tolist())
    print(f"  {b['well']:18s} traces {tr.shape[0]:4d}  peak {pk:5.1f} Hz  band {f_lo:5.1f}-{f_hi:5.1f} Hz  gate(rel-band) {gate['corr_relative_band']:.2f} ms  rms {rms:.1f}")
bands = np.array([[v["f_lo_half_amp_hz"], v["f_hi_half_amp_hz"], v["peak_hz"]] for v in out.values()])
summary = dict(n_blocks=len(out), f_lo_median=float(np.median(bands[:, 0])), f_hi_median=float(np.median(bands[:, 1])), peak_median=float(np.median(bands[:, 2])),
               n_eff_per_second=float(2 * (np.median(bands[:, 1]) - np.median(bands[:, 0]))),
               s_gate_rel_band_ms_median=float(np.median([v["s_gate_ms"]["corr_relative_band"] for v in out.values()])),
               polarity=V.POLARITY)
print("  summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != "polarity"})
payload = dict(item="(5) near-stack band, statistical zero-phase wavelet and timing gate per well block (seismic only)", blocks=out, summary=summary)
o = V.write_evidence("v05_seismic_band.json", payload, V.run_record(os.path.abspath(__file__), inputs, config=vars(a)))
print("->", o)
