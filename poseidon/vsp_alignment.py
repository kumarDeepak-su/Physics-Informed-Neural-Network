#!/usr/bin/env python3
"""
VSP-derived alignment of each well (protocol v1, addendum 2): the time shift
between the surface near-stack trace at the well and the well's VSP corridor
stack, both seismic measurements, estimated without reading any impedance log.

For every tied well with a corridor stack on disk the corridor stack (1 ms,
two-way time from SRD = MSL) is correlated with the surface trace at the well
(5 x 5 bin mean of the inline block, interpolated to 1 ms) over the log window
of the well extended by 100 ms, for lags of -24 to +24 ms; the shift is the lag
of the maximum absolute correlation, refined by parabolic interpolation. The
sign of the correlation at the maximum is recorded (polarity of the corridor
stack relative to the surface data). Wells without a corridor stack keep a zero
shift. Output: prepared/vsp_alignment.json.

Convention: shift_ms > 0 means the surface event arrives later than the VSP
event, the same convention as the verification lags of v06.

usage: python vsp_alignment.py --prepared DIR
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WELLS_DIR = os.path.join(MANUSCRIPT, "DATA_POST_STACK", "Poseidon3D", "wells")
STACKS = {"Kronos 1": "Kronos_1/VSP/CP_Kronos1_corstk.sgy",
          "Pharos 1": "Pharos_1/VSP/COP_Pharos-1_corstk.sgy",
          "Poseidon 2": "Poseidon_2/VSP/CP_Poseidon2_corstk.sgy",
          "Torosa 1": "Torosa_1/VSP/VSPZO_COMPUTED_DCU_ZPH_90.SEGY"}
NOTES = {"Torosa 1": "not a corridor stack: deconvolved upgoing wavefield (120 levels, Read Well Services 2007); "
                     "the deepest level's trace is used"}
WELLS = ["Proteus 1", "Pharos 1", "Poseidon 2", "Kronos 1", "Torosa 1"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def corridor_trace(path):
    import segyio
    with segyio.open(path, "r", ignore_geometry=True) as s:
        dt = segyio.tools.dt(s) / 1e6
        if s.tracecount <= 10:
            tr = np.mean([np.asarray(s.trace[i], float) for i in range(s.tracecount)], axis=0)
        else:
            tr = np.asarray(s.trace[s.tracecount - 1], float)
        return tr, dt, s.tracecount


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--max-lag-ms", type=float, default=24.0)
    ap.add_argument("--margin-s", type=float, default=0.1)
    args = ap.parse_args()
    out = {"item": "VSP-derived alignment (surface near stack against the VSP corridor stack at the well, no log read)",
           "convention": "shift_ms > 0: surface event later than the VSP event (as the v06 lags)", "wells": {}, "inputs_sha256": {}}
    for well in WELLS:
        tag = well.replace(" ", "_")
        L = np.load(os.path.join(args.prepared, "lines", f"{tag}_inline.npz"))
        d = np.load(os.path.join(args.prepared, "wells", f"{tag}.npz"))
        nt, half, dt, t0 = int(L["nt"]), int(L["half_pad"]), float(L["dt"]), float(L["t0"])
        wt = int(L["well_trace"])
        S = L["S_ext"][:, half:half + nt].astype(np.float64)
        surf = S[max(0, wt - 2):wt + 3].mean(axis=0)
        t4 = t0 + dt * np.arange(nt)
        rows = np.flatnonzero(d["log_rows"])
        w0, w1 = t4[rows[0]] - args.margin_s, t4[rows[-1]] + args.margin_s
        entry = dict(window_s=[float(w0), float(w1)], log_based_lag_ms=float(d["tie_lag_subsample_ms"]))
        if well not in STACKS:
            entry.update(shift_ms=0.0, source=None, note="no corridor stack on disk; zero shift")
            out["wells"][well] = entry
            print(f"{well:12s} no corridor stack -> shift 0.0 ms (log-based {entry['log_based_lag_ms']:+.1f})")
            continue
        path = os.path.join(WELLS_DIR, STACKS[well])
        tr, dtv, ntr = corridor_trace(path)
        out["inputs_sha256"][STACKS[well]] = sha256(path)
        tv = dtv * np.arange(tr.size)
        t1 = np.arange(w0, w1, dtv)                                   # 1 ms grid over the window
        surf1 = np.interp(t1, t4, surf)
        lags = np.arange(-args.max_lag_ms, args.max_lag_ms + 0.5, dtv * 1000.0)
        rs = []
        for lag in lags:
            v = np.interp(t1 - lag / 1000.0, tv, tr)                 # VSP delayed by lag: surface later by lag
            rs.append(np.corrcoef(v, surf1)[0, 1])
        rs = np.asarray(rs)
        i = int(np.argmax(np.abs(rs)))
        shift = float(lags[i])
        if 0 < i < lags.size - 1:
            y0, y1, y2 = np.abs(rs[i - 1:i + 2])
            den = y0 - 2 * y1 + y2
            if den < 0:
                shift = float(lags[i] + 0.5 * (y0 - y2) / den * (lags[1] - lags[0]))
        entry.update(shift_ms=shift, correlation=float(rs[i]), polarity_sign=int(np.sign(rs[i])), source=STACKS[well],
                     stack_traces=int(ntr), stack_dt_ms=dtv * 1000.0, lag_grid_ms=[float(lags[0]), float(lags[-1])],
                     note=NOTES.get(well, "corridor stack, mean of the delivered traces"))
        out["wells"][well] = entry
        print(f"{well:12s} VSP shift {shift:+5.1f} ms (r {rs[i]:+.3f}, {ntr} traces)   log-based {entry['log_based_lag_ms']:+.1f} ms")
    out["script_sha256"] = sha256(os.path.abspath(__file__))
    with open(os.path.join(args.prepared, "vsp_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written", os.path.join(args.prepared, "vsp_alignment.json"))


if __name__ == "__main__":
    main()
