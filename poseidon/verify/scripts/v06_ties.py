import sys; sys.dont_write_bytecode = True
"""v06 - tie per well under OUR forward convention with the v05 zero-phase statistical wavelet, the declared
polarity (SEG reverse -> data compared with -synthetic), the delivered time-depth (v04), a declared small lag
search, and two nulls. Item (6). ALIGNMENT CHECK, not a score: no inversion, no impedance prediction, no well
scored against another well's product. The same procedure is applied to every well so that no well is
'blind' afterwards; the frozen protocol must therefore treat every tie statistic here as disclosed.
    ~/.venvs/deepfwi/bin/python v06_ties.py [--lag-ms 12 --n-null 500]
"""
import argparse
import json
import os

import numpy as np

import vlib as V

ap = argparse.ArgumentParser(); ap.add_argument("--lag-ms", type=float, default=12.0); ap.add_argument("--n-null", type=int, default=500)
ap.add_argument("--half", type=int, default=2); a = ap.parse_args()
rec = json.load(open(os.path.join(V.SEISMIC, "near_stack_blocks_record.json")))
v04 = json.load(open(os.path.join(V.EVID, "v04_time_depth.json"))); v05 = json.load(open(os.path.join(V.EVID, "v05_seismic_band.json")))
dt = rec["time_ms"][2] / 1000.0; t_axis = np.arange(rec["time_ms"][0], rec["time_ms"][1] + 1e-9, rec["time_ms"][2]) / 1000.0
rng = np.random.default_rng(20260916)


def impedance_in_time(well):
    """AI(t) on the seismic time axis from the delivered time-domain logs (VSP-processed sets) or the
    checkshot-calibrated LAS; returns (t, AI) restricted to the joint density+sonic window."""
    d = os.path.join(V.WELLS_DIR, V.well_tag(well)); w4 = v04["wells"][well]
    if w4.get("time_domain_logs"):
        c, _, _ = V.las_curves(os.path.join(V.RAW, w4["time_domain_logs"]))
        t = c["TIME"]; dtc = c.get("DTCO", c.get("DT")); rho = c["RHOB"]
        ok = np.isfinite(t) & np.isfinite(dtc) & np.isfinite(rho) & (rho > 1.5)
        vel = 1e6 / dtc[ok] * 0.3048; return t[ok], rho[ok] * 1000.0 * vel, os.path.relpath(os.path.join(V.RAW, w4["time_domain_logs"]), V.RAW)
    if well == "Poseidon 1":
        import csv
        rows = list(csv.DictReader(open(os.path.join(d, "reports", "Poseidon1_VSP_Table33_time_depth.csv"))))
        md_df = np.array([float(r["depth_col2_m"]) for r in rows]); tvd = np.array([float(r["depth_col1_m"]) for r in rows]); twt = 2 * np.array([float(r["vertical_transit_time_s"]) for r in rows])
        o = np.argsort(md_df); md_df, tvd, twt = md_df[o], tvd[o], twt[o]
        cd, _, _ = V.las_curves(os.path.join(d, "wireline", "Poseidon-1_Suite2_Run1_CMR-LDS-CNL-GR_ML.las"))
        cs, _, _ = V.las_curves(os.path.join(d, "wireline", "ConocoPhillips_Poseidon-1_Sonic_Scanner_P&S_S2R3.las"))
        md_s = cs["DEPT"]; dtc = cs["DTCO"]; ok_s = np.isfinite(dtc); md_s, dtc = md_s[ok_s], dtc[ok_s]
        md_d = cd["DEPT"]; rho = cd["HROM"]; ok_d = np.isfinite(rho) & (rho > 1.5); md_d, rho = md_d[ok_d], rho[ok_d]
        o = np.argsort(md_s); md_s, dtc = md_s[o], dtc[o]; o = np.argsort(md_d); md_d, rho = md_d[o], rho[o]
        lo, hi = max(md_s.min(), md_d.min()), min(md_s.max(), md_d.max(), md_df.max())
        md = np.arange(lo, hi, 0.1524); dtc_i = np.interp(md, md_s, dtc); rho_i = np.interp(md, md_d, rho)
        t = np.interp(md, md_df, twt)                       # MD (from DF) -> TWT through the VSP levels (col2 -> vertical time)
        vel = 1e6 / dtc_i * 0.3048
        return t, rho_i * 1000.0 * vel, os.path.relpath(os.path.join(d, "reports", "Poseidon1_VSP_Table33_time_depth.csv"), V.RAW)
    if well == "Torosa 1":
        p = os.path.join(d, "VSP", "TZV_DEPTH_CS_CALIBRATED_LOGS.LAS"); c, _, _ = V.las_curves(p)
        md = c["MD"]; t = c["TIME"] / 1000.0; dtc = c["DTC_CS"]; rho = c["RHO_CS"]
        ok = np.isfinite(t) & np.isfinite(dtc) & np.isfinite(rho) & (md >= 3577.0) & (md <= 4666.0)     # measured density only
        vel = 1e6 / dtc[ok] * 0.3048; return t[ok], rho[ok] * 1000.0 * vel, os.path.relpath(p, V.RAW)
    return None, None, None


def synthetic(ai_t, w, c):
    r = (ai_t[1:] - ai_t[:-1]) / (ai_t[1:] + ai_t[:-1]); r = np.concatenate([r, [0.0]])
    return np.convolve(r, w)[c:c + ai_t.size]


results = {}; inputs = []
for b in sorted(rec["blocks"], key=lambda x: x["well"]):
    well = b["well"]
    if not os.path.exists(os.path.join(V.SEISMIC, b["file"])) or well not in v05["blocks"]:
        continue
    t_log, ai, src = impedance_in_time(well)
    if t_log is None:
        results[well] = {"status": "no delivered impedance in time", "seismic_block": b["file"]}; print(f"  {well:18s} no impedance in time"); continue
    inputs.append(os.path.join(V.RAW, src))
    # resample the log impedance onto the seismic axis (only inside its own window), fill outside with edge values
    k0 = int(np.ceil(t_log.min() / dt)); k1 = int(np.floor(t_log.max() / dt)); tt = t_axis[k0:k1 + 1]
    ai_s = np.interp(tt, t_log, ai)
    w = np.array(v05["blocks"][well]["wavelet_zero_phase_unit_peak"]); c = v05["blocks"][well]["wavelet_centre_index"]
    syn = synthetic(ai_s, w, c)
    data, hdr, blk, _ = V.seismic_block(well); i = b["il"] - b["il_range"][0]; j = b["xl"] - b["xl_range"][0]
    trace = np.asarray(data[i - a.half:i + a.half + 1, j - a.half:j + a.half + 1, :], dtype=np.float64).mean(axis=(0, 1))   # 5x5 bin mean
    pol = -1.0                                                                                                              # SEG reverse
    edge = int(round(0.064 / dt)); rows = slice(edge, syn.size - edge)                                                         # drop wavelet edges
    L = int(round(a.lag_ms / 1000.0 / dt)); best = None; curve = []
    for lag in range(-L, L + 1):
        seg = trace[k0 + lag:k1 + 1 + lag]
        r = V.pearson(pol * syn[rows], seg[rows]); curve.append([lag * dt * 1000.0, float(r)])
        if best is None or r > best[1]: best = (lag, r)
    r0 = V.pearson(pol * syn[rows], trace[k0:k1 + 1][rows])
    cl = np.array([c_[1] for c_ in curve]); ib = int(np.argmax(cl)); sub_lag = best[0] * dt * 1000.0
    if 0 < ib < cl.size - 1:
        y0, y1, y2 = cl[ib - 1], cl[ib], cl[ib + 1]; den = (y0 - 2 * y1 + y2)
        if den < 0: sub_lag = (best[0] + 0.5 * (y0 - y2) / den) * dt * 1000.0
    # null 1: circular shifts of the data trace far from the window (span of +-1.5 s outside +-0.1 s)
    n1 = []
    for _ in range(a.n_null):
        s = int(rng.integers(int(0.1 / dt), int(1.5 / dt))) * int(rng.choice([-1, 1]))
        segn = np.roll(trace, s)[k0:k1 + 1]
        n1.append(max(V.pearson(pol * syn[rows], np.roll(segn, lag)[rows]) for lag in range(-L, L + 1, 2)))
    # null 2: phase-randomised synthetic (same spectrum), same lag search
    n2 = []
    for _ in range(a.n_null):
        sp = V.phase_randomised(syn, rng)
        n2.append(max(V.pearson(pol * sp[rows], trace[k0 + lag:k1 + 1 + lag][rows]) for lag in range(-L, L + 1, 2)))
    n1, n2 = np.array(n1), np.array(n2)
    res = dict(seismic_block=b["file"], impedance_source=src, window_s=[float(tt[0]), float(tt[-1])], rows=int(syn[rows].size),
               r_at_zero_lag=float(r0), best_lag_ms=best[0] * dt * 1000.0, best_lag_subsample_ms=float(sub_lag), r_best=float(best[1]), lag_curve=curve,
               null_circular=dict(p=float((n1 >= best[1]).mean()), q95=float(np.quantile(n1, 0.95)), mean=float(n1.mean())),
               null_phase_random=dict(p=float((n2 >= best[1]).mean()), q95=float(np.quantile(n2, 0.95)), mean=float(n2.mean())),
               polarity_applied=-1, note="5x5-bin mean trace; zero-phase statistical wavelet from v05; no phase fit")
    results[well] = res
    print(f"  {well:18s} rows {res['rows']:4d}  r0 {r0:+.3f}  best {best[1]:+.3f} @ {res['best_lag_ms']:+5.1f} ms (sub {res['best_lag_subsample_ms']:+5.1f})  null95 {res['null_circular']['q95']:.3f}/{res['null_phase_random']['q95']:.3f}  p {res['null_circular']['p']:.3f}/{res['null_phase_random']['p']:.3f}")
payload = dict(item="(6) tie per well: declared time-depth, v05 zero-phase wavelet, declared polarity, +-lag search, two nulls", wells=results,
               config=vars(a), disclosure="all wells are tied here; the frozen protocol must treat every tie statistic as disclosed")
o = V.write_evidence("v06_ties.json", payload, V.run_record(os.path.abspath(__file__), sorted(set(inputs)), config=vars(a)))
print("->", o)
