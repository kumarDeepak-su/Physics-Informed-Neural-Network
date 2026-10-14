import sys; sys.dont_write_bytecode = True
"""v07 - metric rows per well at the measured near-stack band (v05), washout/edit flags where a caliper is
delivered in time, the three effective-sample estimates (2 (f_hi - f_lo) T; Pyper-Peterman on the band-passed
log impedance against a band-passed copy; autocorrelation n_eff), and the Zou minimum detectable difference at
the pooled n_eff for a leave-one-well-out design. Item (7). Nothing here scores an inversion.
    ~/.venvs/deepfwi/bin/python v07_rows_and_power.py
"""
import json
import math
import os

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import norm

import vlib as V

v04 = json.load(open(os.path.join(V.EVID, "v04_time_depth.json"))); v05 = json.load(open(os.path.join(V.EVID, "v05_seismic_band.json")))
band = (v05["summary"]["f_lo_median"], v05["summary"]["f_hi_median"]); dt = 0.004


def fisher_ci(r, n, level=0.95):
    z = math.atanh(max(min(r, 0.999999), -0.999999)); h = norm.ppf(0.5 + level / 2) / math.sqrt(n - 3)
    return math.tanh(z - h), math.tanh(z + h)


def zou_min_detectable(r_b, r_ab, n, level):
    def excludes(r_a):
        l1, u1 = fisher_ci(r_a, n, level); l2, u2 = fisher_ci(r_b, n, level)
        den = (1 - r_a ** 2) * (1 - r_b ** 2)
        c = ((r_ab - 0.5 * r_a * r_b) * (1 - r_a ** 2 - r_b ** 2 - r_ab ** 2) + r_ab ** 3) / den if den > 0 else 1.0
        lo = r_a - r_b - math.sqrt(max((r_a - l1) ** 2 + (u2 - r_b) ** 2 - 2 * c * (r_a - l1) * (u2 - r_b), 0.0))
        return lo > 0
    lo, hi = r_b, 0.99999
    if not excludes(hi):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excludes(mid): hi = mid
        else: lo = mid
    return float(hi - r_b)


def acf(x, J):
    x = x - x.mean(); den = float(np.sum(x * x)); return np.array([1.0] + [float(np.sum(x[:-k] * x[k:])) / den for k in range(1, J + 1)])


def n_eff_pp(x, y):
    n = x.size; J = max(1, n // 5); rx, ry = acf(x, J), acf(y, J); j = np.arange(1, J + 1)
    inv = 1.0 / n + (2.0 / n) * np.sum((n - j) / n * rx[1:] * ry[1:]); return float(min(max(1.0 / inv, 3.0), n))


def bandpass(x, lo, hi):
    b, a = butter(4, [lo * 2 * dt, min(hi * 2 * dt, 0.99)], btype="band"); return filtfilt(b, a, x)


rows = {}; pooled = {"rows": 0, "n_2bt": 0.0, "n_pp": 0.0}
for well, w4 in v04["wells"].items():
    src = w4.get("time_domain_logs")
    if not src and well == "Torosa 1":
        src = "wells/Torosa_1/VSP/TZV_DEPTH_CS_CALIBRATED_LOGS.LAS"
    if not src:
        rows[well] = {"status": "no time-domain impedance on disk"}; continue
    c, _, _ = V.las_curves(os.path.join(V.RAW, src))
    if "TIME" in c and np.nanmax(c["TIME"]) > 100:                       # Torosa: ms and depth-indexed
        t = c["TIME"] / 1000.0; dtc = c["DTC_CS"]; rho = c["RHO_CS"]; md = c["MD"]; ok = np.isfinite(t) & np.isfinite(dtc) & np.isfinite(rho) & (md >= 3577) & (md <= 4666)
    else:
        t = c["TIME"]; dtc = c.get("DTCO", c.get("DT")); rho = c["RHOB"]; ok = np.isfinite(t) & np.isfinite(dtc) & np.isfinite(rho) & (rho > 1.5)
    t, ai = t[ok], rho[ok] * 1000.0 * (1e6 / dtc[ok] * 0.3048)
    k0, k1 = int(np.ceil(t.min() / dt)), int(np.floor(t.max() / dt)); tt = np.arange(k0, k1 + 1) * dt
    ai_s = np.interp(tt, t, ai); lnz = np.log(ai_s)
    n_rows = int(tt.size); T = float(tt[-1] - tt[0]); n_2bt = 2 * (band[1] - band[0]) * T
    bp = bandpass(lnz - lnz.mean(), band[0], band[1]) if n_rows > 40 else lnz - lnz.mean()
    n_pp = n_eff_pp(bp, bp) if n_rows > 40 else float(n_rows)
    cal = c.get("CALI", c.get("CALI:1")); wash = None
    if cal is not None:
        cc = np.interp(tt, t, cal[ok]) if np.isfinite(cal[ok]).any() else None
        if cc is not None:
            med = np.nanmedian(cc); wash = {"caliper_median_in": float(med), "rows_over_median_plus_1in": int(np.sum(cc > med + 1.0)), "fraction": float(np.mean(cc > med + 1.0))}
    rows[well] = {"source": src, "window_s": [float(tt[0]), float(tt[-1])], "T_s": T, "rows_4ms": n_rows, "n_eff_2BT": n_2bt, "n_eff_pyper_peterman_bandpassed": n_pp, "washout": wash}
    pooled["rows"] += n_rows; pooled["n_2bt"] += n_2bt; pooled["n_pp"] += n_pp
    print(f"  {well:18s} rows {n_rows:4d}  T {T:5.2f} s  n_2BT {n_2bt:6.1f}  n_PP {n_pp:6.1f}  washout {wash['fraction'] if wash else 'n/a'}")
level_family = 1 - 0.05 / 40
mdd = {f"n_eff={int(n)}": {f"r_b={rb},r_ab={rab}": zou_min_detectable(rb, rab, n, level_family) for rb in (0.3, 0.5) for rab in (0.5, 0.8, 0.9)}
       for n in (14, pooled["n_2bt"], pooled["n_pp"])}
print(f"  pooled: rows {pooled['rows']}  n_2BT {pooled['n_2bt']:.0f}  n_PP {pooled['n_pp']:.0f}  band {band[0]:.1f}-{band[1]:.1f} Hz")
for k, v in mdd.items():
    print(f"  Zou MDD at {k} (family level {level_family:.4f}): " + ", ".join(f"{kk} -> {vv:.3f}" for kk, vv in v.items()))
payload = dict(item="(7) rows, washouts, three effective-sample estimates, Zou minimum detectable difference at the pooled n_eff", band_hz=band,
               wells=rows, pooled=pooled, zou_min_detectable=mdd, family_level=level_family,
               design_note="leave-one-well-out: each evaluation well is scored on its own rows; the pooled n_eff is what a paired comparison across all evaluation wells sees")
o = V.write_evidence("v07_rows_and_power.json", payload, V.run_record(os.path.abspath(__file__), [os.path.join(V.EVID, "v04_time_depth.json"), os.path.join(V.EVID, "v05_seismic_band.json")], config={"band_hz": band}))
print("->", o)
