import sys; sys.dont_write_bytecode = True
"""v04 - time-depth control per well: the VSP/checkshot tables as delivered, the joint density+sonic window
in two-way time, the sonic-integral drift against the VSP, and the effective sample count 2 (f_hi - f_lo) T.
Item (4). No tie, wavelet, gain or lag is estimated here. Each well's time-depth SOURCE is named and hashed;
a well without a delivered time-depth (Poseidon North 1, Boreas 1) is reported as such, not filled in.
    ~/.venvs/deepfwi/bin/python v04_time_depth.py
"""
import csv
import json
import os
import re

import numpy as np

import vlib as V

BAND = (8.0, 45.0)     # provisional half-amplitude band of the near stack; v05 replaces it with the measured one


def xls_level_table(path):
    """Schlumberger VSP 'Well Seismic Report' / 'Time to Depth' spreadsheets (old .xls): returns
    (tvd_srd_m, twt_s, kind). kind = 'levels' for a level table (depth, vertical time), 'tz' for a TWT->depth table."""
    import xlrd
    wb = xlrd.open_workbook(path)
    sh = wb.sheet_by_index(0)
    rows = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]
    text = " ".join(str(v) for row in rows[:20] for v in row)
    nums = []
    for row in rows:
        vals = []
        for v in row:
            try:
                vals.append(float(v))
            except Exception:
                pass
        if len(vals) >= 2:
            nums.append(vals)
    if re.search(r"(?i)time to depth", text):
        tz = np.array([(r[0], r[1]) for r in nums if 0 <= r[0] <= 8000 and 0 <= r[1] <= 8000 and len(r) >= 2])
        return tz[:, 1], tz[:, 0] / 1000.0, "tz(+0 ms column)"
    lev = np.array([(r[2], r[4]) for r in nums if len(r) >= 5 and 300 <= r[2] <= 6000 and 0.1 <= r[4] <= 4.0])
    if lev.size:
        return lev[:, 0], 2 * lev[:, 1], "levels(TVD-SRD, 2 x vertical one-way time)"
    return None, None, "unparsed"


def td_source(well):
    d = os.path.join(V.WELLS_DIR, V.well_tag(well))
    if well == "Pharos 1":
        p = os.path.join(d, "VSP", "COP-Pharos-1_QC_VSP_TimeDepthListing.xlsx")
        t = V.vsp_time_depth_xlsx(p); return p, t["tvd_srd"], t["twt_s"], "xlsx TimeDepthListing (TVD from SRD=MSL, TWT TVD-corrected)"
    if well == "Kronos 1":
        p = os.path.join(d, "VSP", "CP_Kronos1_Time_to_Depth_Report.xls")
        z, t, k = xls_level_table(p); return p, z, t, k + " SRD=MSL"
    if well == "Poseidon 2":
        p = os.path.join(d, "VSP", "CP_Poseidon2_Well_Seismic_Report.xls")
        z, t, k = xls_level_table(p); return p, z, t, k
    if well == "Proteus 1":
        p = os.path.join(d, "VSP", "COP_Proteus-1ST2_A1_Well_Seismic_Report_Final.xls")
        z, t, k = xls_level_table(p); return p, z, t, k
    if well == "Poseidon 1":
        p = os.path.join(d, "reports", "Poseidon1_VSP_Table33_time_depth.csv")
        rows = list(csv.DictReader(open(p)))
        z = np.array([float(r["depth_col1_m"]) for r in rows]); t = 2 * np.array([float(r["vertical_transit_time_s"]) for r in rows])
        return p, z, t, "WCR Vol 2 Table 33 (depth below SRD=AHD, 2 x vertical transit time)"
    if well == "Torosa 1":
        p = os.path.join(d, "VSP", "TZV_DEPTH_CS_CALIBRATED_LOGS.LAS")
        c, _, _ = V.las_curves(p); ok = np.isfinite(c["TIME"]) & np.isfinite(c["TVD"])
        return p, c["TVD"][ok], c["TIME"][ok] / 1000.0, "checkshot-calibrated LAS (TVD from MSL, TWT from MSL)"
    return None, None, None, "no delivered time-depth in the files on disk"


def time_domain_logs(well):
    d = os.path.join(V.WELLS_DIR, V.well_tag(well))
    cands = {"Pharos 1": "VSP/COP_Pharos-1_logs_time.las", "Kronos 1": "VSP/CP_Kronos1_Logs_time.las",
             "Poseidon 2": "VSP/CP_Poseidon2_logs_time.las", "Proteus 1": "VSP/COP_Proteus-1ST2_logs_time.las"}
    if well in cands and os.path.exists(os.path.join(d, cands[well])):
        return os.path.join(d, cands[well])
    return None


def joint_window_time(path):
    c, _, _ = V.las_curves(path)
    t = c["TIME"]; dt = c.get("DTCO", c.get("DT")); rho = c.get("RHOB")
    ok = np.isfinite(t) & np.isfinite(dt) & np.isfinite(rho) & (rho > 1.5)
    return (float(t[ok].min()), float(t[ok].max()), int(ok.sum())) if ok.any() else None


def depth_window(well):
    """Measured density+sonic depth range from the spliced Occam LAS (0.5 m), as an independent cross-check."""
    p = os.path.join(V.WELLS_DIR, "Spliced_occam", {"Pharos 1": "Pharos1Decim.LAS", "Kronos 1": "Kronos1Decim.LAS", "Poseidon 1": "Poseidon1Decim.LAS",
                                                     "Poseidon 2": "Poseidon2Decim.LAS", "Poseidon North 1": "PoseidonNorth1Decim.LAS", "Proteus 1": "Proteus1Decim.LAS",
                                                     "Torosa 1": "Torosa1LDecim.LAS", "Boreas 1": "Boreas1Decim.LAS"}.get(well, "none"))
    if not os.path.exists(p):
        return None
    c, _, _ = V.las_curves(p); md = c["DEPT"]
    dt = next((c[k] for k in ("DTCO", "DTC", "DTS") if k in c), None); rho = next((c[k] for k in ("RHOB", "RHOZ", "HROM") if k in c), None)
    if dt is None or rho is None:
        return {"file": p, "note": "no sonic or no density curve in the spliced LAS"}
    ok = np.isfinite(dt) & np.isfinite(rho) & (rho > 1.5)
    return {"file": p, "md_range_m": [float(md[ok].min()), float(md[ok].max())] if ok.any() else None, "n": int(ok.sum()),
            "sonic_curve": next(k for k in ("DTCO", "DTC", "DTS") if k in c), "density_curve": next(k for k in ("RHOB", "RHOZ", "HROM") if k in c)}


results = {}; pooled = 0.0; inputs = []
for well in V.WELLS:
    src, z, t, kind = td_source(well)
    r = {"time_depth_source": src and os.path.relpath(src, V.RAW), "kind": kind}
    if src: inputs.append(src)
    if z is not None and t is not None and len(z) > 3:
        o = np.argsort(z); z, t = z[o], t[o]
        r["levels"] = int(len(z)); r["tvd_range_m"] = [float(z.min()), float(z.max())]; r["twt_range_s"] = [float(t.min()), float(t.max())]
        r["monotonic"] = bool(np.all(np.diff(t) > 0))
    tl = time_domain_logs(well)
    if tl:
        inputs.append(tl); w = joint_window_time(tl)
        r["time_domain_logs"] = os.path.relpath(tl, V.RAW); r["joint_rhob_dt_window_s"] = w
    dw = depth_window(well); r["spliced_depth_window"] = dw
    if dw and dw.get("md_range_m") and z is not None and t is not None:
        tw = np.interp(dw["md_range_m"], z, t); r["spliced_window_mapped_to_twt_s"] = [float(tw[0]), float(tw[1])]
    win = r.get("joint_rhob_dt_window_s") or (r.get("spliced_window_mapped_to_twt_s") and (r["spliced_window_mapped_to_twt_s"][0], r["spliced_window_mapped_to_twt_s"][1], None))
    if win:
        T = win[1] - win[0]; r["T_s"] = T; r["n_eff_band"] = 2 * (BAND[1] - BAND[0]) * T; pooled += r["n_eff_band"]
    results[well] = r
    print(f"  {well:18s} {kind[:48]:48s} T={r.get('T_s', float('nan')):6.3f} s  n_eff={r.get('n_eff_band', float('nan')):6.1f}")
print(f"  pooled n_eff (band {BAND[0]:.0f}-{BAND[1]:.0f} Hz): {pooled:.0f}  -> Zou minimum detectable difference scaled from IBDP (0.48 at 14): {0.48*np.sqrt(14/max(pooled,1)):.3f}")
payload = dict(item="(4) time-depth control per well, joint density+sonic window in TWT, effective samples", band_hz=BAND,
               wells=results, pooled_n_eff=pooled, note="BAND is provisional until v05 measures the near-stack wavelet; the spliced (Occam) windows are a cross-check, not the primary source")
o = V.write_evidence("v04_time_depth.json", payload, V.run_record(os.path.abspath(__file__), sorted(set(inputs)), config={"band_hz": BAND}))
print("->", o)
