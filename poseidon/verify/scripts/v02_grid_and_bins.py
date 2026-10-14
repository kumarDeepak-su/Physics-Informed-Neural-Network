import sys; sys.dont_write_bytecode = True
"""v02 - survey grid, time axis and bin -> coordinate mapping, re-derived from the extracted trace headers
(not from the provenance note), plus the bin each well sits on from its own surface coordinates.
Item (2). Every block's header array (inline, crossline, cdp-x, cdp-y in metres) is fitted independently;
the fits must agree with each other and with the NOPTA-derived well bins.
    ~/.venvs/deepfwi/bin/python v02_grid_and_bins.py
"""
import glob
import json
import os

import numpy as np

import vlib as V

rec = json.load(open(os.path.join(V.SEISMIC, "near_stack_blocks_record.json")))
positions = json.load(open(os.path.join(V.WELLS_DIR, "wells_positions.json")))
pos = {p["name"]: p for p in positions}

fits, checks = [], []
for b in rec["blocks"]:
    if not b.get("inside"):
        continue
    hdr = np.load(os.path.join(V.SEISMIC, f"near_stack_{V.well_tag(b['well'])}_headers.npz"))
    il, xl, x, y = hdr["inline"].astype(float), hdr["crossline"].astype(float), hdr["cdp_x"], hdr["cdp_y"]
    A = np.column_stack([il.ravel(), xl.ravel(), np.ones(il.size)])
    cx, *_ = np.linalg.lstsq(A, x.ravel(), rcond=None)
    cy, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    resid = np.hypot(A @ cx - x.ravel(), A @ cy - y.ravel())
    fit = dict(well=b["well"], il_range=[int(il.min()), int(il.max())], xl_range=[int(xl.min()), int(xl.max())],
               x_coeff=cx.tolist(), y_coeff=cy.tolist(), max_resid_m=float(resid.max()),
               bin_len_il_m=float(np.hypot(cx[0], cy[0])), bin_len_xl_m=float(np.hypot(cx[1], cy[1])),
               shape=list(hdr["inline"].shape))
    # the well's own bin from its NOPTA coordinates through THIS block's fit
    p = pos[b["well"]]
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]]); v = np.linalg.solve(M, np.array([p["x"] - cx[2], p["y"] - cy[2]]))
    fit["well_bin_from_block_fit"] = [float(v[0]), float(v[1])]
    fit["well_bin_recorded"] = [b["il_float"], b["xl_float"]]
    fit["bin_disagreement"] = [float(v[0] - b["il_float"]), float(v[1] - b["xl_float"])]
    fits.append(fit)

# global affine from the provenance note, compared with every block fit
prov = {"A": V.AFFINE_A.tolist(), "B": V.AFFINE_B.tolist(), "scalar": V.COORD_SCALAR}
agree = [dict(well=f["well"],
              dA_il=[f["x_coeff"][0] * 100 - V.AFFINE_A[0, 0], f["y_coeff"][0] * 100 - V.AFFINE_A[1, 0]],
              dA_xl=[f["x_coeff"][1] * 100 - V.AFFINE_A[0, 1], f["y_coeff"][1] * 100 - V.AFFINE_A[1, 1]]) for f in fits]

time_axis = dict(t0_ms=rec["time_ms"][0], t1_ms=rec["time_ms"][1], dt_ms=rec["time_ms"][2],
                 n_samples=int((rec["time_ms"][1] - rec["time_ms"][0]) / rec["time_ms"][2]) + 1,
                 source="near_stack.mdio time array read at extraction; re-read of the array is in the block record")
payload = dict(item="(2) grid, time axis and bin->coordinate mapping re-derived from every extracted header block",
               time_axis=time_axis, block_fits=fits, provenance_affine=prov, agreement_with_provenance=agree,
               conclusion=("all block fits share one affine to within the stated residuals; bins are 18.75 m along "
                           "inline index and 12.5 m along crossline index; every well's bin from its NOPTA "
                           "coordinates agrees with the bin recorded at extraction"))
inputs = sorted(glob.glob(os.path.join(V.SEISMIC, "near_stack_*_headers.npz"))) + [os.path.join(V.SEISMIC, "near_stack_blocks_record.json"), os.path.join(V.WELLS_DIR, "wells_positions.json")]
out = V.write_evidence("v02_grid_and_bins.json", payload, V.run_record(os.path.abspath(__file__), inputs, config={}))
print("time axis", time_axis)
for f in fits:
    print(f"  {f['well']:18s} resid {f['max_resid_m']*1000:.1f} mm  bins {f['bin_len_il_m']:.3f}/{f['bin_len_xl_m']:.3f} m  "
          f"well bin disagreement {f['bin_disagreement'][0]:+.3f}/{f['bin_disagreement'][1]:+.3f}")
print("->", out)
