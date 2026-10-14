import sys; sys.dont_write_bytecode = True
"""v01 - inventory and SHA-256 of every Poseidon raw file this workflow can use, and what each well file holds.
Item (1). Opens LAS/VSP files only to state what they contain (curves, units, ranges, sampling, NaN counts,
datum lines from the headers) and the seismic block records. No metric, tie, wavelet, gain or alignment
is computed here for any well.
    ~/.venvs/deepfwi/bin/python v01_inventory.py
"""
import glob
import json
import os

import numpy as np

import vlib as V

files = sorted(p for p in glob.glob(os.path.join(V.RAW, "**", "*"), recursive=True)
               if os.path.isfile(p) and not os.path.basename(p).startswith("."))
inventory = [V.file_record(p) for p in files]

wells = {}
for wdir in sorted(glob.glob(os.path.join(V.WELLS_DIR, "*"))):
    if not os.path.isdir(wdir) or wdir.endswith("info"):
        continue
    name = os.path.basename(wdir).replace("_", " ")
    entry = {"las": [], "vsp_listing": [], "segy": [], "other": []}
    for p in sorted(glob.glob(os.path.join(wdir, "**", "*"), recursive=True)):
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, V.WELLS_DIR)
        low = p.lower()
        try:
            if low.endswith(".las"):   # text scan for every LAS: lasio is too slow; curve ranges live in v03/v04
                # large LWD/time-based LAS: record curves and row count from the text, not through lasio
                names = []; nrows = 0; in_curve = False; in_data = False
                with open(p, "r", errors="ignore") as fh:
                    for line in fh:
                        u = line.strip()
                        if u.startswith("~"):
                            in_curve = u.upper().startswith("~C"); in_data = u.upper().startswith("~A"); continue
                        if in_curve and u and not u.startswith("#"):
                            names.append(u.split(".")[0].strip())
                        elif in_data and u and not u.startswith("#"):
                            nrows += 1
                entry["las"].append({"file": rel, "large_file_text_scan": True, "bytes": os.path.getsize(p), "curves": names[:60], "n_rows": nrows})
                continue
            if low.endswith(".las"):
                curves, meta, clist = V.las_curves(p)
                idx = next(iter(curves.values()))
                summ = {"file": rel, "index": clist[0][0], "index_unit": clist[0][1],
                        "index_range": [float(np.nanmin(idx)), float(np.nanmax(idx))],
                        "step": float(np.nanmedian(np.diff(idx))) if idx.size > 1 else None, "n": int(idx.size),
                        "curves": [{"mnemonic": m, "unit": u, "descr": d[:80],
                                    "valid": int(np.isfinite(curves[m]).sum()),
                                    "range": [float(np.nanmin(curves[m])), float(np.nanmax(curves[m]))]
                                    if np.isfinite(curves[m]).any() else None} for m, u, d in clist],
                        "header": {k: v for k, v in meta.items()
                                   if k in ("WELL", "KB", "EKB", "EGL", "APD", "DREF", "SRD", "STRT", "STOP", "STEP",
                                            "NULL", "COMP", "DATE", "LOC", "UWI", "API", "FLD", "CTRY")}}
                entry["las"].append(summ)
            elif low.endswith(".xlsx") and "timedepth" in low.replace("_", "").replace("-", ""):
                td = V.vsp_time_depth_xlsx(p)
                entry["vsp_listing"].append({"file": rel, "levels": int(td["tvd_srd"].size),
                                             "tvd_srd_range_m": [float(td["tvd_srd"].min()), float(td["tvd_srd"].max())],
                                             "twt_range_s": [float(td["twt_s"].min()), float(td["twt_s"].max())],
                                             "header": td["header"]})
            elif low.endswith((".sgy", ".segy")):
                import segyio
                with segyio.open(p, ignore_geometry=True) as s:
                    txt = s.text[0]
                    txt = txt.decode("ascii", "ignore") if isinstance(txt, (bytes, bytearray)) else str(txt)
                    entry["segy"].append({"file": rel, "traces": int(s.tracecount), "samples": int(s.samples.size),
                                          "dt_us": float(segyio.tools.dt(s)), "t0_ms": float(s.samples[0]),
                                          "text_header_head": txt[:400]})
            else:
                entry["other"].append(rel)
        except Exception as e:  # inventory must never stop on one bad file; the failure is recorded
            entry["other"].append({"file": rel, "error": str(e)[:200]})
    wells[name] = entry

seis = {}
for rec in glob.glob(os.path.join(V.SEISMIC, "*_blocks_record.json")):
    seis[os.path.basename(rec)] = json.load(open(rec))

payload = dict(item="(1) inventory and SHA-256 of every raw file; contents of each well file; seismic block records",
               n_files=len(inventory), total_bytes=int(sum(r["bytes"] for r in inventory)),
               inventory=inventory, wells=wells, seismic_records=seis)
run = V.run_record(os.path.abspath(__file__), files[:0], config={"raw_root": V.RAW})
out = V.write_evidence("v01_inventory.json", payload, run)
print("files", len(inventory), "bytes", payload["total_bytes"], "->", out)
for w, e in wells.items():
    print(f"  {w:18s} LAS {len(e['las']):2d}  VSP listings {len(e['vsp_listing'])}  SEG-Y {len(e['segy'])}  other {len(e['other'])}")
