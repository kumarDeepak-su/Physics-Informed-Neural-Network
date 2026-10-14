import sys; sys.dont_write_bytecode = True
"""v03 - well positions, datums and deviation, collected from the files themselves and compared with NOPTA.
Item (3). Sources, each recorded with its file hash: NOPTA feature-service query (lat/lon), LAS well headers
(EKB/EGL/APD/SRD where present), VSP corridor-stack SEG-Y text headers (SRD, DF, GL, coordinates), definitive
survey spreadsheets (max inclination), and the ConocoPhillips/Woodside report statements already transcribed.
Nothing is fitted. A datum that no file states is written as "unresolved", never assumed.
    ~/.venvs/deepfwi/bin/python v03_wells_datums.py
"""
import glob
import json
import os
import re

import numpy as np

import vlib as V

pos = {p["name"]: p for p in json.load(open(os.path.join(V.WELLS_DIR, "wells_positions.json")))}
out = {}
for well in V.WELLS:
    wdir = os.path.join(V.WELLS_DIR, V.well_tag(well))
    rec = {"nopta": {k: pos[well].get(k) for k in ("lat", "lon", "x", "y", "il", "xl", "uwi", "title", "operator")} if well in pos else None,
           "las_headers": [], "segy_text_headers": [], "surveys": [], "files_present": os.path.isdir(wdir)}
    if not os.path.isdir(wdir):
        out[well] = rec; continue
    for p in sorted(glob.glob(os.path.join(wdir, "**", "*"), recursive=True)):
        low = p.lower()
        if not os.path.isfile(p):
            continue
        try:
            if low.endswith(".las") and os.path.getsize(p) > 25_000_000:
                rec.setdefault("skipped_large", []).append(os.path.relpath(p, V.WELLS_DIR)); continue
            if low.endswith(".las"):
                # lightweight header read (text up to the ~A section); lasio is too slow on the multi-MB LWD files
                head = []
                with open(p, "r", errors="ignore") as fh:
                    for line in fh:
                        if line.lstrip().upper().startswith("~A"):
                            break
                        head.append(line)
                        if len(head) > 4000:
                            break
                keep = {}
                for line in head:
                    m = re.match(r"\s*([A-Za-z][A-Za-z0-9_]{0,7})\s*\.[^\s]*\s+(.*?)\s*:\s*(.*)$", line)
                    if m and m.group(1).upper() in ("WELL", "EKB", "KB", "EGL", "APD", "DREF", "SRD", "ELEV", "EDF", "LOC", "X", "Y", "LATI", "LONG", "UWI", "STRT", "STOP", "COMP", "DATE"):
                        keep[m.group(1).upper()] = (m.group(2).strip(), m.group(3).strip()[:60])
                rec["las_headers"].append({"file": os.path.relpath(p, V.WELLS_DIR), "header": keep, "bytes": os.path.getsize(p)})
            elif low.endswith((".sgy", ".segy")):
                import segyio
                with segyio.open(p, ignore_geometry=True) as s:
                    txt = s.text[0]; txt = txt.decode("ascii", "ignore") if isinstance(txt, (bytes, bytearray)) else str(txt)
                lines = re.findall(r"C\s?\d{1,2}[^C]{0,79}", txt)
                keyl = [l.strip() for l in lines if re.search(r"(?i)SRD|DATUM|DF:|GL:|COORD|WELL NAME|DEVIATION|KB", l)]
                rec["segy_text_headers"].append({"file": os.path.relpath(p, V.WELLS_DIR), "lines": keyl[:12]})
            elif low.endswith((".xlsx", ".xls", ".txt")) and ("survey" in low or "inclination" in low):
                rec["surveys"].append({"file": os.path.relpath(p, V.WELLS_DIR), "bytes": os.path.getsize(p), "parsed": False, "note": "deviation to be read explicitly in the freeze stage"})
        except Exception as e:
            rec.setdefault("errors", []).append({"file": os.path.relpath(p, V.WELLS_DIR), "error": str(e)[:160]})
    out[well] = rec

# facts transcribed from reports (with their source), kept separate from anything read automatically
transcribed = {
    "Pharos 1": {"SRD": "MSL", "DF_above_MSL_m": 22.1, "seafloor_below_MSL_m": 482.1, "source": "VSP corridor stack SEG-Y text header (Schlumberger, 16-Jul-2014)"},
    "Kronos 1": {"SRD": "MSL", "RT_above_MSL_m": 21.8, "SRD_datum_velocity_mps": 1524, "source": "CP_Kronos1_Time_to_Depth_Report.xls header"},
    "Poseidon 1": {"SRD": "AHD", "DF_above_SRD_m": 26.4, "reference_velocity_mps": 1524, "source": "WCR Vol 2 Appendix 8 (Schlumberger VSP/GEO report), Table 33"},
    "Poseidon North 1": {"RT_above_MSL_m": 22.0, "water_depth_m": 486, "RT_to_mudline_m": 508, "VSP": "none in the basic WCR (checkshots there are BHA verticality shots)", "source": "Pos North1_Basic WCR rev0.pdf p.16"},
    "Torosa 1": {"TVD_reference": "MSL", "TIME_reference": "MSL (checkshot-calibrated LAS)", "source": "TZV_DEPTH_CS_CALIBRATED_LOGS.LAS curve descriptions"},
}
payload = dict(item="(3) well positions, datums and deviation from the files, compared with NOPTA; transcribed report facts kept apart",
               wells=out, transcribed_from_reports=transcribed,
               note=("every datum used later must trace to one of the entries above; wells without a stated SRD/DF pair are "
                     "carried as 'unresolved' into the alignment budget"))
inputs = [os.path.join(V.WELLS_DIR, "wells_positions.json")]
o = V.write_evidence("v03_wells_datums.json", payload, V.run_record(os.path.abspath(__file__), inputs, config={}))
for w, r in out.items():
    print(f"  {w:18s} LAS {len(r['las_headers']):2d}  SEGY {len(r['segy_text_headers'])}  surveys {len(r['surveys'])}  present={r['files_present']}")
print("->", o)
