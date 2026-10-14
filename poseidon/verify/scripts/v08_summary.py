import sys; sys.dont_write_bytecode = True
"""v08 - assemble the Poseidon verification evidence into one record: per-well status, the tie table, the
alignment budget (leave-one-well-out lag spread against the v05 gate), effective samples and power, and what
this dataset can and cannot support BEFORE any evaluation score exists. Reads only v02-v07 evidence JSONs and
hashes them; computes nothing new except the cross-references.
    ~/.venvs/deepfwi/bin/python v08_summary.py
"""
import json
import os

import numpy as np

import vlib as V

ev = {}
for name in ("v02_grid_and_bins", "v03_wells_datums", "v04_time_depth", "v05_seismic_band", "v06_ties", "v07_rows_and_power"):
    p = os.path.join(V.EVID, name + ".json")
    ev[name] = json.load(open(p)) if os.path.exists(p) else None
v04, v05, v06, v07 = ev["v04_time_depth"], ev["v05_seismic_band"], ev["v06_ties"], ev["v07_rows_and_power"]

# tie table and leave-one-out alignment budget
ties = {w: r for w, r in (v06["wells"] if v06 else {}).items() if "r_best" in r}
lags = {w: r["best_lag_ms"] for w, r in ties.items()}
budget = {}
for w in lags:
    others = np.array([v for k, v in lags.items() if k != w])
    pred = float(np.median(others)) if others.size else float("nan")
    budget[w] = dict(own_lag_ms=lags[w], loo_predicted_lag_ms=pred, loo_error_ms=abs(lags[w] - pred) if others.size else float("nan"),
                     others_spread_ms=float(others.max() - others.min()) if others.size else float("nan"))
gate = v05["summary"]["s_gate_rel_band_ms_median"] if v05 else float("nan")
per_well = {}
for w in V.WELLS:
    s = dict(seismic_block=bool(v05 and w in v05["blocks"]), time_depth=bool(v04 and v04["wells"].get(w, {}).get("levels")),
             impedance_in_time=bool(v07 and "rows_4ms" in v07["wells"].get(w, {})), tied=w in ties)
    if w in ties:
        s.update(r_best=ties[w]["r_best"], lag_ms=ties[w]["best_lag_ms"], p_circular=ties[w]["null_circular"]["p"], p_phase=ties[w]["null_phase_random"]["p"],
                 beats_both_nulls=bool(ties[w]["r_best"] > max(ties[w]["null_circular"]["q95"], ties[w]["null_phase_random"]["q95"])))
    if w in budget:
        s.update(loo_error_ms=budget[w]["loo_error_ms"], passes_gate_loo=bool(budget[w]["loo_error_ms"] <= gate))
    if v07 and "rows_4ms" in v07["wells"].get(w, {}):
        s.update(rows=v07["wells"][w]["rows_4ms"], n_eff_2BT=v07["wells"][w]["n_eff_2BT"], n_eff_PP=v07["wells"][w]["n_eff_pyper_peterman_bandpassed"])
    per_well[w] = s
usable = [w for w, s in per_well.items() if s["tied"] and s.get("beats_both_nulls")]
verdict = dict(
    q1_timing=dict(gate_ms=gate, wells_with_loo_error=len(budget), wells_passing_gate_loo=[w for w in budget if budget[w]["loo_error_ms"] <= gate],
                   note="LOO error = |own lag - median lag of the other tied wells|; the freeze must declare the bulk shift rule and this budget"),
    q2_effective_samples=dict(pooled_2BT=v07["pooled"]["n_2bt"] if v07 else None, pooled_PP=v07["pooled"]["n_pp"] if v07 else None,
                              zou_mdd=v07["zou_min_detectable"] if v07 else None),
    q3_wells=dict(tied_and_beating_nulls=usable, blocks_present=[w for w, s in per_well.items() if s["seismic_block"]],
                  no_time_depth=[w for w, s in per_well.items() if not s["time_depth"]]),
    status="PROVISIONAL until all ten blocks are extracted and v06 has tied every well with impedance in time")
payload = dict(item="(8) summary of the Poseidon verification evidence and the provisional feasibility picture", per_well=per_well, tie_table=ties and {w: {k: r[k] for k in ("rows", "r_at_zero_lag", "best_lag_ms", "r_best")} for w, r in ties.items()},
               alignment_budget=budget, gate_ms=gate, verdict=verdict,
               evidence_hashes={k: V.sha256(os.path.join(V.EVID, k + ".json")) for k, v in ev.items() if v is not None})
o = V.write_evidence("v08_summary.json", payload, V.run_record(os.path.abspath(__file__), [os.path.join(V.EVID, k + ".json") for k, v in ev.items() if v is not None], config={}))
for w, s in per_well.items():
    print(f"  {w:18s} block={int(s['seismic_block'])} td={int(s['time_depth'])} imp_t={int(s['impedance_in_time'])} tied={int(s['tied'])}"
          + (f"  r {s['r_best']:+.2f} @ {s['lag_ms']:+.0f} ms  loo_err {s.get('loo_error_ms', float('nan')):.1f} ms (gate {gate:.1f})" if s['tied'] else ""))
print("  verdict:", json.dumps(verdict["q1_timing"])[:200]); print("->", o)
