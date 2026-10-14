import sys; sys.dont_write_bytecode = True
"""Shared helpers for the independent Poseidon 3D verification pass (verify/).

Adapted 2026-09-16 from the IBDP verify toolkit (vlib_ibdp_origin.py kept beside this file for the diff).
Nothing here fits anything a later stage will fit. Pure functions plus hashing and run-record helpers.

Forward convention (OURS, inherited from the frozen F3/IBDP protocols):
    r[q] = (Z[q+1] - Z[q]) / (Z[q+1] + Z[q])        at sample q
    synthetic = np.convolve(r, w)[c : c + N]         w[c] at tau = 0
Poseidon polarity: the processing report reversed the polarity so that a TROUGH = impedance INCREASE
(SEG reverse). Our synthetic convention gives a PEAK for an increase with a positive wavelet, so the
data must be compared with w -> -w (or the data negated); this is a declared fact, not a fitted phase.
"""
import datetime
import glob
import hashlib
import json
import math
import os
import platform

import numpy as np

NULL = -999.25
HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
ROOT = os.path.dirname(VERIFY)                       # POSEIDON_FIELD_INVERSION
EVID = os.path.join(VERIFY, "evidence")

# Raw data root (read-only). Poseidon3D/ is being moved from the manuscript root to DATA_POST_STACK/;
# both locations are tried and the one used is written into every run record.
_CANDIDATES = [os.path.join(os.path.dirname(ROOT), "Poseidon3D"),
               os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Poseidon3D")]
RAW = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])
SEISMIC = os.path.join(RAW, "seismic")
WELLS_DIR = os.path.join(RAW, "wells")
REPORTS = os.path.join(RAW, "reports")

WELLS = ["Boreas 1", "Crown 1", "Kronos 1", "Pharos 1", "Poseidon 1", "Poseidon 2", "Poseidon North 1",
         "Proteus 1", "Torosa 1", "Zephyros 1"]

# Survey facts established 2026-09-16 (Poseidon3D/provenance/SOURCE.md); every one is re-derived in v02.
NOMINAL_DT = 0.004
NOMINAL_T0 = 0.0
AFFINE_A = np.array([[1205.226958, -957.555533], [1436.333333, 803.484399]])   # raw header units per bin
AFFINE_B = np.array([40911129.891, 842942630.308])
COORD_SCALAR = -100                                                            # metres = raw / 100
ANGLE_RANGES = {"near": (6, 18), "mid": (18, 30), "far": (30, 42), "full_agc": (6, 42)}
POLARITY = "SEG reverse: trough = impedance increase (CGGVeritas processing report, final sequence step 6)"

# F3/IBDP frozen timing-gate constants: S_gate = c_gate(metric) / f_hi.
GATE_C = {"corr_band_envelope": 1.0638281250000001, "corr_lowpass": 1.0638281250000001,
          "corr_relative_band": 0.13014375, "rel_rmse": 0.0}


def well_tag(name):
    return name.replace(" ", "_")


def seismic_block(well, stack="near_stack"):
    """The extracted sub-volume around a well: float32 [inline, crossline, time] plus its header block."""
    tag = well_tag(well)
    npy = os.path.join(SEISMIC, f"{stack}_{tag}.npy")
    hdr = np.load(os.path.join(SEISMIC, f"{stack}_{tag}_headers.npz"))
    rec = json.load(open(os.path.join(SEISMIC, f"{stack}_blocks_record.json")))
    blk = next(b for b in rec["blocks"] if b["well"] == well)
    return np.load(npy, mmap_mode="r"), hdr, blk, rec


def las_curves(path):
    """LAS -> dict of float arrays (NULL -> NaN) with the index first, plus the raw header dict."""
    import lasio
    l = lasio.read(path, ignore_header_errors=True)
    out = {}
    for c in l.curves:
        v = np.asarray(c.data, float)
        v[v == NULL] = np.nan
        out[c.mnemonic] = v
    meta = {k: (l.well[k].value, l.well[k].unit, l.well[k].descr) for k in l.well.keys()}
    return out, meta, [(c.mnemonic, c.unit, c.descr) for c in l.curves]


def vsp_time_depth_xlsx(path):
    """ConocoPhillips/Schlumberger VSP time-depth listing (xlsx): TVD-from-SRD [m] and TWT [s] per level."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    hdr = [str(c) for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r and r[2] is not None and r[5] is not None]
    tvd = np.array([float(r[2]) for r in rows]); twt = np.array([float(r[5]) for r in rows]) / 1000.0
    md = np.array([float(r[1]) for r in rows])
    o = np.argsort(tvd)
    return dict(md=md[o], tvd_srd=tvd[o], twt_s=twt[o], header=hdr)


# ---------------------------------------------------------------- hashing / records
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_record(path):
    st = os.stat(path)
    return dict(path=path, bytes=int(st.st_size),
                mtime_utc=datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat(timespec="seconds") + "Z",
                sha256=sha256(path))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def run_record(script_path, inputs, config):
    """Run record: SHA-256 of the script, of vlib, of every input, plus the config and the environment."""
    return dict(
        script=dict(path=script_path, sha256=sha256(script_path)),
        vlib=dict(path=os.path.join(HERE, "vlib.py"), sha256=sha256(os.path.join(HERE, "vlib.py"))),
        run_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
        executable=sys.executable,
        release_root_used=RAW,
        release_root_note="Poseidon3D raw folder (AWS near-stack blocks, NOPIMS/Drive well files, ConocoPhillips reports), read-only",
        inputs=[file_record(p) for p in inputs],
        config=config)


def write_evidence(name, payload, run):
    os.makedirs(EVID, exist_ok=True)
    out = os.path.join(EVID, name)
    payload = dict(payload)
    payload["run"] = run
    with open(out, "w") as fh:
        json.dump(jsonable(payload), fh, indent=1, sort_keys=False)
    # self-hash of the written artefact, appended as a sidecar so the output hash is recorded too
    with open(out + ".sha256", "w") as fh:
        fh.write(sha256(out) + "  " + os.path.basename(out) + "\n")
    return out



# ---------------------------------------------------------------- statistics
def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    da, db = math.sqrt(float(a @ a)), math.sqrt(float(b @ b))
    return float(a @ b / (da * db)) if da > 0 and db > 0 else float("nan")


def hilbert_pair(w):
    """Zero-phase member w0 = w and its +90 deg partner w90, via the discrete Hilbert transform of the taps.
    A constant-phase rotation by phi is then w0 cos(phi) + w90 sin(phi)."""
    from scipy.signal import hilbert
    an = hilbert(np.asarray(w, float))
    return np.real(an), -np.imag(an)


def best_constant_phase(d, s0, s90, phi_grid):
    """Pearson r maximised over a constant-phase rotation of the synthetic pair. Returns (r, phi_deg)."""
    dc = np.asarray(d, float) - np.mean(d)
    A = np.asarray(s0, float) - np.mean(s0)
    B = np.asarray(s90, float) - np.mean(s90)
    c, s = np.cos(np.deg2rad(phi_grid)), np.sin(np.deg2rad(phi_grid))
    num = (A @ dc) * c + (B @ dc) * s
    den = np.sqrt((A @ A) * c ** 2 + (B @ B) * s ** 2 + 2 * (A @ B) * c * s)
    den = np.where(den > 0, den, np.inf) * math.sqrt(float(dc @ dc))
    R = num / den
    k = int(np.nanargmax(R))
    return float(R[k]), float(phi_grid[k]), R


def circular_shift_null(trace, span, rng, n):
    """Circular shifts of the trace inside the scoring span (F3 tie-protocol null)."""
    lo, hi = span
    seg = trace[lo:hi]
    out = []
    for _ in range(n):
        k = int(rng.integers(1, len(seg)))
        x = trace.copy()
        x[lo:hi] = np.roll(seg, k)
        out.append(x)
    return out


def phase_randomised(x, rng):
    F = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), len(x))


def amplitude_spectrum(w, dt, centre, n_grid=40001):
    """Continuous-time transform on [0, f_Nyq] with tau = (k - centre) dt (invlib.ctft, reimplemented)."""
    w = np.asarray(w, float)
    tau = (np.arange(len(w)) - centre) * dt
    f = np.linspace(0.0, 0.5 / dt, n_grid)
    W = dt * (np.exp(-2j * np.pi * np.outer(f, tau)) @ w)
    return f, W


def band_stats(w, dt, centre):
    """Peak, half-amplitude (-6 dB) band and -12 dB band of |W(f)|, plus the energy fraction above 80 Hz."""
    f, W = amplitude_spectrum(w, dt, centre)
    a = np.abs(W)
    pk = a.max()
    def band(frac):
        m = f[a >= frac * pk]
        return (float(m.min()), float(m.max())) if m.size else (float("nan"), float("nan"))
    e = a ** 2
    tot = float(np.trapezoid(e, f))
    hi = float(np.trapezoid(np.where(f >= 80.0, e, 0.0), f))
    return dict(peak_hz=float(f[int(np.argmax(a))]),
                half_amplitude_band_hz=band(0.5), minus12dB_band_hz=band(10 ** (-12 / 20)),
                energy_fraction_above_80hz=hi / tot if tot > 0 else float("nan"))
