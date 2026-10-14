import sys; sys.dont_write_bytecode = True
"""Shared helpers for the independent IBDP verification pass (verify/).

Nothing here fits anything a later stage will fit. Pure functions plus hashing and run-record helpers.

Forward convention (OURS, inherited from the frozen F3 protocol, invlib.py header):
    r[q] = (Z[q+1] - Z[q]) / (Z[q+1] + Z[q])        at sample q
    synthetic = np.convolve(r, w)[c : c + N]         w[c] at tau = 0
so data row j is modelled from reflectivity centred on model sample j (placement "OURS").

Authors' convention (SeisInvSVGD Case3 svgd.py / inversion_VW*.py):
    refl_coef = (Ip[1:] - Ip[:-1]) / (Ip[1:] + Ip[:-1]) ; d = wavelet @ refl_coef
    compared against seismic[1:], i.e. reflectivity index j is placed at cube sample j+1
so the authors' placement is OURS advanced by one sample ("AUTHORS").
"""
import datetime
import hashlib
import json
import math
import os
import platform

import numpy as np

NULL = -999.25
HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
ROOT = os.path.dirname(VERIFY)
EVID = os.path.join(VERIFY, "evidence")

# Release location. The task statement names <MANUSCRIPT>/SeisInvSVGD/; that path does not exist on
# this machine. The only copy of the release is the read-only tree below. Recorded in every run record.
RELEASE = os.path.join(os.path.dirname(ROOT), "DATASET_ASSESSMENT_IBDP", "SeisInvSVGD")
CASE3 = os.path.join(RELEASE, "Case3")
DATA = os.path.join(CASE3, "data")
WELLS_DIR = os.path.join(DATA, "wells")
SEISMIC_NPY = os.path.join(DATA, "seismic.npy")
WAVELET_NPY = os.path.join(DATA, "wavelet.npy")
EDX_LAS = os.path.join(os.path.dirname(ROOT), "DATASET_ASSESSMENT_IBDP", "licensing", "edx_members")

WELLS = ["CCS1", "CCS2", "VW1", "VW2"]
CURVES = ["DEPTH", "TVD", "TVDSS", "TIME", "DTS", "PHIT", "GR", "RHOB", "DT", "AI", "SEIS"]

# Nominal cube axes as stated by the earlier assessment; every one of these is re-derived in v02 and
# never taken on trust downstream.
NOMINAL_T0 = 0.700
NOMINAL_DT = 0.002

# F3 frozen timing-gate constants (inversion_v3/code/metrics.py timing_gate, amendment v1.1 A1):
#   S_gate = c_gate(metric) / f_hi   with f_hi the upper half-amplitude frequency of the case wavelet.
GATE_C = {"corr_band_envelope": 1.0638281250000001, "corr_lowpass": 1.0638281250000001,
          "corr_relative_band": 0.13014375, "rel_rmse": 0.0}


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
        release_root_used=RELEASE,
        release_root_note=("the task statement names <MANUSCRIPT>/SeisInvSVGD/, which does not exist on this "
                           "machine; the only copy of the release is DATASET_ASSESSMENT_IBDP/SeisInvSVGD/, "
                           "treated as read-only"),
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


# ---------------------------------------------------------------- well CSV
def load_well(name):
    """Raw CSV as a dict of float arrays with NULL (-999.25) mapped to NaN, plus the raw text hash."""
    path = os.path.join(WELLS_DIR, name + ".csv")
    with open(path, "r") as fh:
        head = fh.readline().strip()
    cols = head.split(",")
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    d = {c: arr[:, i].copy() for i, c in enumerate(cols)}
    raw = {c: arr[:, i].copy() for i, c in enumerate(cols)}
    for c in d:
        d[c][np.isclose(d[c], NULL, atol=1e-6)] = np.nan
    return dict(name=name, path=path, columns=cols, n_rows=arr.shape[0], curves=d, raw=raw)


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
