"""
Banded, multi-process versions of the two classical ADMM inversions.

`admm_l1_traces` computes the same iterations as `admm_l1_single_trace` in
2D_ADMM_PINN_ResCNN_Attention.py for every trace of a section, and
`admm_l1_2d_tv_banded` computes the same iterations as `admm_l1_2d_tv`.

The system matrix of every L-update,

    A = (W D/2)^T (W D/2) + alpha I + lambda (D/2)^T diag(M^2) (D/2),

is symmetric positive definite and banded: the Gram matrix of the forward
operator has half-bandwidth nw (the wavelet length) and the reweighting term is
tridiagonal.  The dense solve of the original code costs O(m^3) per iteration;
the banded Cholesky solve used here costs O(m nw^2), which for the complete
Marmousi-2 model (m = 2801, nw = 96) is about 300 times cheaper for the same
answer.  Traces are independent, so they are distributed over a pool of worker
processes; the lateral TV step of the 2-D solver works on all traces at once
with a tridiagonal solve.
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.linalg import solveh_banded, toeplitz


# ------------------------------------------------------------------ operators

def convolution_matrix(w, n):
    """(n + nw - 1) x n Toeplitz convolution matrix (same as the main code)."""
    nw = len(w)
    col = np.zeros(n + nw - 1)
    col[:nw] = w
    row = np.zeros(n)
    row[0] = w[0]
    return toeplitz(col, row)


def difference_matrix(m):
    """First-difference matrix D, shape (m - 1, m) (same as the main code)."""
    D = np.zeros((m - 1, m))
    idx = np.arange(m - 1)
    D[idx, idx] = -1.0
    D[idx, idx + 1] = 1.0
    return D


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def banded_operators(wavelet, m):
    """
    Forward map WDh = W D/2 (dense, (m + nw - 2) x m) and the upper-band
    storage of its Gram matrix G = WDh^T WDh, shape (nw + 1, m), in the layout
    expected by scipy.linalg.solveh_banded (ab[nw + i - j, j] = G[i, j], i <= j).
    """
    wavelet = np.asarray(wavelet, dtype=np.float64)
    WDh = convolution_matrix(wavelet, m - 1) @ (0.5 * difference_matrix(m))
    G = WDh.T @ WDh
    kb = len(wavelet)
    ab = np.zeros((kb + 1, m))
    for off in range(kb + 1):
        ab[kb - off, off:] = np.diagonal(G, off)
    if m > kb + 1:
        beyond = np.abs(np.diagonal(G, kb + 1)).max()
        if beyond > 1e-10 * np.abs(G).max():
            raise ValueError("Gram matrix is wider than the assumed band")
    return ab, WDh


def forward_model(L, wavelet):
    """
    Zero-phase seismic of every trace of a log-impedance section L (nx, m):
    exact reflectivity followed by full convolution with the wavelet, returned
    with shape (nx, m + nw - 2).  Same numbers as the per-trace loop of the
    main code, computed as one matrix product.
    """
    Z = np.exp(L)
    r = (Z[:, 1:] - Z[:, :-1]) / (Z[:, 1:] + Z[:, :-1])
    W = convolution_matrix(np.asarray(wavelet, dtype=np.float64), L.shape[1] - 1)
    return r @ W.T


# ------------------------------------------------------------- worker process

_W = {}


def _init_worker(ab, params):
    _W["ab"] = ab
    _W["params"] = params
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _trace_iterations(WtS_i, L0_i, ab_G, mu, alpha, lam, eps, max_iter, tol,
                      reweight, R, C, M, n_iter, stop_on_tol):
    """
    Reweighted-L1 ADMM iterations of one trace with banded L-updates.
    Update order, reweighting and stopping rule follow admm_l1_single_trace.
    Returns L, R, C, M and the number of iterations done.
    """
    m = L0_i.size
    kb = ab_G.shape[0] - 1
    L = L0_i.copy()
    done = 0
    for _ in range(n_iter):
        L_old = L
        m2 = M * M
        ab = ab_G.copy()
        ab[kb, :] += alpha
        ab[kb, :-1] += 0.25 * lam * m2
        ab[kb, 1:] += 0.25 * lam * m2
        ab[kb - 1, 1:] -= 0.25 * lam * m2
        v = M * (R - C)
        b = WtS_i + alpha * L0_i
        b[:-1] -= 0.5 * lam * v
        b[1:] += 0.5 * lam * v
        L = solveh_banded(ab, b, overwrite_ab=True, overwrite_b=True,
                          check_finite=False)
        y = 0.5 * (L[1:] - L[:-1])
        R = soft_threshold(M * y + C, mu / lam)
        C = C + (M * y - R)
        if reweight:
            M = 1.0 / (np.abs(y) + eps)
        done += 1
        if stop_on_tol:
            rel = np.linalg.norm(L - L_old) / (np.linalg.norm(L_old) + 1e-12)
            if rel < tol:
                break
    return L, R, C, M, done


def _task_traces(args):
    """Full per-trace inversion of one chunk (used by admm_l1_traces)."""
    WtS, L0, alpha = args
    p = _W["params"]
    n, m = L0.shape
    L_out = np.empty((n, m))
    iters = np.empty(n, dtype=np.int32)
    for i in range(n):
        L_out[i], _, _, _, iters[i] = _trace_iterations(
            WtS[i], L0[i], _W["ab"], p["mu"], float(alpha[i]), p["lam"], p["eps"],
            p["max_iter"], p["tol"], p["reweight"],
            np.zeros(m - 1), np.zeros(m - 1), np.ones(m - 1),
            p["max_iter"], True)
    return L_out, iters


def _task_vertical(args):
    """Fixed number of vertical sweeps of one chunk (used by the 2-D solver)."""
    WtS_T, L0_T, Rv, Cv, Mt, alpha = args
    p = _W["params"]
    n, m = L0_T.shape
    L_out = np.empty((n, m))
    for i in range(n):
        L_out[i], Rv[i], Cv[i], Mt[i], _ = _trace_iterations(
            WtS_T[i], L0_T[i], _W["ab"], p["mu"], float(alpha[i]), p["lam"],
            p["eps"], p["max_iter"], p["tol"], p["reweight"],
            Rv[i], Cv[i], Mt[i], p["vert_inner"], False)
    return L_out, Rv, Cv, Mt


def well_terms(alpha, L0, well_cols, well_logs, well_weight):
    """
    Fold a quadratic well penalty  well_weight/2 * ||L_i - log_i||^2  on the
    traces `well_cols` into the Tikhonov term of every trace: the penalised
    traces get weight alpha + well_weight and the weighted mean of the
    starting model and the log as target.  Returns (alpha_vec, L0_eff).
    """
    nx = L0.shape[0]
    alpha_vec = np.full(nx, float(alpha))
    L0_eff = np.array(L0, dtype=np.float64, copy=True)
    if well_cols is None or well_weight <= 0.0:
        return alpha_vec, L0_eff
    for c, log in zip(well_cols, well_logs):
        a = alpha + well_weight
        L0_eff[c] = (alpha * L0[c] + well_weight * np.asarray(log, dtype=np.float64)) / a
        alpha_vec[c] = a
    return alpha_vec, L0_eff


def _chunks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _pool(n_jobs, ab, params):
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1
    return ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                               initargs=(ab, params))


# ------------------------------------------------------- trace-wise inversion

def admm_l1_traces(S, wavelet, L0, mu, alpha, lambda_, epsilon=1e-4,
                   max_iter=200, tol=1e-6, reweight=True, n_jobs=None,
                   chunk=64, well_cols=None, well_logs=None, well_weight=0.0):
    """
    Reweighted-L1 ADMM inversion of every trace of S (nx, m + nw - 2) from the
    starting log-impedance L0 (nx, m).  Returns the impedance exp(L) (nx, m)
    and the number of iterations used by each trace.  With well_cols /
    well_logs / well_weight the traces well_cols carry an extra quadratic
    penalty towards the given log-impedance logs (see well_terms).
    """
    S = np.asarray(S, dtype=np.float64)
    L0 = np.asarray(L0, dtype=np.float64)
    nx, m = L0.shape
    ab, WDh = banded_operators(wavelet, m)
    WtS = S @ WDh
    alpha_vec, L0 = well_terms(alpha, L0, well_cols, well_logs, well_weight)
    params = dict(mu=mu, alpha=alpha, lam=lambda_, eps=epsilon,
                  max_iter=max_iter, tol=tol, reweight=reweight)
    L = np.empty((nx, m))
    iters = np.empty(nx, dtype=np.int32)
    with _pool(n_jobs, ab, params) as pool:
        parts = _chunks(nx, chunk)
        for sl, (L_c, it_c) in zip(parts, pool.map(
                _task_traces, ((WtS[sl], L0[sl], alpha_vec[sl]) for sl in parts))):
            L[sl] = L_c
            iters[sl] = it_c
    return np.exp(L), iters


# ------------------------------------------------- 2-D coupled TV inversion

def tv1d_columns_banded(A, lam, beta=1.0, n_iter=60, tol=1e-7, n_threads=1):
    """
    Anisotropic 1-D total-variation denoising along axis 0 of every column of
    A, the same ADMM as _tv1d_columns of the main code, with the fixed
    tridiagonal system solved in banded form.  The columns are independent,
    so every inner iteration is carried out block of columns by block of
    columns in `n_threads` threads (LAPACK and the NumPy array operations
    release the interpreter lock); only the convergence test combines the
    blocks.  The numbers do not depend on the thread count.
    """
    nx, m = A.shape
    if nx < 2 or lam <= 0.0:
        return A.copy()
    ab = np.zeros((2, nx))
    ab[1, :] = 1.0 + 2.0 * beta
    ab[1, 0] = ab[1, -1] = 1.0 + beta
    ab[0, 1:] = -beta
    n_threads = max(1, min(int(n_threads), m))
    cols = _chunks(m, -(-m // n_threads))
    thr = lam / beta

    V = A.copy()
    Z = V[1:, :] - V[:-1, :]
    Ud = np.zeros_like(Z)

    def step(sl):
        """One ADMM iteration on the columns sl; returns the squared norms
        of the change of Z and of Z for the convergence test."""
        Q = Z[:, sl] - Ud[:, sl]
        rhs = A[:, sl].copy()
        rhs[:-1, :] -= beta * Q
        rhs[1:, :] += beta * Q
        V[:, sl] = solveh_banded(ab, rhs, check_finite=False)
        DV = V[1:, sl] - V[:-1, sl]
        Z_new = soft_threshold(DV + Ud[:, sl], thr)
        Ud[:, sl] += DV - Z_new
        change = float(np.sum((Z_new - Z[:, sl]) ** 2))
        size = float(np.sum(Z[:, sl] ** 2))
        Z[:, sl] = Z_new
        return change, size

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for _ in range(n_iter):
            parts = list(ex.map(step, cols)) if n_threads > 1 else [step(sl) for sl in cols]
            change = np.sqrt(sum(p[0] for p in parts))
            size = np.sqrt(sum(p[1] for p in parts))
            if change / (size + 1e-12) < tol:
                break
    return V


def admm_l1_2d_tv_banded(S, wavelet, L0, mu_t, alpha, lambda_t, mu_x=None,
                         lambda_x=None, epsilon=1e-4, max_iter=40, tol=1e-5,
                         rho=None, vert_inner=12, reweight=True, n_jobs=None,
                         chunk=64, verbose=False, well_cols=None, well_logs=None,
                         well_weight=0.0):
    """
    Spatially coupled reweighted-L1 + lateral-TV inversion of a section,
    identical in its iterations to admm_l1_2d_tv of the main code.  The
    vertical sub-problem is solved trace by trace in worker processes with
    banded L-updates; the lateral sub-problem is a tridiagonal solve over all
    traces.  Returns exp((L + V) / 2) and the list of outer residuals.
    """
    S = np.asarray(S, dtype=np.float64)
    L0 = np.asarray(L0, dtype=np.float64)
    nx, m = L0.shape
    if mu_x is None:
        mu_x = 8.0 * mu_t
    if rho is None:
        rho = 4.0 * alpha
    alpha_r = alpha + rho

    ab, WDh = banded_operators(wavelet, m)
    WtS = S @ WDh
    # optional well penalty: the penalised traces get the extra weight on top
    # of alpha_r and the weighted mean of consensus target and log as target
    w_alpha = np.zeros(nx)
    w_target = np.zeros((nx, m))
    if well_cols is not None and well_weight > 0.0:
        for c, log in zip(well_cols, well_logs):
            w_alpha[c] = well_weight
            w_target[c] = well_weight * np.asarray(log, dtype=np.float64)
    params = dict(mu=mu_t, alpha=alpha_r, lam=lambda_t, eps=epsilon,
                  max_iter=max_iter, tol=tol, reweight=reweight,
                  vert_inner=vert_inner)

    L = L0.copy()
    V = L0.copy()
    U = np.zeros((nx, m))
    Rv = np.zeros((nx, m - 1))
    Cv = np.zeros((nx, m - 1))
    Mt = np.ones((nx, m - 1))
    lat_lam = mu_x / rho
    convergence = []
    parts = _chunks(nx, chunk)
    n_threads = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)

    with _pool(n_jobs, ab, params) as pool:
        for outer in range(max_iter):
            L_old = L.copy()
            # consensus target folded into the Tikhonov term: the vertical
            # sub-problem then has weight alpha_r and starting model T
            alpha_vec = alpha_r + w_alpha
            T = (alpha * L0 + rho * (V - U) + w_target) / alpha_vec[:, None]
            tasks = ((WtS[sl], T[sl], Rv[sl], Cv[sl], Mt[sl], alpha_vec[sl]) for sl in parts)
            for sl, (L_c, R_c, C_c, M_c) in zip(parts, pool.map(_task_vertical, tasks)):
                L[sl] = L_c
                Rv[sl] = R_c
                Cv[sl] = C_c
                Mt[sl] = M_c
            A_lat = L + U
            V = (tv1d_columns_banded(A_lat, lat_lam, n_threads=n_threads)
                 if mu_x > 0.0 else A_lat.copy())
            U += L - V
            rel = np.linalg.norm(L - L_old) / (np.linalg.norm(L_old) + 1e-12)
            convergence.append(rel)
            if verbose:
                print(f"    outer {outer + 1:3d}/{max_iter}  rel change {rel:.3e}", flush=True)
            if rel < tol:
                break
    return np.exp(0.5 * (L + V)), convergence
