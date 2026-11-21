"""Calibration of the constants of the analytic models on a set of strain states (used by calibrate.py).

Joint nonlinear least squares on the CBM, VBM and gap residuals, with a ridge penalty on the five second-order constants
whose strength is chosen by five-fold cross-validation, and several starting points for the shear coupling D_BP. The
valence band is fitted in the form physics.physics_vbm evaluates. With two_band=True the band separation DELTA_X of the
conduction-band shear term is fitted as a 14th constant (bounded to 0.05-10 eV).
"""

import numpy as np
from scipy.optimize import least_squares

import physics as P

KEYS = P.CBM_KEYS + P.VBM_KEYS  # 13 constants
KEYS2 = KEYS + ["DELTA_X"]  # with the two-band shear term
HIGH = ["XI_Q", "XI_C", "XI_Q2", "AV2", "B2_BP"]
HIGH_IDX = [KEYS.index(k) for k in HIGH]


def cbm_form(s, th):
    return P.cbm_numpy(s, *th[:7], delta=th[13] if len(th) > 13 else 0.0)


def vbm_form(s, th, eps=1e-8):
    v0, av, av2, b, b2, d = th[7:13]
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz
    hydro = av * i1 + av2 * i1**2
    dxx, dyy, dzz = xx - i1 / 3.0, yy - i1 / 3.0, zz - i1 / 3.0
    H = np.zeros((len(s), 3, 3))
    H[:, 0, 0] = hydro + (b + b2 * dxx) * dxx
    H[:, 1, 1] = hydro + (b + b2 * dyy) * dyy
    H[:, 2, 2] = hydro + (b + b2 * dzz) * dzz
    H[:, 0, 1] = H[:, 1, 0] = d * xy
    H[:, 1, 2] = H[:, 2, 1] = d * yz
    H[:, 0, 2] = H[:, 2, 0] = d * zx
    H += eps * np.eye(3)[None]
    return v0 + np.linalg.eigvalsh(H)[:, 2]


def gap_form(s, th):
    return cbm_form(s, th) - vbm_form(s, th)


def _fit(s, c, v, lam, x0, w_gap=1.0):
    n = len(s)
    sc = np.array([max(abs(x0[i]), 1.0) for i in HIGH_IDX])

    def residuals(th):
        cp, vp = cbm_form(s, th), vbm_form(s, th)
        r = [cp - c, vp - v, w_gap * ((cp - vp) - (c - v))]
        if lam > 0:
            r.append(np.sqrt(lam * n) * th[HIGH_IDX] / sc)
        return np.concatenate(r)

    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    if len(x0) > 13:
        lo[13], hi[13] = 0.05, 10.0
    return least_squares(residuals, x0, method="trf", bounds=(lo, hi), max_nfev=4000)


def fit_constants(
    s,
    c,
    v,
    lams=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    d_starts=(-4.0, -1.0, 1.0, 4.0),
    seed=0,
    two_band=False,
    delta_starts=(0.5, 1.0, 1.7, 3.0),
):
    """Fit on the strain, CBM and VBM of a set of states; returns {name: value} (13 constants, 14 with two_band)."""
    base = np.array([float(getattr(P, k)) for k in KEYS])
    best = None
    for d0 in d_starts:  # starting points for D_BP, without penalty
        x0 = base.copy()
        x0[12] = d0
        r = _fit(s, c, v, 0.0, x0)
        if best is None or r.cost < best.cost:
            best = r
    keys = KEYS
    if two_band:  # start from the 13-constant fit; starting points for DELTA_X
        keys, lin, best = KEYS2, best.x, None
        for g0 in delta_starts:
            r = _fit(s, c, v, 0.0, np.append(lin, g0))
            if best is None or r.cost < best.cost:
                best = r
    x_start = best.x

    # ridge strength: lowest gap MAE on the held-out folds of a five-fold cross-validation
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(s)), 5)
    cv = {}
    for lam in lams:
        err = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            th = _fit(s[tr], c[tr], v[tr], lam, x_start).x
            err.append(np.abs(gap_form(s[te], th) - (c[te] - v[te])).mean())
        cv[lam] = float(np.mean(err))
    th = _fit(s, c, v, min(cv, key=cv.get), x_start).x
    return {k: float(x) for k, x in zip(keys, th)}
