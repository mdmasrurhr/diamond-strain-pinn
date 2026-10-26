"""Calibration of the 13 constants of the analytic models (used by calibrate_*.py).
1. The valence band is fitted in exactly the form that physics.physics_vbm evaluates in training: the largest eigenvalue
   of the 3x3 matrix with (B + B2*dev)*dev on the diagonal and D*e_ab off the diagonal.
2. Joint least squares over all 13 constants on the CBM, VBM and gap residuals together, a ridge penalty on the five
   higher-order constants (strength chosen by 5-fold cross-validation inside the training portion), and several starting
   points for the shear coupling D."""
import numpy as np
from scipy.optimize import least_squares
import physics as P

KEYS = P.CBM_KEYS + P.VBM_KEYS            # 7 + 6
HIGH = ["XI_Q", "XI_C", "XI_Q2", "AV2", "B2_BP"]
HIGH_IDX = [KEYS.index(k) for k in HIGH]


def cbm_form(s, th):
    return P.cbm_numpy(s, *th[:7])


def vbm_form(s, th, eps=1e-8):
    v0, av, av2, b, b2, d = th[7:13]
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz; hydro = av * i1 + av2 * i1 ** 2
    dxx, dyy, dzz = xx - i1 / 3.0, yy - i1 / 3.0, zz - i1 / 3.0
    H = np.zeros((len(s), 3, 3))
    H[:, 0, 0] = hydro + (b + b2 * dxx) * dxx; H[:, 1, 1] = hydro + (b + b2 * dyy) * dyy; H[:, 2, 2] = hydro + (b + b2 * dzz) * dzz
    H[:, 0, 1] = H[:, 1, 0] = d * xy; H[:, 1, 2] = H[:, 2, 1] = d * yz; H[:, 0, 2] = H[:, 2, 0] = d * zx
    H += eps * np.eye(3)[None]
    return v0 + np.linalg.eigvalsh(H)[:, 2]


def gap_form(s, th):
    return cbm_form(s, th) - vbm_form(s, th)


def _fit(s, c, v, lam, x0, w_gap=1.0):
    n = len(s); sc = np.array([max(abs(x0[i]), 1.0) for i in HIGH_IDX])
    def res(th):
        cp, vp = cbm_form(s, th), vbm_form(s, th)
        r = [cp - c, vp - v, w_gap * ((cp - vp) - (c - v))]
        if lam > 0: r.append(np.sqrt(lam * n) * th[HIGH_IDX] / sc)
        return np.concatenate(r)
    return least_squares(res, x0, method="trf", max_nfev=4000)


def fit_constants(s, c, v, lams=(0.0, 1e-4, 1e-3, 1e-2, 1e-1), d_starts=(-4.0, -1.0, 1.0, 4.0), seed=0, return_info=False):
    """Fit on (strain, CBM, VBM) of a training portion. Returns dict of 13 constants."""
    base = np.array([float(getattr(P, k)) for k in KEYS])
    # multi-start on D with no penalty, to find the basin
    best = None
    for d0 in d_starts:
        x0 = base.copy(); x0[12] = d0
        r = _fit(s, c, v, 0.0, x0)
        if best is None or r.cost < best.cost: best = r
    x_start = best.x
    # ridge strength by 5-fold cross-validation (gap MAE on the held-out fold)
    rng = np.random.default_rng(seed); folds = np.array_split(rng.permutation(len(s)), 5); cv = {}
    for lam in lams:
        err = []
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            th = _fit(s[tr], c[tr], v[tr], lam, x_start).x
            err.append(np.abs(gap_form(s[te], th) - (c[te] - v[te])).mean())
        cv[lam] = float(np.mean(err))
    lam_best = min(cv, key=cv.get)
    th = _fit(s, c, v, lam_best, x_start).x
    out = {k: float(x) for k, x in zip(KEYS, th)}
    return (out, dict(lambda_=lam_best, cv=cv)) if return_info else out
