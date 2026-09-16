"""Analytic model of how strain moves the band edges of diamond.

Two closed-form models, one per band edge, plus the gap they define:

    CBM(E)  six <100> conduction valleys from a deformation-potential
            expansion; the band edge is the lowest valley
    VBM(E)  largest eigenvalue of a 3x3 Bir-Pikus Hamiltonian
    Eg(E)   CBM(E) - VBM(E)

Both the trainer and the physics-only solver import this module, so they use
identical equations and constants.

Units: strain is dimensionless (Green-Lagrange), energies are in eV.
"""

import json
import os

import numpy as np
import torch
from scipy.optimize import least_squares

# The 13 constants below are fitted once to HSE06 reference data and then held
# fixed. Regenerate them with:  python fit_physics_constants.py --freeze
# The fit uses the training fold only, so the test fold stays unseen.

# --- CBM deformation-potential constants (eV) ---
CBM_0 = 14.534980535287007
XI_D  = -24.84604763603144
XI_U  = 24.328031070892287
XI_UP = 12.126672421097085
XI_Q  = 12.984971351885598
XI_C  = -29.582212007237697
XI_Q2 = 18.18785190955848

# --- VBM extended Bir-Pikus constants (eV) ---
VBM_0  = 8.998824547791076
AV     = -14.238965719572148
AV2    = 13.090080116642534
B_BP   = -7.177161522873087
B2_BP  = 70.41960340594974
D_BP   = -1.9671238909825762e-07


# --- optional per-run constant override ---
# Setting the environment variable PHYSICS_CONST_JSON to a JSON file of
# {name: value} replaces any subset of the constants above when this module is
# imported. run_matched_prior.py uses it to give each run a set fitted on only
# that run's labelled rows. With the variable unset the values above are used.

CONSTANT_NAMES = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2",
                  "VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]


def _load_constant_overrides():
    """Read the override file named by PHYSICS_CONST_JSON, or return nothing."""
    path = os.environ.get("PHYSICS_CONST_JSON", "").strip()
    if not path:
        return {}
    with open(path) as handle:
        overrides = json.load(handle)
    for name in overrides:
        if name not in CONSTANT_NAMES:
            raise KeyError("PHYSICS_CONST_JSON: unknown constant " + str(name))
    return overrides


# Applied one name at a time so an unknown key raises instead of being ignored.
for _name, _value in _load_constant_overrides().items():
    if _name == "CBM_0":
        CBM_0 = float(_value)
    elif _name == "XI_D":
        XI_D = float(_value)
    elif _name == "XI_U":
        XI_U = float(_value)
    elif _name == "XI_UP":
        XI_UP = float(_value)
    elif _name == "XI_Q":
        XI_Q = float(_value)
    elif _name == "XI_C":
        XI_C = float(_value)
    elif _name == "XI_Q2":
        XI_Q2 = float(_value)
    elif _name == "VBM_0":
        VBM_0 = float(_value)
    elif _name == "AV":
        AV = float(_value)
    elif _name == "AV2":
        AV2 = float(_value)
    elif _name == "B_BP":
        B_BP = float(_value)
    elif _name == "B2_BP":
        B2_BP = float(_value)
    elif _name == "D_BP":
        D_BP = float(_value)


def physics_cbm(strain, beta=None):
    # CBM via the 6-valley deformation-potential model.
    # beta=None  -> exact hard minimum (use for evaluation / the solver).
    # beta=float -> log-sum-exp soft minimum (differentiable; PINN training).
    Exx = strain[:, 0]; Eyy = strain[:, 1]; Ezz = strain[:, 2]
    Exy = strain[:, 3]; Eyz = strain[:, 4]; Ezx = strain[:, 5]
    I1  = Exx + Eyy + Ezz

    def valley(e_aa):
        return XI_D*I1 + XI_U*e_aa + XI_Q*e_aa**2 + XI_C*I1*e_aa + XI_Q2*I1**2

    # Each axis contributes two valleys split by shear coupling (XI_UP).
    deltas = torch.stack([
        valley(Ezz) + 2.0*XI_UP*Exy,
        valley(Ezz) - 2.0*XI_UP*Exy,
        valley(Eyy) + 2.0*XI_UP*Ezx,
        valley(Eyy) - 2.0*XI_UP*Ezx,
        valley(Exx) + 2.0*XI_UP*Eyz,
        valley(Exx) - 2.0*XI_UP*Eyz,
    ], dim=1)

    if beta is None:
        return CBM_0 + torch.min(deltas, dim=1).values
    return CBM_0 + (-torch.logsumexp(-beta * deltas, dim=1) / beta)


def physics_vbm(strain, eps=1e-8):
    # VBM = largest eigenvalue of the extended Bir-Pikus 3x3 Hamiltonian.
    # Quadratic terms (AV2, B2_BP) extend the standard linear model to the
    # larger strain amplitudes present in the triaxial subset.
    Exx = strain[:, 0]; Eyy = strain[:, 1]; Ezz = strain[:, 2]
    Exy = strain[:, 3]; Eyz = strain[:, 4]; Ezx = strain[:, 5]
    I1     = Exx + Eyy + Ezz
    hydro  = AV*I1 + AV2*I1**2
    dxx    = Exx - I1/3.0
    dyy    = Eyy - I1/3.0
    dzz    = Ezz - I1/3.0

    N = strain.shape[0]
    H = torch.zeros(N, 3, 3, dtype=strain.dtype, device=strain.device)
    H[:, 0, 0] = hydro + (B_BP + B2_BP*dxx)*dxx
    H[:, 1, 1] = hydro + (B_BP + B2_BP*dyy)*dyy
    H[:, 2, 2] = hydro + (B_BP + B2_BP*dzz)*dzz
    H[:, 0, 1] = D_BP*Exy;  H[:, 1, 0] = D_BP*Exy
    H[:, 1, 2] = D_BP*Eyz;  H[:, 2, 1] = D_BP*Eyz
    H[:, 0, 2] = D_BP*Ezx;  H[:, 2, 0] = D_BP*Ezx
    # Small diagonal jitter stabilises eigvalsh for near-degenerate cases.
    H += eps * torch.eye(3, dtype=strain.dtype, device=strain.device).unsqueeze(0)

    return VBM_0 + torch.linalg.eigvalsh(H)[:, 2]


def physics_eg(strain, beta=None):
    # Convenience: physics-only bandgap Eg = CBM - VBM (no network).
    return physics_cbm(strain, beta=beta) - physics_vbm(strain)


# ==============================================================================
# NUMPY MIRROR + FITTERS
#
# The functions above are torch, because training needs gradients through them.
# Fitting the constants is a scipy least-squares problem, so the same two
# equations are repeated here in numpy. They must stay in step: fit_physics_constants.py
# checks that re-fitting reproduces the frozen constants above, which would fail
# immediately if these drifted from the torch versions.
# ==============================================================================

CBM_KEYS = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2"]
VBM_KEYS = ["VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]


def cbm_numpy(s, cbm_0, xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2):
    # Arguments follow CBM_KEYS order, so a fit result splats straight back in:
    #     cbm_numpy(strain, *fit_cbm(strain, y))
    xx, yy, zz = s[:, 0], s[:, 1], s[:, 2]
    xy, yz, zx = s[:, 3], s[:, 4], s[:, 5]
    i1 = xx + yy + zz

    def valley(e):
        return xi_d * i1 + xi_u * e + xi_q * e ** 2 + xi_c * i1 * e + xi_q2 * i1 ** 2

    d = np.stack([valley(zz) + 2 * xi_up * xy, valley(zz) - 2 * xi_up * xy,
                   valley(yy) + 2 * xi_up * zx, valley(yy) - 2 * xi_up * zx,
                   valley(xx) + 2 * xi_up * yz, valley(xx) - 2 * xi_up * yz], axis=1)
    return cbm_0 + d.min(axis=1)


def vbm_numpy(s, vbm_0, a_v, a_v2, b, b2, d_bp):
    xx, yy, zz = s[:, 0], s[:, 1], s[:, 2]
    xy, yz, zx = s[:, 3], s[:, 4], s[:, 5]
    i1 = xx + yy + zz
    j2 = ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) / 6.0 \
        + xy ** 2 + yz ** 2 + zx ** 2
    out = np.empty(len(s))
    for k in range(len(s)):
        h = np.array([
            [-b * (2 * xx[k] - yy[k] - zz[k]) / 2.0, d_bp * xy[k], d_bp * zx[k]],
            [d_bp * xy[k], -b * (2 * yy[k] - zz[k] - xx[k]) / 2.0, d_bp * yz[k]],
            [d_bp * zx[k], d_bp * yz[k], -b * (2 * zz[k] - xx[k] - yy[k]) / 2.0]])
        out[k] = np.linalg.eigvalsh(h).max()
    return vbm_0 + a_v * i1 + a_v2 * i1 ** 2 + out + b2 * j2


def fit_cbm(strain, y):
    """Fit the 7 conduction-band constants to measured CBM energies.

    Returns them in CBM_KEYS order, so the result can be passed straight back
    into cbm_numpy.
    """

    def residual(params):
        return cbm_numpy(strain, *params) - y

    start = [14.5, -13.0, 7.95, 10.0, 13.0, -2.7, 11.0]
    lower = [10.0, -50.0, -10.0, -30.0, -200.0, -200.0, -300.0]
    upper = [20.0, 50.0, 30.0, 30.0, 200.0, 200.0, 300.0]
    fit = least_squares(residual, start, method="trf", bounds=(lower, upper),
                        max_nfev=100000)
    return fit.x


def fit_vbm(strain, y):
    """Fit the 6 valence-band constants to measured VBM energies.

    Fitted in two passes. The first holds the quadratic terms at zero and fits
    only the linear model, which is well conditioned. Its answer then starts the
    full fit, so the quadratic terms refine a sensible solution instead of
    searching from nothing.
    """

    def linear_residual(params):
        vbm_0, a_v, b, d_bp = params
        return vbm_numpy(strain, vbm_0, a_v, 0.0, b, 0.0, d_bp) - y

    def full_residual(params):
        return vbm_numpy(strain, *params) - y

    linear_start = [8.8, -15.0, -3.0, -16.8]
    linear_fit = least_squares(linear_residual, linear_start, method="trf",
                               max_nfev=100000).x

    # Re-order the linear answer into the full 6-parameter layout, with the two
    # quadratic terms (a_v2, b2) starting at zero.
    vbm_0, a_v, b, d_bp = linear_fit
    start = [vbm_0, a_v, 0.0, b, 0.0, d_bp]
    lower = [-20.0, -60.0, -500.0, -30.0, -500.0, -50.0]
    upper = [40.0, 60.0, 500.0, 30.0, 500.0, 50.0]
    fit = least_squares(full_residual, start, method="trf",
                        bounds=(lower, upper), max_nfev=100000)
    return fit.x
