# ============================================================
# physics.py
#
# The analytic models of the band edges under strain, shared by the trainer,
# the calibration (calib_v2.py) and the analysis.
#
#   CBM(E): six <100> conduction valleys, each shifted by an extended
#     deformation-potential expansion (hydrostatic XI_D, uniaxial XI_U,
#     quadratic XI_Q, cross term XI_C, second invariant XI_Q2, shear
#     splitting 2*XI_UP); the band edge is their minimum. With beta given,
#     a log-sum-exp soft minimum makes it differentiable.
#   VBM(E): largest eigenvalue of an extended 3x3 Bir-Pikus matrix
#     (hydrostatic AV/AV2, deviatoric B_BP/B2_BP, shear D_BP).
#   Eg(E) = CBM(E) - VBM(E).
#
# UNITS: strain dimensionless (Green-Lagrange); energies in eV.
#
# The 13 constants below are default values only. Every run of the paper
# replaces them with the constants fitted on that run's own training portion
# (calibrate_*.py, results/98_constants_v2 and results/100_constants_labelmatched),
# passed through the environment variable PHYSICS_CONST_JSON.
# ============================================================

import torch

# --- CBM deformation-potential constants (eV), defaults ---
CBM_0 = 14.534980535287007
XI_D  = -24.84604763603144
XI_U  = 24.328031070892287
XI_UP = 12.126672421097085
XI_Q  = 12.984971351885598
XI_C  = -29.582212007237697
XI_Q2 = 18.18785190955848
# Separation of the two lowest conduction bands at the valley minimum (eV). 0 selects the linear shear term
# 2*XI_UP*|e_bc|; a positive value selects the two-band form at X (Hensel et al. 1965; Ungersboeck et al. 2007),
# quadratic in shear below |e_bc| = DELTA_X / (4 XI_UP) and linear above.
DELTA_X = 0.0

# --- VBM extended Bir-Pikus constants (eV), defaults ---
VBM_0  = 8.998824547791076
AV     = -14.238965719572148
AV2    = 13.090080116642534
B_BP   = -7.177161522873087
B2_BP  = 70.41960340594974
D_BP   = -1.9671238909825762e-07


# --- Per-run constants ---
# PHYSICS_CONST_JSON names a JSON file of {name: value}; its values replace the
# defaults above at import time. Every training run of the paper sets it.
import os as _os
_CONST_JSON = _os.environ.get("PHYSICS_CONST_JSON", "").strip()
if _CONST_JSON:
    import json as _json
    with open(_CONST_JSON) as _f:
        _ov = _json.load(_f)
    for _k, _v in _ov.items():
        if _k in globals() and isinstance(globals()[_k], float):
            globals()[_k] = float(_v)
        else:
            raise KeyError("PHYSICS_CONST_JSON: unknown constant " + str(_k))


def shear_shift(e, xi_up, delta, lib=torch):
    # Shift of the two valleys on one axis by the shear strain e in the plane normal to that axis. Both valleys
    # (+k and -k) shift alike. delta = 0: linear form -2|xi_up e|. delta > 0: two-band form with
    # eta = 4 xi_up e / delta: -(delta/4) eta^2 for |eta| < 1, -(delta/4)(2|eta| - 1) above.
    if delta <= 0:
        return -2.0 * lib.abs(xi_up * e)
    eta = 4.0 * xi_up * e / delta
    a = lib.abs(eta)
    return lib.where(a < 1.0, -0.25 * delta * eta ** 2, -0.25 * delta * (2.0 * a - 1.0))


def _valleys(valley, Exx, Eyy, Ezz, Exy, Eyz, Ezx, xi_up, lib, stack, delta=None):
    # The six valley energies. With DELTA_X = 0 the pairs are valley +/- 2 xi_up e_bc, as fitted originally
    # (their minimum is the linear form); otherwise each axis contributes two equal valleys.
    delta = DELTA_X if delta is None else delta
    if delta <= 0:
        return stack([valley(Ezz) + 2.0*xi_up*Exy, valley(Ezz) - 2.0*xi_up*Exy,
                      valley(Eyy) + 2.0*xi_up*Ezx, valley(Eyy) - 2.0*xi_up*Ezx,
                      valley(Exx) + 2.0*xi_up*Eyz, valley(Exx) - 2.0*xi_up*Eyz])
    z = valley(Ezz) + shear_shift(Exy, xi_up, delta, lib)
    y = valley(Eyy) + shear_shift(Ezx, xi_up, delta, lib)
    x = valley(Exx) + shear_shift(Eyz, xi_up, delta, lib)
    return stack([z, z, y, y, x, x])


def physics_cbm(strain, beta=None):
    # CBM via the 6-valley deformation-potential model.
    # beta=None  -> exact hard minimum (use for evaluation / the solver).
    # beta=float -> log-sum-exp soft minimum (differentiable; PINN training).
    Exx = strain[:, 0]; Eyy = strain[:, 1]; Ezz = strain[:, 2]
    Exy = strain[:, 3]; Eyz = strain[:, 4]; Ezx = strain[:, 5]
    I1  = Exx + Eyy + Ezz

    def valley(e_aa):
        return XI_D*I1 + XI_U*e_aa + XI_Q*e_aa**2 + XI_C*I1*e_aa + XI_Q2*I1**2

    deltas = _valleys(valley, Exx, Eyy, Ezz, Exy, Eyz, Ezx, XI_UP, torch, lambda v: torch.stack(v, dim=1))

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
# NUMPY VERSION OF THE CONDUCTION-BAND MODEL (used by calib_v2.py)
# ==============================================================================

import numpy as _np

CBM_KEYS = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2"]
VBM_KEYS = ["VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]


def cbm_numpy(s, cbm_0, xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2, delta=None, beta=None):
    # Arguments follow CBM_KEYS order. beta=None: exact minimum over the valleys; beta=float: the soft minimum of training.
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz

    def valley(e):
        return xi_d * i1 + xi_u * e + xi_q * e ** 2 + xi_c * i1 * e + xi_q2 * i1 ** 2

    d = _valleys(valley, xx, yy, zz, xy, yz, zx, xi_up, _np, lambda v: _np.stack(v, axis=1), delta)
    if beta is None:
        return cbm_0 + d.min(axis=1)
    m = d.min(axis=1)
    return cbm_0 + m - _np.log(_np.exp(-beta * (d - m[:, None])).sum(axis=1)) / beta
