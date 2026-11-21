"""Analytic models of the band edges of diamond under strain (energies in eV, Green-Lagrange strain).

CBM: six <100> conduction valleys, each shifted by an extended deformation-potential expansion (XI_D, XI_U, XI_Q, XI_C,
     XI_Q2) and by shear (XI_UP, two-band form when DELTA_X > 0); the band edge is their minimum, or a log-sum-exp soft
     minimum when beta is given.
VBM: largest eigenvalue of an extended 3x3 Bir-Pikus matrix (AV, AV2, B_BP, B2_BP, D_BP).

The constants below are defaults. Every run replaces them with the constants fitted on its own training portion
(constants/), named by the environment variable PHYSICS_CONST_JSON.
"""

import json
import os

import numpy as np
import torch

# conduction band
CBM_0 = 14.534980535287007
XI_D = -24.84604763603144
XI_U = 24.328031070892287
XI_UP = 12.126672421097085
XI_Q = 12.984971351885598
XI_C = -29.582212007237697
XI_Q2 = 18.18785190955848
# Separation of the two lowest conduction bands at X (Hensel et al. 1965; Ungersboeck et al. 2007). 0: linear shear term
# 2 XI_UP |e|; > 0: two-band form, quadratic in shear below |e| = DELTA_X / (4 XI_UP) and linear above.
DELTA_X = 0.0

# valence band
VBM_0 = 8.998824547791076
AV = -14.238965719572148
AV2 = 13.090080116642534
B_BP = -7.177161522873087
B2_BP = 70.41960340594974
D_BP = -1.9671238909825762e-07

CBM_KEYS = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2"]
VBM_KEYS = ["VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]

_CONST_JSON = os.environ.get("PHYSICS_CONST_JSON", "").strip()
if _CONST_JSON:
    with open(_CONST_JSON) as _f:
        for _k, _v in json.load(_f).items():
            if _k in globals() and isinstance(globals()[_k], float):
                globals()[_k] = float(_v)
            else:
                raise KeyError("PHYSICS_CONST_JSON: unknown constant " + str(_k))


def shear_shift(e, xi_up, delta, lib=torch):
    """Shift of both valleys on one axis by the shear e in the plane normal to that axis.

    delta = 0: -2 |xi_up e|. delta > 0, eta = 4 xi_up e / delta: -(delta/4) eta^2 for |eta| < 1, else -(delta/4)(2|eta| - 1).
    """
    if delta <= 0:
        return -2.0 * lib.abs(xi_up * e)
    eta = 4.0 * xi_up * e / delta
    a = lib.abs(eta)
    return lib.where(a < 1.0, -0.25 * delta * eta**2, -0.25 * delta * (2.0 * a - 1.0))


def _valleys(valley, Exx, Eyy, Ezz, Exy, Eyz, Ezx, xi_up, lib, stack, delta=None):
    """The six valley energies (with delta = 0 the pairs are valley +/- 2 xi_up e, whose minimum is the linear form)."""
    delta = DELTA_X if delta is None else delta
    if delta <= 0:
        return stack(
            [
                valley(Ezz) + 2.0 * xi_up * Exy,
                valley(Ezz) - 2.0 * xi_up * Exy,
                valley(Eyy) + 2.0 * xi_up * Ezx,
                valley(Eyy) - 2.0 * xi_up * Ezx,
                valley(Exx) + 2.0 * xi_up * Eyz,
                valley(Exx) - 2.0 * xi_up * Eyz,
            ]
        )
    z = valley(Ezz) + shear_shift(Exy, xi_up, delta, lib)
    y = valley(Eyy) + shear_shift(Ezx, xi_up, delta, lib)
    x = valley(Exx) + shear_shift(Eyz, xi_up, delta, lib)
    return stack([z, z, y, y, x, x])


def physics_cbm(strain, beta=None):
    """Conduction-band minimum; beta=None gives the exact minimum, a float the soft minimum used in training."""
    Exx = strain[:, 0]
    Eyy = strain[:, 1]
    Ezz = strain[:, 2]
    Exy = strain[:, 3]
    Eyz = strain[:, 4]
    Ezx = strain[:, 5]
    I1 = Exx + Eyy + Ezz

    def valley(e_aa):
        return XI_D * I1 + XI_U * e_aa + XI_Q * e_aa**2 + XI_C * I1 * e_aa + XI_Q2 * I1**2

    deltas = _valleys(valley, Exx, Eyy, Ezz, Exy, Eyz, Ezx, XI_UP, torch, lambda v: torch.stack(v, dim=1))
    if beta is None:
        return CBM_0 + torch.min(deltas, dim=1).values
    return CBM_0 + (-torch.logsumexp(-beta * deltas, dim=1) / beta)


def physics_vbm(strain, eps=1e-8):
    """Valence-band maximum: largest eigenvalue of the extended Bir-Pikus matrix (eps: small diagonal shift)."""
    Exx = strain[:, 0]
    Eyy = strain[:, 1]
    Ezz = strain[:, 2]
    Exy = strain[:, 3]
    Eyz = strain[:, 4]
    Ezx = strain[:, 5]
    I1 = Exx + Eyy + Ezz
    hydro = AV * I1 + AV2 * I1**2
    dxx = Exx - I1 / 3.0
    dyy = Eyy - I1 / 3.0
    dzz = Ezz - I1 / 3.0

    H = torch.zeros(strain.shape[0], 3, 3, dtype=strain.dtype, device=strain.device)
    H[:, 0, 0] = hydro + (B_BP + B2_BP * dxx) * dxx
    H[:, 1, 1] = hydro + (B_BP + B2_BP * dyy) * dyy
    H[:, 2, 2] = hydro + (B_BP + B2_BP * dzz) * dzz
    H[:, 0, 1] = D_BP * Exy
    H[:, 1, 0] = D_BP * Exy
    H[:, 1, 2] = D_BP * Eyz
    H[:, 2, 1] = D_BP * Eyz
    H[:, 0, 2] = D_BP * Ezx
    H[:, 2, 0] = D_BP * Ezx
    H += eps * torch.eye(3, dtype=strain.dtype, device=strain.device).unsqueeze(0)
    return VBM_0 + torch.linalg.eigvalsh(H)[:, 2]


def cbm_numpy(s, cbm_0, xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2, delta=None, beta=None):
    """NumPy version of physics_cbm with the constants as arguments (CBM_KEYS order), for the calibration."""
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz

    def valley(e):
        return xi_d * i1 + xi_u * e + xi_q * e**2 + xi_c * i1 * e + xi_q2 * i1**2

    d = _valleys(valley, xx, yy, zz, xy, yz, zx, xi_up, np, lambda v: np.stack(v, axis=1), delta)
    if beta is None:
        return cbm_0 + d.min(axis=1)
    m = d.min(axis=1)
    return cbm_0 + m - np.log(np.exp(-beta * (d - m[:, None])).sum(axis=1)) / beta
