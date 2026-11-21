"""Would a richer analytic model help multi-plane shear? (Supplementary Note 8)

For each of the 17 runs, the analytic models are refitted on the training portion with three extra second-order shear
constants (shear in the other two planes and shear times the deviatoric normal strain for each valley, shear squared
for the valence band), and compared on the test states with the calibrated two-band and one-band constants.
Writes training/results/shear_tests/formula_test.csv (summary/shear_formula_test.csv holds the paper's copy).
Usage: python analysis/shear_formula_test.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))
import calibration as CV
import config as C
import physics as P

STRAIN = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]


def cbm_extended(s, th, ex):
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz
    xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2 = th[1], th[2], th[3], th[4], th[5], th[6]
    xt, xm = ex[:2]
    xs = 0.0

    def valley(e, own, o1, o2):  # own: shear normal to the valley axis; o1, o2: the other two shears
        return xi_d * i1 + xi_u * e + xi_q * e**2 + xi_c * i1 * e + xi_q2 * i1**2 + xs * own**2 + xt * (o1**2 + o2**2)

    d = []
    for e, own, o1, o2, nrm in [
        (zz, xy, yz, zx, xx + yy - 2 * zz),
        (yy, zx, xy, yz, xx + zz - 2 * yy),
        (xx, yz, zx, xy, yy + zz - 2 * xx),
    ]:
        base = valley(e, own, o1, o2)
        sh = P.shear_shift(own, xi_up, th[13], np)
        d += [base + sh + xm * own * nrm, base + sh - xm * own * nrm]
    return th[0] + np.min(np.stack(d, 1), axis=1)


def vbm_extended(s, th, ex):
    v0, av, av2, b, b2, d = th[7:13]
    vs = ex[2]
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz
    hydro = av * i1 + av2 * i1**2
    dxx, dyy, dzz = xx - i1 / 3, yy - i1 / 3, zz - i1 / 3
    H = np.zeros((len(s), 3, 3))
    H[:, 0, 0] = hydro + (b + b2 * dxx) * dxx + vs * (xy**2 + zx**2)
    H[:, 1, 1] = hydro + (b + b2 * dyy) * dyy + vs * (xy**2 + yz**2)
    H[:, 2, 2] = hydro + (b + b2 * dzz) * dzz + vs * (yz**2 + zx**2)
    H[:, 0, 1] = H[:, 1, 0] = d * xy
    H[:, 1, 2] = H[:, 2, 1] = d * yz
    H[:, 0, 2] = H[:, 2, 0] = d * zx
    return v0 + np.linalg.eigvalsh(H + 1e-8 * np.eye(3)[None])[:, 2]


def load_constants(band, seed, keys):
    with open(os.path.join(C.CONSTANTS, band, "symmetry", "const_seed%d.json" % seed)) as f:
        const = json.load(f)
    return np.array([const[k] for k in keys])


trainer = C.import_trainer("--split", "symmetry")
data = pd.read_csv(C.DATA_CSV)
family = data.strain_type.map(lambda x: "isotropic" if str(x) in ("unstrained", "isotropic") else str(x).split("_")[0])
rows = []
for seed in C.SEEDS:
    d = trainer.load_and_split(C.DATA_CSV, seed, 100)
    s, c, v = d["strain_tr"].astype(float), d["cbm_tr"].astype(float), d["vbm_tr"].astype(float)
    th0 = load_constants("two_band", seed, CV.KEYS2)

    def residuals(x):
        th, ex = x[:14], x[14:]
        cp, vp = cbm_extended(s, th, ex), vbm_extended(s, th, ex)
        return np.concatenate([cp - c, vp - v, (cp - vp) - (c - v)])

    lo = np.r_[np.full(13, -np.inf), 0.05, np.full(3, -np.inf)]
    hi = np.r_[np.full(13, np.inf), 10.0, np.full(3, np.inf)]
    fit = least_squares(residuals, np.r_[th0, np.zeros(3)], bounds=(lo, hi), method="trf", max_nfev=3000)
    th, ex = fit.x[:14], fit.x[14:]
    # the same fit without the three extra constants, so that both forms are fitted alike
    fit0 = least_squares(
        lambda x: residuals(np.r_[x, np.zeros(3)]), th0, bounds=(lo[:14], hi[:14]), method="trf", max_nfev=3000
    )

    te = np.array(d["idx_te"])
    st, y = data[STRAIN].values[te], data.Eg_eV.values[te]
    fam, planes = family.values[te], data.n_shear_planes.values[te]
    errors = dict(
        linear=np.abs(CV.gap_form(st, load_constants("one_band", seed, CV.KEYS)) - y) * 1000,
        calibrated=np.abs(CV.gap_form(st, th0) - y) * 1000,
        refit=np.abs(cbm_extended(st, fit0.x, np.zeros(3)) - vbm_extended(st, fit0.x, np.zeros(3)) - y) * 1000,
        extended=np.abs(cbm_extended(st, th, ex) - vbm_extended(st, th, ex) - y) * 1000,
    )
    groups = {
        "all": np.ones(len(te), bool),
        "shear 1 plane": (fam == "shear") & (planes == 1),
        "shear 2 planes": (fam == "shear") & (planes == 2),
        "shear 3 planes": (fam == "shear") & (planes == 3),
        "other families": fam != "shear",
    }
    for name, m in groups.items():
        rows.append(dict(seed=seed, group=name, **{k: e[m].mean() for k, e in errors.items()}))

table = pd.DataFrame(rows)
order = ["all", "other families", "shear 1 plane", "shear 2 planes", "shear 3 planes"]
print("Analytic models alone, gap MAE on the test states (meV), mean of 17 runs")
print(table.groupby("group")[["linear", "calibrated", "refit", "extended"]].mean().round(1).loc[order].to_string())
out = os.path.join(C.RESULTS, "shear_tests")
os.makedirs(out, exist_ok=True)
table.to_csv(os.path.join(out, "formula_test.csv"), index=False)
