"""Test B: would a better analytic model help multi-plane shear? Refit with extra second-order shear terms (no network)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root; run from there
os.makedirs(ROOT + "/results/shear_tests", exist_ok=True)

import sys, json, importlib.util, numpy as np, pandas as pd
from scipy.optimize import least_squares
M = ROOT; REPO = M + ""
sys.path.insert(0, REPO); os.environ["PHYSICS_CONST_JSON"] = REPO + "/results/98_constants_v2/symmetry/const_seed42.json"
import calib_v2 as CV, physics as P
S = pd.read_csv(M + "/results/shear_tests/shear_difficulty_states.csv")
sys.argv = ["03_train_pinn.py", "--data_csv", REPO + "/data/dft_hy_v9.csv", "--results_dir", "/tmp/_fstub", "--device", "cpu", "--split", "symmetry"]
spec = importlib.util.spec_from_file_location("trn", REPO + "/03_train_pinn.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

def cbm_ext(s, th, ex):
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]; i1 = xx + yy + zz
    xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2 = th[1], th[2], th[3], th[4], th[5], th[6]
    xs, xt, xm = ex[:3]
    def val(e, own, o1, o2, nrm):   # own: shear in the plane perpendicular to the valley axis; o1, o2: the other two shears
        return xi_d * i1 + xi_u * e + xi_q * e ** 2 + xi_c * i1 * e + xi_q2 * i1 ** 2 + xs * own ** 2 + xt * (o1 ** 2 + o2 ** 2), nrm
    d = []
    for e, own, o1, o2, nrm in [(zz, xy, yz, zx, (xx + yy - 2 * zz)), (yy, zx, xy, yz, (xx + zz - 2 * yy)), (xx, yz, zx, xy, (yy + zz - 2 * xx))]:
        base, _ = val(e, own, o1, o2, nrm); d += [base + 2 * xi_up * own + xm * own * nrm, base - 2 * xi_up * own - xm * own * nrm]
    return th[0] + np.min(np.stack(d, 1), axis=1)

def vbm_ext(s, th, ex):
    v0, av, av2, b, b2, d = th[7:13]; vs = ex[3]
    xx, yy, zz, xy, yz, zx = [s[:, i] for i in range(6)]
    i1 = xx + yy + zz; hydro = av * i1 + av2 * i1 ** 2; dxx, dyy, dzz = xx - i1 / 3, yy - i1 / 3, zz - i1 / 3
    H = np.zeros((len(s), 3, 3))
    H[:, 0, 0] = hydro + (b + b2 * dxx) * dxx + vs * (xy ** 2 + zx ** 2); H[:, 1, 1] = hydro + (b + b2 * dyy) * dyy + vs * (xy ** 2 + yz ** 2); H[:, 2, 2] = hydro + (b + b2 * dzz) * dzz + vs * (yz ** 2 + zx ** 2)
    H[:, 0, 1] = H[:, 1, 0] = d * xy; H[:, 1, 2] = H[:, 2, 1] = d * yz; H[:, 0, 2] = H[:, 2, 0] = d * zx
    return v0 + np.linalg.eigvalsh(H + 1e-8 * np.eye(3)[None])[:, 2]

rows = []
for seed in range(42, 59):
    d = T.load_and_split(REPO + "/data/dft_hy_v9.csv", seed, 100); s = d["strain_tr"].astype(float); c = d["cbm_tr"].astype(float); v = d["vbm_tr"].astype(float)
    th0 = json.load(open(REPO + "/results/98_constants_v2/symmetry/const_seed%d.json" % seed)); th0 = np.array([th0[k] for k in CV.KEYS])
    def res(x):
        th, ex = x[:13], x[13:]; cp, vp = cbm_ext(s, th, ex), vbm_ext(s, th, ex)
        return np.concatenate([cp - c, vp - v, (cp - vp) - (c - v)])
    fit = least_squares(res, np.r_[th0, np.zeros(4)], method="trf", max_nfev=3000); th, ex = fit.x[:13], fit.x[13:]
    te = np.array(d["idx_te"]); st = d["X_te"].astype(float) if False else S[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values[te]; y = S.Eg_eV.values[te]
    sub = S.iloc[te]; base = np.abs(CV.gap_form(st, th0) - y) * 1000; ext = np.abs(cbm_ext(st, th, ex) - vbm_ext(st, th, ex) - y) * 1000
    for name, m in [("all", np.ones(len(te), bool)), ("shear 1 plane", ((sub.fam == "shear") & (sub.n_shear_planes == 1)).values), ("shear 2 planes", ((sub.fam == "shear") & (sub.n_shear_planes == 2)).values),
                    ("shear 3 planes", ((sub.fam == "shear") & (sub.n_shear_planes == 3)).values), ("other families", (sub.fam != "shear").values)]:
        rows.append(dict(seed=seed, group=name, current=base[m].mean(), extended=ext[m].mean()))
D = pd.DataFrame(rows); print("Analytic models alone, gap MAE on test states (meV), mean of 17 refits")
print(D.groupby("group")[["current", "extended"]].mean().round(1).loc[["all", "other families", "shear 1 plane", "shear 2 planes", "shear 3 planes"]].to_string())
