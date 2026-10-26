#!/usr/bin/env python3
"""Constants for the held-out tests: each deformation family left out in turn (isotropic, uniaxial, biaxial, triaxial,
shear) and three strain cut-offs (training on max|strain| <= 0.03, 0.05 or 0.07), 17 seeds each, fitted with calib_v2 on
each run's own reduced training portion; the held-out states never enter. Existing files are kept.
Shear left out: no training state has an off-diagonal strain, so the shear constants XI_UP and D_BP are not determined
by the data; they are set to 0.
Writes results/98_constants_v2/<scheme>/const_seed<s>.json and results/98_constants_v2/<scheme>/fit_summary.csv.
Usage: python calibrate_heldout.py <scheme>   (one scheme per process; schemes listed below)"""
import os, sys, json, importlib.util, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, calib_v2 as CV
SCHEMES = {"group_isotropic": ["--split", "group", "--holdout_family", "isotropic"],
           "group_uniaxial": ["--split", "group", "--holdout_family", "uniaxial"],
           "group_biaxial": ["--split", "group", "--holdout_family", "biaxial"],
           "group_triaxial": ["--split", "group", "--holdout_family", "triaxial"],
           "group_shear": ["--split", "group", "--holdout_family", "shear"],
           "magnitude_03": ["--split", "magnitude", "--magnitude_thresh", "0.03"],
           "magnitude_05": ["--split", "magnitude", "--magnitude_thresh", "0.05"],
           "magnitude_07": ["--split", "magnitude", "--magnitude_thresh", "0.07"]}
name = sys.argv[1]; flags = SCHEMES[name]
df = pd.read_csv(C.DATA_CSV); E = df[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values.astype(float)
out = os.path.join(C.RESULTS, "98_constants_v2", name); os.makedirs(out, exist_ok=True)
sys.argv = ["03_train_pinn.py", "--data_csv", C.DATA_CSV, "--results_dir", os.path.join(C.RESULTS, "_import_stub"), "--device", "cpu"] + flags
spec = importlib.util.spec_from_file_location("tr_" + name, os.path.join(C.ROOT, "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
rows = []
for seed in C.SEEDS:
    path = os.path.join(out, "const_seed%d.json" % seed)
    d = T.load_and_split(C.DATA_CSV, seed, 100); s = d["strain_tr"].astype(float)
    if os.path.exists(path):
        new = json.load(open(path)); status = "kept"
    else:
        new = CV.fit_constants(s, d["cbm_tr"].astype(float), d["vbm_tr"].astype(float), seed=seed)
        if name == "group_shear":
            assert np.abs(s[:, 3:]).max() < 1e-8, "shear left out but training has off-diagonal strain %g" % np.abs(s[:, 3:]).max()
            new["XI_UP"] = 0.0; new["D_BP"] = 0.0
        json.dump(new, open(path, "w"), indent=1); status = "fitted"
    te = np.array(d["idx_te"]); th = np.array([new[k] for k in CV.KEYS])
    r = dict(scheme=name, seed=seed, status=status, n_train=len(s), n_test=len(te),
             formula_gap_MAE_meV=1000 * np.abs(CV.gap_form(E[te], th) - df.Eg_eV.values[te]).mean())
    rows.append(r); print("%-16s seed %d %-6s train %4d test %4d | formula test gap MAE %7.1f meV" % (name, seed, status, r["n_train"], r["n_test"], r["formula_gap_MAE_meV"]), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "fit_summary.csv"), index=False)
