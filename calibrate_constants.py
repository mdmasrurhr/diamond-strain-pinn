#!/usr/bin/env python3
"""Fit the 13 constants of the analytic models for every run, on that run's training portion only.

Symmetry-grouped division, seeds 42-58. The test and validation states of a run never enter its fit. The fit itself is
in calib_v2.py (joint nonlinear least squares on both band edges and the gap, with a ridge penalty on the second-order
constants chosen by five-fold cross-validation inside the training portion).
Writes results/98_constants_v2/symmetry/const_seed<s>.json and results/98_constants_v2/symmetry/fit_summary.csv
(the analytic models alone on each run's test portion).
Constants for the held-out tests: calibrate_heldout.py. Label-matched constants: calibrate_labelmatched.py.
With --two_band the conduction-band shear term takes its two-band form and DELTA_X is fitted as a 14th constant;
the files go to results/98_constants_2band/symmetry.
With --softmin the conduction-band minimum is fitted in the soft-minimum form that training uses (beta = config.SOFTMIN_BETA);
the files go to <folder>_softmin/symmetry. With --seeds a-b other seeds are fitted (fit summary fit_summary_seeds<a>-<b>.csv).
Usage: python calibrate_constants.py [--two_band] [--softmin] [--seeds 59-75]"""
import os, sys, json, importlib.util, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, calib_v2 as CV
TWO = "--two_band" in sys.argv
SOFT = "--softmin" in sys.argv
SEEDS, SUMMARY = C.SEEDS, "fit_summary.csv"
if "--seeds" in sys.argv:
    a, b = map(int, sys.argv[sys.argv.index("--seeds") + 1].split("-")); SEEDS, SUMMARY = list(range(a, b + 1)), "fit_summary_seeds%d-%d.csv" % (a, b)
if SOFT:
    CV.CBM_BETA = C.SOFTMIN_BETA
df = pd.read_csv(C.DATA_CSV); E = df[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values.astype(float)
out = os.path.join(C.RESULTS, ("98_constants_2band" if TWO else "98_constants_v2") + ("_softmin" if SOFT else ""), "symmetry"); os.makedirs(out, exist_ok=True)
sys.argv = ["03_train_pinn.py", "--data_csv", C.DATA_CSV, "--results_dir", os.path.join(C.RESULTS, "_import_stub"), "--device", "cpu",
            "--split", "symmetry"]
spec = importlib.util.spec_from_file_location("tr_symmetry", os.path.join(C.ROOT, "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
rows = []
for seed in SEEDS:
    d = T.load_and_split(C.DATA_CSV, seed, 100); s = d["strain_tr"].astype(float)
    const = CV.fit_constants(s, d["cbm_tr"].astype(float), d["vbm_tr"].astype(float), seed=seed, two_band=TWO)
    json.dump(const, open(os.path.join(out, "const_seed%d.json" % seed), "w"), indent=1)
    te = np.array(d["idx_te"]); th = np.array([const[k] for k in (CV.KEYS2 if TWO else CV.KEYS)])
    r = dict(seed=seed, n_train=len(s), n_test=len(te), formula_gap_MAE_meV=1000 * np.abs(CV.gap_form(E[te], th) - df.Eg_eV.values[te]).mean())
    rows.append(r); print("seed %d: fitted on %d training states | analytic models on %d test states: %.1f meV" % (seed, r["n_train"], r["n_test"], r["formula_gap_MAE_meV"]), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, SUMMARY), index=False)
