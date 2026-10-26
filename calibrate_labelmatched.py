#!/usr/bin/env python3
"""Label-matched constants: fitted with calib_v2 only on what a run at a reduced labeled fraction may use, namely its
labeled states plus the training states under isotropic strain (whose reference gaps every run sees through the
isotropic term). Symmetry-grouped division, seeds 42-58, labeled fractions 1, 2, 5 and 10%.
Writes results/100_constants_labelmatched/<pct>pct/const_seed<s>.json.
Usage: python calibrate_labelmatched.py"""
import os, sys, json, importlib.util, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, calib_v2 as CV
sys.argv = ["03_train_pinn.py", "--data_csv", C.DATA_CSV, "--results_dir", os.path.join(C.RESULTS, "_import_stub"), "--device", "cpu", "--split", "symmetry"]
spec = importlib.util.spec_from_file_location("tr", os.path.join(C.ROOT, "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
for pct in [1, 2, 5, 10]:
    out = os.path.join(C.RESULTS, "100_constants_labelmatched", "%dpct" % pct); os.makedirs(out, exist_ok=True)
    for seed in C.SEEDS:
        d = T.load_and_split(C.DATA_CSV, seed, pct); st = d["strain_tr"].astype(float)
        iso = (np.abs(st[:, 3:]).sum(1) < 1e-4) & (np.abs(st[:, :3] - st[:, :3].mean(1, keepdims=True)).max(1) < 1e-4)
        use = np.zeros(len(st), bool); use[d["dft_idx"]] = True; use |= iso
        const = CV.fit_constants(st[use], d["cbm_tr"][use].astype(float), d["vbm_tr"][use].astype(float), seed=seed)
        json.dump(const, open(os.path.join(out, "const_seed%d.json" % seed), "w"), indent=1)
        print("%3d%% seed %d: fitted on %d states (%d labeled + %d isotropic) | XI_D %.2f XI_U %.2f AV %.2f D %.2f" % (pct, seed, use.sum(), len(d["dft_idx"]), (iso & ~np.isin(np.arange(len(st)), d["dft_idx"])).sum(), const["XI_D"], const["XI_U"], const["AV"], const["D_BP"]), flush=True)
