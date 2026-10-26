#!/usr/bin/env python3
"""Conventional regression baselines: Gaussian process regression, random forest and histogram gradient boosting.

They use exactly the networks' symmetry-grouped training and test portions and labeled states (taken from
03_train_pinn.load_and_split), 17 seeds x the eight nonzero labeled fractions, predict the gap directly from the
standardized strain and see only the labeled states.
Writes results/classical/runs.csv.   Usage: python classical_baselines.py"""
import os, sys, importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C




def family_of(label):
    s = str(label).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"


def build_models(seed, n_labeled):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1)))
    models = {
        "GaussianProcess": GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=seed,
            n_restarts_optimizer=1),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, random_state=seed, n_jobs=2),
        "GBDT": HistGradientBoostingRegressor(
            random_state=seed, max_iter=400, learning_rate=0.08),
    }
    # A tree needs at least a couple of points to split on; a GP does not.
    if n_labeled < 2:
        models.pop("RandomForest", None)
        models.pop("GBDT", None)
    return models



def run_seed(seed):
    sys.argv = ["03_train_pinn.py", "--data_csv", C.DATA_CSV, "--results_dir", os.path.join(C.RESULTS, "_import_stub"), "--device", "cpu",
                "--split", "symmetry"]
    spec = importlib.util.spec_from_file_location("tr%d" % seed, os.path.join(C.ROOT, "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    rows = []
    for pct in [p for p in C.FRACTIONS if p > 0]:
        d = T.load_and_split(C.DATA_CSV, seed, pct)
        Xl, yl = d["X_tr"][d["dft_idx"]].astype(np.float64), d["eg_tr"][d["dft_idx"]].astype(np.float64)
        Xte, yte = d["X_te"].astype(np.float64), d["eg_te"].astype(np.float64); fam = np.array([family_of(s) for s in d["stype_te"]])
        for name, model in build_models(seed, len(yl)).items():
            try:
                model.fit(Xl, yl); pred = model.predict(Xte)
            except Exception as exc:
                print("  ! %s seed %d pct %d: %s" % (name, seed, pct, exc)); continue
            a = np.abs(pred - yte); r = dict(model=name, seed=seed, dft_pct=pct, n_labeled=len(yl), test_mae_meV=1000 * a.mean(),
                                             test_r2=1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
            for f in C.STRAIN_FAMILIES:
                m = fam == f; r["mae_%s_meV" % f] = 1000 * a[m].mean() if m.any() else np.nan
            rows.append(r)
    print("seed %d done" % seed, flush=True)
    return rows

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(8) as ex:
        rows = sum(ex.map(run_seed, C.SEEDS), [])
    out = os.path.join(C.RESULTS, "classical"); os.makedirs(out, exist_ok=True)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(out, "runs.csv"), index=False)
    print(d.pivot_table(index="dft_pct", columns="model", values="test_mae_meV", aggfunc="mean").round(1).to_string())
