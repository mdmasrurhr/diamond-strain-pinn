"""Conventional regression: Gaussian process, random forest and gradient boosting.

Same symmetry-grouped divisions and labeled states as the networks (17 seeds x the eight nonzero labeled fractions); the
gap is predicted directly from the standardized strain, using the labeled states only.
Writes results/classical/runs.csv.    Usage: python classical_baselines.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import config as C


def build_models(seed, n_labeled):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-3, noise_level_bounds=(1e-8, 1e1)
    )
    models = {
        "GaussianProcess": GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=1
        ),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=2),
        "GBDT": HistGradientBoostingRegressor(random_state=seed, max_iter=400, learning_rate=0.08),
    }
    if n_labeled < 2:  # the trees need at least two states to split
        del models["RandomForest"], models["GBDT"]
    return models


def run_seed(seed):
    trainer = C.import_trainer("--split", "symmetry")
    rows = []
    for pct in [p for p in C.FRACTIONS if p > 0]:
        d = trainer.load_and_split(C.DATA_CSV, seed, pct)
        X, y = d["X_tr"][d["dft_idx"]].astype(np.float64), d["eg_tr"][d["dft_idx"]].astype(np.float64)
        X_te, y_te = d["X_te"].astype(np.float64), d["eg_te"].astype(np.float64)
        family = np.array([trainer.strain_family(s) for s in d["stype_te"]])
        for name, model in build_models(seed, len(y)).items():
            try:
                model.fit(X, y)
                pred = model.predict(X_te)
            except Exception as exc:
                print("  %s seed %d, %d%%: %s" % (name, seed, pct, exc))
                continue
            err = np.abs(pred - y_te)
            row = dict(
                model=name,
                seed=seed,
                dft_pct=pct,
                n_labeled=len(y),
                test_mae_meV=1000 * err.mean(),
                test_r2=1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum(),
            )
            for f in C.STRAIN_FAMILIES:
                m = family == f
                row["mae_%s_meV" % f] = 1000 * err[m].mean() if m.any() else np.nan
            rows.append(row)
    print("seed %d done" % seed, flush=True)
    return rows


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(8) as pool:
        rows = sum(pool.map(run_seed, C.SEEDS), [])
    out = os.path.join(C.RESULTS, "classical")
    os.makedirs(out, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    print(
        runs.pivot_table(index="dft_pct", columns="model", values="test_mae_meV", aggfunc="mean").round(1).to_string()
    )
