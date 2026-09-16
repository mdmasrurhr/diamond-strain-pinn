"""
run_classical_baselines.py -- three standard regressors, same data, same splits.

The data-driven model we compare against is a neural network, which invites a
fair question: is the physics advantage an advantage over data-driven regression
in general, or only over one particular architecture? So this runs three
methods that are usually hard to beat on small tabular problems -- gradient
boosting, a random forest, and a Gaussian process.

Fairness is the point, so each one gets exactly the same treatment as the
networks: the same split, the same labelled subset at each fraction, the same
seeds. They predict the bandgap directly rather than through the two band edges,
which is the easier task and therefore the more favourable setting for them.

No GPU, no training loop -- this finishes in a few minutes.

    python run_classical_baselines.py

Writes results/classical_baselines/{runs,summary}.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fractions", nargs="+", type=int,
                    default=[f for f in C.FRACTIONS if f > 0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[:3] if args.quick else C.SEEDS
    out = C.results_dir("classical_baselines")

    df = pd.read_csv(C.DATA_CSV)
    X = df[C.STRAIN_COLS].values.astype(np.float64)
    y = df[C.TARGET_EG].values.astype(np.float64)
    stype = df["strain_type"].map(family_of).values
    idx = np.arange(len(df))

    rows = []
    for seed in seeds:
        i_tv, i_te = train_test_split(idx, test_size=C.TEST_FRACTION,
                                      random_state=seed, stratify=stype)
        i_tr, _ = train_test_split(i_tv, test_size=C.VAL_FRACTION_OF_REMAINDER,
                                   random_state=seed, stratify=stype[i_tv])
        scaler = StandardScaler().fit(X[i_tr])
        Xtr_all, Xte = scaler.transform(X[i_tr]), scaler.transform(X[i_te])

        for pct in args.fractions:
            rng = np.random.default_rng(seed + C.LABEL_RNG_OFFSET)
            k = max(1, round(len(i_tr) * pct / 100.0))
            pos = rng.choice(len(i_tr), size=k, replace=False)
            Xl, yl = Xtr_all[pos], y[i_tr][pos]

            for name, model in build_models(seed, k).items():
                try:
                    model.fit(Xl, yl)
                    pred = model.predict(Xte)
                    mae = float(np.mean(np.abs(pred - y[i_te])) * 1000.0)
                    ss_res = float(np.sum((y[i_te] - pred) ** 2))
                    ss_tot = float(np.sum((y[i_te] - y[i_te].mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot
                    per = {}
                    for fam in C.STRAIN_FAMILIES:
                        m = stype[i_te] == fam
                        per["mae_%s_meV" % fam] = (
                            round(float(np.mean(np.abs(pred[m] - y[i_te][m])) * 1000), 2)
                            if m.any() else "")
                except Exception as exc:
                    print("  ! %s seed=%d pct=%d failed: %s" % (name, seed, pct, exc))
                    continue
                rows.append(dict(model=name, seed=seed, dft_pct=pct, n_labeled=k,
                                 test_mae_meV=round(mae, 2),
                                 test_r2=round(r2, 4), **per))
        print("seed %d done" % seed)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "runs.csv"), index=False)
    agg = (d.groupby(["model", "dft_pct"])[["test_mae_meV", "test_r2"]]
             .agg(["mean", "std"]).round(2))
    agg.to_csv(os.path.join(out, "summary.csv"))

    print("\ntest MAE (meV), mean over %d seeds:" % len(seeds))
    piv = d.pivot_table(index="dft_pct", columns="model",
                        values="test_mae_meV", aggfunc="mean").round(1)
    print(piv.to_string())
    print("\nwrote %s/{runs,summary}.csv" % out)


if __name__ == "__main__":
    main()
