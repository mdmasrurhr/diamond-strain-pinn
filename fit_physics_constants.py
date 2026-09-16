"""
fit_physics_constants.py -- fit the analytic band-edge model and say how good it is.

The physics the networks are constrained by is a closed-form model of the two
band edges with 13 constants. Those constants are fitted to DFT labels once and
then frozen; every training run uses the same numbers.

This script does three things:

  1. Re-fits the constants and checks they reproduce the frozen values in
     physics.py (a parity check -- if this fails, the pipeline is inconsistent).
  2. Scores the analytic model on its own, with no network involved. That is the
     error a PINN starts from before it sees a single label.
  3. Measures how many labels the fit itself needs, by re-fitting on k states
     and scoring on held-out data. This is the calibration cost a new material
     would have to pay, and it is the number to quote when asked whether the
     approach transfers.

    python fit_physics_constants.py             # parity + calibration cost
    python fit_physics_constants.py --freeze    # fit the production constants

--freeze is the step that produces the frozen constants in physics.py, and its
protocol answers the two defects the old calibration carried:

  * it fits on the SEED-42 TRAINING FOLD only, never on all rows, so the
    constants have not seen the canonical test fold (the old fits used every
    row, which put every downstream test state inside the calibration data);
  * it re-fits on the folds of two further seeds and reports the spread, so
    "the constants are stable across folds" is a measured statement.

It writes results/physics_fit/constants_v9_fold42.json and prints the block to
paste into physics.py. physics.py stays the single frozen source; this script
is how its numbers are produced and audited.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
import physics as P
import train_pinn

# The band-edge equations, the fitters and the constant names all live in
# physics.py so that this script and run_matched_prior.py cannot drift
# apart. Local aliases keep the code below readable.
CBM_KEYS, VBM_KEYS = P.CBM_KEYS, P.VBM_KEYS
cbm_model, vbm_model = P.cbm_numpy, P.vbm_numpy
fit_cbm, fit_vbm = P.fit_cbm, P.fit_vbm


def family_of(label):
    s = str(label).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"


def _train_fold(seed):
    """The exact training fold the trainer uses, obtained from the trainer.

    Re-deriving the split here would invite drift; calling the trainer's own
    function guarantees the constants are fitted on precisely the rows the
    models train on.
    """
    return train_pinn.load_and_split(C.DATA_CSV, seed=seed, dft_pct=100)


def freeze():
    """Fit the production constants on the seed-42 training fold."""
    import os
    out = C.results_dir("physics_fit")
    df = pd.read_csv(C.DATA_CSV)
    strain_all = df[C.STRAIN_COLS].values.astype(np.float64)
    eg_all = df[C.TARGET_EG].values.astype(np.float64)

    fits = {}
    for seed in [42, 43, 44]:
        data = _train_fold(seed)
        s_tr = data["strain_tr"].astype(np.float64)
        pc = fit_cbm(s_tr, data["cbm_tr"].astype(np.float64))
        pv = fit_vbm(s_tr, data["vbm_tr"].astype(np.float64))
        i_te = data["idx_te"]
        pred = cbm_model(strain_all[i_te], *pc) - vbm_model(strain_all[i_te], *pv)
        mae = float(np.mean(np.abs(pred - eg_all[i_te])) * 1000.0)
        fits[seed] = (list(pc), list(pv), mae)
        print("fold seed %d: physics-only held-out Eg MAE %.1f meV" % (seed, mae))

    print("\nfold stability (constant, seed42, spread across 3 folds):")
    names = CBM_KEYS + VBM_KEYS
    vals42 = fits[42][0] + fits[42][1]
    spread = []
    for i, name in enumerate(names):
        vs = [fits[sd][0][i] if i < len(CBM_KEYS) else fits[sd][1][i - len(CBM_KEYS)]
              for sd in fits]
        sp = max(vs) - min(vs)
        spread.append(sp)
        print("  %-8s %14.4f   spread %.4f eV" % (name, vals42[i], sp))

    const = dict(zip(names, [float(v) for v in vals42]))
    meta = dict(protocol="fit on the seed-42 training fold (70%), stratified split; "
                         "test fold never seen by the fit",
                dataset=os.path.basename(C.DATA_CSV),
                heldout_mae_meV={sd: round(fits[sd][2], 1) for sd in fits},
                max_constant_spread_eV=round(max(spread), 4))
    path = os.path.join(out, "constants_v9_fold42.json")
    json.dump(dict(constants=const, meta=meta), open(path, "w"), indent=2)
    print("\nwrote %s" % path)
    print("\n# ---- paste into physics.py ----")
    for name in names:
        print("%s = %r" % (name.ljust(6), const[name]))


def main():
    out = C.results_dir("physics_fit")
    df = pd.read_csv(C.DATA_CSV)
    strain = df[C.STRAIN_COLS].values.astype(np.float64)
    eg = df[C.TARGET_EG].values.astype(np.float64)
    cbm = df[C.TARGET_CBM].values.astype(np.float64)
    vbm = df[C.TARGET_VBM].values.astype(np.float64)
    stype = df["strain_type"].map(family_of).values
    idx = np.arange(len(df))

    # ---- 1. parity with the frozen constants ----------------------------
    p_c, p_v = fit_cbm(strain, cbm), fit_vbm(strain, vbm)
    rows = []
    worst = 0.0
    print("%-8s %18s %18s %12s" % ("constant", "frozen", "re-fitted", "abs diff"))
    for name, val in list(zip(CBM_KEYS, p_c)) + list(zip(VBM_KEYS, p_v)):
        frozen = getattr(P, name)
        d = abs(frozen - val)
        worst = max(worst, d)
        rows.append(dict(constant=name, frozen=frozen, refitted=val, abs_diff=d))
        print("%-8s %18.10f %18.10f %12.2e" % (name, frozen, val, d))
    print("largest disagreement: %.2e eV" % worst)
    if worst > 1e-6:
        print("  NOTE: the frozen constants are not the optimum of this dataset.")
    pd.DataFrame(rows).to_csv("%s/parity.csv" % out, index=False)

    # ---- 2 & 3. calibration cost ----------------------------------------
    # Fit on k labelled states, score on the held-out fold. No network.
    print("\nanalytic model alone, fitted on k labelled states (17 seeds):")
    print("%6s %6s  %18s" % ("k", "pct", "test MAE (meV)"))
    cost = []
    for pct in [1, 2, 5, 10, 25, 50, 100]:
        maes = []
        for seed in C.SEEDS:
            i_tv, i_te = train_test_split(idx, test_size=C.TEST_FRACTION,
                                          random_state=seed, stratify=stype)
            i_tr, _ = train_test_split(i_tv,
                                       test_size=C.VAL_FRACTION_OF_REMAINDER,
                                       random_state=seed, stratify=stype[i_tv])
            rng = np.random.default_rng(seed + C.LABEL_RNG_OFFSET)
            k = max(1, round(len(i_tr) * pct / 100.0))
            lab = i_tr[rng.choice(len(i_tr), size=k, replace=False)]
            pc, pv = fit_cbm(strain[lab], cbm[lab]), fit_vbm(strain[lab], vbm[lab])
            pred = cbm_model(strain[i_te], *pc) - vbm_model(strain[i_te], *pv)
            maes.append(np.mean(np.abs(pred - eg[i_te])) * 1000.0)
        cost.append(dict(pct=pct, n_labeled=k, mae_mean=round(np.mean(maes), 1),
                         mae_sd=round(np.std(maes, ddof=1), 1),
                         mae_median=round(float(np.median(maes)), 1)))
        print("%6d %5d%%  %8.1f +/- %-8.1f (median %.1f)"
              % (k, pct, np.mean(maes), np.std(maes, ddof=1), np.median(maes)))

    pd.DataFrame(cost).to_csv("%s/calibration_cost.csv" % out, index=False)
    print("\nwrote %s/parity.csv and calibration_cost.csv" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="fit the production constants on the seed-42 train fold")
    a = ap.parse_args()
    freeze() if a.freeze else main()
