"""Refit the physics constants on each run's own labelled subset.

In the main experiment the constants are fitted once on the whole dataset and
then frozen, so at 1% labels the supervised term sees nine states while the
physics term carries information from all of them. This removes that asymmetry:
for each seed and fraction the 13 constants are refitted on that run's labelled
rows only, using the same split and subset draw as the trainer.

The difference between this and the main experiment is the part of the
low-label result that comes from the reused calibration.

    python run_matched_prior.py
    python run_matched_prior.py --quick

Writes results/matched_prior/<model>/<pct>pct/seed<n>/.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
import jobs
import physics as P

FRACTIONS = [1, 5]          # where the prior does the most work


def family_of(label):
    s = str(label).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fractions", nargs="+", type=int, default=FRACTIONS)
    ap.add_argument("--models", nargs="+", default=["sa", "rba"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[:3] if args.quick else C.SEEDS
    cdir = C.results_dir("matched_prior", "constants")

    df = pd.read_csv(C.DATA_CSV)
    strain = df[C.STRAIN_COLS].values.astype(np.float64)
    eg = df[C.TARGET_EG].values.astype(np.float64)
    cbm = df[C.TARGET_CBM].values.astype(np.float64)
    vbm = df[C.TARGET_VBM].values.astype(np.float64)
    stype = df["strain_type"].map(family_of).values
    idx = np.arange(len(df))

    cmds, diag = [], []
    for seed in seeds:
        # Exactly the trainer's split.
        i_tv, i_te = train_test_split(idx, test_size=C.TEST_FRACTION,
                                      random_state=seed, stratify=stype)
        i_tr, _ = train_test_split(i_tv, test_size=C.VAL_FRACTION_OF_REMAINDER,
                                   random_state=seed, stratify=stype[i_tv])
        for pct in args.fractions:
            # Exactly the trainer's labelled draw.
            rng = np.random.default_rng(seed + C.LABEL_RNG_OFFSET)
            k = max(1, round(len(i_tr) * pct / 100.0))
            lab = i_tr[rng.choice(len(i_tr), size=k, replace=False)]

            p_c = P.fit_cbm(strain[lab], cbm[lab])
            p_v = P.fit_vbm(strain[lab], vbm[lab])
            const = dict(zip(P.CBM_KEYS, [float(v) for v in p_c]))
            const.update(dict(zip(P.VBM_KEYS, [float(v) for v in p_v])))

            jf = os.path.join(cdir, "const_%dpct_seed%d.json" % (pct, seed))
            with open(jf, "w") as fh:
                json.dump(const, fh, indent=1)

            # What this prior scores on its own, with no network. It bounds what
            # a model built on it can inherit.
            pred = P.cbm_numpy(strain[i_te], *p_c) - P.vbm_numpy(strain[i_te], *p_v)
            prior_mae = float(np.mean(np.abs(pred - eg[i_te])) * 1000.0)
            diag.append(dict(seed=seed, pct=pct, n_labeled=k,
                             prior_test_mae_meV=round(prior_mae, 1)))

            for model in args.models:
                rd = C.results_dir("matched_prior", model, "%dpct" % pct,
                                   "seed%d" % seed)
                cmds.append(jobs.train_cmd(
                    model, seed, pct, rd,
                    env_prefix="PHYSICS_CONST_JSON=%s " % jf))

        print("seed %d: constants refitted" % seed)

    d = pd.DataFrame(diag)
    d.to_csv(os.path.join(C.RESULTS, "matched_prior", "prior_only.csv"),
             index=False)
    print("\nlabel-matched prior on its own (no network):")
    for pct in args.fractions:
        v = d[d.pct == pct].prior_test_mae_meV
        print("  %3d%% (%d labels): %.1f +/- %.1f meV"
              % (pct, d[d.pct == pct].n_labeled.iloc[0], v.mean(), v.std(ddof=1)))

    print("\nretraining %d runs with label-matched priors" % len(cmds))
    jobs.run_jobs(cmds, "matched_prior")


if __name__ == "__main__":
    main()
