"""Fit the constants of the analytic models for every run, on that run's training data only.

  python calibrate.py main               constants/two_band/symmetry       the paper's constants, 17 seeds
  python calibrate.py main --one_band    constants/one_band/symmetry       linear shear term (Supplementary Note 8)
  python calibrate.py heldout [test ...] constants/two_band/<test>         each family left out, three strain cut-offs
  python calibrate.py labelmatched       constants/labelmatched/<pct>pct   labeled and isotropic training states only

main and heldout also write fit_summary.csv: the gap MAE of the analytic models alone on each run's test states.
In the held-out test without shear, the shear constants XI_UP, D_BP and DELTA_X are set to 0 (no data determine them).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import calibration as CV
import config as C

HELDOUT = {
    "group_isotropic": ["--split", "group", "--holdout_family", "isotropic"],
    "group_uniaxial": ["--split", "group", "--holdout_family", "uniaxial"],
    "group_biaxial": ["--split", "group", "--holdout_family", "biaxial"],
    "group_triaxial": ["--split", "group", "--holdout_family", "triaxial"],
    "group_shear": ["--split", "group", "--holdout_family", "shear"],
    "magnitude_03": ["--split", "magnitude", "--magnitude_thresh", "0.03"],
    "magnitude_05": ["--split", "magnitude", "--magnitude_thresh", "0.05"],
    "magnitude_07": ["--split", "magnitude", "--magnitude_thresh", "0.07"],
}
LABELMATCHED_FRACTIONS = [1, 2, 5, 10]

DATA = pd.read_csv(C.DATA_CSV)
STRAIN = DATA[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values.astype(float)


def save(const, folder, seed):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "const_seed%d.json" % seed), "w") as f:
        json.dump(const, f, indent=1)


def test_mae(const, data, keys):
    """Gap MAE (meV) of the analytic models alone on the run's test states."""
    te = np.array(data["idx_te"])
    th = np.array([const[k] for k in keys])
    return 1000 * np.abs(CV.gap_form(STRAIN[te], th) - DATA.Eg_eV.values[te]).mean()


def fit_runs(folder, options, two_band, name):
    """Fit on the training portion of every seed; returns one summary row per seed."""
    trainer = C.import_trainer(*options)
    keys = CV.KEYS2 if two_band else CV.KEYS
    rows = []
    for seed in C.SEEDS:
        d = trainer.load_and_split(C.DATA_CSV, seed, 100)
        s = d["strain_tr"].astype(float)
        const = CV.fit_constants(s, d["cbm_tr"].astype(float), d["vbm_tr"].astype(float), seed=seed, two_band=two_band)
        if name == "group_shear":
            assert np.abs(s[:, 3:]).max() < 1e-8, "training states with shear in the held-out test without shear"
            const["XI_UP"] = 0.0
            const["D_BP"] = 0.0
            if two_band:
                const["DELTA_X"] = 0.0
        save(const, folder, seed)
        rows.append(
            dict(seed=seed, n_train=len(s), n_test=len(d["idx_te"]), formula_gap_MAE_meV=test_mae(const, d, keys))
        )
        print(
            "%-16s seed %d: %4d training states | analytic models on the test states: %.1f meV"
            % (name, seed, len(s), rows[-1]["formula_gap_MAE_meV"]),
            flush=True,
        )
    pd.DataFrame(rows).to_csv(os.path.join(folder, "fit_summary.csv"), index=False)


def fit_labelmatched():
    """Fit on the labeled training states plus the isotropic ones (whose gaps every run sees)."""
    trainer = C.import_trainer("--split", "symmetry")
    for pct in LABELMATCHED_FRACTIONS:
        for seed in C.SEEDS:
            d = trainer.load_and_split(C.DATA_CSV, seed, pct)
            st = d["strain_tr"].astype(float)
            iso = (np.abs(st[:, 3:]).sum(1) < 1e-4) & (
                np.abs(st[:, :3] - st[:, :3].mean(1, keepdims=True)).max(1) < 1e-4
            )
            use = np.zeros(len(st), bool)
            use[d["dft_idx"]] = True
            use |= iso
            const = CV.fit_constants(
                st[use], d["cbm_tr"][use].astype(float), d["vbm_tr"][use].astype(float), seed=seed, two_band=True
            )
            save(const, os.path.join(C.CONSTANTS, "labelmatched", "%dpct" % pct), seed)
            print(
                "%3d%% seed %d: fitted on %d states (%d labeled)" % (pct, seed, use.sum(), len(d["dft_idx"])),
                flush=True,
            )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "main":
        band = "one_band" if "--one_band" in sys.argv else "two_band"
        fit_runs(os.path.join(C.CONSTANTS, band, "symmetry"), ["--split", "symmetry"], band == "two_band", "symmetry")
    elif mode == "heldout":
        for test in sys.argv[2:] or list(HELDOUT):
            fit_runs(os.path.join(C.CONSTANTS, "two_band", test), HELDOUT[test], True, test)
    elif mode == "labelmatched":
        fit_labelmatched()
    else:
        sys.exit(__doc__)
