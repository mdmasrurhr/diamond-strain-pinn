#!/usr/bin/env python3
"""
Analytic models alone: band edges of strained diamond from deformation-potential theory (conduction-band minimum CBM,
six valleys with a two-band shear term) and the Bir-Pikus model (valence-band maximum VBM); gap = CBM - VBM.

The 13 constants and the valley gap DELTA_X are fitted to the training states of each run (joint least squares on CBM,
VBM and gap; ridge penalty on the five second-order constants, its strength by 5-fold cross-validation). The fitted
constants are the theta0 of every physics model of the paper. Evaluation uses the exact minimum over the valleys and
the exact largest eigenvalue (the networks use smooth versions in training).

Name in the paper: "analytic models alone". Campaign scripts: calib_v2.py, calibrate_constants.py,
calibrate_heldout.py, calibrate_labelmatched.py. This file reproduces their constants exactly.

Experiments (--experiment):
  main          symmetry-grouped division; constants fitted on each run's training portion
  heldout       one strain class left out, or every state beyond a strain magnitude; constants fitted on the reduced
                training portion (shear constants set to 0 when shear is left out)
  labelmatched  1, 2, 5, 10% labeled; constants fitted only on the labeled and isotropic training states

Commands (python analytic.py <command> -h shows the options):
  constants   fit the constants for every seed of one experiment   -> <out>/constants/...
  evaluate    test errors of the analytic models alone              -> <out>/analytic_<experiment>.csv
  study       constants, then evaluate
"""

# ======================================================================================================================
# LIBRARIES
# ======================================================================================================================
import argparse
import itertools
import json
import os
import time
import tomllib

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================================================================================================
# FUNCTIONS IN THIS FILE (in the order they appear)
# ======================================================================================================================
# Settings
#   read_config(path, *sections)                   the named sections of config.toml
#   cubic_point_group()                            the 48 cubic symmetry operations (used by the division)
# Analytic band-edge models
#   shear_shift(e, xi_up, delta_x, xp)             shear splitting of a pair of conduction valleys
#   valley_energies(strain, ...)                   energies of the six conduction valleys
#   cbm_numpy(strain, th), vbm_numpy(strain, th), gap_numpy(strain, th)   exact minimum / largest eigenvalue
# Fit of the constants
#   fit_once(strain, cbm, vbm, ridge, start)       one least-squares fit
#   fit_constants(strain, cbm, vbm, seed)          the full fit (several starts, ridge by cross-validation)
# Data
#   strain_class_of, cubic_images, heldout_schemes, load_and_split, isotropic_mask
# Study
#   constants_file(...), read_constants(...), settings_of(...), fit_all_constants(...)
#   evaluate(data_csv, out, seeds, experiment, fractions)
# Command line
#   parse_int_list(text), main()

# ======================================================================================================================
# SETTINGS (all in config.toml)
# ======================================================================================================================
def read_config(path, *sections):
    """The named sections of the settings file, in the order asked for."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return [config[name] for name in sections]


def cubic_point_group():
    """The 48 operations of the cubic point group: axis permutations x sign changes."""
    operations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            operations.append(np.diag(signs) @ np.eye(3)[list(permutation)])
    return operations


MODEL_NAME = "analytic"
EXPERIMENT_NAMES = ["main", "heldout", "labelmatched"]     # experiments (see the top of the file)
# the settings file: config.toml next to this file, or another one given with ANALYTIC_CONFIG=<path>
CONFIG_FILE = os.environ.get("ANALYTIC_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml"))
STUDY, DATA, EXPERIMENTS, CONSTANTS, PHYSICS, FIT = read_config(CONFIG_FILE, "study", "data", "experiments",
                                                                "constants", "physics", "fit")
CUBIC_OPERATIONS = cubic_point_group()


# ======================================================================================================================
# ANALYTIC BAND-EDGE MODELS
# ======================================================================================================================
def shear_shift(e, xi_up, delta_x, xp):
    """Shift of a pair of conduction valleys under the shear component e (two-band form with valley gap delta_x).
    xp is the array library (numpy here)."""
    eta = 4.0 * xi_up * e / delta_x
    size = xp.abs(eta)
    return xp.where(size < 1.0, -0.25 * delta_x * eta ** 2, -0.25 * delta_x * (2.0 * size - 1.0))


def valley_energies(strain, xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2, delta_x):
    """Energies of the six conduction valleys relative to CBM_0, one column per valley (numpy)."""
    e_xx = strain[:, 0]
    e_yy = strain[:, 1]
    e_zz = strain[:, 2]
    e_xy = strain[:, 3]
    e_yz = strain[:, 4]
    e_zx = strain[:, 5]
    i1 = e_xx + e_yy + e_zz                       # volume change (trace of the strain)

    def valley(e_axis):
        return xi_d * i1 + xi_u * e_axis + xi_q * e_axis ** 2 + xi_c * i1 * e_axis + xi_q2 * i1 ** 2

    if delta_x <= 0:
        # one-band shear term: used inside the fit before DELTA_X is added, and when shear is left out
        columns = [valley(e_zz) + 2 * xi_up * e_xy, valley(e_zz) - 2 * xi_up * e_xy,
                   valley(e_yy) + 2 * xi_up * e_zx, valley(e_yy) - 2 * xi_up * e_zx,
                   valley(e_xx) + 2 * xi_up * e_yz, valley(e_xx) - 2 * xi_up * e_yz]
    else:
        z_pair = valley(e_zz) + shear_shift(e_xy, xi_up, delta_x, np)
        y_pair = valley(e_yy) + shear_shift(e_zx, xi_up, delta_x, np)
        x_pair = valley(e_xx) + shear_shift(e_yz, xi_up, delta_x, np)
        columns = [z_pair, z_pair, y_pair, y_pair, x_pair, x_pair]
    return np.stack(columns, axis=1)


# ======================================================================================================================
# FIT OF THE CONSTANTS
# Joint least squares on the CBM, VBM and gap of the training states; a ridge penalty on the five second-order
# constants with its strength chosen by 5-fold cross-validation; several starts for d and for DELTA_X.
# ======================================================================================================================
def cbm_numpy(strain, th):
    """CBM of the fit: exact minimum over the valleys (th has 13 values, or 14 with DELTA_X last)."""
    if len(th) > 13:
        delta_x = th[13]
    else:
        delta_x = 0.0
    valleys = valley_energies(strain, th[1], th[2], th[3], th[4], th[5], th[6], delta_x)
    return th[0] + valleys.min(axis=1)


def vbm_numpy(strain, th):
    """VBM of the fit: exact largest eigenvalue of the Bir-Pikus matrix."""
    vbm_0, a_v, a_v2, b, b2, d = th[7], th[8], th[9], th[10], th[11], th[12]
    e_xx = strain[:, 0]
    e_yy = strain[:, 1]
    e_zz = strain[:, 2]
    e_xy = strain[:, 3]
    e_yz = strain[:, 4]
    e_zx = strain[:, 5]
    i1 = e_xx + e_yy + e_zz
    hydrostatic = a_v * i1 + a_v2 * i1 ** 2
    dev_xx = e_xx - i1 / 3.0
    dev_yy = e_yy - i1 / 3.0
    dev_zz = e_zz - i1 / 3.0
    H = np.zeros((len(strain), 3, 3))
    H[:, 0, 0] = hydrostatic + (b + b2 * dev_xx) * dev_xx
    H[:, 1, 1] = hydrostatic + (b + b2 * dev_yy) * dev_yy
    H[:, 2, 2] = hydrostatic + (b + b2 * dev_zz) * dev_zz
    H[:, 0, 1] = d * e_xy
    H[:, 1, 0] = d * e_xy
    H[:, 1, 2] = d * e_yz
    H[:, 2, 1] = d * e_yz
    H[:, 0, 2] = d * e_zx
    H[:, 2, 0] = d * e_zx
    H += PHYSICS["eigen_eps"] * np.eye(3)[None]
    return vbm_0 + np.linalg.eigvalsh(H)[:, 2]


def gap_numpy(strain, th):
    return cbm_numpy(strain, th) - vbm_numpy(strain, th)


def fit_once(strain, cbm, vbm, ridge, start):
    """One least-squares fit from the starting values `start` with ridge strength `ridge`."""
    n_states = len(strain)
    ridge_index = [CONSTANTS["constant_names"].index(name) for name in FIT["ridge_constants"]]
    ridge_scale = np.array([max(abs(start[i]), 1.0) for i in ridge_index])

    def residuals(th):
        cbm_fit = cbm_numpy(strain, th)
        vbm_fit = vbm_numpy(strain, th)
        parts = [cbm_fit - cbm, vbm_fit - vbm, (cbm_fit - vbm_fit) - (cbm - vbm)]
        if ridge > 0:
            parts.append(np.sqrt(ridge * n_states) * th[ridge_index] / ridge_scale)
        return np.concatenate(parts)

    lower = np.full(len(start), -np.inf)
    upper = np.full(len(start), np.inf)
    if len(start) > 13:
        lower[13] = FIT["delta_x_bounds"][0]
        upper[13] = FIT["delta_x_bounds"][1]
    return least_squares(residuals, start, method="trf", bounds=(lower, upper), max_nfev=FIT["max_function_evals"])


def fit_constants(strain, cbm, vbm, seed):
    """Fit the 13 constants and DELTA_X to the given states. Returns a dict {name: value}."""
    # step 1: the 13 constants, several starting values for d
    best = None
    for d_start in FIT["d_starts"]:
        start = np.array([FIT["start"][name] for name in CONSTANTS["constant_names"]])
        start[12] = d_start
        result = fit_once(strain, cbm, vbm, 0.0, start)
        if best is None or result.cost < best.cost:
            best = result
    thirteen = best.x

    # step 2: add DELTA_X, several starting values
    best = None
    for delta_start in FIT["delta_x_starts"]:
        result = fit_once(strain, cbm, vbm, 0.0, np.append(thirteen, delta_start))
        if best is None or result.cost < best.cost:
            best = result
    fourteen = best.x

    # step 3: ridge strength by 5-fold cross-validation (gap MAE on the held-out fold)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(strain)), FIT["folds"])
    cv_error = {}
    for ridge in FIT["ridge_strengths"]:
        errors = []
        for k in range(FIT["folds"]):
            held_out = folds[k]
            kept = np.concatenate([folds[j] for j in range(FIT["folds"]) if j != k])
            th = fit_once(strain[kept], cbm[kept], vbm[kept], ridge, fourteen).x
            errors.append(np.abs(gap_numpy(strain[held_out], th) - (cbm[held_out] - vbm[held_out])).mean())
        cv_error[ridge] = float(np.mean(errors))
    best_ridge = min(cv_error, key=cv_error.get)

    # step 4: final fit on all given states
    th = fit_once(strain, cbm, vbm, best_ridge, fourteen).x
    constants = {}
    for name, value in zip(CONSTANTS["constant_names"] + ["DELTA_X"], th):
        constants[name] = float(value)
    return constants


# ======================================================================================================================
# DATA
# ======================================================================================================================
def strain_class_of(name):
    """Map the dataset's strain_type (uniaxial_x, biaxial_xy, unstrained, ...) to one of the five classes."""
    name = str(name).lower().strip()
    if name.startswith("uniaxial"):
        return "uniaxial"
    if name.startswith("biaxial"):
        return "biaxial"
    if name in ("isotropic", "triaxial", "shear"):
        return name
    return "isotropic"                      # the unstrained state counts as isotropic


def cubic_images(strain):
    """The 48 images R e R^T of one strain state under the cubic operations (Voigt order)."""
    T = np.array([[strain[0], strain[3], strain[5]],
                  [strain[3], strain[1], strain[4]],
                  [strain[5], strain[4], strain[2]]])
    images = []
    for R in CUBIC_OPERATIONS:
        Tr = R @ T @ R.T
        images.append(np.array([Tr[0, 0], Tr[1, 1], Tr[2, 2], Tr[0, 1], Tr[1, 2], Tr[0, 2]]))
    return images


def heldout_schemes():
    """Names of the held-out divisions: group_<class> (one strain class left out) and magnitude_<03|05|07> (every state
    beyond that largest strain component left out)."""
    names = ["group_" + name for name in EXPERIMENTS["heldout_classes"]]
    for threshold in EXPERIMENTS["heldout_magnitudes"]:
        names.append("magnitude_%02d" % round(100 * threshold))
    return names


def load_and_split(data_csv, seed, fraction, scheme=None):
    """Read the dataset; divide it into training, validation and test states (scheme None: symmetry-grouped division;
    otherwise a held-out division); scale the inputs on the training states only; draw the labeled subset."""
    table = pd.read_csv(data_csv)
    strain = table[DATA["strain_columns"]].values.astype(np.float32)
    gap = table["Eg_eV"].values.astype(np.float32)
    cbm = table["CBM_eV"].values.astype(np.float32)
    vbm = table["VBM_eV"].values.astype(np.float32)
    classes = np.array([strain_class_of(name) for name in table["strain_type"].values])
    n_states = len(table)
    all_index = np.arange(n_states)

    if scheme is None:
        # 1. symmetry classes: states related by a cubic operation share a class (and the same band edges)
        key_to_id = {}
        symmetry_class = np.zeros(n_states, dtype=int)
        for i in range(n_states):
            keys = [tuple(np.round(image, DATA["symmetry_decimals"])) for image in cubic_images(strain[i].astype(np.float64))]
            key = min(keys)
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id)
            symmetry_class[i] = key_to_id[key]

        # 2. per strain class, deal whole symmetry classes in a seeded random order: test first, then validation,
        #    then training, so no test state has a rotated twin in training or validation
        rng = np.random.default_rng(seed)
        train_list = []
        val_list = []
        test_list = []
        for class_name in np.unique(classes):
            members = all_index[classes == class_name]
            class_ids = np.unique(symmetry_class[members])
            rng.shuffle(class_ids)
            n_members = len(members)
            n_test = 0
            n_val = 0
            for class_id in class_ids:
                states = members[symmetry_class[members] == class_id]
                if n_test < round(DATA["test_share"] * n_members):
                    test_list.extend(states)
                    n_test += len(states)
                elif n_val < round(DATA["validation_share"] * n_members):
                    val_list.extend(states)
                    n_val += len(states)
                else:
                    train_list.extend(states)
        index_train = np.array(sorted(train_list))
        index_val = np.array(sorted(val_list))
        index_test = np.array(sorted(test_list))
    elif scheme.startswith("group_"):
        # one strain class is the test set; the rest is divided into training and validation, stratified by class
        left_out = scheme[len("group_"):]
        index_test = all_index[classes == left_out]
        pool = all_index[classes != left_out]
        index_train, index_val = train_test_split(pool, test_size=EXPERIMENTS["heldout_validation_share"],
                                                  random_state=seed, stratify=classes[pool])
    elif scheme.startswith("magnitude_"):
        # every state whose largest strain component exceeds the threshold is the test set
        threshold = int(scheme[len("magnitude_"):]) / 100
        size = np.max(np.abs(strain), axis=1)
        index_test = all_index[size > threshold]
        pool = all_index[size <= threshold]
        index_train, index_val = train_test_split(pool, test_size=EXPERIMENTS["heldout_validation_share"],
                                                  random_state=seed)
    else:
        raise SystemExit("unknown held-out scheme %s (known: %s)" % (scheme, ", ".join(heldout_schemes())))

    # 3. input scaling (mean 0, standard deviation 1), fitted on the training states only. A strain component that is
    #    only round-off in training (the shear components when shear is left out) gets the typical scale instead.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(strain[index_train]).astype(np.float32)
    if (scaler.scale_ < 1e-6).any():
        tiny = scaler.scale_ < 1e-6
        scaler.scale_[tiny] = np.median(scaler.scale_[~tiny])
        scaler.mean_[tiny] = 0.0
        x_train = scaler.transform(strain[index_train]).astype(np.float32)
    x_val = scaler.transform(strain[index_val]).astype(np.float32)
    x_test = scaler.transform(strain[index_test]).astype(np.float32)

    # 4. labeled subset (positions inside the training set)
    rng_labels = np.random.default_rng(seed + DATA["label_seed_offset"])
    if fraction == 0:
        labeled = np.array([], dtype=np.int64)
    else:
        n_labeled = max(1, round(len(index_train) * fraction / 100.0))
        labeled = rng_labels.choice(len(index_train), size=n_labeled, replace=False)

    split = np.full(n_states, "train", dtype=object)
    split[index_val] = "val"
    split[index_test] = "test"

    return {"table": table, "strain": strain, "classes": classes, "split": split, "scaler": scaler,
            "index_train": index_train, "index_val": index_val, "index_test": index_test, "labeled": labeled,
            "x_train": x_train, "x_val": x_val, "x_test": x_test,
            "strain_train": strain[index_train], "gap_train": gap[index_train],
            "cbm_train": cbm[index_train], "vbm_train": vbm[index_train],
            "gap_val": gap[index_val], "gap_test": gap[index_test],
            "gap": gap, "cbm": cbm, "vbm": vbm}


def isotropic_mask(strain):
    """True for states under isotropic strain: no shear and three equal normal components (numpy, float64)."""
    tolerance = DATA["isotropic_tolerance"]
    no_shear = np.abs(strain[:, 3:]).sum(axis=1) < tolerance
    equal_normal = np.abs(strain[:, :3] - strain[:, :3].mean(axis=1, keepdims=True)).max(axis=1) < tolerance
    return no_shear & equal_normal


# ======================================================================================================================
# STUDY
# ======================================================================================================================
def constants_file(out, seed, experiment="main", fraction=None, scheme=None):
    """main: constants/const_seed<s>.json;  heldout: constants/heldout/<scheme>/...;  labelmatched: .../<fraction>pct/..."""
    if experiment == "main":
        return os.path.join(out, "constants", "const_seed%d.json" % seed)
    if experiment == "heldout":
        return os.path.join(out, "constants", "heldout", scheme, "const_seed%d.json" % seed)
    return os.path.join(out, "constants", "labelmatched", "%dpct" % fraction, "const_seed%d.json" % seed)


def read_constants(out, seed, experiment="main", fraction=None, scheme=None):
    path = constants_file(out, seed, experiment, fraction, scheme)
    if not os.path.exists(path):
        raise SystemExit("missing %s: run the 'constants' command first (same --experiment)" % path)
    with open(path) as f:
        return json.load(f)


def settings_of(experiment, fractions=None):
    """The (fraction, scheme) pairs of one experiment."""
    if experiment == "main":
        return [(fraction, None) for fraction in (fractions or STUDY["fractions"])]
    if experiment == "heldout":
        return [(100, scheme) for scheme in heldout_schemes()]
    if experiment == "labelmatched" and experiment in EXPERIMENT_NAMES:
        return [(fraction, None) for fraction in (fractions or EXPERIMENTS["labelmatched_fractions"])]
    raise SystemExit("unknown experiment %s (this model: %s)" % (experiment, ", ".join(EXPERIMENT_NAMES)))


def fit_all_constants(data_csv, out, seeds, experiment="main", fractions=None):
    """Fit theta0 for every seed. main: on all training states (labeled or not, as in the paper); heldout: on the
    training states of each held-out division (shear constants set to 0 when shear is left out); labelmatched: only
    on the labeled and the isotropic training states."""
    if experiment == "main":
        settings = [(100, None)]
    else:
        settings = settings_of(experiment, fractions)
    for seed in seeds:
        for fraction, scheme in settings:
            path = constants_file(out, seed, experiment, fraction, scheme)
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            start = time.time()
            data = load_and_split(data_csv, seed, fraction, scheme)
            strain = data["strain_train"].astype(float)
            use = np.ones(len(strain), dtype=bool)
            if experiment == "labelmatched":
                use = np.zeros(len(strain), dtype=bool)
                use[data["labeled"]] = True
                use |= isotropic_mask(strain)
            constants = fit_constants(strain[use], data["cbm_train"][use].astype(float),
                                      data["vbm_train"][use].astype(float), seed)
            if scheme == "group_shear":
                # no training state carries shear, so the shear constants are not determined by the data
                constants["XI_UP"] = 0.0
                constants["D_BP"] = 0.0
                constants["DELTA_X"] = 0.0
            with open(path, "w") as f:
                json.dump(constants, f, indent=1)
            print("constants for seed %d (%s %s) fitted on %d states (%.0f s)"
                  % (seed, experiment, scheme or "%d%%" % fraction, int(use.sum()), time.time() - start), flush=True)


def evaluate(data_csv, out, seeds, experiment="main", fractions=None):
    """Test errors of the analytic models alone (exact minimum and largest eigenvalue), one row per seed and setting."""
    if experiment == "main":
        settings = [(100, None)]
    else:
        settings = settings_of(experiment, fractions)
    names = CONSTANTS["constant_names"] + ["DELTA_X"]
    rows = []
    for seed in seeds:
        for fraction, scheme in settings:
            constants = read_constants(out, seed, experiment, fraction, scheme)
            th = np.array([constants[name] for name in names])
            data = load_and_split(data_csv, seed, fraction, scheme)
            test = data["index_test"]
            strain = data["table"][DATA["strain_columns"]].values[test]      # float64, as read from the dataset
            prediction = gap_numpy(strain, th)
            truth = data["table"]["Eg_eV"].values[test]
            error = prediction - truth
            row = {"experiment": experiment, "scheme": scheme or "", "seed": seed, "fraction": fraction,
                   "n_fit": "", "test_mae_meV": np.abs(error).mean() * 1000,
                   "test_rmse_meV": np.sqrt((error ** 2).mean()) * 1000, "test_max_meV": np.abs(error).max() * 1000,
                   "test_r2": 1.0 - (error ** 2).sum() / ((truth - truth.mean()) ** 2).sum()}
            classes = data["classes"][test]
            for class_name in DATA["strain_classes"]:
                in_class = classes == class_name
                if in_class.any():
                    row["test_mae_" + class_name + "_meV"] = np.abs(error[in_class]).mean() * 1000
            rows.append(row)
    table = pd.DataFrame(rows).drop(columns=["n_fit"])
    path = os.path.join(out, "analytic_%s.csv" % experiment)
    table.to_csv(path, index=False)
    group = ["scheme"] if experiment == "heldout" else ["fraction"]
    print(table.groupby(group).test_mae_meV.agg(["mean", "std", "median"]).round(1).to_string())
    print("wrote", path)


# ======================================================================================================================
# COMMAND LINE
# ======================================================================================================================
def parse_int_list(text):
    """'42-46' -> [42, 43, 44, 45, 46];  '0,1,5' -> [0, 1, 5]."""
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part == "":
            continue
        if "-" in part:
            first, last = part.split("-")
            values.extend(range(int(first), int(last) + 1))
        else:
            values.append(int(part))
    return values


def main():
    parser = argparse.ArgumentParser(description="analytic models alone (see the top of this file)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["constants", "evaluate", "study"]:
        p = commands.add_parser(name)
        p.add_argument("--data_csv", required=True,
                       help="the dataset (columns E_xx ... E_zx, Eg_eV, CBM_eV, VBM_eV, strain_type)")
        p.add_argument("--out", required=True, help="study folder")
        p.add_argument("--experiment", default="main", choices=EXPERIMENT_NAMES)
        p.add_argument("--seeds", default="%d-%d" % (STUDY["first_seed"], STUDY["last_seed"]))
        p.add_argument("--fractions", default=None, help="labeled fractions (labelmatched); default: config.toml")
    args = parser.parse_args()
    seeds = parse_int_list(args.seeds)
    fractions = parse_int_list(args.fractions) if args.fractions else None
    if args.command in ("constants", "study"):
        fit_all_constants(args.data_csv, args.out, seeds, args.experiment, fractions)
    if args.command in ("evaluate", "study"):
        evaluate(args.data_csv, args.out, seeds, args.experiment, fractions)


if __name__ == "__main__":
    main()
