#!/usr/bin/env python3
"""
Conventional regression: Gaussian process regression, random forest and histogram gradient boosting.

Each method predicts the gap directly from the six standardized strain components, trained only on the labeled
states, with exactly the division, input scaling and labeled states of the networks (seeds 42-58, labeled fractions
1-100%; 0% has no labeled state). The Gaussian process was the most accurate of the three at every fraction.

Name in the paper: "Gaussian process", "random forest", "gradient boosting". Campaign script: classical_baselines.py.
This file reproduces its results exactly.

Commands (python conventional.py <command> -h shows the options):
  run      one seed, every fraction and method          -> <out>/runs/seed<seed>.csv
  study    every seed on this computer                  -> <out>/runs.csv
"""

# ======================================================================================================================
# LIBRARIES
# ======================================================================================================================
import argparse
import itertools
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ======================================================================================================================
# FUNCTIONS IN THIS FILE (in the order they appear)
# ======================================================================================================================
# Settings
#   read_config(path, *sections)                   the named sections of config.toml
#   cubic_point_group()                            the 48 cubic symmetry operations (used by the division)
# Data
#   strain_class_of(name)                          isotropic / uniaxial / biaxial / triaxial / shear
#   cubic_images(strain)                           the 48 cubic-symmetry images of one strain state
#   load_and_split(data_csv, seed, fraction)       division, input scaling, labeled subset
# Methods
#   build_methods(seed, n_labeled)                 the three regressors with the settings of config.toml
# Study
#   run_seed(data_csv, out, seed)                  every fraction and method for one seed
#   run_study(data_csv, out, seeds, jobs)          every seed, then one table of all runs
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


# the settings file: config.toml next to this file, or another one given with CONVENTIONAL_CONFIG=<path>
CONFIG_FILE = os.environ.get("CONVENTIONAL_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml"))
STUDY, DATA, GP, FOREST, BOOSTING = read_config(CONFIG_FILE, "study", "data", "gaussian_process", "random_forest",
                                                "gradient_boosting")
CUBIC_OPERATIONS = cubic_point_group()


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


def load_and_split(data_csv, seed, fraction):
    """Read the dataset; divide it into training, validation and test states (symmetry-grouped division); scale the
    inputs on the training states only; draw the labeled subset."""
    table = pd.read_csv(data_csv)
    strain = table[DATA["strain_columns"]].values.astype(np.float32)
    gap = table["Eg_eV"].values.astype(np.float32)
    cbm = table["CBM_eV"].values.astype(np.float32)
    vbm = table["VBM_eV"].values.astype(np.float32)
    classes = np.array([strain_class_of(name) for name in table["strain_type"].values])
    n_states = len(table)
    all_index = np.arange(n_states)

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

    # 3. input scaling (mean 0, standard deviation 1), fitted on the training states only
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


# ======================================================================================================================
# METHODS
# ======================================================================================================================
def build_methods(seed, n_labeled):
    """The three regressors. Trees need at least two labeled states to split; the Gaussian process does not."""
    kernel = (ConstantKernel(GP["amplitude_start"], tuple(GP["amplitude_bounds"]))
              * RBF(length_scale=GP["length_scale_start"], length_scale_bounds=tuple(GP["length_scale_bounds"]))
              + WhiteKernel(noise_level=GP["noise_start"], noise_level_bounds=tuple(GP["noise_bounds"])))
    methods = {
        "GaussianProcess": GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed,
                                                    n_restarts_optimizer=GP["restarts"]),
        "RandomForest": RandomForestRegressor(n_estimators=FOREST["trees"], random_state=seed, n_jobs=FOREST["jobs"]),
        "GBDT": HistGradientBoostingRegressor(random_state=seed, max_iter=BOOSTING["trees"],
                                              learning_rate=BOOSTING["learning_rate"]),
    }
    if n_labeled < 2:
        methods.pop("RandomForest")
        methods.pop("GBDT")
    return methods


# ======================================================================================================================
# STUDY
# ======================================================================================================================
def run_seed(data_csv, out, seed):
    """Every labeled fraction and method for one seed; returns the rows and writes <out>/runs/seed<seed>.csv."""
    rows = []
    for fraction in STUDY["fractions"]:
        data = load_and_split(data_csv, seed, fraction)
        x_labeled = data["x_train"][data["labeled"]].astype(np.float64)
        y_labeled = data["gap_train"][data["labeled"]].astype(np.float64)
        x_test = data["x_test"].astype(np.float64)
        y_test = data["gap_test"].astype(np.float64)
        test_classes = data["classes"][data["index_test"]]
        for name, method in build_methods(seed, len(y_labeled)).items():
            try:
                method.fit(x_labeled, y_labeled)
                prediction = method.predict(x_test)
            except Exception as error:
                print("  ! %s seed %d, %d%%: %s" % (name, seed, fraction, error))
                continue
            size = np.abs(prediction - y_test)
            row = {"model": name, "seed": seed, "dft_pct": fraction, "n_labeled": len(y_labeled),
                   "test_mae_meV": 1000 * size.mean(),
                   "test_r2": 1 - ((y_test - prediction) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()}
            for class_name in DATA["strain_classes"]:
                in_class = test_classes == class_name
                row["mae_%s_meV" % class_name] = 1000 * size[in_class].mean() if in_class.any() else np.nan
            rows.append(row)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "runs", "seed%d.csv" % seed), index=False)
    print("seed %d done" % seed, flush=True)
    return rows


def run_study(data_csv, out, seeds, jobs):
    """Every seed (`jobs` at a time), then one table of all runs and the mean test MAE per method and fraction."""
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(jobs) as pool:
        results = list(pool.map(run_seed, [data_csv] * len(seeds), [out] * len(seeds), seeds))
    rows = []
    for result in results:
        rows.extend(result)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "runs.csv"), index=False)
    print(table.pivot_table(index="dft_pct", columns="model", values="test_mae_meV", aggfunc="mean").round(1).to_string())


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
    parser = argparse.ArgumentParser(description="conventional regression (see the top of this file)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "study"]:
        p = commands.add_parser(name)
        p.add_argument("--data_csv", required=True,
                       help="the dataset (columns E_xx ... E_zx, Eg_eV, CBM_eV, VBM_eV, strain_type)")
        p.add_argument("--out", required=True, help="study folder")
        if name == "run":
            p.add_argument("--seed", type=int, required=True)
        else:
            p.add_argument("--seeds", default="%d-%d" % (STUDY["first_seed"], STUDY["last_seed"]))
            p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="seeds at the same time")
    args = parser.parse_args()
    if args.command == "run":
        run_seed(args.data_csv, args.out, args.seed)
    else:
        run_study(args.data_csv, args.out, parse_int_list(args.seeds), args.jobs)


if __name__ == "__main__":
    main()
