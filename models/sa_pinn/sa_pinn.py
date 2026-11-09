#!/usr/bin/env python3
"""
SA-PINN: band edges of strained diamond from the six strain components, with the analytic models in the loss.

A fully connected network maps the six strain components to the conduction-band minimum CBM and the valence-band
maximum VBM; gap = CBM - VBM. Physics enters only through the training objective: besides the labeled states and the
isotropic gaps, the network must agree with the analytic band-edge models (deformation potentials for the CBM,
Bir-Pikus for the VBM; constants theta0 fitted to the training states) on the unlabeled training states, keep the band
edges of the unstrained state at the calibrated unstrained values, and keep the slopes dE/dI1 of the states with
little volume change at those of the analytic models. All terms are squared deviations.
The self-adaptive scheme (McClenny & Braga-Neto, J. Comput. Phys. 2023): every state of every physics term has
its own weight lambda, trained by gradient ascent on the loss (Adam) while the network descends it. A physics term is
the mean of lambda^2 e^2; a data term is the mean of e^2 (not weighted).

Name in the paper: "SA-PINN". Campaign key: sa_pinn (03_train_pinn.py --model sa_paper --slope_fix).
This file reproduces those runs exactly (checked iteration by iteration).

Experiments (--experiment):
  main          symmetry-grouped division; labeled fractions 0-100%; constants fitted on each run's training portion
  heldout       100% labeled; one strain class left out, or every state beyond a strain magnitude; constants refitted
                on the reduced training portion

Commands (python sa_pinn.py <command> -h shows the options):
  constants   fit theta0 for every seed of one experiment      -> <out>/constants/...
  run         one training run                                -> <out>/runs/...
  study       constants + every run of one experiment on this computer
  tasks       task list for a cluster, one line per run        -> <out>/sa_pinn_<experiment>.tsv
"""

# ======================================================================================================================
# LIBRARIES
# ======================================================================================================================
import argparse
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================================================================================================
# FUNCTIONS IN THIS FILE (in the order they appear)
# ======================================================================================================================
# Settings
#   read_config(path, *sections)                   the named sections of config.toml
#   cubic_point_group()                            the 48 cubic symmetry operations
# Analytic band-edge models
#   shear_shift, valley_energies, bir_pikus_matrix, cbm_analytic, vbm_analytic
# Fit of the constants theta0
#   cbm_numpy, vbm_numpy, gap_numpy, fit_once, fit_constants
# Data
#   strain_class_of, cubic_images, heldout_schemes, load_and_split, isotropic_mask
# Optimizer
#   class SOAP, make_soap(parameters)
# Network
#   class PINN                                     the network
# Weighting
#   class SelfAdaptiveWeights                      the per-state weights and their gradient ascent
# Training
#   prepare_tensors(data, theta0, delta_x)         everything the loss needs, as tensors
#   residual_terms(model, t)                       the deviations of every loss term
#   compute_loss(model, t, weighting, update)      weighted sum of the terms
#   gap_mae(model, x, gap_true)                    mean absolute gap error (eV)
#   train_one_run(data_csv, out, seed, fraction, experiment, scheme, n_iterations, threads)
#   save_predictions(...), save_summary(...)
# Study driver
#   run_folder(...), constants_file(...), read_constants(...), settings_of(...)
#   fit_all_constants(data_csv, out, seeds, experiment)
#   run_arguments(...), write_task_list(...), run_study_locally(...)
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


MODEL_NAME = "sa_pinn"
EXPERIMENT_NAMES = ["main", "heldout"]                     # experiments this model runs (see the top of the file)
# the settings file: config.toml next to this file, or another one given with SA_PINN_CONFIG=<path>
CONFIG_FILE = os.environ.get("SA_PINN_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml"))
STUDY, DATA, EXPERIMENTS, NETWORK, TRAINING, LOSS, WEIGHTING, PHYSICS, FIT = read_config(
    CONFIG_FILE, "study", "data", "experiments", "network", "training", "loss", "weighting", "physics", "fit")
CUBIC_OPERATIONS = cubic_point_group()


# ======================================================================================================================
# ANALYTIC BAND-EDGE MODELS
# ======================================================================================================================
def shear_shift(e, xi_up, delta_x, xp):
    """Shift of a pair of conduction valleys under the shear component e (two-band form with valley gap delta_x).
    xp is the array library: torch (training) or numpy (constants fit)."""
    eta = 4.0 * xi_up * e / delta_x
    size = xp.abs(eta)
    return xp.where(size < 1.0, -0.25 * delta_x * eta ** 2, -0.25 * delta_x * (2.0 * size - 1.0))


def valley_energies(strain, xi_d, xi_u, xi_up, xi_q, xi_c, xi_q2, delta_x, xp):
    """Energies of the six conduction valleys relative to CBM_0, one column per valley."""
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
        # one-band shear term: used inside the constants fit before DELTA_X is added, and when shear is left out
        columns = [valley(e_zz) + 2 * xi_up * e_xy, valley(e_zz) - 2 * xi_up * e_xy,
                   valley(e_yy) + 2 * xi_up * e_zx, valley(e_yy) - 2 * xi_up * e_zx,
                   valley(e_xx) + 2 * xi_up * e_yz, valley(e_xx) - 2 * xi_up * e_yz]
    else:
        z_pair = valley(e_zz) + shear_shift(e_xy, xi_up, delta_x, xp)
        y_pair = valley(e_yy) + shear_shift(e_zx, xi_up, delta_x, xp)
        x_pair = valley(e_xx) + shear_shift(e_yz, xi_up, delta_x, xp)
        columns = [z_pair, z_pair, y_pair, y_pair, x_pair, x_pair]

    if xp is torch:
        return torch.stack(columns, dim=1)
    return np.stack(columns, axis=1)


def bir_pikus_matrix(strain, a_v, a_v2, b, b2, d):
    """3x3 Bir-Pikus matrix of the three top valence bands, one matrix per strain state."""
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
    H = torch.zeros(strain.shape[0], 3, 3, dtype=strain.dtype, device=strain.device)
    H[:, 0, 0] = hydrostatic + (b + b2 * dev_xx) * dev_xx
    H[:, 1, 1] = hydrostatic + (b + b2 * dev_yy) * dev_yy
    H[:, 2, 2] = hydrostatic + (b + b2 * dev_zz) * dev_zz
    H[:, 0, 1] = d * e_xy
    H[:, 1, 0] = d * e_xy
    H[:, 1, 2] = d * e_yz
    H[:, 2, 1] = d * e_yz
    H[:, 0, 2] = d * e_zx
    H[:, 2, 0] = d * e_zx
    return H


def cbm_analytic(strain, c, delta_x):
    """CBM with the fixed constants c (13 plain numbers): soft minimum over the six valleys."""
    valleys = valley_energies(strain, c[1], c[2], c[3], c[4], c[5], c[6], delta_x, torch)
    return c[0] + (-torch.logsumexp(-PHYSICS["beta_cbm"] * valleys, dim=1) / PHYSICS["beta_cbm"])


def vbm_analytic(strain, c):
    """VBM with the fixed constants c: exact largest eigenvalue of the Bir-Pikus matrix."""
    H = bir_pikus_matrix(strain, c[8], c[9], c[10], c[11], c[12])
    H += PHYSICS["eigen_eps"] * torch.eye(3, dtype=strain.dtype, device=strain.device).unsqueeze(0)
    return c[7] + torch.linalg.eigvalsh(H)[:, 2]


# ======================================================================================================================
# FIT OF THE CONSTANTS THETA0
# Joint least squares on the CBM, VBM and gap of the training states; a ridge penalty on the five second-order
# constants with its strength chosen by 5-fold cross-validation; several starts for d and for DELTA_X.
# ======================================================================================================================
def cbm_numpy(strain, th):
    """CBM of the fit: exact minimum over the valleys (th has 13 values, or 14 with DELTA_X last)."""
    if len(th) > 13:
        delta_x = th[13]
    else:
        delta_x = 0.0
    valleys = valley_energies(strain, th[1], th[2], th[3], th[4], th[5], th[6], delta_x, np)
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
    ridge_index = [NETWORK["constant_names"].index(name) for name in FIT["ridge_constants"]]
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
        start = np.array([FIT["start"][name] for name in NETWORK["constant_names"]])
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
    for name, value in zip(NETWORK["constant_names"] + ["DELTA_X"], th):
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
# OPTIMIZER: SOAP (exact copy of the project's soap.py; https://arxiv.org/abs/2409.11321; parts adapted from
# PyTorch's AdamW and from GaLore's projector). Adam run in the eigenbasis of Shampoo's preconditioner.
# ======================================================================================================================
class SOAP(optim.Optimizer):
    """
    lr                      learning rate
    betas                   Adam's (b1, b2)
    shampoo_beta            moving average of the preconditioner; < 0 uses b2
    eps                     Adam's epsilon
    weight_decay            decoupled weight decay (as AdamW)
    precondition_frequency  iterations between refreshes of the eigenbasis
    max_precond_dim         largest preconditioned dimension
    merge_dims              merge dimensions until their product is <= max_precond_dim
    precondition_1d         also precondition 1-D parameters (biases)
    normalize_grads         normalise the update of each layer
    data_format             'channels_first' or 'channels_last' (for 4-D tensors)
    correct_bias            Adam's bias correction
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1, eps=1e-8, weight_decay=0.01,
                 precondition_frequency=10, max_precond_dim=10000, merge_dims=False, precondition_1d=False,
                 normalize_grads=False, data_format="channels_first", correct_bias=True):
        defaults = {"lr": lr, "betas": betas, "shampoo_beta": shampoo_beta, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """Merges dimensions of the gradient tensor until the product of the dimensions is <= max_precond_dim."""
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        new_grad = grad.reshape(new_shape)
        return new_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        if closure is None:
            loss = None
        else:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)          # moving average of the gradient
                    state["exp_avg_sq"] = torch.zeros_like(grad)       # moving average of the squared gradient
                if "Q" not in state:
                    self.init_preconditioner(grad, state, precondition_frequency=group["precondition_frequency"],
                                             precondition_1d=group["precondition_1d"],
                                             shampoo_beta=(group["shampoo_beta"] if group["shampoo_beta"] >= 0
                                                           else group["betas"][1]),
                                             max_precond_dim=group["max_precond_dim"], merge_dims=group["merge_dims"])
                    self.update_preconditioner(grad, state, max_precond_dim=group["max_precond_dim"],
                                               merge_dims=group["merge_dims"], precondition_1d=group["precondition_1d"])
                    continue    # first step is skipped so that the current gradient is never used in the projection

                # project the gradient to the eigenbases of Shampoo's preconditioner
                grad_projected = self.project(grad, state, merge_dims=group["merge_dims"],
                                              max_precond_dim=group["max_precond_dim"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                exp_avg_projected = exp_avg
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2 ** .5) / bias_correction1

                # project the Adam-preconditioned average back to the original space
                norm_grad = self.project_back(exp_avg_projected / denom, state, merge_dims=group["merge_dims"],
                                              max_precond_dim=group["max_precond_dim"])
                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)
                p.add_(norm_grad, alpha=-step_size)

                # decoupled weight decay, as in AdamW
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # update after the step, so the current gradient is not used in the projection
                self.update_preconditioner(grad, state, max_precond_dim=group["max_precond_dim"],
                                           merge_dims=group["merge_dims"], precondition_1d=group["precondition_1d"])
        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10, shampoo_beta=0.95, max_precond_dim=10000,
                            precondition_1d=False, merge_dims=False):
        """Initializes the preconditioner matrices (L and R in the paper)."""
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Projects the gradient to the eigenbases of the preconditioner."""
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(self, grad, state, max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        """Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper)."""
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(state["exp_avg"], state, merge_dims=merge_dims,
                                                 max_precond_dim=max_precond_dim)
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"])
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        other_dims = [*itertools.chain(range(idx), range(idx + 1, len(new_grad.shape)))]
                        outer_product = torch.tensordot(new_grad, new_grad, dims=[other_dims] * 2)
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        other_dims = [*itertools.chain(range(idx), range(idx + 1, len(grad.shape)))]
                        outer_product = torch.tensordot(grad, grad, dims=[other_dims] * 2)
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
        if state["step"] > 0:
            state["exp_avg"] = self.project(state["exp_avg"], state, merge_dims=merge_dims,
                                            max_precond_dim=max_precond_dim)

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Projects the gradient back to the original space."""
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        """Computes the eigenbases of the preconditioner with torch.linalg.eigh."""
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)
        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except Exception:
                _, Q = torch.linalg.eigh(m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """Computes the eigenbases of the preconditioner by one power iteration followed by a QR decomposition."""
        precond_list = state["GG"]
        orth_list = state["Q"]
        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]
        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)
        state["exp_avg_sq"] = exp_avg_sq
        return final


def make_soap(parameters):
    """SOAP with the settings of training phase 2."""
    return SOAP(parameters, lr=TRAINING["soap_lr"], betas=tuple(TRAINING["soap_betas"]), weight_decay=TRAINING["soap_weight_decay"],
                precondition_frequency=TRAINING["soap_precondition_frequency"])


# ======================================================================================================================
# NETWORK
# ======================================================================================================================
class PINN(nn.Module):
    """6 scaled strains -> 8 hidden layers of 64 SiLU neurons -> 2 outputs: CBM and VBM (eV).
    The output biases start at the mean band edges of the training states."""

    def __init__(self, cbm_mean, vbm_mean):
        super().__init__()
        layer_sizes = [len(DATA["strain_columns"])] + [NETWORK["neurons"]] * NETWORK["hidden_layers"]
        self.hidden = nn.ModuleList()
        for i in range(NETWORK["hidden_layers"]):
            self.hidden.append(nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.SiLU()))
        self.out = nn.Linear(NETWORK["neurons"], NETWORK["outputs"])

        # Xavier initialisation (in the same order as the paper's trainer, so the random-number stream is the same)
        for layer in self.hidden:
            nn.init.xavier_normal_(layer[0].weight)
            nn.init.zeros_(layer[0].bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[0] = cbm_mean
            self.out.bias[1] = vbm_mean

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = layer(h)
        o = self.out(h)
        cbm = o[:, 0]
        vbm = o[:, 1]
        return cbm, vbm, cbm - vbm


# ======================================================================================================================
# WEIGHTING: SELF-ADAPTIVE WEIGHTS (SA-PINN)
# ======================================================================================================================
class SelfAdaptiveWeights:
    """One weight lambda per state on every physics term, started uniformly at random in [0, 1) and raised by gradient
    ascent on the loss after every network step. Physics term: mean of lambda^2 e^2; data term: mean of e^2."""

    def __init__(self):
        self.weights = {}          # term name -> lambda (one per state), created at the first loss evaluation
        self.optimizer = None      # Adam on the weights, created after the first network step

    def term_losses(self, model, terms, update):
        losses = {}
        for name, (e, base_weight, physics_term) in terms.items():
            if physics_term and name not in self.weights:
                self.weights[name] = nn.Parameter(torch.rand(e.shape[0]))
            if physics_term:
                losses[name] = base_weight * (self.weights[name] ** 2 * e ** 2).mean()
            else:
                losses[name] = base_weight * (e ** 2).mean()
        return losses

    def after_network_step(self, model, t):
        """Gradient ascent on the weights: maximise the same loss at the updated network."""
        if self.optimizer is None:
            self.optimizer = optim.Adam(list(self.weights.values()), lr=WEIGHTING["ascent_lr"])
        self.optimizer.zero_grad()
        loss, _ = compute_loss(model, t, self, update=False)
        (-loss).backward()
        self.optimizer.step()

    def save(self, folder):
        rows = []
        for name, value in self.weights.items():
            v = value.detach()
            rows.append({"term": name, "n": int(v.numel()), "mean": float(v.mean()), "min": float(v.min()),
                         "max": float(v.max())})
        pd.DataFrame(rows).to_csv(os.path.join(folder, "term_weights.csv"), index=False)


WEIGHTING_SCHEME = SelfAdaptiveWeights


# ======================================================================================================================
# TRAINING
# ======================================================================================================================
def prepare_tensors(data, theta0, delta_x):
    """Everything the loss needs, as tensors (built once per run)."""
    t = {}
    t["x_train"] = torch.tensor(data["x_train"], dtype=torch.float32)
    t["x_val"] = torch.tensor(data["x_val"], dtype=torch.float32)
    t["x_test"] = torch.tensor(data["x_test"], dtype=torch.float32)
    t["strain_train"] = torch.tensor(data["strain_train"], dtype=torch.float32)
    t["gap_train"] = torch.tensor(data["gap_train"], dtype=torch.float32)
    t["gap_val"] = torch.tensor(data["gap_val"], dtype=torch.float32)
    t["gap_test"] = torch.tensor(data["gap_test"], dtype=torch.float32)

    # labeled states
    t["n_labeled"] = len(data["labeled"])
    t["labeled"] = torch.tensor(data["labeled"], dtype=torch.long)
    t["cbm_labeled"] = torch.tensor(data["cbm_train"][data["labeled"]], dtype=torch.float32)
    t["vbm_labeled"] = torch.tensor(data["vbm_train"][data["labeled"]], dtype=torch.float32)

    # isotropic training states (no shear, equal diagonal): the analytic models are not trusted there, so the
    # agreement terms skip them and their DFT gaps are used directly
    strain = t["strain_train"]
    shear_size = strain[:, 3].abs() + strain[:, 4].abs() + strain[:, 5].abs()
    diagonal = strain[:, :3]
    diagonal_spread = (diagonal - diagonal.mean(dim=1, keepdim=True)).abs().max(dim=1).values
    t["isotropic"] = (shear_size < DATA["isotropic_tolerance"]) & (diagonal_spread < DATA["isotropic_tolerance"])
    t["not_isotropic"] = ~t["isotropic"]

    # targets of the agreement terms: the analytic models with the fixed constants theta0
    with torch.no_grad():
        t["cbm_analytic"] = cbm_analytic(strain, theta0, delta_x)
        t["vbm_analytic"] = vbm_analytic(strain, theta0)
        t["gap_analytic"] = t["cbm_analytic"] - t["vbm_analytic"]

    # zero-strain terms: anchor states (volume change |I1| below anchor_tolerance; in this dataset only the unstrained
    # state, when it is in the training portion) hold the band edges at the calibrated unstrained values; slope states
    # (|I1| below slope_tolerance; most of them sheared) hold the slopes dE/dI1 at those of the analytic models
    volume_change = data["strain_train"][:, 0] + data["strain_train"][:, 1] + data["strain_train"][:, 2]
    t["x_anchor"] = t["x_train"][np.abs(volume_change) < LOSS["anchor_tolerance"]]
    t["x_slope"] = t["x_train"][np.abs(volume_change) < LOSS["slope_tolerance"]]
    t["cbm_0"] = theta0[0]
    t["vbm_0"] = theta0[7]
    t["slope_cbm"] = theta0[1] + theta0[2] / 3.0         # d CBM / d I1 at zero strain: XI_D + XI_U / 3
    t["slope_vbm"] = theta0[8]                           # d VBM / d I1 at zero strain: a_v
    t["input_scale_normal"] = data["scaler"].scale_[:3].astype(np.float32)   # scales of e_xx, e_yy, e_zz
    return t


def residual_terms(model, t):
    """The deviations e of every loss term, in a fixed order: name -> (deviations, base weight, physics term?).
    Terms with base weight 0 are left out."""
    cbm, vbm, gap = model(t["x_train"])
    terms = {}

    # agreement with the analytic models (non-isotropic training states)
    other = t["not_isotropic"]
    if other.sum() > 0:
        terms["cbm"] = (cbm[other] - t["cbm_analytic"][other], LOSS["w_cbm"], True)
        terms["vbm"] = (vbm[other] - t["vbm_analytic"][other], LOSS["w_vbm"], True)
        terms["gap"] = (gap[other] - t["gap_analytic"][other], LOSS["w_gap"], True)

    # anchor: band edges of the anchor states at the calibrated unstrained values
    if t["x_anchor"].shape[0] > 0:
        cbm_anchor, vbm_anchor, _ = model(t["x_anchor"])
        terms["anchor_cbm"] = (cbm_anchor - t["cbm_0"], LOSS["w_anchor"], True)
        terms["anchor_vbm"] = (vbm_anchor - t["vbm_0"], LOSS["w_anchor"], True)

    # slopes dE/dI1 at the slope states, by automatic differentiation through the input scaling: the sum of the three
    # normal-strain partials is the derivative along (t, t, t), that is 3 dE/dI1
    x_slope = t["x_slope"].detach().requires_grad_(True)
    cbm_slope, vbm_slope, _ = model(x_slope)
    grad_cbm = torch.autograd.grad(cbm_slope.sum(), x_slope, create_graph=True)[0]
    grad_vbm = torch.autograd.grad(vbm_slope.sum(), x_slope, create_graph=True)[0]
    sx = t["input_scale_normal"][0]
    sy = t["input_scale_normal"][1]
    sz = t["input_scale_normal"][2]
    d_cbm = grad_cbm[:, 0] / sx + grad_cbm[:, 1] / sy + grad_cbm[:, 2] / sz
    d_vbm = grad_vbm[:, 0] / sx + grad_vbm[:, 1] / sy + grad_vbm[:, 2] / sz
    d_cbm = d_cbm / 3.0
    d_vbm = d_vbm / 3.0
    terms["slope_cbm"] = (d_cbm - t["slope_cbm"], LOSS["w_slope"], True)
    terms["slope_vbm"] = (d_vbm - t["slope_vbm"], LOSS["w_slope"], True)

    # data: CBM and VBM of the labeled states; isotropic gaps (always labeled)
    if t["n_labeled"] > 0:
        terms["data_cbm"] = (cbm[t["labeled"]] - t["cbm_labeled"], LOSS["w_data"], False)
        terms["data_vbm"] = (vbm[t["labeled"]] - t["vbm_labeled"], LOSS["w_data"], False)
    iso = t["isotropic"]
    if iso.sum() > 0:
        terms["isotropic"] = (gap[iso] - t["gap_train"][iso], LOSS["w_isotropic"], False)

    kept = {}
    for name, value in terms.items():
        if value[1] != 0:
            kept[name] = value
    return kept


def compute_loss(model, t, weighting, update):
    """Weighted sum of the squared-deviation terms (weights from the adaptive scheme) and the terms themselves.
    update = True only in the evaluation that precedes the network step."""
    terms = residual_terms(model, t)
    losses = weighting.term_losses(model, terms, update)
    zero = torch.tensor(0.0)

    def part(name):
        return losses.get(name, zero)

    # summed in the order of the paper's trainer, so the floating-point result is the same
    total = (part("cbm") + part("vbm") + part("gap") + (part("anchor_cbm") + part("anchor_vbm"))
             + part("slope_cbm") + part("slope_vbm") + (part("data_cbm") + part("data_vbm")) + part("isotropic") + zero)
    return total, losses


def gap_mae(model, x, gap_true):
    """Mean absolute gap error in eV (evaluation mode, no gradients)."""
    model.eval()
    with torch.no_grad():
        _, _, gap = model(x)
        error = float((gap - gap_true).abs().mean())
    model.train()
    return error


def train_one_run(data_csv, out, seed, fraction, experiment, scheme, n_iterations, threads):
    """Train the network for one seed and one setting; write log, predictions and summary to its run folder."""
    folder = run_folder(out, seed, fraction, experiment, scheme)
    os.makedirs(folder, exist_ok=True)
    shutil.copy(CONFIG_FILE, os.path.join(folder, "config_used.toml"))      # the settings of this run
    if threads > 0:
        torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    constants = read_constants(out, seed, experiment, fraction, scheme)
    theta0 = [float(constants[name]) for name in NETWORK["constant_names"]]
    delta_x = float(constants["DELTA_X"])
    data = load_and_split(data_csv, seed, fraction, scheme)
    t = prepare_tensors(data, theta0, delta_x)
    print("seed %d, %s %s: %d train, %d validation, %d test, %d labeled, %d anchor and %d slope states"
          % (seed, experiment, scheme or "%d%%" % fraction, len(data["index_train"]), len(data["index_val"]),
             len(data["index_test"]), t["n_labeled"], t["x_anchor"].shape[0], t["x_slope"].shape[0]), flush=True)

    model = PINN(float(np.mean(data["cbm_train"])), float(np.mean(data["vbm_train"])))
    weighting = WEIGHTING_SCHEME()
    switch_iteration = int(round(TRAINING["switch_fraction"] * n_iterations))
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING["adamw_lr"], weight_decay=TRAINING["adamw_weight_decay"],
                            betas=tuple(TRAINING["adamw_betas"]))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=TRAINING["restart_period_adamw"],
                                                               T_mult=TRAINING["restart_growth"], eta_min=TRAINING["lr_min"])
    best_val = float("inf")
    best_iteration = 0
    best_state = None
    n_clipped = 0
    log_rows = []
    start_time = time.time()

    for iteration in range(1, n_iterations + 1):
        if iteration == switch_iteration + 1:
            # phase 2: a fresh SOAP optimizer with its own schedule
            optimizer = make_soap(model.parameters())
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=TRAINING["restart_period_soap"],
                                                                       T_mult=TRAINING["restart_growth"], eta_min=TRAINING["lr_min"])
        model.train()
        optimizer.zero_grad()
        loss, terms = compute_loss(model, t, weighting, update=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), TRAINING["clip_norm"]))
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        weighting.after_network_step(model, t)
        if grad_norm > TRAINING["clip_norm"]:
            n_clipped += 1

        # checkpoint: keep the weights with the lowest validation gap MAE
        if iteration % TRAINING["checkpoint_every"] == 0:
            val = gap_mae(model, t["x_val"], t["gap_val"])
            if val < best_val:
                best_val = val
                best_iteration = iteration
                best_state = {}
                for name, value in model.state_dict().items():
                    best_state[name] = value.clone()

        # log (diagnostic only; nothing here changes the training)
        if iteration % TRAINING["log_every"] == 0 or iteration == 1:
            row = {"iteration": iteration, "loss": float(loss.detach())}
            for name, value in terms.items():
                row[name] = float(value.detach())
            row["lr"] = lr_used
            row["grad_norm"] = grad_norm
            row["train_mae"] = gap_mae(model, t["x_train"], t["gap_train"])
            row["val_mae"] = gap_mae(model, t["x_val"], t["gap_val"])
            row["test_mae"] = gap_mae(model, t["x_test"], t["gap_test"])
            log_rows.append(row)
        if iteration % 2000 == 0:
            print("  iteration %5d  lr %.2e  loss %.4f  best validation %.1f meV  (%.0f s)"
                  % (iteration, lr_used, float(loss.detach()), best_val * 1000, time.time() - start_time), flush=True)

    train_time = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(folder, "best_model.pt"))
    pd.DataFrame(log_rows).to_csv(os.path.join(folder, "log.csv"), index=False)
    weighting.save(folder)
    predictions = save_predictions(model, data, theta0, delta_x, folder)
    save_summary(predictions, seed, fraction, experiment, scheme, t, n_iterations, best_iteration, best_val,
                 n_clipped, train_time, folder)


def save_predictions(model, data, theta0, delta_x, folder):
    """Predictions of the kept model for every state, with the analytic models alone for comparison."""
    model.eval()
    x_all = torch.tensor((data["strain"] - data["scaler"].mean_) / data["scaler"].scale_, dtype=torch.float32)
    strain_all = torch.tensor(data["strain"], dtype=torch.float32)
    with torch.no_grad():
        cbm, vbm, gap = model(x_all)
        gap_analytic = cbm_analytic(strain_all, theta0, delta_x) - vbm_analytic(strain_all, theta0)

    p = pd.DataFrame()
    p["state"] = np.arange(len(gap))
    p["split"] = data["split"]
    p["strain_class"] = data["classes"]
    p["strain_type"] = data["table"]["strain_type"].values
    p["max_abs_strain"] = np.abs(data["strain"]).max(axis=1)
    p["gap_true"] = data["gap"]
    p["gap_pred"] = gap.numpy()
    p["cbm_true"] = data["cbm"]
    p["cbm_pred"] = cbm.numpy()
    p["vbm_true"] = data["vbm"]
    p["vbm_pred"] = vbm.numpy()
    p["gap_analytic"] = gap_analytic.numpy()                       # analytic models with theta0 (no network)
    p.to_csv(os.path.join(folder, "predictions.csv"), index=False)
    return p


def save_summary(p, seed, fraction, experiment, scheme, t, n_iterations, best_iteration, best_val, n_clipped,
                 train_time, folder):
    """One row of metrics. Written last: its presence marks the run as finished."""
    test = p[p["split"] == "test"]
    train = p[p["split"] == "train"]
    error = (test["gap_pred"] - test["gap_true"]).values
    total_variance = ((test["gap_true"] - test["gap_true"].mean()) ** 2).sum()
    s = {"model": MODEL_NAME, "experiment": experiment, "scheme": scheme or "", "seed": seed, "fraction": fraction,
         "n_labeled": t["n_labeled"], "n_iterations": n_iterations, "best_iteration": best_iteration,
         "test_mae_meV": np.abs(error).mean() * 1000,
         "test_rmse_meV": np.sqrt((error ** 2).mean()) * 1000,
         "test_max_meV": np.abs(error).max() * 1000,
         "test_r2": 1.0 - (error ** 2).sum() / total_variance,
         "test_cbm_mae_meV": np.abs(test["cbm_pred"] - test["cbm_true"]).mean() * 1000,
         "test_vbm_mae_meV": np.abs(test["vbm_pred"] - test["vbm_true"]).mean() * 1000,
         "val_mae_meV": best_val * 1000,
         "train_mae_meV": np.abs(train["gap_pred"] - train["gap_true"]).mean() * 1000,
         "clipped_share": n_clipped / n_iterations,
         "train_time_s": train_time}
    if "gap_analytic" in test.columns:
        s["analytic_test_mae_meV"] = np.abs(test["gap_analytic"] - test["gap_true"]).mean() * 1000
    for class_name in DATA["strain_classes"]:
        in_class = (test["strain_class"] == class_name).values
        if in_class.any():
            s["test_mae_" + class_name + "_meV"] = np.abs(error[in_class]).mean() * 1000
    pd.DataFrame([s]).to_csv(os.path.join(folder, "metrics_summary.csv"), index=False)
    print("seed %d, %s %s: test gap MAE %.2f meV, best checkpoint at iteration %d, %.0f s"
          % (seed, experiment, scheme or "%d%%" % fraction, s["test_mae_meV"], best_iteration, train_time), flush=True)


# ======================================================================================================================
# STUDY DRIVER
# ======================================================================================================================
def run_folder(out, seed, fraction, experiment="main", scheme=None):
    """main: runs/<fraction>pct/seed<s>;  heldout: runs/heldout/<scheme>/seed<s>;  labelmatched: runs/labelmatched/..."""
    if experiment == "main":
        return os.path.join(out, "runs", "%dpct" % fraction, "seed%d" % seed)
    if experiment == "heldout":
        return os.path.join(out, "runs", "heldout", scheme, "seed%d" % seed)
    return os.path.join(out, "runs", "labelmatched", "%dpct" % fraction, "seed%d" % seed)


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


def run_arguments(data_csv, out, seed, fraction, experiment, scheme, n_iterations, threads):
    arguments = ["run", "--data_csv", data_csv, "--out", out, "--experiment", experiment, "--seed", str(seed),
                 "--fraction", str(fraction), "--iterations", str(n_iterations), "--threads", str(threads)]
    if scheme is not None:
        arguments += ["--scheme", scheme]
    return arguments


def write_task_list(script, data_csv, out, seeds, experiment, fractions, n_iterations):
    """One line per run: <run folder> TAB <command>. Paths are written as given (relative paths stay relative)."""
    path = os.path.join(out, "%s_%s.tsv" % (MODEL_NAME, experiment))
    os.makedirs(out, exist_ok=True)
    n_runs = 0
    with open(path, "w") as f:
        for fraction, scheme in settings_of(experiment, fractions):
            for seed in seeds:
                arguments = run_arguments(data_csv, out, seed, fraction, experiment, scheme, n_iterations, 1)
                f.write("%s\tpython %s %s\n" % (run_folder(out, seed, fraction, experiment, scheme), script,
                                               " ".join(arguments)))
                n_runs += 1
    print("wrote %s (%d runs); fit the constants first with the 'constants' command" % (path, n_runs))


def run_study_locally(data_csv, out, seeds, experiment, fractions, n_iterations, jobs, threads):
    """Constants, then every unfinished run of the experiment (`jobs` at a time, each with `threads` threads)."""
    fit_all_constants(data_csv, out, seeds, experiment, fractions)
    todo = []
    for fraction, scheme in settings_of(experiment, fractions):
        for seed in seeds:
            if not os.path.exists(os.path.join(run_folder(out, seed, fraction, experiment, scheme), "metrics_summary.csv")):
                todo.append((seed, fraction, scheme))
    print("%d runs to do, %d at a time" % (len(todo), jobs), flush=True)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    running = []
    failed = []
    while len(todo) > 0 or len(running) > 0:
        while len(todo) > 0 and len(running) < jobs:
            seed, fraction, scheme = todo.pop(0)
            folder = run_folder(out, seed, fraction, experiment, scheme)
            os.makedirs(folder, exist_ok=True)
            command = [sys.executable, os.path.abspath(__file__)] + run_arguments(data_csv, out, seed, fraction,
                                                                                  experiment, scheme, n_iterations, threads)
            output = open(os.path.join(folder, "stdout.txt"), "w")
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=environment)
            running.append((process, seed, fraction, scheme))
        time.sleep(2)
        still_running = []
        for process, seed, fraction, scheme in running:
            if process.poll() is None:
                still_running.append((process, seed, fraction, scheme))
            else:
                print("  finished seed %d, %s (exit code %d)" % (seed, scheme or "%d%%" % fraction, process.returncode),
                      flush=True)
                if process.returncode != 0:
                    failed.append((seed, fraction, scheme))
        running = still_running
    if len(failed) > 0:
        print("FAILED runs (see stdout.txt in their folders):", failed)


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
    parser = argparse.ArgumentParser(description="%s model (see the top of this file)" % MODEL_NAME)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["constants", "run", "study", "tasks"]:
        p = commands.add_parser(name)
        p.add_argument("--data_csv", required=True,
                       help="the dataset (columns E_xx ... E_zx, Eg_eV, CBM_eV, VBM_eV, strain_type)")
        p.add_argument("--out", required=True, help="study folder")
        p.add_argument("--experiment", default="main", choices=EXPERIMENT_NAMES)
        if name == "run":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--fraction", type=int, default=100, help="percent of the training states with labels")
            p.add_argument("--scheme", default=None, help="held-out division (heldout experiment): " + ", ".join(heldout_schemes()))
        else:
            p.add_argument("--seeds", default="%d-%d" % (STUDY["first_seed"], STUDY["last_seed"]))
            p.add_argument("--fractions", default=None, help="labeled fractions (main, labelmatched); default: config.toml")
        if name in ("run", "study", "tasks"):
            p.add_argument("--iterations", type=int, default=TRAINING["iterations"])
        if name in ("run", "study"):
            p.add_argument("--threads", type=int, default=1, help="threads per run")
        if name == "study":
            p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="runs at the same time")
    args = parser.parse_args()

    if args.command == "run":
        if args.experiment == "heldout" and args.scheme is None:
            raise SystemExit("--experiment heldout needs --scheme")
        if args.experiment == "heldout":
            args.fraction = 100
        train_one_run(args.data_csv, args.out, args.seed, args.fraction, args.experiment, args.scheme,
                      args.iterations, args.threads)
        return
    fractions = parse_int_list(args.fractions) if args.fractions else None
    seeds = parse_int_list(args.seeds)
    if args.command == "constants":
        fit_all_constants(args.data_csv, args.out, seeds, args.experiment, fractions)
    elif args.command == "study":
        run_study_locally(args.data_csv, args.out, seeds, args.experiment, fractions, args.iterations, args.jobs,
                          args.threads)
    elif args.command == "tasks":
        write_task_list(sys.argv[0], args.data_csv, args.out, seeds, args.experiment, fractions, args.iterations)


if __name__ == "__main__":
    main()
