#!/usr/bin/env python3
"""
phm_ablations: the ablation and fix screen of PHM (the final model itself is in phm.py).
Naming (8 Nov 2026): this model was called PHM during the screen; it is now PHM. The earlier 13-output physics-head
model is now "PHM old"; it is not part of this screen.

"base" is PHM as in the paper (identical to 03_train_pinn.py, res_const_symlab_bal). Every other variant changes
one part of it (VARIANTS below); "final" is the combination the screen adopted, identical to phm.py.
Results of the screen: 10_independent_check/phm_screen/ (comparison/decisions.txt).

Commands (python phm_ablations.py <command> -h shows the options):
  constants   fit theta0 for every seed (and labeled fraction, where a variant needs it)  -> <out>/constants/
  run         one training run (one variant, seed and labeled fraction)       -> <out>/<variant>/<fraction>pct/seed<seed>/
  study       constants + every run on this computer + reports
  tasks       task list for a cluster, one line per run                       -> <out>/phm.tsv
  report      tables and figures of one variant                               -> <out>/<variant>/results.txt, figures/
  compare     every variant against base: tables, decisions, figures          -> <out>/comparison/
"""

# ======================================================================================================================
# LIBRARIES
# ======================================================================================================================
import argparse
import glob
import itertools
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")                     # draw into files; no screen needed
import matplotlib.pyplot as plt

# ======================================================================================================================
# FUNCTIONS IN THIS FILE (in the order they appear)
# ======================================================================================================================
# Analytic band-edge models
#   agreement_weight(fraction)                     weight 2 s of the gap-agreement loss term at a labeled fraction
#   shear_shift(e, xi_up, delta_x, xp)             shear splitting of a pair of conduction valleys
#   valley_energies(strain, ..., xp)               energies of the six conduction valleys
#   bir_pikus_matrix(strain, ...)                  3x3 Bir-Pikus matrix of the top valence bands
#   cbm_learned(strain, theta, delta_x)            CBM with per-state constants (soft minimum over the valleys)
#   vbm_learned(strain, theta)                     VBM with per-state constants (smooth maximum eigenvalue)
#   cbm_analytic(strain, c, delta_x)               CBM with the fixed constants theta0 (soft minimum)
#   vbm_analytic(strain, c)                        VBM with the fixed constants theta0 (exact maximum eigenvalue)
# Fit of the constants theta0
#   cbm_numpy(strain, th), vbm_numpy(strain, th), gap_numpy(strain, th)
#   fit_once(strain, cbm, vbm, ridge, start)       one least-squares fit
#   fit_constants(strain, cbm, vbm, seed)          the full fit (several starts, ridge by cross-validation)
# Data
#   strain_class_of(name)                          isotropic / uniaxial / biaxial / triaxial / shear
#   cubic_images(strain)                           the 48 cubic-symmetry images of one strain state
#   load_and_split(data_csv, seed, fraction)       symmetry split, input scaling, labeled subset
#   copied_labels(data, seed)                      cubic images of the labeled states, carrying their labels
# Optimizer
#   class SOAP                                     the SOAP optimizer (exact copy of the project's soap.py)
#   make_soap(parameters, lr)                      SOAP with the settings of training phase 2
# Network
#   options_for(variant)                           the options of one model variant
#   class PHM                                   the network and its physics output layer
# Training
#   absolute_error(difference, options)            |difference| (or its smooth version)
#   isotropic_mask(strain)                         training states under isotropic strain
#   hold_out_for_selection(data, isotropic, seed)  states taken out of training to choose the checkpoint
#   prepare_tensors(data, seed, theta0, delta_x, options)
#   compute_loss(model, t, options, w_agree)       the loss terms and their sum
#   gap_mae(model, x, gap_true)                    mean absolute gap error (eV)
#   gradient_norms(model)                          gradient norm of the hidden layers and of the output layer
#   symmetry_violation(model, data)                largest band-edge change over the 48 cubic images (test states)
#   train_one_run(data_csv, out, variant, seed, fraction, n_iterations, threads)
#   save_predictions(...), save_summary(...)
# Study driver
#   run_folder(...), constants_file(...), read_constants(...), fit_all_constants(...)
#   runs_in_study(...), run_command(...), write_task_list(...), run_study_locally(...)
# Report (one variant)
#   load_summaries, load_log, load_predictions, mean_and_ci, median_curve, write_tables
#   setup_matplotlib, log_axis, light_grid, save_figure, available, fraction_colors
#   plot_learning_curves, plot_loss_terms, plot_lr_and_gradients, plot_parity, plot_residuals, plot_error_ecdf,
#   plot_seed_spread, plot_per_class, plot_data_efficiency, plot_generalization_gap, plot_strain_sweeps,
#   plot_constants_vs_correction, make_report(data_csv, out, variant)
# Comparison of variants
#   paired_difference(...), compare_variants(out), write_decisions(...), plot_variant_differences(...),
#   plot_issue_diagnostics(...)
# Command line
#   parse_int_list(text), parse_variants(text), main()

# ======================================================================================================================
# CONSTANTS AND HYPERPARAMETERS
# ======================================================================================================================
# ---- study
SEEDS = list(range(42, 59))                       # 17 seeds, as in the paper
FRACTIONS = [0, 1, 2, 5, 10, 25, 50, 75, 100]     # percent of the training states that have DFT labels

# ---- data
STRAIN_COLUMNS = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]   # Green-Lagrange strain, Voigt order
STRAIN_CLASSES = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]
TEST_SHARE = 0.20                 # share of each strain class that goes to the test set
VALIDATION_SHARE = 0.10           # share of each strain class that goes to the validation set
SYMMETRY_DECIMALS = 5             # rounding used to recognise two states as cubic images of each other
IMAGE_DECIMALS = 7                # rounding used to recognise repeated images of one labeled state
LABEL_SEED_OFFSET = 9999          # random stream of the labeled subset: seed + this
COPY_SEED_OFFSET = 4241           # random stream of the copied-label subsample: seed + this
MAX_COPIED_LABELS = 2000          # at most this many copied symmetry images are used
ISOTROPIC_TOLERANCE = 1e-4        # a state is isotropic when shear and diagonal spread are below this

# ---- network
N_INPUTS = 6                      # the six strain components (scaled)
N_HIDDEN_LAYERS = 8
N_NEURONS = 64                    # per hidden layer, SiLU activation
N_CORRECTIONS = 2                 # dCBM, dVBM
N_CONSTANTS = 13                  # changes to the 13 analytic constants
CONSTANT_STEP = 0.25              # theta = theta0 + 0.25 * max(|theta0|, 1) * network output
D_BP_STEP = 2.0                   # fixed step for d, whose fitted value is close to 0

# ---- training
N_ITERATIONS = 24000              # full-batch iterations
SWITCH_FRACTION = 0.5             # AdamW for the first half, SOAP for the second
ADAMW_LR = 2.636e-3
ADAMW_WEIGHT_DECAY = 1.793e-5
ADAMW_BETAS = (0.9, 0.999)
SOAP_LR = 7.325e-3
SOAP_WEIGHT_DECAY = 1.224e-3
SOAP_BETAS = (0.95, 0.95)
SOAP_PRECONDITION_FREQUENCY = 10
RESTART_PERIOD_ADAMW = 500        # cosine schedule with warm restarts: first period (iterations)
RESTART_PERIOD_SOAP = 1000
RESTART_GROWTH = 2                # each period twice as long as the one before
LR_MIN = 1e-5
CLIP_NORM = 1.0                   # gradient-norm clipping
CHECKPOINT_EVERY = 250            # validation check; the best checkpoint is kept
LOG_EVERY = 10                    # iterations between rows of log.csv

# ---- loss
W_DATA = 5.0                      # CBM and VBM of the labeled states (and their copied images)
W_ISOTROPIC = 5.0                 # gap of the isotropic training states
W_SYMMETRY = 1.0                  # same band edges for a strain state and a random cubic image of it
AGREE_N_FULL = 9                  # gap agreement has full weight up to this many labeled states, then (9/n)^3
N_TRAIN_NOMINAL = 903             # nominal training-set size used in the agreement rule (as in the paper's runs)

# ---- analytic band-edge models
BETA_CBM = 50.0                   # sharpness of the soft minimum over the conduction valleys (1/eV)
BETA_VBM = 200.0                  # sharpness of the smooth maximum over the valence bands (1/eV)
EIGEN_EPS = 1e-8                  # added to the Bir-Pikus diagonal before the exact eigenvalues
CONSTANT_NAMES = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2",       # conduction band (7)
                  "VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]                    # valence band (6)
CONSTANT_SYMBOLS = ["CBM$_0$", r"$\Xi_d$", r"$\Xi_u$", r"$\Xi_u'$", r"$\Xi_q$", r"$\Xi_c$", r"$\Xi_{q2}$",
                    "VBM$_0$", "$a_v$", "$a_{v2}$", "$b$", "$b_2$", "$d$"]

# ---- fit of theta0
CONSTANT_START = {"CBM_0": 14.534980535287007, "XI_D": -24.84604763603144, "XI_U": 24.328031070892287,
                  "XI_UP": 12.126672421097085, "XI_Q": 12.984971351885598, "XI_C": -29.582212007237697,
                  "XI_Q2": 18.18785190955848, "VBM_0": 8.998824547791076, "AV": -14.238965719572148,
                  "AV2": 13.090080116642534, "B_BP": -7.177161522873087, "B2_BP": 70.41960340594974,
                  "D_BP": -1.9671238909825762e-07}        # first guess only; the fit replaces every value
RIDGE_CONSTANTS = ["XI_Q", "XI_C", "XI_Q2", "AV2", "B2_BP"]   # second-order constants, held small by a ridge penalty
RIDGE_STRENGTHS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]               # chosen by cross-validation
D_STARTS = [-4.0, -1.0, 1.0, 4.0]                             # starting values for d
DELTA_X_STARTS = [0.5, 1.0, 1.7, 3.0]                         # starting values for the valley gap DELTA_X (eV)
DELTA_X_BOUNDS = (0.05, 10.0)
N_FOLDS = 5
MAX_FUNCTION_EVALS = 4000

# ---- report
FIGURE_DPI = 300
SHOW_FRACTIONS = [0, 1, 5, 25, 100]       # fractions drawn in the per-fraction figures
TWO_FRACTIONS = [1, 100]                  # fractions drawn in the detailed figures
ROLLING_POINTS = 5                        # rolling median over this many log rows (= 50 iterations)
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_TEST = "#1baf7a"
COLOR_MODEL = "#2a78d6"
COLOR_ANALYTIC = "#6e7781"
COLOR_CONSTANTS = "#2a78d6"
COLOR_CORRECTION = "#eb6834"
CLASS_COLORS = {"isotropic": "#7a5af8", "uniaxial": "#2a78d6", "biaxial": "#1baf7a",
                "triaxial": "#eda100", "shear": "#eb6834"}
TERM_COLORS = {"data": "#2a78d6", "isotropic": "#7a5af8", "agreement": "#1baf7a", "symmetry": "#eb6834"}

# ---- model options. "base" uses DEFAULT_OPTIONS; every other variant changes only the options listed for it.
DEFAULT_OPTIONS = {
    "theta0_fit": "all",           # theta0 fitted on all training states ("all") or on labeled + isotropic ones ("labeled")
    "adjust_offsets": True,        # False: CBM_0 and VBM_0 stay at theta0 (the corrections already shift the edges)
    "constant_map": "linear",      # "tanh": theta = theta0 + step * tanh(u), so a constant moves at most one step
    "constant_penalty": 0.0,       # weight of mean(u^2) in the loss (u = network outputs for the constants)
    "correction_penalty": 0.0,     # weight of mean|dCBM| + mean|dVBM| in the loss
    "edge_agreement": 0.0,         # weight (times s) of |CBM - CBM(theta0)| + |VBM - VBM(theta0)|, non-isotropic states
    "edge_anchor": "edges",        # "edges": the term above; "midpoint": only the mean level (CBM + VBM)/2 is pulled
                                   # toward theta0, |mid - mid(theta0)|, so the gap itself is left free
    "isotropic_edges": False,      # True: the isotropic states supervise CBM and VBM, not only the gap
    "selection": "validation",     # kept checkpoint: "validation" (best validation gap MAE), "holdout" (best on part of
                                   # the labeled + isotropic training states, held out of training), "last" (final)
    "stop_at": 0,                  # stop after this iteration (0 = run all; the learning-rate schedule is unchanged)
    "patience": 0,                 # early stopping: stop after this many checkpoints without improvement (0 = off)
    "soap_warmup": 0,              # iterations of linear learning-rate warm-up after the switch to SOAP
    "soap_lr": SOAP_LR,            # peak learning rate of SOAP
    "output_init_std": 0.0,        # standard deviation of the starting output weights (0 = exactly zero)
    "loss_shape": "l1",            # "l1" (absolute error) or "smooth_l1" (quadratic below SMOOTH_L1_BETA)
    "n_rotations": 1,              # random cubic operations per training state and iteration in the symmetry term
    "dropout": 0.0,                # dropout after every hidden layer during training
    "agreement_vbm": "exact",      # VBM form of the agreement targets: "exact" largest eigenvalue, or "smooth" (the
                                   # same smooth maximum the network uses, so the untrained model matches exactly)
    "selection_metric": "gap",     # checkpoint score: "gap" (gap MAE) or "gap_edges" (gap MAE + 0.5 (CBM + VBM MAE))
}
VARIANTS = {
    "base":                {},
    # issue 1: CBM_0 / VBM_0 and the corrections dCBM / dVBM do the same job
    "freeze_offsets":      {"adjust_offsets": False},
    "penalize_correction": {"correction_penalty": 0.1},
    # issue 2: the constant changes are unbounded
    "tanh_constants":      {"constant_map": "tanh"},
    "penalize_constants":  {"constant_penalty": 0.05},
    # issue 3: with few labels only gaps anchor the band edges
    "edge_agreement":      {"edge_agreement": 0.5},
    "isotropic_edges":     {"isotropic_edges": True},
    # issue 4: theta0 is fitted on all training states (dual reporting)
    "theta0_labeled":      {"theta0_fit": "labeled"},
    # issue 5: 129 labeled validation states choose the checkpoint
    "select_last":         {"selection": "last"},
    "select_last_2500":    {"selection": "last", "stop_at": 2500},
    "select_holdout":      {"selection": "holdout"},
    "early_stop":          {"patience": 20},
    # issue 6: SOAP starts cold at about 7 times the learning rate AdamW had at iteration 12,000
    "soap_warmup":         {"soap_warmup": 200},
    "soap_lower_lr":       {"soap_lr": ADAMW_LR},
    # issue 7: zero output layer -> no hidden-layer gradient at iteration 1
    "small_output_init":   {"output_init_std": 1e-5},
    # issue 8: absolute-error kinks and one random rotation per step
    "smooth_l1":           {"loss_shape": "smooth_l1"},
    "four_rotations":      {"n_rotations": 4},
    # regularisation
    "dropout":             {"dropout": 0.05},
    # the changes adopted by the screen of 8 Nov 2026 (edge anchor on both edges; replaced by final_midpoint)
    "final":               {"constant_map": "tanh", "edge_agreement": 0.5, "soap_warmup": 200},
    # the final model (= phm.py): only the mean band-edge level anchored (weight 1.0 s gives the same pull on a common shift
    # of both edges as 0.5 s (|dCBM| + |dVBM|))
    "final_midpoint":      {"constant_map": "tanh", "edge_agreement": 1.0, "edge_anchor": "midpoint",
                            "soap_warmup": 200},    # candidate changes to the final model (review of 8 Nov 2026): CBM_0 / VBM_0 fixed (corrections and offset changes
    # were found cancelling each other, correlation -0.99), and a small starting output layer
    "final_frozen":        {"constant_map": "tanh", "edge_agreement": 1.0, "edge_anchor": "midpoint",
                            "soap_warmup": 200, "adjust_offsets": False},
    "final_small_init":    {"constant_map": "tanh", "edge_agreement": 1.0, "edge_anchor": "midpoint",
                            "soap_warmup": 200, "output_init_std": 1e-5},    # reporting check: the final model with theta0 fitted only on the labeled + isotropic training states
    # review of 8 Nov 2026 (2): agreement targets in the network's smooth VBM form; checkpoint chosen on gap + edges
    "final_smooth_target": {"constant_map": "tanh", "edge_agreement": 1.0, "edge_anchor": "midpoint",
                            "soap_warmup": 200, "agreement_vbm": "smooth"},
    "final_select_edges":  {"constant_map": "tanh", "edge_agreement": 1.0, "edge_anchor": "midpoint",
                            "soap_warmup": 200, "selection_metric": "gap_edges"},
    "final_theta0_labeled": {"constant_map": "tanh", "edge_agreement": 1.0, "edge_anchor": "midpoint",
                             "soap_warmup": 200, "theta0_fit": "labeled"},
}
SMOOTH_L1_BETA = 1e-3             # eV: smooth_l1 is quadratic below 1 meV and equal to |x| - 0.5 meV above
HOLDOUT_SHARE = 0.2               # share of the labeled + isotropic training states held out for selection="holdout"
HOLDOUT_SEED_OFFSET = 7777        # random stream of that hold-out: seed + this
EARLY_LOG_ITERATIONS = 50         # every one of the first 50 iterations is logged (start-up behaviour)

# ---- screening study and decision rules (written down before the screen is run)
SCREEN_SEEDS = list(range(42, 47))
SCREEN_FRACTIONS = [0, 1, 2, 5, 25, 100]
TOLERANCE_MEV = 0.5               # "not worse": paired mean difference to base at most +0.5 meV
TOLERANCE_TANH_MEV = 1.0          # bounded constants: 100%-label validation gap MAE at most +1 meV
TOLERANCE_SELECTION_MEV = 3.0     # a validation-free selection is usable when within 3 meV of base at 0-2%

# ---- the 48 operations of the cubic point group: axis permutations x sign changes
CUBIC_OPERATIONS = []
for _permutation in itertools.permutations(range(3)):
    for _signs in itertools.product([1, -1], repeat=3):
        CUBIC_OPERATIONS.append(np.diag(_signs) @ np.eye(3)[list(_permutation)])


# ======================================================================================================================
# ANALYTIC BAND-EDGE MODELS
# ======================================================================================================================
def agreement_weight(fraction):
    """Weight 2 s of the gap-agreement term. s = 1 for n <= 9 labeled states, s = (9/n)^3 above. As in the paper's
    runs, n is computed from the nominal training size and the weight is rounded to 6 significant digits."""
    n_labeled = round(N_TRAIN_NOMINAL * fraction / 100)
    if n_labeled <= AGREE_N_FULL:
        s = 1.0
    else:
        s = (AGREE_N_FULL / n_labeled) ** 3
    return float("%g" % (2.0 * s))


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
        # one-band shear term: used only inside the constants fit, before DELTA_X is added
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


def cbm_learned(strain, theta, delta_x):
    """CBM with per-state constants theta (N x 13): soft minimum over the six valleys."""
    valleys = valley_energies(strain, theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4], theta[:, 5], theta[:, 6],
                              delta_x, torch)
    return theta[:, 0] - torch.logsumexp(-BETA_CBM * valleys, dim=1) / BETA_CBM


def vbm_learned(strain, theta):
    """VBM with per-state constants theta: smooth maximum (1/beta) log trace exp(beta H) of the Bir-Pikus matrix.
    It is shifted by the exact largest eigenvalue (no gradient through the shift) so it never overflows."""
    H = bir_pikus_matrix(strain, theta[:, 8], theta[:, 9], theta[:, 10], theta[:, 11], theta[:, 12])
    with torch.no_grad():
        largest = torch.linalg.eigvalsh(torch.nan_to_num(H.double()))[:, 2].to(H.dtype)
    identity = torch.eye(3, dtype=H.dtype, device=H.device).unsqueeze(0)
    exponential = torch.linalg.matrix_exp(BETA_VBM * (H - largest[:, None, None] * identity))
    trace = torch.diagonal(exponential, dim1=1, dim2=2).sum(1)
    return theta[:, 7] + (largest + torch.log(trace) / BETA_VBM)


def cbm_analytic(strain, c, delta_x):
    """CBM with the fixed constants c (13 plain numbers): soft minimum over the six valleys."""
    valleys = valley_energies(strain, c[1], c[2], c[3], c[4], c[5], c[6], delta_x, torch)
    return c[0] + (-torch.logsumexp(-BETA_CBM * valleys, dim=1) / BETA_CBM)


def vbm_analytic(strain, c):
    """VBM with the fixed constants c: exact largest eigenvalue of the Bir-Pikus matrix."""
    H = bir_pikus_matrix(strain, c[8], c[9], c[10], c[11], c[12])
    H += EIGEN_EPS * torch.eye(3, dtype=strain.dtype, device=strain.device).unsqueeze(0)
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
    H += EIGEN_EPS * np.eye(3)[None]
    return vbm_0 + np.linalg.eigvalsh(H)[:, 2]


def gap_numpy(strain, th):
    return cbm_numpy(strain, th) - vbm_numpy(strain, th)


def fit_once(strain, cbm, vbm, ridge, start):
    """One least-squares fit from the starting values `start` with ridge strength `ridge`."""
    n_states = len(strain)
    ridge_index = [CONSTANT_NAMES.index(name) for name in RIDGE_CONSTANTS]
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
        lower[13] = DELTA_X_BOUNDS[0]
        upper[13] = DELTA_X_BOUNDS[1]
    return least_squares(residuals, start, method="trf", bounds=(lower, upper), max_nfev=MAX_FUNCTION_EVALS)


def fit_constants(strain, cbm, vbm, seed):
    """Fit the 13 constants and DELTA_X to the training states. Returns a dict {name: value}."""
    # step 1: the 13 constants, several starting values for d
    best = None
    for d_start in D_STARTS:
        start = np.array([CONSTANT_START[name] for name in CONSTANT_NAMES])
        start[12] = d_start
        result = fit_once(strain, cbm, vbm, 0.0, start)
        if best is None or result.cost < best.cost:
            best = result
    thirteen = best.x

    # step 2: add DELTA_X, several starting values
    best = None
    for delta_start in DELTA_X_STARTS:
        result = fit_once(strain, cbm, vbm, 0.0, np.append(thirteen, delta_start))
        if best is None or result.cost < best.cost:
            best = result
    fourteen = best.x

    # step 3: ridge strength by 5-fold cross-validation (gap MAE on the held-out fold)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(strain)), N_FOLDS)
    cv_error = {}
    for ridge in RIDGE_STRENGTHS:
        errors = []
        for k in range(N_FOLDS):
            held_out = folds[k]
            kept = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
            th = fit_once(strain[kept], cbm[kept], vbm[kept], ridge, fourteen).x
            errors.append(np.abs(gap_numpy(strain[held_out], th) - (cbm[held_out] - vbm[held_out])).mean())
        cv_error[ridge] = float(np.mean(errors))
    best_ridge = min(cv_error, key=cv_error.get)

    # step 4: final fit on all training states
    th = fit_once(strain, cbm, vbm, best_ridge, fourteen).x
    constants = {}
    for name, value in zip(CONSTANT_NAMES + ["DELTA_X"], th):
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


def load_and_split(data_csv, seed, fraction):
    """Read the dataset; split it by symmetry class into training, validation and test; scale the inputs on the
    training states only; draw the labeled subset of the training states."""
    table = pd.read_csv(data_csv)
    strain = table[STRAIN_COLUMNS].values.astype(np.float32)
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
        keys = [tuple(np.round(image, SYMMETRY_DECIMALS)) for image in cubic_images(strain[i].astype(np.float64))]
        key = min(keys)
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        symmetry_class[i] = key_to_id[key]

    # 2. per strain class, deal whole symmetry classes in a seeded random order: test first, then validation, then
    #    training, so no test state has a rotated twin in training or validation
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
            if n_test < round(TEST_SHARE * n_members):
                test_list.extend(states)
                n_test += len(states)
            elif n_val < round(VALIDATION_SHARE * n_members):
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
    rng_labels = np.random.default_rng(seed + LABEL_SEED_OFFSET)
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


def copied_labels(data, seed):
    """Every distinct cubic image of each labeled training state, with that state's CBM and VBM. At most
    MAX_COPIED_LABELS are kept. Balanced: a labeled state and its k images share one unit of weight, 1/(k+1) each."""
    parents = []
    images = []
    for i in data["labeled"]:
        state = data["strain_train"][i].astype(np.float64)
        seen = {tuple(np.round(state, IMAGE_DECIMALS))}
        for image in cubic_images(state):
            key = tuple(np.round(image, IMAGE_DECIMALS))
            if key not in seen:
                seen.add(key)
                parents.append(int(i))
                images.append(image)
    if len(parents) > MAX_COPIED_LABELS:
        rng = np.random.default_rng(seed + COPY_SEED_OFFSET)
        keep = np.sort(rng.choice(len(parents), MAX_COPIED_LABELS, replace=False))
        parents = [parents[j] for j in keep]
        images = [images[j] for j in keep]
    if len(parents) == 0:
        return None

    n_images_of = {}
    for parent in parents:
        n_images_of[parent] = n_images_of.get(parent, 0) + 1
    weight_images = np.array([1.0 / (n_images_of[parent] + 1) for parent in parents])
    weight_labeled = np.array([1.0 / (n_images_of.get(int(i), 0) + 1) for i in data["labeled"]])
    x_images = (np.array(images) - data["scaler"].mean_) / data["scaler"].scale_
    return {"x": x_images, "parent": np.array(parents), "weight_images": weight_images, "weight_labeled": weight_labeled}


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


def make_soap(parameters, lr):
    """SOAP with the settings of training phase 2 (peak learning rate lr)."""
    return SOAP(parameters, lr=lr, betas=SOAP_BETAS, weight_decay=SOAP_WEIGHT_DECAY,
                precondition_frequency=SOAP_PRECONDITION_FREQUENCY)


# ======================================================================================================================
# NETWORK
# ======================================================================================================================
def options_for(variant):
    """DEFAULT_OPTIONS with the changes of one variant."""
    if variant not in VARIANTS:
        raise SystemExit("unknown variant %s; known: %s" % (variant, ", ".join(VARIANTS)))
    options = dict(DEFAULT_OPTIONS)
    for name, value in VARIANTS[variant].items():
        options[name] = value
    return options


class PHM(nn.Module):
    """6 scaled strains -> 8 hidden layers of 64 SiLU neurons -> output layer:
         outputs 0-1   corrections dCBM, dVBM (eV)
         outputs 2-    changes u of the adjusted constants: theta = theta0 + step * u   (13 constants, or 11 when
                       CBM_0 and VBM_0 are kept at theta0)
       CBM = soft minimum of the valleys with theta + dCBM;  VBM = smooth maximum of Bir-Pikus with theta + dVBM."""

    def __init__(self, scaler, theta0, delta_x, options):
        super().__init__()
        self.options = options
        layer_sizes = [N_INPUTS] + [N_NEURONS] * N_HIDDEN_LAYERS
        self.hidden = nn.ModuleList()
        for i in range(N_HIDDEN_LAYERS):
            layers = [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.SiLU()]
            if options["dropout"] > 0:
                layers.append(nn.Dropout(options["dropout"]))
            self.hidden.append(nn.Sequential(*layers))

        # The paper's trainer first builds a 2-output layer and initialises it, then replaces it. That layer is built
        # here too, only so that the random-number stream (and so the hidden-layer weights) is exactly the same.
        temporary_output = nn.Linear(N_NEURONS, 2)
        for layer in self.hidden:
            nn.init.xavier_normal_(layer[0].weight)
            nn.init.zeros_(layer[0].bias)
        nn.init.xavier_normal_(temporary_output.weight)
        nn.init.zeros_(temporary_output.bias)

        # which constants the network adjusts
        self.adjusted = []
        for k in range(N_CONSTANTS):
            if options["adjust_offsets"] or CONSTANT_NAMES[k] not in ("CBM_0", "VBM_0"):
                self.adjusted.append(k)

        # output layer starts at zero: the untrained model is exactly the analytic models with theta0
        self.out = nn.Linear(N_NEURONS, N_CORRECTIONS + len(self.adjusted))
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()
        if options["output_init_std"] > 0:
            nn.init.normal_(self.out.weight, std=options["output_init_std"])

        step = []
        for value in theta0:
            step.append(max(abs(value), 1.0) * CONSTANT_STEP)
        step[-1] = D_BP_STEP
        self.register_buffer("theta0", torch.tensor(theta0, dtype=torch.float32))
        self.register_buffer("theta_step", torch.tensor(step, dtype=torch.float32))
        self.register_buffer("input_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.delta_x = delta_x

    def internals(self, x):
        """Physical strain, per-state constants theta, the two corrections and the raw constant outputs u."""
        h = x
        for layer in self.hidden:
            h = layer(h)
        o = self.out(h)
        strain = x * self.input_scale + self.input_mean      # back to physical strain for the formulas
        u = o[:, 2:]
        if self.options["constant_map"] == "tanh":
            change = torch.tanh(u)
        else:
            change = u
        if len(self.adjusted) < N_CONSTANTS:                  # constants that are not adjusted stay at theta0
            full_change = torch.zeros(x.shape[0], N_CONSTANTS, dtype=x.dtype, device=x.device)
            full_change[:, self.adjusted] = change
            change = full_change
        theta = self.theta0 + self.theta_step * change
        return strain, theta, o[:, 0], o[:, 1], u

    def forward_with_parts(self, x):
        strain, theta, d_cbm, d_vbm, u = self.internals(x)
        cbm = d_cbm + cbm_learned(strain, theta, self.delta_x)
        vbm = d_vbm + vbm_learned(strain, theta)
        parts = {"strain": strain, "theta": theta, "d_cbm": d_cbm, "d_vbm": d_vbm, "u": u}
        return cbm, vbm, cbm - vbm, parts

    def forward(self, x):
        cbm, vbm, gap, parts = self.forward_with_parts(x)
        return cbm, vbm, gap


# ======================================================================================================================
# TRAINING
# ======================================================================================================================
def absolute_error(difference, options):
    """|difference|; with loss_shape = smooth_l1 the smooth version: quadratic below SMOOTH_L1_BETA, |x| - beta/2 above."""
    if options["loss_shape"] == "l1":
        return difference.abs()
    size = difference.abs()
    return torch.where(size < SMOOTH_L1_BETA, 0.5 * difference ** 2 / SMOOTH_L1_BETA, size - 0.5 * SMOOTH_L1_BETA)


def isotropic_mask(strain):
    """True for states under isotropic strain (no shear, equal diagonal). The analytic models are not trusted there,
    so the gap agreement skips them, and their DFT gaps are used directly."""
    shear_size = strain[:, 3].abs() + strain[:, 4].abs() + strain[:, 5].abs()
    diagonal = strain[:, :3]
    diagonal_spread = (diagonal - diagonal.mean(dim=1, keepdim=True)).abs().max(dim=1).values
    return (shear_size < ISOTROPIC_TOLERANCE) & (diagonal_spread < ISOTROPIC_TOLERANCE)


def hold_out_for_selection(data, isotropic, seed):
    """selection = holdout: HOLDOUT_SHARE of the labeled + isotropic training states leave training and choose the
    checkpoint instead of the validation set. Returns their positions in the training set (data["labeled"] shrinks)."""
    candidates = np.union1d(data["labeled"], np.where(isotropic)[0])
    rng = np.random.default_rng(seed + HOLDOUT_SEED_OFFSET)
    n_held = max(1, round(HOLDOUT_SHARE * len(candidates)))
    held = np.sort(rng.choice(candidates, n_held, replace=False))
    data["labeled"] = np.setdiff1d(data["labeled"], held)
    return held


def prepare_tensors(data, seed, theta0, delta_x, options):
    """Everything the loss needs, as tensors (built once per run)."""
    t = {}
    t["x_train"] = torch.tensor(data["x_train"], dtype=torch.float32)
    t["x_val"] = torch.tensor(data["x_val"], dtype=torch.float32)
    t["x_test"] = torch.tensor(data["x_test"], dtype=torch.float32)
    t["strain_train"] = torch.tensor(data["strain_train"], dtype=torch.float32)
    t["gap_train"] = torch.tensor(data["gap_train"], dtype=torch.float32)
    t["cbm_train"] = torch.tensor(data["cbm_train"], dtype=torch.float32)
    t["vbm_train"] = torch.tensor(data["vbm_train"], dtype=torch.float32)
    t["gap_val"] = torch.tensor(data["gap_val"], dtype=torch.float32)
    t["cbm_val"] = torch.tensor(data["cbm"][data["index_val"]], dtype=torch.float32)
    t["vbm_val"] = torch.tensor(data["vbm"][data["index_val"]], dtype=torch.float32)
    t["gap_test"] = torch.tensor(data["gap_test"], dtype=torch.float32)
    t["input_mean"] = torch.tensor(data["scaler"].mean_, dtype=torch.float32)
    t["input_scale"] = torch.tensor(data["scaler"].scale_, dtype=torch.float32)
    t["operations"] = torch.tensor(np.array(CUBIC_OPERATIONS), dtype=torch.float32)

    # isotropic training states: excluded from the agreement; their DFT values supervise training
    strain = t["strain_train"]
    isotropic = isotropic_mask(strain)
    t["not_isotropic"] = ~isotropic
    t["n_held_out"] = 0
    if options["selection"] == "holdout":
        held = hold_out_for_selection(data, isotropic.numpy(), seed)
        t["x_held_out"] = t["x_train"][held]
        t["gap_held_out"] = t["gap_train"][held]
        t["n_held_out"] = len(held)
        isotropic = isotropic.clone()
        isotropic[held] = False                     # held-out isotropic states no longer supervise training
    t["isotropic"] = isotropic

    # labeled states
    t["n_labeled"] = len(data["labeled"])
    t["labeled"] = torch.tensor(data["labeled"], dtype=torch.long)
    t["cbm_labeled"] = torch.tensor(data["cbm_train"][data["labeled"]], dtype=torch.float32)
    t["vbm_labeled"] = torch.tensor(data["vbm_train"][data["labeled"]], dtype=torch.float32)

    # copied symmetry labels
    copies = None
    if t["n_labeled"] > 0:
        copies = copied_labels(data, seed)
    t["has_copies"] = copies is not None
    t["n_copies"] = 0
    if copies is not None:
        t["n_copies"] = len(copies["parent"])
        t["x_copies"] = torch.tensor(copies["x"], dtype=torch.float32)
        t["cbm_copies"] = torch.tensor(data["cbm_train"][copies["parent"]], dtype=torch.float32)
        t["vbm_copies"] = torch.tensor(data["vbm_train"][copies["parent"]], dtype=torch.float32)
        t["weight_copies"] = torch.tensor(copies["weight_images"], dtype=torch.float32)
        t["weight_labeled"] = torch.tensor(copies["weight_labeled"], dtype=torch.float32)

    # targets of the agreement terms: the analytic models with the fixed constants theta0
    with torch.no_grad():
        t["cbm_analytic"] = cbm_analytic(strain, theta0, delta_x)
        if options["agreement_vbm"] == "smooth":
            theta0_rows = torch.tensor(theta0, dtype=torch.float32).expand(strain.shape[0], -1)
            t["vbm_analytic"] = vbm_learned(strain, theta0_rows)
        else:
            t["vbm_analytic"] = vbm_analytic(strain, theta0)
        t["gap_analytic"] = t["cbm_analytic"] - t["vbm_analytic"]
    return t


def compute_loss(model, t, options, w_agree):
    """Sum of the loss terms (absolute errors unless loss_shape says otherwise) and the terms themselves."""
    zero = torch.tensor(0.0)
    cbm, vbm, gap, parts = model.forward_with_parts(t["x_train"])

    # 1. data: CBM and VBM of the labeled states and their copied images (balanced), averaged over labeled states
    n = t["n_labeled"]
    if n > 0 and t["has_copies"]:
        cbm_copies, vbm_copies, _ = model(t["x_copies"])
        cbm_sum = ((t["weight_labeled"] * absolute_error(cbm[t["labeled"]] - t["cbm_labeled"], options)).sum()
                   + (t["weight_copies"] * absolute_error(cbm_copies - t["cbm_copies"], options)).sum())
        vbm_sum = ((t["weight_labeled"] * absolute_error(vbm[t["labeled"]] - t["vbm_labeled"], options)).sum()
                   + (t["weight_copies"] * absolute_error(vbm_copies - t["vbm_copies"], options)).sum())
        loss_data = W_DATA * (cbm_sum / n + vbm_sum / n)
    elif n > 0:
        loss_data = W_DATA * (absolute_error(cbm[t["labeled"]] - t["cbm_labeled"], options).mean()
                              + absolute_error(vbm[t["labeled"]] - t["vbm_labeled"], options).mean())
    else:
        loss_data = zero

    # 2. isotropic states (always labeled): their gaps, or (isotropic_edges) their CBM and VBM
    iso = t["isotropic"]
    if iso.any() and options["isotropic_edges"]:
        loss_isotropic = W_ISOTROPIC * (absolute_error(cbm[iso] - t["cbm_train"][iso], options).mean()
                                        + absolute_error(vbm[iso] - t["vbm_train"][iso], options).mean())
    elif iso.any():
        loss_isotropic = W_ISOTROPIC * absolute_error(gap[iso] - t["gap_train"][iso], options).mean()
    else:
        loss_isotropic = zero

    # 3. gap agreement with the analytic models (theta0), on the non-isotropic training states
    other = t["not_isotropic"]
    loss_agreement = w_agree * absolute_error(gap[other] - t["gap_analytic"][other], options).mean()

    # 4. symmetry: random cubic operations R; the image R e R^T must give the same band edges
    strain = t["strain_train"]
    n_train = strain.shape[0]
    T = strain.new_zeros(n_train, 3, 3)
    T[:, 0, 0] = strain[:, 0]
    T[:, 1, 1] = strain[:, 1]
    T[:, 2, 2] = strain[:, 2]
    T[:, 0, 1] = strain[:, 3]
    T[:, 1, 0] = strain[:, 3]
    T[:, 1, 2] = strain[:, 4]
    T[:, 2, 1] = strain[:, 4]
    T[:, 0, 2] = strain[:, 5]
    T[:, 2, 0] = strain[:, 5]
    symmetry_terms = []
    for k in range(options["n_rotations"]):
        R = t["operations"][torch.randint(0, 48, (n_train,))]
        T_rotated = R @ T @ R.transpose(1, 2)
        strain_rotated = torch.stack([T_rotated[:, 0, 0], T_rotated[:, 1, 1], T_rotated[:, 2, 2],
                                      T_rotated[:, 0, 1], T_rotated[:, 1, 2], T_rotated[:, 0, 2]], dim=1)
        cbm_rotated, vbm_rotated, _ = model((strain_rotated - t["input_mean"]) / t["input_scale"])
        symmetry_terms.append(W_SYMMETRY * (absolute_error(cbm_rotated - cbm, options)
                                            + absolute_error(vbm_rotated - vbm, options)).mean())
    if len(symmetry_terms) == 1:
        loss_symmetry = symmetry_terms[0]
    else:
        loss_symmetry = sum(symmetry_terms) / len(symmetry_terms)

    total = loss_agreement + loss_data + loss_isotropic + loss_symmetry
    terms = {"data": loss_data, "isotropic": loss_isotropic, "agreement": loss_agreement, "symmetry": loss_symmetry}

    # optional terms (zero in base)
    if options["edge_agreement"] > 0:
        s = w_agree / 2.0
        if options["edge_anchor"] == "midpoint":
            midpoint = (cbm[other] + vbm[other]) / 2.0
            midpoint_analytic = (t["cbm_analytic"][other] + t["vbm_analytic"][other]) / 2.0
            loss_edges = options["edge_agreement"] * s * absolute_error(midpoint - midpoint_analytic, options).mean()
        else:
            loss_edges = options["edge_agreement"] * s * (
                absolute_error(cbm[other] - t["cbm_analytic"][other], options).mean()
                + absolute_error(vbm[other] - t["vbm_analytic"][other], options).mean())
        total = total + loss_edges
        terms["edge_agreement"] = loss_edges
    if options["constant_penalty"] > 0:
        loss_constants = options["constant_penalty"] * (parts["u"] ** 2).mean()
        total = total + loss_constants
        terms["constant_penalty"] = loss_constants
    if options["correction_penalty"] > 0:
        loss_corrections = options["correction_penalty"] * (parts["d_cbm"].abs().mean() + parts["d_vbm"].abs().mean())
        total = total + loss_corrections
        terms["correction_penalty"] = loss_corrections
    return total, terms


def gap_mae(model, x, gap_true):
    """Mean absolute gap error in eV (evaluation mode, no gradients)."""
    model.eval()
    with torch.no_grad():
        _, _, gap = model(x)
        error = float((gap - gap_true).abs().mean())
    model.train()
    return error


def gradient_norms(model):
    """L2 norm of the gradient of the hidden layers and of the output layer (before clipping)."""
    hidden_sum = 0.0
    for parameter in model.hidden.parameters():
        if parameter.grad is not None:
            hidden_sum += float((parameter.grad ** 2).sum())
    output_sum = 0.0
    for parameter in model.out.parameters():
        if parameter.grad is not None:
            output_sum += float((parameter.grad ** 2).sum())
    return np.sqrt(hidden_sum), np.sqrt(output_sum)


def symmetry_violation(model, data):
    """Mean over the test states of the largest change of CBM or VBM over all 48 cubic images (eV)."""
    model.eval()
    strain = torch.tensor(data["strain"][data["index_test"]], dtype=torch.float32)
    mean = torch.tensor(data["scaler"].mean_, dtype=torch.float32)
    scale = torch.tensor(data["scaler"].scale_, dtype=torch.float32)
    n_states = strain.shape[0]
    T = strain.new_zeros(n_states, 3, 3)
    T[:, 0, 0] = strain[:, 0]
    T[:, 1, 1] = strain[:, 1]
    T[:, 2, 2] = strain[:, 2]
    T[:, 0, 1] = strain[:, 3]
    T[:, 1, 0] = strain[:, 3]
    T[:, 1, 2] = strain[:, 4]
    T[:, 2, 1] = strain[:, 4]
    T[:, 0, 2] = strain[:, 5]
    T[:, 2, 0] = strain[:, 5]
    largest = torch.zeros(n_states)
    with torch.no_grad():
        cbm, vbm, _ = model((strain - mean) / scale)
        for operation in CUBIC_OPERATIONS:
            R = torch.tensor(operation, dtype=torch.float32)
            T_rotated = R @ T @ R.T
            strain_rotated = torch.stack([T_rotated[:, 0, 0], T_rotated[:, 1, 1], T_rotated[:, 2, 2],
                                          T_rotated[:, 0, 1], T_rotated[:, 1, 2], T_rotated[:, 0, 2]], dim=1)
            cbm_rotated, vbm_rotated, _ = model((strain_rotated - mean) / scale)
            change = torch.maximum((cbm_rotated - cbm).abs(), (vbm_rotated - vbm).abs())
            largest = torch.maximum(largest, change)
    return float(largest.mean())


def train_one_run(data_csv, out, variant, seed, fraction, n_iterations, threads):
    """Train one variant of PHM for one seed and one labeled fraction; write log, predictions and summary."""
    options = options_for(variant)
    folder = run_folder(out, variant, seed, fraction)
    os.makedirs(folder, exist_ok=True)
    if threads > 0:
        torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    constants = read_constants(out, seed, fraction, options["theta0_fit"])
    with open(os.path.join(folder, "constants.json"), "w") as f:
        json.dump(constants, f, indent=1)
    with open(os.path.join(folder, "options.json"), "w") as f:
        json.dump(options, f, indent=1)
    theta0 = [float(constants[name]) for name in CONSTANT_NAMES]
    delta_x = float(constants["DELTA_X"])
    data = load_and_split(data_csv, seed, fraction)
    t = prepare_tensors(data, seed, theta0, delta_x, options)
    w_agree = agreement_weight(fraction)
    print("%s, seed %d, %d%% labeled: %d train, %d validation, %d test, %d labeled, %d copied images, %d held out, "
          "agreement weight %g" % (variant, seed, fraction, len(data["index_train"]), len(data["index_val"]),
                                   len(data["index_test"]), t["n_labeled"], t["n_copies"], t["n_held_out"], w_agree),
          flush=True)

    model = PHM(data["scaler"], theta0, delta_x, options)
    switch_iteration = int(round(SWITCH_FRACTION * n_iterations))
    last_iteration = n_iterations
    if options["stop_at"] > 0:
        last_iteration = min(n_iterations, options["stop_at"])
    optimizer = optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=ADAMW_WEIGHT_DECAY, betas=ADAMW_BETAS)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=RESTART_PERIOD_ADAMW,
                                                               T_mult=RESTART_GROWTH, eta_min=LR_MIN)
    best_score = float("inf")
    best_iteration = 0
    best_state = None
    stale_checkpoints = 0
    n_clipped = 0
    log_rows = []
    start_time = time.time()

    iteration = 0
    for iteration in range(1, last_iteration + 1):
        if iteration == switch_iteration + 1:
            # phase 2: a fresh SOAP optimizer with its own schedule
            optimizer = make_soap(model.parameters(), options["soap_lr"])
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=RESTART_PERIOD_SOAP,
                                                                       T_mult=RESTART_GROWTH, eta_min=LR_MIN)
        logging_now = iteration <= EARLY_LOG_ITERATIONS or iteration % LOG_EVERY == 0
        model.train()
        optimizer.zero_grad()
        loss, terms = compute_loss(model, t, options, w_agree)
        loss.backward()
        if logging_now:
            hidden_norm, output_norm = gradient_norms(model)
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM))
        if grad_norm > CLIP_NORM:
            n_clipped += 1

        # SOAP warm-up: the scheduled learning rate is scaled by k / soap_warmup in the first iterations of phase 2
        warmup_step = iteration - switch_iteration
        if options["soap_warmup"] > 0 and 1 <= warmup_step <= options["soap_warmup"]:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * warmup_step / options["soap_warmup"]
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        # checkpoint: keep the weights with the lowest selection score (validation or held-out gap MAE)
        if iteration % CHECKPOINT_EVERY == 0 and options["selection"] != "last":
            if options["selection"] == "validation" and options["selection_metric"] == "gap_edges":
                model.eval()
                with torch.no_grad():
                    cbm_v, vbm_v, gap_v = model(t["x_val"])
                model.train()
                score = float((gap_v - t["gap_val"]).abs().mean()
                              + 0.5 * ((cbm_v - t["cbm_val"]).abs().mean() + (vbm_v - t["vbm_val"]).abs().mean()))
            elif options["selection"] == "validation":
                score = gap_mae(model, t["x_val"], t["gap_val"])
            else:
                score = gap_mae(model, t["x_held_out"], t["gap_held_out"])
            if score < best_score:
                best_score = score
                best_iteration = iteration
                best_state = {}
                for name, value in model.state_dict().items():
                    best_state[name] = value.clone()
                stale_checkpoints = 0
            else:
                stale_checkpoints += 1

        # log (diagnostic only; nothing here changes the training)
        if logging_now:
            row = {"iteration": iteration, "loss": float(loss.detach())}
            for name, value in terms.items():
                row[name] = float(value.detach())
            row["lr"] = lr_used
            row["grad_norm"] = grad_norm
            row["grad_norm_after_clip"] = grad_norm * min(1.0, CLIP_NORM / (grad_norm + 1e-6))
            row["grad_norm_hidden"] = hidden_norm
            row["grad_norm_output"] = output_norm
            row["train_mae"] = gap_mae(model, t["x_train"], t["gap_train"])
            row["val_mae"] = gap_mae(model, t["x_val"], t["gap_val"])
            row["test_mae"] = gap_mae(model, t["x_test"], t["gap_test"])
            log_rows.append(row)
        if iteration % 2000 == 0:
            print("  iteration %5d  lr %.2e  loss %.4f  best selection score %.1f meV  (%.0f s)"
                  % (iteration, lr_used, float(loss.detach()), best_score * 1000, time.time() - start_time), flush=True)
        if options["patience"] > 0 and stale_checkpoints >= options["patience"]:
            print("  early stop at iteration %d (%d checkpoints without improvement)" % (iteration, stale_checkpoints))
            break

    train_time = time.time() - start_time
    iterations_run = iteration
    if options["selection"] == "last" or best_state is None:
        best_iteration = iterations_run                       # the final weights are kept
    else:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(folder, "best_model.pt"))
    log = pd.DataFrame(log_rows)
    log.to_csv(os.path.join(folder, "log.csv"), index=False)
    predictions = save_predictions(model, data, theta0, delta_x, folder)
    violation = symmetry_violation(model, data)
    save_summary(predictions, log, variant, seed, fraction, t, n_iterations, iterations_run, switch_iteration,
                 best_iteration, best_score, n_clipped, violation, train_time, folder)


def save_predictions(model, data, theta0, delta_x, folder):
    """Predictions of the kept model for every state, and how the gap change splits between constants and correction."""
    model.eval()
    x_all = torch.tensor((data["strain"] - data["scaler"].mean_) / data["scaler"].scale_, dtype=torch.float32)
    with torch.no_grad():
        cbm, vbm, gap, parts = model.forward_with_parts(x_all)
        strain = parts["strain"]
        theta = parts["theta"]
        theta_start = model.theta0.expand(len(gap), -1)
        gap_untrained = cbm_learned(strain, theta_start, delta_x) - vbm_learned(strain, theta_start)
        gap_with_constants = cbm_learned(strain, theta, delta_x) - vbm_learned(strain, theta)
        cbm_formula = cbm_analytic(strain, theta0, delta_x)
        vbm_formula = vbm_analytic(strain, theta0)

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
    p["cbm_analytic"] = cbm_formula.numpy()                       # analytic models with theta0 (no network)
    p["vbm_analytic"] = vbm_formula.numpy()
    p["gap_analytic"] = (cbm_formula - vbm_formula).numpy()
    p["gap_from_constants"] = (gap_with_constants - gap_untrained).numpy()
    p["gap_from_correction"] = (parts["d_cbm"] - parts["d_vbm"]).numpy()
    p["d_cbm"] = parts["d_cbm"].numpy()
    p["d_vbm"] = parts["d_vbm"].numpy()
    for k, name in enumerate(CONSTANT_NAMES):
        change = (theta[:, k] - model.theta0[k]).numpy()
        p["change_" + name] = change
        p["relative_change_" + name] = 100.0 * change / max(abs(theta0[k]), 1.0)     # percent
    p.to_csv(os.path.join(folder, "predictions.csv"), index=False)
    return p


def save_summary(p, log, variant, seed, fraction, t, n_iterations, iterations_run, switch_iteration, best_iteration,
                 best_score, n_clipped, violation, train_time, folder):
    """One row of metrics. Written last: its presence marks the run as finished."""
    test = p[p["split"] == "test"]
    val = p[p["split"] == "val"]
    train = p[p["split"] == "train"]
    error = (test["gap_pred"] - test["gap_true"]).values
    error_analytic = (test["gap_analytic"] - test["gap_true"]).values
    total_variance = ((test["gap_true"] - test["gap_true"].mean()) ** 2).sum()
    relative_changes = test[["relative_change_" + name for name in CONSTANT_NAMES]].abs().values

    # training-dynamics numbers from the log
    loss_by_iteration = log.set_index("iteration")["loss"]
    val_by_iteration = log.set_index("iteration")["val_mae"]
    spike_ratio = np.nan
    if switch_iteration in loss_by_iteration.index and switch_iteration + LOG_EVERY in loss_by_iteration.index:
        spike_ratio = loss_by_iteration[switch_iteration + LOG_EVERY] / loss_by_iteration[switch_iteration]
    phase_two = val_by_iteration[val_by_iteration.index > switch_iteration]
    best_phase_two = phase_two.min() * 1000 if len(phase_two) > 0 else np.nan
    val_at_500 = val_by_iteration[500] * 1000 if 500 in val_by_iteration.index else np.nan
    early = log[log["iteration"] <= EARLY_LOG_ITERATIONS]
    hidden_to_output = (early["grad_norm_hidden"] / early["grad_norm_output"].replace(0.0, np.nan)).median()

    s = {"variant": variant, "seed": seed, "fraction": fraction, "n_labeled": t["n_labeled"],
         "n_copies": t["n_copies"], "n_held_out": t["n_held_out"],
         "n_iterations": n_iterations, "iterations_run": iterations_run, "best_iteration": best_iteration,
         "test_mae_meV": np.abs(error).mean() * 1000,
         "test_rmse_meV": np.sqrt((error ** 2).mean()) * 1000,
         "test_max_meV": np.abs(error).max() * 1000,
         "test_r2": 1.0 - (error ** 2).sum() / total_variance,
         "test_cbm_mae_meV": np.abs(test["cbm_pred"] - test["cbm_true"]).mean() * 1000,
         "test_vbm_mae_meV": np.abs(test["vbm_pred"] - test["vbm_true"]).mean() * 1000,
         "val_mae_meV": np.abs(val["gap_pred"] - val["gap_true"]).mean() * 1000,
         "val_cbm_mae_meV": np.abs(val["cbm_pred"] - val["cbm_true"]).mean() * 1000,
         "val_vbm_mae_meV": np.abs(val["vbm_pred"] - val["vbm_true"]).mean() * 1000,
         "train_mae_meV": np.abs(train["gap_pred"] - train["gap_true"]).mean() * 1000,
         "selection_score_meV": best_score * 1000,
         "analytic_test_mae_meV": np.abs(error_analytic).mean() * 1000,
         "analytic_cbm_mae_meV": np.abs(test["cbm_analytic"] - test["cbm_true"]).mean() * 1000,
         "analytic_vbm_mae_meV": np.abs(test["vbm_analytic"] - test["vbm_true"]).mean() * 1000,
         "symmetry_violation_meV": violation * 1000,
         "max_constant_change_pct": relative_changes.max(),
         "mean_constant_change_pct": relative_changes.mean(),
         "mean_abs_correction_meV": (test["d_cbm"].abs().mean() + test["d_vbm"].abs().mean()) * 1000,
         "soap_spike_ratio": spike_ratio,
         "best_phase2_val_meV": best_phase_two,
         "val_at_500_meV": val_at_500,
         "early_hidden_to_output_gradient": hidden_to_output,
         "clipped_share": n_clipped / iterations_run,
         "train_time_s": train_time,
         "seconds_per_1000_iterations": train_time / iterations_run * 1000}
    for class_name in STRAIN_CLASSES:
        in_class = (test["strain_class"] == class_name).values
        s["test_mae_" + class_name + "_meV"] = np.abs(error[in_class]).mean() * 1000
        s["analytic_mae_" + class_name + "_meV"] = np.abs(error_analytic[in_class]).mean() * 1000
    pd.DataFrame([s]).to_csv(os.path.join(folder, "metrics_summary.csv"), index=False)
    print("%s, seed %d, %d%% labeled: test gap MAE %.2f meV (analytic models %.2f meV), kept iteration %d of %d, %.0f s"
          % (variant, seed, fraction, s["test_mae_meV"], s["analytic_test_mae_meV"], best_iteration, iterations_run,
             train_time), flush=True)


# ======================================================================================================================
# STUDY DRIVER
# ======================================================================================================================
def run_folder(out, variant, seed, fraction):
    return os.path.join(out, variant, "%dpct" % fraction, "seed%d" % seed)


def constants_file(out, seed, fraction, fit):
    """theta0 fitted on all training states: one file per seed. Fitted on the labeled + isotropic training states:
    one file per seed and labeled fraction."""
    if fit == "all":
        return os.path.join(out, "constants", "const_seed%d.json" % seed)
    return os.path.join(out, "constants", "labeled", "const_seed%d_%dpct.json" % (seed, fraction))


def read_constants(out, seed, fraction, fit):
    path = constants_file(out, seed, fraction, fit)
    if not os.path.exists(path):
        raise SystemExit("missing %s: run the 'constants' command first" % path)
    with open(path) as f:
        return json.load(f)


def fit_all_constants(data_csv, out, variants, seeds, fractions):
    """Fit theta0 for every seed (on all training states, as in the paper) and, if a variant asks for it, for every
    seed and labeled fraction on the labeled + isotropic training states only."""
    fits = set()
    for variant in variants:
        fits.add(options_for(variant)["theta0_fit"])
    jobs = []
    for seed in seeds:
        if "all" in fits:
            jobs.append((seed, None, "all"))
        if "labeled" in fits:
            for fraction in fractions:
                if fraction > 0:
                    jobs.append((seed, fraction, "labeled"))
    for seed, fraction, fit in jobs:
        path = constants_file(out, seed, fraction, fit)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        start = time.time()
        if fit == "all":
            data = load_and_split(data_csv, seed, 100)
            use = np.arange(len(data["index_train"]))
        else:
            data = load_and_split(data_csv, seed, fraction)
            strain = data["strain_train"]
            isotropic = ((np.abs(strain[:, 3:]).sum(axis=1) < ISOTROPIC_TOLERANCE)
                         & (np.abs(strain[:, :3] - strain[:, :3].mean(axis=1, keepdims=True)).max(axis=1)
                            < ISOTROPIC_TOLERANCE))
            use = np.union1d(data["labeled"], np.where(isotropic)[0])
        constants = fit_constants(data["strain_train"][use].astype(float), data["cbm_train"][use].astype(float),
                                  data["vbm_train"][use].astype(float), seed)
        with open(path, "w") as f:
            json.dump(constants, f, indent=1)
        print("constants %s: fitted on %d training states (%.0f s)" % (path, len(use), time.time() - start), flush=True)


def runs_in_study(variants, seeds, fractions):
    """(variant, seed, fraction) of every run. theta0_labeled has no run at 0% (no labeled states to fit on)."""
    runs = []
    for variant in variants:
        for fraction in fractions:
            if options_for(variant)["theta0_fit"] == "labeled" and fraction == 0:
                continue
            for seed in seeds:
                runs.append((variant, seed, fraction))
    return runs


def run_command(script, data_csv, out, variant, seed, fraction, n_iterations, threads):
    return ("python %s run --data_csv %s --out %s --variant %s --seed %d --fraction %d --iterations %d --threads %d"
            % (script, data_csv, out, variant, seed, fraction, n_iterations, threads))


def write_task_list(script, data_csv, out, variants, seeds, fractions, n_iterations):
    """One line per run: <run folder> TAB <command>. Paths are written as given (relative paths stay relative)."""
    path = os.path.join(out, "phm.tsv")
    os.makedirs(out, exist_ok=True)
    runs = runs_in_study(variants, seeds, fractions)
    with open(path, "w") as f:
        for variant, seed, fraction in runs:
            command = run_command(script, data_csv, out, variant, seed, fraction, n_iterations, 1)
            f.write("%s\t%s\n" % (run_folder(out, variant, seed, fraction), command))
    print("wrote %s (%d runs)" % (path, len(runs)))


def run_study_locally(data_csv, out, variants, seeds, fractions, n_iterations, jobs, threads):
    """Constants, then every unfinished run (`jobs` at a time, each with `threads` threads), then the reports."""
    fit_all_constants(data_csv, out, variants, seeds, fractions)
    todo = []
    for variant, seed, fraction in runs_in_study(variants, seeds, fractions):
        if not os.path.exists(os.path.join(run_folder(out, variant, seed, fraction), "metrics_summary.csv")):
            todo.append((variant, seed, fraction))
    print("%d runs to do, %d at a time" % (len(todo), jobs), flush=True)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    running = []
    failed = []
    while len(todo) > 0 or len(running) > 0:
        while len(todo) > 0 and len(running) < jobs:
            variant, seed, fraction = todo.pop(0)
            folder = run_folder(out, variant, seed, fraction)
            os.makedirs(folder, exist_ok=True)
            command = [sys.executable, os.path.abspath(__file__), "run", "--data_csv", data_csv, "--out", out,
                       "--variant", variant, "--seed", str(seed), "--fraction", str(fraction),
                       "--iterations", str(n_iterations), "--threads", str(threads)]
            output = open(os.path.join(folder, "stdout.txt"), "w")
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=environment)
            running.append((process, variant, seed, fraction))
        time.sleep(2)
        still_running = []
        for process, variant, seed, fraction in running:
            if process.poll() is None:
                still_running.append((process, variant, seed, fraction))
            else:
                print("  finished %s seed %d, %d%% (exit code %d)" % (variant, seed, fraction, process.returncode),
                      flush=True)
                if process.returncode != 0:
                    failed.append((variant, seed, fraction))
        running = still_running
    if len(failed) > 0:
        print("FAILED runs (see stdout.txt in their folders):", failed)
    for variant in variants:
        make_report(data_csv, out, variant)
    if len(variants) > 1:
        compare_variants(out)


# ======================================================================================================================
# REPORT
# ======================================================================================================================
def load_summaries(out, variant):
    files = sorted(glob.glob(os.path.join(out, variant, "*pct", "seed*", "metrics_summary.csv")))
    if len(files) == 0:
        raise SystemExit("no finished runs under %s" % os.path.join(out, variant))
    rows = [pd.read_csv(f).iloc[0] for f in files]
    return pd.DataFrame(rows).sort_values(["fraction", "seed"]).reset_index(drop=True)


def load_log(out, variant, seed, fraction):
    path = os.path.join(run_folder(out, variant, seed, fraction), "log.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_predictions(out, variant, seeds, fraction, split="test"):
    """Predictions of all seeds at one fraction, pooled (only the rows of one split, or all rows if split is None)."""
    tables = []
    for seed in seeds:
        path = os.path.join(run_folder(out, variant, seed, fraction), "predictions.csv")
        if os.path.exists(path):
            p = pd.read_csv(path)
            p["seed"] = seed
            tables.append(p)
    if len(tables) == 0:
        return None
    p = pd.concat(tables, ignore_index=True)
    if split is not None:
        p = p[p["split"] == split]
    return p


def mean_and_ci(values):
    """Mean and half-width of the 95% t interval."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.0
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half_width


def median_curve(out, variant, seeds, fraction, column):
    """Median over seeds of one column of log.csv, indexed by iteration (the first 20 iterations are left out)."""
    curves = []
    for seed in seeds:
        log = load_log(out, variant, seed, fraction)
        if log is not None and column in log.columns:
            curve = log.set_index("iteration")[column]
            curves.append(curve[curve.index > 20])
    if len(curves) == 0:
        return None
    return pd.concat(curves, axis=1).median(axis=1)


def write_tables(summaries, out):
    """results.txt (readable), summary_by_fraction.csv and all_runs.csv."""
    lines = []
    seeds = sorted(summaries["seed"].unique())
    fractions = sorted(summaries["fraction"].unique())
    lines.append("PHM, variant %s: test results over %d seeds (%s), %d iterations per run"
                 % (summaries["variant"].iloc[0], len(seeds), ", ".join(str(int(s)) for s in seeds), int(summaries["n_iterations"].max())))
    lines.append("test gap MAE: mean ± sd [95% t interval]. Analytic models = the same constants theta0, no network.")
    lines.append("")
    lines.append("%8s %5s %26s %7s %7s %8s %8s %8s %8s %9s %9s %8s" % (
        "labeled", "seeds", "test gap MAE (meV)", "RMSE", "max", "R2", "CBM MAE", "VBM MAE", "val MAE", "train MAE",
        "best iter", "clipped"))
    table_rows = []
    for fraction in fractions:
        rows = summaries[summaries["fraction"] == fraction]
        mean, half = mean_and_ci(rows["test_mae_meV"])
        mae_text = "%.1f ± %.1f [%.1f, %.1f]" % (mean, rows["test_mae_meV"].std(), mean - half, mean + half)
        lines.append("%7d%% %5d %26s %7.1f %7.1f %8.4f %8.1f %8.1f %8.1f %9.1f %9.0f %7.0f%%" % (
            fraction, len(rows), mae_text, rows["test_rmse_meV"].mean(), rows["test_max_meV"].mean(),
            rows["test_r2"].mean(), rows["test_cbm_mae_meV"].mean(), rows["test_vbm_mae_meV"].mean(),
            rows["val_mae_meV"].mean(), rows["train_mae_meV"].mean(), rows["best_iteration"].median(),
            100 * rows["clipped_share"].mean()))
        difference = rows["test_mae_meV"].values - rows["analytic_test_mae_meV"].values
        d_mean, d_half = mean_and_ci(difference)
        table_rows.append({"fraction": fraction, "n_seeds": len(rows), "test_mae_mean_meV": mean,
                           "test_mae_sd_meV": rows["test_mae_meV"].std(), "test_mae_ci95_meV": half,
                           "minus_analytic_mean_meV": d_mean, "minus_analytic_ci95_meV": d_half})

    lines.append("")
    one_row_per_seed = summaries.drop_duplicates("seed")
    analytic_mean, analytic_half = mean_and_ci(one_row_per_seed["analytic_test_mae_meV"])
    lines.append("analytic models alone (theta0, no network): %.1f meV [%.1f, %.1f]"
                 % (analytic_mean, analytic_mean - analytic_half, analytic_mean + analytic_half))
    lines.append("PHM minus analytic models, paired by seed (negative = PHM better):")
    for row in table_rows:
        low = row["minus_analytic_mean_meV"] - row["minus_analytic_ci95_meV"]
        high = row["minus_analytic_mean_meV"] + row["minus_analytic_ci95_meV"]
        lines.append("  %3d%%: %+6.1f [%+.1f, %+.1f] meV" % (row["fraction"], row["minus_analytic_mean_meV"], low, high))

    lines.append("")
    lines.append("test gap MAE per strain class (meV), mean over seeds")
    lines.append("%9s" % "labeled" + "".join("%11s" % name for name in STRAIN_CLASSES))
    for fraction in fractions:
        rows = summaries[summaries["fraction"] == fraction]
        cells = ""
        for name in STRAIN_CLASSES:
            cells += "%11.1f" % rows["test_mae_" + name + "_meV"].mean()
        lines.append("%8d%%" % fraction + cells)
    cells = ""
    for name in STRAIN_CLASSES:
        cells += "%11.1f" % one_row_per_seed["analytic_mae_" + name + "_meV"].mean()
    lines.append("%9s" % "analytic" + cells)

    with open(os.path.join(out, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(table_rows).to_csv(os.path.join(out, "summary_by_fraction.csv"), index=False)
    summaries.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    print("\n".join(lines))


# ---- small figure helpers
def setup_matplotlib():
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333", "xtick.color": "#555555",
                         "ytick.color": "#555555", "axes.titlesize": 10, "axes.titleweight": "bold",
                         "legend.frameon": False, "savefig.dpi": FIGURE_DPI})


def log_axis(ax):
    """Logarithmic y axis with plain numbers as tick labels (call it after plotting). Ticks at 1, 2, 5 x 10^k when
    the axis spans fewer than 3 decades, otherwise only at 10^k."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    if np.log10(high / low) < 3:
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5), numticks=12))
    else:
        ax.yaxis.set_major_locator(LogLocator(subs=(1,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: "%g" % value))
    ax.yaxis.set_minor_formatter(NullFormatter())


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, figure_folder, name, caption, captions):
    fig.savefig(os.path.join(figure_folder, name + ".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(figure_folder, name + ".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    captions.append("%s: %s" % (name, caption))


def available(wanted, fractions):
    """The wanted fractions that were run (or all of them if none of the wanted ones were)."""
    chosen = [f for f in wanted if f in fractions]
    if len(chosen) == 0:
        chosen = list(fractions)
    return chosen


def fraction_colors(n):
    """n shades of one blue, light (few labels) to dark (many labels)."""
    return [plt.cm.Blues(value) for value in np.linspace(0.45, 0.95, n)]


# ---- 1. training dynamics
def plot_learning_curves(out, variant, summaries, seeds, fractions, figure_folder, captions):
    shown = available(SHOW_FRACTIONS, fractions)
    switch = int(round(SWITCH_FRACTION * summaries["n_iterations"].max()))
    fig, axes = plt.subplots(1, len(shown), figsize=(3.9 * len(shown), 3.5), squeeze=False)
    for ax, fraction in zip(axes[0], shown):
        for column, color, style, label in (("train_mae", COLOR_TRAIN, ":", "training set"),
                                            ("val_mae", COLOR_VAL, "-", "validation"),
                                            ("test_mae", COLOR_TEST, "--", "test")):
            curve = median_curve(out, variant, seeds, fraction, column)
            if curve is not None:
                ax.plot(curve.index / 1000, curve.values * 1000, style, color=color, linewidth=1.3, label=label)
        best = summaries[summaries["fraction"] == fraction]["best_iteration"].median()
        ax.axvline(best / 1000, color="#222222", linewidth=1.0, linestyle="-.", label="kept checkpoint (median)")
        ax.axvline(switch / 1000, color="#999999", linewidth=0.8, linestyle="--", label="AdamW → SOAP")
        log_axis(ax)
        light_grid(ax)
        ax.set_title("%d%% labeled" % fraction, loc="left")
        ax.set_xlabel("iteration (thousands)")
        ax.set_ylabel("gap MAE (meV)")
    axes[0][0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Learning curves (median over seeds)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig01_learning_curves",
                "gap MAE on the training, validation and test states during training (median over seeds); dash-dot = "
                "median iteration of the kept checkpoint; grey dashed = switch from AdamW to SOAP.", captions)


def plot_loss_terms(out, variant, seeds, fractions, figure_folder, captions):
    shown = available(SHOW_FRACTIONS, fractions)
    labels = {"data": "data", "isotropic": "isotropic gap", "agreement": "gap agreement", "symmetry": "symmetry"}
    fig, axes = plt.subplots(2, len(shown), figsize=(3.9 * len(shown), 6.2), sharex=True, squeeze=False)
    for j, fraction in enumerate(shown):
        weights = {"data": W_DATA, "isotropic": W_ISOTROPIC, "agreement": agreement_weight(fraction),
                   "symmetry": W_SYMMETRY}
        for term in ["data", "isotropic", "agreement", "symmetry"]:
            curve = median_curve(out, variant, seeds, fraction, term)
            if curve is None or not (curve > 0).any():
                continue                                    # the term is zero at this fraction (data at 0%)
            curve = curve.rolling(ROLLING_POINTS, min_periods=1).median()
            axes[0][j].plot(curve.index / 1000, curve.values, color=TERM_COLORS[term], linewidth=1.2,
                            label="%s (× %g)" % (labels[term], weights[term]))
            axes[1][j].plot(curve.index / 1000, curve.values / weights[term] * 1000, color=TERM_COLORS[term],
                            linewidth=1.2, label=labels[term])
        total = median_curve(out, variant, seeds, fraction, "loss")
        if total is not None:
            total = total.rolling(ROLLING_POINTS, min_periods=1).median()
            axes[0][j].plot(total.index / 1000, total.values, color="#222222", linewidth=1.0, linestyle="--",
                            label="total")
        for i in range(2):
            log_axis(axes[i][j])
            light_grid(axes[i][j])
        axes[0][j].set_title("%d%% labeled" % fraction, loc="left")
        axes[1][j].set_xlabel("iteration (thousands)")
        axes[0][j].legend(fontsize=7)
    axes[0][0].set_ylabel("term as weighted in the loss")
    axes[1][0].set_ylabel("term without its weight (meV)")
    fig.suptitle("Loss terms during training (median over seeds, 50-iteration rolling median)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig02_loss_terms",
                "each loss term during training: top as weighted in the loss, bottom without its weight (a mean "
                "absolute error in meV). A term is not drawn where it is zero (no labels at 0%).", captions)


def plot_lr_and_gradients(out, variant, summaries, seeds, fractions, figure_folder, captions):
    shown = available(TWO_FRACTIONS, fractions)
    fig, axes = plt.subplots(2, len(shown), figsize=(5.0 * len(shown), 5.6), sharex=True, squeeze=False)
    for j, fraction in enumerate(shown):
        log = load_log(out, variant, seeds[0], fraction)
        if log is not None:
            axes[0][j].plot(log["iteration"] / 1000, log["lr"], color="#eda100", linewidth=1.2)
        before = median_curve(out, variant, seeds, fraction, "grad_norm")
        after = median_curve(out, variant, seeds, fraction, "grad_norm_after_clip")
        if before is not None:
            before = before.rolling(ROLLING_POINTS, min_periods=1).median()
            after = after.rolling(ROLLING_POINTS, min_periods=1).median()
            axes[1][j].plot(before.index / 1000, before.values, color=COLOR_TRAIN, linewidth=1.0,
                            label="before clipping")
            axes[1][j].plot(after.index / 1000, after.values, color=COLOR_VAL, linewidth=1.0, linestyle="--",
                            label="after clipping (limit %g)" % CLIP_NORM)
        clipped = 100 * summaries[summaries["fraction"] == fraction]["clipped_share"].mean()
        axes[1][j].text(0.02, 0.04, "clipping active in %.0f%% of iterations" % clipped,
                        transform=axes[1][j].transAxes, fontsize=8, color="#333333")
        for i in range(2):
            log_axis(axes[i][j])
            light_grid(axes[i][j])
        axes[0][j].set_title("%d%% labeled" % fraction, loc="left")
        axes[1][j].set_xlabel("iteration (thousands)")
    axes[0][0].set_ylabel("learning rate")
    axes[1][0].set_ylabel("gradient norm (L2)")
    axes[1][0].legend(fontsize=8)
    fig.suptitle("Learning-rate schedule (warm restarts; SOAP in the second half) and gradient norms", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig03_lr_gradients",
                "learning rate (first seed) and the L2 gradient norm before and after clipping (median over seeds).",
                captions)


# ---- 2. accuracy
def plot_parity(out, variant, seeds, fractions, figure_folder, captions):
    shown = available(SHOW_FRACTIONS, fractions)
    fig, axes = plt.subplots(1, len(shown), figsize=(3.9 * len(shown), 4.0), squeeze=False)
    for ax, fraction in zip(axes[0], shown):
        p = load_predictions(out, variant, seeds, fraction)
        low = min(p["gap_true"].min(), p["gap_pred"].min()) - 0.05
        high = max(p["gap_true"].max(), p["gap_pred"].max()) + 0.05
        ax.plot([low, high], [low, high], color="#999999", linewidth=0.9)
        for class_name in STRAIN_CLASSES:
            rows = p[p["strain_class"] == class_name]
            ax.scatter(rows["gap_true"], rows["gap_pred"], s=6, color=CLASS_COLORS[class_name], alpha=0.75,
                       linewidth=0, label=class_name)
        error = p["gap_pred"] - p["gap_true"]
        r2 = 1.0 - (error ** 2).sum() / ((p["gap_true"] - p["gap_true"].mean()) ** 2).sum()
        ax.text(0.04, 0.96, "MAE %.1f meV\nRMSE %.1f meV\nR² %.4f" % (error.abs().mean() * 1000,
                np.sqrt((error ** 2).mean()) * 1000, r2), transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_title("%d%% labeled" % fraction, loc="left")
        ax.set_xlabel("DFT gap (eV)")
        ax.set_ylabel("predicted gap (eV)")
    axes[0][0].legend(fontsize=7.5, loc="lower right", markerscale=2)
    fig.suptitle("Predicted vs DFT gap on the test states (all seeds pooled)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig04_parity", "predicted vs DFT gap of the test states; colour = strain class; "
                "diagonal = perfect prediction.", captions)


def plot_residuals(out, variant, seeds, fractions, figure_folder, captions):
    shown = available(TWO_FRACTIONS, fractions)
    fig, axes = plt.subplots(len(shown), 2, figsize=(10.5, 3.5 * len(shown)), squeeze=False)
    for i, fraction in enumerate(shown):
        p = load_predictions(out, variant, seeds, fraction)
        residual = (p["gap_pred"] - p["gap_true"]).values * 1000
        for j, (column, label) in enumerate((("gap_true", "DFT gap (eV)"),
                                             ("max_abs_strain", "largest strain component |ε|"))):
            ax = axes[i][j]
            for class_name in STRAIN_CLASSES:
                in_class = (p["strain_class"] == class_name).values
                ax.scatter(p[column].values[in_class], residual[in_class], s=6, color=CLASS_COLORS[class_name],
                           alpha=0.7, linewidth=0, label=class_name)
            ax.axhline(0, color="#999999", linewidth=0.9)
            order = np.argsort(p[column].values)
            window = max(5, len(order) // 25)
            x_sorted = pd.Series(p[column].values[order])
            y_sorted = pd.Series(residual[order])
            ax.plot(x_sorted.rolling(window, center=True).median(), y_sorted.rolling(window, center=True).median(),
                    color="#222222", linewidth=1.2, label="running median")
            ax.set_xlabel(label)
            ax.set_ylabel("predicted − DFT gap (meV)")
            ax.set_title("%d%% labeled" % fraction, loc="left")
            light_grid(ax)
    axes[0][0].legend(fontsize=7.5, ncol=3, markerscale=2)
    fig.suptitle("Test residuals vs gap and vs strain size (all seeds pooled)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig05_residuals", "residuals of the test states against the DFT gap and the "
                "largest strain component; black = running median (a trend shows a systematic bias).", captions)


def plot_error_ecdf(out, variant, seeds, fractions, figure_folder, captions):
    shown = available(SHOW_FRACTIONS, fractions)
    colors = fraction_colors(len(shown))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for fraction, color in zip(shown, colors):
        p = load_predictions(out, variant, seeds, fraction)
        error = np.sort(np.abs(p["gap_pred"] - p["gap_true"]).values * 1000)
        share = np.arange(1, len(error) + 1) / len(error)
        ax.plot(error, share, color=color, linewidth=1.5,
                label="%d%% labeled (95%% below %.0f meV)" % (fraction, np.percentile(error, 95)))
    p = load_predictions(out, variant, seeds, shown[0])
    error = np.sort(np.abs(p["gap_analytic"] - p["gap_true"]).values * 1000)
    ax.plot(error, np.arange(1, len(error) + 1) / len(error), color=COLOR_ANALYTIC, linewidth=1.5, linestyle="--",
            label="analytic models θ₀ (95%% below %.0f meV)" % np.percentile(error, 95))
    ax.axhline(0.95, color="#bbbbbb", linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("|predicted − DFT gap| (meV)")
    ax.set_ylabel("share of test states")
    light_grid(ax)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Error distribution of the test states (all seeds pooled)", loc="left")
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig06_error_ecdf", "cumulative distribution of the absolute test errors; the "
                "legend gives the error below which 95% of the test states fall.", captions)


def plot_seed_spread(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(1.0 * len(fractions) + 2.5, 4.0))
    for k, fraction in enumerate(fractions):
        values = summaries[summaries["fraction"] == fraction]["test_mae_meV"].values
        ax.boxplot([values], positions=[k], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": COLOR_MODEL},
                   medianprops={"color": COLOR_MODEL, "linewidth": 1.6},
                   whiskerprops={"color": COLOR_MODEL}, capprops={"color": COLOR_MODEL})
        offsets = np.linspace(-0.15, 0.15, len(values))
        ax.scatter(np.full(len(values), k) + offsets, values, s=10, color=COLOR_MODEL, zorder=3)
    analytic = summaries.drop_duplicates("seed")["analytic_test_mae_meV"].mean()
    ax.axhline(analytic, color=COLOR_ANALYTIC, linewidth=1.2, linestyle="--", label="analytic models θ₀ (mean)")
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(["%d%%" % f for f in fractions])
    ax.set_xlabel("labeled training states")
    ax.set_ylabel("test gap MAE per seed (meV)")
    log_axis(ax)
    light_grid(ax)
    ax.legend(fontsize=8)
    ax.set_title("Seed-to-seed spread of the test error", loc="left")
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig07_seed_spread", "test gap MAE of every seed (dots) with box plots per "
                "labeled fraction.", captions)


# ---- 3. subgroups, data efficiency, generalization gap
def plot_per_class(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.array(fractions, dtype=float)
    one_row_per_seed = summaries.drop_duplicates("seed")
    smallest_mean = np.inf
    for class_name in STRAIN_CLASSES:
        means = []
        halves = []
        for fraction in fractions:
            mean, half = mean_and_ci(summaries[summaries["fraction"] == fraction]["test_mae_" + class_name + "_meV"])
            means.append(mean)
            halves.append(half)
        means = np.array(means)
        halves = np.array(halves)
        smallest_mean = min(smallest_mean, np.nanmin(means))
        color = CLASS_COLORS[class_name]
        ax.plot(x, means, "o-", color=color, linewidth=1.5, markersize=4, label=class_name)
        ax.fill_between(x, means - halves, means + halves, color=color, alpha=0.08, linewidth=0)
        analytic = one_row_per_seed["analytic_mae_" + class_name + "_meV"].mean()
        ax.axhline(analytic, color=color, linewidth=0.9, linestyle=":", alpha=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, x.max() * 1.25)               # no empty space left of 0 %
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % f for f in fractions])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.6 * smallest_mean)        # the axis follows the means, not the interval bands
    log_axis(ax)
    light_grid(ax)
    ax.set_xlabel("labeled training states (%)")
    ax.set_ylabel("test gap MAE (meV)")
    ax.legend(fontsize=8, title="strain class", title_fontsize=8)
    ax.set_title("Test error per strain class (band = 95% interval; dotted = analytic models θ₀)", loc="left",
                 fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig08_per_class", "test gap MAE per strain class vs the labeled fraction; bands "
                "are 95% t intervals over seeds; dotted lines are the analytic models with θ₀ for each class.", captions)


def plot_data_efficiency(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    x = np.array(fractions, dtype=float)
    means = []
    halves = []
    for fraction in fractions:
        mean, half = mean_and_ci(summaries[summaries["fraction"] == fraction]["test_mae_meV"])
        means.append(mean)
        halves.append(half)
    means = np.array(means)
    halves = np.array(halves)
    ax.plot(x, means, "o-", color=COLOR_MODEL, linewidth=1.7, markersize=5, label="PHM")
    ax.fill_between(x, means - halves, means + halves, color=COLOR_MODEL, alpha=0.15, linewidth=0)
    analytic_mean, analytic_half = mean_and_ci(summaries.drop_duplicates("seed")["analytic_test_mae_meV"])
    ax.axhline(analytic_mean, color=COLOR_ANALYTIC, linewidth=1.4, linestyle="--", label="analytic models θ₀")
    ax.axhspan(analytic_mean - analytic_half, analytic_mean + analytic_half, color=COLOR_ANALYTIC, alpha=0.1)
    for x_value, mean in zip(x, means):
        ax.annotate("%.1f" % mean, (x_value, mean), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, x.max() * 1.25)               # no empty space left of 0 %
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % f for f in fractions])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.6 * np.nanmin(means))     # the axis follows the means, not the interval band
    log_axis(ax)
    light_grid(ax)
    ax.set_xlabel("labeled training states (%)")
    ax.set_ylabel("test gap MAE (meV)")
    ax.legend(fontsize=8.5)
    ax.set_title("Data efficiency (mean over seeds; band = 95% t interval)", loc="left")
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig09_data_efficiency", "test gap MAE vs the labeled fraction, with the analytic "
                "models (θ₀, no network) as reference.", captions)


def plot_generalization_gap(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(1.1 * len(fractions) + 2.5, 4.0))
    bar_width = 0.26
    for k, (column, color, label) in enumerate((("train_mae_meV", COLOR_TRAIN, "training set"),
                                                ("val_mae_meV", COLOR_VAL, "validation (kept checkpoint)"),
                                                ("test_mae_meV", COLOR_TEST, "test"))):
        means = []
        halves = []
        for fraction in fractions:
            mean, half = mean_and_ci(summaries[summaries["fraction"] == fraction][column])
            means.append(mean)
            halves.append(half)
        positions = np.arange(len(fractions)) + (k - 1) * bar_width
        ax.bar(positions, means, bar_width * 0.92, yerr=halves, color=color, label=label,
               error_kw={"linewidth": 0.8, "capsize": 1.5})
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(["%d%%" % f for f in fractions])
    ax.set_xlabel("labeled training states")
    ax.set_ylabel("gap MAE (meV)")
    log_axis(ax)
    light_grid(ax)
    ax.legend(fontsize=8)
    ax.set_title("Training / validation / test error of the kept model (mean, 95% interval)", loc="left",
                 fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig10_generalization_gap", "gap MAE of the kept checkpoint on the training, "
                "validation and test states.", captions)


# ---- 4. physical consistency and internals
def plot_strain_sweeps(data_csv, out, variant, seeds, fractions, figure_folder, captions):
    table = pd.read_csv(data_csv)
    paths = [("isotropic", table["strain_type"] == "isotropic", "E_xx", "ε_xx = ε_yy = ε_zz"),
             ("uniaxial [100]", table["strain_type"] == "uniaxial_x", "E_xx", "ε_xx"),
             ("biaxial (001)", table["strain_type"] == "biaxial_xy", "E_xx", "ε_xx = ε_yy")]
    if "shear_planes" in table.columns:
        paths.append(("shear xy", table["shear_planes"].astype(str) == "xy", "E_xy", "ε_xy"))
    shown = available(TWO_FRACTIONS, fractions)
    seed = seeds[0]
    options = options_for(variant)
    fig, axes = plt.subplots(len(shown), len(paths), figsize=(4.3 * len(paths), 3.6 * len(shown)), squeeze=False)
    for i, fraction in enumerate(shown):
        folder = run_folder(out, variant, seed, fraction)
        with open(os.path.join(folder, "constants.json")) as f:
            constants = json.load(f)                 # the constants this run used
        theta0 = [float(constants[name]) for name in CONSTANT_NAMES]
        delta_x = float(constants["DELTA_X"])
        data = load_and_split(data_csv, seed, fraction)
        model = PHM(data["scaler"], theta0, delta_x, options)
        model.load_state_dict(torch.load(os.path.join(folder, "best_model.pt")))
        model.eval()
        for j, (name, on_path, parameter, label) in enumerate(paths):
            ax = axes[i][j]
            states = table[on_path]
            if len(states) < 3:
                ax.axis("off")
                continue
            t_values = states[parameter].values
            strains = states[STRAIN_COLUMNS].values
            # the states of a path lie on a straight line in strain space: fit that line, then sample it finely
            t_line = np.linspace(t_values.min(), t_values.max(), 300)
            line = np.zeros((len(t_line), 6), dtype=np.float32)
            for c in range(6):
                slope, intercept = np.polyfit(t_values, strains[:, c], 1)
                line[:, c] = slope * t_line + intercept
            line_tensor = torch.tensor(line)
            x_line = torch.tensor((line - data["scaler"].mean_) / data["scaler"].scale_, dtype=torch.float32)
            with torch.no_grad():
                gap_analytic = (cbm_analytic(line_tensor, theta0, delta_x)
                                - vbm_analytic(line_tensor, theta0)).numpy()
                gap_model = model(x_line)[2].numpy()
            ax.plot(t_line, gap_analytic, color=COLOR_ANALYTIC, linewidth=1.4, linestyle="--",
                    label="analytic models θ₀")
            ax.plot(t_line, gap_model, color=COLOR_MODEL, linewidth=1.6, label="PHM")
            split = data["split"][states.index.values]
            for kind, marker, face, kind_label in (("test", "o", "#222222", "DFT, test"),
                                                   ("train", "o", "white", "DFT, training"),
                                                   ("val", "D", "white", "DFT, validation")):
                on = split == kind
                ax.scatter(t_values[on], states["Eg_eV"].values[on], s=18, marker=marker, facecolor=face,
                           edgecolor="#222222", linewidth=0.8, zorder=4, label=kind_label)
            ax.set_xlabel(label)
            ax.set_ylabel("gap (eV)")
            ax.set_title("%s · %d%% labeled" % (name, fraction), loc="left")
            light_grid(ax)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Gap along continuous strain paths (seed %d) with the DFT states on each path" % seed, fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig11_strain_sweeps", "predicted gap along four strain paths (lines) and the DFT "
                "states on them (filled = test, open = training, diamond = validation).", captions)


def plot_constants_vs_correction(out, variant, seeds, fractions, figure_folder, captions):
    shown = available(TWO_FRACTIONS, fractions)
    fig, axes = plt.subplots(2, len(shown), figsize=(6.6 * len(shown), 7.6), squeeze=False)
    for j, fraction in enumerate(shown):
        p = load_predictions(out, variant, seeds, fraction)
        ax = axes[0][j]
        changes = [p["relative_change_" + name].values for name in CONSTANT_NAMES]
        boxes = ax.boxplot(changes, showfliers=False, patch_artist=True, widths=0.6)
        for k, box in enumerate(boxes["boxes"]):
            if k < 7:
                box.set_facecolor("#e8f1fb")
                box.set_edgecolor(COLOR_CONSTANTS)
            else:
                box.set_facecolor("#fcedf3")
                box.set_edgecolor("#e87ba4")
        for median_line in boxes["medians"]:
            median_line.set_color("#222222")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_xticks(range(1, 14))
        ax.set_xticklabels(CONSTANT_SYMBOLS, fontsize=8.5)
        ax.set_ylabel("(θ − θ₀) / max(|θ₀|, 1)  (%)")
        ax.set_title("%d%% labeled: change of each constant (blue = conduction, pink = valence)" % fraction,
                     loc="left", fontsize=9.5)
        light_grid(ax)

        ax = axes[1][j]
        from_constants = p["gap_from_constants"].values * 1000
        from_correction = p["gap_from_correction"].values * 1000
        low = min(from_constants.min(), from_correction.min())
        high = max(from_constants.max(), from_correction.max()) + 1e-9
        bins = np.linspace(low, high, 60)
        ax.hist(from_constants, bins=bins, color=COLOR_CONSTANTS, alpha=0.6,
                label="from the adjusted constants (mean |·| %.1f meV)" % np.abs(from_constants).mean())
        ax.hist(from_correction, bins=bins, color=COLOR_CORRECTION, alpha=0.6,
                label="from the free correction (mean |·| %.1f meV)" % np.abs(from_correction).mean())
        ax.set_xlabel("change of the gap relative to the untrained model (meV)")
        ax.set_ylabel("test states")
        ax.set_title("%d%% labeled: how the two paths share the work" % fraction, loc="left", fontsize=9.5)
        ax.legend(fontsize=8)
        light_grid(ax)
    fig.suptitle("Inside PHM: learned constant changes and constants vs correction (test states, all seeds)",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig12_constants_vs_correction", "top: change of each constant relative to θ₀ "
                "over the test states; bottom: the part of the gap change that comes from the adjusted constants and "
                "the part that comes from the free correction.", captions)


def make_report(data_csv, out, variant):
    """Tables and figures of one variant, written to <out>/<variant>/."""
    setup_matplotlib()
    summaries = load_summaries(out, variant)
    seeds = [int(s) for s in sorted(summaries["seed"].unique())]
    fractions = [int(f) for f in sorted(summaries["fraction"].unique())]
    variant_folder = os.path.join(out, variant)
    write_tables(summaries, variant_folder)
    figure_folder = os.path.join(variant_folder, "figures")
    os.makedirs(figure_folder, exist_ok=True)
    captions = []
    plot_learning_curves(out, variant, summaries, seeds, fractions, figure_folder, captions)
    plot_loss_terms(out, variant, seeds, fractions, figure_folder, captions)
    plot_lr_and_gradients(out, variant, summaries, seeds, fractions, figure_folder, captions)
    plot_parity(out, variant, seeds, fractions, figure_folder, captions)
    plot_residuals(out, variant, seeds, fractions, figure_folder, captions)
    plot_error_ecdf(out, variant, seeds, fractions, figure_folder, captions)
    plot_seed_spread(summaries, fractions, figure_folder, captions)
    plot_per_class(summaries, fractions, figure_folder, captions)
    plot_data_efficiency(summaries, fractions, figure_folder, captions)
    plot_generalization_gap(summaries, fractions, figure_folder, captions)
    plot_strain_sweeps(data_csv, out, variant, seeds, fractions, figure_folder, captions)
    plot_constants_vs_correction(out, variant, seeds, fractions, figure_folder, captions)
    with open(os.path.join(figure_folder, "captions.txt"), "w") as f:
        f.write("\n".join(captions) + "\n")
    print("tables: %s   figures: %s" % (os.path.join(variant_folder, "results.txt"), figure_folder))


# ======================================================================================================================
# COMPARISON OF VARIANTS (ablations and candidate fixes against base, paired by seed)
# ======================================================================================================================
COMPARE_COLUMNS = [("test_mae_meV", "test gap MAE (meV)"), ("val_mae_meV", "validation gap MAE (meV)"),
                   ("val_edges_meV", "validation CBM + VBM MAE (meV)"), ("test_edges_meV", "test CBM + VBM MAE (meV)"),
                   ("symmetry_violation_meV", "48-image symmetry violation (meV)"),
                   ("max_constant_change_pct", "largest constant change (%)"),
                   ("soap_spike_ratio", "loss ratio iteration 12,010 / 12,000"),
                   ("best_phase2_val_meV", "best validation gap MAE after the SOAP switch (meV)"),
                   ("val_at_500_meV", "validation gap MAE at iteration 500 (meV)"),
                   ("iterations_run", "iterations run"), ("seconds_per_1000_iterations", "seconds per 1000 iterations")]


def paired_difference(all_runs, variant, fraction, column):
    """Mean of (variant - base) over the seeds both have, the half-width of its 95% t interval, and the seed count."""
    rows = all_runs[all_runs["fraction"] == fraction]
    base = rows[rows["variant"] == "base"].set_index("seed")[column]
    other = rows[rows["variant"] == variant].set_index("seed")[column]
    difference = (other - base).dropna()
    if len(difference) == 0:
        return np.nan, np.nan, 0
    mean, half = mean_and_ci(difference.values)
    return mean, half, len(difference)


def compare_variants(out):
    folder = os.path.join(out, "comparison")
    os.makedirs(folder, exist_ok=True)
    tables = []
    for variant in VARIANTS:
        if len(glob.glob(os.path.join(out, variant, "*pct", "seed*", "metrics_summary.csv"))) > 0:
            tables.append(load_summaries(out, variant))
    if len(tables) == 0 or "base" not in [table["variant"].iloc[0] for table in tables]:
        raise SystemExit("compare needs finished runs of base and at least one other variant under %s" % out)
    all_runs = pd.concat(tables, ignore_index=True)
    all_runs["val_edges_meV"] = all_runs["val_cbm_mae_meV"] + all_runs["val_vbm_mae_meV"]
    all_runs["test_edges_meV"] = all_runs["test_cbm_mae_meV"] + all_runs["test_vbm_mae_meV"]
    all_runs.to_csv(os.path.join(folder, "all_runs.csv"), index=False)
    variants = [v for v in VARIANTS if v in set(all_runs["variant"])]
    fractions = [int(f) for f in sorted(all_runs["fraction"].unique())]

    # table: mean of every metric, and the paired difference to base
    rows = []
    lines = ["Variants of PHM against base. Seeds: %s. Paired by seed: mean difference [95%% t interval]; "
             "negative = lower than base." % ", ".join(str(int(s)) for s in sorted(all_runs["seed"].unique())), ""]
    for column, label in COMPARE_COLUMNS:
        lines.append("== %s ==" % label)
        lines.append("%-22s" % "variant" + "".join("%26s" % ("%d%%" % f) for f in fractions))
        for variant in variants:
            cells = ""
            for fraction in fractions:
                values = all_runs[(all_runs["variant"] == variant) & (all_runs["fraction"] == fraction)][column]
                mean_value = values.mean() if len(values) > 0 else np.nan
                if variant == "base":
                    cells += "%26s" % ("%.2f" % mean_value)
                else:
                    d_mean, d_half, n = paired_difference(all_runs, variant, fraction, column)
                    if n == 0 or np.isnan(d_mean):
                        cells += "%26s" % "-"
                    else:
                        cells += "%26s" % ("%.2f (%+.2f ± %.2f)" % (mean_value, d_mean, d_half))
                    rows.append({"variant": variant, "fraction": fraction, "metric": column, "mean": mean_value,
                                 "minus_base_mean": d_mean, "minus_base_ci95": d_half, "n_seeds": n})
            lines.append("%-22s" % variant + cells)
        lines.append("")
    with open(os.path.join(folder, "comparison.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(folder, "comparison.csv"), index=False)

    write_decisions(all_runs, variants, fractions, folder)
    setup_matplotlib()
    plot_variant_differences(all_runs, variants, fractions, "val_mae_meV", "validation gap MAE minus base (meV)",
                             os.path.join(folder, "compare_validation_gap"))
    plot_variant_differences(all_runs, variants, fractions, "test_mae_meV", "test gap MAE minus base (meV)",
                             os.path.join(folder, "compare_test_gap"))
    plot_issue_diagnostics(all_runs, fractions, os.path.join(folder, "issue_diagnostics"))
    print("comparison written to %s" % folder)


def write_decisions(all_runs, variants, fractions, folder):
    """Apply the decision rules (fixed before the screen, see SCREEN_* and TOLERANCE_* constants) and write them out."""
    lines = []

    def d(variant, fraction, column):
        return paired_difference(all_runs, variant, fraction, column)

    def text(variant, fraction, column):
        mean, half, n = d(variant, fraction, column)
        if n == 0:
            return "%d%%: not run" % fraction
        return "%d%%: %+.2f [%+.2f, %+.2f] (n=%d)" % (fraction, mean, mean - half, mean + half, n)

    def present(variant, needed):
        return variant in variants and all(f in fractions for f in needed)

    def mean_value(variant, fraction, column):
        values = all_runs[(all_runs["variant"] == variant) & (all_runs["fraction"] == fraction)][column]
        return values.mean()

    lines.append("DECISIONS from the screening seeds. Each rule was written down before the runs. A change adopted here")
    lines.append("still has to be combined with the other adopted changes and confirmed before the held-out seeds 59-75.")
    lines.append("")

    lines.append("Issue 1: CBM_0 / VBM_0 overlap with the corrections")
    lines.append("  rule: adopt freeze_offsets if validation CBM + VBM MAE at 5%% and 100%% is at most +%.1f meV vs base"
                 % TOLERANCE_MEV)
    if present("freeze_offsets", [5, 100]):
        worst = max(d("freeze_offsets", 5, "val_edges_meV")[0], d("freeze_offsets", 100, "val_edges_meV")[0])
        lines.append("  freeze_offsets: " + " | ".join(text("freeze_offsets", f, "val_edges_meV") for f in (5, 100)))
        lines.append("  -> %s" % ("ADOPT freeze_offsets" if worst <= TOLERANCE_MEV else "keep the 13 adjusted constants"))
    if present("penalize_correction", [5, 100]):
        lines.append("  (secondary) penalize_correction: "
                     + " | ".join(text("penalize_correction", f, "val_edges_meV") for f in (5, 100)))
    lines.append("")

    lines.append("Issue 2: unbounded constant changes")
    lines.append("  rule: adopt tanh_constants if the 100%%-label validation gap MAE is at most +%.1f meV vs base; "
                 "otherwise penalize_constants under the same rule" % TOLERANCE_TANH_MEV)
    for variant in ("base", "tanh_constants", "penalize_constants"):
        if present(variant, [100]):
            lines.append("  %-20s largest constant change at 100%%: %.0f %%" % (variant, mean_value(variant, 100,
                                                                                    "max_constant_change_pct")))
    verdict = "keep unbounded constants"
    if present("penalize_constants", [100]):
        lines.append("  penalize_constants: " + text("penalize_constants", 100, "val_mae_meV"))
        if d("penalize_constants", 100, "val_mae_meV")[0] <= TOLERANCE_TANH_MEV:
            verdict = "ADOPT penalize_constants"
    if present("tanh_constants", [100]):
        lines.append("  tanh_constants: " + text("tanh_constants", 100, "val_mae_meV"))
        if d("tanh_constants", 100, "val_mae_meV")[0] <= TOLERANCE_TANH_MEV:
            verdict = "ADOPT tanh_constants"
    lines.append("  -> " + verdict)
    lines.append("")

    lines.append("Issue 3: with few labels only gaps anchor the band edges")
    lines.append("  rule: among edge_agreement and isotropic_edges, adopt the one that lowers validation CBM + VBM MAE")
    lines.append("        most (mean over 0, 1, 2%%) while validation gap MAE at 5, 25, 100%% stays within +%.1f meV"
                 % TOLERANCE_MEV)
    low = [f for f in (0, 1, 2) if f in fractions]
    high = [f for f in (5, 25, 100) if f in fractions]
    if present("base", low):
        for f in low:
            lines.append("  base at %d%%: test gap MAE %.1f, test CBM MAE %.1f, test VBM MAE %.1f meV"
                         % (f, mean_value("base", f, "test_mae_meV"), mean_value("base", f, "test_cbm_mae_meV"),
                            mean_value("base", f, "test_vbm_mae_meV")))
    best_variant, best_gain = None, 0.0
    for variant in ("edge_agreement", "isotropic_edges"):
        if not present(variant, low + high):
            continue
        gain = np.mean([d(variant, f, "val_edges_meV")[0] for f in low])
        high_ok = all(d(variant, f, "val_mae_meV")[0] <= TOLERANCE_MEV for f in high)
        lines.append("  %s: validation CBM+VBM %s" % (variant, " | ".join(text(variant, f, "val_edges_meV") for f in low)))
        lines.append("  %s: test CBM+VBM       %s" % (variant, " | ".join(text(variant, f, "test_edges_meV") for f in low)))
        lines.append("  %s: validation gap     %s  (%s)" % (variant, " | ".join(text(variant, f, "val_mae_meV")
                                                                                 for f in high),
                                                             "within tolerance" if high_ok else "worse"))
        if high_ok and gain < best_gain:
            best_variant, best_gain = variant, gain
    lines.append("  -> %s" % ("ADOPT " + best_variant if best_variant else "keep gap-only anchoring"))
    lines.append("")

    lines.append("Issue 4: theta0 fitted on all training states (dual reporting; nothing is adopted)")
    if "theta0_labeled" in variants:
        lines.append("  theta0_labeled test gap MAE minus base: "
                     + " | ".join(text("theta0_labeled", f, "test_mae_meV") for f in fractions if f > 0))
    lines.append("")

    lines.append("Issue 5: 129 labeled validation states choose the checkpoint")
    lines.append("  rule: a validation-free selection is usable as a verification if its test gap MAE at 0, 1, 2%% is")
    lines.append("        within +%.0f meV of base" % TOLERANCE_SELECTION_MEV)
    for variant in ("select_last", "select_last_2500", "select_holdout"):
        if present(variant, low):
            worst = max(d(variant, f, "test_mae_meV")[0] for f in low)
            lines.append("  %-18s %s -> %s" % (variant, " | ".join(text(variant, f, "test_mae_meV") for f in low),
                                               "usable" if worst <= TOLERANCE_SELECTION_MEV else "not usable"))
    lines.append("")

    lines.append("Issue 6: SOAP starts cold")
    lines.append("  rule: adopt a variant if its loss jump at the switch (ratio 12,010 / 12,000) is lower than base at")
    lines.append("        25%% and 100%% and its best validation gap MAE after the switch is within +%.1f meV"
                 % TOLERANCE_MEV)
    adopted = []
    for variant in ("soap_warmup", "soap_lower_lr"):
        if not present(variant, [25, 100]):
            continue
        lower_spike = all(mean_value(variant, f, "soap_spike_ratio") < mean_value("base", f, "soap_spike_ratio")
                          for f in (25, 100))
        phase_two_ok = all(d(variant, f, "best_phase2_val_meV")[0] <= TOLERANCE_MEV for f in (25, 100))
        lines.append("  %s: spike ratio %s vs base %s; best phase-2 validation %s" % (
            variant, " / ".join("%.2f" % mean_value(variant, f, "soap_spike_ratio") for f in (25, 100)),
            " / ".join("%.2f" % mean_value("base", f, "soap_spike_ratio") for f in (25, 100)),
            " | ".join(text(variant, f, "best_phase2_val_meV") for f in (25, 100))))
        if lower_spike and phase_two_ok:
            adopted.append(variant)
    lines.append("  -> %s" % ("ADOPT " + " and ".join(adopted) if adopted else "keep the cold start"))
    lines.append("")

    lines.append("Issue 7: no hidden-layer gradient at iteration 1")
    lines.append("  rule: adopt small_output_init if validation gap MAE at iteration 500 is lower (mean over fractions)")
    lines.append("        and the validation gap MAE of the kept model is within +%.1f meV at every fraction" % TOLERANCE_MEV)
    if present("small_output_init", fractions):
        early = np.mean([d("small_output_init", f, "val_at_500_meV")[0] for f in fractions])
        final_ok = all(d("small_output_init", f, "val_mae_meV")[0] <= TOLERANCE_MEV for f in fractions)
        lines.append("  validation at 500: mean difference %+.2f meV; kept model: %s" % (
            early, " | ".join(text("small_output_init", f, "val_mae_meV") for f in fractions)))
        lines.append("  -> %s" % ("ADOPT small_output_init" if early < 0 and final_ok else "keep the zero start"))
    lines.append("")

    lines.append("Issue 8: absolute-error kinks and one rotation per step")
    lines.append("  rule: adopt a variant if its validation gap MAE is within +%.1f meV at every fraction and its"
                 % TOLERANCE_MEV)
    lines.append("        48-image symmetry violation is lower than base (mean over fractions)")
    adopted = []
    for variant in ("smooth_l1", "four_rotations"):
        if not present(variant, fractions):
            continue
        val_ok = all(d(variant, f, "val_mae_meV")[0] <= TOLERANCE_MEV for f in fractions)
        symmetry_change = np.mean([d(variant, f, "symmetry_violation_meV")[0] for f in fractions])
        lines.append("  %s: symmetry violation %+.2f meV vs base; validation gap %s" % (
            variant, symmetry_change, " | ".join(text(variant, f, "val_mae_meV") for f in fractions)))
        if val_ok and symmetry_change < 0:
            adopted.append(variant)
    lines.append("  -> %s" % ("ADOPT " + " and ".join(adopted) if adopted else "keep absolute errors, one rotation"))
    lines.append("")

    lines.append("Regularisation: dropout and early stopping")
    lines.append("  rule: adopt dropout if validation gap MAE is lower on average and within +%.1f meV at every fraction;"
                 % TOLERANCE_MEV)
    lines.append("        adopt early_stop (it saves compute) if validation and test gap MAE are within +%.1f meV "
                 "at every fraction" % TOLERANCE_MEV)
    if present("dropout", fractions):
        mean_change = np.mean([d("dropout", f, "val_mae_meV")[0] for f in fractions])
        ok = all(d("dropout", f, "val_mae_meV")[0] <= TOLERANCE_MEV for f in fractions)
        lines.append("  dropout: validation gap %s" % " | ".join(text("dropout", f, "val_mae_meV") for f in fractions))
        lines.append("  -> %s" % ("ADOPT dropout" if mean_change < 0 and ok else "no dropout"))
    if present("early_stop", fractions):
        ok = all(d("early_stop", f, "val_mae_meV")[0] <= TOLERANCE_MEV and d("early_stop", f, "test_mae_meV")[0]
                 <= TOLERANCE_MEV for f in fractions)
        saved = 1.0 - (all_runs[all_runs["variant"] == "early_stop"]["iterations_run"].mean()
                       / all_runs[all_runs["variant"] == "base"]["iterations_run"].mean())
        lines.append("  early_stop: test gap %s; iterations saved %.0f%%" % (
            " | ".join(text("early_stop", f, "test_mae_meV") for f in fractions), 100 * saved))
        lines.append("  -> %s" % ("ADOPT early_stop" if ok else "no early stopping"))
    lines.append("")
    lines.append("Issue 9: zero-weight slope points. Not applicable here: this file never evaluates them.")

    with open(os.path.join(folder, "decisions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_variant_differences(all_runs, variants, fractions, column, label, path):
    """One panel per labeled fraction: paired difference to base for every variant (blue = lower than base,
    orange = higher; pale when the 95% interval includes 0)."""
    others = [v for v in variants if v != "base"]
    fig, axes = plt.subplots(1, len(fractions), figsize=(2.9 * len(fractions), 0.34 * len(others) + 1.6),
                             sharey=True, squeeze=False)
    for j, fraction in enumerate(fractions):
        ax = axes[0][j]
        for k, variant in enumerate(others):
            mean, half, n = paired_difference(all_runs, variant, fraction, column)
            if n == 0 or np.isnan(mean):
                continue
            if mean < 0:
                color = COLOR_TRAIN
            else:
                color = COLOR_VAL
            alpha = 1.0
            if abs(mean) <= half:
                alpha = 0.35
            ax.barh(k, mean, xerr=half, color=color, alpha=alpha, height=0.65, error_kw={"linewidth": 0.8, "capsize": 2})
        ax.axvline(0, color="#444444", linewidth=0.9)
        light_grid(ax, "x")
        ax.set_title("%d%% labeled" % fraction, loc="left")
        ax.set_xlabel("meV")
    axes[0][0].set_yticks(range(len(others)))
    axes[0][0].set_yticklabels(others)
    axes[0][0].invert_yaxis()
    fig.suptitle("%s (paired by seed, 95%% t interval; pale = interval includes 0)" % label, fontsize=11)
    fig.tight_layout()
    fig.savefig(path + ".png", facecolor="white", bbox_inches="tight")
    fig.savefig(path + ".pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_issue_diagnostics(all_runs, fractions, path):
    """The decision metric of each issue against the labeled fraction, for base and the candidate fixes."""
    panels = [("Issue 1 and 3: band edges", "test_edges_meV", "test CBM + VBM MAE (meV)",
               ["base", "freeze_offsets", "edge_agreement", "isotropic_edges", "penalize_correction"]),
              ("Issue 2: constant drift", "max_constant_change_pct", "largest constant change (%)",
               ["base", "tanh_constants", "penalize_constants", "freeze_offsets"]),
              ("Issue 5: checkpoint choice", "test_mae_meV", "test gap MAE (meV)",
               ["base", "select_last", "select_last_2500", "select_holdout", "early_stop"]),
              ("Issue 6: SOAP switch", "soap_spike_ratio", "loss ratio 12,010 / 12,000",
               ["base", "soap_warmup", "soap_lower_lr"]),
              ("Issue 7: start", "val_at_500_meV", "validation gap MAE at iteration 500 (meV)",
               ["base", "small_output_init"]),
              ("Issue 8 and dropout: symmetry", "symmetry_violation_meV", "48-image symmetry violation (meV)",
               ["base", "smooth_l1", "four_rotations", "dropout"])]
    line_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), squeeze=False)
    x = np.array(fractions, dtype=float)
    for panel, (title, column, label, variants) in zip(axes.flat, panels):
        color_index = 0
        for variant in variants:
            rows = all_runs[all_runs["variant"] == variant]
            if len(rows) == 0:
                continue
            means = []
            for fraction in fractions:
                means.append(rows[rows["fraction"] == fraction][column].mean())
            if variant == "base":
                panel.plot(x, means, "o-", color="#222222", linewidth=1.8, markersize=4, label="base")
            else:
                panel.plot(x, means, "o-", color=line_colors[color_index], linewidth=1.4, markersize=4, label=variant)
                color_index += 1
        panel.set_xscale("symlog", linthresh=1)
        panel.set_xlim(-0.3, x.max() * 1.25)
        panel.set_xticks(x)
        panel.set_xticklabels(["%g" % f for f in fractions])
        panel.set_xlabel("labeled training states (%)")
        panel.set_ylabel(label)
        panel.set_title(title, loc="left")
        light_grid(panel)
        panel.legend(fontsize=7.5)
    fig.suptitle("Decision metric of each issue (mean over the screening seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path + ".png", facecolor="white", bbox_inches="tight")
    fig.savefig(path + ".pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


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


def parse_variants(text):
    """'base' -> ['base'];  'all' -> every variant;  'base,dropout' -> ['base', 'dropout']."""
    if text == "all":
        return list(VARIANTS)
    variants = []
    for name in text.split(","):
        name = name.strip()
        if name == "":
            continue
        if name not in VARIANTS:
            raise SystemExit("unknown variant %s; known: %s" % (name, ", ".join(VARIANTS)))
        variants.append(name)
    return variants


def main():
    parser = argparse.ArgumentParser(description="PHM (see the top of this file)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["constants", "run", "study", "tasks", "report", "compare"]:
        p = commands.add_parser(name)
        if name != "compare":
            p.add_argument("--data_csv", required=True,
                           help="the dataset (columns E_xx ... E_zx, Eg_eV, CBM_eV, VBM_eV, strain_type)")
        p.add_argument("--out", required=True, help="study folder")
        if name in ("run", "report"):
            p.add_argument("--variant", default="base", help="one of: " + ", ".join(VARIANTS))
        if name == "run":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--fraction", type=int, required=True, help="percent of the training states with labels")
        if name in ("constants", "study", "tasks"):
            p.add_argument("--variants", default="base", help="comma-separated names, or 'all'")
            p.add_argument("--seeds", default="%d-%d" % (SEEDS[0], SEEDS[-1]))
            p.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
            p.add_argument("--screen", action="store_true", help="screening study: every variant, seeds %d-%d, "
                           "fractions %s" % (SCREEN_SEEDS[0], SCREEN_SEEDS[-1], ",".join(str(f) for f in SCREEN_FRACTIONS)))
        if name in ("run", "study", "tasks"):
            p.add_argument("--iterations", type=int, default=N_ITERATIONS)
        if name in ("run", "study"):
            p.add_argument("--threads", type=int, default=1, help="threads per run")
        if name == "study":
            p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                           help="runs at the same time")
    args = parser.parse_args()

    if args.command in ("constants", "study", "tasks"):
        if args.screen:
            variants = list(VARIANTS)
            seeds = SCREEN_SEEDS
            fractions = SCREEN_FRACTIONS
        else:
            variants = parse_variants(args.variants)
            seeds = parse_int_list(args.seeds)
            fractions = parse_int_list(args.fractions)

    if args.command == "constants":
        fit_all_constants(args.data_csv, args.out, variants, seeds, fractions)
    elif args.command == "run":
        train_one_run(args.data_csv, args.out, args.variant, args.seed, args.fraction, args.iterations, args.threads)
    elif args.command == "study":
        run_study_locally(args.data_csv, args.out, variants, seeds, fractions, args.iterations, args.jobs,
                          args.threads)
    elif args.command == "tasks":
        write_task_list(sys.argv[0], args.data_csv, args.out, variants, seeds, fractions, args.iterations)
    elif args.command == "report":
        make_report(args.data_csv, args.out, args.variant)
    elif args.command == "compare":
        compare_variants(args.out)


if __name__ == "__main__":
    main()
