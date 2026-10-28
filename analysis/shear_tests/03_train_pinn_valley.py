# Scratch variant of 03_train_pinn.py for the tests of Supplementary Note 8. Identical to the trainer except for three options that
# are off by default (--valley_head, --valley_sup, --focus); with them off it reproduces the reference runs bit for bit.
# Run from the repository root: python analysis/shear_tests/03_train_pinn_valley.py ...
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
# ==============================================================================
# 03_train_pinn.py -- trains ONE physics-informed network and writes its results.
#
# This is the only file in the repository that trains a PINN. Every study script
# (05-11, 15) runs this file as a subprocess with different flags, so there is
# exactly one copy of the training numerics to read, check, or change.
#
# Three models share this backbone and differ only in how each training sample
# is weighted in the loss:
#
#   --model rba   weights follow a fixed residual rule (EMA, updated every 100 epochs)
#   --model sa    weights are learned adversarially (gradient ascent on the same loss)
#   --model mlp   no physics terms at all -- the physics-free control
#
# The Shi ANN baseline is a different architecture and lives in 04_train_shi.py.
#
# Usage
#   python 03_train_pinn.py --model rba --seed 42 --dft_pct 100
#
# Outputs (into --results_dir)
#   epoch_log.csv        one row per epoch
#   test_results.csv     one row per test sample
#   metrics_summary.csv  one summary row for the run
# ==============================================================================

# ============================================================
# SECTION 1: IMPORTS AND ARGUMENT PARSING
# ============================================================
# Defines every command-line option and parses them into `args`. Besides the run
# basics and tuned hyperparameters (defaults reproduce the thesis), this model
# also exposes the ABLATION switches the ablation drivers rely on: --no_rba
# (freeze weights uniform), --no_soap (AdamW only), --drop_loss <name> (zero a
# loss term), and --init_seed (vary init while holding the split, for the
# deep-ensemble UQ). All default to "off", so a bare run is the full model.

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Model name is set from --model once arguments are parsed (see below).
MODEL_NAMES = {"sa": "SA_PINN", "rba": "Dia_RBA_PINN", "mlp": "PhysicsFree_MLP", "sa_paper": "SA_PINN_paper", "rba_paper": "RBA_PINN_paper"}

parser = argparse.ArgumentParser(
    description="Train one physics-informed network (SA, RBA, or physics-free).")
parser.add_argument("--data_csv",    type=str, required=True,
                    help="Path to the dataset CSV")
parser.add_argument("--results_dir", type=str, default=None,
                    help="Output folder for CSVs and checkpoints "
                         "(default: ./results/<model name>)")
parser.add_argument("--seed",        type=int, default=42,
                    help="Random seed (controls the train/val/test split)")
parser.add_argument("--init_seed",   type=int, default=None,
                    help="If set, seeds network initialization independently of "
                         "--seed, holding the data split fixed. Used to build a "
                         "deep ensemble for uncertainty quantification.")
parser.add_argument("--dft_pct",     type=int, default=100,
                    choices=[0, 1, 2, 5, 10, 25, 50, 75, 100],
                    help="Labeled DFT data fraction 0-100")
parser.add_argument("--epochs",      type=int, default=C.TUNED["epochs"],
                    help="Total training epochs")
parser.add_argument("--device",      type=str, default="auto",
                    help="cpu | cuda | auto")
parser.add_argument("--iso_dp_loss", action="store_true",
                    help="Add isotropic DP loss term (Bardeen-Shockley closed-form). "
                         "Ablation flag — off by default.")
parser.add_argument("--model", type=str, default="rba", choices=["sa", "rba", "mlp", "sa_paper", "rba_paper"],
                    help="Which weighting to use. rba = fixed residual rule; "
                         "sa = learned adversarial weights; mlp = no physics at all "
                         "(the physics-free control).")
parser.add_argument("--sa_lr", type=float, default=8.07e-3,
                    help="Ascent learning rate for the self-adaptive weights (tuned).")
parser.add_argument("--no_rba", action="store_true",
                    help="Ablation: freeze RBA weights at 1.0 (uniform).")
parser.add_argument("--input_inv", action="store_true",
                    help="Ablation: feed Oh-invariant scalars instead of the raw 6 "
                         "strain components. The physics loss still sees raw strain, "
                         "so this isolates the input representation.")
parser.add_argument("--no_soap", action="store_true",
                    help="Ablation: keep AdamW for all epochs (no SOAP phase 2).")
parser.add_argument("--drop_loss", action="append", default=[],
                    help="Ablation: zero a loss weight by name "
                         "(cbm, vbm, cons, anchor, pcbm, pvbm, data, iso_eg). "
                         "Repeatable.")

# Config-determining hyperparameters, exposed for the tuning/sensitivity studies.
# Defaults equal the originally hand-set values, so behavior is unchanged when no
# flag is passed. These mirror the SA-PINN backbone so the chosen configuration
# can be applied identically to both weighting variants.
parser.add_argument("--width",      type=int,   default=C.TUNED["width"])
parser.add_argument("--depth",      type=int,   default=C.TUNED["depth"])
# Architecture elements that the thesis never varied. Defaults reproduce the
# thesis network exactly, so behaviour is unchanged unless a flag is passed; the
# point of exposing them is that "not used" then becomes a measured result
# instead of a silence.
parser.add_argument("--norm",       type=str,   default="none",
                    choices=["none", "layer", "batch"],
                    help="normalisation inserted after each hidden activation")
parser.add_argument("--dropout",    type=float, default=0.0,
                    help="dropout probability after each hidden activation")
parser.add_argument("--skip",       type=str,   default="none",
                    choices=["none", "residual"],
                    help="residual connections between equal-width hidden layers")
parser.add_argument("--width_shape", type=str,  default="uniform",
                    choices=["uniform", "taper", "widen"],
                    help="uniform keeps every hidden layer at --width; taper "
                         "halves toward the output; widen doubles toward it")
# Optimizer defaults are the tuned values from the Phase-2 hyperparameter search
# on the shared backbone (see results/hyperparameter/hp_search/), applied identically to both
# weighting variants so the RBA-vs-SA comparison stays paired.
parser.add_argument("--adamw_lr",   type=float, default=2.636e-3)  # tuned
parser.add_argument("--adamw_wd",   type=float, default=1.793e-5)  # tuned
parser.add_argument("--soap_lr",    type=float, default=7.325e-3)  # tuned
parser.add_argument("--soap_wd",    type=float, default=1.224e-3)  # tuned
parser.add_argument("--switch_frac",type=float, default=0.5)       # unchanged
parser.add_argument("--clip_norm",  type=float, default=1.0)       # unchanged
parser.add_argument("--rba_gamma",  type=float, default=0.999)  # RBA EMA decay
parser.add_argument("--rba_eta",    type=float, default=0.1)    # RBA update rate
parser.add_argument("--ckpt_interval", type=int, default=250)
parser.add_argument("--nz_thresh",  type=float, default=0.01)
parser.add_argument("--anc_thresh", type=float, default=1e-4)
parser.add_argument("--scaler",     type=str,   default="standard",
                    choices=["standard", "minmax", "robust"])
parser.add_argument("--w_data",     type=float, default=5.0)
parser.add_argument("--w_cons",     type=float, default=2.0)
parser.add_argument("--w_cbm",      type=float, default=1.0)
parser.add_argument("--w_vbm",      type=float, default=1.0)
parser.add_argument("--w_anchor",   type=float, default=0.5)
parser.add_argument("--w_pcbm",     type=float, default=0.5)
parser.add_argument("--w_pvbm",     type=float, default=0.5)
parser.add_argument("--w_iso_eg",   type=float, default=5.0)
# Cross-validation: when --cv_fold is set, the fixed split is replaced by a
# stratified k-fold partition (test = the given fold). Used by the CV driver only.
parser.add_argument("--cv_fold",    type=int, default=None)
parser.add_argument("--cv_nfolds",  type=int, default=5)
# Partitioning scheme. "stratified" is the adopted 70/10/20 keyed to the seed.
# The others exist so that what the split measures -- interpolation within the
# sampled region, or extrapolation beyond it -- becomes an explicit choice.
parser.add_argument("--split",      type=str,   default="stratified",
                    choices=["stratified", "random", "group", "magnitude", "symmetry"])
parser.add_argument("--holdout_family", type=str, default="shear",
                    help="--split group: the deformation family held out entirely")
parser.add_argument("--magnitude_thresh", type=float, default=0.05,
                    help="--split magnitude: train below this |strain|, test above")
# Early stopping. Off by default: with best-checkpoint selection nothing is lost
# by training on, so stopping early is a compute decision, not an accuracy one.
parser.add_argument("--patience",   type=int,   default=0,
                    help="stop after this many checkpoints without improvement "
                         "(0 = never stop early)")
parser.add_argument("--min_delta",  type=float, default=0.0,
                    help="improvement in eV that counts as progress for --patience")
parser.add_argument("--sa_clamp",   type=float, default=8.0,
                    help="self-adaptive log-weights are clamped to +/- this")
# Design-choice ablations (defaults = adopted choices; behavior unchanged when unset).
parser.add_argument("--activation", type=str, default="silu",
                    choices=["silu", "relu", "tanh", "gelu"])
parser.add_argument("--init",       type=str, default="xavier",
                    choices=["xavier", "kaiming", "orthogonal", "default"])
parser.add_argument("--out_bias_init", type=str, default="mean",
                    choices=["mean", "zero"])
parser.add_argument("--optimizer",  type=str, default="adamw_soap",
                    choices=["adamw_soap", "soap", "adamw", "adam", "lbfgs",
                             "rmsprop", "sgd"])
parser.add_argument("--beta",       type=float, default=50.0)
parser.add_argument("--residual_phys", action="store_true",
                    help="residual-learning baseline: the network predicts corrections to the analytic band edges")
parser.add_argument("--grad_probe_every", type=int, default=0,
                    help="every N epochs log the norms and cosine of the data-term and physics-term gradients to grad_probe.csv")
# Physics-branch switches (18 Oct 2026). All default off; with all off the trainer is unchanged.
parser.add_argument("--sym_colloc", type=float, default=0.0,
                    help="S: weight of the cubic-symmetry collocation loss (rotated training strains must give the same band edges)")
parser.add_argument("--structured_head", action="store_true",
                    help="Q: the network predicts strain-dependent values of the 13 analytic constants; band edges follow from the "
                         "deformation-potential valleys (CBM) and the Bir-Pikus 3x3 eigenvalue problem (VBM)")
parser.add_argument("--curv_w", type=float, default=0.0,
                    help="C: weight of the zero-strain curvature condition d2E/dI1^2 (targets: cubic fit to the training isotropic states)")
parser.add_argument("--loss_pow", type=int, default=1, choices=[1, 2],
                    help="1: mean absolute deviations (default); 2: mean squared deviations. sa_paper / rba_paper always use 2")
parser.add_argument("--sa_paper_lr", type=float, default=5e-3, help="SA-PINN (McClenny & Braga-Neto 2023): ascent learning rate")
parser.add_argument("--rba_paper_gamma", type=float, default=0.999, help="RBA-PINN (Anagnostopoulos et al. 2024): decay")
parser.add_argument("--rba_paper_eta", type=float, default=0.01, help="RBA-PINN: learning rate eta* (upper bound eta*/(1-gamma) = 10)")
parser.add_argument("--nz_mode", type=str, default="volume", choices=["volume", "strain"],
                    help="states the zero-strain slope terms act on: |I1| < nz_thresh (volume, default: includes pure shear) "
                         "or max|eps_ij| < nz_thresh (strain: only small total strain)")
parser.add_argument("--phys_anneal", type=float, default=1.0,
                    help="multiply every physics term by a factor falling linearly from 1 at the first epoch to this value at the last "
                         "(1.0 = constant weight, the default)")
parser.add_argument("--slope_fix", action="store_true",
                    help="divide the summed diagonal partials by 3 so the zero-strain slope terms compare dE/dI1 with its target "
                         "(off reproduces all runs before 18 Oct 2026, which compared 3 dE/dI1 with the target)")
parser.add_argument("--valley_head", action="store_true",
                    help="structured head with six extra outputs: an offset for each conduction valley (zero at initialization)")
parser.add_argument("--valley_sup", type=float, default=0.0,
                    help="weight of a cross-entropy term that matches the model's lowest valley axis to the axis of the labeled CBM k-point; 0 = off")
parser.add_argument("--focus", type=float, default=0.0,
                    help="extra weight (1 + focus) on training states sheared in two or more planes whose two lowest valleys lie within --focus_margin; 0 = off")
parser.add_argument("--focus_margin", type=float, default=0.3, help="valley margin in eV for --focus")
parser.add_argument("--trust_scale", type=float, default=0.0,
                    help="V: weight the analytic-agreement terms per state by exp(-max|eps|/trust_scale); 0 = off")
parser.add_argument("--colloc_n", type=int, default=0,
                    help="formula-only collocation: number of synthetic strain states (six components uniform in "
                         "[-colloc_max, colloc_max], fixed per seed) where the network must agree with the analytic models; 0 = off")
parser.add_argument("--colloc_max", type=float, default=0.10, help="largest strain component of the collocation states")
parser.add_argument("--w_colloc", type=float, default=1.0, help="weight of the collocation agreement term")
args = parser.parse_args()

# The physics-free control is the same network with every physics term switched
# off, so it is expressed as a --drop_loss set rather than a separate model.
if args.model == "mlp":
    for _t in ("cbm", "vbm", "cons", "anchor", "pcbm", "pvbm"):
        if _t not in args.drop_loss:
            args.drop_loss.append(_t)
MODEL_NAME = MODEL_NAMES[args.model]
if args.residual_phys:
    MODEL_NAME += "_residual"
if args.results_dir is None:
    args.results_dir = os.path.join(".", "results", MODEL_NAME)

# Resolve device before any CUDA calls so later code can branch cleanly
if args.device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)

os.makedirs(args.results_dir, exist_ok=True)


# ============================================================
# SECTION 2: REPRODUCIBILITY SETUP
# ============================================================
# Seeds every RNG (Python, NumPy, PyTorch CPU+GPU) and forces deterministic
# cuDNN, so the same --seed gives bit-identical results. Note: the data SPLIT is
# always keyed to --seed, but network INITIALIZATION uses --init_seed when given
# (else --seed) -- that is how the deep ensemble trains many nets on one split.

# Fix all RNG sources so identical seeds produce identical runs.
# The data split is always keyed to --seed; network initialization uses
# --init_seed when provided (deep-ensemble mode), else --seed.
random.seed(args.seed)
np.random.seed(args.seed)
_init_seed = args.seed if args.init_seed is None else args.init_seed
torch.manual_seed(_init_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_init_seed)

# Deterministic cuDNN ops cost a small speed penalty but ensure bit-exact results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# SECTION 3: DATA LOADING AND FEATURE CONSTRUCTION
# ============================================================
# Turns the CSV into ready-to-train arrays: reads the 6 strain inputs and
# Eg/CBM/VBM targets; splits 70/10/20 STRATIFIED by strain type and keyed to
# --seed; feature-scales fitting the scaler on TRAIN ONLY (no leakage); picks the
# labeled subset of size --dft_pct (the data-scarcity lever; 0% = physics only);
# and builds the near-zero-strain masks the boundary losses need. Returns one
# `data` dict. Identical to the SA-PINN loader so the two are directly comparable.

def compute_raw_features(df):
    # Raw 6 Green-Lagrange strain components as network inputs.
    # Keeps shear cross-coupling (Exy, Eyz, Ezx as independent axes) that the
    # Oh-invariant I5 = Exy²+Eyz²+Ezx² collapses, which caused shear MAE saturation.
    cols = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
    return df[cols].values.astype(np.float32)


def compute_invariant_features(df):
    # Oh-symmetric scalar invariant inputs -- the alternative representation the
    # raw components are compared against. Six invariants so the input width is
    # unchanged and only the REPRESENTATION differs: the three principal
    # invariants of the strain tensor plus the cubic-symmetric shear terms.
    # Every shear enters through an even-degree combination, which is exactly the
    # sign information this representation is unable to carry.
    xx = df["E_xx"].values; yy = df["E_yy"].values; zz = df["E_zz"].values
    xy = df["E_xy"].values; yz = df["E_yz"].values; zx = df["E_zx"].values
    i1 = xx + yy + zz
    i2 = xx*yy + yy*zz + zz*xx - (xy**2 + yz**2 + zx**2)
    i3 = xx*yy*zz + 2.0*xy*yz*zx - xx*yz**2 - yy*zx**2 - zz*xy**2
    i4 = xx**2 + yy**2 + zz**2
    i5 = xy**2 + yz**2 + zx**2
    i6 = xx*(xy**2 + zx**2) + yy*(xy**2 + yz**2) + zz*(yz**2 + zx**2)
    return np.stack([i1, i2, i3, i4, i5, i6], axis=1).astype(np.float32)


def _normalize_strain_type(s):
    # Map CSV subtypes (uniaxial_x, biaxial_xy, unstrained, …) to the 5 canonical
    # types used throughout the output CSVs and thesis analysis.
    s = str(s).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"   # 'unstrained' and anything else → isotropic


def load_and_split(data_csv, seed, dft_pct, cv_fold=None, cv_nfolds=5,
                   input_inv=False):
    df = pd.read_csv(data_csv)

    # Representation lever: raw components (default) vs Oh-invariant scalars.
    # strain_all below stays RAW either way, so the physics loss is unchanged and
    # the ablation isolates the input representation alone.
    X_all     = compute_invariant_features(df) if input_inv else compute_raw_features(df)
    strain_all = df[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values.astype(np.float32)
    eg_all    = df["Eg_eV"].values.astype(np.float32)
    cbm_all   = df["CBM_eV"].values.astype(np.float32)
    vbm_all   = df["VBM_eV"].values.astype(np.float32)
    stype_all = np.array([_normalize_strain_type(s) for s in df["strain_type"].values])
    shear_all = df["shear_planes"].values if "shear_planes" in df.columns \
                else np.full(len(df), "", dtype=object)
    idx_all   = np.arange(len(df))
    K = np.abs(df[["cbm_kx", "cbm_ky", "cbm_kz"]].values); clear = (K.min(1) < 0.05) & (K.max(1) > 0.2)
    axis_all = np.where(clear, np.array([2, 1, 0])[K.argmin(1)], -1)      # k on a zone-face line: the zero coordinate names the valley axis

    if cv_fold is None and args.split == "stratified":
        # Adopted: stratified 70/10/20. Stratifying keeps the rare families
        # represented (isotropic is only 47 rows), and measures interpolation
        # within the sampled region.
        idx_trval, idx_te = train_test_split(
            idx_all, test_size=0.20, random_state=seed, stratify=stype_all
        )
        # 0.125 of the 80% train+val pool = 10% of total
        idx_tr, idx_va = train_test_split(
            idx_trval, test_size=0.125, random_state=seed, stratify=stype_all[idx_trval]
        )
    elif cv_fold is None and args.split == "random":
        # The same proportions with no stratification. Its only purpose is to
        # show what stratification is worth: if the two agree, stratification is
        # not carrying the result.
        idx_trval, idx_te = train_test_split(
            idx_all, test_size=0.20, random_state=seed)
        idx_tr, idx_va = train_test_split(
            idx_trval, test_size=0.125, random_state=seed)
    elif cv_fold is None and args.split == "group":
        # One deformation family held out entirely. This is extrapolation to an
        # unseen kind of deformation, not interpolation, and it is the harder
        # claim a strain-engineering reader actually wants.
        held = _normalize_strain_type(args.holdout_family)
        idx_te = idx_all[stype_all == held]
        pool   = idx_all[stype_all != held]
        if len(idx_te) == 0:
            raise SystemExit("--holdout_family %s matches no rows" % args.holdout_family)
        idx_tr, idx_va = train_test_split(
            pool, test_size=0.125, random_state=seed, stratify=stype_all[pool])
    elif cv_fold is None and args.split == "symmetry":
        # Symmetry-grouped split: strains related by one of the 48 cubic operations carry the same labels, so every
        # symmetry class goes to a single partition. Classes are drawn per family (70/10/20 of the family's states),
        # with this seed, so no test state has a rotated twin in training or validation.
        import itertools
        ops = [np.diag(sg) @ np.eye(3)[list(pm)] for pm in itertools.permutations(range(3)) for sg in itertools.product([1, -1], repeat=3)]
        keys = []
        for v in strain_all.astype(np.float64):
            T = np.array([[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]])
            imgs = []
            for R in ops:
                Tr = R @ T @ R.T
                imgs.append(tuple(np.round([Tr[0, 0], Tr[1, 1], Tr[2, 2], Tr[0, 1], Tr[1, 2], Tr[0, 2]], 5)))
            keys.append(min(imgs))
        uniq = {}
        cls = np.array([uniq.setdefault(k, len(uniq)) for k in keys])
        rng_s = np.random.default_rng(seed)
        idx_tr, idx_va, idx_te = [], [], []
        for fam in np.unique(stype_all):
            fam_idx = idx_all[stype_all == fam]; fam_cls = np.unique(cls[fam_idx]); rng_s.shuffle(fam_cls)
            n = len(fam_idx); te_n, va_n = 0, 0
            for c in fam_cls:
                members = fam_idx[cls[fam_idx] == c]
                if te_n < round(0.20 * n): idx_te.extend(members); te_n += len(members)
                elif va_n < round(0.10 * n): idx_va.extend(members); va_n += len(members)
                else: idx_tr.extend(members)
        idx_tr, idx_va, idx_te = np.array(sorted(idx_tr)), np.array(sorted(idx_va)), np.array(sorted(idx_te))
    elif cv_fold is None and args.split == "magnitude":
        # Train on small strains, test on large ones. This asks whether the model
        # extrapolates beyond the strain range it was shown -- the question a
        # device application turns on.
        mag = np.max(np.abs(strain_all), axis=1)
        idx_te = idx_all[mag > args.magnitude_thresh]
        pool   = idx_all[mag <= args.magnitude_thresh]
        if len(idx_te) == 0 or len(pool) < 10:
            raise SystemExit("--magnitude_thresh %g leaves %d train / %d test"
                             % (args.magnitude_thresh, len(pool), len(idx_te)))
        idx_tr, idx_va = train_test_split(pool, test_size=0.125, random_state=seed)
    else:
        # Stratified k-fold (test = fold cv_fold); used only by the CV driver.
        skf = StratifiedKFold(n_splits=cv_nfolds, shuffle=True, random_state=seed)
        folds = list(skf.split(idx_all, stype_all))
        idx_trval, idx_te = folds[cv_fold][0], folds[cv_fold][1]
        idx_tr, idx_va = train_test_split(
            idx_trval, test_size=0.125, random_state=seed,
            stratify=stype_all[idx_trval]
        )

    # Scaler fitted on training features only — prevents leakage into val/test normalization.
    # Choice exposed via --scaler for the input-scaling sensitivity study.
    if args.scaler == "minmax":
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    elif args.scaler == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[idx_tr]).astype(np.float32)
    if isinstance(scaler, StandardScaler) and (scaler.scale_ < 1e-6).any():
        # A strain component that is only round-off in training (the shear components when shear is held out, ~1e-17)
        # would otherwise be divided by ~1e-17 and blow up on the test states; give it the typical strain scale.
        tiny = scaler.scale_ < 1e-6
        scaler.scale_[tiny] = np.median(scaler.scale_[~tiny]); scaler.mean_[tiny] = 0.0
        X_tr = scaler.transform(X_all[idx_tr]).astype(np.float32)
        print("  scaler: %d near-constant strain component(s) given the typical scale %.4f" % (int(tiny.sum()), scaler.scale_[tiny][0]))
    X_va = scaler.transform(X_all[idx_va]).astype(np.float32)
    X_te = scaler.transform(X_all[idx_te]).astype(np.float32)

    # Labeled DFT subset — separate RNG offset avoids coupling seed to split seed
    rng = np.random.default_rng(seed + 9999)
    if dft_pct == 0:
        dft_idx = np.array([], dtype=np.int64)
    else:
        n_labeled = max(1, round(len(idx_tr) * dft_pct / 100.0))
        dft_idx = rng.choice(len(idx_tr), size=n_labeled, replace=False)

    # Near-zero-strain masks for boundary condition losses
    vol_tr = strain_all[idx_tr, 0] + strain_all[idx_tr, 1] + strain_all[idx_tr, 2]
    nz_mask  = np.abs(vol_tr) < args.nz_thresh    # gradient BC: dCBM/dI1, dVBM/dI1
    if args.nz_mode == "strain":
        nz_mask = np.abs(strain_all[idx_tr]).max(axis=1) < args.nz_thresh
    anc_mask = np.abs(vol_tr) < args.anc_thresh   # anchor BC: predict reference band positions

    return dict(
        X_tr=X_tr, X_va=X_va, X_te=X_te,
        strain_tr=strain_all[idx_tr],
        eg_tr=eg_all[idx_tr],   cbm_tr=cbm_all[idx_tr],   vbm_tr=vbm_all[idx_tr],
        eg_va=eg_all[idx_va],   cbm_va=cbm_all[idx_va],   vbm_va=vbm_all[idx_va],
        eg_te=eg_all[idx_te],   cbm_te=cbm_all[idx_te],   vbm_te=vbm_all[idx_te],
        stype_te=stype_all[idx_te],
        stype_tr=stype_all[idx_tr],
        shear_te=shear_all[idx_te],
        idx_te=idx_te,
        axis_tr=axis_all[idx_tr],
        dft_idx=dft_idx,
        nz_mask=nz_mask,
        anc_mask=anc_mask,
        cbm_mean_tr=float(np.mean(cbm_all[idx_tr])),
        vbm_mean_tr=float(np.mean(vbm_all[idx_tr])),
        scaler=scaler,
        N_tr=len(idx_tr),
    )


def make_iso_synth_points(scaler, n=200, eps_range=0.08):
    # Synthetic isotropic strain points: Exx=Eyy=Ezz=eps, all shear=0.
    # Spans the full isotropic range in the dataset (~±8% volumetric strain).
    # These are physics-only — no DFT labels needed — so they inject
    # Bardeen-Shockley DP theory directly into the sparse isotropic subspace.
    eps_vals = np.linspace(-eps_range, eps_range, n).astype(np.float32)
    raw = np.zeros((n, 6), dtype=np.float32)
    raw[:, 0] = eps_vals   # Exx
    raw[:, 1] = eps_vals   # Eyy
    raw[:, 2] = eps_vals   # Ezz
    # shear columns stay zero — pure isotropic
    return scaler.transform(raw).astype(np.float32)


# ============================================================
# SECTION 4: PHYSICS MODEL
# ============================================================
# The analytic theory the loss compares against: physics_cbm (6-valley
# deformation-potential model) and physics_vbm (Bir-Pikus eigenvalue model),
# imported -- not redefined -- from physics_terms.py so this model, the SA PINN,
# and the zero-label solver all use byte-identical equations. The local wrapper
# just supplies the soft-min sharpness --beta when none is given.

# Physics is defined ONCE in physics_terms.py (the single source of truth shared
# with the SA PINN and the 0% solver baselines). Import the constants and the
# VBM model directly; wrap physics_cbm so that, with no explicit beta, training
# uses the soft-min at args.beta exactly as before.
from physics import (physics_cbm as _physics_cbm_softmin,
                           physics_vbm, CBM_0, VBM_0, AV, XI_D, XI_U)


def physics_cbm(strain, beta=None):
    if beta is None:
        beta = args.beta
    return _physics_cbm_softmin(strain, beta=beta)


import physics as _PHYS
THETA_KEYS = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2", "VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]


def cbm_from_theta(strain, th, beta, offsets=None, valleys=False):
    # Same deformation-potential valley model as physics.physics_cbm, with per-state constants th[:, 0:7].
    Exx, Eyy, Ezz, Exy, Eyz, Ezx = [strain[:, i] for i in range(6)]
    I1 = Exx + Eyy + Ezz
    c0, xd, xu, xup, xq, xc, xq2 = [th[:, i] for i in range(7)]
    def valley(e):
        return xd * I1 + xu * e + xq * e ** 2 + xc * I1 * e + xq2 * I1 ** 2
    d = torch.stack([valley(Ezz) + 2 * xup * Exy, valley(Ezz) - 2 * xup * Exy, valley(Eyy) + 2 * xup * Ezx,
                     valley(Eyy) - 2 * xup * Ezx, valley(Exx) + 2 * xup * Eyz, valley(Exx) - 2 * xup * Eyz], dim=1)
    if offsets is not None:
        d = d + offsets
    if valleys:
        return d
    return c0 - torch.logsumexp(-beta * d, dim=1) / beta


def vbm_from_theta(strain, th, eps=1e-8):
    # Same Bir-Pikus 3x3 model as physics.physics_vbm, with per-state constants th[:, 7:13].
    Exx, Eyy, Ezz, Exy, Eyz, Ezx = [strain[:, i] for i in range(6)]
    I1 = Exx + Eyy + Ezz
    v0, av, av2, b, b2, dbp = [th[:, i] for i in range(7, 13)]
    hydro = av * I1 + av2 * I1 ** 2
    dxx, dyy, dzz = Exx - I1 / 3.0, Eyy - I1 / 3.0, Ezz - I1 / 3.0
    H = torch.zeros(strain.shape[0], 3, 3, dtype=strain.dtype, device=strain.device)
    H[:, 0, 0] = hydro + (b + b2 * dxx) * dxx
    H[:, 1, 1] = hydro + (b + b2 * dyy) * dyy
    H[:, 2, 2] = hydro + (b + b2 * dzz) * dzz
    H[:, 0, 1] = dbp * Exy; H[:, 1, 0] = dbp * Exy
    H[:, 1, 2] = dbp * Eyz; H[:, 2, 1] = dbp * Eyz
    H[:, 0, 2] = dbp * Ezx; H[:, 2, 0] = dbp * Ezx
    return v0 + _max_eig_sym3(H)


def _max_eig_sym3(H, beta=200.0):
    # Smooth maximum over the three eigenvalues of a symmetric 3x3 matrix: (1/beta) log tr exp(beta H), the counterpart
    # of the soft minimum used for the conduction valleys. Smooth to all orders, including where the valence bands are
    # degenerate (zero strain, pure shear), where eigvalsh and the closed form give unusable second derivatives.
    # Deviation from the exact maximum on the dataset: 1.6 meV mean, 5.5 meV at the triple degeneracy (log 3 / beta).
    # Shift by the exact largest eigenvalue (no gradient through it), so the largest exponent is exactly 0 and the trace
    # is >= 1: no underflow to log(0) however far the learned constants drift (a Gershgorin bound was too loose).
    with torch.no_grad():
        u = torch.linalg.eigvalsh(torch.nan_to_num(H.double()))[:, 2].to(H.dtype)
    eye = torch.eye(3, dtype=H.dtype, device=H.device).unsqueeze(0)
    ex = torch.linalg.matrix_exp(beta * (H - u[:, None, None] * eye))
    return u + torch.log(torch.diagonal(ex, dim1=1, dim2=2).sum(1)) / beta


# ============================================================
# SECTION 5: NEURAL NETWORK ARCHITECTURE
# ============================================================
# A plain MLP: 6 scaled strain components -> [depth x width] hidden layers -> 2
# outputs (CBM, VBM). The bandgap is computed as Eg = CBM - VBM in forward(), so
# it is exact by construction. Same architecture as the SA PINN (default SiLU /
# Xavier / bias=train means); SiLU's smoothness is required by the gradient-BC
# loss in Section 6.

def _make_activation(name):
    return {"silu": nn.SiLU, "relu": nn.ReLU,
            "tanh": nn.Tanh, "gelu": nn.GELU}[name]()


def _apply_init(linear, scheme):
    if scheme == "xavier":
        nn.init.xavier_normal_(linear.weight)
    elif scheme == "kaiming":
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
    elif scheme == "orthogonal":
        nn.init.orthogonal_(linear.weight)
    if scheme != "default":
        nn.init.zeros_(linear.bias)


def _hidden_sizes():
    """Widths of the hidden layers, before the 2-unit output.

    Uniform is the thesis network. Taper and widen exist so that "every hidden
    layer is the same width" stops being an unexamined default: they keep the
    depth and roughly the parameter budget while moving capacity toward the input
    or toward the output.
    """
    if args.width_shape == "uniform":
        return [args.width] * args.depth
    out = []
    w = args.width
    for i in range(args.depth):
        out.append(max(8, int(round(w))))
        w = w * (0.75 if args.width_shape == "taper" else 1.0 / 0.75)
    return out


class PINNModel(nn.Module):
    """MLP: 6 strain components -> depth x width hidden -> (CBM, VBM).

    Eg = CBM - VBM is structural, not learned, so the identity holds exactly.

    Normalisation, dropout and residual connections are all off by default,
    which reproduces the thesis network. They are wired in rather than absent so
    that their exclusion rests on a measurement (11_decide_architecture.py)
    rather than on never having tried them.
    """

    def __init__(self, cbm_mean, vbm_mean, scaler=None):
        super().__init__()
        widths = _hidden_sizes()
        sizes = [6] + widths

        self.blocks = nn.ModuleList()
        for i in range(len(sizes) - 1):
            block = [nn.Linear(sizes[i], sizes[i + 1])]
            # Normalisation goes after the activation: the residual add (below)
            # must see an already-normalised tensor, or the two interact.
            block.append(_make_activation(args.activation))
            if args.norm == "layer":
                block.append(nn.LayerNorm(sizes[i + 1]))
            elif args.norm == "batch":
                block.append(nn.BatchNorm1d(sizes[i + 1]))
            if args.dropout > 0.0:
                block.append(nn.Dropout(args.dropout))
            self.blocks.append(nn.Sequential(*block))

        # A residual add needs matching shapes, so it is only possible where a
        # block does not change the width. Recorded per block rather than assumed.
        self.residual = [args.skip == "residual" and sizes[i] == sizes[i + 1]
                         for i in range(len(sizes) - 1)]

        self.out = nn.Linear(widths[-1], 2)

        for blk in self.blocks:
            for m in blk:
                if isinstance(m, nn.Linear):
                    _apply_init(m, args.init)
        _apply_init(self.out, args.init)

        # Output biases initialised to training-set means (default); --out_bias_init
        # zero disables this for the ablation.
        if args.out_bias_init == "mean":
            with torch.no_grad():
                self.out.bias[0] = cbm_mean
                self.out.bias[1] = vbm_mean

        # Residual-learning baseline: outputs are corrections added to the analytic
        # band edges evaluated on the raw strain, recovered from the scaled input.
        self.residual_phys = bool(args.residual_phys)
        if self.residual_phys:
            if scaler is None or not hasattr(scaler, "mean_"):
                raise SystemExit("--residual_phys needs the standard scaler (mean_/scale_)")
            self.register_buffer("in_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("in_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            with torch.no_grad():
                self.out.bias.zero_()      # the analytic models supply the level

        # Structured head: 13 outputs = changes to the analytic constants, zero at initialisation, so the
        # untrained model reproduces the analytic models exactly.
        self.structured = bool(args.structured_head)
        if self.structured:
            if scaler is None or not hasattr(scaler, "mean_"):
                raise SystemExit("--structured_head needs the standard scaler (mean_/scale_)")
            self.valley = bool(args.valley_head)
            self.out = nn.Linear(widths[-1], 13 + (6 if self.valley else 0))
            with torch.no_grad():
                self.out.weight.zero_(); self.out.bias.zero_()
            th0 = [float(getattr(_PHYS, k)) for k in THETA_KEYS]
            sc = [max(abs(v), 1.0) * 0.25 for v in th0]; sc[-1] = 2.0     # D_BP is fitted to ~0
            self.register_buffer("theta0", torch.tensor(th0, dtype=torch.float32))
            self.register_buffer("theta_scale", torch.tensor(sc, dtype=torch.float32))
            self.register_buffer("in_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("in_scale", torch.tensor(scaler.scale_, dtype=torch.float32))

    def valley_axis_logp(self, x):
        # log-probability of each cubic valley axis (z, y, x) from the soft minimum over the two valleys of the axis
        h = x
        for blk, res in zip(self.blocks, self.residual):
            y = blk(h); h = h + y if res else y
        o = self.out(h); raw = x * self.in_scale + self.in_mean
        th = self.theta0 + self.theta_scale * o[:, :13]
        d = cbm_from_theta(raw, th, args.beta, offsets=0.2 * o[:, 13:] if self.valley else None, valleys=True)
        la = torch.stack([torch.logsumexp(-args.beta * d[:, 2 * a:2 * a + 2], dim=1) for a in range(3)], dim=1)
        return torch.log_softmax(la, dim=1)

    def forward(self, x):
        h = x
        for blk, res in zip(self.blocks, self.residual):
            y = blk(h)
            h = h + y if res else y
        o = self.out(h)
        if self.structured:
            raw = x * self.in_scale + self.in_mean
            if self.valley:
                th = self.theta0 + self.theta_scale * o[:, :13]
                cbm = cbm_from_theta(raw, th, args.beta, offsets=0.2 * o[:, 13:])
            else:
                th = self.theta0 + self.theta_scale * o
                cbm = cbm_from_theta(raw, th, args.beta)
            vbm = vbm_from_theta(raw, th)
            return cbm, vbm, cbm - vbm
        cbm = o[:, 0]
        vbm = o[:, 1]
        if self.residual_phys:
            raw = x * self.in_scale + self.in_mean
            cbm = cbm + physics_cbm(raw)
            vbm = vbm + physics_vbm(raw)
        eg  = cbm - vbm
        return cbm, vbm, eg


# ============================================================
# SECTION 6: LOSS FUNCTION
# ============================================================
# The composite loss the network minimises: a weighted sum of data + physics +
# boundary terms (same six terms as the SA PINN -- see compute_loss below). What
# is UNIQUE to this model lives here: the RBA weighting. Each sample has a weight
# w = 1 + r, where r is an exponential moving average of that sample's normalized
# physics residual (init_rba_weights starts r at 0 -> w = 1 uniform; update_rba
# grows r for stubborn samples). Unlike the SA PINN there is no learned min-max
# step -- the weights follow a fixed rule (gamma, eta). The block right below also
# applies the --drop_loss ablation by zeroing chosen W_* term weights.

# Loss term weights — taken from the CLI (defaults equal the hand-set values) so
# the tuning/sensitivity studies can vary them. The --drop_loss ablation below
# still zeroes selected weights on top of whatever base value is set here.
W_CBM    = args.w_cbm     # physics residual: predicted CBM vs deformation-potential CBM
W_VBM    = args.w_vbm     # physics residual: predicted VBM vs Bir-Pikus VBM
W_CONS   = args.w_cons    # Eg consistency: predicted Eg vs physics Eg = CBM_phys - VBM_phys
W_ANCHOR = args.w_anchor  # anchor: near-zero strain predicts reference band positions
W_PCBM   = args.w_pcbm    # gradient BC: dCBM/dI1 at small strain matches deformation potential
W_PVBM   = args.w_pvbm    # gradient BC: dVBM/dI1 at small strain matches Bir-Pikus reference
W_DATA   = args.w_data    # supervised data: predicted CBM/VBM vs DFT labels (labeled set only)
W_ISO    = 1.0            # isotropic DP loss: enforce Bardeen-Shockley closed-form on synthetic iso points (ablation flag)
W_ISO_EG = args.w_iso_eg  # direct Eg supervision on isotropic training samples — same weight as W_DATA
                          # because the DP/BP physics model has >89 meV systematic error there

# Reference gradients at zero strain used for gradient BCs
# Slope of the conduction edge with respect to the trace of the strain
# tensor at zero strain. Derived from the constants rather than written
# out, so it cannot fall out of step with them when they are refitted.
A_C_REF = XI_D + XI_U / 3.0
A_V_REF = AV           # dVBM/dI1 [eV] — equals linear hydrostatic term at zero strain

# Isotropic Eg gradient BC — fitted directly from HSE06-VASP isotropic subset.
# The physics model's dEg/dI1 = -1.20 eV but DFT gives -2.36 eV (factor-of-2 error)
# because the isotropic limit couples CBM valley degeneracy and VBM hydrostatic terms
# in a way the individual DP/BP constants don't reproduce.
# Quadratic: Eg(I1) = A_EG_ISO_0 + A_EG_ISO_L*I1 + A_EG_ISO_Q*I1^2
# -> dEg/dI1(I1) = A_EG_ISO_L + 2*A_EG_ISO_Q*I1
# Enforced on all near-isotropic training points (shear~0, diagonal~equal).
A_EG_ISO_L =  -2.3643   # linear isotropic deformation potential [eV] from HSE06 fit
A_EG_ISO_Q =   1.6653   # quadratic correction [eV] from HSE06 fit
W_PISO     =   0.5      # weight matching existing gradient BC terms

# RBA hyper-parameters (exposed via CLI; defaults are the hand-set values)
RBA_GAMMA = args.rba_gamma
RBA_ETA   = args.rba_eta

# Ablation: zero selected loss weights named via --drop_loss.
# compute_loss reads these as module globals, so reassigning them here
# takes effect for the whole run.
_DROP_MAP = {
    "cbm": "W_CBM", "vbm": "W_VBM", "cons": "W_CONS",
    "anchor": "W_ANCHOR", "pcbm": "W_PCBM", "pvbm": "W_PVBM",
    "data": "W_DATA", "iso_eg": "W_ISO_EG",
}
for _drop_name in args.drop_loss:
    _gvar = _DROP_MAP.get(_drop_name)
    if _gvar is not None:
        globals()[_gvar] = 0.0


def init_rba_weights(N_tr, device):
    # The RBA state r: one scalar per training sample. Start at 0 so the first
    # epochs run with uniform weight w = 1 + r = 1 (no attention yet); attention
    # only builds up as update_rba accumulates residuals.
    return torch.zeros(N_tr, device=device)


def update_rba(rba_w, model, X_tr_t, eg_phys_tr):
    # The RBA rule (called periodically from the training loop). For each sample,
    # measure how far the current prediction is from the physics target, normalize
    # by the worst residual in the batch (-> 0..1), and blend it into the running
    # state with an exponential moving average:  r <- gamma*r + eta*normalized_res.
    # gamma (0.999) = memory of past residuals; eta (0.1) = how fast new residuals
    # enter. Net effect: persistently hard samples drift to high weight, easy ones
    # stay near 1. In-place (rba_w[:]=) so the same tensor is reused every update.
    model.eval()
    with torch.no_grad():
        _, _, eg_pred = model(X_tr_t)
    residuals = torch.abs(eg_pred - eg_phys_tr)
    norm_r = residuals / (residuals.max() + 1e-10)
    rba_w[:] = RBA_GAMMA * rba_w + RBA_ETA * norm_r
    model.train()


# ------------------------------------------------------------------------------
# compute_loss() — OUTLINE (read this first; full detail at each term below)
# ------------------------------------------------------------------------------
# One weighted scalar the network minimises, blending theory (physics) with DFT
# data so the model stays accurate even with few labels. Same six terms as the
# SA PINN; the only difference is that the per-sample weight w comes from the RBA
# rule (w = 1 + EMA residual) instead of a learned parameter. Read top to bottom:
#
#   PRELUDE        forward pass; analytic physics targets; RBA per-sample weights
#   ISO MASK       split isotropic vs non-isotropic (physics unreliable on iso)
#   TERM 1  L_cbm/L_vbm/L_cons  physics residuals  -> predicted bands vs theory
#   TERM 2  L_anchor            anchor BC          -> bands sit right at eps=0
#   TERM 3  L_pcbm/L_pvbm       gradient BC        -> band SLOPES right at eps=0
#   TERM 4  L_data              supervised loss    -> fit DFT labels (if any)
#   TERM 5  L_iso_eg            isotropic Eg sup.  -> fit DFT gap on isotropic
#   TERM 6  L_iso               synthetic iso DP   -> ablation only (off default)
#   SUM            total = weighted sum of all terms; returned to the optimizer
#
# Boundary/physics terms (1-3) need no labels, so they keep working at 0% data.
# ------------------------------------------------------------------------------
def compute_loss(model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                 w, dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                 eg_tr_t=None, X_iso_t=None, strain_iso_t=None, return_parts=False):
    # PRELUDE: predictions, analytic physics targets, and RBA per-sample weights.
    cbm_pred, vbm_pred, eg_pred = model(X_tr_t)
    if args.focus > 0:
        w = w * BRANCH["focus_w"]

    cbm_phys = physics_cbm(strain_tr_t)
    vbm_phys = physics_vbm(strain_tr_t)
    eg_phys  = cbm_phys - vbm_phys

    # Sample weights: hard samples (large residual) get w > 1, easy samples stay near 1
    # `w` arrives ready to use: (rba_w + 1) for RBA, softplus(lambda) for SA,
    # all-ones for the physics-free control. Computing it in the caller is what
    # lets one loss function serve all three models.

    # Isotropic mask: shear near zero AND diagonal components nearly equal.
    # The DP/BP physics model has >89 meV systematic error in the isotropic subspace
    # (verified against HSE06 DFT) so physics residual losses are suppressed there.
    # The gradient BC and DFT data loss remain active for all samples including isotropic.
    shear_mag = (strain_tr_t[:, 3].abs() + strain_tr_t[:, 4].abs() + strain_tr_t[:, 5].abs())
    diag = strain_tr_t[:, :3]
    diag_spread = (diag - diag.mean(dim=1, keepdim=True)).abs().max(dim=1).values
    iso_mask = (shear_mag < 1e-4) & (diag_spread < 1e-4)   # True = isotropic, exclude from phys loss
    non_iso  = ~iso_mask

    if args.trust_scale > 0:
        # V: trust in the analytic models fades with strain size (they are a small-strain expansion)
        w_ag = w * torch.exp(-strain_tr_t.abs().max(dim=1).values / args.trust_scale)
    else:
        w_ag = w
    if non_iso.sum() > 0:
        L_cbm  = W_CBM  * (w_ag[non_iso] * torch.abs(cbm_pred[non_iso] - cbm_phys[non_iso])).mean()
        L_vbm  = W_VBM  * (w_ag[non_iso] * torch.abs(vbm_pred[non_iso] - vbm_phys[non_iso])).mean()
        L_cons = W_CONS * (w_ag[non_iso] * torch.abs(eg_pred[non_iso]  - eg_phys[non_iso] )).mean()
    else:
        L_cbm  = torch.tensor(0.0, device=X_tr_t.device)
        L_vbm  = torch.tensor(0.0, device=X_tr_t.device)
        L_cons = torch.tensor(0.0, device=X_tr_t.device)

    # Anchor BC: essentially-unstrained samples should predict reference positions
    if X_anc_t.shape[0] > 0:
        cbm_a, vbm_a, _ = model(X_anc_t)
        L_anchor = W_ANCHOR * (
            torch.abs(cbm_a - CBM_0).mean() +
            torch.abs(vbm_a - VBM_0).mean()
        )
    else:
        L_anchor = torch.tensor(0.0, device=X_tr_t.device)

    # Gradient BCs: require autograd through the network at near-zero-strain points
    X_nz_g = X_nz_t.detach().requires_grad_(True)
    cbm_nz, vbm_nz, _ = model(X_nz_g)
    grad_cbm = torch.autograd.grad(cbm_nz.sum(), X_nz_g, create_graph=True)[0]
    grad_vbm = torch.autograd.grad(vbm_nz.sum(), X_nz_g, create_graph=True)[0]
    # dCBM/dI1 = sum of partials over Exx,Eyy,Ezz (indices 0,1,2), each divided by
    # its own scaler scale to convert from scaled-input space back to eV/strain units.
    sx = i1_scale[0]
    sy = i1_scale[1]
    sz = i1_scale[2]
    dCBM_dI1 = grad_cbm[:, 0]/sx + grad_cbm[:, 1]/sy + grad_cbm[:, 2]/sz
    dVBM_dI1 = grad_vbm[:, 0]/sx + grad_vbm[:, 1]/sy + grad_vbm[:, 2]/sz
    if args.slope_fix:
        # The sum of the three partials is the derivative along (t, t, t), i.e. 3 dE/dI1 (I1 = 3t).
        # Without this flag the target (a dE/dI1 value) is compared with 3 dE/dI1, as in every run before 18 Oct 2026.
        dCBM_dI1 = dCBM_dI1 / 3.0
        dVBM_dI1 = dVBM_dI1 / 3.0
    L_pcbm = W_PCBM * torch.abs(dCBM_dI1 - A_C_REF).mean()
    L_pvbm = W_PVBM * torch.abs(dVBM_dI1 - A_V_REF).mean()

    # Data terms — only active when labeled DFT samples exist
    if dft_idx_t.numel() > 0:
        w_dft = w[dft_idx_t]
        L_data = W_DATA * (
            (w_dft * torch.abs(cbm_pred[dft_idx_t] - cbm_dft_t)).mean() +
            (w_dft * torch.abs(vbm_pred[dft_idx_t] - vbm_dft_t)).mean()
        )
    else:
        L_data = torch.tensor(0.0, device=X_tr_t.device)

    # Direct Eg supervision on isotropic training samples.
    # The DP/BP physics model has wrong slope and offset in the isotropic subspace
    # (verified: dEg/dI1_phys=-1.20 eV vs DFT=-2.36 eV, +61 meV offset).
    # Physics residual loss is already masked for these samples above.
    # This term uses DFT Eg labels directly — active for all isotropic samples
    # regardless of dft_idx, since isotropic samples are always fully labeled.
    if iso_mask.sum() > 0 and eg_tr_t is not None:
        L_iso_eg = W_ISO_EG * (w[iso_mask] * torch.abs(
            eg_pred[iso_mask] - eg_tr_t[iso_mask]
        )).mean()
    else:
        L_iso_eg = torch.tensor(0.0, device=X_tr_t.device)

    # Isotropic DP loss (Bardeen-Shockley): enforce physics_cbm/vbm closed-form
    # on synthetic isotropic points spanning the full volumetric strain range.
    # Active only when --iso_dp_loss is set — kept as an ablation flag.
    if X_iso_t is not None and strain_iso_t is not None:
        cbm_iso_pred, vbm_iso_pred, _ = model(X_iso_t)
        cbm_iso_phys = physics_cbm(strain_iso_t)
        vbm_iso_phys = physics_vbm(strain_iso_t)
        L_iso = W_ISO * (
            torch.abs(cbm_iso_pred - cbm_iso_phys).mean() +
            torch.abs(vbm_iso_pred - vbm_iso_phys).mean()
        )
    else:
        L_iso = torch.tensor(0.0, device=X_tr_t.device)

    if PAPER_MODE:
        # Squared deviations, and per-point weights exactly as in the original papers:
        #   SA-PINN  (McClenny & Braga-Neto, J. Comput. Phys. 2023): loss_term = mean(m(lambda_i) e_i^2), m = lambda^2, lambda per
        #            point on the physics-residual and boundary-type terms, learned by gradient ascent; data terms NOT weighted.
        #   RBA-PINN (Anagnostopoulos et al., Comput. Methods Appl. Mech. Eng. 2024): loss_term = mean((lambda_i e_i)^2), lambda
        #            per point and per term from that term's own residual, lambda <- gamma lambda + eta |e|/max|e| every iteration.
        dev = X_tr_t.device
        trust = (torch.exp(-strain_tr_t.abs().max(dim=1).values / args.trust_scale) if args.trust_scale > 0
                 else torch.ones(strain_tr_t.shape[0], device=dev))
        E = {}
        if non_iso.sum() > 0:
            E["cbm"] = (cbm_pred[non_iso] - cbm_phys[non_iso], trust[non_iso], W_CBM, True)
            E["vbm"] = (vbm_pred[non_iso] - vbm_phys[non_iso], trust[non_iso], W_VBM, True)
            E["cons"] = (eg_pred[non_iso] - eg_phys[non_iso], trust[non_iso], W_CONS, True)
        if X_anc_t.shape[0] > 0:
            E["anc_c"] = (cbm_a - CBM_0, None, W_ANCHOR, True); E["anc_v"] = (vbm_a - VBM_0, None, W_ANCHOR, True)
        E["pcbm"] = (dCBM_dI1 - A_C_REF, None, W_PCBM, True); E["pvbm"] = (dVBM_dI1 - A_V_REF, None, W_PVBM, True)
        if dft_idx_t.numel() > 0:
            E["data_c"] = (cbm_pred[dft_idx_t] - cbm_dft_t, None, W_DATA, False); E["data_v"] = (vbm_pred[dft_idx_t] - vbm_dft_t, None, W_DATA, False)
        if iso_mask.sum() > 0 and eg_tr_t is not None:
            E["iso"] = (eg_pred[iso_mask] - eg_tr_t[iso_mask], None, W_ISO_EG, False)
        L = {}
        for k, (e, t, wk, physics_term) in E.items():
            uses_w = (args.model == "rba_paper") or (args.model == "sa_paper" and physics_term)
            if uses_w and k not in PAPER_W:   # lazy init after the network is built, so its initialisation is unchanged
                if args.model == "sa_paper":
                    PAPER_W[k] = nn.Parameter(torch.rand(e.shape[0], device=dev))
                else:
                    PAPER_W[k] = torch.zeros(e.shape[0], device=dev)
            if args.model == "rba_paper" and PAPER_UPDATE[0]:
                with torch.no_grad():
                    a_ = e.detach().abs(); PAPER_W[k].mul_(args.rba_paper_gamma).add_(args.rba_paper_eta * a_ / (a_.max() + 1e-12))
            if args.model == "sa_paper":
                m = PAPER_W[k] ** 2 if physics_term else torch.ones_like(e)
                term = m * e ** 2
            elif args.model == "rba_paper":
                term = (PAPER_W[k] * e) ** 2
            else:
                term = e ** 2
            if t is not None:
                term = term * t
            L[k] = wk * term.mean()
        z = torch.tensor(0.0, device=dev)
        L_cbm, L_vbm, L_cons = L.get("cbm", z), L.get("vbm", z), L.get("cons", z)
        L_anchor = L.get("anc_c", z) + L.get("anc_v", z)
        L_pcbm, L_pvbm = L["pcbm"], L["pvbm"]
        L_data = L.get("data_c", z) + L.get("data_v", z)
        L_iso_eg = L.get("iso", z)
    if args.phys_anneal != 1.0:
        f = PHYS_MULT[0]
        L_cbm, L_vbm, L_cons, L_anchor, L_pcbm, L_pvbm = [f * t for t in (L_cbm, L_vbm, L_cons, L_anchor, L_pcbm, L_pvbm)]
    total = L_cbm + L_vbm + L_cons + L_anchor + L_pcbm + L_pvbm + L_data + L_iso_eg + L_iso
    extra = {}
    if args.sym_colloc > 0:
        # S: a random cubic operation per training state; the rotated strain must give the same band edges
        ops = BRANCH["ops"][torch.randint(0, 48, (strain_tr_t.shape[0],), device=strain_tr_t.device)]
        T = strain_tr_t.new_zeros(strain_tr_t.shape[0], 3, 3)
        T[:, 0, 0] = strain_tr_t[:, 0]; T[:, 1, 1] = strain_tr_t[:, 1]; T[:, 2, 2] = strain_tr_t[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = strain_tr_t[:, 3]; T[:, 1, 2] = T[:, 2, 1] = strain_tr_t[:, 4]; T[:, 0, 2] = T[:, 2, 0] = strain_tr_t[:, 5]
        Tr = ops @ T @ ops.transpose(1, 2)
        raw_r = torch.stack([Tr[:, 0, 0], Tr[:, 1, 1], Tr[:, 2, 2], Tr[:, 0, 1], Tr[:, 1, 2], Tr[:, 0, 2]], dim=1)
        cbm_r, vbm_r, _ = model((raw_r - BRANCH["mean"]) / BRANCH["scale"])
        _pw = 2 if PAPER_MODE else 1
        extra["sym"] = args.sym_colloc * (torch.abs(cbm_r - cbm_pred) ** _pw + torch.abs(vbm_r - vbm_pred) ** _pw).mean()
    if args.curv_w > 0:
        # C: second derivative along equal expansion of the three axes, at the unstrained point
        x0 = BRANCH["x0"].clone().requires_grad_(True)
        c0, v0, _ = model(x0)
        dirv = BRANCH["dir"]
        curv = []
        for e in (c0, v0):
            g = torch.autograd.grad(e.sum(), x0, create_graph=True)[0]
            d1 = (g * dirv).sum()
            g2 = torch.autograd.grad(d1, x0, create_graph=True)[0]
            curv.append((g2 * dirv).sum())
        _pw = 2 if PAPER_MODE else 1
        extra["curv"] = args.curv_w * (torch.abs(curv[0] - BRANCH["curv_c"]) ** _pw + torch.abs(curv[1] - BRANCH["curv_v"]) ** _pw)
    if args.colloc_n > 0 and args.w_colloc > 0:
        # Collocation: agreement with the analytic models on the synthetic strain states (follows --phys_anneal)
        cbm_c, vbm_c, _ = model(BRANCH["col_x"])
        _pw = 2 if PAPER_MODE else 1
        f_c = PHYS_MULT[0] if args.phys_anneal != 1.0 else 1.0
        extra["colloc"] = f_c * args.w_colloc * (torch.abs(cbm_c - BRANCH["col_cbm"]) ** _pw + torch.abs(vbm_c - BRANCH["col_vbm"]) ** _pw).mean()
    if args.valley_sup > 0 and dft_idx_t.numel() > 0:
        ax = BRANCH["axis_dft"]; ok = ax >= 0
        if bool(ok.any()):
            lp = model.valley_axis_logp(X_tr_t[dft_idx_t][ok])
            extra["valley"] = args.valley_sup * (-lp[torch.arange(int(ok.sum()), device=lp.device), ax[ok]]).mean()
    for v in extra.values():
        total = total + v
    if return_parts:
        return total, eg_phys, dict(data=L_data, iso=L_iso_eg, cons=L_cons, cbm=L_cbm, vbm=L_vbm,
                                    anchor=L_anchor, pcbm=L_pcbm, pvbm=L_pvbm, iso_dp=L_iso, **extra)
    return total, eg_phys


# ============================================================
# SECTION 7: TRAINING LOOP
# ============================================================
# train() runs `epochs` of full-batch training. Each epoch: ONE network step that
# descends compute_loss() (AdamW for the first half, then the curvature-aware
# SOAP optimizer for the second). Every 100 epochs it calls update_rba() to
# refresh the attention weights (skipped under --no_rba, which keeps them
# uniform). Every --ckpt_interval epochs it saves the model if VALIDATION MAE
# improved (selection never uses test -> no leakage). It logs one row per epoch
# and returns the best checkpoint. (This is the simpler sibling of the SA PINN's
# two-step min-max loop -- here the weights follow a fixed rule, not a gradient.)

import torch.optim as optim

try:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from soap import SOAP
    HAS_SOAP = True
except ImportError:
    HAS_SOAP = False


def _to(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


PHYS_MULT = [1.0]   # current physics-term multiplier under --phys_anneal
PAPER_MODE = (args.model in ("sa_paper", "rba_paper")) or (args.loss_pow == 2)
PAPER_W = {}        # per-term, per-point weights of sa_paper / rba_paper (created at the first loss evaluation)
PAPER_UPDATE = [False]   # rba_paper: update the multipliers only in the network-update call
BRANCH = {}     # tensors for the physics-branch switches (filled in train() only when a switch is on)


def train(data, device):
    t0 = time.time()

    X_tr_t     = _to(data["X_tr"], device)
    X_va_t     = _to(data["X_va"], device)
    X_te_t     = _to(data["X_te"], device)
    strain_tr_t = _to(data["strain_tr"], device)

    X_anc_t = X_tr_t[data["anc_mask"]]
    X_nz_t  = X_tr_t[data["nz_mask"]]

    # BatchNorm is structurally incompatible with this loss, and it is worth
    # saying why rather than letting it fail obscurely deeper in. The anchor
    # boundary condition evaluates the network at the unstrained reference state,
    # of which there is exactly one row. BatchNorm in training mode normalises by
    # within-batch variance, and a batch of one has none. Making it run would
    # mean either dropping the anchor term or evaluating it under different
    # normalisation statistics than the data term -- both change the objective.
    # LayerNorm has no such problem: it normalises per sample.
    if args.norm == "batch" and len(X_anc_t) < 2:
        raise SystemExit(
            "--norm batch is not applicable: the anchor term evaluates the model "
            "on %d sample(s) and BatchNorm needs at least 2. Use --norm layer."
            % len(X_anc_t))

    # Scales for Exx, Eyy, Ezz (indices 0,1,2) — needed to convert raw-component
    # gradients back to physical eV/strain units for the gradient BC loss terms
    i1_scale = data["scaler"].scale_[:3].astype(np.float32)

    if len(data["dft_idx"]) > 0:
        dft_idx_t = torch.tensor(data["dft_idx"], dtype=torch.long, device=device)
        cbm_dft_t = _to(data["cbm_tr"][data["dft_idx"]], device)
        vbm_dft_t = _to(data["vbm_tr"][data["dft_idx"]], device)
    else:
        dft_idx_t = torch.zeros(0, dtype=torch.long, device=device)
        cbm_dft_t = torch.zeros(0, device=device)
        vbm_dft_t = torch.zeros(0, device=device)

    if args.valley_sup > 0:
        BRANCH["axis_dft"] = torch.tensor(data["axis_tr"][data["dft_idx"]], dtype=torch.long, device=device) if len(data["dft_idx"]) > 0 else torch.zeros(0, dtype=torch.long, device=device)
    if args.focus > 0:
        st = torch.tensor(data["strain_tr"], dtype=torch.float64); I1f = st[:, :3].sum(1)
        P_ = _PHYS
        vf = lambda e: P_.XI_D * I1f + P_.XI_U * e + P_.XI_Q * e ** 2 + P_.XI_C * I1f * e + P_.XI_Q2 * I1f ** 2
        dv = torch.stack([vf(st[:, 2]) + 2 * P_.XI_UP * st[:, 3], vf(st[:, 2]) - 2 * P_.XI_UP * st[:, 3], vf(st[:, 1]) + 2 * P_.XI_UP * st[:, 5],
                          vf(st[:, 1]) - 2 * P_.XI_UP * st[:, 5], vf(st[:, 0]) + 2 * P_.XI_UP * st[:, 4], vf(st[:, 0]) - 2 * P_.XI_UP * st[:, 4]], 1)
        srt = dv.sort(1).values; margin = srt[:, 1] - srt[:, 0]
        region = ((st[:, 3:].abs() > 1e-8).sum(1) >= 2) & (margin < args.focus_margin)
        BRANCH["focus_w"] = _to((1.0 + args.focus * region.double()).numpy(), device)
        print("  focus: %d of %d training states in the competing-valley region, weight %.1f" % (int(region.sum()), len(region), 1 + args.focus), flush=True)

    # Full training-set Eg labels — used by L_iso_eg on isotropic samples
    eg_tr_t = _to(data["eg_tr"], device)

    # Isotropic DP loss tensors — built once, reused every epoch
    if args.iso_dp_loss:
        iso_raw_scaled = make_iso_synth_points(data["scaler"])
        X_iso_t = _to(iso_raw_scaled, device)
        # Unscaled raw strain for physics functions (all shear = 0)
        n_iso = iso_raw_scaled.shape[0]
        eps_vals = np.linspace(-0.08, 0.08, n_iso).astype(np.float32)
        strain_iso_np = np.zeros((n_iso, 6), dtype=np.float32)
        strain_iso_np[:, 0] = eps_vals
        strain_iso_np[:, 1] = eps_vals
        strain_iso_np[:, 2] = eps_vals
        strain_iso_t = _to(strain_iso_np, device)
    else:
        X_iso_t = None
        strain_iso_t = None

    if args.sym_colloc > 0 or args.curv_w > 0:
        import itertools
        mean_t = _to(data["scaler"].mean_, device); scale_t = _to(data["scaler"].scale_, device)
        ops = [np.diag(sg) @ np.eye(3)[list(pm)] for pm in itertools.permutations(range(3)) for sg in itertools.product([1, -1], repeat=3)]
        BRANCH.update(ops=_to(np.array(ops), device), mean=mean_t, scale=scale_t)
        x0 = ((np.zeros(6) - data["scaler"].mean_) / data["scaler"].scale_)[None, :]
        dirv = np.zeros(6); dirv[:3] = 1.0 / (3.0 * data["scaler"].scale_[:3])     # d/dI1 in scaled-input space
        BRANCH.update(x0=_to(x0, device), dir=_to(dirv[None, :], device))
        # curvature targets: cubic fit to the isotropic states of the TRAINING portion (always labeled)
        st = data["strain_tr"]; i1 = st[:, :3].sum(1)
        iso = (np.abs(st[:, 3:]).sum(1) < 1e-4) & (np.abs(st[:, :3] - st[:, :3].mean(1, keepdims=True)).max(1) < 1e-4)
        if iso.sum() >= 4:
            cc = np.polyfit(i1[iso], data["cbm_tr"][iso], 3); cv = np.polyfit(i1[iso], data["vbm_tr"][iso], 3)
            BRANCH.update(curv_c=float(2 * cc[-3]), curv_v=float(2 * cv[-3]))
            print("  branch: curvature targets d2CBM/dI1^2 = %.2f, d2VBM/dI1^2 = %.2f eV (%d isotropic training states)"
                  % (BRANCH["curv_c"], BRANCH["curv_v"], int(iso.sum())), flush=True)
        elif args.curv_w > 0:
            # Isotropic family held out: the curvature targets cannot be measured, so C is switched off for this run.
            args.curv_w = 0.0
            print("  branch: %d isotropic training states -> curvature term off" % int(iso.sum()), flush=True)

    if args.colloc_n > 0:
        # Collocation states: synthetic strains where only the analytic models are enforced (no DFT label), so the
        # physics also constrains the network where there are no calculations. Own generator: the global RNG is untouched.
        rng_c = np.random.default_rng(args.seed + 7919)
        raw_c = rng_c.uniform(-args.colloc_max, args.colloc_max, size=(args.colloc_n, 6))
        raw_ct = _to(raw_c, device)
        BRANCH.update(col_x=_to((raw_c - data["scaler"].mean_) / data["scaler"].scale_, device),
                      col_cbm=physics_cbm(raw_ct).detach(), col_vbm=physics_vbm(raw_ct).detach())
        print("  collocation: %d formula-only states, |strain| <= %.3f, weight %g" % (args.colloc_n, args.colloc_max, args.w_colloc), flush=True)

    model = PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"], scaler=data["scaler"]).to(device)

    # Weight state. Both start at zero, so neither draws from the RNG and the
    # network initialisation is identical across all three models.
    if args.model == "sa":
        sa_w = nn.Parameter(torch.zeros(data["N_tr"], device=device))
        sa_opt = optim.Adam([sa_w], lr=args.sa_lr)
        rba_w = None
    else:
        rba_w = init_rba_weights(data["N_tr"], device)
        sa_w, sa_opt = None, None

    sa_paper_opt = [None]

    def _weights(detach=True):
        # The per-sample weights the loss should use this epoch.
        if args.model == "sa":
            return torch.nn.functional.softplus(sa_w.detach() if detach else sa_w)
        return rba_w + 1.0

    # Pre-compute physics targets once — they don't change during training
    with torch.no_grad():
        eg_phys_tr = physics_cbm(strain_tr_t) - physics_vbm(strain_tr_t)

    # Optimizer family. Adopted = two-phase AdamW -> SOAP; the ablation can replace
    # it with a single optimizer for all epochs. (--no_soap also forces single-phase.)
    two_phase = (args.optimizer == "adamw_soap") and (not args.no_soap)
    use_lbfgs = (args.optimizer == "lbfgs")
    half = int(round(args.switch_frac * args.epochs)) if two_phase else args.epochs + 1

    if args.optimizer in ("adamw_soap", "adamw"):
        opt = optim.AdamW(model.parameters(), lr=args.adamw_lr,
                          weight_decay=args.adamw_wd, betas=(0.9, 0.999))
    elif args.optimizer == "soap":
        # SOAP for the whole run. Without this arm the two-phase result cannot be
        # attributed: dropping SOAP is worse, but that could be SOAP itself or the
        # switch. It takes SOAP's own hyperparameters, not AdamW's, so the
        # comparison is against SOAP at its best rather than SOAP misconfigured.
        if not HAS_SOAP:
            raise SystemExit("--optimizer soap needs soap.py on the path")
        opt = SOAP(model.parameters(), lr=args.soap_lr, betas=(0.95, 0.95),
                   weight_decay=args.soap_wd, precondition_frequency=10)
    elif args.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=args.adamw_lr)
    elif args.optimizer == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=args.adamw_lr)
    elif args.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.adamw_lr, momentum=0.9)
    elif args.optimizer == "lbfgs":
        opt = optim.LBFGS(model.parameters(), lr=args.adamw_lr,
                          max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2,
                                                          eta_min=1e-5) if not use_lbfgs else None

    best_val_mae = float("inf")
    best_state   = None
    gpu_peak_mb  = 0.0
    # Validation is measured only at checkpoint epochs, so between them the log
    # carries the last measurement rather than a stale best.
    last_val_mae = float("nan")
    best_ckpt_epoch = 0
    stale_ckpts = 0

    epoch_rows = []

    def _val_mae():
        model.eval()
        with torch.no_grad():
            _, _, eg_va = model(X_va_t)
        model.train()
        return float(torch.abs(eg_va - _to(data["eg_va"], device)).mean().item())

    # Held-out test MAE per epoch — logged only for the learning-curve figure;
    # never used for checkpoint selection (selection stays on val to avoid leakage).
    def _test_mae():
        model.eval()
        with torch.no_grad():
            _, _, eg_te = model(X_te_t)
        model.train()
        return float(torch.abs(eg_te - _to(data["eg_te"], device)).mean().item())

    def _train_mae(eg_pred_t):
        eg_tr_t = _to(data["eg_tr"], device)
        return float(torch.abs(eg_pred_t.detach() - eg_tr_t).mean().item())

    print(f"\nTraining on {device}  |  {data['N_tr']} train  |  "
          f"{len(data['dft_idx'])} labeled  |  {args.epochs} epochs")

    probe_rows = []
    for ep in range(1, args.epochs + 1):
        if args.phys_anneal != 1.0:
            PHYS_MULT[0] = 1.0 + (args.phys_anneal - 1.0) * (ep - 1) / max(1, args.epochs - 1)

        # Switch to SOAP at the halfway point (two-phase only)
        if two_phase and ep == half + 1:
            if HAS_SOAP:
                opt = SOAP(model.parameters(), lr=args.soap_lr, betas=(0.95, 0.95),
                           weight_decay=args.soap_wd, precondition_frequency=10)
            else:
                opt = optim.AdamW(model.parameters(), lr=args.soap_lr,
                                  weight_decay=args.soap_wd, betas=(0.95, 0.95))
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000,
                                                                   T_mult=2,
                                                                   eta_min=1e-5)

        model.train()
        if args.grad_probe_every and (ep == 1 or ep % args.grad_probe_every == 0):
            # Diagnostic only: norms and cosine of the data-term and physics-term
            # gradients with the weights the loss uses this epoch. Nothing is stepped,
            # and the random-number state is restored afterwards: the symmetry term
            # draws random cubic operations inside compute_loss, so without this a
            # probed run would train differently from an unprobed one.
            _rng_cpu = torch.get_rng_state()
            _rng_gpu = torch.cuda.get_rng_state(X_tr_t.device) if X_tr_t.is_cuda else None
            _, _, _parts = compute_loss(
                model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                _weights(), dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                eg_tr_t=eg_tr_t, X_iso_t=X_iso_t, strain_iso_t=strain_iso_t, return_parts=True)
            _params = [p for p in model.parameters() if p.requires_grad]
            _Ld = _parts["data"] + _parts["iso"]
            _Lp = (_parts["cons"] + _parts["cbm"] + _parts["vbm"] + _parts["anchor"]
                   + _parts["pcbm"] + _parts["pvbm"] + _parts["iso_dp"])
            def _flat(L):
                if not (torch.is_tensor(L) and L.requires_grad):
                    return torch.zeros(sum(p.numel() for p in _params), device=device)
                gs = torch.autograd.grad(L, _params, retain_graph=True, allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                  for g, p in zip(gs, _params)])
            _gd, _gp = _flat(_Ld), _flat(_Lp)
            _nd, _npn = float(_gd.norm()), float(_gp.norm())
            probe_rows.append(dict(epoch=ep, L_data=float(_Ld), L_phys=float(_Lp),
                                   grad_data_norm=_nd, grad_phys_norm=_npn,
                                   cos=(float(_gd @ _gp) / (_nd * _npn)) if (_nd > 0 and _npn > 0) else float("nan"),
                                   **{"term_" + k: float(v) for k, v in _parts.items()}))
            opt.zero_grad()
            torch.set_rng_state(_rng_cpu)
            if _rng_gpu is not None:
                torch.cuda.set_rng_state(_rng_gpu, X_tr_t.device)
        if use_lbfgs:
            eg_phys_out = None
            def _closure():
                opt.zero_grad()
                lo, _ = compute_loss(
                    model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                    _weights(), dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                    eg_tr_t=eg_tr_t, X_iso_t=X_iso_t, strain_iso_t=strain_iso_t)
                lo.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                return lo
            loss = opt.step(_closure)
        else:
            opt.zero_grad()
            PAPER_UPDATE[0] = True
            loss, eg_phys_out = compute_loss(
                model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                _weights(), dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                eg_tr_t=eg_tr_t, X_iso_t=X_iso_t, strain_iso_t=strain_iso_t
            )
            PAPER_UPDATE[0] = False
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            opt.step()
            if sch is not None:
                sch.step()

        if args.model == "sa":
            # Maximise the same loss with respect to the weights: samples the
            # network handles badly get pushed up. Clamped to keep them bounded.
            sa_opt.zero_grad()
            loss_sa, _ = compute_loss(
                model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                _weights(detach=False), dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                eg_tr_t=eg_tr_t, X_iso_t=X_iso_t, strain_iso_t=strain_iso_t)
            (-loss_sa).backward()          # Adam minimises; we want to maximise
            sa_opt.step()
            with torch.no_grad():
                sa_w.clamp_(-args.sa_clamp, args.sa_clamp)
        elif args.model == "sa_paper":
            # Gradient ascent on the self-adaptation weights (the loss is maximised w.r.t. them).
            if sa_paper_opt[0] is None:
                sa_paper_opt[0] = optim.Adam(list(PAPER_W.values()), lr=args.sa_paper_lr)
            sa_paper_opt[0].zero_grad()
            loss_sa, _ = compute_loss(
                model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                _weights(), dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                eg_tr_t=eg_tr_t, X_iso_t=X_iso_t, strain_iso_t=strain_iso_t)
            (-loss_sa).backward()
            sa_paper_opt[0].step()
        elif args.model == "rba_paper":
            pass                           # multipliers already updated inside the network-update call
        elif (not args.no_rba) and (ep % 100 == 0):
            update_rba(rba_w, model, X_tr_t, eg_phys_tr)

        # Checkpoint at every 250 epochs — cheap enough, catches fast early drops
        if ep % args.ckpt_interval == 0:
            v_mae = _val_mae()
            last_val_mae = v_mae
            if v_mae < best_val_mae - args.min_delta:
                best_val_mae = v_mae
                best_ckpt_epoch = ep
                stale_ckpts = 0
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            else:
                stale_ckpts += 1

        cur_lr = sch.get_last_lr()[0] if sch is not None else args.adamw_lr
        if DEVICE.type == "cuda":
            gm = torch.cuda.memory_allocated(device) / 1e6
            gpu_peak_mb = max(gpu_peak_mb, gm)
        else:
            gm = 0.0

        # Forward pass for train MAE — reuse the graph already computed this step
        with torch.no_grad():
            _, _, eg_pred_t = model(X_tr_t)
        tr_mae = _train_mae(eg_pred_t)

        # Two different quantities, under two different names. The thesis code
        # wrote the running best into val_mae_eV while the Shi baseline wrote the
        # current value into the same column, so any cross-model reading of it
        # compared "the checkpoint we kept" against "the luckiest single epoch".
        # Both are logged here, in every trainer, and neither name is reused.
        #   val_mae_eV   -- validation error as last measured (checkpoint cadence)
        #   val_best_eV  -- best validation error seen so far, i.e. the kept model
        epoch_rows.append({
            "epoch":        ep,
            "train_mae_eV": round(tr_mae, 6),
            "val_mae_eV":   round(last_val_mae, 6),
            "val_best_eV":  round(best_val_mae, 6) if best_val_mae < float("inf") else "",
            "test_mae_eV":  round(_test_mae(), 6),
            "lr":           cur_lr,
            "elapsed_s":    round(time.time() - t0, 2),
            "gpu_mem_mb":   round(gm, 1),
            "cpu_percent":  0.0,
            "ram_mb":       0.0,
        })

        # Early stopping, off unless --patience is set. Nothing is lost by
        # training on when the best checkpoint is kept, so this is a compute
        # decision; it is measured, not assumed, by 13_decide_budget.py.
        if args.patience > 0 and stale_ckpts >= args.patience:
            print("  early stop at epoch %d: %d checkpoints without improvement"
                  % (ep, stale_ckpts))
            break

        if ep % 500 == 0:
            phase = "AdamW" if ep <= half else ("SOAP" if HAS_SOAP else "AdamW-p2")
            print(f"  ep {ep:>5} [{phase}]  lr={cur_lr:.2e}  "
                  f"train={tr_mae*1000:.1f} meV  val={best_val_mae*1000:.1f} meV")

    elapsed = time.time() - t0

    # Restore best checkpoint before handing model to evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(),
               os.path.join(args.results_dir, "best_model.pt"))

    pd.DataFrame(epoch_rows).to_csv(
        os.path.join(args.results_dir, "epoch_log.csv"), index=False
    )
    if probe_rows:
        pd.DataFrame(probe_rows).to_csv(os.path.join(args.results_dir, "grad_probe.csv"), index=False)

    # Dump final-epoch RBA sample weights with their strain type, so the
    # weight-by-mode diagnostic (which modes the attention scheme up-weights)
    # can be plotted without re-training. Skipped under --no_rba (weights flat).
    # Per-sample weights are only meaningful for the two weighted models.
    if PAPER_W:
        pd.DataFrame([dict(term=k, n=int(v.numel()), mean=float(v.detach().mean()), min=float(v.detach().min()), max=float(v.detach().max()))
                      for k, v in PAPER_W.items()]).to_csv(os.path.join(args.results_dir, "paper_weights.csv"), index=False)
    if args.model in ("rba", "sa") and not args.no_rba:
        _w = (torch.nn.functional.softplus(sa_w.detach()) if args.model == "sa"
              else rba_w + 1.0)
        pd.DataFrame({
            "strain_type": data["stype_tr"],
            "sample_weight": _w.detach().cpu().numpy(),
        }).to_csv(os.path.join(args.results_dir, "sample_weights.csv"), index=False)

    # The epoch the kept model came from. If this equals the budget, training was
    # still improving when it was cut off and the accuracy is budget-limited.
    args._best_ckpt_epoch = best_ckpt_epoch
    args._epochs_run = ep

    return model, elapsed, gpu_peak_mb, best_val_mae


# ============================================================
# SECTION 8: EVALUATION ON TEST SET
# ============================================================
# Runs the best checkpoint on the held-out test set and computes every reported
# metric: overall Eg MAE/RMSE/R2, CBM and VBM MAE, and the per-strain-type
# breakdown. Then writes the two fixed-schema output-contract files,
# test_results.csv (one row per test sample) and metrics_summary.csv (one summary
# row), so the aggregation and comparison-table scripts can read them.

from sklearn.metrics import r2_score


def evaluate(model, data, elapsed, gpu_peak_mb, best_val_mae, device):
    model.eval()
    with torch.no_grad():
        cbm_pred, vbm_pred, eg_pred = model(_to(data["X_te"], device))
        _, _, eg_tr_pred = model(_to(data["X_tr"], device))
        _, _, eg_va_pred = model(_to(data["X_va"], device))

    eg_pred_np  = eg_pred.cpu().numpy()
    cbm_pred_np = cbm_pred.cpu().numpy()
    vbm_pred_np = vbm_pred.cpu().numpy()

    eg_true  = data["eg_te"]
    cbm_true = data["cbm_te"]
    vbm_true = data["vbm_te"]

    abs_err   = np.abs(eg_pred_np - eg_true)
    test_mae  = float(abs_err.mean())
    test_rmse = float(np.sqrt((abs_err**2).mean()))
    test_max  = float(abs_err.max())
    test_p95  = float(np.percentile(abs_err, 95))
    # MAPE: mean |pred-true|/|true| × 100 — relative error across wide Eg range
    test_mape = float((abs_err / np.abs(eg_true)).mean() * 100.0)
    # Sign accuracy: fraction of test samples where predicted ΔEg sign matches DFT
    # ΔEg = Eg - Eg_ref, where Eg_ref is the unstrained value (CBM_0 - VBM_0)
    EG_REF    = CBM_0 - VBM_0
    sign_acc  = float(np.mean(np.sign(eg_pred_np - EG_REF) == np.sign(eg_true - EG_REF)) * 100.0)

    cbm_mae   = float(np.abs(cbm_pred_np - cbm_true).mean())
    vbm_mae   = float(np.abs(vbm_pred_np - vbm_true).mean())
    test_r2   = float(r2_score(eg_true, eg_pred_np))
    val_mae   = float(np.abs(eg_va_pred.cpu().numpy() - data["eg_va"]).mean())
    train_mae = float(np.abs(eg_tr_pred.cpu().numpy() - data["eg_tr"]).mean())

    # Per-strain-type MAE — iterate once, bucket by type
    buckets = {t: [] for t in ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]}
    for i, st in enumerate(data["stype_te"]):
        key = str(st).lower()
        if key in buckets:
            buckets[key].append(abs_err[i])

    def _mae(lst):
        return round(float(np.mean(lst)) * 1000, 2) if lst else float("nan")

    print(f"\n  test  MAE : {test_mae*1000:.2f} meV   RMSE: {test_rmse*1000:.2f} meV   R²={test_r2:.4f}")
    print(f"  max err   : {test_max*1000:.2f} meV   p95: {test_p95*1000:.2f} meV   MAPE: {test_mape:.2f}%")
    print(f"  sign acc  : {sign_acc:.1f}%")
    print(f"  val   MAE : {val_mae*1000:.2f} meV")
    print(f"  train MAE : {train_mae*1000:.2f} meV")
    print(f"  CBM MAE   : {cbm_mae*1000:.2f} meV   VBM MAE: {vbm_mae*1000:.2f} meV")
    for k, v in buckets.items():
        print(f"  {k:>12}: {_mae(v):.1f} meV")

    # --- test_results.csv ---
    shear_col = [str(s) if str(s) != "nan" else "" for s in data["shear_te"]]
    test_df = pd.DataFrame({
        "sample_id":       data["idx_te"],
        "strain_type":     data["stype_te"],
        "shear_planes":    shear_col,
        "Eg_true_eV":      eg_true,
        "Eg_pred_eV":      eg_pred_np,
        "CBM_true_eV":     cbm_true,
        "CBM_pred_eV":     cbm_pred_np,
        "VBM_true_eV":     vbm_true,
        "VBM_pred_eV":     vbm_pred_np,
        "abs_error_eV":    abs_err,
        "signed_error_eV": eg_pred_np - eg_true,
        "seed":            args.seed,
        "dft_pct":         args.dft_pct,
    })
    test_df.to_csv(os.path.join(args.results_dir, "test_results.csv"), index=False)

    # --- metrics_summary.csv ---
    summary = {
        "model_name":          MODEL_NAME,
        "seed":                args.seed,
        "dft_pct":             args.dft_pct,
        "n_labeled":           len(data["dft_idx"]),
        "test_mae_meV":        round(test_mae  * 1000, 2),
        "test_rmse_meV":       round(test_rmse * 1000, 2),
        "test_max_err_meV":    round(test_max  * 1000, 2),
        "test_p95_err_meV":    round(test_p95  * 1000, 2),
        "test_mape_pct":       round(test_mape, 3),
        "sign_accuracy_pct":   round(sign_acc,  2),
        "val_mae_meV":         round(val_mae   * 1000, 2),
        "train_mae_meV":       round(train_mae * 1000, 2),
        "test_r2":             round(test_r2, 4),
        "cbm_mae_meV":         round(cbm_mae   * 1000, 2),
        "vbm_mae_meV":         round(vbm_mae   * 1000, 2),
        "mae_isotropic_meV":   _mae(buckets["isotropic"]),
        "mae_uniaxial_meV":    _mae(buckets["uniaxial"]),
        "mae_biaxial_meV":     _mae(buckets["biaxial"]),
        "mae_triaxial_meV":    _mae(buckets["triaxial"]),
        "mae_shear_meV":       _mae(buckets["shear"]),
        "total_epochs":        args.epochs,
        # Whether the budget bound the result is a property of the run, so it
        # belongs in the summary rather than only in the epoch log. When
        # best_ckpt_epoch equals epochs_run, training was still improving when
        # it stopped and the reported accuracy is budget-limited.
        "epochs_run":          getattr(args, "_epochs_run", args.epochs),
        "best_ckpt_epoch":     getattr(args, "_best_ckpt_epoch", -1),
        "n_params":            sum(p.numel() for p in model.parameters()),
        "elapsed_s":           round(elapsed, 1),
        "gpu_mem_peak_mb":     round(gpu_peak_mb, 1),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.results_dir, "metrics_summary.csv"), index=False
    )

    return summary


# ============================================================
# SECTION 9: RESOURCE LOGGING
# ============================================================
# Records compute usage. Sampling CPU/RAM every epoch would skew the tight loop's
# timings, so one psutil snapshot is taken at the end and patched into every row
# of epoch_log.csv. psutil is optional -- without it these fall back to 0.0.

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _cpu_percent():
    return psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0


def _ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6 if HAS_PSUTIL else 0.0


def patch_epoch_log_resources(results_dir):
    # Re-read epoch_log and fill cpu_percent / ram_mb with a single psutil snapshot.
    # Resource sampling inside the tight training loop would skew timing, so we
    # record one representative value per run instead.
    path = os.path.join(results_dir, "epoch_log.csv")
    df = pd.read_csv(path)
    df["cpu_percent"] = _cpu_percent()
    df["ram_mb"]      = _ram_mb()
    df.to_csv(path, index=False)


# ============================================================
# SECTION 10: README LOGGER CALL
# ============================================================
# The entry point. main() wires the pipeline together: load_and_split() ->
# train() -> evaluate() -> patch the resource log -> log_run(), which appends a
# one-line record (params + headline metrics) to RESULTS_LOG.md so every run is
# traceable. Runs only when executed as a script (the __main__ guard at bottom).



def main():
    os.makedirs(args.results_dir, exist_ok=True)
    # Record the exact invocation so the trained model can be rebuilt later (explainability, benchmarks).
    with open(os.path.join(args.results_dir, "command.txt"), "w") as _f:
        _env = os.environ.get("PHYSICS_CONST_JSON", "")
        _f.write(("PHYSICS_CONST_JSON=%s " % _env if _env else "") + "python " + " ".join(sys.argv) + "\n")

    print(f"Loading data: {args.data_csv}")
    data = load_and_split(args.data_csv, args.seed, args.dft_pct,
                          cv_fold=args.cv_fold, cv_nfolds=args.cv_nfolds,
                          input_inv=args.input_inv)
    print(f"  train={data['N_tr']}  val={len(data['eg_va'])}  "
          f"test={len(data['eg_te'])}  labeled={len(data['dft_idx'])}")

    model, elapsed, gpu_peak_mb, best_val_mae = train(data, DEVICE)

    summary = evaluate(model, data, elapsed, gpu_peak_mb, best_val_mae, DEVICE)

    patch_epoch_log_resources(args.results_dir)

    print("\n  wrote %s" % args.results_dir)


if __name__ == "__main__":
    main()
