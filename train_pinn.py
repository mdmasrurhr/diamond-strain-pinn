# ==============================================================================
# train_pinn.py -- trains ONE physics-informed network and writes its results.
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
# The Shi ANN baseline is a different architecture and lives in train_baseline.py.
#
# Usage
#   python train_pinn.py --model rba --seed 42 --dft_pct 100
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
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import config as C
# The analytic band-edge model. It lives in physics.py so that this trainer and
# the zero-label solvers use the identical equations and constants.
from physics import physics_cbm as _physics_cbm_softmin
from physics import physics_vbm, CBM_0, VBM_0, AV

# SOAP is the second-phase optimizer. It is optional so the trainer still runs
# on a checkout without it; the fallback keeps AdamW for the whole budget.
try:
    from soap import SOAP
    HAS_SOAP = True
except ImportError:
    HAS_SOAP = False

# Model name is set from --model once arguments are parsed (see below).
MODEL_NAMES = {"sa": "SA_PINN", "rba": "Dia_RBA_PINN", "mlp": "PhysicsFree_MLP"}

parser = argparse.ArgumentParser(
    description="Train one physics-informed network (SA, RBA, or physics-free).")
# Defaults to the dataset named in config.py. Giving it a default rather than
# marking it required is what lets the diagnostic scripts do a plain
# "import train_pinn" to reuse the model and the data pipeline.
parser.add_argument("--data_csv",    type=str, default=C.DATA_CSV,
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
                         "Ablation flag -- off by default.")
parser.add_argument("--model", type=str, default="rba", choices=["sa", "rba", "mlp"],
                    help="Which weighting to use. rba = fixed residual rule; "
                         "sa = learned adversarial weights; mlp = no physics at all "
                         "(the physics-free control).")
parser.add_argument("--sa_lr", type=float, default=C.SA_LR,
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
# Every default comes from config.py, which is the single place the adopted
# configuration is recorded. Passing a flag on the command line overrides it,
# which is how the ablation and sensitivity scripts vary one setting at a time.
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
parser.add_argument("--adamw_lr",   type=float, default=C.TUNED["adamw_lr"])
parser.add_argument("--adamw_wd",   type=float, default=C.TUNED["adamw_wd"])
parser.add_argument("--soap_lr",    type=float, default=C.TUNED["soap_lr"])
parser.add_argument("--soap_wd",    type=float, default=C.TUNED["soap_wd"])
parser.add_argument("--switch_frac",type=float, default=C.TUNED["switch_frac"])
parser.add_argument("--clip_norm",  type=float, default=C.TUNED["clip_norm"])
parser.add_argument("--rba_gamma",  type=float, default=C.RBA_GAMMA)
parser.add_argument("--rba_eta",    type=float, default=C.RBA_ETA)
parser.add_argument("--ckpt_interval", type=int, default=C.TUNED["ckpt_interval"])
parser.add_argument("--nz_thresh",  type=float, default=C.NZ_THRESH)
parser.add_argument("--anc_thresh", type=float, default=C.ANC_THRESH)
parser.add_argument("--scaler",     type=str,   default=C.TUNED["scaler"],
                    choices=["standard", "minmax", "robust"])
parser.add_argument("--w_data",     type=float, default=C.LOSS_WEIGHTS["data"])
parser.add_argument("--w_cons",     type=float, default=C.LOSS_WEIGHTS["cons"])
parser.add_argument("--w_cbm",      type=float, default=C.LOSS_WEIGHTS["cbm"])
parser.add_argument("--w_vbm",      type=float, default=C.LOSS_WEIGHTS["vbm"])
parser.add_argument("--w_anchor",   type=float, default=C.LOSS_WEIGHTS["anchor"])
parser.add_argument("--w_pcbm",     type=float, default=C.LOSS_WEIGHTS["pcbm"])
parser.add_argument("--w_pvbm",     type=float, default=C.LOSS_WEIGHTS["pvbm"])
parser.add_argument("--w_iso_eg",   type=float, default=C.LOSS_WEIGHTS["iso_eg"])
# Cross-validation: when --cv_fold is set, the fixed split is replaced by a
# stratified k-fold partition (test = the given fold). Used by the CV driver only.
parser.add_argument("--cv_fold",    type=int, default=None)
parser.add_argument("--cv_nfolds",  type=int, default=5)
# Partitioning scheme. "stratified" is the adopted 70/10/20 keyed to the seed.
# The others exist so that what the split measures -- interpolation within the
# sampled region, or extrapolation beyond it -- becomes an explicit choice.
parser.add_argument("--split",      type=str,   default="stratified",
                    choices=["stratified", "random", "group", "magnitude"])
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
parser.add_argument("--sa_clamp",   type=float, default=C.SA_CLAMP[1],
                    help="self-adaptive log-weights are clamped to +/- this")
# Design-choice ablations (defaults = adopted choices; behavior unchanged when unset).
parser.add_argument("--activation", type=str, default=C.TUNED["activation"],
                    choices=["silu", "relu", "tanh", "gelu"])
parser.add_argument("--init",       type=str, default="xavier",
                    choices=["xavier", "kaiming", "orthogonal", "default"])
parser.add_argument("--out_bias_init", type=str, default="mean",
                    choices=["mean", "zero"])
parser.add_argument("--optimizer",  type=str, default="adamw_soap",
                    choices=["adamw_soap", "soap", "adamw", "adam", "lbfgs",
                             "rmsprop", "sgd"])
parser.add_argument("--beta",       type=float, default=C.SOFTMIN_BETA)
args = parser.parse_args()

# The physics-free control is the same network with every physics term switched
# off, so it is expressed as a --drop_loss set rather than a separate model.
if args.model == "mlp":
    for _t in ("cbm", "vbm", "cons", "anchor", "pcbm", "pvbm"):
        if _t not in args.drop_loss:
            args.drop_loss.append(_t)
MODEL_NAME = MODEL_NAMES[args.model]
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
    # Oh-invariant I5 = Exy^2+Eyz^2+Ezx^2 collapses, which caused shear MAE saturation.
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
    # Map CSV subtypes (uniaxial_x, biaxial_xy, unstrained, ...) to the 5 canonical
    # types used throughout the output CSVs and thesis analysis.
    s = str(s).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"   # 'unstrained' and anything else -> isotropic


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

    # Scaler fitted on training features only -- prevents leakage into val/test normalization.
    # Choice exposed via --scaler for the input-scaling sensitivity study.
    if args.scaler == "minmax":
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    elif args.scaler == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[idx_tr]).astype(np.float32)
    X_va = scaler.transform(X_all[idx_va]).astype(np.float32)
    X_te = scaler.transform(X_all[idx_te]).astype(np.float32)

    # Labeled DFT subset -- separate RNG offset avoids coupling seed to split seed
    rng = np.random.default_rng(seed + 9999)
    if dft_pct == 0:
        dft_idx = np.array([], dtype=np.int64)
    else:
        n_labeled = max(1, round(len(idx_tr) * dft_pct / 100.0))
        dft_idx = rng.choice(len(idx_tr), size=n_labeled, replace=False)

    # Near-zero-strain masks for boundary condition losses
    vol_tr = strain_all[idx_tr, 0] + strain_all[idx_tr, 1] + strain_all[idx_tr, 2]
    nz_mask  = np.abs(vol_tr) < args.nz_thresh    # gradient BC: dCBM/dI1, dVBM/dI1
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
    # Spans the full isotropic range in the dataset (~+/-8% volumetric strain).
    # These are physics-only -- no DFT labels needed -- so they inject
    # Bardeen-Shockley DP theory directly into the sparse isotropic subspace.
    eps_vals = np.linspace(-eps_range, eps_range, n).astype(np.float32)
    raw = np.zeros((n, 6), dtype=np.float32)
    raw[:, 0] = eps_vals   # Exx
    raw[:, 1] = eps_vals   # Eyy
    raw[:, 2] = eps_vals   # Ezz
    # shear columns stay zero -- pure isotropic
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
def physics_cbm(strain, beta=None):
    if beta is None:
        beta = args.beta
    return _physics_cbm_softmin(strain, beta=beta)


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
    that their exclusion rests on a measurement (decide_architecture.py)
    rather than on never having tried them.
    """

    def __init__(self, cbm_mean, vbm_mean):
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

    def forward(self, x):
        h = x
        for blk, res in zip(self.blocks, self.residual):
            y = blk(h)
            h = h + y if res else y
        o = self.out(h)
        cbm = o[:, 0]
        vbm = o[:, 1]
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

# Loss term weights -- taken from the CLI (defaults equal the hand-set values) so
# the tuning/sensitivity studies can vary them. The --drop_loss ablation below
# still zeroes selected weights on top of whatever base value is set here.
# One weight per loss term, held in a dictionary so that the --drop_loss
# ablation below can set a term to zero by name.
LOSS_W = {
    # physics residual: predicted CBM vs the deformation-potential CBM
    "cbm":     args.w_cbm,
    # physics residual: predicted VBM vs the Bir-Pikus VBM
    "vbm":     args.w_vbm,
    # consistency: predicted Eg vs the physics Eg = CBM_phys - VBM_phys
    "cons":    args.w_cons,
    # anchor: at near-zero strain, predict the reference band positions
    "anchor":  args.w_anchor,
    # boundary condition on the slope dCBM/dI1 at small strain
    "pcbm":    args.w_pcbm,
    # boundary condition on the slope dVBM/dI1 at small strain
    "pvbm":    args.w_pvbm,
    # supervised data term, active only on the labelled subset
    "data":    args.w_data,
    # direct Eg supervision on isotropic samples, where the analytic model is
    # known to be inaccurate
    "iso_eg":  args.w_iso_eg,
    # Bardeen-Shockley form on synthetic isotropic points; only used when
    # --iso_dp_loss is given
    "iso":     1.0,
}
                          # because the DP/BP physics model has >89 meV systematic error there

# Reference gradients at zero strain used for gradient BCs
A_C_REF = -16.342843   # dCBM/dI1 [eV] -- derived from DP constants at zero strain
A_V_REF = AV           # dVBM/dI1 [eV] -- equals linear hydrostatic term at zero strain

# Isotropic Eg gradient BC -- fitted directly from HSE06-VASP isotropic subset.
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

# Ablation: --drop_loss NAME sets that term's weight to zero, which removes it
# from the sum without changing anything else about the run.
for drop_name in args.drop_loss:
    if drop_name in LOSS_W:
        LOSS_W[drop_name] = 0.0


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
# compute_loss() -- OUTLINE (read this first; full detail at each term below)
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
                 eg_tr_t=None, X_iso_t=None, strain_iso_t=None):
    # PRELUDE: predictions, analytic physics targets, and RBA per-sample weights.
    cbm_pred, vbm_pred, eg_pred = model(X_tr_t)

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

    if non_iso.sum() > 0:
        L_cbm  = LOSS_W["cbm"]  * (w[non_iso] * torch.abs(cbm_pred[non_iso] - cbm_phys[non_iso])).mean()
        L_vbm  = LOSS_W["vbm"]  * (w[non_iso] * torch.abs(vbm_pred[non_iso] - vbm_phys[non_iso])).mean()
        L_cons = LOSS_W["cons"] * (w[non_iso] * torch.abs(eg_pred[non_iso]  - eg_phys[non_iso] )).mean()
    else:
        L_cbm  = torch.tensor(0.0, device=X_tr_t.device)
        L_vbm  = torch.tensor(0.0, device=X_tr_t.device)
        L_cons = torch.tensor(0.0, device=X_tr_t.device)

    # Anchor BC: essentially-unstrained samples should predict reference positions
    if X_anc_t.shape[0] > 0:
        cbm_a, vbm_a, _ = model(X_anc_t)
        L_anchor = LOSS_W["anchor"] * (
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
    L_pcbm = LOSS_W["pcbm"] * torch.abs(dCBM_dI1 - A_C_REF).mean()
    L_pvbm = LOSS_W["pvbm"] * torch.abs(dVBM_dI1 - A_V_REF).mean()

    # Data terms -- only active when labeled DFT samples exist
    if dft_idx_t.numel() > 0:
        w_dft = w[dft_idx_t]
        L_data = LOSS_W["data"] * (
            (w_dft * torch.abs(cbm_pred[dft_idx_t] - cbm_dft_t)).mean() +
            (w_dft * torch.abs(vbm_pred[dft_idx_t] - vbm_dft_t)).mean()
        )
    else:
        L_data = torch.tensor(0.0, device=X_tr_t.device)

    # Direct Eg supervision on isotropic training samples.
    # The DP/BP physics model has wrong slope and offset in the isotropic subspace
    # (verified: dEg/dI1_phys=-1.20 eV vs DFT=-2.36 eV, +61 meV offset).
    # Physics residual loss is already masked for these samples above.
    # This term uses DFT Eg labels directly -- active for all isotropic samples
    # regardless of dft_idx, since isotropic samples are always fully labeled.
    if iso_mask.sum() > 0 and eg_tr_t is not None:
        L_iso_eg = LOSS_W["iso_eg"] * (w[iso_mask] * torch.abs(
            eg_pred[iso_mask] - eg_tr_t[iso_mask]
        )).mean()
    else:
        L_iso_eg = torch.tensor(0.0, device=X_tr_t.device)

    # Isotropic DP loss (Bardeen-Shockley): enforce physics_cbm/vbm closed-form
    # on synthetic isotropic points spanning the full volumetric strain range.
    # Active only when --iso_dp_loss is set -- kept as an ablation flag.
    if X_iso_t is not None and strain_iso_t is not None:
        cbm_iso_pred, vbm_iso_pred, _ = model(X_iso_t)
        cbm_iso_phys = physics_cbm(strain_iso_t)
        vbm_iso_phys = physics_vbm(strain_iso_t)
        L_iso = LOSS_W["iso"] * (
            torch.abs(cbm_iso_pred - cbm_iso_phys).mean() +
            torch.abs(vbm_iso_pred - vbm_iso_phys).mean()
        )
    else:
        L_iso = torch.tensor(0.0, device=X_tr_t.device)

    total = L_cbm + L_vbm + L_cons + L_anchor + L_pcbm + L_pvbm + L_data + L_iso_eg + L_iso
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


def _to(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


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

    # Scales for Exx, Eyy, Ezz (indices 0,1,2) -- needed to convert raw-component
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

    # Full training-set Eg labels -- used by L_iso_eg on isotropic samples
    eg_tr_t = _to(data["eg_tr"], device)

    # Isotropic DP loss tensors -- built once, reused every epoch
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

    model = PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(device)

    # Weight state. Both start at zero, so neither draws from the RNG and the
    # network initialisation is identical across all three models.
    if args.model == "sa":
        sa_w = nn.Parameter(torch.zeros(data["N_tr"], device=device))
        sa_opt = optim.Adam([sa_w], lr=args.sa_lr)
        rba_w = None
    else:
        rba_w = init_rba_weights(data["N_tr"], device)
        sa_w, sa_opt = None, None

    def _weights(detach=True):
        # The per-sample weights the loss should use this epoch.
        if args.model == "sa":
            return torch.nn.functional.softplus(sa_w.detach() if detach else sa_w)
        return rba_w + 1.0

    # Pre-compute physics targets once -- they don't change during training
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

    # Held-out test MAE per epoch -- logged only for the learning-curve figure;
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

    for ep in range(1, args.epochs + 1):

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
            loss, eg_phys_out = compute_loss(
                model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t,
                _weights(), dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale,
                eg_tr_t=eg_tr_t, X_iso_t=X_iso_t, strain_iso_t=strain_iso_t
            )
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
        elif (not args.no_rba) and (ep % 100 == 0):
            update_rba(rba_w, model, X_tr_t, eg_phys_tr)

        # Checkpoint at every 250 epochs -- cheap enough, catches fast early drops
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

        # Forward pass for train MAE -- reuse the graph already computed this step
        with torch.no_grad():
            _, _, eg_pred_t = model(X_tr_t)
        tr_mae = _train_mae(eg_pred_t)

        # Two different quantities, under two different names. The thesis code
        # wrote the running best into val_mae_eV while the Shi baseline wrote the
        # current value into the same column, so any cross-model reading of it
        # compared the kept checkpoint against the luckiest single epoch.
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
        # decision; it is measured, not assumed, by decide_budget.py.
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

    # Dump final-epoch RBA sample weights with their strain type, so the
    # weight-by-mode diagnostic (which modes the attention scheme up-weights)
    # can be plotted without re-training. Skipped under --no_rba (weights flat).
    # Per-sample weights are only meaningful for the two weighted models.
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
    # MAPE: mean |pred-true|/|true| x 100 -- relative error across wide Eg range
    test_mape = float((abs_err / np.abs(eg_true)).mean() * 100.0)
    # Sign accuracy: fraction of test samples where predicted DeltaEg sign matches DFT
    # DeltaEg = Eg - Eg_ref, where Eg_ref is the unstrained value (CBM_0 - VBM_0)
    EG_REF    = CBM_0 - VBM_0
    sign_acc  = float(np.mean(np.sign(eg_pred_np - EG_REF) == np.sign(eg_true - EG_REF)) * 100.0)

    cbm_mae   = float(np.abs(cbm_pred_np - cbm_true).mean())
    vbm_mae   = float(np.abs(vbm_pred_np - vbm_true).mean())
    test_r2   = float(r2_score(eg_true, eg_pred_np))
    val_mae   = float(np.abs(eg_va_pred.cpu().numpy() - data["eg_va"]).mean())
    train_mae = float(np.abs(eg_tr_pred.cpu().numpy() - data["eg_tr"]).mean())

    # Per-strain-type MAE -- iterate once, bucket by type
    buckets = {t: [] for t in ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]}
    for i, st in enumerate(data["stype_te"]):
        key = str(st).lower()
        if key in buckets:
            buckets[key].append(abs_err[i])

    def _mae(lst):
        return round(float(np.mean(lst)) * 1000, 2) if lst else float("nan")

    print(f"\n  test  MAE : {test_mae*1000:.2f} meV   RMSE: {test_rmse*1000:.2f} meV   R^2={test_r2:.4f}")
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
    # Resource sampling inside the tight training loop would skew timing, so it
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
