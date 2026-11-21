"""Train one network on the strained-diamond dataset and write its results.

Every training run of the paper is one call of this program; make_tasks.py writes the exact commands. The models
differ only in their options:

  --model mlp                                 MLP (data only; every physics term off)
  --model mlp --aphynity                      APHYNITY (analytic models with trainable constants + minimal correction)
  --model sa_paper | rba_paper | ntk_paper    SA-PINN, RBA-PINN and NTK-PINN, weighted as published
  --model rba --no_rba --residual_phys        residual model (analytic band edges + learned correction)
  ... --zero_out --const_head --const_tanh    PHM (the network also sets the 13 analytic constants per strain state)

Writes into --results_dir: command.txt, epoch_log.csv, test_results.csv, metrics_summary.csv, best_model.pt and, where
they apply, grad_probe.csv, paper_weights.csv, ntk_weights.csv and aphynity_lambda.csv.

The options are parsed when the module is loaded; other scripts reuse load_and_split() and PINNModel through
config.import_trainer().
"""

import argparse
import itertools
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import physics as _PHYS
from physics import physics_cbm as _physics_cbm_softmin, physics_vbm, CBM_0, VBM_0, AV, XI_D, XI_U
from soap import SOAP

# ======================================================================================================================
# OPTIONS
# ======================================================================================================================
PAPER_MODELS = ("sa_paper", "rba_paper", "ntk_paper")
MODEL_NAMES = {
    "rba": "Dia_RBA_PINN",
    "mlp": "PhysicsFree_MLP",
    "sa_paper": "SA_PINN_paper",
    "rba_paper": "RBA_PINN_paper",
    "ntk_paper": "NTK_weighted_PINN",
}  # model_name in metrics_summary.csv

parser = argparse.ArgumentParser(description="Train one network on the strained-diamond dataset.")
# run
parser.add_argument("--data_csv", type=str, required=True, help="dataset CSV")
parser.add_argument("--results_dir", type=str, default=None, help="output folder (default: results/<model name>)")
parser.add_argument(
    "--seed", type=int, default=42, help="seed of the data division, labeled states and initial weights"
)
parser.add_argument(
    "--dft_pct", type=int, default=100, choices=C.FRACTIONS, help="labeled share of the training states (%%)"
)
parser.add_argument("--epochs", type=int, default=C.TUNED["epochs"], help="training iterations (full batch)")
parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
# data division
parser.add_argument(
    "--split",
    type=str,
    default="stratified",
    choices=["stratified", "symmetry", "group", "magnitude"],
    help="stratified 70/10/20; symmetry: symmetry-equivalent states in one portion; group: one family "
    "held out; magnitude: train below --magnitude_thresh, test above",
)
parser.add_argument("--holdout_family", type=str, default="shear", help="--split group: the family held out")
parser.add_argument("--magnitude_thresh", type=float, default=0.05, help="--split magnitude: largest training |strain|")
# network and optimizer
parser.add_argument("--width", type=int, default=C.TUNED["width"])
parser.add_argument("--depth", type=int, default=C.TUNED["depth"])
parser.add_argument("--adamw_lr", type=float, default=C.TUNED["adamw_lr"])
parser.add_argument("--adamw_wd", type=float, default=C.TUNED["adamw_wd"])
parser.add_argument("--soap_lr", type=float, default=C.TUNED["soap_lr"])
parser.add_argument("--soap_wd", type=float, default=C.TUNED["soap_wd"])
parser.add_argument("--switch_frac", type=float, default=C.TUNED["switch_frac"], help="share of iterations with AdamW")
parser.add_argument("--soap_warmup", type=int, default=0, help="linear warm-up of the SOAP learning rate (iterations)")
parser.add_argument("--clip_norm", type=float, default=C.TUNED["clip_norm"])
parser.add_argument(
    "--ckpt_interval", type=int, default=C.TUNED["ckpt_interval"], help="iterations between validations"
)
# model
parser.add_argument(
    "--model",
    type=str,
    default="rba",
    choices=list(MODEL_NAMES),
    help="rba: fixed or residual-based per-state weights; mlp: no physics terms; "
    "sa_paper, rba_paper, ntk_paper: the published adaptive PINNs",
)
parser.add_argument("--no_rba", action="store_true", help="--model rba or mlp: keep every per-state weight at 1")
parser.add_argument("--rba_gamma", type=float, default=0.999, help="--model rba: decay of the residual-based weights")
parser.add_argument("--rba_eta", type=float, default=0.1, help="--model rba: update rate of the residual-based weights")
parser.add_argument("--residual_phys", action="store_true", help="outputs are corrections to the analytic band edges")
parser.add_argument("--zero_out", action="store_true", help="with --residual_phys: the correction starts at zero")
parser.add_argument(
    "--const_head",
    action="store_true",
    help="with --residual_phys --zero_out: the network also sets the 13 analytic constants per state",
)
parser.add_argument("--const_tanh", action="store_true", help="with --const_head: theta = theta0 + scale * tanh(u)")
parser.add_argument(
    "--aphynity",
    action="store_true",
    help="APHYNITY (Yin et al. 2021): trainable analytic constants + network correction F_a; "
    "minimizes mean |F_a|^2 + lambda * squared data loss",
)
parser.add_argument("--aph_lambda0", type=float, default=1.0, help="APHYNITY: initial lambda")
parser.add_argument("--aph_tau2", type=float, default=1.0, help="APHYNITY: lambda step")
parser.add_argument("--sa_paper_lr", type=float, default=5e-3, help="SA-PINN: ascent learning rate of the weights")
parser.add_argument("--rba_paper_gamma", type=float, default=0.999, help="RBA-PINN: decay")
parser.add_argument("--rba_paper_eta", type=float, default=0.01, help="RBA-PINN: learning rate")
parser.add_argument("--ntk_every", type=int, default=1000, help="NTK-PINN: iterations between weight updates")
parser.add_argument("--beta", type=float, default=C.SOFTMIN_BETA, help="sharpness of the soft minimum over the valleys")
# objective
parser.add_argument("--w_data", type=float, default=C.LOSS_WEIGHTS["data"])
parser.add_argument("--w_cons", type=float, default=C.LOSS_WEIGHTS["cons"])
parser.add_argument("--w_cbm", type=float, default=C.LOSS_WEIGHTS["cbm"])
parser.add_argument("--w_vbm", type=float, default=C.LOSS_WEIGHTS["vbm"])
parser.add_argument("--w_anchor", type=float, default=C.LOSS_WEIGHTS["anchor"])
parser.add_argument("--w_pcbm", type=float, default=C.LOSS_WEIGHTS["pcbm"])
parser.add_argument("--w_pvbm", type=float, default=C.LOSS_WEIGHTS["pvbm"])
parser.add_argument("--w_iso_eg", type=float, default=C.LOSS_WEIGHTS["iso_eg"])
parser.add_argument(
    "--w_midpoint",
    type=float,
    default=0.0,
    help="weight of the band-edge level anchor (mean of CBM and VBM against the analytic models)",
)
parser.add_argument(
    "--drop_loss",
    action="append",
    default=[],
    help="switch a term off: cbm, vbm, cons, anchor, pcbm, pvbm, data, iso_eg (repeatable)",
)
parser.add_argument("--nz_thresh", type=float, default=0.01, help="|I1| below which the zero-strain slope terms act")
parser.add_argument("--anc_thresh", type=float, default=1e-4, help="|I1| below which the anchor term acts")
parser.add_argument(
    "--slope_fix",
    action="store_true",
    help="compare dE/dI1 (not 3 dE/dI1) with its target in the zero-strain slope terms",
)
parser.add_argument(
    "--sym_colloc",
    type=float,
    default=0.0,
    help="weight of the symmetry term (rotated training strains give the same band edges)",
)
parser.add_argument(
    "--sym_labels",
    type=int,
    default=0,
    help="N > 0: labeled states also label their distinct cubic images, at most N images",
)
parser.add_argument(
    "--sym_labels_balanced",
    action="store_true",
    help="with --sym_labels: a labeled state and its k images share one unit of weight",
)
parser.add_argument(
    "--grad_probe_every",
    type=int,
    default=0,
    help="every N iterations log the data and physics gradients to grad_probe.csv",
)
args = parser.parse_args()

if args.model == "mlp":
    for _term in ("cbm", "vbm", "cons", "anchor", "pcbm", "pvbm"):
        if _term not in args.drop_loss:
            args.drop_loss.append(_term)
if args.aphynity:
    args.residual_phys = True
if args.zero_out and (not args.residual_phys or args.aphynity):
    sys.exit("--zero_out applies to the correction of --residual_phys only")
if args.const_head and (not args.residual_phys or not args.zero_out or args.aphynity):
    sys.exit("--const_head needs --residual_phys and --zero_out, and excludes --aphynity")
if args.sym_labels_balanced and args.sym_labels <= 0:
    sys.exit("--sym_labels_balanced needs --sym_labels N > 0")
if args.sym_labels > 0 and (args.model in PAPER_MODELS or not args.no_rba):
    sys.exit("--sym_labels applies to the fixed-weight objective (--model rba or mlp, with --no_rba)")

MODEL_NAME = MODEL_NAMES[args.model] + ("_residual" if args.residual_phys else "")
if args.results_dir is None:
    args.results_dir = os.path.join(".", "results", MODEL_NAME)
DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ======================================================================================================================
# DATA
# ======================================================================================================================
STRAIN_COLUMNS = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]


def strain_family(strain_type):
    """Deformation family of a dataset row (uniaxial_x -> uniaxial, unstrained -> isotropic)."""
    s = str(strain_type).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"


def cubic_operations():
    """The 48 operations of the cubic point group (signed permutation matrices)."""
    return [
        np.diag(sg) @ np.eye(3)[list(pm)]
        for pm in itertools.permutations(range(3))
        for sg in itertools.product([1, -1], repeat=3)
    ]


def strain_matrix(v):
    return np.array([[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]])


def strain_vector(m):
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[1, 2], m[0, 2]])


def symmetry_split(strain_all, stype_all, idx_all, seed):
    """70/10/20 of each family, drawn by symmetry class, so no test state has a rotated copy in training."""
    ops = cubic_operations()
    keys = [
        min(tuple(np.round(strain_vector(R @ strain_matrix(v) @ R.T), 5)) for R in ops)
        for v in strain_all.astype(np.float64)
    ]
    classes = {}
    cls = np.array([classes.setdefault(k, len(classes)) for k in keys])
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for fam in np.unique(stype_all):
        fam_idx = idx_all[stype_all == fam]
        fam_cls = np.unique(cls[fam_idx])
        rng.shuffle(fam_cls)
        n, te_n, va_n = len(fam_idx), 0, 0
        for c in fam_cls:
            members = fam_idx[cls[fam_idx] == c]
            if te_n < round(0.20 * n):
                idx_te.extend(members)
                te_n += len(members)
            elif va_n < round(0.10 * n):
                idx_va.extend(members)
                va_n += len(members)
            else:
                idx_tr.extend(members)
    return np.array(sorted(idx_tr)), np.array(sorted(idx_va)), np.array(sorted(idx_te))


def load_and_split(data_csv, seed, dft_pct):
    """Divide the data (--split), scale the inputs on the training portion and draw the labeled states."""
    df = pd.read_csv(data_csv)
    strain_all = df[STRAIN_COLUMNS].values.astype(np.float32)
    eg_all = df["Eg_eV"].values.astype(np.float32)
    cbm_all = df["CBM_eV"].values.astype(np.float32)
    vbm_all = df["VBM_eV"].values.astype(np.float32)
    stype_all = np.array([strain_family(s) for s in df["strain_type"].values])
    shear_all = df["shear_planes"].values if "shear_planes" in df.columns else np.full(len(df), "", dtype=object)
    idx_all = np.arange(len(df))

    if args.split == "stratified":
        idx_trval, idx_te = train_test_split(idx_all, test_size=0.20, random_state=seed, stratify=stype_all)
        idx_tr, idx_va = train_test_split(idx_trval, test_size=0.125, random_state=seed, stratify=stype_all[idx_trval])
    elif args.split == "symmetry":
        idx_tr, idx_va, idx_te = symmetry_split(strain_all, stype_all, idx_all, seed)
    elif args.split == "group":
        held = strain_family(args.holdout_family)
        idx_te = idx_all[stype_all == held]
        pool = idx_all[stype_all != held]
        if len(idx_te) == 0:
            sys.exit("--holdout_family %s matches no rows" % args.holdout_family)
        idx_tr, idx_va = train_test_split(pool, test_size=0.125, random_state=seed, stratify=stype_all[pool])
    else:
        magnitude = np.max(np.abs(strain_all), axis=1)
        idx_te = idx_all[magnitude > args.magnitude_thresh]
        pool = idx_all[magnitude <= args.magnitude_thresh]
        if len(idx_te) == 0 or len(pool) < 10:
            sys.exit(
                "--magnitude_thresh %g leaves %d train / %d test" % (args.magnitude_thresh, len(pool), len(idx_te))
            )
        idx_tr, idx_va = train_test_split(pool, test_size=0.125, random_state=seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(strain_all[idx_tr]).astype(np.float32)
    tiny = scaler.scale_ < 1e-6
    if tiny.any():
        # A component that is only round-off in training (shear when shear is held out) gets the typical scale.
        scaler.scale_[tiny] = np.median(scaler.scale_[~tiny])
        scaler.mean_[tiny] = 0.0
        X_tr = scaler.transform(strain_all[idx_tr]).astype(np.float32)
        print(
            "  scaler: %d near-constant strain component(s) given the typical scale %.4f"
            % (int(tiny.sum()), scaler.scale_[tiny][0])
        )
    X_va = scaler.transform(strain_all[idx_va]).astype(np.float32)
    X_te = scaler.transform(strain_all[idx_te]).astype(np.float32)

    rng = np.random.default_rng(seed + 9999)
    if dft_pct == 0:
        dft_idx = np.array([], dtype=np.int64)
    else:
        dft_idx = rng.choice(len(idx_tr), size=max(1, round(len(idx_tr) * dft_pct / 100.0)), replace=False)

    vol_tr = strain_all[idx_tr, 0] + strain_all[idx_tr, 1] + strain_all[idx_tr, 2]
    return dict(
        X_tr=X_tr,
        X_va=X_va,
        X_te=X_te,
        strain_tr=strain_all[idx_tr],
        eg_tr=eg_all[idx_tr],
        cbm_tr=cbm_all[idx_tr],
        vbm_tr=vbm_all[idx_tr],
        eg_va=eg_all[idx_va],
        cbm_va=cbm_all[idx_va],
        vbm_va=vbm_all[idx_va],
        eg_te=eg_all[idx_te],
        cbm_te=cbm_all[idx_te],
        vbm_te=vbm_all[idx_te],
        stype_te=stype_all[idx_te],
        stype_tr=stype_all[idx_tr],
        shear_te=shear_all[idx_te],
        idx_te=idx_te,
        dft_idx=dft_idx,
        nz_mask=np.abs(vol_tr) < args.nz_thresh,  # zero-strain slope terms
        anc_mask=np.abs(vol_tr) < args.anc_thresh,  # anchor term
        cbm_mean_tr=float(np.mean(cbm_all[idx_tr])),
        vbm_mean_tr=float(np.mean(vbm_all[idx_tr])),
        scaler=scaler,
        N_tr=len(idx_tr),
    )


# ======================================================================================================================
# ANALYTIC MODELS
# ======================================================================================================================
THETA_KEYS = ["CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2", "VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"]


def physics_cbm(strain, beta=None):
    return _physics_cbm_softmin(strain, beta=args.beta if beta is None else beta)


def cbm_from_theta(strain, th, beta):
    """Conduction-band model of physics.physics_cbm with per-state constants th[:, 0:7]."""
    Exx, Eyy, Ezz, Exy, Eyz, Ezx = [strain[:, i] for i in range(6)]
    I1 = Exx + Eyy + Ezz
    c0, xd, xu, xup, xq, xc, xq2 = [th[:, i] for i in range(7)]

    def valley(e):
        return xd * I1 + xu * e + xq * e**2 + xc * I1 * e + xq2 * I1**2

    if _PHYS.DELTA_X <= 0:
        d = torch.stack(
            [
                valley(Ezz) + 2 * xup * Exy,
                valley(Ezz) - 2 * xup * Exy,
                valley(Eyy) + 2 * xup * Ezx,
                valley(Eyy) - 2 * xup * Ezx,
                valley(Exx) + 2 * xup * Eyz,
                valley(Exx) - 2 * xup * Eyz,
            ],
            dim=1,
        )
    else:
        d = _PHYS._valleys(valley, Exx, Eyy, Ezz, Exy, Eyz, Ezx, xup, torch, lambda v: torch.stack(v, dim=1))
    return c0 - torch.logsumexp(-beta * d, dim=1) / beta


def vbm_from_theta(strain, th, eps=1e-8):
    """Bir-Pikus valence-band model of physics.physics_vbm with per-state constants th[:, 7:13]."""
    Exx, Eyy, Ezz, Exy, Eyz, Ezx = [strain[:, i] for i in range(6)]
    I1 = Exx + Eyy + Ezz
    v0, av, av2, b, b2, dbp = [th[:, i] for i in range(7, 13)]
    hydro = av * I1 + av2 * I1**2
    dxx, dyy, dzz = Exx - I1 / 3.0, Eyy - I1 / 3.0, Ezz - I1 / 3.0
    H = torch.zeros(strain.shape[0], 3, 3, dtype=strain.dtype, device=strain.device)
    H[:, 0, 0] = hydro + (b + b2 * dxx) * dxx
    H[:, 1, 1] = hydro + (b + b2 * dyy) * dyy
    H[:, 2, 2] = hydro + (b + b2 * dzz) * dzz
    H[:, 0, 1] = dbp * Exy
    H[:, 1, 0] = dbp * Exy
    H[:, 1, 2] = dbp * Eyz
    H[:, 2, 1] = dbp * Eyz
    H[:, 0, 2] = dbp * Ezx
    H[:, 2, 0] = dbp * Ezx
    return v0 + _max_eig_sym3(H)


def _max_eig_sym3(H, beta=200.0):
    """Smooth maximum of the eigenvalues of symmetric 3x3 matrices, (1/beta) log tr exp(beta H).

    Shifted by the exact largest eigenvalue (no gradient through it), so the trace is at least 1."""
    with torch.no_grad():
        u = torch.linalg.eigvalsh(torch.nan_to_num(H.double()))[:, 2].to(H.dtype)
    eye = torch.eye(3, dtype=H.dtype, device=H.device).unsqueeze(0)
    ex = torch.linalg.matrix_exp(beta * (H - u[:, None, None] * eye))
    return u + torch.log(torch.diagonal(ex, dim1=1, dim2=2).sum(1)) / beta


# ======================================================================================================================
# NETWORK
# ======================================================================================================================
class PINNModel(nn.Module):
    """depth x width SiLU network from the six scaled strain components to (CBM, VBM); gap = CBM - VBM.

    --residual_phys: the outputs are corrections to the analytic band edges. --aphynity: the analytic constants are
    trainable and shared by all states. --const_head: 13 further outputs set the analytic constants per state."""

    def __init__(self, cbm_mean, vbm_mean, scaler=None):
        super().__init__()
        sizes = [6] + [args.width] * args.depth
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.SiLU()) for i in range(args.depth)
        )
        self.out = nn.Linear(args.width, 2)
        for linear in [blk[0] for blk in self.blocks] + [self.out]:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
        with torch.no_grad():
            self.out.bias[0] = cbm_mean
            self.out.bias[1] = vbm_mean

        self.residual_phys = bool(args.residual_phys)
        if self.residual_phys:
            self.register_buffer("in_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("in_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            with torch.no_grad():
                self.out.bias.zero_()
                if args.zero_out:
                    self.out.weight.zero_()
            if args.aphynity:
                self._register_constants()
                self.aph_u = nn.Parameter(torch.zeros(len(THETA_KEYS)))
            if args.const_head:
                self.out = nn.Linear(args.width, 2 + len(THETA_KEYS))
                with torch.no_grad():
                    self.out.weight.zero_()
                    self.out.bias.zero_()
                self._register_constants()

    def _register_constants(self):
        """Calibrated constants theta0 and the step of each constant (a quarter of its size, at least 0.25)."""
        th0 = [float(getattr(_PHYS, k)) for k in THETA_KEYS]
        scale = [max(abs(v), 1.0) * 0.25 for v in th0]
        scale[-1] = 2.0
        self.register_buffer("theta0", torch.tensor(th0, dtype=torch.float32))
        self.register_buffer("theta_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        h = x
        for blk in self.blocks:
            h = blk(h)
        o = self.out(h)
        cbm, vbm = o[:, 0], o[:, 1]
        if self.residual_phys:
            raw = x * self.in_scale + self.in_mean
            if args.aphynity:
                self.last_fa = o[:, :2]
                th = (self.theta0 + self.theta_scale * self.aph_u).expand(raw.shape[0], -1)
                cbm = cbm + cbm_from_theta(raw, th, args.beta)
                vbm = vbm + vbm_from_theta(raw, th)
            elif args.const_head:
                u = torch.tanh(o[:, 2:]) if args.const_tanh else o[:, 2:]
                th = self.theta0 + self.theta_scale * u
                cbm = cbm + cbm_from_theta(raw, th, args.beta)
                vbm = vbm + vbm_from_theta(raw, th)
            else:
                cbm = cbm + physics_cbm(raw)
                vbm = vbm + physics_vbm(raw)
        return cbm, vbm, cbm - vbm


# ======================================================================================================================
# OBJECTIVE
# ======================================================================================================================
W_CBM, W_VBM, W_CONS = args.w_cbm, args.w_vbm, args.w_cons  # agreement with the analytic models
W_ANCHOR, W_PCBM, W_PVBM = args.w_anchor, args.w_pcbm, args.w_pvbm  # zero-strain level and slopes
W_DATA, W_ISO_EG = args.w_data, args.w_iso_eg  # labeled band edges, isotropic gaps
_DROP = {
    "cbm": "W_CBM",
    "vbm": "W_VBM",
    "cons": "W_CONS",
    "anchor": "W_ANCHOR",
    "pcbm": "W_PCBM",
    "pvbm": "W_PVBM",
    "data": "W_DATA",
    "iso_eg": "W_ISO_EG",
}
for _name in args.drop_loss:
    if _name in _DROP:
        globals()[_DROP[_name]] = 0.0

A_C_REF = XI_D + XI_U / 3.0  # dCBM/dI1 of the analytic models at zero strain
A_V_REF = AV  # dVBM/dI1 of the analytic models at zero strain

PAPER_MODE = args.model in PAPER_MODELS  # squared deviations and the published weight updates
PAPER_W = {}  # per-term, per-state weights of SA-PINN and RBA-PINN
PAPER_UPDATE = [False]  # update the weights only in the network-update call
NTK_W, NTK_STEP, NTK_LOG = {}, [0], []  # NTK-PINN term weights, update counter, history
APH = dict(lam=args.aph_lambda0, step=0, log=[])
BRANCH = {}  # tensors of the symmetry term and the symmetry-copied labels


def update_rba(rba_w, model, X_tr_t, eg_phys_tr):
    """Residual-based weights: r <- gamma r + eta |Eg - Eg_analytic| / max."""
    model.eval()
    with torch.no_grad():
        _, _, eg_pred = model(X_tr_t)
    residuals = torch.abs(eg_pred - eg_phys_tr)
    rba_w[:] = args.rba_gamma * rba_w + args.rba_eta * (residuals / (residuals.max() + 1e-10))
    model.train()


def _ntk_trace(e, params, chunk=128):
    """Trace of this term's block of the neural tangent kernel: sum over points of |d e_i / d theta|^2."""
    n, tot = e.shape[0], 0.0
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        go = torch.zeros(m, n, device=e.device, dtype=e.dtype)
        go[torch.arange(m), torch.arange(s, s + m)] = 1.0
        gs = torch.autograd.grad(
            e, params, grad_outputs=go, retain_graph=True, is_grads_batched=True, allow_unused=True
        )
        tot += sum(float((g.double() ** 2).sum()) for g in gs if g is not None)
    return tot


def _sym_rotated(model, strain_tr_t):
    """Band edges predicted for each training strain under a random cubic operation."""
    ops = BRANCH["ops"][torch.randint(0, 48, (strain_tr_t.shape[0],), device=strain_tr_t.device)]
    T = strain_tr_t.new_zeros(strain_tr_t.shape[0], 3, 3)
    T[:, 0, 0] = strain_tr_t[:, 0]
    T[:, 1, 1] = strain_tr_t[:, 1]
    T[:, 2, 2] = strain_tr_t[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = strain_tr_t[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = strain_tr_t[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = strain_tr_t[:, 5]
    Tr = ops @ T @ ops.transpose(1, 2)
    raw_r = torch.stack([Tr[:, 0, 0], Tr[:, 1, 1], Tr[:, 2, 2], Tr[:, 0, 1], Tr[:, 1, 2], Tr[:, 0, 2]], dim=1)
    cbm_r, vbm_r, _ = model((raw_r - BRANCH["mean"]) / BRANCH["scale"])
    return cbm_r, vbm_r


def compute_loss(
    model,
    X_tr_t,
    strain_tr_t,
    X_anc_t,
    X_nz_t,
    w,
    dft_idx_t,
    cbm_dft_t,
    vbm_dft_t,
    i1_scale,
    eg_tr_t=None,
    return_parts=False,
):
    """Weighted sum of the terms of the objective; w holds the per-state weights."""
    cbm_pred, vbm_pred, eg_pred = model(X_tr_t)
    fa_tr = model.last_fa if args.aphynity else None
    cbm_phys = physics_cbm(strain_tr_t)
    vbm_phys = physics_vbm(strain_tr_t)
    eg_phys = cbm_phys - vbm_phys
    zero = torch.tensor(0.0, device=X_tr_t.device)

    # Isotropic states: no shear, equal normal strains. Agreement terms skip them; their gaps are fitted directly.
    shear_mag = strain_tr_t[:, 3].abs() + strain_tr_t[:, 4].abs() + strain_tr_t[:, 5].abs()
    diag = strain_tr_t[:, :3]
    diag_spread = (diag - diag.mean(dim=1, keepdim=True)).abs().max(dim=1).values
    iso_mask = (shear_mag < 1e-4) & (diag_spread < 1e-4)
    non_iso = ~iso_mask

    # agreement with the analytic models
    if non_iso.sum() > 0:
        L_cbm = W_CBM * (w[non_iso] * torch.abs(cbm_pred[non_iso] - cbm_phys[non_iso])).mean()
        L_vbm = W_VBM * (w[non_iso] * torch.abs(vbm_pred[non_iso] - vbm_phys[non_iso])).mean()
        L_cons = W_CONS * (w[non_iso] * torch.abs(eg_pred[non_iso] - eg_phys[non_iso])).mean()
    else:
        L_cbm, L_vbm, L_cons = zero, zero, zero

    # zero-strain band levels
    if X_anc_t.shape[0] > 0:
        cbm_a, vbm_a, _ = model(X_anc_t)
        L_anchor = W_ANCHOR * (torch.abs(cbm_a - CBM_0).mean() + torch.abs(vbm_a - VBM_0).mean())
    else:
        L_anchor = zero

    # zero-strain slopes dE/dI1, from the gradients in scaled-input space
    X_nz_g = X_nz_t.detach().requires_grad_(True)
    cbm_nz, vbm_nz, _ = model(X_nz_g)
    grad_cbm = torch.autograd.grad(cbm_nz.sum(), X_nz_g, create_graph=True)[0]
    grad_vbm = torch.autograd.grad(vbm_nz.sum(), X_nz_g, create_graph=True)[0]
    sx, sy, sz = i1_scale[0], i1_scale[1], i1_scale[2]
    dCBM_dI1 = grad_cbm[:, 0] / sx + grad_cbm[:, 1] / sy + grad_cbm[:, 2] / sz
    dVBM_dI1 = grad_vbm[:, 0] / sx + grad_vbm[:, 1] / sy + grad_vbm[:, 2] / sz
    if args.slope_fix:
        # the sum of the three partials is the derivative along (t, t, t), i.e. 3 dE/dI1
        dCBM_dI1 = dCBM_dI1 / 3.0
        dVBM_dI1 = dVBM_dI1 / 3.0
    L_pcbm = W_PCBM * torch.abs(dCBM_dI1 - A_C_REF).mean()
    L_pvbm = W_PVBM * torch.abs(dVBM_dI1 - A_V_REF).mean()

    # labeled band edges (with --sym_labels also their cubic images, weighted like their parent)
    if dft_idx_t.numel() > 0 and "aug_x" in BRANCH:
        w_dft = w[dft_idx_t]
        w_aug = w[BRANCH["aug_parent"]]
        cbm_g, vbm_g, _ = model(BRANCH["aug_x"])
        n_all = dft_idx_t.numel() + w_aug.numel()
        if "aug_f" in BRANCH:
            w_dft = w_dft * BRANCH["orig_f"]
            w_aug = w_aug * BRANCH["aug_f"]
            n_all = dft_idx_t.numel()
        L_data = W_DATA * (
            (
                (w_dft * torch.abs(cbm_pred[dft_idx_t] - cbm_dft_t)).sum()
                + (w_aug * torch.abs(cbm_g - BRANCH["aug_cbm"])).sum()
            )
            / n_all
            + (
                (w_dft * torch.abs(vbm_pred[dft_idx_t] - vbm_dft_t)).sum()
                + (w_aug * torch.abs(vbm_g - BRANCH["aug_vbm"])).sum()
            )
            / n_all
        )
    elif dft_idx_t.numel() > 0:
        w_dft = w[dft_idx_t]
        L_data = W_DATA * (
            (w_dft * torch.abs(cbm_pred[dft_idx_t] - cbm_dft_t)).mean()
            + (w_dft * torch.abs(vbm_pred[dft_idx_t] - vbm_dft_t)).mean()
        )
    else:
        L_data = zero

    # gaps of the isotropic training states (always labeled)
    if iso_mask.sum() > 0 and eg_tr_t is not None:
        L_iso_eg = W_ISO_EG * (w[iso_mask] * torch.abs(eg_pred[iso_mask] - eg_tr_t[iso_mask])).mean()
    else:
        L_iso_eg = zero

    if PAPER_MODE:
        # Published forms: squared deviations; SA-PINN weights m(lambda) = lambda^2 on the physics-type terms (McClenny
        # and Braga-Neto 2023); RBA-PINN weights lambda <- gamma lambda + eta |e| / max|e| on every term (Anagnostopoulos
        # et al. 2024); NTK-PINN term weights lambda_k = Tr(K) / Tr(K_kk) (Wang, Yu and Perdikaris 2022).
        dev = X_tr_t.device
        E = {}
        if non_iso.sum() > 0:
            E["cbm"] = (cbm_pred[non_iso] - cbm_phys[non_iso], W_CBM, True)
            E["vbm"] = (vbm_pred[non_iso] - vbm_phys[non_iso], W_VBM, True)
            E["cons"] = (eg_pred[non_iso] - eg_phys[non_iso], W_CONS, True)
        if X_anc_t.shape[0] > 0:
            E["anc_c"] = (cbm_a - CBM_0, W_ANCHOR, True)
            E["anc_v"] = (vbm_a - VBM_0, W_ANCHOR, True)
        E["pcbm"] = (dCBM_dI1 - A_C_REF, W_PCBM, True)
        E["pvbm"] = (dVBM_dI1 - A_V_REF, W_PVBM, True)
        if dft_idx_t.numel() > 0:
            E["data_c"] = (cbm_pred[dft_idx_t] - cbm_dft_t, W_DATA, False)
            E["data_v"] = (vbm_pred[dft_idx_t] - vbm_dft_t, W_DATA, False)
        if iso_mask.sum() > 0 and eg_tr_t is not None:
            E["iso"] = (eg_pred[iso_mask] - eg_tr_t[iso_mask], W_ISO_EG, False)
        E = {k: v for k, v in E.items() if v[1] != 0}
        if args.model == "ntk_paper" and PAPER_UPDATE[0]:
            if NTK_STEP[0] % args.ntk_every == 0:
                params = [p for p in model.parameters() if p.requires_grad]
                tr = {k: _ntk_trace(v[0], params) for k, v in E.items()}
                tot = sum(tr.values())
                for k in tr:
                    NTK_W[k] = tot / tr[k] if tr[k] > 0 else 1.0
                NTK_LOG.append(
                    dict(
                        epoch=NTK_STEP[0],
                        **{"lambda_" + k: v for k, v in NTK_W.items()},
                        **{"trace_" + k: v for k, v in tr.items()},
                    )
                )
            NTK_STEP[0] += 1
        L = {}
        for k, (e, wk, physics_term) in E.items():
            uses_w = args.model == "rba_paper" or (args.model == "sa_paper" and physics_term)
            if uses_w and k not in PAPER_W:  # created after the network, so its initialization is unchanged
                if args.model == "sa_paper":
                    PAPER_W[k] = nn.Parameter(torch.rand(e.shape[0], device=dev))
                else:
                    PAPER_W[k] = torch.zeros(e.shape[0], device=dev)
            if args.model == "rba_paper" and PAPER_UPDATE[0]:
                with torch.no_grad():
                    a_ = e.detach().abs()
                    PAPER_W[k].mul_(args.rba_paper_gamma).add_(args.rba_paper_eta * a_ / (a_.max() + 1e-12))
            if args.model == "sa_paper":
                term = (PAPER_W[k] ** 2 if physics_term else torch.ones_like(e)) * e**2
            elif args.model == "rba_paper":
                term = (PAPER_W[k] * e) ** 2
            else:
                term = e**2
            L[k] = (NTK_W.get(k, 1.0) if args.model == "ntk_paper" else wk) * term.mean()
        L_cbm, L_vbm, L_cons = L.get("cbm", zero), L.get("vbm", zero), L.get("cons", zero)
        L_anchor = L.get("anc_c", zero) + L.get("anc_v", zero)
        L_pcbm, L_pvbm = L.get("pcbm", zero), L.get("pvbm", zero)
        L_data = L.get("data_c", zero) + L.get("data_v", zero)
        L_iso_eg = L.get("iso", zero)

    total = L_cbm + L_vbm + L_cons + L_anchor + L_pcbm + L_pvbm + L_data + L_iso_eg

    if args.aphynity:
        # Yin et al. 2021, Algorithm 1: L = |F_a|^2 + lambda L_traj; lambda <- lambda + tau2 L_traj after each step
        L_traj = zero
        if dft_idx_t.numel() > 0:
            L_traj = (
                L_traj
                + ((cbm_pred[dft_idx_t] - cbm_dft_t) ** 2).mean()
                + ((vbm_pred[dft_idx_t] - vbm_dft_t) ** 2).mean()
            )
        if iso_mask.sum() > 0 and eg_tr_t is not None and W_ISO_EG > 0:
            L_traj = L_traj + ((eg_pred[iso_mask] - eg_tr_t[iso_mask]) ** 2).mean()
        if PAPER_UPDATE[0]:
            if APH["step"] > 0:
                APH["lam"] += args.aph_tau2 * float(L_traj.detach())
            APH["step"] += 1
        L_fa = (fa_tr**2).sum(dim=1).mean()
        total = L_fa + APH["lam"] * L_traj
        L_data, L_cbm = L_traj, L_fa
        if APH["step"] % 1000 == 1 and PAPER_UPDATE[0]:
            APH["log"].append(
                dict(epoch=APH["step"], lam=APH["lam"], L_fa=float(L_fa.detach()), L_traj=float(L_traj.detach()))
            )

    extra = {}
    if args.sym_colloc > 0:
        cbm_r, vbm_r = _sym_rotated(model, strain_tr_t)
        p = 2 if PAPER_MODE else 1
        extra["sym"] = args.sym_colloc * (torch.abs(cbm_r - cbm_pred) ** p + torch.abs(vbm_r - vbm_pred) ** p).mean()
    if args.w_midpoint > 0 and non_iso.sum() > 0:
        midpoint = (cbm_pred[non_iso] + vbm_pred[non_iso]) / 2.0
        midpoint_phys = (cbm_phys[non_iso] + vbm_phys[non_iso]) / 2.0
        extra["midpoint"] = args.w_midpoint * (midpoint - midpoint_phys).abs().mean()
    for v in extra.values():
        total = total + v

    if return_parts:
        return (
            total,
            eg_phys,
            dict(
                data=L_data,
                iso=L_iso_eg,
                cons=L_cons,
                cbm=L_cbm,
                vbm=L_vbm,
                anchor=L_anchor,
                pcbm=L_pcbm,
                pvbm=L_pvbm,
                **extra,
            ),
        )
    return total, eg_phys


# ======================================================================================================================
# TRAINING
# ======================================================================================================================
def _to(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _prepare_symmetry(data, device):
    """Cubic operations and input scaling for the symmetry term; cubic images of the labeled states."""
    if args.sym_colloc > 0:
        BRANCH.update(
            ops=_to(np.array(cubic_operations()), device),
            mean=_to(data["scaler"].mean_, device),
            scale=_to(data["scaler"].scale_, device),
        )
    if args.sym_labels > 0 and len(data["dft_idx"]) > 0:
        parents, raws = [], []
        for i in data["dft_idx"]:
            e = data["strain_tr"][i].astype(np.float64)
            T = strain_matrix(e)
            seen = {tuple(np.round(e, 7))}
            for R in cubic_operations():
                r = strain_vector(R @ T @ R.T)
                key = tuple(np.round(r, 7))
                if key not in seen:
                    seen.add(key)
                    parents.append(int(i))
                    raws.append(r)
        n_images = len(parents)
        if n_images > args.sym_labels:
            keep = np.sort(np.random.default_rng(args.seed + 4241).choice(n_images, args.sym_labels, replace=False))
            parents = [parents[j] for j in keep]
            raws = [raws[j] for j in keep]
        if parents:
            raws = np.array(raws)
            BRANCH.update(
                aug_x=_to((raws - data["scaler"].mean_) / data["scaler"].scale_, device),
                aug_parent=torch.tensor(parents, dtype=torch.long, device=device),
                aug_cbm=_to(data["cbm_tr"][parents], device),
                aug_vbm=_to(data["vbm_tr"][parents], device),
            )
            if args.sym_labels_balanced:
                count = {}
                for q in parents:
                    count[q] = count.get(q, 0) + 1
                BRANCH.update(
                    aug_f=_to(np.array([1.0 / (count[q] + 1) for q in parents]), device),
                    orig_f=_to(np.array([1.0 / (count.get(int(i), 0) + 1) for i in data["dft_idx"]]), device),
                )
        print(
            "  symmetry labels: %d labeled states -> %d distinct images, %d used"
            % (len(data["dft_idx"]), n_images, len(parents)),
            flush=True,
        )


def train(data, device):
    """AdamW, then SOAP, with cosine warm restarts; keeps the checkpoint with the lowest validation MAE."""
    t0 = time.time()
    X_tr_t, X_va_t, X_te_t = _to(data["X_tr"], device), _to(data["X_va"], device), _to(data["X_te"], device)
    strain_tr_t = _to(data["strain_tr"], device)
    X_anc_t = X_tr_t[data["anc_mask"]]
    X_nz_t = X_tr_t[data["nz_mask"]]
    i1_scale = data["scaler"].scale_[:3].astype(np.float32)
    if len(data["dft_idx"]) > 0:
        dft_idx_t = torch.tensor(data["dft_idx"], dtype=torch.long, device=device)
        cbm_dft_t = _to(data["cbm_tr"][data["dft_idx"]], device)
        vbm_dft_t = _to(data["vbm_tr"][data["dft_idx"]], device)
    else:
        dft_idx_t = torch.zeros(0, dtype=torch.long, device=device)
        cbm_dft_t = torch.zeros(0, device=device)
        vbm_dft_t = torch.zeros(0, device=device)
    eg_tr_t = _to(data["eg_tr"], device)
    _prepare_symmetry(data, device)

    model = PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"], scaler=data["scaler"]).to(device)
    rba_w = torch.zeros(data["N_tr"], device=device)  # per-state weights are rba_w + 1
    sa_paper_opt = None
    with torch.no_grad():
        eg_phys_tr = physics_cbm(strain_tr_t) - physics_vbm(strain_tr_t)

    def loss_inputs():
        return (model, X_tr_t, strain_tr_t, X_anc_t, X_nz_t, rba_w + 1.0, dft_idx_t, cbm_dft_t, vbm_dft_t, i1_scale)

    def mae(X_t, y):
        model.eval()
        with torch.no_grad():
            _, _, eg = model(X_t)
        model.train()
        return float(torch.abs(eg - _to(y, device)).mean().item())

    half = int(round(args.switch_frac * args.epochs))
    opt = optim.AdamW(model.parameters(), lr=args.adamw_lr, weight_decay=args.adamw_wd, betas=(0.9, 0.999))
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2, eta_min=1e-5)

    best_val_mae, best_state, best_ckpt_epoch = float("inf"), None, 0
    last_val_mae, gpu_peak_mb = float("nan"), 0.0
    epoch_rows, probe_rows = [], []
    print(
        f"\nTraining on {device}  |  {data['N_tr']} train  |  {len(data['dft_idx'])} labeled  |  {args.epochs} epochs"
    )

    for ep in range(1, args.epochs + 1):
        if ep == half + 1:
            opt = SOAP(
                model.parameters(),
                lr=args.soap_lr,
                betas=(0.95, 0.95),
                weight_decay=args.soap_wd,
                precondition_frequency=10,
            )
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=2, eta_min=1e-5)
        model.train()

        if args.grad_probe_every and (ep == 1 or ep % args.grad_probe_every == 0):
            probe_rows.append(gradient_probe(ep, loss_inputs(), eg_tr_t, device))
            opt.zero_grad()

        opt.zero_grad()
        PAPER_UPDATE[0] = True
        loss, _ = compute_loss(*loss_inputs(), eg_tr_t=eg_tr_t)
        PAPER_UPDATE[0] = False
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        if args.soap_warmup > 0 and 1 <= ep - half <= args.soap_warmup:
            for group in opt.param_groups:
                group["lr"] = group["lr"] * (ep - half) / args.soap_warmup
        opt.step()
        sch.step()

        if args.model == "sa_paper":
            # gradient ascent on the self-adaptive weights
            if sa_paper_opt is None:
                sa_paper_opt = optim.Adam(list(PAPER_W.values()), lr=args.sa_paper_lr)
            sa_paper_opt.zero_grad()
            loss_sa, _ = compute_loss(*loss_inputs(), eg_tr_t=eg_tr_t)
            (-loss_sa).backward()
            sa_paper_opt.step()
        elif args.model in ("rba_paper", "ntk_paper"):
            pass  # updated inside the network-update call
        elif not args.no_rba and ep % 100 == 0:
            update_rba(rba_w, model, X_tr_t, eg_phys_tr)

        if ep % args.ckpt_interval == 0:
            last_val_mae = mae(X_va_t, data["eg_va"])
            if last_val_mae < best_val_mae:
                best_val_mae, best_ckpt_epoch = last_val_mae, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        cur_lr = sch.get_last_lr()[0]
        gm = torch.cuda.memory_allocated(device) / 1e6 if DEVICE.type == "cuda" else 0.0
        gpu_peak_mb = max(gpu_peak_mb, gm)
        with torch.no_grad():
            _, _, eg_pred_t = model(X_tr_t)
        tr_mae = float(torch.abs(eg_pred_t.detach() - _to(data["eg_tr"], device)).mean().item())
        epoch_rows.append(
            {
                "epoch": ep,
                "train_mae_eV": round(tr_mae, 6),
                "val_mae_eV": round(last_val_mae, 6),  # last validation (every ckpt_interval)
                "val_best_eV": round(best_val_mae, 6) if best_val_mae < float("inf") else "",
                "test_mae_eV": round(mae(X_te_t, data["eg_te"]), 6),  # logged only; never used for selection
                "lr": cur_lr,
                "elapsed_s": round(time.time() - t0, 2),
                "gpu_mem_mb": round(gm, 1),
                "cpu_percent": 0.0,
                "ram_mb": 0.0,
            }
        )
        if ep % 500 == 0:
            print(
                f"  ep {ep:>5} [{'AdamW' if ep <= half else 'SOAP'}]  lr={cur_lr:.2e}  "
                f"train={tr_mae * 1000:.1f} meV  val={best_val_mae * 1000:.1f} meV"
            )

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model.pt"))
    pd.DataFrame(epoch_rows).to_csv(os.path.join(args.results_dir, "epoch_log.csv"), index=False)
    if probe_rows:
        pd.DataFrame(probe_rows).to_csv(os.path.join(args.results_dir, "grad_probe.csv"), index=False)
    if APH["log"]:
        pd.DataFrame(APH["log"]).to_csv(os.path.join(args.results_dir, "aphynity_lambda.csv"), index=False)
    if NTK_LOG:
        pd.DataFrame(NTK_LOG).to_csv(os.path.join(args.results_dir, "ntk_weights.csv"), index=False)
    if PAPER_W:
        pd.DataFrame(
            [
                dict(
                    term=k,
                    n=int(v.numel()),
                    mean=float(v.detach().mean()),
                    min=float(v.detach().min()),
                    max=float(v.detach().max()),
                )
                for k, v in PAPER_W.items()
            ]
        ).to_csv(os.path.join(args.results_dir, "paper_weights.csv"), index=False)
    if args.model == "rba" and not args.no_rba:
        pd.DataFrame({"strain_type": data["stype_tr"], "sample_weight": (rba_w + 1.0).cpu().numpy()}).to_csv(
            os.path.join(args.results_dir, "sample_weights.csv"), index=False
        )
    return model, elapsed, gpu_peak_mb, best_ckpt_epoch


def gradient_probe(ep, inputs, eg_tr_t, device):
    """Norms and cosine of the data and physics gradients; restores the random state, so training is unchanged."""
    rng_cpu = torch.get_rng_state()
    model = inputs[0]
    rng_gpu = torch.cuda.get_rng_state(model.out.weight.device) if model.out.weight.is_cuda else None
    _, _, parts = compute_loss(*inputs, eg_tr_t=eg_tr_t, return_parts=True)
    params = [p for p in model.parameters() if p.requires_grad]
    L_data = parts["data"] + parts["iso"]
    L_phys = parts["cons"] + parts["cbm"] + parts["vbm"] + parts["anchor"] + parts["pcbm"] + parts["pvbm"]

    def flat(L):
        if not (torch.is_tensor(L) and L.requires_grad):
            return torch.zeros(sum(p.numel() for p in params), device=device)
        gs = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])

    gd, gp = flat(L_data), flat(L_phys)
    nd, npn = float(gd.norm()), float(gp.norm())
    torch.set_rng_state(rng_cpu)
    if rng_gpu is not None:
        torch.cuda.set_rng_state(rng_gpu, model.out.weight.device)
    return dict(
        epoch=ep,
        L_data=float(L_data),
        L_phys=float(L_phys),
        grad_data_norm=nd,
        grad_phys_norm=npn,
        cos=float(gd @ gp) / (nd * npn) if nd > 0 and npn > 0 else float("nan"),
        **{"term_" + k: float(v) for k, v in parts.items()},
    )


# ======================================================================================================================
# EVALUATION
# ======================================================================================================================
def evaluate(model, data, elapsed, gpu_peak_mb, best_ckpt_epoch, device):
    """Test metrics of the kept checkpoint; writes test_results.csv and metrics_summary.csv."""
    model.eval()
    with torch.no_grad():
        cbm_pred, vbm_pred, eg_pred = model(_to(data["X_te"], device))
        _, _, eg_tr_pred = model(_to(data["X_tr"], device))
        _, _, eg_va_pred = model(_to(data["X_va"], device))
    eg_pred_np, cbm_pred_np, vbm_pred_np = eg_pred.cpu().numpy(), cbm_pred.cpu().numpy(), vbm_pred.cpu().numpy()
    eg_true, cbm_true, vbm_true = data["eg_te"], data["cbm_te"], data["vbm_te"]

    abs_err = np.abs(eg_pred_np - eg_true)
    test_mae = float(abs_err.mean())
    test_rmse = float(np.sqrt((abs_err**2).mean()))
    test_max = float(abs_err.max())
    test_p95 = float(np.percentile(abs_err, 95))
    test_mape = float((abs_err / np.abs(eg_true)).mean() * 100.0)
    eg_ref = CBM_0 - VBM_0
    sign_acc = float(np.mean(np.sign(eg_pred_np - eg_ref) == np.sign(eg_true - eg_ref)) * 100.0)
    cbm_mae = float(np.abs(cbm_pred_np - cbm_true).mean())
    vbm_mae = float(np.abs(vbm_pred_np - vbm_true).mean())
    test_r2 = float(r2_score(eg_true, eg_pred_np))
    val_mae = float(np.abs(eg_va_pred.cpu().numpy() - data["eg_va"]).mean())
    train_mae = float(np.abs(eg_tr_pred.cpu().numpy() - data["eg_tr"]).mean())

    buckets = {t: [] for t in C.STRAIN_FAMILIES}
    for i, st in enumerate(data["stype_te"]):
        if str(st).lower() in buckets:
            buckets[str(st).lower()].append(abs_err[i])

    def family_mae(errors):
        return round(float(np.mean(errors)) * 1000, 2) if errors else float("nan")

    print(f"\n  test  MAE : {test_mae * 1000:.2f} meV   RMSE: {test_rmse * 1000:.2f} meV   R²={test_r2:.4f}")
    print(f"  max err   : {test_max * 1000:.2f} meV   p95: {test_p95 * 1000:.2f} meV   MAPE: {test_mape:.2f}%")
    print(f"  sign acc  : {sign_acc:.1f}%")
    print(f"  val   MAE : {val_mae * 1000:.2f} meV")
    print(f"  train MAE : {train_mae * 1000:.2f} meV")
    print(f"  CBM MAE   : {cbm_mae * 1000:.2f} meV   VBM MAE: {vbm_mae * 1000:.2f} meV")
    for k, v in buckets.items():
        print(f"  {k:>12}: {family_mae(v):.1f} meV")

    pd.DataFrame(
        {
            "sample_id": data["idx_te"],
            "strain_type": data["stype_te"],
            "shear_planes": [str(s) if str(s) != "nan" else "" for s in data["shear_te"]],
            "Eg_true_eV": eg_true,
            "Eg_pred_eV": eg_pred_np,
            "CBM_true_eV": cbm_true,
            "CBM_pred_eV": cbm_pred_np,
            "VBM_true_eV": vbm_true,
            "VBM_pred_eV": vbm_pred_np,
            "abs_error_eV": abs_err,
            "signed_error_eV": eg_pred_np - eg_true,
            "seed": args.seed,
            "dft_pct": args.dft_pct,
        }
    ).to_csv(os.path.join(args.results_dir, "test_results.csv"), index=False)

    summary = {
        "model_name": MODEL_NAME,
        "seed": args.seed,
        "dft_pct": args.dft_pct,
        "n_labeled": len(data["dft_idx"]),
        "test_mae_meV": round(test_mae * 1000, 2),
        "test_rmse_meV": round(test_rmse * 1000, 2),
        "test_max_err_meV": round(test_max * 1000, 2),
        "test_p95_err_meV": round(test_p95 * 1000, 2),
        "test_mape_pct": round(test_mape, 3),
        "sign_accuracy_pct": round(sign_acc, 2),
        "val_mae_meV": round(val_mae * 1000, 2),
        "train_mae_meV": round(train_mae * 1000, 2),
        "test_r2": round(test_r2, 4),
        "cbm_mae_meV": round(cbm_mae * 1000, 2),
        "vbm_mae_meV": round(vbm_mae * 1000, 2),
        **{"mae_%s_meV" % t: family_mae(buckets[t]) for t in C.STRAIN_FAMILIES},
        "total_epochs": args.epochs,
        "epochs_run": args.epochs,
        "best_ckpt_epoch": best_ckpt_epoch,
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed, 1),
        "gpu_mem_peak_mb": round(gpu_peak_mb, 1),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.results_dir, "metrics_summary.csv"), index=False)
    return summary


def record_resources(results_dir):
    """One CPU and memory reading per run, written into every row of epoch_log.csv."""
    try:
        import psutil
    except ImportError:
        return
    path = os.path.join(results_dir, "epoch_log.csv")
    df = pd.read_csv(path)
    df["cpu_percent"] = psutil.cpu_percent(interval=None)
    df["ram_mb"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
    df.to_csv(path, index=False)


def main():
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "command.txt"), "w") as f:
        const = os.environ.get("PHYSICS_CONST_JSON", "")
        f.write(("PHYSICS_CONST_JSON=%s " % const if const else "") + "python " + " ".join(sys.argv) + "\n")
    print(f"Loading data: {args.data_csv}")
    data = load_and_split(args.data_csv, args.seed, args.dft_pct)
    print(
        f"  train={data['N_tr']}  val={len(data['eg_va'])}  test={len(data['eg_te'])}  labeled={len(data['dft_idx'])}"
    )
    model, elapsed, gpu_peak_mb, best_ckpt_epoch = train(data, DEVICE)
    evaluate(model, data, elapsed, gpu_peak_mb, best_ckpt_epoch, DEVICE)
    record_resources(args.results_dir)
    print("\n  wrote %s" % args.results_dir)


if __name__ == "__main__":
    main()
