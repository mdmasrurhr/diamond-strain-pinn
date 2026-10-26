# ==============================================================================
# config.py -- shared settings of the study.
#
# The trainer (03_train_pinn.py) takes every model setting as a command-line
# option; make_tasks.py writes the exact command of every run of the paper, using
# the values below. The other scripts read the paths, seeds and labeled fractions
# from here.
# ==============================================================================

import os

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "dft_hy_v9.csv")
RESULTS = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- dataset
STRAIN_FAMILIES = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]

# ---------------------------------------------------------------- experiment grid
# 17 runs per model and setting; the seed sets the data division, the labeled
# states and the initial weights. FRACTIONS are the labeled fractions of the
# training portion, in percent.
SEEDS = list(range(42, 59))
FRACTIONS = [0, 1, 2, 5, 10, 25, 50, 75, 100]

# ---------------------------------------------------------------- network and optimizer
# Fixed in preliminary studies on a stratified random division (Supplementary
# Tables 9 and 10 of the paper): the smallest network within one standard
# deviation of the best (8 x 64), 24,000 iterations (12,000 AdamW, then 12,000
# SOAP), gradient clipping at 1.0, validation every 250 iterations. Not retuned
# for any model of the paper.
TUNED = dict(
    width=64,
    depth=8,
    activation="silu",
    scaler="standard",
    epochs=24000,
    adamw_lr=2.636e-3,
    adamw_wd=1.793e-5,
    soap_lr=7.325e-3,
    soap_wd=1.224e-3,
    switch_frac=0.5,
    clip_norm=1.0,
    ckpt_interval=250,
)

# ---------------------------------------------------------------- loss weights
# Base weights of the objective (Table 5 of the paper). make_tasks.py passes each
# one to the trainer explicitly and then appends the model's own options, which
# override them where a model differs (argparse keeps the last value given).
LOSS_WEIGHTS = dict(data=5.0, cons=2.0, cbm=1.0, vbm=1.0, anchor=0.5,
                    pcbm=0.5, pvbm=0.5, iso_eg=5.0)

# Sharpness of the soft minimum over the six conduction valleys (eV^-1).
SOFTMIN_BETA = 50.0


def results_dir(*parts):
    """results/<parts...>, created on demand."""
    p = os.path.join(RESULTS, *parts)
    os.makedirs(p, exist_ok=True)
    return p
