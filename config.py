"""Settings shared by every script in the pipeline.

Scripts import this module instead of hard-coding values, so a setting appears
in exactly one place. Command-line flags override these defaults.
"""

import os
import sys

# --- paths ---
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "dft_hy_v9.csv")
RESULTS = os.path.join(ROOT, "results")

# --- dataset columns ---
# Model inputs: the six independent components of the strain tensor.
STRAIN_COLS = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
# Model targets: the bandgap and the two band edges it is the difference of.
TARGET_EG, TARGET_CBM, TARGET_VBM = "Eg_eV", "CBM_eV", "VBM_eV"

# --- train / validation / test split ---
TEST_FRACTION = 0.20
# 0.125 of the remaining 80% gives 10% of the total.
VAL_FRACTION_OF_REMAINDER = 0.125
# The split is stratified on these, so each appears in every fold.
STRAIN_FAMILIES = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]

# Offset applied to the seed when drawing the labelled subset. Keeping it on a
# separate stream means the subset does not change when the split does.
LABEL_RNG_OFFSET = 9999

# --- experiment grid ---
SEEDS = list(range(42, 59))
# Percentage of the training pool whose labels the model may use.
FRACTIONS = [0, 1, 2, 5, 10, 25, 50, 75, 100]
# sa and rba are the physics-informed variants, shi the literature baseline,
# mlp the same network with the physics terms switched off.
MODELS = ["sa", "rba", "shi", "mlp"]

# --- network and optimizer ---
TUNED = dict(
    width=64,
    depth=8,
    activation="silu",
    scaler="standard",
    epochs=24000,
    # AdamW runs the first phase, SOAP the second.
    adamw_lr=2.636e-3,
    adamw_wd=1.793e-5,
    soap_lr=7.325e-3,
    soap_wd=1.224e-3,
    # Fraction of the budget spent in the first phase before switching.
    switch_frac=0.5,
    clip_norm=1.0,
    # Epochs between validation checks. The best checkpoint is kept.
    ckpt_interval=250,
)

# --- per-sample weighting ---
# RBA: weight follows an exponential moving average of the physics residual.
RBA_GAMMA, RBA_ETA, RBA_UPDATE_EVERY = 0.999, 0.1, 100
# SA: weight is a learned parameter, bounded by the clamp to stay finite.
SA_LR, SA_CLAMP = 8.07e-3, (-8.0, 8.0)

# --- loss term weights ---
# One per term in the training objective. Setting a weight to zero removes that
# term, which is how the ablation scripts work.
LOSS_WEIGHTS = dict(data=5.0, cons=2.0, cbm=1.0, vbm=1.0, anchor=0.5,
                    pcbm=0.5, pvbm=0.5, iso_eg=5.0)

# --- boundary condition masks ---
# Samples below these strain magnitudes are treated as near zero strain by the
# gradient terms and as zero strain by the anchor term.
NZ_THRESH = 0.01
ANC_THRESH = 1e-4
# Sharpness of the smooth minimum over the six conduction valleys. Larger is
# closer to an exact minimum but gives sharper gradients.
SOFTMIN_BETA = 50.0

# --- compute ---
GPUS = [0, 1]
# Concurrent runs per GPU. Each run needs about 20 MB.
RUNS_PER_GPU = 4

# Thread count for CPU-side maths. Pinned because the number of threads changes
# the order floating-point sums are accumulated in, which moves the baseline
# model by up to 0.5% between machines.
OMP_THREADS = 2

# Interpreter used to launch training jobs. Defaults to the one running this
# file; override with the DSP_PYTHON environment variable.
PYTHON = os.environ.get("DSP_PYTHON", sys.executable)


def results_dir(*parts):
    """Return results/<parts...>, creating the directory if needed."""
    path = os.path.join(RESULTS, *parts)
    os.makedirs(path, exist_ok=True)
    return path
