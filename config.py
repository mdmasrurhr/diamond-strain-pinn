# ==============================================================================
# config.py -- every setting for every experiment, in one file.
#
# Nothing here is computed and nothing here is clever. If you want to know what
# value a study used, it is on one of the lines below. Scripts import this file
# and never hard-code a number of their own.
#
# Each setting below carries its provenance, because a value with no provenance
# is a value nobody can defend:
#
#   [decided]  measured, and this is what the measurement chose
#   [pending]  a stage-1 script measures this; the value here is the thesis
#              default and stands only until that script has run
#   [fixed]    theory or convention, argued rather than swept
#
# Changing one of these changes every study that follows, so change them
# deliberately and re-run from the stage that owns them.
# ==============================================================================

import os
import sys

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "dft_hy_v9.csv")
RESULTS = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- dataset
# The six inputs and three targets, exactly as named in the CSV.
STRAIN_COLS = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
TARGET_EG, TARGET_CBM, TARGET_VBM = "Eg_eV", "CBM_eV", "VBM_eV"

# [pending decide_split.py] 70/10/20, stratified by deformation family, keyed
# to the seed. Stratification keeps the rare families represented and measures
# interpolation within the sampled region. The group and magnitude splits measure
# extrapolation instead; they are reported alongside, never instead.
TEST_FRACTION = 0.20
VAL_FRACTION_OF_REMAINDER = 0.125      # 0.125 of the remaining 80% = 10% of total
STRAIN_FAMILIES = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]

# The labelled subset is drawn from the training fold by this offset stream, so
# every model at a given (seed, fraction) sees the identical labelled rows.
LABEL_RNG_OFFSET = 9999

# ---------------------------------------------------------------- experiment grid
# [decided decide_budget.py] 17 seeds. The power analysis on the recorded
# spread says this resolves ~1.8 meV for the PINNs (sigma 1.9 meV), which is the
# right order for the differences the paper claims. It does NOT resolve the Shi
# baseline, whose sigma is 8.9 meV and which would need ~308 seeds for 2 meV --
# so no claim about the baseline finer than ~8.5 meV is supported at this N.
SEEDS = list(range(42, 59))
FRACTIONS = [0, 1, 2, 5, 10, 25, 50, 75, 100]     # percent of the training pool
MODELS = ["sa", "rba", "shi", "mlp"]    # mlp = physics-free control

# ---------------------------------------------------------------- tuned config
# Optimizer values selected by random search on validation error at 100% labels.
TUNED = dict(
    # [decided decide_architecture.py, v9, 3 seeds] The parsimony rule --
    # smallest within 1 s.d. of the best -- selects 8 x 64 (12.84 meV, 29,698
    # params) over the thesis 5 x 100 (17.48 meV, 41,302 params). Every depth
    # >= 6 beat depth 5. Dropout was REJECTED at p = 0.000 (+145 to +278 meV);
    # LayerNorm / residual / taper / widen showed no measurable effect and are
    # excluded as unearned complexity.
    width=64,
    depth=8,
    activation="silu",
    scaler="standard",
    # [decided decide_budget.py] 24,000 is the first NON-BINDING budget: 0%
    # of runs kept their final checkpoint, where 6,000 was 100% binding and left
    # ~5.6 meV on the table. Early stopping rejected on evidence: patience-8
    # costs 1.9 meV to save 7k epochs, shorter patience is catastrophic.
    epochs=24000,
    adamw_lr=2.636e-3,
    adamw_wd=1.793e-5,
    soap_lr=7.325e-3,
    soap_wd=1.224e-3,
    # [decided decide_optimizer.py] The three-arm study settled attribution:
    # AdamW-only 55.2, SOAP-only 16.7, two-phase 17.5 meV. The switch buys
    # nothing beyond SOAP (-0.7, noise); the two-phase form is kept for
    # continuity with the thesis and is statistically indistinguishable from
    # SOAP-only -- but the CREDIT goes to SOAP, and the paper must say so.
    # switch_frac screened {0.25, 0.5, 0.75}: 0.5 best. clip_norm screened
    # {0.1..off}: flat -- a validated nuisance.
    switch_frac=0.5,
    clip_norm=1.0,
    # [decided] Screened over {50, 100, 250, 500, 1000}, flat to 0.01 meV.
    ckpt_interval=250,
)
# The budget question is settled (see TUNED["epochs"] above).

# Per-sample weighting. RBA follows a fixed residual rule; SA learns its weights.
#
# [decided] Keeping BOTH variants is a deliberate choice and the paper has to
# justify it, so the justification lives here rather than being reconstructed
# later. The two differ in exactly one respect -- how each training sample is
# weighted -- with architecture, optimizer, schedule, data and seeds identical.
# That makes it a clean single-factor comparison and it answers a real question:
# whether LEARNING the weights is worth the extra loss evaluation and backward
# pass per epoch that SA costs. At full data it is (12.89 against 16.44 meV). In
# the scarce-label regime the two are statistically indistinguishable, which is
# itself the result: where the physics dominates, the weighting rule stops
# mattering. Reporting only the winner would discard that.
RBA_GAMMA, RBA_ETA, RBA_UPDATE_EVERY = 0.999, 0.1, 100
# [decided decide_optimizer.py, group sa_clamp] Screened {2, 4, 8, 16}:
# flat (clamp_16 better by 0.9 meV, p = 0.69 -- noise). 8 kept.
SA_LR, SA_CLAMP = 8.07e-3, (-8.0, 8.0)

# Loss term weights. Dropping a term (script 06) sets its weight to zero.
# [pending] Four of these sit on CLIFFS in their own screens, not plateaus:
# w_data, w_cons, w_cbm and w_vbm each have a neighbouring value more than twice
# as bad. w_cons is the worst case -- 2.0 gives 12.45 meV and 4.0 gives 117.9,
# a tenfold degradation one step away. A nuisance parameter is only validated by
# a FLAT screen, so these are not yet validated: either move to a stable
# interior point or report the screens and say the value is sensitive.
# These values are now AUTHORITATIVE: jobs.train_cmd passes each one to the
# trainer explicitly, so what this dict says is what a run uses. (They used to
# be decorative -- read by nothing, and disagreeing with the trainer's argparse
# defaults on pcbm, pvbm and iso_eg. The values below are the trainer's, i.e.
# what every thesis run actually used.)
LOSS_WEIGHTS = dict(data=5.0, cons=2.0, cbm=1.0, vbm=1.0, anchor=0.5,
                    pcbm=0.5, pvbm=0.5, iso_eg=5.0)

# Masks for the boundary-condition terms.
# [decided] Screened; nz_thresh is a slope rather than a plateau (12.45 -> 19.89
# across its range), so it is documented as sensitive rather than as flat.
NZ_THRESH = 0.01                        # "near zero strain" for the slope terms
ANC_THRESH = 1e-4                       # "at zero strain" for the anchor term
# [pending] The analytic error table says beta=50 carries a 4.57 meV systematic
# error against an exact six-valley minimum -- about a third of the model's own
# 12.9 meV accuracy, introduced by a numerical convenience. beta=100 halves it
# (2.00 meV) and beta=200 quarters it again. The only argument for the smaller
# value is gradient smoothness, which is asserted nowhere quantified. Raise it,
# or measure the gradient cost.
SOFTMIN_BETA = 50.0                     # sharpness of the 6-valley soft minimum

# ---------------------------------------------------------------- compute
GPUS = [0, 1]
RUNS_PER_GPU = 4                        # 8 concurrent; each run needs ~20 MB

# Thread count for the CPU-side maths, pinned rather than left to the machine.
# This is not only about speed. Thread count changes the order in which
# floating-point reductions are summed, which changes low-order bits. The two
# PINNs are unaffected (full batch, on the GPU), but the Shi baseline shifts by
# up to ~0.5% because it batches on the CPU -- so leaving this to whatever the
# host happens to have would make its numbers machine-dependent. Pinned here,
# every run on any machine gives the same answer.
OMP_THREADS = 2
# The interpreter used to launch training jobs. Defaults to the one running
# this file, which is what you want in a virtual environment. Override with the
# DSP_PYTHON environment variable to use a different one.
PYTHON = os.environ.get("DSP_PYTHON", sys.executable)


def results_dir(*parts):
    """results/<parts...>, created on demand. Every script writes through this."""
    p = os.path.join(RESULTS, *parts)
    os.makedirs(p, exist_ok=True)
    return p
