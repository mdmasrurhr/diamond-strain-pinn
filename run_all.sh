#!/usr/bin/env bash
# run_all.sh -- the whole study, in order, from a clean checkout.
#
# The stages must run in order, because each one settles something the next
# assumes:
#
#   setup     what data we have and what the physics constants are
#   decide    settle every training setting BEFORE the large run
#   measure   the main experiment: accuracy against how many labels were used
#   ablate    what each part of the model contributes
#   validate  what the trained model can and cannot do
#   tune      a large hyperparameter search, AFTER the main experiment
#   report    tables and figures
#
# The "decide" stage is the one it is tempting to skip. Each of its scripts
# writes a decision file. Read them, copy the chosen values into config.py, and
# only then run "measure". Running the main experiment first means measuring a
# configuration that the decide stage then tells you was the wrong one.
#
#   ./run_all.sh                    the full study
#   ./run_all.sh --quick            3 seeds instead of 17, for a smoke test
#   ./run_all.sh --stage decide     one stage only
#
# Every stage is safe to re-run: a job whose output already exists is skipped,
# so an interrupted study picks up where it stopped.

set -u
cd "$(dirname "$0")"

# Uses whichever python is on PATH; override with DSP_PYTHON if needed.
PY="${DSP_PYTHON:-python3}"

QUICK=""
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK="--quick"; shift ;;
    --stage) ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

# The decide-stage scripts screen at 3 seeds by default and take --full for all
# 17. A quick run keeps the 3.
FULL=""
[ -z "$QUICK" ] && FULL="--full"

banner () { printf '\n=========== %s ===========\n' "$1"; }
stage  () { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# --------------------------------------------------------------------- setup
if stage setup; then
  banner "environment manifest"
  $PY check_environment.py

  banner "dataset checks"
  $PY check_data.py

  banner "physics fit and calibration cost"
  $PY fit_physics_constants.py
fi

# -------------------------------------------------------------------- decide
if stage decide; then
  banner "sanity checks (cheap, and they fail loudly -- run these first)"
  $PY check_sanity.py --all

  banner "which data split"
  $PY decide_split.py $FULL

  banner "which architecture"
  $PY decide_architecture.py $FULL

  banner "which optimizer"
  $PY decide_optimizer.py $FULL

  banner "how many epochs and seeds"
  $PY decide_budget.py $FULL --patience

  echo
  echo ">>> STOP HERE. Read the decision files in results/, update config.py,"
  echo ">>> then run the measure stage. Do not run the main experiment on"
  echo ">>> settings the decide stage has not confirmed."
fi

# ------------------------------------------------------------------- measure
if stage measure; then
  banner "confirm the settings still hold"
  $PY confirm_settings.py
  $PY confirm_settings.py --guard || { echo "settings not confirmed; stopping"; exit 1; }

  banner "accuracy against labelled fraction (the main experiment)"
  $PY run_label_sweep.py $QUICK

  banner "physics prior fitted on the same labels"
  $PY run_matched_prior.py $QUICK

  banner "classical baselines"
  $PY run_classical_baselines.py
fi

# -------------------------------------------------------------------- ablate
if stage ablate; then
  banner "remove one loss term at a time"
  $PY ablate_loss_terms.py $QUICK

  banner "vary one design choice at a time"
  $PY ablate_design_choices.py $QUICK

  banner "repeat on distinct strain states only"
  $PY ablate_duplicates.py $QUICK
fi

# ------------------------------------------------------------------ validate
if stage validate; then
  # The validation scripts read one finished run. This is the best model from
  # the main experiment.
  BEST=results/label_sweep/sa/100pct/seed42

  banner "does it obey physics it was not fitted to"
  $PY validate_physics.py

  banner "memorisation, unseen families, unseen strain magnitudes"
  $PY validate_generalization.py

  banner "noisy input, cubic symmetry, shear sign"
  $PY validate_robustness.py --run_dir "$BEST"

  banner "right answer for the right reason"
  $PY validate_explainability.py --run_dir "$BEST"

  banner "speed, memory, and what it replaces"
  $PY validate_performance.py --run_dir "$BEST"
fi

# ---------------------------------------------------------------------- tune
# This comes last on purpose. Searching for the best settings and then reporting
# that best number as the headline result is selecting on the quantity being
# reported. The search answers a different question: how much accuracy was left
# on the table.
if stage tune; then
  banner "hyperparameter search"
  $PY search_hyperparameters.py --trials 300
  $PY search_hyperparameters.py --confirm
fi

# -------------------------------------------------------------------- report
if stage report; then
  banner "collect every result into tables"
  $PY make_tables.py

  banner "draw the figures"
  $PY make_figures.py

  echo
  echo "done. tables are in results/tables/ and figures in results/figures/"
fi
