#!/usr/bin/env bash
# run_all.sh -- the whole study, in order, from a clean checkout.
#
# Scripts are numbered in the order they run. They group into stages, and the
# stages must run in order because each settles something the next assumes.
#
# The decide stage writes a decision file for each setting. Read those, copy the
# chosen values into config.py, and only then run the measure stage.
#
#   ./run_all.sh                    the full study
#   ./run_all.sh --quick            3 seeds instead of 17, for a smoke test
#   ./run_all.sh --stage decide     one stage only
#
# Every stage can be re-run: a job whose output already exists is skipped, so an
# interrupted study picks up where it stopped.

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

# The decide scripts screen at 3 seeds by default and take --full for all 17.
FULL=""
[ -z "$QUICK" ] && FULL="--full"

banner () { printf '\n=========== %s ===========\n' "$1"; }
stage  () { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

if stage setup; then
  banner "checks: environment, dataset, training sanity"
  $PY 01_check.py all

  banner "fit the physics constants"
  $PY 02_fit_physics_constants.py
fi

if stage decide; then
  banner "which data split"
  $PY 03_decide_split.py $FULL

  banner "which architecture"
  $PY 04_decide_architecture.py $FULL

  banner "which optimizer"
  $PY 05_decide_optimizer.py $FULL

  banner "how many epochs and seeds"
  $PY 06_decide_budget.py $FULL --patience

  echo
  echo ">>> STOP HERE. Read the decision files in results/, update config.py,"
  echo ">>> then run the measure stage."
fi

if stage measure; then
  banner "confirm the settings still hold"
  $PY 07_confirm_settings.py
  $PY 07_confirm_settings.py --guard || { echo "settings not confirmed; stopping"; exit 1; }

  banner "accuracy against labelled fraction (the main experiment)"
  $PY 08_run_label_sweep.py $QUICK

  banner "physics prior fitted on the same labels"
  $PY 09_run_matched_prior.py $QUICK

  banner "classical baselines"
  $PY 10_run_classical_baselines.py
fi

if stage ablate; then
  banner "ablations: loss terms, design choices, duplicates"
  $PY 11_ablate.py all $QUICK
fi

if stage validate; then
  # The last three checks read one finished run.
  BEST=results/label_sweep/sa/100pct/seed42

  banner "checks on the trained model"
  $PY 12_validate.py all --run_dir "$BEST"
fi

# The search comes last on purpose. Reporting the best of several hundred trials
# as the headline result selects on the quantity being reported; the search
# answers a different question, namely how much accuracy was left on the table.
if stage tune; then
  banner "hyperparameter search"
  $PY 13_search_hyperparameters.py --trials 300
  $PY 13_search_hyperparameters.py --confirm
fi

if stage report; then
  banner "tables and figures"
  $PY 14_report.py all

  echo
  echo "done. tables are in results/tables/ and figures in results/figures/"
fi
