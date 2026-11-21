# Training runs of the paper

`train.py` trains one network and writes its results; every training run of the paper is one call of it. Run the
scripts from this folder.

| File | What it does |
|---|---|
| `train.py` | Trains one network (`python train.py -h` lists the options) |
| `make_tasks.py` | Writes the command of every run: `paper` (2,924 runs) or `smoke` (21 short runs) |
| `constants/` | Fitted constants of the analytic models, one file per run (`two_band/`: the paper's; `labelmatched/`; `one_band/`: the linear shear term of Supplementary Note 8) |
| `calibrate.py`, `calibration.py` | Fit those constants on each run's training data |
| `physics.py` | The analytic models of the band edges |
| `soap.py` | The SOAP optimizer (Vyas et al. 2025) |
| `config.py` | Paths, seeds, labeled fractions and the fixed network and optimizer settings |
| `classical_baselines.py` | Gaussian process, random forest and gradient boosting on the same divisions |
| `diag_run.py`, `run_diagnostics.py` | Physical-consistency diagnostics of trained runs (deformation potential, shear slope, symmetry drift) |
| `benchmark_inference.py` | Prediction time of a trained network |
| `metrics_report.py` | Metrics of finished runs from their `test_results.csv` |
| `check_env.py` | Checks the packages, the dataset and the task lists |
| `compare_smoke.py`, `reference_smoke/` | Installation check against reference runs |
| `run_task.sh`, `submit.sh`, `status.sh`, `setup_env.sh` | Running the task lists as SLURM job arrays, one core per run |

## Runs

    python make_tasks.py paper          # tasks/paper.tsv: one line per run, "<results folder>\t<command>"

| Runs | Folder | What |
|---|---|---|
| 1,071 | `results/B_main` | 7 models × 9 labeled fractions × 17 seeds |
| 765 | `results/B_ablation` | 5 changes to the PHM × 9 fractions × 17 seeds |
| 952 | `results/B_extrap` | the 7 models × 8 held-out tests (5 families left out, 3 strain cut-offs) × 17 seeds |
| 136 | `results/B_labelmatched` | PHM and the residual model with label-matched constants × 4 fractions × 17 seeds |

Results folders of the models: `phm2` (PHM), `res_symlab_bal` (residual model), `aphynity`, `sa_pinn`, `rba_pinn`,
`ntk_pinn`, `mlp`; ablations `phm2_drop_agreement`, `phm2_drop_iso_eg`, `phm2_drop_copies` (one part removed),
`phm_new_drop_anchor` (symmetry term added) and `phm_new_drop_sym` (band-edge level anchor added).

A run takes 5 to 45 minutes on one core. On a workstation, run the lines in parallel, one thread each:

    export OMP_NUM_THREADS=1
    cut -f2- tasks/paper.tsv | xargs -d '\n' -P 8 -I{} bash -c '{}'

On a SLURM cluster: `./setup_env.sh` once (conda environment with the paper's package versions), then
`./submit.sh tasks/paper.tsv` and `./status.sh tasks/paper.tsv`. Finished runs are skipped when resubmitted.

## Installation check (about 2 minutes on 8 cores)

    python make_tasks.py smoke
    export OMP_NUM_THREADS=1
    cut -f2- tasks/smoke.tsv | xargs -d '\n' -P 8 -I{} bash -c '{}'
    python compare_smoke.py

One 300-iteration run per model and experiment, compared with `reference_smoke/`. On the processor that made the
references (Intel Xeon W-2255) the predictions agree exactly; on other processors a run passes if its test and
validation errors agree within max(2 meV, 5%).

## Constants

The constants in `constants/` are included. To refit them:

    python calibrate.py main                # constants/two_band/symmetry
    python calibrate.py heldout             # constants/two_band/<held-out test>
    python calibrate.py labelmatched        # constants/labelmatched/<fraction>pct
    python calibrate.py main --one_band     # constants/one_band/symmetry

## After training

    python classical_baselines.py                                                          # results/classical
    python run_diagnostics.py results/diagnostics.jsonl "results/B_main/*/*pct/seed*" "results/B_ablation/*/*pct/seed*"
    python benchmark_inference.py results/B_main/phm2/100pct/seed42
    python ../analysis/shear_formula_test.py                                               # results/shear_tests

`../analysis/export_summary.py` condenses the results into `../summary/`, from which `../analysis/` computes the
tables and figures.
