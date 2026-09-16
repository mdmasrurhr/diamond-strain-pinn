# Predicting the bandgap of strained diamond with a physics-informed neural network

This code predicts the electronic bandgap of diamond under an arbitrary
mechanical strain. The strain is described by six numbers (the Green-Lagrange
strain tensor), and the model returns the bandgap in electron-volts.

The hard part is that the reference data is expensive. Each training label comes
from an HSE06 hybrid-functional density-functional-theory calculation costing
hours of compute, so there are only about a thousand of them. A plain neural
network needs far more data than that to learn a six-dimensional function.

The approach here is to put the physics into the training objective. Two
textbook models describe how strain moves the band edges: deformation-potential
theory for the conduction band and the Bir-Pikus Hamiltonian for the valence
band. Neither is accurate enough on its own, but both are approximately right
everywhere. The network is trained to fit the expensive labels **and** to agree
with the analytic physics, so the physics supplies structure wherever labels are
missing. The result stays usable when labelled data is scarce.

---

## What you need

Python 3.10 or newer, and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended. A single training run takes a few minutes on a
GPU; the full study is thousands of runs. Everything also runs on CPU if you
only want to try one model.

## The quickest thing that shows it works

```bash
python train_pinn.py --model sa --seed 42 --dft_pct 100 --results_dir out/demo
```

Every option defaults to the value recorded in `config.py`, so this uses the
configuration the study adopted. Passing a flag overrides it, which is how the
ablation scripts vary one setting at a time.

This trains one model on the full dataset and writes three files to `out/demo/`:
per-epoch history, per-sample test predictions, and a one-line summary of the
metrics. `--dft_pct` is the percentage of training labels the model is allowed
to use, so `--dft_pct 5` is the scarce-label case the physics is there for.

## How the repository is organised

Scripts are numbered in the order they run, and the name says what each does.
They group into seven stages; each stage settles something the next assumes.

| stage | scripts | purpose |
|---|---|---|
| **setup** | `01_check_environment.py`, `02_check_data.py`, `03_fit_physics_constants.py` | record the environment, describe the data, fit the physics constants |
| **decide** | `04_check_sanity.py`, `05_decide_split.py`, `06_decide_architecture.py`, `07_decide_optimizer.py`, `08_decide_budget.py` | settle every training setting, before any large run |
| **measure** | `09_confirm_settings.py`, `10_run_label_sweep.py`, `11_run_matched_prior.py`, `12_run_classical_baselines.py` | the main experiment: accuracy against how many labels were used |
| **ablate** | `13_ablate_loss_terms.py`, `14_ablate_design_choices.py`, `15_ablate_duplicates.py` | what each part of the model contributes |
| **validate** | `16_validate_physics.py`, `17_validate_generalization.py`, `18_validate_robustness.py`, `19_validate_explainability.py`, `20_validate_performance.py` | what the trained model can and cannot do |
| **tune** | `21_search_hyperparameters.py` | a large hyperparameter search, run after the main experiment |
| **report** | `22_make_tables.py`, `23_make_figures.py` | tables and figures |

Two scripts are not stages. `train_pinn.py` trains one physics-informed model
and `train_baseline.py` trains one comparison model; the stage scripts call them
many times with different settings.

The rest are shared building blocks:

| file | what it holds |
|---|---|
| `config.py` | every setting, in one place, with a note on how each was chosen |
| `physics.py` | the analytic band-edge equations and their 13 constants |
| `jobs.py` | runs many training jobs across the available GPUs |
| `decide.py` | reads finished runs back and applies the selection rules |
| `style.py` | one consistent look for every figure |
| `soap.py` | a third-party optimizer, vendored unchanged (see its header) |

## Running the whole study

```bash
./run_all.sh                  # the full study
./run_all.sh --quick          # 3 seeds instead of 17, for a smoke test
./run_all.sh --stage decide   # one stage only
```

Every stage can be re-run safely: a job whose output already exists is skipped,
so an interrupted study picks up where it stopped.

**The decide stage is the part it is tempting to skip.** Each of its scripts
writes a decision file into `results/`. Read them, copy the chosen values into
`config.py`, and only then run the measure stage. Running the main experiment
first means measuring a configuration that the decide stage then tells you was
the wrong one.

The tune stage comes last for a related reason: searching for the best hyperparameters
and then reporting that best number as the headline result is selecting on the
quantity being reported. The search answers a different question, namely how
much accuracy was left on the table.

## The three models

| name | what it is |
|---|---|
| **SA-PINN** | physics-informed; the weight of each training sample is *learned* during training |
| **RBA-PINN** | physics-informed; sample weights follow a fixed rule based on the physics residual |
| **Baseline ANN** | a purely data-driven network reproduced from the literature (`train_baseline.py`) |

A fourth configuration, the same network with every physics term switched off,
is used as a control. It isolates what the physics contributes, because it
differs from the physics-informed models in nothing else.

SA-PINN and RBA-PINN differ in exactly one respect: how each sample is weighted.
Architecture, optimizer, schedule, data and random seeds are identical, so any
difference between them is attributable to the weighting rule alone.

## The data

`data/dft_hy_v9.csv` holds 1,292 strained states. Each row has six strain
components, the bandgap, and the two band-edge energies it is the difference of.

```
E_xx E_yy E_zz E_xy E_yz E_zx    strain (dimensionless)
Eg_eV                            bandgap
CBM_eV VBM_eV                    conduction and valence band edges
strain_type                      which family of deformation this is
```

## What every run writes

Three files with fixed column names, so any script can read any run:

- `epoch_log.csv` - one row per epoch
- `test_results.csv` - one row per test sample
- `metrics_summary.csv` - one row summarising the run

Two columns in `epoch_log.csv` are easy to confuse:

| column | meaning |
|---|---|
| `val_mae_eV` | the validation error as last measured |
| `val_best_eV` | the best validation error so far, which is the model that gets kept |

`metrics_summary.csv` also records `best_ckpt_epoch` and `epochs_run`. When
those two are equal, the model was still improving when training stopped, which
means the reported accuracy was limited by the epoch budget rather than by the
model.

## Reproducibility

`01_check_environment.py` records the interpreter, the package versions, the GPUs, a
checksum of the dataset and the random-seed policy. Run it before and after a
long study: if the two records differ, the results straddle a change in the
environment and are not a single experiment.

Random seeds control the data split, the network initialisation and the choice
of which labels are visible. The thread count is pinned in `config.py`, because
the number of threads changes the order in which floating-point sums are
accumulated and that is enough to move the baseline by a few tenths of a
percent.

## Citation

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The vendored optimizer in `soap.py` is third-party code;
its origin and authors are named in its file header.
