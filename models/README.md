# Standalone code for every model

One folder per model. Each script is self-contained: it reads the dataset, divides it, fits the constants of the
analytic models where the model uses them, trains and writes the results. Nothing is imported from another folder.
Every setting is in the `config.toml` next to the script (another settings file can be given with
`<MODEL>_CONFIG=<path>`, for example `PHM_CONFIG=other.toml`).

| Folder | Model in the paper | What it is |
|---|---|---|
| `phm/` | PHM (proposed hybrid model) | analytic models whose 13 constants the network sets for each strain state (tanh-bounded), plus a correction to each band edge; copied symmetry labels; gap agreement with weight 2s |
| `residual/` | residual model | analytic models with fixed constants plus a learned correction to each band edge; copied symmetry labels; gap agreement and a symmetry term |
| `aphynity/` | APHYNITY | analytic models with trainable constants plus the smallest correction that fits the data |
| `sa_pinn/` | SA-PINN | network trained against the data and the analytic models, self-adaptive weights |
| `rba_pinn/` | RBA-PINN | the same terms, residual-based attention |
| `ntk_pinn/` | NTK-PINN | the same terms, neural-tangent-kernel weights |
| `mlp/` | MLP | the same network, data only |
| `conventional/` | Gaussian process, random forest, gradient boosting | the gap directly from the labeled states |
| `analytic/` | analytic models alone | fit of the 13 constants and of the band separation; their test errors |

Each script has the same layout: libraries, a list of its functions, the settings read from `config.toml`, the
functions in that order, and the command line.

## Commands

The network models share the same commands (`python <model>.py <command> -h` lists the options):

    python phm.py constants --data_csv ../../data/dft_hy_v9.csv --out study      # constants for seeds 42-58
    python phm.py run       --data_csv ../../data/dft_hy_v9.csv --out study --seed 42 --fraction 5
    python phm.py study     --data_csv ../../data/dft_hy_v9.csv --out study      # every run, on this computer
    python phm.py tasks     --data_csv ../../data/dft_hy_v9.csv --out study      # one line per run, for a cluster

`--experiment` selects the experiment; give the same value to `constants` first.

| Experiment | What | Models |
|---|---|---|
| `main` (default) | symmetry-grouped division, labeled fractions 0, 1, 2, 5, 10, 25, 50, 75, 100% | all |
| `heldout` | 100% labeled; one strain family left out (`group_<family>`), or every state beyond a strain magnitude (`magnitude_03`, `_05`, `_07`); constants refitted on the reduced training portion | network models, analytic |
| `labelmatched` | 1, 2, 5, 10% labeled; constants fitted only on the labeled and isotropic training states | PHM, residual model, analytic |

The MLP has no `constants` command. `conventional.py` has `run` (one seed) and `study`; `analytic.py` has
`constants`, `evaluate` and `study`. Finished runs are skipped.

Outputs: `<out>/constants/` (one JSON file per seed and setting) and one folder per run,
`<out>/runs/<fraction>pct/seed<seed>/` (main), `<out>/runs/heldout/<test>/seed<seed>/` or
`<out>/runs/labelmatched/<fraction>pct/seed<seed>/`. A run folder holds `log.csv` (every 10 iterations),
`predictions.csv` (every state of the dataset), `best_model.pt`, `config_used.toml` and `metrics_summary.csv`, which is
written last and marks the run as finished.

## Agreement with the paper's runs

Each script was run next to `training/train.py` on one thread, with the same constants and shortened runs (300 to
1,100 iterations), and the train and test gap errors were compared at every logged iteration:

| Model | Main (0, 2, 100%) | Held-out | Label-matched | Largest difference |
|---|---|---|---|---|
| PHM | identical | group_shear, magnitude_03 identical | 5% identical | 5e-7 eV (rounding of the log) |
| residual model | identical | group_shear, group_isotropic, magnitude_05 identical | 2% identical | 5e-7 eV |
| MLP | identical | group_shear, magnitude_05 identical | – | 5e-7 eV |
| SA-PINN | identical | magnitude_05 identical | – | 5e-7 eV |
| RBA-PINN | identical | magnitude_05 identical | – | 5e-7 eV |
| NTK-PINN | identical (also 5% over 1,100 iterations, two weight updates) | group_shear identical | – | 5e-7 eV |
| APHYNITY | identical | group_shear, magnitude_07 identical | – | 5e-7 eV |

The constants fitted by `analytic.py` are bit-identical to those in `training/constants/`, and `conventional.py`
reproduces the conventional regression results of the paper.

Each run uses one processor thread, which fixes the order of floating-point sums: a run repeated with the same seed
gives the same numbers. With more threads the results differ in the last digits.

## Requirements

Python 3.11 or later (for `tomllib`), NumPy, pandas, SciPy, scikit-learn and PyTorch (the CPU build is enough); the
versions used for the paper are in `../requirements.txt`. The SOAP optimizer is included in every network script.
