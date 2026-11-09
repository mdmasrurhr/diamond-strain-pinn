# PHM

**Naming (from 8 Nov 2026):** **PHM** is this model (it was called r15_phm before). **PHM old** is the earlier model with 13 constant outputs (structured head). In the old pipeline (`03_train_pinn.py`, `make_tasks.py`, `results/B_main/phm`, CHANGE_LOG entries before 8 Nov, the manuscript draft), the name "phm" means PHM old. Those identifiers were kept so existing results stay traceable.

PHM predicts the band edges of strained diamond from the six strain components. A neural network adjusts the 13 constants of the analytic band-edge models for each strain state, and adds a small correction to each band edge. The outputs are CBM and VBM, and the gap is CBM − VBM.

## Files
| file | job |
|---|---|
| `config.toml` | **the settings**: study seeds and fractions, network size, training, loss weights, θ₀ fit and figure settings, each with its meaning. Read by the two scripts below. |
| `phm.py` | **the model**: physics, θ₀ fit, data split, SOAP, network, training. It writes the raw results of every run and contains no plotting code. |
| `phm_display_items.py` | **the display items** (journal term for tables + figures): results tables and 12 figures, read from the run folders |
| `phm_ablations.py` | **the record of the ablation screen**: every variant tested and the comparison against base. Kept as it was run. |

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, torch and matplotlib (matplotlib only for the display items). The SOAP optimizer is inside `phm.py`.

Each script has the same layout: libraries, a list of its functions, its settings (from `config.toml`) and fixed definitions, then the functions, then the command line. Fixed definitions the formulas depend on stay in `phm.py`: the 13 constant names, the strain columns and the 2 corrections.

The settings file is TOML, not YAML. Python reads TOML without any extra package, both here and on LEAP2. TOML also reads numbers like `1e-5` as numbers, while the common YAML reader turns them into text. To use another settings file: `PHM_CONFIG=other.toml python phm.py ...`. Every run saves a copy of its settings as `config_used.toml` in its run folder.

## The final model
It is the first version (res_const_symlab_bal) plus two changes adopted by the ablation screen:
- the constants are bounded with tanh (each moves at most 25%)
- a 200-iteration learning-rate warm-up after the switch to SOAP

The loss has three terms: data (with copied symmetry labels), isotropic gaps and gap agreement with θ₀. Two more terms, a symmetry loss and a band-edge level anchor, were removed on 8 Nov 2026 after the 17-seed ablations in the trainer (key `phm_new`). Without the symmetry loss the test gap error fell by 1.9 meV at 50% and 100% labels. The anchor changed nothing. The trainer key of this model is `phm2`.

The evidence is in `10_independent_check/phm_screen/comparison/decisions.txt` and in `04_plans/CHANGE_LOG.md`.

## Commands
```
python phm.py constants --data_csv ../../data/dft_hy_v9.csv --out study        # fit θ₀ for seeds 42-58
python phm.py study     --data_csv ../../data/dft_hy_v9.csv --out study        # every run on this computer
python phm.py tasks     --data_csv ../../data/dft_hy_v9.csv --out study        # cluster task list: study/phm_main.tsv
python phm.py run       --data_csv ../../data/dft_hy_v9.csv --out study --seed 42 --fraction 1
python phm_display_items.py --data_csv ../../data/dft_hy_v9.csv --out study    # tables and figures
```
The defaults are 17 seeds (42–58), labeled fractions 0, 1, 2, 5, 10, 25, 50, 75 and 100%, and 24,000 iterations. Finished runs are skipped.

`--experiment heldout` (one strain class left out, or every state beyond a strain magnitude; 100% labeled) and
`--experiment labelmatched` (constants fitted only on the labeled and isotropic training states; 1, 2, 5, 10%) run the
paper's other two experiments; give the same `--experiment` to `constants` first. The display items cover the main
experiment. See `../README.md` for the layout shared by all models.

## Outputs
**`phm.py`**, per run, in `study/runs/<fraction>pct/seed<seed>/`:
- `metrics_summary.csv`: written last, so its presence marks the run as finished
- `log.csv`, `predictions.csv`, `best_model.pt`, `config_used.toml`

**`phm_display_items.py`**, in `study/`:
- tables: `results.txt`, `summary_by_fraction.csv`, `all_runs.csv`
- figures: `figures/fig01` … `fig12`, as PNG (300 dpi) and PDF, with `captions.txt`
  - training dynamics: learning curves, loss terms, learning rate and gradient norms
  - accuracy: parity, residuals, error ECDF, seed spread
  - per strain class, data efficiency, training / validation / test error
  - physics: strain sweeps, constants vs correction

## Checks
- The ablation script's `base` variant reproduces the paper's trainer (03_train_pinn.py) exactly.
- Before 8 Nov, `phm.py` matched the ablation script's `final_midpoint` variant exactly. The current version (no symmetry loss, no anchor) matches the trainer's `phm2` runs iteration by iteration (to the log's rounding, 5e-7 eV) in all three experiments: main at 0, 2 and 100%, held-out group_shear and magnitude_03, label-matched 5%.
- The θ₀ fit gives exactly the paper's constant files for all 17 seeds.
- Splitting out the display items changed no number: the tables are byte-identical and a training run is unchanged.
- Moving the settings to `config.toml` changed no value or type (all 80 settings compared), and no result (training run identical, tables byte-identical). The code reads them by section and key, for example `TRAINING["adamw_lr"]`, so the names in the code match the names in `config.toml`.

## Open point
θ₀ is fitted on all training states. At 1% labels, fitting it only on what a 1% run may see raises the test gap error from 25.7 to 57.0 meV (seeds 42–46). So low-label results depend on this θ₀, and the paper must state it or use a fairer θ₀.
