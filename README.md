# Physics-informed neural networks for strain-dependent bandgaps in diamond

Code and data for the paper *Physics-informed neural networks for strain-dependent bandgaps in diamond*
(M. M. H. Rakin, N. Rabelo Martins, L. M. Scolfaro, R. Shadman, A. Haque, A. Dutta, A. Bora and D. Valles;
manuscript submitted to *Communications Materials*).

The models predict the bandgap of diamond under an arbitrary strain, given as the six components of the Green–Lagrange
strain tensor. They learn from 1,292 hybrid-functional DFT calculations and use two analytic models of how strain shifts
the band edges: deformation-potential theory for the conduction band and the Bir–Pikus model for the valence band. The
paper compares several ways of combining the two, from a plain neural network (MLP) to physics-informed networks
(SA-PINN, RBA-PINN, NTK-PINN), hybrid models (the residual model, APHYNITY and the proposed hybrid model, PHM) and
conventional regression, across labeled fractions from 0 to 100% of the training data.

**Start with `models/`:** one standalone script per model of the paper, with its settings in a `config.toml`
(`models/README.md`). Each script reproduces the paper's runs of that model exactly. The files in the repository root
are the campaign code that produced the paper's runs, kept as the record.

This repository holds everything needed to repeat every training run of the paper and to recompute every number, table
and figure from the results.

## Contents

| Path | What it is |
|---|---|
| `models/` | **Standalone code for every model of the paper**: PHM, residual model, APHYNITY, SA-PINN, RBA-PINN, NTK-PINN, MLP, conventional regression, analytic models alone (`models/README.md`) |
| `data/dft_hy_v9.csv` | The reference dataset: 1,292 deformation states, strain tensor and band edges (`data/README.md`) |
| `dft/` | VASP input files, the relaxed structure of every state and the absolute energies (`dft/README.md`) |
| `03_train_pinn.py` | The training program for every network of the paper (one run per call) |
| `physics.py`, `calib_v2.py` | The analytic models and their calibration |
| `symsplit.py` | The division of the data that keeps symmetry-equivalent states in the same portion |
| `soap.py`, `metrics_report.py`, `config.py` | SOAP optimizer, evaluation metrics, shared settings |
| `calibrate_constants.py`, `calibrate_heldout.py`, `calibrate_labelmatched.py` | Fit the constants of the analytic models for every run, on that run's training portion only (`--two_band` for the paper's two-band shear term) |
| `results/98_constants_2band/`, `results/100_constants_labelmatched_2band/` | The fitted constants used by the paper's runs (`results/98_constants_v2/`: the one-band shear term, for the comparison in Supplementary Note 8) |
| `make_tasks.py` | Writes the exact command of every training run of the paper (task lists) |
| `run_task.sh`, `submit.sh`, `status.sh`, `setup_env.sh`, `check_env.py` | Run the task lists as SLURM job arrays, one core per run |
| `classical_baselines.py` | Gaussian process, random forest and gradient boosting on the same divisions and labeled states |
| `select_weighting.py`, `select_phm_weight.py` | Choice of the form and weight of the agreement term from validation errors only (made with earlier versions of the model; the weight rule carried over to the PHM) |
| `diag_run.py`, `run_diagnostics.py` | Physical-validity diagnostics of trained runs (deformation potential, shear slope, symmetry drift) |
| `benchmark_inference.py` | Prediction time of a trained model |
| `reference_smoke/`, `compare_smoke.py` | Short reference runs for checking an installation |
| `analysis/shear_tests/` | Tests behind Supplementary Note 8 (`analysis/shear_tests/README.md`) |
| `analysis/` | The paper's analysis: `make_tables.py` (every table, number and statistical test), `make_figures.py` (Figures 2–5, Supplementary Figures 1–6), `paper_data.py` (reads `summary/`), `fig1_workflow.tex` (Figure 1) |
| `summary/` | Condensed results of the paper's 2,924 training runs and other inputs (about 4 MB), the input of `analysis/` |
| `outputs/` | The tables (LaTeX), `numbers_v3.json` (every value quoted in the text) and the figures, as produced by `analysis/` |

## Model names

| In the paper | Standalone code | Campaign key (`make_tasks.py`, results folders) |
|---|---|---|
| PHM (proposed hybrid model) | `models/phm/phm.py` | `phm2` (list `phm2`); its ablations `phm2_drop_*` |
| Residual model | `models/residual/residual.py` | `res_symlab_bal` |
| APHYNITY | `models/aphynity/aphynity.py` | `aphynity` |
| SA-PINN | `models/sa_pinn/sa_pinn.py` | `sa_pinn` (`--model sa_paper`) |
| RBA-PINN | `models/rba_pinn/rba_pinn.py` | `rba_pinn` (`--model rba_paper`) |
| NTK-PINN | `models/ntk_pinn/ntk_pinn.py` | `ntk_pinn` (`--model ntk_paper`) |
| MLP | `models/mlp/mlp.py` | `mlp` |
| Gaussian process, random forest, gradient boosting | `models/conventional/conventional.py` | `classical_baselines.py` |
| Analytic models alone | `models/analytic/analytic.py` | `calibrate_*.py` |

In the campaign code, the key `phm` is an earlier version of the model (13 constant outputs), and `phm_new` is PHM
with a symmetry term and a band-edge level anchor, which the final model dropped; both are kept only as records.

## Installation

Python 3.12 and the package versions in `requirements.txt`. On Linux, `./setup_env.sh` creates a conda environment
`dspinn` with exactly these versions (PyTorch as the CPU build) and runs `check_env.py`. Or, in any environment:

    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
    pip install -r requirements.txt
    python check_env.py

All runs use the CPU with one thread; a GPU is not needed.

## Check an installation (about 10 minutes on 8 cores)

    python make_tasks.py smoke                     # 16 short runs, one per command shape of the paper
    export OMP_NUM_THREADS=1
    cut -f2- tasks/smoke.tsv | xargs -d '\n' -P 8 -I{} bash -c '{}'
    python compare_smoke.py

`compare_smoke.py` compares each run with `reference_smoke/`. On the CPU that made the references (Intel Xeon W-2255)
the predictions agree exactly; on other CPUs, rounding differs and a run passes if its test and validation errors agree
within max(2 meV, 5%).

## Reproduce the tables and figures from the included summaries (about 30 seconds)

    python analysis/make_tables.py     # outputs/tables/: every table and numbers_v3.json with every value quoted in the text
    python analysis/make_figures.py    # outputs/figures/: Figures 2-5 and Supplementary Figures 1-6

`make_tables.py` also evaluates the 67 claims the text makes (all pass) and runs every statistical test of the paper
(Supplementary Table 7).

## Repeat the whole study

1. **Constants.** The fitted constants are included. To refit them: `python models/analytic/analytic.py constants
   --data_csv data/dft_hy_v9.csv --out <folder>` (with `--experiment heldout` and `--experiment labelmatched` for the
   other two sets), or the campaign scripts `calibrate_constants.py --two_band`, `calibrate_heldout.py <scheme>
   --two_band` and `calibrate_labelmatched.py --two_band`. Both give the included files exactly.
2. **Training runs.** Either the standalone scripts in `models/` (`python models/<model>/<model>.py study ...`; see
   `models/README.md`), or the campaign trainer: `python make_tasks.py paper` writes `tasks/paper.tsv`, the 2,924
   training runs of the paper:

   | Runs | What |
   |---|---|
   | 1,071 | 7 models (PHM, residual model, SA-PINN, RBA-PINN, NTK-PINN, APHYNITY, MLP) × 9 labeled fractions × 17 seeds |
   | 765 | 5 changes to PHM (three parts removed, two terms added) × 9 fractions × 17 seeds |
   | 952 | the 7 models × (5 families left out + 3 strain cut-offs) × 17 seeds |
   | 136 | PHM and the residual model with label-matched constants, 4 fractions × 17 seeds |

   On a SLURM cluster: `./submit.sh tasks/paper.tsv` (`./status.sh tasks/paper.tsv` follows progress). Each line can
   also be run directly: the part after the tab is a shell command, run from this folder. A run takes about 15–50
   minutes on one core, depending on the model.
3. **After training:** `python classical_baselines.py` (conventional regression); `python run_diagnostics.py ...` (the
   call in its header: physical-validity diagnostics); `python benchmark_inference.py results/B_main/phm2/100pct/seed42`
   (prediction time).
4. **Analysis:** `summary/` was condensed from the raw run folders with `paper_data.export_summary` (in
   `analysis/paper_data.py`); then step "Reproduce the tables and figures" above. The raw run folders (about 18 GB) are
   not included; `summary/` contains everything the analysis reads from them.

## How this repository was checked

- The 2,924 commands of `make_tasks.py paper` are identical, character for character, to those recorded in the run
  folders of the paper.
- Every standalone script in `models/` reproduces the campaign trainer's runs of its model iteration by iteration
  (`models/README.md`).
- From `summary/`, `make_tables.py` reproduces every table of the paper byte for byte, every value quoted in the text
  and every statistical test (the 26 output files are identical to the manuscript build). `make_figures.py` reproduces
  the figures; the learning curves are drawn from every 10th iteration, and the horizontal jitter of the dots that
  mark individual runs can differ.
- The smoke runs, started from this repository, reproduce the reference runs exactly on the reference CPU.

## Notes

- **Calibration.** The 14 constants of the analytic models are fitted separately for every run, on that run's training
  portion only; test and validation states never enter. The label-matched constants use only the labeled states plus the
  training states under isotropic strain, whose gaps every run sees.
- **Preliminary studies.** The network size, optimizer and training length (Supplementary Tables 8 and 9 of the paper)
  were fixed in preliminary studies made in September 2026 on a stratified random division. Their results are in
  `summary/preliminary/`; the code that produced them is the earlier version of this repository, tagged `thesis-2026-09`.
  That version is otherwise superseded: it calibrated the constants once and reused them for every run, divided the data
  at random, and contained a valence-band fit and a zero-strain slope term that were later corrected.

## License and citation

MIT license (`LICENSE`). If you use the code or the data, please cite the paper and this repository (`CITATION.cff`).
