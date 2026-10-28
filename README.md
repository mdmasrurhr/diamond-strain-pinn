# Physics-informed neural networks for strain-dependent bandgaps in diamond

Code and data for the paper *Physics-informed neural networks for strain-dependent bandgaps in diamond*
(M. M. H. Rakin, N. Rabelo Martins, L. M. Scolfaro, R. Shadman, A. Haque, A. Dutta, A. Bora and D. Valles;
manuscript submitted to *Communications Materials*).

The models predict the bandgap of diamond under an arbitrary strain, given as the six components of the Green–Lagrange
strain tensor. They learn from 1,292 hybrid-functional DFT calculations and use two analytic models of how strain shifts
the band edges: deformation-potential theory for the conduction band and the Bir–Pikus model for the valence band. The
paper compares several ways of combining the two, from a plain neural network (MLP) to physics-informed networks
(SA-PINN, RBA-PINN), Δ-learning and our model, across labeled fractions from 0 to 100% of the training data.

This repository holds everything needed to repeat every training run of the paper and to recompute every number, table
and figure from the results.

## Contents

| Path | What it is |
|---|---|
| `data/dft_hy_v9.csv` | The reference dataset: 1,292 deformation states, strain tensor and band edges (`data/README.md`) |
| `dft/` | VASP input files, the relaxed structure of every state and the absolute energies (`dft/README.md`) |
| `03_train_pinn.py` | The training program for every network of the paper (one run per call) |
| `physics.py`, `calib_v2.py` | The analytic models and their calibration |
| `symsplit.py` | The division of the data that keeps symmetry-equivalent states in the same portion |
| `soap.py`, `metrics_report.py`, `config.py` | SOAP optimizer, evaluation metrics, shared settings |
| `calibrate_constants.py`, `calibrate_heldout.py`, `calibrate_labelmatched.py` | Fit the 13 constants of the analytic models for every run, on that run's training portion only |
| `results/98_constants_v2/`, `results/100_constants_labelmatched/` | The fitted constants used by the paper's runs |
| `make_tasks.py` | Writes the exact command of every training run of the paper (task lists) |
| `run_task.sh`, `submit.sh`, `status.sh`, `setup_env.sh`, `check_env.py` | Run the task lists as SLURM job arrays, one core per run |
| `classical_baselines.py` | Gaussian process, random forest and gradient boosting on the same divisions and labeled states |
| `diag_run.py`, `run_diagnostics.py` | Physical-validity diagnostics of trained runs (deformation potential, shear slope, symmetry drift) |
| `select_weighting.py` | Selection of our model's weighting rule from validation errors |
| `benchmark_inference.py` | Prediction time of a trained model |
| `reference_smoke/`, `compare_smoke.py` | Short reference runs for checking an installation |
| `analysis/` | The paper's analysis: `export_summary.py` → `make_tables.py` → `make_figures.py`, and `fig1_workflow.tex` |
| `summary/` | Condensed results of all runs (about 3 MB), the input of `make_tables.py` and `make_figures.py` |
| `outputs/` | The tables (LaTeX), `numbers.json` and the figures of the paper, as produced by `analysis/` |

## Model names

| In the paper | ID in the code | Trainer options (in addition to the shared ones; see `make_tasks.py`) |
|---|---|---|
| Our model | `hyb_sdp_gap_ld9` | constants of the analytic models as network outputs, agreement of the gap with the analytic models weighted by s = min(1, 9/n), symmetry and curvature terms |
| SA-PINN | `sa_pinn` | `--model sa_paper` (self-adaptive weights as published) |
| RBA-PINN | `rba_pinn` | `--model rba_paper` (residual-based attention as published) |
| Δ-learning | `delta` | `--model mlp --residual_phys` (learned correction to the analytic band edges) |
| MLP | `mlp` | `--model mlp` (data and isotropic terms only) |
| Our model without the agreement term | `sdp_net` | as our model, without the agreement term |
| Weighting variants and other models in the Supplementary Information | `hyb_sdp_ld9`, `hyb_sdp_ld45`, `hyb_ld9`, `hyb_ld45`, `hyb_ld45_col`, `hyb_fixed`, `delta_zs`, `delta_zs_sym` | see `make_tasks.py` |

The trainer also accepts options that the paper does not use (all off by default). Every run of the paper passes its
options explicitly, as listed by `make_tasks.py`.

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

## Reproduce the tables and figures from the included summaries (about 1 minute)

    python analysis/make_tables.py     # outputs/tables/: every table, numbers.json with every value quoted in the text
    python analysis/make_figures.py    # outputs/figures/: Figures 2-5 and Supplementary Figures 1-3

`make_tables.py` also evaluates the claims the text makes (50 checks) and reports how many fail; all pass.

## Repeat the whole study

1. **Constants.** The fitted constants are included. To refit them: `python calibrate_constants.py`,
   `python calibrate_heldout.py <scheme>` for each scheme listed in that script, and `python calibrate_labelmatched.py`.
2. **Training runs.** `python make_tasks.py all` writes four task lists, 3,757 runs in total:

   | List | Runs | What |
   |---|---|---|
   | `tasks/main.tsv` | 2,142 | 14 models × 9 labeled fractions × 17 seeds |
   | `tasks/heldout.tsv` | 1,088 | 8 models × (5 families left out + 3 strain cut-offs) × 17 seeds |
   | `tasks/labelmatched.tsv` | 204 | 3 models with constants fitted on the labeled states, 4 fractions × 17 seeds |
   | `tasks/ablation.tsv` | 323 | our model with one term of the objective removed |

   On a SLURM cluster: `./submit.sh tasks/main.tsv` (and so on; `./status.sh tasks/main.tsv` follows progress). Each line
   can also be run directly: the part after the tab is a shell command, run from this folder. A run takes about 15–50
   minutes on one core, depending on the model.
3. **After training:** `python classical_baselines.py`; `python run_diagnostics.py ...` (the call in its header);
   `python select_weighting.py`; `mkdir -p results/inference_benchmark_gap && python benchmark_inference.py results/B_main/hyb_sdp_gap_ld9/100pct/seed42 > results/inference_benchmark_gap/inference_times.txt`.
4. **Analysis:** `python analysis/export_summary.py` (reads `results/`, writes `summary/`), then step "Reproduce the
   tables and figures" above.

The raw run folders of the paper (about 18 GB) are not included because of their size; `summary/` contains everything
the analysis reads from them.

## How this repository was checked

- The commands written by `make_tasks.py` are identical, character for character, to those of the 3,757 training runs
  behind the paper.
- `03_train_pinn.py`, `physics.py`, `soap.py`, `symsplit.py`, `metrics_report.py` and `calib_v2.py` are the files that
  produced the paper's results, unchanged.
- The smoke runs, started from this repository, reproduce the reference runs exactly on the reference CPU.
- From `summary/`, `make_tables.py` reproduces every table of the paper byte for byte and every value quoted in the
  text. `make_figures.py` reproduces the figures; only the horizontal jitter of the dots that mark individual runs can
  differ.

## Notes

- **Calibration.** The 13 constants of the analytic models are fitted separately for every run, on that run's training
  portion only; test and validation states never enter. The label-matched constants use only the labeled states plus the
  training states under isotropic strain, whose gaps every run sees.
- **Preliminary studies.** The network size, optimizer and training length (Supplementary Tables 9 and 10 of the paper)
  were fixed in preliminary studies made in September 2026 on a stratified random division. Their results are in
  `summary/preliminary/`; the code that produced them is the earlier version of this repository, tagged `thesis-2026-09`.
  That version is otherwise superseded: it calibrated the constants once and reused them for every run, divided the data
  at random, and contained a valence-band fit and a zero-strain slope term that were later corrected.
- **Model name.** "Our model" is the paper's working name; the code calls it `hyb_sdp_gap_ld9`.

## License and citation

MIT license (`LICENSE`). If you use the code or the data, please cite the paper and this repository (`CITATION.cff`).
