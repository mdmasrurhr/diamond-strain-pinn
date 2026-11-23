# A physics-informed neural network for strain-dependent bandgaps in diamond

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24483464.svg)](https://doi.org/10.5281/zenodo.24483464)

Code and data for the paper *A physics-informed neural network for strain-dependent bandgaps in diamond*
(M. M. H. Rakin, N. R. Martins, L. M. Scolfaro, R. Shadman, A. Haque, A. Dutta, A. Bora and D. Valles;
submitted to *Communications Materials*).

The models predict the bandgap of diamond under an arbitrary strain, given as the six components of the Green–Lagrange
strain tensor. They learn from 1,292 hybrid-functional DFT calculations and use two analytic models of how strain shifts
the band edges: deformation-potential theory for the conduction band and the Bir–Pikus model for the valence band. The
paper's proposed hybrid model (PHM) keeps these analytic models in its output and lets a neural network set their
constants for each strain state. It is compared with a data-only network (MLP), three physics-informed networks
(SA-PINN, RBA-PINN, NTK-PINN), two other hybrid models (the residual model and APHYNITY) and conventional regression,
with 0 to 100% of the training states labeled.

## Contents

| Folder | What it holds |
|---|---|
| `data/` | The reference dataset: 1,292 strain states with their strain tensors and band edges ([`data/README.md`](data/README.md)) |
| `dft/` | VASP input files, the relaxed structure of every state and the absolute energies ([`dft/README.md`](dft/README.md)) |
| `models/` | One standalone script per model of the paper, with its settings in `config.toml` ([`models/README.md`](models/README.md)) |
| `training/` | The program and the exact settings of the paper's 2,924 training runs, the fitted constants of the analytic models, and tools to run them on a workstation or a SLURM cluster ([`training/README.md`](training/README.md)) |
| `analysis/` | Every table, number, statistical test and figure of the paper, computed from `summary/` |
| `summary/` | Condensed results of every training run (about 4 MB), the input of `analysis/` |
| `outputs/` | The tables and figures as produced by `analysis/` |

## Installation

Python 3.12 with the package versions in `requirements.txt` (PyTorch as the CPU build; no GPU is needed):

    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
    pip install -r requirements.txt
    python training/check_env.py

## Reproduce the tables and figures (about 30 seconds)

    python analysis/make_tables.py      # outputs/tables: every table, and numbers_v3.json with every value in the text
    python analysis/make_figures.py     # outputs/figures: Figures 2-5 and Supplementary Figures 1-6

`make_tables.py` also runs every statistical test of the paper (Supplementary Table 7) and checks the 70 claims the
text makes. Figure 1 is drawn with TikZ (`analysis/fig1_workflow.tex`). `analysis/shear_formula_test.py` repeats the
refit of the analytic models with extra shear terms (Supplementary Note 8).

## Train the models

- **One model, step by step:** the standalone scripts in `models/`, for example
  `python models/phm/phm.py study --data_csv data/dft_hy_v9.csv --out study`.
- **The paper's runs:** `training/`, which holds the training program that produced them and the command of every run.

## How the repository was checked

- The commands written by `training/make_tasks.py` give the training program the same settings as the commands
  recorded in the run folders of the paper, and the full runs reproduce the recorded results exactly on the reference
  processor.
- Every standalone script in `models/` reproduces the training program's runs of its model iteration by iteration
  ([`models/README.md`](models/README.md)), and the fitted constants refit exactly.
- From `summary/`, `make_tables.py` reproduces every table, every value quoted in the text and every statistical test of
  the paper byte for byte.

## Notes

- **Calibration.** The 14 constants of the analytic models are fitted separately for every run, on that run's training
  portion only; test and validation states never enter. The label-matched constants use only the labeled states plus
  the training states under isotropic strain, whose gaps every run sees.
- **Preliminary studies.** The network size, optimizer and training length (Supplementary Tables 8 and 9 of the paper)
  were fixed in preliminary studies on a stratified random division. Their results are in `summary/preliminary/`; the
  code that produced them is the earlier version of this repository tagged `thesis-2026-09`.
- **Reproducibility.** Every run uses one CPU thread, which fixes the order of floating-point sums; a run repeated with
  the same seed on the same processor gives the same numbers.

## License and citation

MIT license (`LICENSE`). If you use the code or the data, please cite the paper and this repository (`CITATION.cff`).
Version 1.0.0, the version used in the paper, is archived on Zenodo: https://doi.org/10.5281/zenodo.24483464.
