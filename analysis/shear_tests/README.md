# Tests behind Supplementary Note 8 (decomposition of the shear error)

These scripts need the raw run folders in `results/` (see the main README). Run them from the repository root, in this order.

| Script | What it does |
|---|---|
| `shear_difficulty.py`, `shear_difficulty_2.py` | Per-state analysis of the shear error: sheared planes, valley changes, sampling density. Its summary is part of `analysis/make_tables.py`. |
| `beta_test.py` | Earlier test, not reported in the paper: re-evaluates trained runs with a sharper minimum over the conduction valleys (no retraining). |
| `formula_test.py` | Refits the analytic models with three extra second-order shear constants (Supplementary Note 8). |
| `03_train_pinn_valley.py`, `variant_tasks.py`, `evaluate_variants.py` | Earlier tests, not reported in the paper, made with a previous version of the model: four variants aimed at the valley crossings (valley-resolved offsets, valley supervision from the k-point of the minimum, extra weight on the region, and the first two together), their task lines and their evaluation. |

None of these variants improved on the model they were built on; the trainer is kept apart from `03_train_pinn.py` so that the reported runs stay unchanged.
