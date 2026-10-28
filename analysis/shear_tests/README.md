# Tests behind Supplementary Note 8 (why shear is the hardest family)

These scripts need the raw run folders in `results/` (see the main README). Run them from the repository root, in this order.

| Script | What it does |
|---|---|
| `shear_difficulty.py`, `shear_difficulty_2.py` | Per-state analysis of the shear error: sheared planes, valley changes, sampling density. Its summary is part of `analysis/make_tables.py`. |
| `beta_test.py` | Re-evaluates the trained runs with a sharper minimum over the conduction valleys (no retraining). |
| `formula_test.py` | Refits the analytic models with four extra second-order shear constants. |
| `03_train_pinn_valley.py`, `variant_tasks.py`, `evaluate_variants.py` | Four variants of our model aimed at the valley crossings (valley-resolved offsets, valley supervision from the k-point of the minimum, extra weight on the region, and the first two together), their task lines and their evaluation. |

None of the variants improves on the final model; the trainer is kept apart from `03_train_pinn.py` so that the reported runs stay unchanged.
