# Model checklist

Every element of the model, what settles it, and where the evidence lives. A row
with no script is one that cannot yet be defended, and it says so.

This file is a record of what has and has not been validated. It is included so
that a reader can see the limits of the study without having to find them.

**Status** means: **decided** -- measured, and the measurement chose this;
**pending** -- a script exists and has not run; **gap** -- nothing covers it yet.

---

## 1. Feasibility and scientific formulation

| element | status | where | what it says |
|---|---|---|---|
| Feasibility: is the surrogate worth building | **decided** | `validate_performance.py` | One prediction takes 0.2 ms against ~8 core-hours of HSE06. Training costs less than one DFT calculation, so the model repays itself after one prediction and the dataset after 939. |
| Baseline to beat | **decided** | `run_classical_baselines.py`, `train_baseline.py` | Random forest, GP and kernel ridge alongside the literature ANN. The complexity tax is paid only if the PINN beats the simplest thing that works. |
| Hardware dependency | **decided** | `check_environment.py` | 2x Quadro RTX 6000. A run uses ~20 MB, so the GPU is a convenience, not a requirement -- the CPU forward pass is 0.09 ms. |
| Naive benchmark | **gap** | -- | A constant predictor and a linear fit are implied by the classical baselines but not reported as their own row. One line to add in `22`. |

## 2. Data integrity and provenance

| element | status | where | what it says |
|---|---|---|---|
| Provenance and bias | **decided** | dataset audit (see the paper's supplementary material) | 2,680 VASP runs, all converged. Sampling is by design, not by nature, so the "bias" is a stated design: 47 isotropic, 120 uniaxial, 120 biaxial, 512 triaxial, 540 shear. |
| Feature choice | **decided** | `check_data.py`, analysis 01 Q9 | The six strain components are complete; everything else admissible is an algebraic function of them. A naive selection prefers stresses and energies, which are *outputs of the calculation the model replaces* and therefore inadmissible. |
| Dimensionality and scaling | **decided** | `train_pinn.py --scaler`, screened | Z-score. MinMax and Robust screened; standard is best and the screen is categorical, not a plateau. |
| Splitting method | **pending** | `decide_split.py` | Six schemes, and the point that a split decides *what question the number answers*: stratified measures interpolation, group and magnitude measure extrapolation. |
| Leakage proof | **decided** | `check_sanity.py` | Scaler fitted on the training fold only (proved by its mean differing from the full-data mean); zero identical input rows across folds; zero duplicate strain states in v9. |
| Augmentation | **n/a, argued** | -- | None used. The only physically valid augmentation is cubic symmetry, and applying it would manufacture rows the DFT campaign did not compute -- `validate_robustness.py` uses those transformations as a *test* instead, which is the stronger use. |

## 3. Architectural topology

| element | status | where | what it says |
|---|---|---|---|
| Depth x width | **pending** | `decide_architecture.py` | 24-point grid with the parsimony rule applied explicitly. The thesis stated that rule and did not follow it: its own sweep selects 7x64 over the adopted 5x100. |
| Parameter count | **decided** | logged per run as `n_params` | 41,302 for 5x100 against 938 training samples -- 44 parameters per sample, which is why regularisation is worth testing rather than assuming. |
| Activation | **decided** | `ablate_design_choices.py` | SiLU. GELU within noise (+0.23 meV), tanh +4.4, ReLU +12.6. |
| Weight initialisation | **decided** | `ablate_design_choices.py` | Xavier. Kaiming within noise (+0.04), orthogonal +1.4, default +1.9. |
| Output bias initialisation | **decided** | `ablate_design_choices.py` | Training-fold means. Zero costs +3.8 meV. |
| Normalisation | **pending** | `decide_architecture.py` | LayerNorm now testable. **BatchNorm is excluded on grounds**: the anchor boundary condition evaluates the network at the single unstrained state, and BatchNorm has no within-batch variance in a batch of one. `train_pinn.py` refuses it with that message. |
| Dropout | **pending** | `decide_architecture.py` | Three rates. Never previously tried despite 6,000 epochs on 938 samples. |
| Residual connections | **pending** | `decide_architecture.py` | With and without LayerNorm. |
| Uniform layer width | **pending** | `decide_architecture.py` | Taper and widen at matched depth. |

## 4. Optimization dynamics

| element | status | where | what it says |
|---|---|---|---|
| Loss formulation | **decided** | `ablate_loss_terms.py` | Leave-one-out at 17 seeds. Two terms **hurt** at full data: gradient BCs cost 4.2 meV of 16.4, band-edge residuals 0.8. That is the physics-vs-accuracy trade, and the paper must name its price. |
| Loss weights | **pending** | `12`, and see `config.py` | Four sit on **cliffs**, not plateaus. `w_cons=2.0` gives 12.45 meV and 4.0 gives 117.9 -- one step from a tenfold failure. |
| Optimizer family | **decided** | `ablate_design_choices.py` | AdamW->SOAP. Adam +20, RMSprop +36, SGD +157, L-BFGS +237 meV. |
| Optimizer **attribution** | **pending** | `decide_optimizer.py --group arms` | The SOAP-only arm, absent until now. Dropping SOAP costs +31 meV, the largest single effect in the ablation table, and without this arm it cannot be split between SOAP and the two-phase schedule. |
| Switch point, clip norm, SA clamp | **pending** | `decide_optimizer.py` | None ever screened alone. |
| Gradient flow | **decided** | `check_sanity.py` | Per-layer gradient norms span 1.8x at initialisation. Healthy; no vanishing path. |
| Batch size | **n/a, argued** | -- | Full batch. 938 samples fit in 20 MB, and the physics residuals are evaluated on masked subsets of the same forward pass, so mini-batching would change what the loss means, not just how it is computed. |
| Epochs | **pending** | `decide_budget.py` | On **every** run of the reproduction campaign the kept checkpoint was the last in the budget. The grid lands on learning-rate floors: 1500 / 4000 / 6000 / 12000 / 24000. |
| Early stopping | **pending** | `decide_budget.py --patience` | None existed. With best-checkpoint selection it is a compute decision, not an accuracy one -- worth adopting only if it saves epochs at no cost in meV. |

## 5. Generalizability and performance bounds

| element | status | where | what it says |
|---|---|---|---|
| Memorisation vs learning | **pending** | `validate_generalization.py` | Train-test gap per label fraction. |
| Label-shuffle control | **decided** | `check_sanity.py --all` | Trained on shuffled labels the model scores 1,191 meV against 649 for a constant predictor -- worse than guessing, which is the correct direction. Nothing leaks. |
| Can it fit at all | **decided** | `check_sanity.py` | Memorises 8 samples to 0.0000 meV. The architecture and gradient path are sound. |
| Out-of-distribution | **pending** | `decide_split.py`, `validate_generalization.py` | Unseen deformation family and strain beyond the trained range. Expected to be much worse -- that is the bound on the claim. |
| Regularisation | **pending** | `decide_architecture.py` | Only weight decay at present. |
| Evaluation metric | **decided** | output contract | MAE in meV, chosen because the error budget is compared against a physical scale (HSE06's own 58 meV scatter), not a relative one. RMSE, R^2, P95 and max error are logged alongside so the choice can be re-examined. |
| Error and residual analysis | **decided** | per-family columns, `validate_generalization.py` | Error reported per deformation family in every run. Shear is consistently worst. |

## 6. Sensitivity, robustness and explainability

| element | status | where | what it says |
|---|---|---|---|
| One-at-a-time sensitivity | **decided** | thesis OAT screens | Ten settings; four cliffs, two slopes, three flat, one categorical. |
| Global sensitivity | **partial** | Sobol study | `w_data` dominates (S1 0.687). But it ran at 1,500 epochs, not 6,000, and several indices are negative -- under-sampled. Qualitative only. |
| Robustness to input noise | **pending** | `validate_robustness.py` | Error against perturbation size, so the required input precision can be read off. |
| Robustness to symmetry | **pending** | `validate_robustness.py` | Permuting the three normal components must not change the gap in a cubic crystal, and nothing told the model so. Stronger than noise: it has a known right answer no label supplied. |
| Explainability | **pending** | `validate_explainability.py` | Not just which input matters -- the model's own derivatives compared against values deformation-potential theory fixes in advance: dEg/dI1, zero shear derivative at zero shear, equal normal derivatives at isotropic strain. |

## 7. Reproducibility and governance

| element | status | where | what it says |
|---|---|---|---|
| Environment manifest | **decided** | `check_environment.py` | Interpreter, five package versions, GPUs, dataset SHA-256, seed policy, thread policy. Comparable before and after a campaign. |
| Seed control | **decided** | `check_environment.py`, `config.py` | 17 seeds; each controls the split, the initialisation and the labelled-subset draw, the last through a separate offset so the subset does not move when the split does. |
| Seed count justified | **decided** | `decide_budget.py --seeds_only` | N=17 resolves ~1.8 meV for the PINNs (sigma=1.9), which matches the differences claimed. It does **not** resolve the baseline (sigma=8.9, needs ~308 seeds for 2 meV) -- no baseline claim finer than ~8.5 meV is supported. |
| Thread determinism | **decided** | `config.py` `OMP_THREADS` | Pinned at 2. Thread count changes floating-point reduction order; the CPU-batched baseline shifts ~0.5% if left to the host. |
| Output contract | **decided** | `README.md` | Three files, fixed schemas. `val_mae_eV` and `val_best_eV` are now distinct columns in every trainer -- the thesis reused one name for two quantities. |

## 8. Deployability, scalability and maintenance

| element | status | where | what it says |
|---|---|---|---|
| Inference latency | **decided** | `validate_performance.py` | 0.20 ms single sample on GPU, 0.09 ms on CPU, 0.0006 ms/sample batched. |
| Memory footprint | **decided** | `validate_performance.py` | 41,302 parameters, 161 kB at fp32. Size is not the deployment constraint. |
| Scalability | **partial** | `validate_performance.py` | Batched latency is flat to 1,292 samples. Training scaling with dataset size is not measured -- the dataset cannot grow without more DFT. |
| Containerisation | **gap** | -- | No Dockerfile or pinned lockfile. `check_environment.py` records the environment but does not pin it. |
| Concept drift | **n/a, argued** | -- | The mapping is a property of diamond, not of a changing world, so there is no drift to monitor. The analogous risk is *domain* drift -- applying it to a different material -- which is a transfer study, not a monitoring problem. |

---

## The silent killers

The failures that do not raise an exception. All five are checked; four run in
`check_sanity.py` in under a minute, and it runs **first** in stage 1.

| trap | covered | result |
|---|---|---|
| **Pre-processing leak** -- scaling or selecting on the full dataset before splitting | yes `check_sanity.py` | Scaler's mean differs from the full-data mean by 3.4e-4, proving it saw the training fold only. Zero identical input rows across folds. |
| **Clever Hans** -- right answer, wrong reason | yes `validate_explainability.py` | Tests the model's derivatives against values the physics fixes independently, rather than only reporting which feature it leans on. |
| **Metric-to-reality mismatch** | yes argued above | MAE in meV against a physical scale. This is a regression with a bounded target, so the imbalanced-accuracy trap does not apply -- but the per-family breakdown exists because the *average* hides that shear is worst. |
| **Complexity tax** -- a simpler model would do | yes `run_classical_baselines.py` | Random forest, GP and kernel ridge. The PINN's case rests on the scarce-label regime, not on full-data accuracy -- where the physics-free control is in fact the most accurate model. |
| **Silent gradient trap** -- can it overfit one batch? | yes `check_sanity.py` | 8 samples to 0.0000 meV; per-layer gradient norms within 1.8x. |

---

## What is still open

1. **Run stage 1.** Five scripts, all written, none yet run at scale. They settle
   the architecture, the split, the optimizer attribution and the epoch budget --
   every one of which is currently a thesis default rather than a measurement.
2. **Four `gap` rows**: a naive-predictor row in `22`, and a pinned environment
   (lockfile or container) for the reproduction package.
3. **Re-run the Sobol study at 6,000 epochs** with a larger base sample, or drop
   it to a qualitative statement.
4. **Verify `DFT_CORE_HOURS`** in `validate_performance.py` against the OUTCAR timings
   before any feasibility number goes in the paper.
