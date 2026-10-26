#!/usr/bin/env python3
"""Stage 1 of the paper's analysis: condense the raw run folders into the small files in summary/.

The raw results (several gigabytes: one folder per training run with test predictions, training logs and the trained
model) are produced by the task lists of make_tasks.py, the calibration scripts and classical_baselines.py, followed by
run_diagnostics.py, collect_metrics.py and select_weighting.py. This script reads them and writes summary/;
make_tables.py and make_figures.py then need only summary/, the dataset and the constants, all of which are in the
repository.

  runs_main.csv            one row per run of results/B_main: every accuracy metric, the physical-validity diagnostics
                           (Table 2), the lowest validation MAE and the training MAE at the retained checkpoint
  runs_ablation.csv        the same for results/B_ablation (our model with one term removed)
  runs_labelmatched.csv    the same for results/B_labelmatched (constants fitted on the labeled states)
  runs_analytic.csv        the analytic models alone on each run's test portion
  calibration_labelmatched.csv   the analytic models with label-matched constants, per labeled fraction and seed
  heldout_runs.csv         held-out families and strain cut-offs, every model and the analytic models alone
  classical_runs.csv       Gaussian process, random forest and gradient boosting (copy of results/classical/runs.csv)
  hybrid_choice.csv        validation errors of the six weighting variants (copy of results/hybrid_choice.csv)
  gradients_checkpoint.csv physics-to-data gradient ratio and cosine at the retained checkpoint
  gradients_training.csv   the same every 250 iterations, median ratio and mean cosine over the 17 runs
  sa_weights.csv           mean SA-PINN weight of each physics term at the end of training, per run
  curves.csv               mean training and lowest-validation MAE curves, every 10th iteration (Figure 4a,
                           Supplementary Figure 2)
  parity_median_run.csv    test predictions of the median run of our model with all training states labeled (Figure 5a)
  cost.csv                 number of jobs, CPU time and mean training time per model
  inference_times.txt      prediction time of a trained model
Usage: python analysis/export_summary.py [raw_results_dir]   (default: results/)"""
import glob, json, os, re, shutil, sys
import numpy as np, pandas as pd, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import physics as P
import metrics_report as MR
R = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results"))
OUT = os.path.join(ROOT, "summary"); os.makedirs(OUT, exist_ok=True)
FINAL, SA, RBA, MLP = "hyb_sdp_ld9", "sa_pinn", "rba_pinn", "mlp"
PAPER = [FINAL, SA, RBA, MLP, "delta", "delta_zs", "delta_zs_sym", "sdp_net", "hyb_sdp_ld45", "hyb_fixed", "hyb_ld45", "hyb_ld9", "hyb_ld45_col"]
SEEDS = list(range(42, 59)); SC = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
DATA = pd.read_csv(os.path.join(ROOT, "data", "dft_hy_v9.csv"))
K = os.path.join(ROOT, "results", "98_constants_v2"); KL = os.path.join(ROOT, "results", "100_constants_labelmatched")

# sign accuracy is measured against the gap at zero strain from a cubic fit to the isotropic states with |strain| <= 0.02
iso = DATA[DATA.strain_type == "isotropic"]; win = iso.E_xx.abs() <= 0.02
MR.EG0 = float(np.polyfit(iso.E_xx[win], iso.Eg_eV[win], 3)[-1])


def rel(d):
    """Run folder relative to the raw results directory, e.g. B_main/mlp/5pct/seed42."""
    return os.path.relpath(d, R)


def load(pattern, models=None):
    rows = []
    for d in sorted(glob.glob(os.path.join(R, pattern))):
        if not os.path.exists(d + "/metrics_summary.csv"): continue
        parts = d.split("/")
        if models is not None and parts[-3] not in models: continue
        s = pd.read_csv(d + "/metrics_summary.csv").iloc[0]
        r = MR.metrics(pd.read_csv(d + "/test_results.csv"))
        r.update(run=rel(d), model=parts[-3], level=int(parts[-2][:-3]), seed=int(parts[-1][4:]), n_labeled=int(s.n_labeled),
                 train_mae=float(s.train_mae_meV), best_epoch=float(s.get("best_ckpt_epoch", np.nan)))
        log = pd.read_csv(d + "/epoch_log.csv")
        r["val_best_final"] = float((pd.to_numeric(log.val_best_eV, errors="coerce").ffill() * 1000).iloc[-1])
        r["train_at_best"] = float(pd.to_numeric(log.train_mae_eV, errors="coerce").iloc[(log.epoch - r["best_epoch"]).abs().argmin()] * 1000) \
            if np.isfinite(r["best_epoch"]) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows); out["sign"] = 100 * out.sign_acc
    return out


def analytic_bands(ids, const_path):
    for k, v in json.load(open(const_path)).items(): setattr(P, k, float(v))
    x = torch.tensor(DATA.loc[ids, SC].values, dtype=torch.float64)
    return P.physics_cbm(x).numpy(), P.physics_vbm(x).numpy()


def test_ids(seed, scheme="B_main/%s/0pct" % FINAL):
    return pd.read_csv(os.path.join(R, scheme, "seed%d" % seed, "test_results.csv"))


# ---------------------------------------------------------------- per-run metrics and diagnostics
M = load("B_main/*/*pct/seed*", PAPER)
A = load("B_ablation/%s_drop_*/*pct/seed*" % FINAL)
L = load("B_labelmatched/*/*pct/seed*", [FINAL, "delta", "sdp_net"])
DG = pd.concat([pd.DataFrame([json.loads(l) for l in open(os.path.join(R, f))]) for f in ("diagnostics.jsonl", "diagnostics_ablation.jsonl")])
DG["run"] = DG.run.map(lambda p: p[p.index("/B_") + 1:]); DG = DG.drop_duplicates("run", keep="last")
DG["dp"] = DG.slope_hydrostatic / 3.0; DG["shear_slope"] = DG.slope_shear.abs(); DG["dp_an"] = DG.ref_slope_hydrostatic / 3.0
DG = DG.rename(columns={"axis_relabel_drift_meV": "axis", "shear_sign_drift_meV": "shearsign", "noise_rel_change": "noise"})
DGC = ["run", "dp", "shear_slope", "axis", "shearsign", "noise", "dp_an"]
M = M.merge(DG[DGC], on="run", how="left"); A = A.merge(DG[DGC], on="run", how="left")
M.to_csv(os.path.join(OUT, "runs_main.csv"), index=False); A.to_csv(os.path.join(OUT, "runs_ablation.csv"), index=False)
L.to_csv(os.path.join(OUT, "runs_labelmatched.csv"), index=False)

# ---------------------------------------------------------------- the analytic models alone
form, cal = [], []
for s in SEEDS:
    t = test_ids(s)
    cb, vb = analytic_bands(t.sample_id.values, os.path.join(K, "symmetry", "const_seed%d.json" % s))
    f = t.copy(); f["CBM_pred_eV"], f["VBM_pred_eV"], f["Eg_pred_eV"] = cb, vb, cb - vb
    r = MR.metrics(f); r.update(model="formula", seed=s, sign=100 * r["sign_acc"]); form.append(r)
    for p in (1, 2, 5, 10):
        cb_, vb_ = analytic_bands(t.sample_id.values, os.path.join(KL, "%dpct" % p, "const_seed%d.json" % s))
        cal.append(dict(level=p, seed=s, MAE_meV=1000 * np.abs(cb_ - vb_ - t.Eg_true_eV.values).mean()))
pd.DataFrame(form).to_csv(os.path.join(OUT, "runs_analytic.csv"), index=False)
pd.DataFrame(cal).to_csv(os.path.join(OUT, "calibration_labelmatched.csv"), index=False)

# ---------------------------------------------------------------- held-out families and strain cut-offs
hs = []
for d in sorted(glob.glob(os.path.join(R, "B_extrap", "*", "*", "seed*"))):
    model, scheme, seed = d.split("/")[-3], d.split("/")[-2], int(d.split("/")[-1][4:])
    if model not in PAPER or not os.path.exists(d + "/test_results.csv"): continue
    t = pd.read_csv(d + "/test_results.csv"); y, e = t.Eg_true_eV.values, t.Eg_pred_eV.values
    hs.append(dict(model=model, level=scheme, seed=seed, MAE_meV=1000 * np.abs(e - y).mean(), R2=1 - ((e - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                   max_meV=1000 * np.abs(e - y).max()))
    if model == FINAL:   # the analytic models alone on the same held-out states, with the constants of that run
        cb, vb = analytic_bands(t.sample_id.values, os.path.join(K, scheme, "const_seed%d.json" % seed)); eg = cb - vb
        hs.append(dict(model="formula", level=scheme, seed=seed, MAE_meV=1000 * np.abs(eg - y).mean(),
                       R2=1 - ((eg - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), max_meV=1000 * np.abs(eg - y).max()))
pd.DataFrame(hs).to_csv(os.path.join(OUT, "heldout_runs.csv"), index=False)

# ---------------------------------------------------------------- copies of small summaries
shutil.copy(os.path.join(R, "classical", "runs.csv"), os.path.join(OUT, "classical_runs.csv"))
shutil.copy(os.path.join(R, "hybrid_choice.csv"), os.path.join(OUT, "hybrid_choice.csv"))
shutil.copy(os.path.join(R, "inference_benchmark_final", "inference_times.txt"), os.path.join(OUT, "inference_times.txt"))

# ---------------------------------------------------------------- gradients and SA-PINN weights
G, GT = [], []
for m in (SA, RBA, FINAL):
    for p in (0, 1, 2, 5, 10, 25, 50, 75, 100):
        for d in sorted(glob.glob(os.path.join(R, "B_main", m, "%dpct" % p, "seed*"))):
            g = pd.read_csv(d + "/grad_probe.csv"); be = pd.read_csv(d + "/metrics_summary.csv").best_ckpt_epoch.iloc[0]
            g["ratio"] = g.grad_phys_norm / g.grad_data_norm
            row = g.iloc[(g.epoch - be).abs().argmin()]
            G.append(dict(model=m, level=p, seed=int(d.split("seed")[-1]), cos=row.cos, ratio=row.ratio))
            GT.append(g[["epoch", "ratio", "cos"]].assign(model=m, level=p))
pd.DataFrame(G).to_csv(os.path.join(OUT, "gradients_checkpoint.csv"), index=False)
pd.concat(GT).groupby(["model", "level", "epoch"]).agg(ratio_median=("ratio", "median"), cos_mean=("cos", "mean"), runs=("ratio", "size")) \
    .reset_index().to_csv(os.path.join(OUT, "gradients_training.csv"), index=False)
lam = []
for d in sorted(glob.glob(os.path.join(R, "B_main", SA, "*pct", "seed*", "paper_weights.csv"))):
    w = pd.read_csv(d)
    for _, row in w.iterrows(): lam.append(dict(level=int(d.split("/")[-3][:-3]), seed=int(d.split("/")[-2][4:]), term=row.term, mean=row["mean"]))
pd.DataFrame(lam).to_csv(os.path.join(OUT, "sa_weights.csv"), index=False)

# ---------------------------------------------------------------- mean training curves
cur = []
for m in (SA, RBA, MLP, FINAL):
    for p in (50, 100):
        tr_, va_ = [], []
        for path in sorted(glob.glob(os.path.join(R, "B_main", m, "%dpct" % p, "seed*", "epoch_log.csv"))):
            log = pd.read_csv(path); tr_.append(pd.to_numeric(log.train_mae_eV, errors="coerce").values * 1000)
            va_.append(pd.to_numeric(log.val_best_eV, errors="coerce").ffill().values * 1000)
        n = min(len(x) for x in va_); st = np.array([x[:n] for x in va_]); keep = np.isfinite(st).all(axis=0)
        cur.append(pd.DataFrame(dict(model=m, level=p, epoch=log.epoch.values[:n], val_best_mean=np.nanmean(st, axis=0),
                                     val_best_mean_complete=np.where(keep, st.mean(axis=0), np.nan),
                                     train_mean=np.nanmean([x[:n] for x in tr_], axis=0))))
cur = pd.concat(cur); cur = cur[(cur.epoch % 10 == 0) | (cur.epoch == 1)]   # every 10th iteration is enough to draw the curves
cur.to_csv(os.path.join(OUT, "curves.csv"), index=False, float_format="%.6g")

# ---------------------------------------------------------------- Figure 5a: the median run of our model, all labeled
f100 = M[(M.model == FINAL) & (M.level == 100)].sort_values("MAE_meV"); med = f100.iloc[len(f100) // 2]
pd.read_csv(os.path.join(R, med.run, "test_results.csv")).assign(run=med.run).to_csv(os.path.join(OUT, "parity_median_run.csv"), index=False)

# ---------------------------------------------------------------- cost of the campaign
rm = pd.read_csv(os.path.join(R, "run_metrics.csv"))   # every job of the campaign except the low-label ablation, read below
p7dirs = [d for d in glob.glob(os.path.join(R, "B_ablation", "%s_drop_*" % FINAL, "*pct", "seed*"))
          if int(d.split("/")[-2][:-3]) in (0, 1, 2) and d.split("/")[-3].split("_drop_")[1] in ("agreement", "cbm_vbm", "cons")]
p7 = [float(re.search(r"User time \(seconds\): ([0-9.]+)", open(d + "/resources.txt").read()).group(1)) for d in p7dirs]
wall = rm[rm.kind == "train"].groupby("model").wall_s.mean() / 60
cost = [dict(item="jobs", value=len(rm) + len(p7dirs)), dict(item="cpu_hours", value=(rm.cpu_s.sum() + sum(p7)) / 3600),
        dict(item="low_label_ablation_jobs", value=len(p7dirs))]
cost += [dict(item="train_minutes_%s" % m, value=wall[m]) for m in PAPER if m in wall.index]
pd.DataFrame(cost).to_csv(os.path.join(OUT, "cost.csv"), index=False)
print("summary written to %s: %d main runs, %d ablation runs, %d label-matched runs, %d held-out rows" % (OUT, len(M), len(A), len(L), len(hs)))
