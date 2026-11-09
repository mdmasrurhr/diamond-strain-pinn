"""Data access for the paper's numbers and figures.

Used by v3_numbers.py and v3_figures.py (in the code repository: analysis/make_tables.py and analysis/make_figures.py).
Two sources give the same values:
  raw      the campaign's run folders and files (the authors' machine, about 18 GB)
  summary  the condensed files in summary/ of the code repository, written from the raw source by export_summary()
configure() selects the source. Every function returns the same values from both sources, except the mean learning
curves of the figures, which the summary keeps at every 10th iteration.
Run folders are named relative to the results folder, for example B_main/phm2/5pct/seed42."""
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

PAPER_MODELS = ["phm2", "res_symlab_bal", "sa_pinn", "rba_pinn", "ntk_pinn", "aphynity", "mlp"]
ABLATION_FOLDERS = ["phm2_drop_agreement", "phm2_drop_iso_eg", "phm2_drop_copies", "phm_new_drop_anchor", "phm_new_drop_sym"]
LABELMATCHED_MODELS = ["phm2", "res_symlab_bal"]
GRADIENT_MODELS = ["sa_pinn", "rba_pinn", "phm2", "res_symlab_bal"]       # at 1, 5, 25 and 100% labeled
CURVE_MODELS = ["sa_pinn", "rba_pinn", "mlp", "phm2"]
HELDOUT_SUITE_MODELS = ["formula", "sa_pinn", "rba_pinn", "ntk_pinn", "aphynity", "mlp"]
SEEDS = list(range(42, 59))
PRELIMINARY = {"capacity_decision": "11_arch", "elements_decision": "11_arch", "optimizer_decision": "12_optim",
               "epoch_decision": "13_budget", "patience_decision": "13_budget", "split_decision": "10_split"}
CFG = {}


def configure(mode, master=None, repo=None):
    """mode 'raw': master = the DiamondPINN_MASTER folder; mode 'summary': repo = the code repository."""
    CFG.clear()
    CFG["mode"] = mode
    if mode == "raw":
        CFG["code"] = os.path.join(master, "06_cluster_bundle_leap2")
        CFG["results"] = os.path.join(master, "07_results", "leap2_campaign_v3")
        CFG["dft"] = os.path.join(master, "08_dft_raw_outputs")
        CFG["preliminary"] = os.path.join(master, "07_results", "preliminary_studies")
    elif mode == "summary":
        CFG["code"] = repo
        CFG["summary"] = os.path.join(repo, "summary")
    else:
        raise ValueError(mode)
    if CFG["code"] not in sys.path:
        sys.path.insert(0, CFG["code"])


def raw():
    return CFG["mode"] == "raw"


def summary_file(name):
    return os.path.join(CFG["summary"], name)


def _read(path):
    """A summary table, read back exactly as written (17 significant digits, round-trip parsing)."""
    return pd.read_csv(path, float_precision="round_trip")


def code_root():
    return CFG["code"]


def data_csv():
    return os.path.join(CFG["code"], "data", "dft_hy_v9.csv")


def constants_file(kind, seed, fraction=None):
    """kind 2band: the paper's constants; linear: the one-band shear term; labelmatched: fitted on labeled states."""
    if kind == "2band":
        return os.path.join(CFG["code"], "results", "98_constants_2band", "symmetry", "const_seed%d.json" % seed)
    if kind == "linear":
        return os.path.join(CFG["code"], "results", "98_constants_v2", "symmetry", "const_seed%d.json" % seed)
    return os.path.join(CFG["code"], "results", "100_constants_labelmatched_2band", "%dpct" % fraction,
                        "const_seed%d.json" % seed)


# ---------------------------------------------------------------------------------------------------- run metrics
def _run_folders(kind):
    if kind == "main":
        folders = ["B_main/%s" % m for m in PAPER_MODELS]
    elif kind == "ablation":
        folders = ["B_ablation/%s" % m for m in ABLATION_FOLDERS]
    else:
        folders = ["B_labelmatched/%s" % m for m in LABELMATCHED_MODELS]
    runs = []
    for folder in folders:
        runs += sorted(glob.glob(os.path.join(CFG["results"], folder, "*pct", "seed*")))
    if kind != "ablation":
        runs = sorted(runs)          # one alphabetical list, as the campaign's analysis read them
    return runs


def runs(kind, metrics):
    """One row per run (kind: main, ablation, labelmatched): every metric of metrics_report.metrics on the test
    predictions, plus run, model, level, seed, n_labeled, train_mae and best_epoch."""
    if not raw():
        t = _read(summary_file("runs_%s.csv" % kind))
        t["best_epoch"] = t.best_epoch.astype(float)          # a float in the raw source (NaN where missing)
        return t
    rows = []
    for d in _run_folders(kind):
        if not os.path.exists(d + "/metrics_summary.csv"):
            continue
        s = pd.read_csv(d + "/metrics_summary.csv").iloc[0]
        parts = d.split("/")
        r = metrics(pd.read_csv(d + "/test_results.csv"))
        r.update(run=os.path.relpath(d, CFG["results"]), model=parts[-3], level=int(parts[-2][:-3]), seed=int(parts[-1][4:]),
                 n_labeled=int(s.n_labeled), train_mae=float(s.train_mae_meV), best_epoch=float(s.get("best_ckpt_epoch", np.nan)))
        rows.append(r)
    return pd.DataFrame(rows)


def diagnostics():
    """Physical-validity diagnostics of every run (diag_run.py), one row per run, run relative to the results folder."""
    if not raw():
        return _read(summary_file("diagnostics.csv"))
    frames = []
    for name in ("diagnostics.jsonl", "diagnostics_res_symlab_bal.jsonl", "diagnostics_phm2.jsonl", "diagnostics_phm_new.jsonl"):
        frames.append(pd.DataFrame([json.loads(line) for line in open(os.path.join(CFG["results"], name))]))
    d = pd.concat(frames)
    d["run"] = d.run.map(lambda q: q[q.index("/B_") + 1:])
    d = d.drop_duplicates("run", keep="last")
    keep = set(os.path.relpath(r, CFG["results"]) for kind in ("main", "ablation") for r in _run_folders(kind))
    return d[d.run.isin(keep)].reset_index(drop=True)


def test_frame(seed):
    """Test predictions of PHM's 0% run of a seed: the test states of that seed's division, with their reference values."""
    if not raw():
        t = _read(summary_file("test_frames.csv"))
        return t[t.seed == seed].reset_index(drop=True)
    return pd.read_csv(os.path.join(CFG["results"], "B_main", "phm2", "0pct", "seed%d" % seed, "test_results.csv"))


def run_test_results(run):
    """Test predictions of one run (the summary holds only the median PHM run at 100%, drawn in Fig. 5a)."""
    if not raw():
        t = _read(summary_file("parity_run.csv"))
        assert t.run.iloc[0] == run, "the summary holds %s, not %s" % (t.run.iloc[0], run)
        return t.drop(columns=["run"])
    return pd.read_csv(os.path.join(CFG["results"], run, "test_results.csv"))


def state_errors(models, fraction):
    """Absolute gap error per test state and run (meV), for the given models at one labeled fraction."""
    if not raw():
        t = _read(summary_file("state_errors.csv"))
        return t[t.model.isin(models) & (t.level == fraction)].reset_index(drop=True)
    rows = []
    for m in models:
        for f in sorted(glob.glob(os.path.join(CFG["results"], "B_main", m, "%dpct" % fraction, "seed*", "test_results.csv"))):
            t = pd.read_csv(f)
            for i, e in zip(t.sample_id.values, (t.Eg_pred_eV - t.Eg_true_eV).abs().values * 1000):
                rows.append({"model": m, "level": fraction, "seed": int(f.split("seed")[-1].split("/")[0]), "sample_id": int(i),
                             "error_meV": e})
    return pd.DataFrame(rows)


def classical():
    if not raw():
        return _read(summary_file("classical_runs.csv"))
    return pd.read_csv(os.path.join(CFG["results"], "classical", "runs.csv"))


# ---------------------------------------------------------------------------------------------------- training logs
def checkpoint_floor(model, fraction):
    """Per run: lowest validation MAE at the end of training and training MAE at the retained checkpoint (meV)."""
    if not raw():
        t = _read(summary_file("checkpoint_floor.csv"))
        return t[(t.model == model) & (t.level == fraction)].reset_index(drop=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(CFG["results"], "B_main", model, "%dpct" % fraction, "seed*"))):
        log = pd.read_csv(d + "/epoch_log.csv")
        best = pd.to_numeric(log.val_best_eV, errors="coerce").ffill() * 1000
        epoch = pd.read_csv(d + "/metrics_summary.csv").best_ckpt_epoch.iloc[0]
        train = pd.to_numeric(log.train_mae_eV, errors="coerce").iloc[(log.epoch - epoch).abs().argmin()] * 1000
        rows.append({"model": model, "level": fraction, "seed": int(d.split("seed")[-1]), "best": best.iloc[-1], "train": train})
    return pd.DataFrame(rows)


def sa_weights():
    """SA-PINN: squared mean weight lambda^2 of each physics term at the end of training, per run."""
    if not raw():
        return _read(summary_file("sa_weights.csv"))
    rows = []
    for f in sorted(glob.glob(os.path.join(CFG["results"], "B_main", "sa_pinn", "*pct", "seed*", "paper_weights.csv"))):
        w = pd.read_csv(f)
        level = int(f.split("/")[-3][:-3])
        for _, row in w.iterrows():
            rows.append({"level": level, "term": row.term, "lam2": row["mean"] ** 2})
    return pd.DataFrame(rows)


def gradients_at_checkpoint():
    """Physics-to-data gradient ratio and cosine at the probe nearest the retained checkpoint, per run: the four models
    of GRADIENT_MODELS at 1, 5, 25 and 100% labeled, and NTK-PINN at every fraction."""
    if not raw():
        return _read(summary_file("gradients_checkpoint.csv"))
    rows = []
    jobs = [(m, p) for m in GRADIENT_MODELS for p in (1, 5, 25, 100)]
    jobs += [("ntk_pinn", p) for p in (0, 1, 2, 5, 10, 25, 50, 75, 100)]
    for m, p in jobs:
        for d in sorted(glob.glob(os.path.join(CFG["results"], "B_main", m, "%dpct" % p, "seed*"))):
            if not os.path.exists(d + "/grad_probe.csv"):
                continue
            g = pd.read_csv(d + "/grad_probe.csv")
            epoch = pd.read_csv(d + "/metrics_summary.csv").best_ckpt_epoch.iloc[0]
            row = g.iloc[(g.epoch - epoch).abs().argmin()]
            rows.append({"model": m, "level": p, "seed": int(d.split("seed")[-1]), "cos": row.cos,
                         "ratio": row.grad_phys_norm / row.grad_data_norm})
    return pd.DataFrame(rows)


def gradient_trajectory(model, fraction):
    """Every 250 iterations from 250 on: median physics-to-data gradient ratio and mean cosine over the runs."""
    if not raw():
        t = _read(summary_file("gradients_training.csv"))
        t = t[(t.model == model) & (t.level == fraction)]
        return t.epoch.values, t.ratio.values, t.cos.values
    rs, cs = [], []
    for f in sorted(glob.glob(os.path.join(CFG["results"], "B_main", model, "%dpct" % fraction, "seed*", "grad_probe.csv"))):
        g = pd.read_csv(f)
        g = g[g.epoch >= 250]
        rs.append((g.grad_phys_norm / g.grad_data_norm).values)
        cs.append(g.cos.values)
        ep = g.epoch.values
    n = min(len(r) for r in rs)
    return ep[:n], np.median(np.array([r[:n] for r in rs]), axis=0), np.mean(np.array([c[:n] for c in cs]), axis=0)


def validation_floor_curve(model, fraction):
    """Lowest validation MAE reached up to each iteration, mean over the runs (meV), where every run has a value."""
    if not raw():
        t = _read(summary_file("curves.csv"))
        t = t[(t.model == model) & (t.level == fraction) & t.val_best_all.notna()]
        return t.epoch.values, t.val_best_all.values
    curves = []
    for path in sorted(glob.glob(os.path.join(CFG["results"], "B_main", model, "%dpct" % fraction, "seed*", "epoch_log.csv"))):
        log = pd.read_csv(path)
        curves.append(pd.to_numeric(log.val_best_eV, errors="coerce").ffill().values * 1000)
    n = min(len(c) for c in curves)
    st = np.array([c[:n] for c in curves])
    keep = np.isfinite(st).all(axis=0)
    return log.epoch.values[:n][keep], st[:, keep].mean(axis=0)


def training_curves(model, fraction):
    """Training MAE and lowest validation MAE, mean over the runs ignoring missing values (meV), at every iteration."""
    if not raw():
        t = _read(summary_file("curves.csv"))
        t = t[(t.model == model) & (t.level == fraction)]
        return t.epoch.values, t.train.values, t.val_best.values
    tr_, va_ = [], []
    for d in sorted(glob.glob(os.path.join(CFG["results"], "B_main", model, "%dpct" % fraction, "seed*"))):
        log = pd.read_csv(d + "/epoch_log.csv")
        tr_.append(pd.to_numeric(log.train_mae_eV, errors="coerce").values * 1000)
        va_.append(pd.to_numeric(log.val_best_eV, errors="coerce").ffill().values * 1000)
    n = min(len(x) for x in tr_)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return log.epoch.values[:n], np.nanmean([x[:n] for x in tr_], axis=0), np.nanmean([x[:n] for x in va_], axis=0)


def run_minutes(model):
    """Wall time of every run of a model in B_main (minutes, one core)."""
    if not raw():
        t = _read(summary_file("run_minutes.csv"))
        return list(t[t.model == model].minutes.values)
    out = []
    for d in sorted(glob.glob(os.path.join(CFG["results"], "B_main", model, "*pct", "seed*"))):
        w = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)", open(d + "/resources.txt").read()).group(1)
        out.append(sum(float(x) * 60 ** i for i, x in enumerate(reversed(w.split(":")))) / 60)
    return out


# ---------------------------------------------------------------------------------------------------- held-out tests
def heldout_suite():
    """Held-out families and strain cut-offs of the analytic models alone and the baselines (one row per run)."""
    if not raw():
        return _read(summary_file("heldout_suite.csv"))
    hs = pd.read_csv(os.path.join(CFG["results"], "heldout_suite_runs.csv"))
    return hs[hs.model.isin(HELDOUT_SUITE_MODELS)].reset_index(drop=True)


def heldout_runs(model):
    """Held-out runs of a model from its run folders: MAE, R2 and maximum error per run."""
    if not raw():
        t = _read(summary_file("heldout_runs.csv"))
        return t[t.model == model].reset_index(drop=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(CFG["results"], "B_extrap", model, "*", "seed*"))):
        t = pd.read_csv(d + "/test_results.csv")
        y, e = t.Eg_true_eV.values, t.Eg_pred_eV.values
        rows.append({"model": model, "level": d.split("/")[-2], "seed": int(d.split("/")[-1][4:]),
                     "MAE_meV": 1000 * np.abs(e - y).mean(), "R2": 1 - ((e - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     "max_meV": 1000 * np.abs(e - y).max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- other inputs
def aphynity_choice():
    path = summary_file("aphynity_choice.csv") if not raw() else os.path.join(CFG["results"], "aphynity_choice.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def shear_formula_test():
    if not raw():
        return _read(summary_file("shear_formula_test.csv"))
    return pd.read_csv(os.path.join(CFG["results"], "shear_tests", "formula_test.csv"))


def inference_text():
    if not raw():
        return open(summary_file("inference_times.txt")).read()
    return open(os.path.join(CFG["results"], "inference_benchmark_gap", "inference_times.txt")).read()


def preliminary(name):
    """A table of the preliminary studies (capacity_decision, elements_decision, ...)."""
    if not raw():
        return _read(os.path.join(CFG["summary"], "preliminary", name + ".csv"))
    return pd.read_csv(os.path.join(CFG["preliminary"], PRELIMINARY[name], name + ".csv"))


def hse_core_hours():
    """Core-hours of every hybrid-functional calculation (wall time x cores, from its OUTCAR)."""
    if not raw():
        return _read(summary_file("hse_core_hours.csv")).core_hours.values
    values = []
    for path in sorted(glob.glob(os.path.join(CFG["dft"], "*", "*", "2-HSE06", "OUTCAR"))):
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            tail = f.read().decode(errors="ignore")
        wall = re.search(r"Elapsed time \(sec\):\s+([0-9.]+)", tail)
        with open(path, errors="ignore") as f:
            head = f.read(200000)
        cores = re.search(r"running on\s+(\d+) total cores", head)
        if wall and cores:
            values.append(float(wall.group(1)) * int(cores.group(1)) / 3600)
    return np.array(values)


def gamma_x_bands():
    """Two lowest conduction bands along Gamma-X in the unstrained HSE06 calculation: (fraction of Gamma-X, E5, E6)."""
    if not raw():
        t = _read(summary_file("gamma_x_bands.csv"))
        return [tuple(r) for r in t[["fraction", "band5_eV", "band6_eV"]].itertuples(index=False, name=None)]
    path = os.path.join(CFG["dft"], "reference_2_atoms", "HSE06")
    L = open(os.path.join(path, "EIGENVAL")).read().split("\n")
    nk, nb = map(int, L[5].split()[1:3])
    poscar = open(os.path.join(path, "POSCAR")).read().split("\n")
    A_ = np.array([[float(x) for x in l.split()[:3]] for l in poscar[2:5]]) * float(poscar[1])
    B_ = np.linalg.inv(A_).T
    X_ = 1.0 / (np.sqrt(2) * np.linalg.norm(A_[0]))          # |Gamma-X| = 1/a without the 2 pi
    i, out = 7, []
    axes = [A_[1] + A_[2] - A_[0], A_[0] + A_[2] - A_[1], A_[0] + A_[1] - A_[2]]
    for _ in range(nk):
        while not L[i].strip():
            i += 1
        kc = np.array(L[i].split()[:3], float) @ B_
        i += 1
        E_ = [float(L[i + j].split()[1]) for j in range(nb)]
        i += nb
        for ax in axes:
            u = ax / np.linalg.norm(ax)
            par = kc @ u
            if abs(par) > 1e-6 and np.linalg.norm(kc - par * u) < 1e-5:
                out.append((round(abs(par) / X_, 3), E_[4], E_[5]))
    return sorted(set(out))


# ---------------------------------------------------------------------------------------------------- export
def export_summary(metrics, summary_dir, thin=10):
    """Write every summary file from the raw source (call after configure('raw', ...))."""
    os.makedirs(os.path.join(summary_dir, "preliminary"), exist_ok=True)

    def write(frame, name):
        frame.to_csv(os.path.join(summary_dir, name), index=False, float_format="%.17g")

    for kind in ("main", "ablation", "labelmatched"):
        write(runs(kind, metrics), "runs_%s.csv" % kind)
    write(diagnostics(), "diagnostics.csv")
    write(pd.concat([test_frame(s) for s in SEEDS], ignore_index=True), "test_frames.csv")
    main_runs = runs("main", metrics)
    f100 = main_runs[(main_runs.model == "phm2") & (main_runs.level == 100)].sort_values("MAE_meV")
    median_run = f100.iloc[len(f100) // 2].run
    parity = run_test_results(median_run)
    parity.insert(0, "run", median_run)
    write(parity, "parity_run.csv")
    write(state_errors(["phm2", "mlp"], 100), "state_errors.csv")
    write(classical(), "classical_runs.csv")
    write(pd.concat([checkpoint_floor(m, 50) for m in CURVE_MODELS], ignore_index=True), "checkpoint_floor.csv")
    write(sa_weights(), "sa_weights.csv")
    write(gradients_at_checkpoint(), "gradients_checkpoint.csv")
    rows = []
    for m in ("sa_pinn", "rba_pinn", "ntk_pinn", "phm2", "res_symlab_bal"):
        for p in (5, 25, 100):
            ep, r, c = gradient_trajectory(m, p)
            rows.append(pd.DataFrame({"model": m, "level": p, "epoch": ep, "ratio": r, "cos": c}))
    write(pd.concat(rows, ignore_index=True), "gradients_training.csv")
    rows = []
    for m in CURVE_MODELS:
        for p in (50, 100):
            ep, train, val = training_curves(m, p)
            ep_all, val_all = validation_floor_curve(m, p)
            t = pd.DataFrame({"model": m, "level": p, "epoch": ep, "train": train, "val_best": val})
            t = t.merge(pd.DataFrame({"epoch": ep_all, "val_best_all": val_all}), on="epoch", how="left")
            rows.append(t[(t.epoch % thin == 0) | (t.epoch == 1)])
    write(pd.concat(rows, ignore_index=True), "curves.csv")
    rows = []
    for m in ("phm2", "sa_pinn", "rba_pinn"):
        rows.append(pd.DataFrame({"model": m, "minutes": run_minutes(m)}))
    write(pd.concat(rows, ignore_index=True), "run_minutes.csv")
    write(heldout_suite(), "heldout_suite.csv")
    write(pd.concat([heldout_runs(m) for m in ("phm2", "res_symlab_bal")], ignore_index=True), "heldout_runs.csv")
    choice = aphynity_choice()
    if choice is not None:
        write(choice, "aphynity_choice.csv")
    write(shear_formula_test(), "shear_formula_test.csv")
    open(os.path.join(summary_dir, "inference_times.txt"), "w").write(inference_text())
    for name in PRELIMINARY:
        write(preliminary(name), os.path.join("preliminary", name + ".csv"))
    write(pd.DataFrame({"core_hours": hse_core_hours()}), "hse_core_hours.csv")
    write(pd.DataFrame(gamma_x_bands(), columns=["fraction", "band5_eV", "band6_eV"]), "gamma_x_bands.csv")
