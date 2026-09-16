"""Checks to run before training: environment, dataset, and training sanity.

    python 01_check.py environment   record interpreter, versions, hardware
    python 01_check.py data          dataset contents and integrity
    python 01_check.py sanity        faults that do not raise an exception
    python 01_check.py all           all three
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

import config as C
# Imported for its model and data pipeline, so the sanity checks test the real
# training code rather than a copy of it.
import train_pinn

# Packages whose version can change a result.
TRACKED = ["numpy", "torch", "pandas", "scikit-learn", "scipy"]


TRACKED = ["numpy", "torch", "pandas", "scikit-learn", "scipy"]


def _sha256(path, cap=None):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def _gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        return [l.strip() for l in out.strip().splitlines()]
    except Exception:
        return []


def manifest():
    m = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict((p, _version(p)) for p in TRACKED),
        "gpus": _gpus(),
        "dataset": {
            "path": os.path.relpath(C.DATA_CSV, C.ROOT),
            "sha256": _sha256(C.DATA_CSV) if os.path.exists(C.DATA_CSV) else None,
            "bytes": os.path.getsize(C.DATA_CSV) if os.path.exists(C.DATA_CSV) else None,
        },
        # The seed policy, not just the seeds: what each one controls matters
        # more than its value.
        "seeds": {
            "list": C.SEEDS,
            "controls": "split partition, weight initialisation, and the labelled "
                        "subset draw (via LABEL_RNG_OFFSET, so the subset does not "
                        "move when the split does)",
            "label_rng_offset": C.LABEL_RNG_OFFSET,
        },
        "thread_policy": {
            "OMP_NUM_THREADS": C.OMP_THREADS,
            "why": "thread count changes floating-point reduction order; the Shi "
                   "baseline batches on the CPU and shifts by ~0.5% if this is "
                   "left to the host",
        },
        "tuned_config": C.TUNED,
    }
    return m


def check_environment(args):

    m = manifest()
    print("ENVIRONMENT")
    print("  python    %s  (%s)" % (m["python"], m["python_executable"]))
    print("  platform  %s" % m["platform"])
    for p, v in m["packages"].items():
        print("  %-14s %s" % (p, v or "NOT INSTALLED"))
    for g in m["gpus"]:
        print("  gpu       %s" % g)
    print("  dataset   %s" % m["dataset"]["path"])
    print("            sha256 %s" % (m["dataset"]["sha256"] or "MISSING")[:32])
    print("  seeds     %d (%d..%d)" % (len(C.SEEDS), C.SEEDS[0], C.SEEDS[-1]))

    out = C.results_dir("environment")
    path = os.path.join(out, "env_manifest.json")

    if args.compare and os.path.exists(args.compare):
        old = json.load(open(args.compare))
        diffs = []
        for k in ("python", "platform", "packages", "gpus"):
            if old.get(k) != m.get(k):
                diffs.append((k, old.get(k), m.get(k)))
        if old.get("dataset", {}).get("sha256") != m["dataset"]["sha256"]:
            diffs.append(("dataset sha256",
                          old.get("dataset", {}).get("sha256"),
                          m["dataset"]["sha256"]))
        print("\nCOMPARISON with %s" % args.compare)
        if not diffs:
            print("  identical -- results from before and after are one experiment")
        else:
            for k, a, b in diffs:
                print("  CHANGED %s\n    was: %s\n    now: %s" % (k, a, b))
            print("\n  Results produced either side of this change are not directly")
            print("  comparable. Re-run the affected stages or report them separately.")

    json.dump(m, open(path, "w"), indent=2)
    print("\nwrote %s" % os.path.relpath(path, C.ROOT))


def family_of(label):
    """Collapse the CSV's sub-labels (uniaxial_x, unstrained, ...) to 5 families."""
    s = str(label).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"          # 'unstrained' counts as the isotropic reference


def check_data(args):
    df = pd.read_csv(C.DATA_CSV)
    out = C.results_dir("data_checks")

    # ---- structure ------------------------------------------------------
    needed = C.STRAIN_COLS + [C.TARGET_EG, C.TARGET_CBM, C.TARGET_VBM]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit("dataset is missing required columns: %s" % missing)

    n_null = int(df[needed].isnull().sum().sum())
    print("rows: %d    required columns: all present    missing values: %d"
          % (len(df), n_null))
    if n_null:
        raise SystemExit("dataset has missing values -- stop and fix the CSV")

    # ---- composition ----------------------------------------------------
    fam = df[C.TARGET_EG].groupby(df["strain_type"].map(family_of))
    rows = []
    print("\n%-11s %6s  %18s  %18s" % ("family", "n", "Eg range (eV)", "Eg mean (eV)"))
    for name in C.STRAIN_FAMILIES:
        if name not in fam.groups:
            continue
        v = fam.get_group(name)
        rows.append(dict(family=name, n=len(v), eg_min=round(v.min(), 4),
                         eg_max=round(v.max(), 4), eg_mean=round(v.mean(), 4)))
        print("%-11s %6d  %8.3f - %-8.3f %12.4f"
              % (name, len(v), v.min(), v.max(), v.mean()))
    print("%-11s %6d" % ("TOTAL", len(df)))

    # ---- strain magnitudes ----------------------------------------------
    s = df[C.STRAIN_COLS].values
    print("\nstrain components: min %.4f  max %.4f  (dimensionless Green-Lagrange)"
          % (s.min(), s.max()))

    # ---- repeated strain states -----------------------------------------
    # Some strain vectors appear more than once. Where they do, the labels are
    # identical too, so these are the same calculation stored repeatedly rather
    # than a reproducibility probe. It matters because the split is by ROW: two
    # copies of one state can land on opposite sides of the train/test boundary,
    # and the test copy is then trivially predictable.
    key = df[C.STRAIN_COLS].round(9).apply(tuple, axis=1)
    counts = key.value_counts()
    repeated = counts[counts > 1]
    n_rep_rows = int(repeated.sum())
    print("\ndistinct strain states: %d of %d rows" % (key.nunique(), len(df)))
    if len(repeated):
        def spread(values):
            """Range of the bandgap within one group of duplicate states."""
            return values.max() - values.min()

        lab_spread = df.groupby(key)[C.TARGET_EG].agg(spread)
        print("repeated states: %d, covering %d rows (%.1f%%); copies per state %s"
              % (len(repeated), n_rep_rows, 100.0 * n_rep_rows / len(df),
                 sorted(repeated.unique())))
        print("largest label disagreement between copies: %.3f meV"
              % (lab_spread.max() * 1000.0))

    # ---- split integrity -------------------------------------------------
    stype = df["strain_type"].map(family_of).values
    idx = np.arange(len(df))
    twins = []
    for seed in C.SEEDS:
        i_tv, i_te = train_test_split(idx, test_size=C.TEST_FRACTION,
                                      random_state=seed, stratify=stype)
        i_tr, _ = train_test_split(i_tv, test_size=C.VAL_FRACTION_OF_REMAINDER,
                                   random_state=seed, stratify=stype[i_tv])
        train_keys = set(key.iloc[i_tr])
        twins.append(sum(1 for k in key.iloc[i_te] if k in train_keys))
    n_te = len(i_te)
    print("\ntest rows holding an exact copy of a training row: "
          "mean %.1f of %d (%.1f%%), range %d-%d over %d seeds"
          % (np.mean(twins), n_te, 100.0 * np.mean(twins) / n_te,
             min(twins), max(twins), len(C.SEEDS)))
    print("  -> every model is scored on the same rows, so comparisons are fair,")
    print("     but absolute errors are slightly optimistic. 11_ablate.py duplicates")
    print("     re-runs the headline models on distinct states only.")

    summary = pd.DataFrame(rows)
    summary.to_csv("%s/dataset_summary.csv" % out, index=False)
    pd.DataFrame(dict(metric=["rows", "distinct_states", "repeated_states",
                              "repeated_rows", "test_rows_with_train_copy_mean"],
                      value=[len(df), key.nunique(), len(repeated), n_rep_rows,
                             round(float(np.mean(twins)), 1)])).to_csv(
        "%s/integrity_summary.csv" % out, index=False)
    print("\nwrote %s/dataset_summary.csv and integrity_summary.csv" % out)


def _load_trainer(extra_argv=()):
    """Return the trainer module. The argument is accepted but unused."""
    return train_pinn


# --------------------------------------------------------------- checks
def check_overfit_batch(rows):
    """Memorise 8 samples. Anything but near-zero error means a broken graph."""
    T = _load_trainer([])
    data = T.load_and_split(C.DATA_CSV, seed=42, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 8
    X = torch.tensor(data["X_tr"][:n], dtype=torch.float32, device=dev)
    cbm = torch.tensor(data["cbm_tr"][:n], dtype=torch.float32, device=dev)
    vbm = torch.tensor(data["vbm_tr"][:n], dtype=torch.float32, device=dev)

    model = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Data term only. The physics terms are deliberately excluded: this asks
    # whether the network CAN fit, not whether the physics lets it.
    for _ in range(2000):
        opt.zero_grad()
        c, v, _ = model(X)
        loss = ((c - cbm) ** 2).mean() + ((v - vbm) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        c, v, eg = model(X)
        mae = float((eg - (cbm - vbm)).abs().mean()) * 1000.0

    ok = mae < 1.0
    print("  overfit 8 samples : Eg MAE %.4f meV -> %s"
          % (mae, "OK" if ok else "BROKEN -- the network cannot fit 8 points"))
    rows.append(dict(check="overfit_batch", value=round(mae, 4), unit="meV",
                     passed=ok, note="data term only, 2000 Adam steps"))


def check_gradients(rows):
    """Per-layer gradient norms at initialisation, on the real composite loss."""
    T = _load_trainer([])
    data = T.load_and_split(C.DATA_CSV, seed=42, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)

    X = torch.tensor(data["X_tr"], dtype=torch.float32, device=dev)
    cbm = torch.tensor(data["cbm_tr"], dtype=torch.float32, device=dev)
    vbm = torch.tensor(data["vbm_tr"], dtype=torch.float32, device=dev)
    c, v, _ = model(X)
    loss = ((c - cbm) ** 2).mean() + ((v - vbm) ** 2).mean()
    loss.backward()

    norms = [(n, float(p.grad.norm())) for n, p in model.named_parameters()
             if p.grad is not None and n.endswith("weight")]
    vals = [x for _, x in norms]
    spread = max(vals) / max(min(vals), 1e-12)
    for n, x in norms:
        print("    %-28s %.3e" % (n, x))
    # Two orders of magnitude across layers is normal for a plain MLP; four is
    # the regime where the deepest layers stop moving.
    ok = spread < 1e4 and min(vals) > 1e-8
    print("  gradient spread   : %.1fx across %d layers -> %s"
          % (spread, len(norms), "OK" if ok else "SUSPECT"))
    rows.append(dict(check="gradient_spread", value=round(spread, 1), unit="ratio",
                     passed=ok, note="max/min weight-grad norm at init"))


def check_leakage(rows):
    """The scaler must be fitted on the training fold alone, and folds disjoint."""
    T = _load_trainer([])
    data = T.load_and_split(C.DATA_CSV, seed=42, dft_pct=100)

    df = pd.read_csv(C.DATA_CSV)
    X_all = T.compute_raw_features(df)

    # If the scaler had seen everything, its mean would equal the full-data mean.
    # Fitted on the training fold it must not, and the gap is the proof.
    full_mean = X_all.mean(axis=0)
    gap = float(np.abs(data["scaler"].mean_ - full_mean).max())
    ok_scaler = gap > 1e-9
    print("  scaler fitted on  : train fold only (max |mean - full mean| = %.3e) -> %s"
          % (gap, "OK" if ok_scaler else "LEAK: scaler saw all rows"))
    rows.append(dict(check="scaler_leak", value=float("%.3e" % gap), unit="",
                     passed=ok_scaler, note="scaler mean must differ from full-data mean"))

    # Compare inputs, not targets. Symmetry-equivalent deformations share a
    # bandgap by construction, so repeated targets are expected; repeated
    # inputs across folds are not.
    tr = set(map(tuple, np.round(data["X_tr"], 6)))
    te = set(map(tuple, np.round(data["X_te"], 6)))
    va = set(map(tuple, np.round(data["X_va"], 6)))
    overlap = len(tr & te) + len(tr & va) + len(va & te)
    ok = overlap == 0
    print("  train/test overlap: %d identical input rows -> %s"
          % (overlap, "OK" if ok else "LEAK: the same state is in two folds"))
    rows.append(dict(check="fold_overlap", value=overlap, unit="rows",
                     passed=ok, note="identical scaled input vectors across folds"))


def check_duplicates(rows):
    """Exact-duplicate strain states are memorisation wearing generalisation's coat."""
    df = pd.read_csv(C.DATA_CSV)
    key = df[C.STRAIN_COLS].round(9).astype(str).agg("|".join, axis=1)
    dup = int(len(key) - key.nunique())
    ok = dup == 0
    print("  duplicate states  : %d of %d rows -> %s"
          % (dup, len(df), "OK" if ok else
             "these can straddle the split; use the deduplicated CSV"))
    rows.append(dict(check="duplicate_states", value=dup, unit="rows", passed=ok,
                     note="rows sharing an identical 6-component strain vector"))


def check_label_shuffle(rows):
    """Train on shuffled labels. Good performance here means a leak somewhere."""
    T = _load_trainer([])
    data = T.load_and_split(C.DATA_CSV, seed=42, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data["cbm_tr"]))
    X = torch.tensor(data["X_tr"], dtype=torch.float32, device=dev)
    cbm = torch.tensor(data["cbm_tr"][perm], dtype=torch.float32, device=dev)
    vbm = torch.tensor(data["vbm_tr"][perm], dtype=torch.float32, device=dev)
    Xte = torch.tensor(data["X_te"], dtype=torch.float32, device=dev)
    eg_te = torch.tensor(data["eg_te"], dtype=torch.float32, device=dev)

    model = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3000):
        opt.zero_grad()
        c, v, _ = model(X)
        (((c - cbm) ** 2).mean() + ((v - vbm) ** 2).mean()).backward()
        opt.step()
    with torch.no_grad():
        _, _, eg = model(Xte)
        mae = float((eg - eg_te).abs().mean()) * 1000.0

    # With destroyed labels the model can do no better than predicting a constant,
    # whose error is the mean absolute deviation of the target.
    baseline = float((eg_te - eg_te.mean()).abs().mean()) * 1000.0
    ok = mae > 0.5 * baseline
    print("  label shuffle     : %.1f meV vs %.1f meV for a constant -> %s"
          % (mae, baseline, "OK" if ok else
             "SUSPECT: performs well on shuffled labels, something leaks"))
    rows.append(dict(check="label_shuffle", value=round(mae, 1), unit="meV",
                     passed=ok, note="constant-predictor error is %.1f meV" % baseline))


CHECKS = {
    "overfit_batch": check_overfit_batch,
    "gradients":     check_gradients,
    "leakage":       check_leakage,
    "duplicates":    check_duplicates,
    "label_shuffle": check_label_shuffle,
}
DEFAULT = ["overfit_batch", "gradients", "leakage", "duplicates"]


def check_sanity(args):
    which = args.check or (sorted(CHECKS) if args.all else DEFAULT)
    rows = []
    for name in which:
        print("\n[%s]" % name)
        CHECKS[name](rows)

    out = C.results_dir("sanity_checks")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sanity_checks.csv"), index=False)
    n_fail = int((~df.passed).sum())
    print("\n%d checks, %d failed" % (len(df), n_fail))
    if n_fail:
        print("failed: %s" % ", ".join(df[~df.passed].check))
    print("wrote %s" % os.path.join(out, "sanity_checks.csv"))
    return 1 if n_fail else 0

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all",
                    choices=["environment", "data", "sanity", "all"])
    ap.add_argument("--compare", help="environment: an earlier manifest to diff against")
    ap.add_argument("--check", nargs="+", default=None, choices=sorted(CHECKS),
                    help="sanity: run only these checks")
    ap.add_argument("--all", action="store_true",
                    help="sanity: include the label-shuffle check")
    args = ap.parse_args()

    status = 0
    if args.what in ("environment", "all"):
        check_environment(args)
    if args.what in ("data", "all"):
        check_data(args)
    if args.what in ("sanity", "all"):
        status = check_sanity(args) or 0
    return status


if __name__ == "__main__":
    sys.exit(main())
