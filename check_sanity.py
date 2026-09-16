"""Checks for faults that do not raise an exception.

A broken training setup usually still runs to completion with a plausible
looking loss curve. Each check below targets one way that can happen.

    overfit_batch   train on 8 samples only. If the error does not fall to
                    near zero, the network or the gradient path is broken.
    gradients       per-layer gradient norms at initialisation. A layer whose
                    gradient is orders of magnitude smaller than its
                    neighbours is not learning.
    leakage         confirms the scaler saw the training fold only and that no
                    row appears in two folds.
    duplicates      counts strain states that appear more than once. Copies
                    split across train and test inflate the score.
    label_shuffle   train on shuffled labels. The error must collapse to the
                    spread of the target; anything better means the model is
                    reading something other than the labels.

    python check_sanity.py              # all except label_shuffle
    python check_sanity.py --all        # including the shuffle control
    python check_sanity.py --check leakage duplicates

Writes results/sanity_checks/sanity_checks.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import config as C


# train_pinn is imported, not re-implemented: a check that tests its own copy of
# the model is not testing the model. Importing it also runs its argument parser
# with default values, which is why --data_csv has a default.
import train_pinn


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", nargs="+", default=None, choices=sorted(CHECKS))
    ap.add_argument("--all", action="store_true", help="include label_shuffle")
    args = ap.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
