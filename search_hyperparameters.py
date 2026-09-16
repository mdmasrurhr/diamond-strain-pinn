"""Random search over the training hyperparameters.

Random rather than grid: over this many settings a grid is either too coarse to
be useful or too large to run.

Protocol:

  * Trials are scored on validation error only; the test fold is not consulted.
  * Each trial runs at three seeds and is scored on the mean, because one seed
    cannot separate configurations that differ by less than the seed spread.
  * The winner is re-run at all 17 seeds before being reported.
  * The full ranking is written out, not just the winner, so the shape of the
    search space is visible.

Run this after the main experiment, not before: reporting the best of several
hundred trials as the headline result selects on the quantity being reported.

    python search_hyperparameters.py --trials 300
    python search_hyperparameters.py --trials 300 --space wide
    python search_hyperparameters.py --confirm

Writes results/hyperparameter_search/trial<NNN>/seed<n>/ and ranking.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

import config as C
import jobs

# Each knob is drawn either log-uniformly (learning rates, decays) or from a
# short list. Ranges span about a decade either side of the default.
SPACE = {
    "adamw_lr":  ("loguniform", 3e-4, 1e-2),
    "adamw_wd":  ("loguniform", 1e-6, 1e-3),
    "soap_lr":   ("loguniform", 1e-3, 3e-2),
    "soap_wd":   ("loguniform", 1e-4, 1e-2),
    "clip_norm": ("choice", [0.5, 1.0, 2.0, 5.0]),
    "width":     ("choice", [50, 100, 150, 200]),
    # Depth reaches 8 because every depth-7 configuration in the capacity sweep
    # beat every depth-5 one; a search capped at 6 could not have found that.
    "depth":     ("choice", [3, 4, 5, 6, 7, 8]),
    "switch_frac": ("choice", [0.25, 0.5, 0.75]),
    "sa_lr":     ("loguniform", 1e-3, 3e-2),
}

# --space wide adds the architecture options and two loss weights. It is a much
# larger space and needs proportionally more trials, so it is kept separate.
WIDE_EXTRA = {
    "norm":        ("choice", ["none", "layer"]),
    "dropout":     ("choice", [0.0, 0.0, 0.05, 0.1]),   # weighted toward off
    "skip":        ("choice", ["none", "residual"]),
    "width_shape": ("choice", ["uniform", "taper", "widen"]),
    "w_data":      ("loguniform", 1.0, 20.0),
    "w_cons":      ("loguniform", 0.2, 3.0),
}


def sample(rng, space=None):
    out = {}
    for name, spec in (space or SPACE).items():
        if spec[0] == "loguniform":
            lo, hi = spec[1], spec[2]
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[name] = spec[1][rng.integers(len(spec[1]))]
    return out


def trial_flags(p):
    flags = []
    for k, v in p.items():
        if isinstance(v, str):
            val = v                                    # norm, skip, width_shape
        elif isinstance(v, (int, np.integer)):
            val = "%d" % v
        else:
            val = "%.6g" % v
        flags += ["--%s" % k, val]
    return flags


def rank():
    """Read every finished trial and rank by mean validation error."""
    base = os.path.join(C.RESULTS, "hyperparameter_search")
    rows = []
    for tdir in sorted(glob.glob(os.path.join(base, "trial*"))):
        vals = []
        for f in glob.glob(os.path.join(tdir, "seed*", "metrics_summary.csv")):
            try:
                d = pd.read_csv(f)
                vals.append(float(d.iloc[-1]["val_mae_meV"]))
            except Exception:
                pass
        if not vals:
            continue
        cfg = {}
        cfg_path = os.path.join(tdir, "config.csv")
        if os.path.exists(cfg_path):
            cfg = pd.read_csv(cfg_path).iloc[0].to_dict()
        rows.append(dict(trial=os.path.basename(tdir), n_seeds=len(vals),
                         val_mae_mean=round(float(np.mean(vals)), 3),
                         val_mae_sd=round(float(np.std(vals, ddof=1)), 3)
                         if len(vals) > 1 else 0.0, **cfg))
    if not rows:
        print("no finished trials yet")
        return None
    d = pd.DataFrame(rows).sort_values("val_mae_mean")
    d.to_csv(os.path.join(base, "ranking.csv"), index=False)
    print("\ntop 10 trials by mean validation MAE (meV):")
    cols = [c for c in ["trial", "val_mae_mean", "val_mae_sd", "n_seeds"]
            if c in d.columns]
    print(d[cols].head(10).to_string(index=False))
    print("\nwrote %s/ranking.csv" % base)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--seeds_per_trial", type=int, default=3)
    ap.add_argument("--model", default="rba", choices=["rba", "sa"])
    ap.add_argument("--space", default="default", choices=["default", "wide"],
                    help="wide adds the architecture elements and loss weights")
    ap.add_argument("--rank_only", action="store_true",
                    help="just re-read finished trials and rank them")
    ap.add_argument("--confirm", action="store_true",
                    help="re-run the best trial at all 17 seeds")
    args = ap.parse_args()

    if args.rank_only:
        rank()
        return

    if args.confirm:
        d = rank()
        if d is None:
            return
        best = d.iloc[0]
        searched = dict(SPACE); searched.update(WIDE_EXTRA)
        params = {k: best[k] for k in searched if k in best}
        print("\nconfirming %s at %d seeds" % (best["trial"], len(C.SEEDS)))
        cmds = []
        for seed in C.SEEDS:
            rd = C.results_dir("hyperparameter_search", "confirm", "seed%d" % seed)
            cmds.append(jobs.train_cmd(args.model, seed, 100, rd,
                                       extra=trial_flags(params)))
        jobs.run_jobs(cmds, "hyperparameter_confirm")
        return

    rng = np.random.default_rng(0)          # the search itself is reproducible
    space = dict(SPACE)
    if args.space == "wide":
        space.update(WIDE_EXTRA)
        print("wide space: %d knobs. A larger space needs more trials to cover;"
              % len(space))
        print("300 is a floor here, not a target.")
    cmds = []
    for t in range(args.trials):
        p = sample(rng, space)
        tdir = C.results_dir("hyperparameter_search", "trial%03d" % t)
        pd.DataFrame([p]).to_csv(os.path.join(tdir, "config.csv"), index=False)
        for seed in C.SEEDS[:args.seeds_per_trial]:
            rd = C.results_dir("hyperparameter_search", "trial%03d" % t, "seed%d" % seed)
            cmds.append(jobs.train_cmd(args.model, seed, 100, rd,
                                       extra=trial_flags(p)))

    print("hyperparameter search: %d trials x %d seeds = %d runs"
          % (args.trials, args.seeds_per_trial, len(cmds)))
    jobs.run_jobs(cmds, "hyperparameter_search")
    rank()


if __name__ == "__main__":
    main()
