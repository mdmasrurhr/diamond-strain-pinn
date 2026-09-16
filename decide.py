"""
decide.py -- read finished runs back and apply the selection rules. No training.

jobs.py launches runs; this reads their metrics_summary.csv files, aggregates
over seeds, and applies the rule that turns a table of numbers into a choice.
The rules live here rather than in each study script so that "smallest within
one standard deviation of the best" means the same thing everywhere.

Nothing here decides anything a human would disagree with silently: every
function returns the chosen row AND the table it chose from, and the study
scripts print both.
"""

import os

import numpy as np
import pandas as pd


def collect(root, level_names):
    """Every metrics_summary.csv under `root`, one row each.

    `level_names` names the directory levels between root and the run folder,
    e.g. ["variant", "pct", "seed"] for results/design_ablation/relu/100pct/seed42/.
    Those become columns, so the caller can group by them.
    """
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "metrics_summary.csv" not in filenames:
            continue
        rel = os.path.relpath(dirpath, root).split(os.sep)
        if len(rel) != len(level_names):
            continue
        df = pd.read_csv(os.path.join(dirpath, "metrics_summary.csv"))
        if not len(df):
            continue
        r = df.iloc[0].to_dict()
        for name, value in zip(level_names, rel):
            r[name] = value
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate(df, by, metric="test_mae_meV"):
    """mean, sd and n of `metric` for each group, sorted best first."""
    if not len(df):
        return pd.DataFrame()
    g = df.groupby(by, sort=False)[metric].agg(["mean", "std", "count"])
    g = g.rename(columns={"mean": metric + "_mean", "std": metric + "_std",
                          "count": "n_seeds"}).reset_index()
    return g.sort_values(metric + "_mean").reset_index(drop=True)


def parsimony(agg, cost_col, metric="test_mae_meV"):
    """The framework's architecture rule: smallest within 1 s.d. of the best.

    Returns (chosen_row, best_row). `cost_col` is what "smallest" means --
    parameter count, runtime, whatever the study is trading against. A choice
    that is not the best performer needs this rule stated, not assumed, which is
    why both rows come back.
    """
    m, s = metric + "_mean", metric + "_std"
    best = agg.iloc[0]
    band = best[m] + (best[s] if np.isfinite(best[s]) else 0.0)
    within = agg[agg[m] <= band]
    chosen = within.loc[within[cost_col].idxmin()]
    return chosen, best


def welch(a, b):
    """Welch t-test p-value for two unequal-variance samples. scipy-free.

    Used to say whether a difference between two variants is real at the seed
    count actually run, rather than eyeballing overlapping error bars.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    denom = np.sqrt(va + vb)
    if denom == 0:
        return 1.0
    t = (a.mean() - b.mean()) / denom
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    # Normal approximation to the t distribution; exact enough at n >= 3 for the
    # accept/reject calls made here, and avoids a scipy dependency.
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


def shape(values, metric_mean):
    """flat / slope / cliff, from the spread across a screened parameter.

    The distinction matters: a nuisance parameter is only validated by a FLAT
    screen. A value sitting on a cliff is a sensitive parameter that needs
    tuning, not documentation.
    """
    v = np.asarray(metric_mean, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return "single"
    ratio = v.max() / v.min()
    if ratio > 2.0:
        return "CLIFF"
    if ratio > 1.25:
        return "slope"
    return "flat"


def report(agg, title, metric="test_mae_meV", extra_cols=()):
    """Print an aggregate table the same way in every study."""
    m, s = metric + "_mean", metric + "_std"
    key = [c for c in agg.columns if c not in (m, s, "n_seeds") + tuple(extra_cols)]
    print("\n%s" % title)
    head = "  %-26s %10s %8s %6s" % ("", "mean meV", "sd", "n")
    for c in extra_cols:
        head += " %12s" % c
    print(head)
    for r in agg.itertuples():
        name = " / ".join(str(getattr(r, k)) for k in key)
        sd = getattr(r, s)
        line = "  %-26s %10.2f %8s %6d" % (
            name[:26], getattr(r, m),
            ("%.2f" % sd) if np.isfinite(sd) else "-", getattr(r, "n_seeds"))
        for c in extra_cols:
            line += " %12s" % getattr(r, c, "")
        print(line)
