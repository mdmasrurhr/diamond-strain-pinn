"""Measure the train-test gap, accuracy on an unseen deformation family, and
accuracy beyond the strain range seen in training.
"""

import os

import numpy as np
import pandas as pd

import config as C
import decide


def memorisation():
    """Train-test gap per label fraction, from the main sweep."""
    df = decide.collect(os.path.join(C.RESULTS, "label_sweep"), ["model", "pct", "seed"])
    if not len(df):
        return pd.DataFrame()
    df["pct_n"] = df.pct.str.replace("pct", "").astype(int)
    g = df.groupby(["model", "pct_n"])[["train_mae_meV", "test_mae_meV"]].mean()
    g = g.reset_index()
    g["gap_meV"] = g.test_mae_meV - g.train_mae_meV
    # A ratio is the more readable form: 1.0 is no gap, 2.0 means test error is
    # twice train error.
    g["ratio"] = (g.test_mae_meV / g.train_mae_meV.clip(lower=1e-9)).round(2)

    print("\nMEMORISATION  (test error against train error)")
    print("  %-10s %6s %12s %12s %10s %8s"
          % ("model", "pct", "train meV", "test meV", "gap meV", "ratio"))
    for r in g.itertuples():
        print("  %-10s %6d %12.2f %12.2f %10.2f %8.2f"
              % (r.model, r.pct_n, r.train_mae_meV, r.test_mae_meV, r.gap_meV, r.ratio))
    print("\n  A ratio near 1 means the model generalises as well as it fits. A")
    print("  large ratio at low label fractions is expected and is not a defect:")
    print("  it is what the physics prior exists to bound.")
    return g


def out_of_sample():
    """Extrapolation, from the split study's holdout runs."""
    root = os.path.join(C.RESULTS, "split_study")
    df = decide.collect(root, ["scheme", "run"])
    if not len(df):
        print("\nno split-study runs found -- run 05_decide_split.py first")
        return pd.DataFrame()

    agg = decide.aggregate(df, ["scheme"])
    base = agg[agg.scheme == "stratified"]
    b = float(base.test_mae_meV_mean.iloc[0]) if len(base) else float("nan")

    print("\nOUT-OF-SAMPLE  (against the in-distribution number, %.2f meV)" % b)
    print("  %-18s %12s %12s  %s" % ("scheme", "meV", "x worse", "what it measures"))
    what = {"group_shear": "unseen family: shear",
            "group_triaxial": "unseen family: triaxial",
            "group_biaxial": "unseen family: biaxial",
            "magnitude_05": "strain beyond |E| = 0.05"}
    rows = []
    for r in agg.itertuples():
        if r.scheme not in what:
            continue
        x = r.test_mae_meV_mean / b if np.isfinite(b) and b > 0 else float("nan")
        print("  %-18s %12.2f %12.1f  %s"
              % (r.scheme, r.test_mae_meV_mean, x, what[r.scheme]))
        rows.append(dict(scheme=r.scheme, mae_meV=round(r.test_mae_meV_mean, 2),
                         times_worse=round(x, 1), measures=what[r.scheme]))
    print("\n  These are the bounds of the claim. A strain-engineering reader wants")
    print("  the second column, and a paper that reports only the in-distribution")
    print("  number has not answered them.")
    return pd.DataFrame(rows)


def per_family():
    """Where the error concentrates, from the per-family columns already logged."""
    df = decide.collect(os.path.join(C.RESULTS, "label_sweep"), ["model", "pct", "seed"])
    if not len(df):
        return pd.DataFrame()
    df = df[df.pct == "100pct"]
    cols = [c for c in df.columns if c.startswith("mae_") and c.endswith("_meV")]
    if not cols:
        return pd.DataFrame()
    g = df.groupby("model")[cols].mean().round(2)
    print("\nERROR BY DEFORMATION FAMILY  (100% labels)")
    print("  %-10s %s" % ("model", " ".join("%12s" % c[4:-4] for c in cols)))
    for model, row in g.iterrows():
        print("  %-10s %s" % (model, " ".join("%12.2f" % row[c] for c in cols)))
    print("\n  The family with the largest error is where the model fails first, and")
    print("  The largest of these is where the model fails first.")
    return g.reset_index()


def main():
    out = C.results_dir("validate_generalization")
    mem = memorisation()
    oos = out_of_sample()
    fam = per_family()
    for name, d in (("memorisation", mem), ("out_of_sample", oos),
                    ("per_family", fam)):
        if len(d):
            d.to_csv(os.path.join(out, "%s.csv" % name), index=False)
    print("\nwrote %s/" % os.path.relpath(out, C.ROOT))


if __name__ == "__main__":
    main()
