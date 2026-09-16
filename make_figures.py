"""
make_figures.py -- every figure for the paper, from the collected results.

One script, one style module, one colour per model. The thesis had eight figure
scripts each setting their own rcParams and one assigning colour by position in a
loop, so the same model changed colour between figures. Here `style.py` decides
the look and `style.series(model)` decides the colour, keyed by name.

Each figure below states the claim it supports. A figure that supports no claim
should not be drawn, and a claim with no figure is an assertion -- so this list
is meant to be read against the paper's claims, not extended for its own sake.

    python make_figures.py                # every figure
    python make_figures.py --only scarcity parity

Writes results/make_figures/*.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import decide
import style


def _sweep():
    for sub in ("label_sweep",):
        df = decide.collect(os.path.join(C.RESULTS, sub), ["model", "pct", "seed"])
        if len(df):
            df["pct_n"] = df.pct.str.replace("pct", "").astype(int)
            return df
    return pd.DataFrame()


def _ci95(g):
    """95% CI of the mean, for a Series or a grouped Series.

    Plotting the standard deviation instead overstates the uncertainty ON THE
    MEAN by sqrt(n); it is the most common error-bar mistake in this kind of
    figure, and it makes real differences look like noise.
    """
    n = g.count()
    sd = g.std()
    n = np.maximum(np.asarray(n, float), 1.0)
    return 1.96 * np.asarray(sd, float) / np.sqrt(n)


# ---------------------------------------------------------------- figures
def fig_scarcity(out):
    """CLAIM: the physics prior buys accuracy where labels are scarce."""
    df = _sweep()
    if not len(df):
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for model, g in df.groupby("model"):
        s = g.groupby("pct_n").test_mae_meV
        m, ci = s.mean(), _ci95(s)
        x = m.index.values.astype(float)
        x = np.where(x == 0, 0.5, x)          # 0% on a log axis
        ax.errorbar(x, m.values, yerr=ci, **style.series(model))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Labelled fraction of the training pool (%)")
    ax.set_ylabel("Test MAE (meV)")
    # The claim lives in the left half of this plot, so say so on the figure.
    # The label goes in axes coordinates just under the top spine, where no
    # series runs, rather than at a data coordinate that the curves cross.
    ax.axvspan(0.4, 5, color=style.HIGHLIGHT, alpha=0.06, zorder=0)
    ax.text(0.02, 1.02, "shaded: scarce-label regime", transform=ax.transAxes,
            fontsize=8, color=style.REFERENCE, alpha=0.85, va="bottom")
    # Curves run from upper-left to lower-right, so the upper-right corner is
    # the one empty region; the seed count sits in the other one.
    ax.legend(ncol=1, loc="upper right")
    style.annotate_n(ax, int(df.groupby(["model", "pct_n"]).size().median()),
                     "lower left")
    return fig


def fig_parity(out):
    """CLAIM: predictions track DFT across the whole gap range, not just the bulk."""
    src = None
    for sub in ("label_sweep",):
        p = os.path.join(C.RESULTS, sub, "sa", "100pct", "seed42", "test_results.csv")
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return None
    d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for fam, g in d.groupby("strain_type"):
        ax.scatter(g.Eg_true_eV, g.Eg_pred_eV, s=12, alpha=0.75,
                   color=style.family_color(fam), label=str(fam), linewidths=0)
    lo = min(d.Eg_true_eV.min(), d.Eg_pred_eV.min())
    hi = max(d.Eg_true_eV.max(), d.Eg_pred_eV.max())
    ax.plot([lo, hi], [lo, hi], color=style.REFERENCE, lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("HSE06 bandgap (eV)")
    ax.set_ylabel("Predicted bandgap (eV)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, ncol=2)
    mae = (d.Eg_true_eV - d.Eg_pred_eV).abs().mean() * 1000
    ax.text(0.04, 0.95, "MAE %.1f meV" % mae, transform=ax.transAxes,
            va="top", fontsize=8, color=style.REFERENCE)
    return fig


def fig_residuals(out):
    """CLAIM: the error is unbiased -- it is scatter, not a systematic offset."""
    src = None
    for sub in ("label_sweep",):
        p = os.path.join(C.RESULTS, sub, "sa", "100pct", "seed42", "test_results.csv")
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return None
    d = pd.read_csv(src)
    r = (d.Eg_pred_eV - d.Eg_true_eV) * 1000.0
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    sc = ax.scatter(d.Eg_true_eV, r, c=r, cmap=style.DIVERGING,
                    norm=style.diverging_norm(r), s=14, linewidths=0)
    ax.axhline(0, color=style.REFERENCE, lw=0.9)
    ax.set_xlabel("HSE06 bandgap (eV)")
    ax.set_ylabel("Residual (meV)")
    fig.colorbar(sc, ax=ax, label="Residual (meV)")
    ax.text(0.02, 0.94, "mean %+.1f meV" % r.mean(), transform=ax.transAxes,
            va="top", fontsize=8, color=style.REFERENCE)
    return fig


def fig_family(out):
    """CLAIM: accuracy is not uniform -- shear is where the model fails first."""
    df = _sweep()
    if not len(df):
        return None
    df = df[df.pct_n == 100]
    cols = [c for c in df.columns if c.startswith("mae_") and c.endswith("_meV")]
    if not cols:
        return None
    fams = [c[4:-4] for c in cols]
    models = sorted(df.model.unique())
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    w = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        g = df[df.model == model]
        vals = [g[c].mean() for c in cols]
        errs = [_ci95(g[c]) for c in cols]
        ax.bar(np.arange(len(fams)) + i * w, vals, w * 0.9, yerr=errs,
               color=style.series(model)["color"],
               label=style.series(model)["label"])
    ax.set_xticks(np.arange(len(fams)) + 0.4 - w / 2)
    ax.set_xticklabels(fams, rotation=15)
    ax.set_ylabel("Test MAE (meV)")
    ax.set_yscale("log")
    ax.legend(ncol=2)
    style.annotate_n(ax, int(df.groupby("model").size().median()), "upper left")
    return fig


def fig_symmetry(out):
    """CLAIM: raw inputs do not give cubic symmetry; invariant inputs do."""
    import glob
    fs = sorted(glob.glob(os.path.join(
        C.RESULTS, "..", "..", "DIA_THESIS_CODEBASE", "results", "diagnostics",
        "deep_probe", "ofair", "ofair_seed*.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    g = d.groupby("model")[["std_mae_meV", "fair_mae_meV"]].mean().sort_values(
        "fair_mae_meV")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = np.arange(len(g))
    ax.barh(y - 0.19, g.std_mae_meV, 0.36, color=style.HIGHLIGHT,
            label="sampled orientation")
    ax.barh(y + 0.19, g.fair_mae_meV, 0.36, color=style.MUTED,
            label="all 48 cubic images")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index)
    ax.set_xlabel("Test MAE (meV)")
    ax.set_xscale("log")
    ax.legend()
    return fig


FIGURES = {
    "scarcity": fig_scarcity,
    "parity": fig_parity,
    "residuals": fig_residuals,
    "family": fig_family,
    "symmetry": fig_symmetry,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=sorted(FIGURES),
                    choices=sorted(FIGURES))
    ap.add_argument("--context", default="paper", choices=["paper", "slides"])
    args = ap.parse_args()

    style.apply(args.context)
    out = C.results_dir("make_figures")
    made = 0
    for name in args.only:
        fig = FIGURES[name](out)
        if fig is None:
            print("  %-12s no input data -- skipped" % name)
            continue
        p = os.path.join(out, "fig_%s.pdf" % name)
        fig.savefig(p)
        plt.close(fig)
        print("  %-12s -> %s" % (name, os.path.relpath(p, C.ROOT)))
        made += 1
    print("\n%d figures in %s" % (made, os.path.relpath(out, C.ROOT)))


if __name__ == "__main__":
    main()
