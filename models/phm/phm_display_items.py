#!/usr/bin/env python3
"""
phm_display_items: the tables and figures (display items) of an PHM study.

It only reads what phm.py wrote into the run folders (metrics_summary.csv, log.csv, predictions.csv,
best_model.pt) and the constants files; it never trains. Output:
  <out>/results.txt, summary_by_fraction.csv, all_runs.csv      tables
  <out>/figures/fig01 ... fig12 (.png at 300 dpi and .pdf)       figures, with captions.txt

Usage:  python phm_display_items.py --data_csv dft_hy_v9.csv --out <study folder>
"""

# ======================================================================================================================
# LIBRARIES
# ======================================================================================================================
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

import matplotlib
matplotlib.use("Agg")                     # draw into files; no screen needed
import matplotlib.pyplot as plt

# the model file: data split, model, formulas and settings are taken from it, so tables and figures always match the
# model that produced the runs
from phm import (CONFIG_FILE, DATA, LOSS, NETWORK, PHM, TRAINING, agreement_weight, cbm_analytic, load_and_split,
                 read_config, read_constants, run_folder, vbm_analytic)

# ======================================================================================================================
# FUNCTIONS IN THIS FILE (in the order they appear)
# ======================================================================================================================
# Reading the runs
#   load_summaries(out), load_log(...), load_predictions(...), mean_and_ci(values), median_curve(...)
# Tables
#   write_tables(summaries, out)
# Figure helpers
#   setup_matplotlib(), log_axis(ax), light_grid(ax), save_figure(...), available(...), fraction_colors(n)
# Figures
#   plot_learning_curves, plot_loss_terms, plot_lr_and_gradients            training dynamics
#   plot_parity, plot_residuals, plot_error_ecdf, plot_seed_spread          accuracy
#   plot_per_class, plot_data_efficiency, plot_generalization_gap           subgroups and data efficiency
#   plot_strain_sweeps, plot_constants_vs_correction                        physics and model internals
# All of it
#   make_display_items(data_csv, out), main()

# ======================================================================================================================
# SETTINGS OF THE TABLES AND FIGURES
# ======================================================================================================================
DISPLAY = read_config(CONFIG_FILE, "display")[0]          # the [display] section of config.toml
CONSTANT_SYMBOLS = ["CBM$_0$", r"$\Xi_d$", r"$\Xi_u$", r"$\Xi_u'$", r"$\Xi_q$", r"$\Xi_c$", r"$\Xi_{q2}$",
                    "VBM$_0$", "$a_v$", "$a_{v2}$", "$b$", "$b_2$", "$d$"]          # axis labels of the 13 constants
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_TEST = "#1baf7a"
COLOR_MODEL = "#2a78d6"
COLOR_ANALYTIC = "#6e7781"
COLOR_CONSTANTS = "#2a78d6"
COLOR_CORRECTION = "#eb6834"
CLASS_COLORS = {"isotropic": "#7a5af8", "uniaxial": "#2a78d6", "biaxial": "#1baf7a",
                "triaxial": "#eda100", "shear": "#eb6834"}
TERM_COLORS = {"data": "#2a78d6", "isotropic": "#7a5af8", "agreement": "#1baf7a"}


# ======================================================================================================================
# TABLES AND FIGURES
# ======================================================================================================================
def load_summaries(out):
    files = sorted(glob.glob(os.path.join(out, "runs", "*pct", "seed*", "metrics_summary.csv")))
    if len(files) == 0:
        raise SystemExit("no finished runs under %s" % os.path.join(out, "runs"))
    rows = [pd.read_csv(f).iloc[0] for f in files]
    return pd.DataFrame(rows).sort_values(["fraction", "seed"]).reset_index(drop=True)


def load_log(out, seed, fraction):
    path = os.path.join(run_folder(out, seed, fraction), "log.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_predictions(out, seeds, fraction, split="test"):
    """Predictions of all seeds at one fraction, pooled (only the rows of one split, or all rows if split is None)."""
    tables = []
    for seed in seeds:
        path = os.path.join(run_folder(out, seed, fraction), "predictions.csv")
        if os.path.exists(path):
            p = pd.read_csv(path)
            p["seed"] = seed
            tables.append(p)
    if len(tables) == 0:
        return None
    p = pd.concat(tables, ignore_index=True)
    if split is not None:
        p = p[p["split"] == split]
    return p


def mean_and_ci(values):
    """Mean and half-width of the 95% t interval."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.0
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half_width


def median_curve(out, seeds, fraction, column):
    """Median over seeds of one column of log.csv, indexed by iteration (the first 20 iterations are left out)."""
    curves = []
    for seed in seeds:
        log = load_log(out, seed, fraction)
        if log is not None and column in log.columns:
            curve = log.set_index("iteration")[column]
            curves.append(curve[curve.index > 20])
    if len(curves) == 0:
        return None
    return pd.concat(curves, axis=1).median(axis=1)


def write_tables(summaries, out):
    """results.txt (readable), summary_by_fraction.csv and all_runs.csv."""
    lines = []
    seeds = sorted(summaries["seed"].unique())
    fractions = sorted(summaries["fraction"].unique())
    lines.append("PHM: test results over %d seeds (%s), %d iterations per run"
                 % (len(seeds), ", ".join(str(int(s)) for s in seeds), int(summaries["n_iterations"].max())))
    lines.append("test gap MAE: mean ± sd [95% t interval]. Analytic models = the same constants theta0, no network.")
    lines.append("")
    lines.append("%8s %5s %26s %7s %7s %8s %8s %8s %8s %9s %9s %8s" % (
        "labeled", "seeds", "test gap MAE (meV)", "RMSE", "max", "R2", "CBM MAE", "VBM MAE", "val MAE", "train MAE",
        "best iter", "clipped"))
    table_rows = []
    for fraction in fractions:
        rows = summaries[summaries["fraction"] == fraction]
        mean, half = mean_and_ci(rows["test_mae_meV"])
        mae_text = "%.1f ± %.1f [%.1f, %.1f]" % (mean, rows["test_mae_meV"].std(), mean - half, mean + half)
        lines.append("%7d%% %5d %26s %7.1f %7.1f %8.4f %8.1f %8.1f %8.1f %9.1f %9.0f %7.0f%%" % (
            fraction, len(rows), mae_text, rows["test_rmse_meV"].mean(), rows["test_max_meV"].mean(),
            rows["test_r2"].mean(), rows["test_cbm_mae_meV"].mean(), rows["test_vbm_mae_meV"].mean(),
            rows["val_mae_meV"].mean(), rows["train_mae_meV"].mean(), rows["best_iteration"].median(),
            100 * rows["clipped_share"].mean()))
        difference = rows["test_mae_meV"].values - rows["analytic_test_mae_meV"].values
        d_mean, d_half = mean_and_ci(difference)
        table_rows.append({"fraction": fraction, "n_seeds": len(rows), "test_mae_mean_meV": mean,
                           "test_mae_sd_meV": rows["test_mae_meV"].std(), "test_mae_ci95_meV": half,
                           "minus_analytic_mean_meV": d_mean, "minus_analytic_ci95_meV": d_half})

    lines.append("")
    one_row_per_seed = summaries.drop_duplicates("seed")
    analytic_mean, analytic_half = mean_and_ci(one_row_per_seed["analytic_test_mae_meV"])
    lines.append("analytic models alone (theta0, no network): %.1f meV [%.1f, %.1f]"
                 % (analytic_mean, analytic_mean - analytic_half, analytic_mean + analytic_half))
    lines.append("PHM minus analytic models, paired by seed (negative = PHM better):")
    for row in table_rows:
        low = row["minus_analytic_mean_meV"] - row["minus_analytic_ci95_meV"]
        high = row["minus_analytic_mean_meV"] + row["minus_analytic_ci95_meV"]
        lines.append("  %3d%%: %+6.1f [%+.1f, %+.1f] meV" % (row["fraction"], row["minus_analytic_mean_meV"], low, high))

    lines.append("")
    lines.append("test gap MAE per strain class (meV), mean over seeds")
    lines.append("%9s" % "labeled" + "".join("%11s" % name for name in DATA["strain_classes"]))
    for fraction in fractions:
        rows = summaries[summaries["fraction"] == fraction]
        cells = ""
        for name in DATA["strain_classes"]:
            cells += "%11.1f" % rows["test_mae_" + name + "_meV"].mean()
        lines.append("%8d%%" % fraction + cells)
    cells = ""
    for name in DATA["strain_classes"]:
        cells += "%11.1f" % one_row_per_seed["analytic_mae_" + name + "_meV"].mean()
    lines.append("%9s" % "analytic" + cells)

    with open(os.path.join(out, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(table_rows).to_csv(os.path.join(out, "summary_by_fraction.csv"), index=False)
    summaries.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    print("\n".join(lines))


# ---- small figure helpers
def setup_matplotlib():
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333", "xtick.color": "#555555",
                         "ytick.color": "#555555", "axes.titlesize": 10, "axes.titleweight": "bold",
                         "legend.frameon": False, "savefig.dpi": DISPLAY["figure_dpi"]})


def log_axis(ax):
    """Logarithmic y axis with plain numbers as tick labels (call it after plotting). Ticks at 1, 2, 5 x 10^k when
    the axis spans fewer than 3 decades, otherwise only at 10^k."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    if np.log10(high / low) < 3:
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5), numticks=12))
    else:
        ax.yaxis.set_major_locator(LogLocator(subs=(1,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: "%g" % value))
    ax.yaxis.set_minor_formatter(NullFormatter())


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, figure_folder, name, caption, captions):
    fig.savefig(os.path.join(figure_folder, name + ".png"), facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(figure_folder, name + ".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    captions.append("%s: %s" % (name, caption))


def available(wanted, fractions):
    """The wanted fractions that were run (or all of them if none of the wanted ones were)."""
    chosen = [f for f in wanted if f in fractions]
    if len(chosen) == 0:
        chosen = list(fractions)
    return chosen


def fraction_colors(n):
    """n shades of one blue, light (few labels) to dark (many labels)."""
    return [plt.cm.Blues(value) for value in np.linspace(0.45, 0.95, n)]


# ---- 1. training dynamics
def plot_learning_curves(out, summaries, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["show_fractions"], fractions)
    switch = int(round(TRAINING["switch_fraction"] * summaries["n_iterations"].max()))
    fig, axes = plt.subplots(1, len(shown), figsize=(3.9 * len(shown), 3.5), squeeze=False)
    for ax, fraction in zip(axes[0], shown):
        for column, color, style, label in (("train_mae", COLOR_TRAIN, ":", "training set"),
                                            ("val_mae", COLOR_VAL, "-", "validation"),
                                            ("test_mae", COLOR_TEST, "--", "test")):
            curve = median_curve(out, seeds, fraction, column)
            if curve is not None:
                ax.plot(curve.index / 1000, curve.values * 1000, style, color=color, linewidth=1.3, label=label)
        best = summaries[summaries["fraction"] == fraction]["best_iteration"].median()
        ax.axvline(best / 1000, color="#222222", linewidth=1.0, linestyle="-.", label="kept checkpoint (median)")
        ax.axvline(switch / 1000, color="#999999", linewidth=0.8, linestyle="--", label="AdamW → SOAP")
        log_axis(ax)
        light_grid(ax)
        ax.set_title("%d%% labeled" % fraction, loc="left")
        ax.set_xlabel("iteration (thousands)")
        ax.set_ylabel("gap MAE (meV)")
    axes[0][0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Learning curves (median over seeds)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig01_learning_curves",
                "gap MAE on the training, validation and test states during training (median over seeds); dash-dot = "
                "median iteration of the kept checkpoint; grey dashed = switch from AdamW to SOAP.", captions)


def plot_loss_terms(out, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["show_fractions"], fractions)
    labels = {"data": "data", "isotropic": "isotropic gap", "agreement": "gap agreement"}
    fig, axes = plt.subplots(2, len(shown), figsize=(3.9 * len(shown), 6.2), sharex=True, squeeze=False)
    for j, fraction in enumerate(shown):
        weights = {"data": LOSS["w_data"], "isotropic": LOSS["w_isotropic"], "agreement": agreement_weight(fraction)}
        for term in ["data", "isotropic", "agreement"]:
            curve = median_curve(out, seeds, fraction, term)
            if curve is None or not (curve > 0).any():
                continue                                    # the term is zero at this fraction (data at 0%)
            curve = curve.rolling(DISPLAY["rolling_points"], min_periods=1).median()
            axes[0][j].plot(curve.index / 1000, curve.values, color=TERM_COLORS[term], linewidth=1.2,
                            label="%s (× %g)" % (labels[term], weights[term]))
            axes[1][j].plot(curve.index / 1000, curve.values / weights[term] * 1000, color=TERM_COLORS[term],
                            linewidth=1.2, label=labels[term])
        total = median_curve(out, seeds, fraction, "loss")
        if total is not None:
            total = total.rolling(DISPLAY["rolling_points"], min_periods=1).median()
            axes[0][j].plot(total.index / 1000, total.values, color="#222222", linewidth=1.0, linestyle="--",
                            label="total")
        for i in range(2):
            log_axis(axes[i][j])
            light_grid(axes[i][j])
        axes[0][j].set_title("%d%% labeled" % fraction, loc="left")
        axes[1][j].set_xlabel("iteration (thousands)")
        axes[0][j].legend(fontsize=7)
    axes[0][0].set_ylabel("term as weighted in the loss")
    axes[1][0].set_ylabel("term without its weight (meV)")
    fig.suptitle("Loss terms during training (median over seeds, 50-iteration rolling median)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig02_loss_terms",
                "each loss term during training: top as weighted in the loss, bottom without its weight (a mean "
                "absolute error in meV). A term is not drawn where it is zero (no labels at 0%).", captions)


def plot_lr_and_gradients(out, summaries, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["two_fractions"], fractions)
    fig, axes = plt.subplots(2, len(shown), figsize=(5.0 * len(shown), 5.6), sharex=True, squeeze=False)
    for j, fraction in enumerate(shown):
        log = load_log(out, seeds[0], fraction)
        if log is not None:
            axes[0][j].plot(log["iteration"] / 1000, log["lr"], color="#eda100", linewidth=1.2)
        before = median_curve(out, seeds, fraction, "grad_norm")
        after = median_curve(out, seeds, fraction, "grad_norm_after_clip")
        if before is not None:
            before = before.rolling(DISPLAY["rolling_points"], min_periods=1).median()
            after = after.rolling(DISPLAY["rolling_points"], min_periods=1).median()
            axes[1][j].plot(before.index / 1000, before.values, color=COLOR_TRAIN, linewidth=1.0,
                            label="before clipping")
            axes[1][j].plot(after.index / 1000, after.values, color=COLOR_VAL, linewidth=1.0, linestyle="--",
                            label="after clipping (limit %g)" % TRAINING["clip_norm"])
        clipped = 100 * summaries[summaries["fraction"] == fraction]["clipped_share"].mean()
        axes[1][j].text(0.02, 0.04, "clipping active in %.0f%% of iterations" % clipped,
                        transform=axes[1][j].transAxes, fontsize=8, color="#333333")
        for i in range(2):
            log_axis(axes[i][j])
            light_grid(axes[i][j])
        axes[0][j].set_title("%d%% labeled" % fraction, loc="left")
        axes[1][j].set_xlabel("iteration (thousands)")
    axes[0][0].set_ylabel("learning rate")
    axes[1][0].set_ylabel("gradient norm (L2)")
    axes[1][0].legend(fontsize=8)
    fig.suptitle("Learning-rate schedule (warm restarts; SOAP in the second half) and gradient norms", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig03_lr_gradients",
                "learning rate (first seed) and the L2 gradient norm before and after clipping (median over seeds).",
                captions)


# ---- 2. accuracy
def plot_parity(out, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["show_fractions"], fractions)
    fig, axes = plt.subplots(1, len(shown), figsize=(3.9 * len(shown), 4.0), squeeze=False)
    for ax, fraction in zip(axes[0], shown):
        p = load_predictions(out, seeds, fraction)
        low = min(p["gap_true"].min(), p["gap_pred"].min()) - 0.05
        high = max(p["gap_true"].max(), p["gap_pred"].max()) + 0.05
        ax.plot([low, high], [low, high], color="#999999", linewidth=0.9)
        for class_name in DATA["strain_classes"]:
            rows = p[p["strain_class"] == class_name]
            ax.scatter(rows["gap_true"], rows["gap_pred"], s=6, color=CLASS_COLORS[class_name], alpha=0.75,
                       linewidth=0, label=class_name)
        error = p["gap_pred"] - p["gap_true"]
        r2 = 1.0 - (error ** 2).sum() / ((p["gap_true"] - p["gap_true"].mean()) ** 2).sum()
        ax.text(0.04, 0.96, "MAE %.1f meV\nRMSE %.1f meV\nR² %.4f" % (error.abs().mean() * 1000,
                np.sqrt((error ** 2).mean()) * 1000, r2), transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_title("%d%% labeled" % fraction, loc="left")
        ax.set_xlabel("DFT gap (eV)")
        ax.set_ylabel("predicted gap (eV)")
    axes[0][0].legend(fontsize=7.5, loc="lower right", markerscale=2)
    fig.suptitle("Predicted vs DFT gap on the test states (all seeds pooled)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig04_parity", "predicted vs DFT gap of the test states; colour = strain class; "
                "diagonal = perfect prediction.", captions)


def plot_residuals(out, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["two_fractions"], fractions)
    fig, axes = plt.subplots(len(shown), 2, figsize=(10.5, 3.5 * len(shown)), squeeze=False)
    for i, fraction in enumerate(shown):
        p = load_predictions(out, seeds, fraction)
        residual = (p["gap_pred"] - p["gap_true"]).values * 1000
        for j, (column, label) in enumerate((("gap_true", "DFT gap (eV)"),
                                             ("max_abs_strain", "largest strain component |ε|"))):
            ax = axes[i][j]
            for class_name in DATA["strain_classes"]:
                in_class = (p["strain_class"] == class_name).values
                ax.scatter(p[column].values[in_class], residual[in_class], s=6, color=CLASS_COLORS[class_name],
                           alpha=0.7, linewidth=0, label=class_name)
            ax.axhline(0, color="#999999", linewidth=0.9)
            order = np.argsort(p[column].values)
            window = max(5, len(order) // 25)
            x_sorted = pd.Series(p[column].values[order])
            y_sorted = pd.Series(residual[order])
            ax.plot(x_sorted.rolling(window, center=True).median(), y_sorted.rolling(window, center=True).median(),
                    color="#222222", linewidth=1.2, label="running median")
            ax.set_xlabel(label)
            ax.set_ylabel("predicted − DFT gap (meV)")
            ax.set_title("%d%% labeled" % fraction, loc="left")
            light_grid(ax)
    axes[0][0].legend(fontsize=7.5, ncol=3, markerscale=2)
    fig.suptitle("Test residuals vs gap and vs strain size (all seeds pooled)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig05_residuals", "residuals of the test states against the DFT gap and the "
                "largest strain component; black = running median (a trend shows a systematic bias).", captions)


def plot_error_ecdf(out, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["show_fractions"], fractions)
    colors = fraction_colors(len(shown))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for fraction, color in zip(shown, colors):
        p = load_predictions(out, seeds, fraction)
        error = np.sort(np.abs(p["gap_pred"] - p["gap_true"]).values * 1000)
        share = np.arange(1, len(error) + 1) / len(error)
        ax.plot(error, share, color=color, linewidth=1.5,
                label="%d%% labeled (95%% below %.0f meV)" % (fraction, np.percentile(error, 95)))
    p = load_predictions(out, seeds, shown[0])
    error = np.sort(np.abs(p["gap_analytic"] - p["gap_true"]).values * 1000)
    ax.plot(error, np.arange(1, len(error) + 1) / len(error), color=COLOR_ANALYTIC, linewidth=1.5, linestyle="--",
            label="analytic models θ₀ (95%% below %.0f meV)" % np.percentile(error, 95))
    ax.axhline(0.95, color="#bbbbbb", linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("|predicted − DFT gap| (meV)")
    ax.set_ylabel("share of test states")
    light_grid(ax)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Error distribution of the test states (all seeds pooled)", loc="left")
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig06_error_ecdf", "cumulative distribution of the absolute test errors, pooled over seeds "
                "(each seed has its own test set, so a state can appear once per seed in which it is a test state); the "
                "legend gives the error below which 95% of the pooled test errors fall.", captions)


def plot_seed_spread(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(1.0 * len(fractions) + 2.5, 4.0))
    for k, fraction in enumerate(fractions):
        values = summaries[summaries["fraction"] == fraction]["test_mae_meV"].values
        ax.boxplot([values], positions=[k], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": COLOR_MODEL},
                   medianprops={"color": COLOR_MODEL, "linewidth": 1.6},
                   whiskerprops={"color": COLOR_MODEL}, capprops={"color": COLOR_MODEL})
        offsets = np.linspace(-0.15, 0.15, len(values))
        ax.scatter(np.full(len(values), k) + offsets, values, s=10, color=COLOR_MODEL, zorder=3)
    analytic = summaries.drop_duplicates("seed")["analytic_test_mae_meV"].mean()
    ax.axhline(analytic, color=COLOR_ANALYTIC, linewidth=1.2, linestyle="--", label="analytic models θ₀ (mean)")
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(["%d%%" % f for f in fractions])
    ax.set_xlabel("labeled training states")
    ax.set_ylabel("test gap MAE per seed (meV)")
    log_axis(ax)
    light_grid(ax)
    ax.legend(fontsize=8)
    ax.set_title("Seed-to-seed spread of the test error", loc="left")
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig07_seed_spread", "test gap MAE of every seed (dots) with box plots per "
                "labeled fraction.", captions)


# ---- 3. subgroups, data efficiency, generalization gap
def plot_per_class(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.array(fractions, dtype=float)
    one_row_per_seed = summaries.drop_duplicates("seed")
    smallest_mean = np.inf
    for class_name in DATA["strain_classes"]:
        means = []
        halves = []
        for fraction in fractions:
            mean, half = mean_and_ci(summaries[summaries["fraction"] == fraction]["test_mae_" + class_name + "_meV"])
            means.append(mean)
            halves.append(half)
        means = np.array(means)
        halves = np.array(halves)
        smallest_mean = min(smallest_mean, np.nanmin(means))
        color = CLASS_COLORS[class_name]
        ax.plot(x, means, "o-", color=color, linewidth=1.5, markersize=4, label=class_name)
        ax.fill_between(x, means - halves, means + halves, color=color, alpha=0.08, linewidth=0)
        analytic = one_row_per_seed["analytic_mae_" + class_name + "_meV"].mean()
        ax.axhline(analytic, color=color, linewidth=0.9, linestyle=":", alpha=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, x.max() * 1.25)               # no empty space left of 0 %
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % f for f in fractions])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.6 * smallest_mean)        # the axis follows the means, not the interval bands
    log_axis(ax)
    light_grid(ax)
    ax.set_xlabel("labeled training states (%)")
    ax.set_ylabel("test gap MAE (meV)")
    ax.legend(fontsize=8, title="strain class", title_fontsize=8)
    ax.set_title("Test error per strain class (band = 95% interval; dotted = analytic models θ₀)", loc="left",
                 fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig08_per_class", "test gap MAE per strain class vs the labeled fraction; bands "
                "are 95% t intervals over seeds; dotted lines are the analytic models with θ₀ for each class.", captions)


def plot_data_efficiency(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    x = np.array(fractions, dtype=float)
    means = []
    halves = []
    for fraction in fractions:
        mean, half = mean_and_ci(summaries[summaries["fraction"] == fraction]["test_mae_meV"])
        means.append(mean)
        halves.append(half)
    means = np.array(means)
    halves = np.array(halves)
    ax.plot(x, means, "o-", color=COLOR_MODEL, linewidth=1.7, markersize=5, label="PHM")
    ax.fill_between(x, means - halves, means + halves, color=COLOR_MODEL, alpha=0.15, linewidth=0)
    analytic_mean, analytic_half = mean_and_ci(summaries.drop_duplicates("seed")["analytic_test_mae_meV"])
    ax.axhline(analytic_mean, color=COLOR_ANALYTIC, linewidth=1.4, linestyle="--", label="analytic models θ₀")
    ax.axhspan(analytic_mean - analytic_half, analytic_mean + analytic_half, color=COLOR_ANALYTIC, alpha=0.1)
    for x_value, mean in zip(x, means):
        ax.annotate("%.1f" % mean, (x_value, mean), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, x.max() * 1.25)               # no empty space left of 0 %
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % f for f in fractions])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.6 * np.nanmin(means))     # the axis follows the means, not the interval band
    log_axis(ax)
    light_grid(ax)
    ax.set_xlabel("labeled training states (%)")
    ax.set_ylabel("test gap MAE (meV)")
    ax.legend(fontsize=8.5)
    ax.set_title("Data efficiency (mean over seeds; band = 95% t interval)", loc="left")
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig09_data_efficiency", "test gap MAE vs the labeled fraction, with the analytic "
                "models (θ₀, no network) as reference.", captions)


def plot_generalization_gap(summaries, fractions, figure_folder, captions):
    fig, ax = plt.subplots(figsize=(1.1 * len(fractions) + 2.5, 4.0))
    bar_width = 0.26
    for k, (column, color, label) in enumerate((("train_mae_meV", COLOR_TRAIN, "training set"),
                                                ("val_mae_meV", COLOR_VAL, "validation (kept checkpoint)"),
                                                ("test_mae_meV", COLOR_TEST, "test"))):
        means = []
        halves = []
        for fraction in fractions:
            mean, half = mean_and_ci(summaries[summaries["fraction"] == fraction][column])
            means.append(mean)
            halves.append(half)
        positions = np.arange(len(fractions)) + (k - 1) * bar_width
        ax.bar(positions, means, bar_width * 0.92, yerr=halves, color=color, label=label,
               error_kw={"linewidth": 0.8, "capsize": 1.5})
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(["%d%%" % f for f in fractions])
    ax.set_xlabel("labeled training states")
    ax.set_ylabel("gap MAE (meV)")
    log_axis(ax)
    light_grid(ax)
    ax.legend(fontsize=8)
    ax.set_title("Training / validation / test error of the kept model (mean, 95% interval)", loc="left",
                 fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig10_generalization_gap", "gap MAE of the kept checkpoint on the training, "
                "validation and test states.", captions)


# ---- 4. physical consistency and internals
def plot_strain_sweeps(data_csv, out, seeds, fractions, figure_folder, captions):
    table = pd.read_csv(data_csv)
    paths = [("isotropic", table["strain_type"] == "isotropic", "E_xx", "ε_xx = ε_yy = ε_zz"),
             ("uniaxial [100]", table["strain_type"] == "uniaxial_x", "E_xx", "ε_xx"),
             ("biaxial (001)", table["strain_type"] == "biaxial_xy", "E_xx", "ε_xx = ε_yy")]
    if "shear_planes" in table.columns:
        paths.append(("shear xy", table["shear_planes"].astype(str) == "xy", "E_xy", "ε_xy"))
    shown = available(DISPLAY["two_fractions"], fractions)
    seed = seeds[0]
    constants = read_constants(out, seed)
    theta0 = [float(constants[name]) for name in NETWORK["constant_names"]]
    delta_x = float(constants["DELTA_X"])
    fig, axes = plt.subplots(len(shown), len(paths), figsize=(4.3 * len(paths), 3.6 * len(shown)), squeeze=False)
    for i, fraction in enumerate(shown):
        data = load_and_split(data_csv, seed, fraction)
        model = PHM(data["scaler"], theta0, delta_x)
        model.load_state_dict(torch.load(os.path.join(run_folder(out, seed, fraction), "best_model.pt")))
        model.eval()
        for j, (name, on_path, parameter, label) in enumerate(paths):
            ax = axes[i][j]
            states = table[on_path]
            if len(states) < 3:
                ax.axis("off")
                continue
            t_values = states[parameter].values
            strains = states[DATA["strain_columns"]].values
            # the states of a path lie on a straight line in strain space: fit that line, then sample it finely
            t_line = np.linspace(t_values.min(), t_values.max(), 300)
            line = np.zeros((len(t_line), 6), dtype=np.float32)
            for c in range(6):
                slope, intercept = np.polyfit(t_values, strains[:, c], 1)
                line[:, c] = slope * t_line + intercept
            line_tensor = torch.tensor(line)
            x_line = torch.tensor((line - data["scaler"].mean_) / data["scaler"].scale_, dtype=torch.float32)
            with torch.no_grad():
                gap_analytic = (cbm_analytic(line_tensor, theta0, delta_x)
                                - vbm_analytic(line_tensor, theta0)).numpy()
                gap_model = model(x_line)[2].numpy()
            ax.plot(t_line, gap_analytic, color=COLOR_ANALYTIC, linewidth=1.4, linestyle="--",
                    label="analytic models θ₀")
            ax.plot(t_line, gap_model, color=COLOR_MODEL, linewidth=1.6, label="PHM")
            split = data["split"][states.index.values]
            for kind, marker, face, kind_label in (("test", "o", "#222222", "DFT, test"),
                                                   ("train", "o", "white", "DFT, training"),
                                                   ("val", "D", "white", "DFT, validation")):
                on = split == kind
                ax.scatter(t_values[on], states["Eg_eV"].values[on], s=18, marker=marker, facecolor=face,
                           edgecolor="#222222", linewidth=0.8, zorder=4, label=kind_label)
            ax.set_xlabel(label)
            ax.set_ylabel("gap (eV)")
            ax.set_title("%s · %d%% labeled" % (name, fraction), loc="left")
            light_grid(ax)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Gap along continuous strain paths (seed %d) with the DFT states on each path" % seed, fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig11_strain_sweeps", "predicted gap along four strain paths (lines) and the DFT "
                "states on them (filled = test, open = training, diamond = validation).", captions)


def plot_constants_vs_correction(out, seeds, fractions, figure_folder, captions):
    shown = available(DISPLAY["two_fractions"], fractions)
    fig, axes = plt.subplots(2, len(shown), figsize=(6.6 * len(shown), 7.6), squeeze=False)
    for j, fraction in enumerate(shown):
        p = load_predictions(out, seeds, fraction)
        ax = axes[0][j]
        changes = [p["relative_change_" + name].values for name in NETWORK["constant_names"]]
        boxes = ax.boxplot(changes, showfliers=False, patch_artist=True, widths=0.6)
        for k, box in enumerate(boxes["boxes"]):
            if k < 7:
                box.set_facecolor("#e8f1fb")
                box.set_edgecolor(COLOR_CONSTANTS)
            else:
                box.set_facecolor("#fcedf3")
                box.set_edgecolor("#e87ba4")
        for median_line in boxes["medians"]:
            median_line.set_color("#222222")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_xticks(range(1, 14))
        ax.set_xticklabels(CONSTANT_SYMBOLS, fontsize=8.5)
        ax.set_ylabel("(θ − θ₀) / max(|θ₀|, 1)  (%)")
        ax.set_title("%d%% labeled: change of each constant (blue = conduction, pink = valence)" % fraction,
                     loc="left", fontsize=9.5)
        light_grid(ax)

        ax = axes[1][j]
        from_constants = p["gap_from_constants"].values * 1000
        from_correction = p["gap_from_correction"].values * 1000
        low = min(from_constants.min(), from_correction.min())
        high = max(from_constants.max(), from_correction.max()) + 1e-9
        bins = np.linspace(low, high, 60)
        ax.hist(from_constants, bins=bins, color=COLOR_CONSTANTS, alpha=0.6,
                label="from the adjusted constants (mean |·| %.1f meV)" % np.abs(from_constants).mean())
        ax.hist(from_correction, bins=bins, color=COLOR_CORRECTION, alpha=0.6,
                label="from the free correction (mean |·| %.1f meV)" % np.abs(from_correction).mean())
        ax.set_xlabel("change of the gap relative to the untrained model (meV)")
        ax.set_ylabel("test states")
        ax.set_title("%d%% labeled: how the two paths share the work" % fraction, loc="left", fontsize=9.5)
        ax.legend(fontsize=8)
        light_grid(ax)
    fig.suptitle("Inside PHM: learned constant changes and constants vs correction (test states, all seeds)",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, figure_folder, "fig12_constants_vs_correction", "top: change of each constant relative to θ₀ "
                "over the test states; bottom: the part of the gap change that comes from the adjusted constants and "
                "the part that comes from the free correction.", captions)


def make_display_items(data_csv, out):
    setup_matplotlib()
    summaries = load_summaries(out)
    seeds = [int(s) for s in sorted(summaries["seed"].unique())]
    fractions = [int(f) for f in sorted(summaries["fraction"].unique())]
    write_tables(summaries, out)
    figure_folder = os.path.join(out, "figures")
    os.makedirs(figure_folder, exist_ok=True)
    captions = []
    plot_learning_curves(out, summaries, seeds, fractions, figure_folder, captions)
    plot_loss_terms(out, seeds, fractions, figure_folder, captions)
    plot_lr_and_gradients(out, summaries, seeds, fractions, figure_folder, captions)
    plot_parity(out, seeds, fractions, figure_folder, captions)
    plot_residuals(out, seeds, fractions, figure_folder, captions)
    plot_error_ecdf(out, seeds, fractions, figure_folder, captions)
    plot_seed_spread(summaries, fractions, figure_folder, captions)
    plot_per_class(summaries, fractions, figure_folder, captions)
    plot_data_efficiency(summaries, fractions, figure_folder, captions)
    plot_generalization_gap(summaries, fractions, figure_folder, captions)
    plot_strain_sweeps(data_csv, out, seeds, fractions, figure_folder, captions)
    plot_constants_vs_correction(out, seeds, fractions, figure_folder, captions)
    with open(os.path.join(figure_folder, "captions.txt"), "w") as f:
        f.write("\n".join(captions) + "\n")
    print("tables: %s   figures: %s" % (os.path.join(out, "results.txt"), figure_folder))


# ======================================================================================================================
# COMMAND LINE
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser(description="tables and figures of an PHM study (see the top of this file)")
    parser.add_argument("--data_csv", required=True, help="the dataset the study was run on")
    parser.add_argument("--out", required=True, help="study folder (the --out of phm.py)")
    args = parser.parse_args()
    make_display_items(args.data_csv, args.out)


if __name__ == "__main__":
    main()
