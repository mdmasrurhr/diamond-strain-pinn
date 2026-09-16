"""Measure how many epochs a run needs and how many seeds a comparison needs."""

import argparse
import math
import os

import numpy as np
import pandas as pd

import config as C
import decide
import jobs

PCT = 100

# Budgets that land on a learning-rate floor of the adopted schedule.
EPOCH_GRID = [1500, 4000, 6000, 12000, 24000]

# Patience is counted in checkpoints, not epochs: with --ckpt_interval 250, a
# patience of 4 is 1,000 epochs without improvement.
PATIENCE_GRID = [2, 4, 8]


def build_epochs(seeds):
    cmds = []
    for ep in EPOCH_GRID:
        for seed in seeds:
            rd = C.results_dir("budget_study", "epochs", "ep%d" % ep, "seed%d" % seed)
            cmds.append(jobs.train_cmd("rba", seed, PCT, rd,
                                       extra=["--epochs", str(ep)]))
    return cmds


def build_patience(seeds):
    cmds = []
    for pa in PATIENCE_GRID:
        for seed in seeds:
            rd = C.results_dir("budget_study", "patience", "pa%d" % pa, "seed%d" % seed)
            cmds.append(jobs.train_cmd("rba", seed, PCT, rd,
                                       extra=["--epochs", str(max(EPOCH_GRID)),
                                              "--patience", str(pa)]))
    return cmds


def seed_power(alpha=0.05, power=0.80):
    """What effect size do N seeds resolve, and what N does a target need?

    Reads the seed-to-seed spread of whatever runs exist. No training.

    Two-sample Welch comparison at equal n:
        n = 2 ((z_{1-alpha/2} + z_{1-beta}) sigma / epsilon)^2
    which inverts to the effect size resolvable at a given n. An effect smaller
    than the run-to-run sigma is not worth designing for -- it cannot be
    separated from the noise floor of the training process itself.
    """
    # z values for the standard two-sided 5% / 80% convention, hard-coded so the
    # script needs no scipy.
    z_a, z_b = 1.959964, 0.841621
    k = 2.0 * (z_a + z_b) ** 2

    rows = []
    df = decide.collect(os.path.join(C.RESULTS, "label_sweep"),
                        ["model", "pct", "seed"])
    if len(df):
        rows.append(df)
    if not rows:
        print("no sweep runs found to estimate the seed spread from")
        return None
    df = pd.concat(rows)
    df = df[df.pct.astype(str).str.replace("pct", "") == "100"]

    print("\nSEED POWER  (two-sided alpha=%.2f, power=%.2f)" % (alpha, power))
    print("  %-16s %6s %10s %14s %16s"
          % ("model", "n", "sigma meV", "resolves at n", "n for 2 meV"))
    out = []
    for model, g in df.groupby("model"):
        v = g.test_mae_meV.values
        if len(v) < 3:
            continue
        sigma = float(np.std(v, ddof=1))
        n = len(v)
        eps_at_n = sigma * math.sqrt(k / n)        # resolvable difference at this n
        n_for_2 = k * (sigma / 2.0) ** 2           # n needed to resolve 2 meV
        print("  %-16s %6d %10.2f %14.2f %16.1f"
              % (model, n, sigma, eps_at_n, n_for_2))
        out.append(dict(model=model, n_seeds=n, sigma_meV=round(sigma, 2),
                        resolvable_meV_at_n=round(eps_at_n, 2),
                        n_for_2meV=round(n_for_2, 1)))

    print("\n  Read this as: at the seed count actually used, a difference smaller")
    print("  than 'resolves at n' cannot be called real, however suggestive the")
    print("  means look. Any claimed improvement below that number needs more")
    print("  seeds, not more argument.")
    return pd.DataFrame(out)


def report():
    out = C.results_dir("budget_study")

    root = os.path.join(C.RESULTS, "budget_study", "epochs")
    df = decide.collect(root, ["budget", "seed"])
    if len(df):
        df["epochs_set"] = df.budget.str.replace("ep", "").astype(int)
        agg = decide.aggregate(df, ["budget"])

        def binding_share(group):
            """Share of runs whose kept checkpoint was the last one available.

            A run is "budget binding" when the best checkpoint is the final one:
            the model was still improving when training stopped.
            """
            last_window = group.epochs_set - C.TUNED["ckpt_interval"]
            return float((group.best_ckpt_epoch >= last_window).mean())

        bind = df.groupby("budget").apply(binding_share, include_groups=False)
        agg["budget_binding"] = ["%.0f%%" % (100 * bind[b]) for b in agg.budget]
        decide.report(agg, "EPOCH BUDGET", extra_cols=("budget_binding",))
        print("\n  'budget_binding' is the share of runs whose kept checkpoint was")
        print("  the last one available. At 100%% the number reported is a property")
        print("  of the budget; the model had not stopped improving.")

        free = agg[agg.budget_binding == "0%"]
        print("\nDECISION")
        if len(free):
            r = free.iloc[0]
            print("  smallest non-binding budget: %s at %.2f meV -> adopt this"
                  % (r.budget, r.test_mae_meV_mean))
        else:
            print("  every budget tested is still binding -- the grid does not reach")
            print("  convergence and must be extended before a budget can be chosen.")
        agg.to_csv(os.path.join(out, "epoch_decision.csv"), index=False)

    root = os.path.join(C.RESULTS, "budget_study", "patience")
    dfp = decide.collect(root, ["variant", "seed"])
    if len(dfp):
        aggp = decide.aggregate(dfp, ["variant"])
        ran = dfp.groupby("variant").epochs_run.mean().round(0).astype(int)
        aggp["epochs_used"] = [ran[v] for v in aggp.variant]
        decide.report(aggp, "EARLY STOPPING", extra_cols=("epochs_used",))
        print("\n  Early stopping is a compute decision here, not an accuracy one:")
        print("  the best checkpoint is kept either way. It is worth adopting only")
        print("  if it saves epochs without costing meV.")
        aggp.to_csv(os.path.join(out, "patience_decision.csv"), index=False)

    sp = seed_power()
    if sp is not None:
        sp.to_csv(os.path.join(out, "seed_power.csv"), index=False)
        print("\nwrote %s/seed_power.csv" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds_only", action="store_true",
                    help="run only the power analysis (no training)")
    ap.add_argument("--patience", action="store_true",
                    help="run the early-stopping grid as well")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.seeds_only:
        sp = seed_power()
        if sp is not None:
            out = C.results_dir("budget_study")
            sp.to_csv(os.path.join(out, "seed_power.csv"), index=False)
            print("\nwrote %s/seed_power.csv" % out)
        return
    if args.report:
        return report()

    seeds = C.SEEDS if args.full else C.SEEDS[:3]
    cmds = build_epochs(seeds)
    if args.patience:
        cmds += build_patience(seeds)
    print("budget study: %d runs (epoch grid %s%s, %d seeds)"
          % (len(cmds), EPOCH_GRID, " + patience" if args.patience else "",
             len(seeds)))
    if args.dry_run:
        return
    jobs.run_jobs(cmds, "budget_study")
    report()


if __name__ == "__main__":
    main()
