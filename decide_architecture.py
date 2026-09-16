"""
decide_architecture.py -- decide every element of the network, by measurement.

The thesis network is 5 x 100 SiLU with Xavier init and mean-initialised output
biases. Four of those were screened and are right. Four more elements were never
varied at all -- normalisation, dropout, residual connections, and the fact that
every hidden layer has the same width -- so "we do not use them" rested on
nothing. This script measures all of it, in two passes.

    capacity     depth x width. Applies the parsimony rule explicitly: the
                 smallest model within one standard deviation of the best. The
                 thesis stated this rule and then did not follow it.
    elements     the four that were never varied, one at a time from the
                 reference, so each difference is attributable.

An element that loses here is not "out of scope" -- it is measured and rejected,
which is a different and defensible claim.

    python decide_architecture.py                 # both passes, 3 seeds
    python decide_architecture.py --pass capacity
    python decide_architecture.py --full          # 17 seeds
    python decide_architecture.py --report

Results in results/architecture_study/<pass>/<variant>/seed<n>/.
"""

import argparse
import os

import config as C
import decide
import jobs

PCT = 100

# Depth matters more than width in every sweep run so far, so the grid is
# deliberately denser in depth and includes narrow-but-deep configurations that
# a width-first search would never reach.
CAPACITY = {}
for _d in [3, 4, 5, 6, 7, 8]:
    for _w in [32, 64, 100, 200]:
        CAPACITY["d%d_w%d" % (_d, _w)] = ["--depth", str(_d), "--width", str(_w)]

ELEMENTS = {
    "ref":            [],
    # Normalisation. BatchNorm is not in this list: the anchor boundary condition
    # evaluates the network on the single unstrained state, and BatchNorm needs
    # a within-batch variance it does not have. train_pinn.py refuses it with
    # that reason rather than failing obscurely -- an exclusion on grounds, not
    # an omission.
    "layernorm":      ["--norm", "layer"],
    # Dropout. The network trains 6,000 epochs on 938 samples, which is the
    # regime where a regulariser usually earns its place -- but it also perturbs
    # the physics residuals, which are evaluated on the same forward pass.
    "dropout_005":    ["--dropout", "0.05"],
    "dropout_010":    ["--dropout", "0.10"],
    "dropout_020":    ["--dropout", "0.20"],
    # Residual connections, possible only between equal-width layers.
    "residual":       ["--skip", "residual"],
    "residual_norm":  ["--skip", "residual", "--norm", "layer"],
    # Non-uniform widths at matched depth.
    "taper":          ["--width_shape", "taper"],
    "widen":          ["--width_shape", "widen"],
}


def build(seeds, which):
    cmds = []
    grids = {"capacity": CAPACITY, "elements": ELEMENTS}
    for name in which:
        for variant, extra in grids[name].items():
            for seed in seeds:
                rd = C.results_dir("architecture_study", name, variant, "seed%d" % seed)
                cmds.append(jobs.train_cmd("rba", seed, PCT, rd, extra=extra))
    return cmds


def _report_pass(name, cost_col):
    root = os.path.join(C.RESULTS, "architecture_study", name)
    df = decide.collect(root, ["variant", "seed"])
    if not len(df):
        print("\n[%s] no runs found" % name)
        return None
    agg = decide.aggregate(df, ["variant"])
    # carry the cost column through, averaged (it is identical across seeds)
    cost = df.groupby("variant")[cost_col].mean().round(0).astype(int)
    agg[cost_col] = [cost[v] for v in agg.variant]
    decide.report(agg, "ARCHITECTURE -- %s" % name.upper(),
                  extra_cols=(cost_col,))

    chosen, best = decide.parsimony(agg, cost_col)
    print("\n  best performer      : %-14s %8.2f meV  %8d %s"
          % (best.variant, best.test_mae_meV_mean, best[cost_col], cost_col))
    print("  parsimony rule says : %-14s %8.2f meV  %8d %s"
          % (chosen.variant, chosen.test_mae_meV_mean, chosen[cost_col], cost_col))
    ref = agg[agg.variant.isin(["d5_w100", "ref"])]
    if len(ref):
        r = ref.iloc[0]
        print("  thesis network      : %-14s %8.2f meV  %8d %s"
              % (r.variant, r.test_mae_meV_mean, r[cost_col], cost_col))
        print("  thesis follows the rule: %s"
              % ("yes" if r.variant == chosen.variant else "NO"))
    return agg, df, chosen


def report():
    out = C.results_dir("architecture_study")
    cap = _report_pass("capacity", "n_params")
    ele = _report_pass("elements", "n_params")

    if ele is not None:
        agg, df, _ = ele
        print("\n  Each element against the reference, at the seeds actually run:")
        ref_vals = df[df.variant == "ref"].test_mae_meV.values
        for r in agg.itertuples():
            if r.variant == "ref":
                continue
            v = df[df.variant == r.variant].test_mae_meV.values
            p = decide.welch(v, ref_vals)
            d = r.test_mae_meV_mean - agg[agg.variant == "ref"].test_mae_meV_mean.iloc[0]
            verdict = ("ADOPT" if (d < 0 and p < 0.05) else
                       "reject" if (d > 0 and p < 0.05) else "no evidence either way")
            print("    %-16s %+8.2f meV  p=%.3f  -> %s" % (r.variant, d, p, verdict))
        print("\n  An element with 'no evidence either way' is not adopted: it costs")
        print("  a parameter to explain and buys nothing measurable. That is a")
        print("  measured exclusion, which is what the paper should say.")

    for name, res in (("capacity", cap), ("elements", ele)):
        if res is not None:
            res[0].to_csv(os.path.join(out, "%s_decision.csv" % name), index=False)
    print("\nwrote %s/{capacity,elements}_decision.csv" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass", dest="which", nargs="+",
                    default=["capacity", "elements"],
                    choices=["capacity", "elements"])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()

    seeds = C.SEEDS if args.full else C.SEEDS[:3]
    cmds = build(seeds, args.which)
    print("architecture study: %s, %d seeds -> %d runs"
          % ("+".join(args.which), len(seeds), len(cmds)))
    if args.dry_run:
        return
    jobs.run_jobs(cmds, "architecture_study")
    report()


if __name__ == "__main__":
    main()
