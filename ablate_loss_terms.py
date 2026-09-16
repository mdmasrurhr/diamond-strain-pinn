"""
ablate_loss_terms.py -- remove one loss term at a time and measure what it cost.

The training loss is a sum of terms: supervised band-edge errors, physics
residuals against the analytic model, a gap-consistency term, an anchor holding
the bands at their unstrained values, and slope conditions at zero strain. This
script switches each off in turn and retrains.

The thesis ran this at full data only. That turns out to be the least
informative place to run it: the physics terms are worth little when labels are
plentiful and a great deal when they are scarce, so an ablation at 100% labels
understates them. Here every variant is repeated at four fractions, which is
what makes the result interpretable.

    python ablate_loss_terms.py
    python ablate_loss_terms.py --fractions 1 100    # a subset

Results in results/loss_ablation/<variant>/<pct>pct/seed<n>/.
"""

import argparse

import config as C
import jobs

# name -> the --drop_loss flags that define it. "full" drops nothing.
VARIANTS = {
    "full":          [],
    "drop_cbm":      ["--drop_loss", "cbm"],
    "drop_vbm":      ["--drop_loss", "vbm"],
    "drop_cbm_vbm":  ["--drop_loss", "cbm", "--drop_loss", "vbm"],
    "drop_cons":     ["--drop_loss", "cons"],
    "drop_anchor":   ["--drop_loss", "anchor"],
    "drop_slopes":   ["--drop_loss", "pcbm", "--drop_loss", "pvbm"],
    "physics_free":  ["--drop_loss", "cbm", "--drop_loss", "vbm",
                      "--drop_loss", "cons", "--drop_loss", "anchor",
                      "--drop_loss", "pcbm", "--drop_loss", "pvbm"],
}

# Where the physics matters most, and least. 0% is excluded: with no labels and
# no physics there is nothing to train on, so the variants are not comparable.
FRACTIONS = [1, 5, 25, 100]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="rba", choices=["rba", "sa"],
                    help="which weighting to ablate on (default rba)")
    ap.add_argument("--fractions", nargs="+", type=int, default=FRACTIONS)
    ap.add_argument("--variants", nargs="+", default=sorted(VARIANTS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[:3] if args.quick else C.SEEDS
    cmds = []
    for variant in args.variants:
        for pct in args.fractions:
            for seed in seeds:
                rd = C.results_dir("loss_ablation", variant, "%dpct" % pct,
                                   "seed%d" % seed)
                cmds.append(jobs.train_cmd(args.model, seed, pct, rd,
                                           extra=VARIANTS[variant]))

    print("loss ablations: %d variants x %d fractions x %d seeds"
          % (len(args.variants), len(args.fractions), len(seeds)))
    jobs.run_jobs(cmds, "loss_ablation")


if __name__ == "__main__":
    main()
