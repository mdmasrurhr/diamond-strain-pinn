"""
run_label_sweep.py -- the headline experiment: accuracy vs labelled data.

Trains every model at every labelled fraction, at every seed. This is the run
that produces the data-efficiency curve and the full-data accuracy numbers;
everything in 06-11 is a variation on it.

    python run_label_sweep.py              # the whole grid
    python run_label_sweep.py --quick      # 3 seeds, for a smoke test

Grid
    models     sa, rba  (physics-informed)
               mlp      (same network, physics removed -- the control)
               shi      (the published data-driven baseline)
    fractions  0,1,2,5,10,25,50,75,100 percent of the 938-state training pool
    seeds      42-58

The 0% fraction trains on the physics terms alone, so it applies only to the
physics-informed models: mlp and shi have nothing to learn from without labels.

Results land in results/label_sweep/<model>/<pct>pct/seed<n>/.
Re-running skips jobs that already finished.
"""

import argparse
import os

import config as C
import jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="3 seeds instead of 17, for a smoke test")
    ap.add_argument("--models", nargs="+", default=["sa", "rba", "mlp", "shi"])
    ap.add_argument("--per_gpu", type=int, default=C.RUNS_PER_GPU)
    args = ap.parse_args()

    seeds = C.SEEDS[:3] if args.quick else C.SEEDS

    cmds = []
    for model in args.models:
        for pct in C.FRACTIONS:
            # Without labels there is nothing to fit unless physics supplies it.
            if pct == 0 and model in ("mlp", "shi"):
                continue
            for seed in seeds:
                rd = C.results_dir("label_sweep", model, "%dpct" % pct,
                                   "seed%d" % seed)
                cmds.append(jobs.train_cmd(model, seed, pct, rd))

    print("sweep: %d models x %d fractions x %d seeds"
          % (len(args.models), len(C.FRACTIONS), len(seeds)))
    jobs.run_jobs(cmds, "label_sweep", per_gpu=args.per_gpu)


if __name__ == "__main__":
    main()
