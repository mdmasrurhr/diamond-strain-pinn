"""Train every model at every labelled-data fraction. This is the main experiment."""

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
