"""Repeat the sweep on distinct strain states only."""

import argparse
import os

import pandas as pd

import config as C
import jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["sa", "rba", "mlp", "shi"])
    ap.add_argument("--fractions", nargs="+", type=int, default=[1, 5, 25, 100])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # ---- build the deduplicated dataset ---------------------------------
    df = pd.read_csv(C.DATA_CSV)
    key = df[C.STRAIN_COLS].round(9).apply(tuple, axis=1)
    before = len(df)
    # Keep the first occurrence. The copies carry identical labels, so which one
    # survives cannot matter -- 01 checks that and reports the spread.
    dedup = df[~key.duplicated(keep="first")].reset_index(drop=True)
    out_csv = os.path.join(C.ROOT, "data", "dft_hy_v7_dedup.csv")
    dedup.to_csv(out_csv, index=False)
    print("deduplicated: %d rows -> %d distinct states (%d removed)"
          % (before, len(dedup), before - len(dedup)))

    # ---- re-run the headline grid on it ---------------------------------
    seeds = C.SEEDS[:3] if args.quick else C.SEEDS
    cmds = []
    for model in args.models:
        for pct in args.fractions:
            for seed in seeds:
                rd = C.results_dir("duplicate_ablation", model, "%dpct" % pct,
                                   "seed%d" % seed)
                cmd = jobs.train_cmd(model, seed, pct, rd)
                # point the run at the deduplicated CSV instead
                cmd = cmd.replace("--data_csv %s" % C.DATA_CSV,
                                  "--data_csv %s" % out_csv)
                cmds.append(cmd)

    print("dedup re-run: %d models x %d fractions x %d seeds"
          % (len(args.models), len(args.fractions), len(seeds)))
    jobs.run_jobs(cmds, "duplicate_ablation")


if __name__ == "__main__":
    main()
