"""Vary one design choice at a time and retrain."""

import argparse

import config as C
import jobs

W = C.TUNED

# group -> {variant name: (model, extra flags)}
GROUPS = {
    "weighting": {
        "ref_rba":        ("rba", []),
        "ref_sa":         ("sa",  []),
        "uniform_weights": ("rba", ["--no_rba"]),
        "frozen_sa":      ("sa",  ["--sa_lr", "0.0"]),
    },
    "optimizer": {
        "adamw_only":     ("rba", ["--no_soap"]),
        "plain_adam":     ("rba", ["--optimizer", "adam"]),
    },
    "architecture": {
        "width_50":       ("rba", ["--width", "50"]),
        "width_200":      ("rba", ["--width", "200"]),
        "depth_3":        ("rba", ["--depth", "3"]),
        "depth_8":        ("rba", ["--depth", "8"]),
    },
    "activation": {
        "relu":           ("rba", ["--activation", "relu"]),
        "tanh":           ("rba", ["--activation", "tanh"]),
        "gelu":           ("rba", ["--activation", "gelu"]),
    },
    "representation": {
        "oh_invariants":  ("rba", ["--input_inv"]),
    },
}

# Design choices are compared where they are most likely to matter: at full data
# (where capacity and optimizer dominate) and at 5% (where the prior dominates).
FRACTIONS = [5, 100]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", nargs="+", default=sorted(GROUPS),
                    choices=sorted(GROUPS))
    ap.add_argument("--fractions", nargs="+", type=int, default=FRACTIONS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[:3] if args.quick else C.SEEDS
    cmds, n_var = [], 0
    for g in args.group:
        for variant, (model, extra) in GROUPS[g].items():
            n_var += 1
            for pct in args.fractions:
                for seed in seeds:
                    rd = C.results_dir("design_ablation", variant, "%dpct" % pct,
                                       "seed%d" % seed)
                    cmds.append(jobs.train_cmd(model, seed, pct, rd, extra=extra))

    print("design ablations: %d variants x %d fractions x %d seeds"
          % (n_var, len(args.fractions), len(seeds)))
    jobs.run_jobs(cmds, "design_ablation")


if __name__ == "__main__":
    main()
