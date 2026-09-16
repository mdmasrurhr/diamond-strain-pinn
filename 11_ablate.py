"""Retrain with one thing changed at a time, to measure what each part contributes.

    python 11_ablate.py losses       remove one loss term at a time
    python 11_ablate.py design       vary one design choice at a time
    python 11_ablate.py duplicates   repeat the sweep on distinct strain states only
    python 11_ablate.py all          all three
"""

import argparse
import os

import pandas as pd

import config as C
import jobs

# --- loss terms ---
# Variant name -> the --drop_loss flags that define it. "full" drops nothing.
LOSS_VARIANTS = {
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
# 0% is excluded: with no labels and no physics there is nothing to train on, so
# the variants would not be comparable.
LOSS_FRACTIONS = [1, 5, 25, 100]

# --- design choices ---
# Group -> variant -> (model, extra flags).
DESIGN_GROUPS = {
    "weighting": {
        "ref_rba":         ("rba", []),
        "ref_sa":          ("sa",  []),
        "uniform_weights": ("rba", ["--no_rba"]),
        "frozen_sa":       ("sa",  ["--sa_lr", "0.0"]),
    },
    "optimizer": {
        "adamw_only":      ("rba", ["--no_soap"]),
        "plain_adam":      ("rba", ["--optimizer", "adam"]),
    },
    "architecture": {
        "width_50":        ("rba", ["--width", "50"]),
        "width_200":       ("rba", ["--width", "200"]),
        "depth_3":         ("rba", ["--depth", "3"]),
        "depth_8":         ("rba", ["--depth", "8"]),
    },
    "activation": {
        "relu":            ("rba", ["--activation", "relu"]),
        "tanh":            ("rba", ["--activation", "tanh"]),
        "gelu":            ("rba", ["--activation", "gelu"]),
    },
    "representation": {
        "oh_invariants":   ("rba", ["--input_inv"]),
    },
}
# Compared at full data, where capacity and optimizer dominate, and at 5%,
# where the physics prior dominates.
DESIGN_FRACTIONS = [5, 100]

DEDUP_FRACTIONS = [1, 5, 25, 100]


def ablate_losses(args, seeds):
    """Switch off one loss term at a time and retrain."""
    cmds = []
    for variant in sorted(LOSS_VARIANTS):
        for pct in LOSS_FRACTIONS:
            for seed in seeds:
                out = C.results_dir("loss_ablation", variant, "%dpct" % pct,
                                    "seed%d" % seed)
                cmds.append(jobs.train_cmd(args.model, seed, pct, out,
                                           extra=LOSS_VARIANTS[variant]))
    print("loss ablation: %d variants x %d fractions x %d seeds"
          % (len(LOSS_VARIANTS), len(LOSS_FRACTIONS), len(seeds)))
    jobs.run_jobs(cmds, "loss_ablation")


def ablate_design(args, seeds):
    """Change one design choice at a time and retrain."""
    cmds, n = [], 0
    for group in sorted(DESIGN_GROUPS):
        for variant, (model, extra) in DESIGN_GROUPS[group].items():
            n += 1
            for pct in DESIGN_FRACTIONS:
                for seed in seeds:
                    out = C.results_dir("design_ablation", variant, "%dpct" % pct,
                                        "seed%d" % seed)
                    cmds.append(jobs.train_cmd(model, seed, pct, out, extra=extra))
    print("design ablation: %d variants x %d fractions x %d seeds"
          % (n, len(DESIGN_FRACTIONS), len(seeds)))
    jobs.run_jobs(cmds, "design_ablation")


def ablate_duplicates(args, seeds):
    """Repeat the main sweep with repeated strain states removed."""
    df = pd.read_csv(C.DATA_CSV)
    key = df[C.STRAIN_COLS].round(9).apply(tuple, axis=1)
    before = len(df)
    # Keep the first occurrence. Copies carry identical labels, so which one
    # survives cannot change the result.
    dedup = df[~key.duplicated(keep="first")].reset_index(drop=True)
    out_csv = os.path.join(C.ROOT, "data", "deduplicated.csv")
    dedup.to_csv(out_csv, index=False)
    print("deduplicated: %d rows -> %d distinct states (%d removed)"
          % (before, len(dedup), before - len(dedup)))

    cmds = []
    for model in C.MODELS:
        for pct in DEDUP_FRACTIONS:
            for seed in seeds:
                out = C.results_dir("duplicate_ablation", model, "%dpct" % pct,
                                    "seed%d" % seed)
                cmd = jobs.train_cmd(model, seed, pct, out)
                cmd = cmd.replace("--data_csv %s" % C.DATA_CSV,
                                  "--data_csv %s" % out_csv)
                cmds.append(cmd)
    print("duplicate ablation: %d models x %d fractions x %d seeds"
          % (len(C.MODELS), len(DEDUP_FRACTIONS), len(seeds)))
    jobs.run_jobs(cmds, "duplicate_ablation")


STUDIES = {
    "losses":     ablate_losses,
    "design":     ablate_design,
    "duplicates": ablate_duplicates,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all",
                    choices=sorted(STUDIES) + ["all"])
    ap.add_argument("--model", default="rba", choices=["rba", "sa"],
                    help="losses: which weighting to ablate on")
    ap.add_argument("--quick", action="store_true", help="3 seeds instead of 17")
    args = ap.parse_args()

    seeds = C.SEEDS[:3] if args.quick else C.SEEDS
    for name in (sorted(STUDIES) if args.what == "all" else [args.what]):
        print("\n===== %s =====" % name)
        STUDIES[name](args, seeds)


if __name__ == "__main__":
    main()
