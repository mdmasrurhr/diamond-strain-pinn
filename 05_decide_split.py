"""Compare five ways of splitting the data into train, validation and test sets."""

import argparse
import os

import pandas as pd

import config as C
import decide
import jobs

# Held at 100% labels: a split scheme is about which rows go where, and the
# scarce-label regime would confound that with how many labels there are.
PCT = 100

SCHEMES = {
    "stratified":    [],
    "random":        ["--split", "random"],
    "group_shear":   ["--split", "group", "--holdout_family", "shear"],
    "group_triaxial":["--split", "group", "--holdout_family", "triaxial"],
    "group_biaxial": ["--split", "group", "--holdout_family", "biaxial"],
    "magnitude_05":  ["--split", "magnitude", "--magnitude_thresh", "0.05"],
}
# k-fold is expressed as five runs rather than a scheme flag, because each fold
# is a separate training run with a different test set.
KFOLDS = 5

WHAT_IT_MEASURES = {
    "stratified":     "interpolation, all families seen",
    "random":         "interpolation, no stratification",
    "kfold":          "interpolation, all data used",
    "group_shear":    "extrapolation to an unseen family (shear)",
    "group_triaxial": "extrapolation to an unseen family (triaxial)",
    "group_biaxial":  "extrapolation to an unseen family (biaxial)",
    "magnitude_05":   "extrapolation beyond |strain| = 0.05",
}


def build(seeds, model):
    cmds = []
    for name, extra in SCHEMES.items():
        for seed in seeds:
            rd = C.results_dir("split_study", name, "seed%d" % seed)
            cmds.append(jobs.train_cmd(model, seed, PCT, rd, extra=extra))
    for fold in range(KFOLDS):
        for seed in seeds:
            rd = C.results_dir("split_study", "kfold", "seed%d_fold%d" % (seed, fold))
            cmds.append(jobs.train_cmd(model, seed, PCT, rd,
                                       extra=["--cv_fold", str(fold),
                                              "--cv_nfolds", str(KFOLDS)]))
    return cmds


def report():
    root = os.path.join(C.RESULTS, "split_study")
    df = decide.collect(root, ["scheme", "run"])
    if not len(df):
        raise SystemExit("no runs found under %s -- run the study first" % root)

    agg = decide.aggregate(df, ["scheme"])
    agg["measures"] = [WHAT_IT_MEASURES.get(s, "") for s in agg.scheme]
    decide.report(agg, "SPLIT SCHEMES", extra_cols=("measures",))

    # The decision is between the schemes that measure the same thing. An
    # extrapolation split is not a competitor to an interpolation split; it is a
    # different reported quantity, so it never wins this comparison.
    interp = agg[agg.scheme.isin(["stratified", "random", "kfold"])]
    print("\nDECISION")
    if len(interp) < 2:
        print("  not enough interpolation schemes finished to decide")
    else:
        best = interp.iloc[0]
        strat = interp[interp.scheme == "stratified"]
        print("  best interpolation scheme : %s (%.2f meV)"
              % (best.scheme, best.test_mae_meV_mean))
        if len(strat):
            s = strat.iloc[0]
            a = df[df.scheme == best.scheme].test_mae_meV.values
            b = df[df.scheme == "stratified"].test_mae_meV.values
            p = decide.welch(a, b)
            print("  adopted (stratified)      : %.2f meV" % s.test_mae_meV_mean)
            print("  difference                : %.2f meV, Welch p = %.3f -> %s"
                  % (s.test_mae_meV_mean - best.test_mae_meV_mean, p,
                     "real" if p < 0.05 else "within seed noise"))
            print("\n  k-fold uses 80%% of the data for training against the fixed")
            print("  split's 70%%, so it is expected to be the kinder number. It is")
            print("  a better ESTIMATE; it is not a better PROTOCOL for a study that")
            print("  also reports per-run diagnostics. Quote one consistently.")

    extrap = agg[~agg.scheme.isin(["stratified", "random", "kfold"])]
    if len(extrap):
        print("\n  Extrapolation splits -- reported, never competed against the above:")
        for r in extrap.itertuples():
            print("    %-16s %8.2f meV   %s"
                  % (r.scheme, r.test_mae_meV_mean, r.measures))

    out = C.results_dir("split_study")
    agg.to_csv(os.path.join(out, "split_decision.csv"), index=False)
    print("\nwrote %s" % os.path.join(out, "split_decision.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="all 17 seeds")
    ap.add_argument("--model", default="rba", choices=["rba", "sa"])
    ap.add_argument("--report", action="store_true", help="read results, decide")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()

    seeds = C.SEEDS if args.full else C.SEEDS[:3]
    cmds = build(seeds, args.model)
    print("split study: %d schemes + %d folds, %d seeds -> %d runs"
          % (len(SCHEMES), KFOLDS, len(seeds), len(cmds)))
    if args.dry_run:
        for c in cmds[:4]:
            print("  " + c)
        print("  ... (%d total)" % len(cmds))
        return
    jobs.run_jobs(cmds, "split_study")
    report()


if __name__ == "__main__":
    main()
