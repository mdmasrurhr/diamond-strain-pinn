"""
decide_optimizer.py -- close the gaps in the optimizer evidence.

The optimizer family is already well evidenced: AdamW -> SOAP beats Adam by 20
meV, RMSprop by 36, SGD by 157 and L-BFGS by 237. Nothing here disputes that.

What is missing is attribution and the surrounding settings.

    arms        AdamW-only, SOAP-only, and the two-phase switch. Dropping SOAP
                costs +31 meV, the largest single effect in the whole ablation
                table -- but with no SOAP-only arm that could be SOAP itself or
                the switch, and the paper currently implies the second while
                evidencing only the first. One run per seed closes it.
    switch      where to switch. Fixed at 0.5 and never screened on its own.
    clip        gradient clip norm. Varied inside the hyperparameter search,
                never screened, so its flatness is assumed rather than shown.
    sa_clamp    the [-8, 8] bound on the self-adaptive log-weights. It bounds
                how far the adaptation can go, which makes it a claim of the
                work by the project's own taxonomy, and it has no evidence.

    python decide_optimizer.py                # every group, 3 seeds
    python decide_optimizer.py --group arms
    python decide_optimizer.py --full --group arms   # 17 seeds, for the paper
    python decide_optimizer.py --report

Results in results/optimizer_study/<group>/<variant>/seed<n>/.
"""

import argparse
import os

import config as C
import decide
import jobs

PCT = 100

GROUPS = {
    # The three-arm comparison the framework asks for. Each arm gets the
    # hyperparameters belonging to its own optimizer, so SOAP-only is compared
    # at SOAP's settings rather than at AdamW's.
    "arms": {
        "two_phase":  ("rba", []),
        "adamw_only": ("rba", ["--optimizer", "adamw"]),
        "soap_only":  ("rba", ["--optimizer", "soap"]),
    },
    "switch": {
        "switch_025": ("rba", ["--switch_frac", "0.25"]),
        "switch_050": ("rba", ["--switch_frac", "0.50"]),
        "switch_075": ("rba", ["--switch_frac", "0.75"]),
    },
    "clip": {
        "clip_01":  ("rba", ["--clip_norm", "0.1"]),
        "clip_05":  ("rba", ["--clip_norm", "0.5"]),
        "clip_10":  ("rba", ["--clip_norm", "1.0"]),
        "clip_50":  ("rba", ["--clip_norm", "5.0"]),
        "clip_off": ("rba", ["--clip_norm", "1e9"]),
    },
    # The clamp is an SA-only mechanism, so these run the SA model.
    "sa_clamp": {
        "clamp_2":  ("sa", ["--sa_clamp", "2.0"]),
        "clamp_4":  ("sa", ["--sa_clamp", "4.0"]),
        "clamp_8":  ("sa", ["--sa_clamp", "8.0"]),
        "clamp_16": ("sa", ["--sa_clamp", "16.0"]),
    },
}

ADOPTED = {"arms": "two_phase", "switch": "switch_050",
           "clip": "clip_10", "sa_clamp": "clamp_8"}


def build(seeds, which):
    cmds = []
    for g in which:
        for variant, (model, extra) in GROUPS[g].items():
            for seed in seeds:
                rd = C.results_dir("optimizer_study", g, variant, "seed%d" % seed)
                cmds.append(jobs.train_cmd(model, seed, PCT, rd, extra=extra))
    return cmds


def report():
    out = C.results_dir("optimizer_study")
    all_agg = []
    for g in GROUPS:
        root = os.path.join(C.RESULTS, "optimizer_study", g)
        df = decide.collect(root, ["variant", "seed"])
        if not len(df):
            continue
        agg = decide.aggregate(df, ["variant"])
        decide.report(agg, "OPTIMIZER -- %s" % g.upper())
        print("  shape across the screen: %s"
              % decide.shape(agg.variant, agg.test_mae_meV_mean))

        adopted = ADOPTED[g]
        if adopted in set(agg.variant):
            a = df[df.variant == adopted].test_mae_meV.values
            best = agg.iloc[0]
            if best.variant != adopted:
                b = df[df.variant == best.variant].test_mae_meV.values
                p = decide.welch(a, b)
                print("  adopted %s is not the best; %s is better by %.2f meV, p=%.3f -> %s"
                      % (adopted, best.variant,
                         agg[agg.variant == adopted].test_mae_meV_mean.iloc[0]
                         - best.test_mae_meV_mean, p,
                         "change it" if p < 0.05 else "within noise, keep adopted"))
            else:
                print("  adopted %s is the best in this screen" % adopted)

        if g == "arms":
            # The attribution the paper needs, stated explicitly.
            m = dict(zip(agg.variant, agg.test_mae_meV_mean))
            if {"two_phase", "adamw_only", "soap_only"} <= set(m):
                print("\n  ATTRIBUTION")
                print("    AdamW only         %8.2f meV" % m["adamw_only"])
                print("    SOAP only          %8.2f meV" % m["soap_only"])
                print("    AdamW -> SOAP      %8.2f meV" % m["two_phase"])
                gain_soap = m["adamw_only"] - m["soap_only"]
                gain_switch = m["soap_only"] - m["two_phase"]
                print("    attributable to SOAP itself : %+8.2f meV" % gain_soap)
                print("    attributable to the switch  : %+8.2f meV" % gain_switch)
                if gain_switch <= 0:
                    print("    -> the switch buys nothing beyond SOAP; the paper should")
                    print("       say SOAP, not the two-phase schedule.")
                else:
                    print("    -> both contribute; the two-phase claim is supported.")
        agg["group"] = g
        all_agg.append(agg)

    if all_agg:
        import pandas as pd
        pd.concat(all_agg).to_csv(
            os.path.join(out, "optimizer_decision.csv"), index=False)
        print("\nwrote %s/optimizer_decision.csv" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", nargs="+", default=sorted(GROUPS),
                    choices=sorted(GROUPS))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()

    seeds = C.SEEDS if args.full else C.SEEDS[:3]
    cmds = build(seeds, args.group)
    print("optimizer study: %s, %d seeds -> %d runs"
          % ("+".join(args.group), len(seeds), len(cmds)))
    if args.dry_run:
        return
    jobs.run_jobs(cmds, "optimizer_study")
    report()


if __name__ == "__main__":
    main()
