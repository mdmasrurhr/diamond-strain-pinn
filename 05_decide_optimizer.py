"""Measure the optimizer schedule, the switch point between its two phases, the
gradient clipping norm and the sample-weight clamp.
"""

import argparse
import os

import config as C
import decide
import jobs

PCT = 100

GROUPS = {
    # Each arm uses the hyperparameters belonging to its own optimizer, so
    # SOAP alone runs at SOAP's settings rather than at AdamW's.
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
            # Split the total effect between the optimizer and the switch.
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
                    print("    -> the switch adds nothing beyond SOAP alone")
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
