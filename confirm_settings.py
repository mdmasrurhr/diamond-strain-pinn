"""Re-run a small set of configurations to confirm the chosen settings still hold.

The settings were selected in separate studies. This trains the few
configurations that carry each choice, at the full budget, and checks the
orderings are unchanged before a long run starts.

    architecture   the chosen network against the best and the runner-up
    optimizer      SOAP alone and AdamW alone against the two-phase default
    weighting      the self-adaptive variant on the chosen network

    python confirm_settings.py            # run the configurations
    python confirm_settings.py --guard    # pass/fail verdict, exit 1 on fail

--guard exits non-zero on failure so it can gate a long run from a shell script.

Writes results/settings_confirmation/<variant>/seed<n>/.
"""

import argparse
import os
import sys

import config as C
import decide
import jobs

SEEDS = [42, 43, 44]

# The base command already carries the adopted config (d8_w64, 24k epochs, the
# authoritative loss weights) via jobs.train_cmd, so the base variant needs no
# extra flags and every other variant overrides exactly one thing.
VARIANTS = {
    "base_d8_w64":  ("rba", []),
    "d8_w200":      ("rba", ["--depth", "8", "--width", "200"]),
    "d7_w100":      ("rba", ["--depth", "7", "--width", "100"]),
    "d5_w100":      ("rba", ["--depth", "5", "--width", "100"]),
    "soap_only":    ("rba", ["--optimizer", "soap"]),
    "adamw_only":   ("rba", ["--optimizer", "adamw"]),
    "sa_base":      ("sa",  []),
}


def build():
    cmds = []
    for variant, (model, extra) in VARIANTS.items():
        for seed in SEEDS:
            rd = C.results_dir("confirm_settings", variant, "seed%d" % seed)
            cmds.append(jobs.train_cmd(model, seed, 100, rd, extra=extra))
    return cmds


def guard():
    df = decide.collect(os.path.join(C.RESULTS, "confirm_settings"),
                        ["variant", "seed"])
    need = set(VARIANTS)
    have = set(df.variant.unique()) if len(df) else set()
    if need - have:
        print("GUARD: missing variants: %s" % ", ".join(sorted(need - have)))
        return 1
    agg = decide.aggregate(df, ["variant"])
    decide.report(agg, "RECONFIRMATION under the refit constants")
    m = dict(zip(agg.variant, agg.test_mae_meV_mean))

    ok = True
    # 1. the architecture decision holds
    if m["base_d8_w64"] > m["d5_w100"]:
        p = decide.welch(df[df.variant == "base_d8_w64"].test_mae_meV,
                         df[df.variant == "d5_w100"].test_mae_meV)
        if p < 0.05:
            print("GUARD FAIL: d5_w100 beats the adopted d8_w64 (p=%.3f) -- the"
                  " architecture decision does not survive the refit." % p)
            ok = False
        else:
            print("guard note: d5_w100 nominally ahead but within noise "
                  "(p=%.3f); the parsimony pick stands." % p)
    else:
        print("guard: adopted d8_w64 (%.2f) <= thesis d5_w100 (%.2f) -- holds."
              % (m["base_d8_w64"], m["d5_w100"]))

    # 2. optimizer attribution holds
    if m["adamw_only"] < 1.5 * min(m["base_d8_w64"], m["soap_only"]):
        print("GUARD FAIL: AdamW-only is no longer clearly worse -- the SOAP"
              " attribution needs re-examining before the campaign.")
        ok = False
    else:
        print("guard: AdamW-only %.1f vs SOAP arms %.1f/%.1f -- attribution holds."
              % (m["adamw_only"], m["soap_only"], m["base_d8_w64"]))

    # 3. SA trains sanely on the new backbone
    if m["sa_base"] > 2.0 * m["base_d8_w64"]:
        print("GUARD FAIL: SA at %.1f meV vs RBA %.1f -- sa_lr does not carry"
              " to the new backbone; retune before the campaign."
              % (m["sa_base"], m["base_d8_w64"]))
        ok = False
    else:
        print("guard: SA %.1f meV on the new backbone -- sane." % m["sa_base"])

    print("\nGUARD: %s" % ("PASS -- campaign may start" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard", action="store_true")
    a = ap.parse_args()
    if a.guard:
        sys.exit(guard())
    cmds = build()
    print("reconfirmation: %d variants x %d seeds = %d runs at %d epochs"
          % (len(VARIANTS), len(SEEDS), len(cmds), C.TUNED["epochs"]))
    jobs.run_jobs(cmds, "confirm_settings")
    guard()


if __name__ == "__main__":
    main()
