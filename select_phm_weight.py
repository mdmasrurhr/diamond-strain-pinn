#!/usr/bin/env python3
"""Select the agreement weight of the proposed hybrid model (PHM) from VALIDATION errors only (no test number is read).

Second selection stage. The first (select_weighting.py) chose the form of the model among six variants that carried the
zero-strain terms; with those terms removed, the weight s(n) of the agreement term is chosen again among five rules:
  ld9    s = min(1, 9/n)          (hyb_sdp_gap_ld9)
  ld9sq  s = min(1, (9/n)^2)      (phm_ld9sq)
  ld9cu  s = min(1, (9/n)^3)      (phm)
  ld4    s = min(1, 4/n)          (phm_ld4)
  cut18  s = min(1, 9/n) up to 18 labeled states, 0 beyond (hyb_sdp_gap_ld9 up to 2%, sdp_net beyond)
At a labeled fraction where a rule gives the same weight as ld9, its runs are those of hyb_sdp_gap_ld9 (identical commands).
Criterion, fixed before the runs: the lowest validation gap MAE averaged over the 17 seeds at each of the nine labeled
fractions and then over the fractions; the mean rank and the geometric mean over fractions are printed as checks.
Writes <results>/phm_weight_choice.csv.
Usage: python select_phm_weight.py [results folder, default results] [--allow-incomplete]"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import config as C

R = next((a for a in sys.argv[1:] if not a.startswith("--")), os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
NL = {p: round(903 * p / 100) for p in C.FRACTIONS}
RULES = {"ld9": lambda n: min(1.0, 9 / n) if n else 1.0, "ld9sq": lambda n: min(1.0, (9 / n) ** 2) if n else 1.0,
         "ld9cu": lambda n: min(1.0, (9 / n) ** 3) if n else 1.0, "ld4": lambda n: min(1.0, 4 / n) if n else 1.0,
         "cut18": lambda n: (min(1.0, 9 / n) if n else 1.0) if n <= 18 else 0.0}
FOLDER = {"ld9": "hyb_sdp_gap_ld9", "ld9sq": "phm_ld9sq", "ld9cu": "phm", "ld4": "phm_ld4"}


def folder(rule, p):
    s, s9 = RULES[rule](NL[p]), RULES["ld9"](NL[p])
    if np.isclose(s, s9):
        return "hyb_sdp_gap_ld9"
    if s == 0.0:
        return "sdp_net"
    return FOLDER[rule]


rows, missing = [], []
for rule in RULES:
    for p in C.FRACTIONS:
        for sd in C.SEEDS:
            f = os.path.join(R, "B_main", folder(rule, p), "%dpct" % p, "seed%d" % sd, "metrics_summary.csv")
            if os.path.exists(f):
                rows.append(dict(rule=rule, pct=p, seed=sd, val_mae_meV=float(pd.read_csv(f)["val_mae_meV"].iloc[0])))
            else:
                missing.append(f)
if missing:
    print("%d candidate runs are not finished, e.g. %s" % (len(missing), missing[0]))
    if "--allow-incomplete" not in sys.argv:
        sys.exit("finish the runs first, or pass --allow-incomplete for a provisional look")
df = pd.DataFrame(rows)
level = df.groupby(["rule", "pct"]).val_mae_meV.mean().unstack("pct").reindex(list(RULES))
crit = pd.DataFrame({"mean_over_levels": level.mean(axis=1), "mean_rank_over_levels": level.rank(axis=0).mean(axis=1),
                     "geomean_over_levels": np.exp(np.log(level).mean(axis=1))})
per_seed = df.groupby(["rule", "seed"]).val_mae_meV.mean().unstack("rule")
crit["sd_over_seeds"] = per_seed.std(ddof=1)
pd.set_option("display.width", 200)
print("Validation gap MAE (meV), mean over seeds, by labeled fraction (%):")
print(level.round(2).to_string())
print("\nCriterion (lowest mean_over_levels wins) and checks:")
print(crit.round(3).to_string())
best = crit.mean_over_levels.idxmin(); runner = crit.mean_over_levels.drop(best).idxmin()
both = per_seed[[best, runner]].dropna()
t = stats.ttest_rel(both[runner], both[best])
print("\n%s vs runner-up %s: %.2f meV lower on average over fractions; paired over %d seeds P = %.3g"
      % (best, runner, (both[runner] - both[best]).mean(), len(both), t.pvalue))
others = {c: crit[c].idxmin() for c in ["mean_rank_over_levels", "geomean_over_levels"]}
if any(v != best for v in others.values()):
    print("NOTE: the checks disagree: %s." % others)
print("Selected rule: %s" % best)
if not missing:
    out = level.copy(); out.columns = ["val_mae_%dpct" % c for c in out.columns]
    pd.concat([out, crit], axis=1).assign(chosen=lambda d: d.index == best).to_csv(os.path.join(R, "phm_weight_choice.csv"))
    print("wrote", os.path.join(R, "phm_weight_choice.csv"))
