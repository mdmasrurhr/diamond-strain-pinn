#!/usr/bin/env python3
"""Select the weighting rule of our model from VALIDATION errors only (no test number is read).

Rule, fixed before the runs: among the six weighting variants (make_tasks.VARIANTS), the one with the lowest validation
gap MAE, averaged over the 17 seeds at each of the nine labeled fractions and then over the fractions, on the
symmetry-grouped division (results/B_main). Two robustness checks are printed next to it, the mean rank over fractions
and the geometric mean over fractions. The selected variant is hyb_sdp_ld9; its weighting rule is the one used by our model (hyb_sdp_gap_ld9).
Writes results/hybrid_choice.csv.
Usage: python select_weighting.py [--allow-incomplete]   (incomplete = provisional look, nothing is written)"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import config as C
from make_tasks import VARIANTS

os.chdir(os.path.dirname(os.path.abspath(__file__)))
CANDS = VARIANTS
OUT_CSV = "results/hybrid_choice.csv"
rows, missing = [], []
for m in CANDS:
    for p in C.FRACTIONS:
        for s in C.SEEDS:
            f = "results/B_main/%s/%dpct/seed%d/metrics_summary.csv" % (m, p, s)
            if os.path.exists(f):
                rows.append(dict(model=m, pct=p, seed=s, val_mae_meV=float(pd.read_csv(f)["val_mae_meV"].iloc[0])))
            else:
                missing.append(f)
provisional = bool(missing)
if missing:
    print("%d of %d candidate runs are not finished, e.g. %s" % (len(missing), len(CANDS) * len(C.FRACTIONS) * len(C.SEEDS), missing[0]))
    if "--allow-incomplete" not in sys.argv:
        sys.exit("finish the runs first (./status.sh tasks/main.tsv), or pass --allow-incomplete for a provisional look")
df = pd.DataFrame(rows)
bad = df[~np.isfinite(df.val_mae_meV)]
if len(bad):
    print("WARNING: %d runs have no validation MAE (failed?):\n%s" % (len(bad), bad.to_string(index=False)))
    df = df[np.isfinite(df.val_mae_meV)]

level = df.groupby(["model", "pct"]).val_mae_meV.mean().unstack("pct").reindex(CANDS)   # seed mean per level
crit = pd.DataFrame({"mean_over_levels": level.mean(axis=1),
                     "mean_rank_over_levels": level.rank(axis=0).mean(axis=1),
                     "geomean_over_levels": np.exp(np.log(level).mean(axis=1))})
per_seed = df.groupby(["model", "seed"]).val_mae_meV.mean().unstack("model")                   # level mean per seed
crit["sd_over_seeds"] = per_seed.std(ddof=1)

pd.set_option("display.width", 200)
print("\nValidation gap MAE (meV), mean over seeds, by label level (%):")
print(level.round(2).to_string())
print("\nSelection criterion (lowest mean_over_levels wins) and robustness checks:")
print(crit.round(2).to_string())
best = crit.mean_over_levels.idxmin()
others = {c: crit[c].idxmin() for c in ["mean_rank_over_levels", "geomean_over_levels"]}
runner = crit.mean_over_levels.drop(best).idxmin()
both = per_seed[[best, runner]].dropna()
if len(both) > 2:
    t = stats.ttest_rel(both[runner], both[best])
    print("\n%s vs runner-up %s: %.2f meV lower on average over levels; paired over %d seeds p = %.3g (two-sided t-test)"
          % (best, runner, (both[runner] - both[best]).mean(), len(both), t.pvalue))
if any(v != best for v in others.values()):
    print("NOTE: the robustness checks disagree: %s." % others)
print("\nSelected variant: %s" % best)

if provisional:
    print("(provisional: runs are missing, nothing written)")
else:
    out = level.copy(); out.columns = ["val_mae_%dpct" % c for c in out.columns]
    pd.concat([out, crit], axis=1).assign(chosen=lambda d: d.index == best).to_csv(OUT_CSV)
    print("wrote %s" % OUT_CSV)
