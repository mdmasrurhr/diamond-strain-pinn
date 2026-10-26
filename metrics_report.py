#!/usr/bin/env python3
"""Standard metrics for any set of finished runs, computed from each run's test_results.csv.

Absolute (meV): MAE, RMSE, median, 95th percentile and maximum absolute error, bias (mean signed error), CBM and VBM MAE,
                MAE per strain family.
Dimensionless (decimal): R^2 (coefficient of determination), Pearson r, Spearman rank correlation, NMAE = MAE / mean gap,
                NRMSE = RMSE / standard deviation of the true gaps, MAPE (as a fraction), fraction of states within 10, 25 and
                50 meV, sign accuracy of the strain-induced gap change (relative to the unstrained gap).
Usage: python metrics_report.py <results_root> [--group N]   groups runs by the first N path parts below <results_root>
       (default: all parts except the seed), writes <results_root>/metrics_report.csv and prints mean values per group."""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats

EG0 = 5.5042     # gap at zero strain (eV), from a cubic fit to the isotropic states with |strain| <= 0.02; the paper's
                 # analysis sets it to the unrounded fit value
FAMS = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]


def fam(s):
    s = str(s)
    return "isotropic" if s in ("unstrained", "isotropic") else s.split("_")[0]


def metrics(t):
    y, p = t.Eg_true_eV.values, t.Eg_pred_eV.values; e = p - y; a = np.abs(e)
    r = dict(n=len(t), MAE_meV=1e3 * a.mean(), RMSE_meV=1e3 * np.sqrt((e ** 2).mean()), median_AE_meV=1e3 * np.median(a),
             P95_meV=1e3 * np.quantile(a, 0.95), max_meV=1e3 * a.max(), bias_meV=1e3 * e.mean(),
             CBM_MAE_meV=1e3 * np.abs(t.CBM_pred_eV - t.CBM_true_eV).mean(), VBM_MAE_meV=1e3 * np.abs(t.VBM_pred_eV - t.VBM_true_eV).mean(),
             R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), pearson_r=stats.pearsonr(y, p)[0], spearman=stats.spearmanr(y, p)[0],
             NMAE=a.mean() / np.abs(y).mean(), NRMSE=np.sqrt((e ** 2).mean()) / y.std(), MAPE=np.mean(a / np.abs(y)),
             within_10meV=np.mean(a < 0.010), within_25meV=np.mean(a < 0.025), within_50meV=np.mean(a < 0.050),
             sign_acc=np.mean(np.sign(p - EG0) == np.sign(y - EG0)))
    f = t.strain_type.map(fam).values
    for k in FAMS:
        m = f == k; r["MAE_%s_meV" % k] = 1e3 * a[m].mean() if m.any() else np.nan
    return r


if __name__ == "__main__":
    root = sys.argv[1]; rows = []
    for f in glob.glob(os.path.join(root, "**", "test_results.csv"), recursive=True):
        rel = os.path.relpath(os.path.dirname(f), root).split(os.sep)
        r = metrics(pd.read_csv(f)); r["run"] = "/".join(rel); r["group"] = "/".join(rel[:-1]); r["seed"] = rel[-1]; rows.append(r)
    df = pd.DataFrame(rows).sort_values("run"); df.to_csv(os.path.join(root, "metrics_report.csv"), index=False)
    cols = ["MAE_meV", "RMSE_meV", "R2", "NMAE", "NRMSE", "MAPE", "within_25meV", "P95_meV", "max_meV", "bias_meV", "MAE_shear_meV", "sign_acc"]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    g = df.groupby("group")[cols].mean()
    fmt = {c: ("{:.4f}" if c in ("R2", "NMAE", "NRMSE", "MAPE", "within_25meV", "sign_acc") else "{:.1f}") for c in cols}
    print("%d runs, %d groups (means over seeds)" % (len(df), len(g)))
    print(g.apply(lambda col: col.map(fmt[col.name].format)).to_string())
