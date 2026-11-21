"""Compare the smoke runs (make_tasks.py smoke) with reference_smoke/.

The references were made with the same commands on an Intel Xeon W-2255 (CPU, one thread per run). On that processor
the predictions agree exactly; other processors round differently, so a run passes if its test and validation MAE agree
with the reference within max(2 meV, 5%). The time per iteration gives the projected time of a full run here.
Usage: python compare_smoke.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

import config as C

os.chdir(C.HERE)


def full_run_minutes(run_dir):
    """Projected minutes for C.TUNED["epochs"] iterations, timing the AdamW and SOAP halves separately."""
    e = pd.read_csv(os.path.join(run_dir, "epoch_log.csv")).set_index("epoch").elapsed_s
    last = int(e.index.max())
    half = last // 2
    t1 = (e.loc[half] - e.loc[20]) / (half - 20)
    t2 = (e.loc[last] - e.loc[half]) / (last - half)
    return C.TUNED["epochs"] / 2 * (t1 + t2) / 60.0


names = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob("reference_smoke/*/metrics_summary.csv"))
fails, ratios = [], []
print(
    "%-34s %9s %9s %9s %9s %9s %9s  %s"
    % ("smoke run", "test MAE", "ref", "val MAE", "ref", "max|dEg|", "time/ref", "full run here")
)
for name in names:
    run, ref = "results/smoke/" + name, "reference_smoke/" + name
    if not os.path.exists(run + "/metrics_summary.csv"):
        print("%-34s not finished" % name)
        fails.append(name)
        continue
    x, y = pd.read_csv(run + "/metrics_summary.csv").iloc[0], pd.read_csv(ref + "/metrics_summary.csv").iloc[0]
    max_diff = np.abs(
        pd.read_csv(run + "/test_results.csv").Eg_pred_eV.values
        - pd.read_csv(ref + "/test_results.csv").Eg_pred_eV.values
    ).max()
    good = all(abs(x[c] - y[c]) <= max(2.0, 0.05 * abs(y[c])) for c in ["test_mae_meV", "val_mae_meV"])
    minutes = full_run_minutes(run)
    ratios.append(minutes / full_run_minutes(ref))
    print(
        "%-34s %9.2f %9.2f %9.2f %9.2f %9.1e %8.2fx  %5.1f min  %s"
        % (
            name,
            x.test_mae_meV,
            y.test_mae_meV,
            x.val_mae_meV,
            y.val_mae_meV,
            max_diff,
            ratios[-1],
            minutes,
            "PASS" if good else "FAIL",
        )
    )
    if not good:
        fails.append(name)
if ratios:
    print("\nThis machine needs %.2f times the time of the reference machine (median)." % np.median(ratios))
print("\n" + ("SMOKE TEST PASSED" if not fails else "SMOKE TEST FAILED: %s" % ", ".join(fails)))
sys.exit(1 if fails else 0)
