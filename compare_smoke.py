#!/usr/bin/env python3
"""Smoke-test check, after the runs of tasks/smoke.tsv have finished (make_tasks.py smoke; ./submit.sh tasks/smoke.tsv).

Each smoke run (300 iterations) is compared with the reference in reference_smoke/, made with the same code and command
on the authors' workstation (Intel Xeon W-2255, CPU, one thread per run). On the same kind of CPU the predictions agree
exactly; other CPUs round differently, so a run PASSES if its test and validation MAE agree with the reference within
max(2 meV, 5%). The time per iteration gives the projected time of a full 24,000-iteration run.
Usage: python compare_smoke.py"""
import csv, glob, os, sys
import numpy as np, pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))



def full_run_minutes(run_dir):
    # 03_train_pinn.py trains AdamW for the first half of the epochs and SOAP for the second, so time each half
    e = pd.read_csv(os.path.join(run_dir, "epoch_log.csv")).set_index("epoch").elapsed_s
    last = int(e.index.max()); half = last // 2
    t1 = (e.loc[half] - e.loc[20]) / (half - 20)
    t2 = (e.loc[last] - e.loc[half]) / (last - half)
    return 12000 * (t1 + t2) / 60.0


names = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob("reference_smoke/*/metrics_summary.csv"))
fails, minutes, ratio = [], {}, []
print("%-28s %9s %9s %9s %9s %10s %9s  %s" % ("smoke run", "test MAE", "ref", "val MAE", "ref", "max|dEg|", "time/ref", "full run here"))
for n in names:
    a, b = "results/smoke/" + n, "reference_smoke/" + n
    if not os.path.exists(a + "/metrics_summary.csv"):
        print("%-28s not finished" % n); fails.append(n); continue
    x, y = pd.read_csv(a + "/metrics_summary.csv").iloc[0], pd.read_csv(b + "/metrics_summary.csv").iloc[0]
    d = np.abs(pd.read_csv(a + "/test_results.csv").Eg_pred_eV.values - pd.read_csv(b + "/test_results.csv").Eg_pred_eV.values).max()
    good = all(abs(x[c] - y[c]) <= max(2.0, 0.05 * abs(y[c])) for c in ["test_mae_meV", "val_mae_meV"])
    mins, r = full_run_minutes(a), full_run_minutes(a) / full_run_minutes(b)
    ratio.append(r)
    print("%-28s %9.2f %9.2f %9.2f %9.2f %10.1e %8.2fx  %5.1f min  %s" % (n, x.test_mae_meV, y.test_mae_meV, x.val_mae_meV, y.val_mae_meV, d, r, mins, "PASS" if good else "FAIL"))
    if not good:
        fails.append(n)
print("\nThis machine vs the reference workstation (same runs): median %.2fx the time" % np.median(ratio) if ratio else "")
print("\n%s" % ("SMOKE TEST PASSED" if not fails else "SMOKE TEST: %d run(s) failed or missing: %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)
