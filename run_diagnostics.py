#!/usr/bin/env python3
"""Run diag_run.py on every finished run matching the given patterns and append one JSON line per run to a file.
The paper's analysis needs (a superset of runs is fine; the files are merged by run folder):
  python run_diagnostics.py results/diagnostics.jsonl "results/B_main/*/5pct/seed*" "results/B_main/*/100pct/seed*"
  python run_diagnostics.py results/diagnostics_ablation.jsonl "results/B_ablation/*/*pct/seed*" "results/B_main/hyb_sdp_ld9/*pct/seed*"
Runs already in the output file are skipped. Usage: python run_diagnostics.py <out.jsonl> <glob> [<glob> ...]"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

os.chdir(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1]
done = set(json.loads(l)["run"] for l in open(out)) if os.path.exists(out) else set()
runs = sorted(set(d for g in sys.argv[2:] for d in glob.glob(g) if os.path.exists(os.path.join(d, "best_model.pt"))))
todo = [d for d in runs if d not in done]
print("%d runs, %d already done, %d to do" % (len(runs), len(runs) - len(todo), len(todo)), flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def one(d):
    return subprocess.run([sys.executable, "diag_run.py", d], capture_output=True, text=True, env=env).stdout.strip().splitlines()[-1]


with ThreadPoolExecutor(int(os.environ.get("NPROC", "8"))) as ex, open(out, "a") as f:
    for line in ex.map(one, todo):
        f.write(line + "\n"); f.flush()
