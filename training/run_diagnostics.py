"""Run diag_run.py on every finished run matching the patterns and append one JSON line per run to a file.

The analysis needs the diagnostics of the main and ablation runs:
  python run_diagnostics.py results/diagnostics.jsonl "results/B_main/*/*pct/seed*" "results/B_ablation/*/*pct/seed*"
Runs already in the file are skipped. NPROC sets the number of parallel processes (default 8).
"""

import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 3:
    sys.exit(__doc__)
out = sys.argv[1]
done = {json.loads(line)["run"] for line in open(out)} if os.path.exists(out) else set()
runs = sorted({d for g in sys.argv[2:] for d in glob.glob(g) if os.path.exists(os.path.join(d, "best_model.pt"))})
todo = [d for d in runs if d not in done]
print("%d runs, %d already done, %d to do" % (len(runs), len(runs) - len(todo), len(todo)), flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def diagnose(run):
    return (
        subprocess.run([sys.executable, "diag_run.py", run], capture_output=True, text=True, env=env, check=True)
        .stdout.strip()
        .splitlines()[-1]
    )


with ThreadPoolExecutor(int(os.environ.get("NPROC", "8"))) as pool, open(out, "a") as f:
    for line in pool.map(diagnose, todo):
        f.write(line + "\n")
        f.flush()
