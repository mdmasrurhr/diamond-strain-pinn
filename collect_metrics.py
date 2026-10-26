#!/usr/bin/env python3
"""Hardware and time metrics of every job under results/<subfolder> (default: all of results/), from the hardware.txt
(node, CPU model, start/end, exit code) and resources.txt (wall time, CPU time, peak memory) that run_task.sh writes.
Writes results/run_metrics.csv and prints totals by group and model.
Usage: python collect_metrics.py [B_main]"""
import glob, os, re, sys
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join("results", sys.argv[1] if len(sys.argv) > 1 else "")


def grab(text, pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else None


rows = []
for hw in glob.glob(os.path.join(root, "**", "hardware.txt"), recursive=True):
    d = os.path.dirname(hw)
    info = dict(l.rstrip("\n").split("=", 1) for l in open(hw) if "=" in l)
    rel = os.path.relpath(d, "results").split(os.sep)
    r = dict(run=d, group=rel[0], model=rel[1] if len(rel) > 1 else "", level=rel[2] if len(rel) > 2 else "",
             seed=rel[3] if len(rel) > 3 else "", kind="xai" if rel[-1] == "xai_batch" else "train",
             host=info.get("host"), cpu_model=info.get("cpu_model"), slurm_job=info.get("slurm_job"),
             exit_code=info.get("exit_code"), start=info.get("start"), end=info.get("end"))
    res = os.path.join(d, "resources.txt")
    if os.path.exists(res):
        t = open(res).read()
        wall = grab(t, r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)")
        if wall:
            parts = [float(x) for x in wall.split(":")]
            r["wall_s"] = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
        else:
            r["wall_s"] = float(grab(t, r"Elapsed \(wall clock\) seconds: ([\d.]+)") or "nan")
        r["cpu_s"] = float(grab(t, r"User time \(seconds\): ([\d.]+)") or 0) + float(grab(t, r"System time \(seconds\): ([\d.]+)") or 0)
        r["peak_rss_MB"] = float(grab(t, r"Maximum resident set size \(kbytes\): (\d+)") or "nan") / 1024.0
    rows.append(r)

df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no jobs with hardware.txt under " + root)
df.to_csv(os.path.join("results", "run_metrics.csv"), index=False)
ok = df.exit_code == "0"
print("jobs: %d (%d finished without error) | CPU models: %s" % (len(df), ok.sum(), df.cpu_model.value_counts().to_dict()))
print("total CPU time: %.1f core-hours | summed wall time: %.1f hours" % (df.cpu_s.sum() / 3600, df.wall_s.sum() / 3600))
g = df[ok].groupby(["kind", "group", "model"]).agg(runs=("run", "size"), wall_min_mean=("wall_s", lambda s: s.mean() / 60),
                                                    wall_min_max=("wall_s", lambda s: s.max() / 60), cpu_hours=("cpu_s", lambda s: s.sum() / 3600),
                                                    peak_MB_max=("peak_rss_MB", "max"))
pd.set_option("display.width", 200)
print(g.round(1).to_string())
