"""Task lines for the four variants of Supplementary Note 8 (17 seeds, 5% and 25% labeled), built from the commands of the
proposed hybrid model's runs (make_tasks.py). Writes tasks/shear_variants.tsv; run each line from the repository root (one core each)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import config as C, make_tasks as MT
VAR = {"head": "--valley_head", "sup": "--valley_sup 0.1", "focus": "--focus 3", "both": "--valley_head --valley_sup 0.1"}
lines = []
for seed in C.SEEDS:
    for pct in (5, 25):
        out0, base = MT.main_runs(MT.FINAL, [pct], [seed])[0][0].split("\t")
        for v, fl in VAR.items():
            out = "results/B_shear_variants/%s/%dpct/seed%d" % (v, pct, seed)
            cmd = base.replace(out0, out).replace("python 03_train_pinn.py", "python analysis/shear_tests/03_train_pinn_valley.py") + " " + fl
            lines.append("%s\t%s" % (out, cmd))
os.makedirs("tasks", exist_ok=True); open("tasks/shear_variants.tsv", "w").write("\n".join(lines) + "\n"); print(len(lines), "runs")
