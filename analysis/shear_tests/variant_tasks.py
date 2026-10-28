"""Task lines for the four variants of Supplementary Note 8 (seeds 42-44, 5% and 25% labeled), built from the commands of the
final model's runs. Writes tasks/shear_variants.tsv; run each line from the repository root (one core each)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VAR = {"head": "--valley_head", "sup": "--valley_sup 0.1", "focus": "--focus 3", "both": "--valley_head --valley_sup 0.1"}
lines = []
for seed in (42, 43, 44):
    for pct in (5, 25):
        base = open("%s/results/B_main/hyb_sdp_gap_ld9/%dpct/seed%d/command.txt" % (ROOT, pct, seed)).read().strip()
        for v, fl in VAR.items():
            out = "results/B_shear_variants/%s/%dpct/seed%d" % (v, pct, seed)
            cmd = base.replace("results/B_main/hyb_sdp_gap_ld9/%dpct/seed%d" % (pct, seed), out).replace("python 03_train_pinn.py", "python analysis/shear_tests/03_train_pinn_valley.py") + " " + fl
            lines.append("%s\t%s" % (out, cmd))
os.makedirs(ROOT + "/tasks", exist_ok=True); open(ROOT + "/tasks/shear_variants.tsv", "w").write("\n".join(lines) + "\n"); print(len(lines), "runs")
