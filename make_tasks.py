#!/usr/bin/env python3
"""Task lists of every training run reported in the paper: one line per run, "<results_dir>\t<command>", with paths
relative to this folder. Each line runs 03_train_pinn.py on one processor core; run_task.sh executes one line,
submit.sh submits a whole list as SLURM job arrays.

  python make_tasks.py smoke         tasks/smoke.tsv         one short run (300 iterations) per command shape below;
                                                            compare_smoke.py checks them against reference_smoke/
  python make_tasks.py main          tasks/main.tsv          every model x 9 labeled fractions x 17 seeds (results/B_main)
  python make_tasks.py heldout       tasks/heldout.tsv       every family left out in turn and three strain cut-offs,
                                                            all training states labeled, 17 seeds (results/B_extrap)
  python make_tasks.py labelmatched  tasks/labelmatched.tsv  constants fitted on the labeled states (results/B_labelmatched)
  python make_tasks.py ablation      tasks/ablation.tsv      our model with one term of the objective removed (results/B_ablation)
  python make_tasks.py all           all of the above except smoke

Before training, the constants must exist: calibrate_constants.py, calibrate_heldout.py <scheme> and
calibrate_labelmatched.py (their outputs are included in results/). The commands are exactly those of the campaign that
produced the paper.
"""
import os, sys
import config as C

K = "results/98_constants_v2"
PROBE = ["--grad_probe_every", "250"]          # log the physics and data gradients every 250 iterations
ZS = ["--w_anchor", "0.0625", "--w_pcbm", "0.0625", "--w_pvbm", "0.0625"]
NEW = ["--no_rba", "--slope_fix", "--w_cons", "0", "--w_cbm", "0", "--w_vbm", "0"] + ZS
DZS = ("rba", NEW + ["--residual_phys", "--sym_colloc", "1.0"])                              # Delta-learning + zero-strain + symmetry terms
SDP = ("rba", NEW + ["--structured_head", "--sym_colloc", "1.0", "--curv_w", "0.0625"])      # constants as outputs, no agreement terms

# Model names of the paper (trainer mode, extra options). The trainer's --model selects the weighting rule:
# sa_paper / rba_paper = the published SA-PINN / RBA-PINN updates; rba with --no_rba = fixed weights.
MODELS = {
    "hyb_sdp_ld9":  SDP,                                               # OUR MODEL: constants as outputs + agreement terms, s = min(1, 9/n)
    "sa_pinn":      ("sa_paper", ["--slope_fix"]),                      # SA-PINN as published
    "rba_pinn":     ("rba_paper", ["--slope_fix"]),                     # RBA-PINN as published
    "mlp":          ("mlp", ["--no_rba"]),                              # MLP: data and isotropic terms only
    "delta":        ("mlp", ["--residual_phys", "--no_rba"]),           # Delta-learning
    "delta_zs":     ("rba", NEW + ["--residual_phys"]),                 # Delta-learning + zero-strain terms
    "delta_zs_sym": DZS,                                                # Delta-learning + zero-strain + symmetry terms
    "sdp_net":      SDP,                                                # our model without the agreement terms
    # the six weighting variants among which our model was selected on validation error (select_weighting.py)
    "hyb_sdp_ld45": SDP,                                                # our model, weight falling from 45 labeled states
    "hyb_fixed":    DZS,                                                # on Delta-learning, fixed weight
    "hyb_ld45":     DZS,                                                # on Delta-learning, weight falling from 45
    "hyb_ld9":      DZS,                                                # on Delta-learning, weight falling from 9
    "hyb_ld45_col": DZS,                                                # hyb_ld45 + formula-only collocation states
}
FINAL = "hyb_sdp_ld9"
VARIANTS = ["hyb_fixed", "hyb_ld45", "hyb_ld9", "hyb_sdp_ld45", "hyb_sdp_ld9", "hyb_ld45_col"]   # order of the selection table
HELDOUT_MODELS = [FINAL, "sa_pinn", "rba_pinn", "mlp", "delta", "delta_zs", "delta_zs_sym", "sdp_net"]
LM_FRACTIONS = [1, 2, 5, 10]

# single-core minutes per run (workstation, all cores busy); only used to order the lists, longest first
MIN = {"mlp": 14.9, "sa_pinn": 21.3, "rba_pinn": 18.7, "delta": 19.9, "delta_zs": 20.0, "delta_zs_sym": 22.5, "sdp_net": 49.8,
       "hyb_fixed": 22.5, "hyb_ld45": 22.5, "hyb_ld9": 22.5, "hyb_sdp_ld45": 49.8, "hyb_sdp_ld9": 49.8, "hyb_ld45_col": 40.0}


def agree(n0):
    # agreement terms weighted by s = min(1, n0 / n) for n labeled states (n0 None: fixed weight)
    def f(p):
        n = round(903 * p / 100); s_ = 1.0 if (n0 is None or n <= n0) else n0 / n
        return ["--w_cons", "%g" % (2 * s_), "--w_cbm", "%g" % s_, "--w_vbm", "%g" % s_]
    return f


def colloc(n0):
    # formula-only collocation states over the dataset's strain range, with the same falling weight
    def f(p):
        n = round(903 * p / 100); s_ = 1.0 if (n0 is None or n <= n0) else n0 / n
        return ["--colloc_n", "2000", "--colloc_max", "0.105", "--w_colloc", "%g" % s_]
    return f


LEVEL_FLAGS = {"hyb_fixed": [agree(None)], "hyb_ld45": [agree(45)], "hyb_ld9": [agree(9)], "hyb_sdp_ld45": [agree(45)],
               "hyb_sdp_ld9": [agree(9)], "hyb_ld45_col": [agree(45), colloc(45)]}
SUITE = {"group_isotropic": ["--split", "group", "--holdout_family", "isotropic"],
         "group_uniaxial": ["--split", "group", "--holdout_family", "uniaxial"],
         "group_biaxial": ["--split", "group", "--holdout_family", "biaxial"],
         "group_triaxial": ["--split", "group", "--holdout_family", "triaxial"],
         "group_shear": ["--split", "group", "--holdout_family", "shear"],
         "magnitude_03": ["--split", "magnitude", "--magnitude_thresh", "0.03"],
         "magnitude_05": ["--split", "magnitude", "--magnitude_thresh", "0.05"],
         "magnitude_07": ["--split", "magnitude", "--magnitude_thresh", "0.07"]}
ABLATION = {"drop_cbm_vbm": ["--drop_loss", "cbm", "--drop_loss", "vbm"], "drop_cons": ["--drop_loss", "cons"],
            "drop_agreement": ["--drop_loss", "cons", "--drop_loss", "cbm", "--drop_loss", "vbm"],
            "drop_zero_strain": ["--drop_loss", "anchor", "--drop_loss", "pcbm", "--drop_loss", "pvbm"],
            "drop_iso_eg": ["--drop_loss", "iso_eg"], "drop_sym": ["--sym_colloc", "0"], "drop_curv": ["--curv_w", "0"]}
ABLATION_LEVELS = {"drop_cbm_vbm": [0, 1, 2, 5, 25, 100], "drop_cons": [0, 1, 2, 5, 25, 100], "drop_agreement": [0, 1, 2, 5, 25, 100],
                   "drop_iso_eg": [0, 5, 25, 100], "drop_zero_strain": [5, 25, 100], "drop_sym": [5, 25, 100], "drop_curv": [5, 25, 100]}


def lflags(name, p):
    return sum([f(p) for f in LEVEL_FLAGS.get(name, [])], [])


def cmd(model, seed, pct, out, extra, const, epochs=None):
    c = "python 03_train_pinn.py --data_csv data/dft_hy_v9.csv --results_dir %s --seed %d --dft_pct %d --epochs %d --device cpu --model %s --depth %d --width %d" % (
        out, seed, pct, epochs or C.TUNED["epochs"], model, C.TUNED["depth"], C.TUNED["width"])
    c += "".join(" --w_%s %g" % (k, v) for k, v in C.LOSS_WEIGHTS.items())
    return "%s\tPHYSICS_CONST_JSON=%s %s %s" % (out, const, c, " ".join(extra))


def main_runs(name, fractions=None, seeds=None, root="results/B_main", epochs=None):
    m, ex = MODELS[name]
    return [(cmd(m, s, p, "%s/%s/%dpct/seed%d" % (root, name, p, s), ["--split", "symmetry"] + list(ex) + PROBE + lflags(name, p),
                 "%s/symmetry/const_seed%d.json" % (K, s), epochs), MIN[name]) for p in (fractions or C.FRACTIONS) for s in (seeds or C.SEEDS)]


def heldout_runs(name):
    m, ex = MODELS[name]
    return [(cmd(m, s, 100, "results/B_extrap/%s/%s/seed%d" % (name, k, s), sx + list(ex) + lflags(name, 100), "%s/%s/const_seed%d.json" % (K, k, s)), MIN[name])
            for k, sx in SUITE.items() for s in C.SEEDS]


def labelmatched_runs():
    out = []
    for name in [FINAL, "delta", "sdp_net"]:
        m, ex = MODELS[name]
        out += [(cmd(m, s, p, "results/B_labelmatched/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + lflags(name, p),
                     "results/100_constants_labelmatched/%dpct/const_seed%d.json" % (p, s)), MIN[name]) for p in LM_FRACTIONS for s in C.SEEDS]
    return out


def ablation_runs():
    m, ex = MODELS[FINAL]
    return [(cmd(m, s, p, "results/B_ablation/%s_%s/%dpct/seed%d" % (FINAL, dname, p, s), ["--split", "symmetry"] + list(ex) + lflags(FINAL, p) + fl,
                 "%s/symmetry/const_seed%d.json" % (K, s)), MIN[FINAL])
            for dname, fl in ABLATION.items() for p in ABLATION_LEVELS[dname] for s in C.SEEDS]


def smoke_runs():
    out = []
    for name in MODELS:
        line = main_runs(name, [25], [42], "results/smoke", 300)[0][0]
        out.append((line.replace("results/smoke/%s/25pct/seed42" % name, "results/smoke/%s" % name), 1.0))
    m, ex = MODELS[FINAL]
    out.append((cmd(m, 42, 100, "results/smoke/x_%s_group_shear" % FINAL, SUITE["group_shear"] + list(ex) + lflags(FINAL, 100),
                    "%s/group_shear/const_seed42.json" % K, 300), 1.0))
    out.append((cmd(m, 42, 5, "results/smoke/a_%s_drop_agreement" % FINAL, ["--split", "symmetry"] + list(ex) + lflags(FINAL, 5) + ABLATION["drop_agreement"],
                    "%s/symmetry/const_seed42.json" % K, 300), 1.0))
    out.append((cmd(m, 42, 1, "results/smoke/lm_%s_1pct" % FINAL, ["--split", "symmetry"] + list(ex) + lflags(FINAL, 1),
                    "results/100_constants_labelmatched/1pct/const_seed42.json", 300), 1.0))
    return out


def write(name, runs, longest_first=True):
    if longest_first:
        runs = sorted(runs, key=lambda r: -r[1])
    os.makedirs("tasks", exist_ok=True)
    path = "tasks/%s.tsv" % name
    with open(path, "w") as f:
        f.write("".join(line + "\n" for line, _ in runs))
    outs = [line.split("\t")[0] for line, _ in runs]
    assert len(outs) == len(set(outs)), "two runs share a results folder in " + path
    print("wrote %-24s %5d runs  %6.0f workstation core-hours" % (path, len(runs), sum(mn for _, mn in runs) / 60.0))


LISTS = {"main": lambda: sum([main_runs(n) for n in MODELS], []),
         "heldout": lambda: sum([heldout_runs(n) for n in HELDOUT_MODELS], []),
         "labelmatched": labelmatched_runs, "ablation": ablation_runs}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "smoke":
        write("smoke", smoke_runs(), longest_first=False)
    elif which in LISTS:
        write(which, LISTS[which]())
    elif which == "all":
        for k, f in LISTS.items():
            write(k, f())
    else:
        sys.exit(__doc__)
