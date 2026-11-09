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
  python make_tasks.py ablation      tasks/ablation.tsv      the PHM with one term of the objective removed (results/B_ablation)
  python make_tasks.py all           all of the above except smoke

Before training, the constants must exist: calibrate_constants.py, calibrate_heldout.py <scheme> and
calibrate_labelmatched.py (their outputs are included in results/). The commands are exactly those of the campaign that
produced the paper.
"""
import os, sys
import config as C

K = "results/98_constants_2band"                  # 14 constants per run, two-band conduction-band shear term
KLM = "results/100_constants_labelmatched_2band"
PROBE = ["--grad_probe_every", "250"]          # log the physics and data gradients every 250 iterations
ZS = ["--w_anchor", "0.0625", "--w_pcbm", "0.0625", "--w_pvbm", "0.0625"]   # zero-strain terms (selection variants; tested as an addition)
NOZS = ["--w_anchor", "0", "--w_pcbm", "0", "--w_pvbm", "0"]
NEW0 = ["--no_rba", "--slope_fix", "--w_cons", "0", "--w_cbm", "0", "--w_vbm", "0"]
NEW = NEW0 + ZS
DZS = ("rba", NEW + ["--residual_phys", "--sym_colloc", "1.0"])                              # Delta-learning + zero-strain + symmetry terms
SDP = ("rba", NEW + ["--structured_head", "--sym_colloc", "1.0"])                           # constants as outputs, zero-strain terms, no agreement terms
PHM = ("rba", NEW0 + NOZS + ["--structured_head", "--sym_colloc", "1.0"])                  # constants as outputs, no zero-strain or agreement terms

# Model names of the paper (trainer mode, extra options). The trainer's --model selects the weighting rule:
# sa_paper / rba_paper = the published SA-PINN / RBA-PINN updates; rba with --no_rba = fixed weights.
GAP = PHM[1] + ["--drop_loss", "cbm", "--drop_loss", "vbm"]            # agreement with the analytic gap only
# Delta-learning on exactly the objective of the PHM (data, isotropic gap, gap agreement with the falling weight s, symmetry); only the way the
# theory enters differs: an additive correction to the analytic band edges (--residual_phys) instead of strain-dependent constants (--structured_head)
GAP_RESIDUAL = NEW0 + NOZS + ["--residual_phys", "--sym_colloc", "1.0", "--drop_loss", "cbm", "--drop_loss", "vbm"]
# Improvement screen for the matched residual (4 Nov 2026; output stays 2 band edges): r01 starts the correction at zero,
# r03 also keeps both band-edge corrections small (band-edge agreement terms on, same falling weight)
RES_ZERO = GAP_RESIDUAL + ["--zero_out"]
RES_VOL = RES_ZERO + ["--agree_volfix"]     # round 2: agreement target = theory + the isotropic correction at the same volume change
RES_EDGES = NEW0 + NOZS + ["--residual_phys", "--sym_colloc", "1.0", "--zero_out"]
MATCHED = NOZS + ["--drop_loss", "cbm", "--drop_loss", "vbm", "--sym_colloc", "1.0", "--sym_adaptive"]   # the PHM's terms, for the PINN control
MODELS = {
    "phm":          (SDP[0], GAP),                                      # PROPOSED HYBRID MODEL (PHM): constants as outputs, gap agreement weighted by s = min(1, (9/n)^3)
    "hyb_sdp_gap_fixed": (SDP[0], GAP),                                # PHM with a fixed agreement weight (s = 1)
    # other candidate weights of the PHM, compared on validation error (select_phm_weight.py)
    "hyb_sdp_gap_ld9": (SDP[0], GAP),                                  # s = min(1, 9/n)
    "phm_ld9sq":    (SDP[0], GAP),                                      # s = min(1, (9/n)^2)
    "phm_ld4":      (SDP[0], GAP),                                      # s = min(1, 4/n)
    "sa_pinn":      ("sa_paper", ["--slope_fix"]),                      # SA-PINN as published
    "rba_pinn":     ("rba_paper", ["--slope_fix"]),                     # RBA-PINN as published
    "ntk_pinn":     ("ntk_paper", ["--slope_fix"]),                     # NTK-weighted PINN (Wang, Yu & Perdikaris 2022, Algorithm 1)
    # the three PINNs with the terms of the PHM (data, isotropic, gap agreement, symmetry), their own weighting rules applied to every
    # physics term including the symmetry term
    "sa_pinn_phmterms":  ("sa_paper", ["--slope_fix"] + MATCHED),
    "rba_pinn_phmterms": ("rba_paper", ["--slope_fix"] + MATCHED),
    "ntk_pinn_phmterms": ("ntk_paper", ["--slope_fix"] + MATCHED),
    "aphynity":     ("mlp", ["--aphynity", "--no_rba", "--aph_tau2", "10"]),   # APHYNITY (Yin et al. 2021); tau2 = 10 chosen on validation error
    "mlp":          ("mlp", ["--no_rba"]),                              # MLP: data and isotropic terms only
    "delta":        ("mlp", ["--residual_phys", "--no_rba"]),           # Delta-learning
    "delta_zs":     ("rba", NEW + ["--residual_phys"]),                 # Delta-learning + zero-strain terms
    "delta_zs_sym": DZS,                                                # Delta-learning + zero-strain + symmetry terms
    "delta_phmterms": ("rba", GAP_RESIDUAL),                            # matched control: Delta-learning on the PHM's objective, agreement weighted by s = min(1, (9/n)^3)
    "sdp_net":      PHM,                                                # PHM without the agreement term
    # improvement screen for the matched residual (B_resfix); weights in RESFIX_FLAGS below
    "res_zero":       ("rba", RES_ZERO),                                # r01: correction starts at zero
    "res_zero_a5":    ("rba", RES_ZERO),                                # r02: r01 + gap agreement weight 5s
    "res_zero_a10":   ("rba", RES_ZERO),                                # r02: r01 + gap agreement weight 10s
    "res_zero_edges": ("rba", RES_EDGES),                               # r03: r01 + band-edge agreement terms (weight s each)
    "res_zero_ld9":   ("rba", RES_ZERO),                                # r04: r01 with s = min(1, 9/n)
    "res_zero_ld9sq": ("rba", RES_ZERO),                                # r04: r01 with s = min(1, (9/n)^2)
    "res_zero_ld4":   ("rba", RES_ZERO),                                # r04: r01 with s = min(1, 4/n)
    "res_zero_cut18": ("rba", RES_ZERO),                                # r04: r01 with s = min(1, 9/n) up to 18 labeled states, 0 beyond
    "res_vol":        ("rba", RES_VOL),                                 # r05: r01 + volume-fixed agreement target
    "res_iso2p5":     ("rba", RES_ZERO + ["--w_iso_eg", "2.5"]),        # r06: r01 + isotropic term weight 2.5 (5 elsewhere)
    "res_iso1":       ("rba", RES_ZERO + ["--w_iso_eg", "1"]),          # r07: r01 + isotropic term weight 1
    "res_vol_iso2p5": ("rba", RES_VOL + ["--w_iso_eg", "2.5"]),         # r08: r05 + isotropic weight 2.5
    "res_vol_iso1":   ("rba", RES_VOL + ["--w_iso_eg", "1"]),           # r09: r05 + isotropic weight 1
    "res_vol_a5":     ("rba", RES_VOL),                                 # r10: r05 + gap agreement weight 5s
    "res_symlab":     ("rba", RES_ZERO + ["--sym_labels", "2000"]),     # r11: r01 + symmetry-copied labels (at most 2000 images)
    "res_curric":     ("rba", RES_ZERO + ["--curric_shear", "0.25"]),   # r12: r01 + shear states left out for the first quarter
    "res_lowstrain":  ("rba", RES_ZERO + ["--lowstrain_w"]),            # r13: r01 + low-strain weighting 1 + 4 exp(-M/0.01)
    "res_symlab_low": ("rba", RES_ZERO + ["--sym_labels", "2000", "--lowstrain_w"]),   # r14: r11 + r13
    "res_symlab_bal": ("rba", RES_ZERO + ["--sym_labels", "2000", "--sym_labels_balanced"]),   # r15: r11, each state and its images share one weight
    # fairness controls for the copied labels: the MLP and Delta-learning of the paper with the same balanced copied labels
    "mlp_symlab_bal":   ("mlp", ["--no_rba", "--sym_labels", "2000", "--sym_labels_balanced"]),
    "delta_symlab_bal": ("mlp", ["--residual_phys", "--no_rba", "--sym_labels", "2000", "--sym_labels_balanced"]),
    # the six weighting variants among which the weighting rule was selected on validation error (select_weighting.py)
    "hyb_sdp_ld9":  SDP,                                                # constants as outputs, weight falling from 9 labeled states (selected)
    "hyb_sdp_ld45": SDP,                                                # constants as outputs, weight falling from 45 labeled states
    "hyb_fixed":    DZS,                                                # on Delta-learning, fixed weight
    "hyb_ld45":     DZS,                                                # on Delta-learning, weight falling from 45
    "hyb_ld9":      DZS,                                                # on Delta-learning, weight falling from 9
    "hyb_ld45_col": DZS,                                                # hyb_ld45 + formula-only collocation states
}
FINAL = "phm"
VARIANTS = ["hyb_fixed", "hyb_ld45", "hyb_ld9", "hyb_sdp_ld45", "hyb_sdp_ld9", "hyb_ld45_col"]   # order of the selection table
HELDOUT_MODELS = [FINAL, "hyb_sdp_gap_fixed", "sa_pinn", "rba_pinn", "ntk_pinn", "aphynity", "mlp", "delta", "delta_zs", "delta_zs_sym", "sdp_net"]
LM_FRACTIONS = [1, 2, 5, 10]

# single-core minutes per run (workstation, all cores busy); only used to order the lists, longest first
MIN = {"phm": 49.8, "hyb_sdp_gap_ld9": 49.8, "hyb_sdp_gap_fixed": 49.8, "phm_ld9sq": 49.8, "phm_ld4": 49.8, "delta_phmterms": 25.0, "ntk_pinn": 30.0, "sa_pinn_phmterms": 30.0, "rba_pinn_phmterms": 30.0, "ntk_pinn_phmterms": 40.0, "aphynity": 20.0, "mlp": 14.9, "sa_pinn": 21.3, "rba_pinn": 18.7, "delta": 19.9, "delta_zs": 20.0, "delta_zs_sym": 22.5, "sdp_net": 49.8,
       "res_zero": 25.0, "res_zero_a5": 25.0, "res_zero_a10": 25.0, "res_zero_edges": 25.0, "res_zero_ld9": 25.0, "res_zero_ld9sq": 25.0,
       "res_zero_ld4": 25.0, "res_zero_cut18": 25.0, "res_vol": 30.0, "res_iso2p5": 25.0, "res_iso1": 25.0, "res_vol_iso2p5": 30.0,
       "res_vol_iso1": 30.0, "res_vol_a5": 30.0, "res_symlab": 35.0, "res_curric": 25.0, "res_lowstrain": 25.0, "res_symlab_low": 35.0, "res_symlab_bal": 35.0, "mlp_symlab_bal": 20.0, "delta_symlab_bal": 20.0,
       "hyb_fixed": 22.5, "hyb_ld45": 22.5, "hyb_ld9": 22.5, "hyb_sdp_ld45": 49.8, "hyb_sdp_ld9": 49.8, "hyb_ld45_col": 40.0}


def agree(n0, power=1):
    # agreement terms weighted by s = min(1, (n0 / n)^power) for n labeled states (n0 None: fixed weight)
    def f(p):
        n = round(903 * p / 100); s_ = 1.0 if (n0 is None or n <= n0) else (n0 / n) ** power
        return ["--w_cons", "%g" % (2 * s_), "--w_cbm", "%g" % s_, "--w_vbm", "%g" % s_]
    return f


def agree_s(rule, mult=1.0):
    # agreement terms weighted by s = rule(n) for n labeled states; the gap term takes 2 * mult * s, each band-edge term s
    def f(p):
        n = round(903 * p / 100); s_ = rule(n)
        return ["--w_cons", "%g" % (2 * mult * s_), "--w_cbm", "%g" % s_, "--w_vbm", "%g" % s_]
    return f


RULES = {"cube": lambda n: 1.0 if n <= 9 else (9 / n) ** 3, "sq": lambda n: 1.0 if n <= 9 else (9 / n) ** 2,
         "lin": lambda n: 1.0 if n <= 9 else 9 / n, "ld4": lambda n: 1.0 if n <= 4 else 4 / n,
         "cut18": lambda n: (1.0 if n <= 9 else 9 / n) if n <= 18 else 0.0}


def colloc(n0):
    # formula-only collocation states over the dataset's strain range, with the same falling weight
    def f(p):
        n = round(903 * p / 100); s_ = 1.0 if (n0 is None or n <= n0) else n0 / n
        return ["--colloc_n", "2000", "--colloc_max", "0.105", "--w_colloc", "%g" % s_]
    return f


LEVEL_FLAGS = {"hyb_fixed": [agree(None)], "hyb_ld45": [agree(45)], "hyb_ld9": [agree(9)], "hyb_sdp_ld45": [agree(45)],
               "hyb_sdp_ld9": [agree(9)], "hyb_sdp_gap_ld9": [agree(9)], "hyb_sdp_gap_fixed": [agree(None)],
               "phm_ld9sq": [agree(9, 2)], "phm": [agree(9, 3)], "delta_phmterms": [agree(9, 3)], "phm_ld4": [agree(4)], "hyb_ld45_col": [agree(45), colloc(45)]}
# improvement screen for the matched residual: weighting rule and agreement multiplier, and the fractions screened (seeds 42-46)
RESFIX_FLAGS = {"res_zero": [agree_s(RULES["cube"])], "res_zero_a5": [agree_s(RULES["cube"], 2.5)], "res_zero_a10": [agree_s(RULES["cube"], 5.0)],
                "res_zero_edges": [agree_s(RULES["cube"])], "res_zero_ld9": [agree_s(RULES["lin"])], "res_zero_ld9sq": [agree_s(RULES["sq"])],
                "res_zero_ld4": [agree_s(RULES["ld4"])], "res_zero_cut18": [agree_s(RULES["cut18"])],
                "res_vol": [agree_s(RULES["cube"])], "res_iso2p5": [agree_s(RULES["cube"])], "res_iso1": [agree_s(RULES["cube"])],
                "res_vol_iso2p5": [agree_s(RULES["cube"])], "res_vol_iso1": [agree_s(RULES["cube"])], "res_vol_a5": [agree_s(RULES["cube"], 2.5)],
                "res_symlab": [agree_s(RULES["cube"])], "res_curric": [agree_s(RULES["cube"])], "res_lowstrain": [agree_s(RULES["cube"])],
                "res_symlab_low": [agree_s(RULES["cube"])], "res_symlab_bal": [agree_s(RULES["cube"])]}
LEVEL_FLAGS.update(RESFIX_FLAGS)
RESFIX_LEVELS = {"res_zero": [0, 1, 2, 5, 25, 100], "res_zero_edges": [0, 1, 2, 5, 25, 100], "res_zero_a5": [0, 1, 2, 5], "res_zero_a10": [0, 1, 2, 5],
                 "res_zero_ld9": [2, 5, 25, 100], "res_zero_ld9sq": [2, 5, 25, 100], "res_zero_cut18": [2, 5, 25, 100], "res_zero_ld4": [1, 2, 5, 25, 100]}
RESFIX_SEEDS = [42, 43, 44, 45, 46]
RESFIX2 = ["res_vol", "res_iso2p5", "res_iso1", "res_vol_iso2p5", "res_vol_iso1", "res_vol_a5"]   # round 2 (cause 2: isotropic correction spreading)
RESFIX_LEVELS.update({n: [0, 1, 2, 5, 25, 100] for n in RESFIX2})
RESFIX3 = ["res_symlab", "res_curric", "res_lowstrain", "res_symlab_low"]   # round 3: copied labels, curriculum, low-strain weight
RESFIX_LEVELS.update({"res_symlab": [1, 2, 5, 25, 100], "res_curric": [0, 1, 2, 5, 25, 100], "res_lowstrain": [0, 1, 2, 5, 25, 100],
                      "res_symlab_low": [1, 2, 5, 25, 100], "res_symlab_bal": [1, 2, 5, 25, 100]})
RESFIX4 = ["res_symlab_bal"]
SUITE = {"group_isotropic": ["--split", "group", "--holdout_family", "isotropic"],
         "group_uniaxial": ["--split", "group", "--holdout_family", "uniaxial"],
         "group_biaxial": ["--split", "group", "--holdout_family", "biaxial"],
         "group_triaxial": ["--split", "group", "--holdout_family", "triaxial"],
         "group_shear": ["--split", "group", "--holdout_family", "shear"],
         "magnitude_03": ["--split", "magnitude", "--magnitude_thresh", "0.03"],
         "magnitude_05": ["--split", "magnitude", "--magnitude_thresh", "0.05"],
         "magnitude_07": ["--split", "magnitude", "--magnitude_thresh", "0.07"]}
ABLATION = {"drop_iso_eg": ["--drop_loss", "iso_eg"], "drop_sym": ["--sym_colloc", "0"]}
ABLATION_LEVELS = {"drop_iso_eg": [0, 5, 25, 100], "drop_sym": [5, 25, 100]}
# curvature condition at zero strain, tested as an addition (not part of the final objective): all fractions and the held-out suite
ADD_CURV = ["--curv_w", "0.0625"]
# without the agreement term: no agreement term at all (a folder of its own, since it is a different objective)
NOAGREE = ["--drop_loss", "cons", "--drop_loss", "cbm", "--drop_loss", "vbm"]; NOAGREE_LEVELS = [0, 1, 2, 5, 25, 100]


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
    for name in [FINAL, "delta", "sdp_net", "delta_phmterms"]:
        m, ex = MODELS[name]
        out += [(cmd(m, s, p, "results/B_labelmatched/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + lflags(name, p),
                     "%s/%dpct/const_seed%d.json" % (KLM, p, s)), MIN[name]) for p in LM_FRACTIONS for s in C.SEEDS]
    return out


def ablation_runs():
    m, ex = MODELS[FINAL]
    runs = [(cmd(m, s, p, "results/B_ablation/%s_%s/%dpct/seed%d" % (FINAL, dname, p, s), ["--split", "symmetry"] + list(ex) + lflags(FINAL, p) + fl,
                 "%s/symmetry/const_seed%d.json" % (K, s)), MIN[FINAL])
            for dname, fl in ABLATION.items() for p in ABLATION_LEVELS[dname] for s in C.SEEDS]
    runs += [(cmd(m, s, p, "results/B_ablation/%s_drop_agreement/%dpct/seed%d" % (FINAL, p, s), ["--split", "symmetry"] + list(ex) + lflags(FINAL, p) + NOAGREE,
                  "%s/symmetry/const_seed%d.json" % (K, s)), MIN[FINAL]) for p in NOAGREE_LEVELS for s in C.SEEDS]
    # the zero-strain terms, tested as an addition (not part of the final objective): all fractions
    runs += [(cmd(m, s, p, "results/B_ablation/%s_add_zero_strain/%dpct/seed%d" % (FINAL, p, s), ["--split", "symmetry"] + list(ex) + PROBE + lflags(FINAL, p) + ZS,
                  "%s/symmetry/const_seed%d.json" % (K, s)), MIN[FINAL]) for p in C.FRACTIONS for s in C.SEEDS]
    # the linear shear term 2 Xi_u' |e_bc| in place of the two-band form (13 constants, results/98_constants_v2): all fractions and the held-out suite
    name, K1 = FINAL + "_linear_shear", "results/98_constants_v2"
    runs += [(cmd(m, s, p, "results/B_ablation/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + PROBE + lflags(FINAL, p),
                  "%s/symmetry/const_seed%d.json" % (K1, s)), MIN[FINAL]) for p in C.FRACTIONS for s in C.SEEDS]
    runs += [(cmd(m, s, 100, "results/B_extrap/%s/%s/seed%d" % (name, k, s), sx + list(ex) + lflags(FINAL, 100), "%s/%s/const_seed%d.json" % (K1, k, s)), MIN[FINAL])
             for k, sx in SUITE.items() for s in C.SEEDS]
    name = FINAL + "_add_curv"
    runs += [(cmd(m, s, p, "results/B_ablation/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + PROBE + lflags(FINAL, p) + ADD_CURV,
                  "%s/symmetry/const_seed%d.json" % (K, s)), MIN[FINAL]) for p in C.FRACTIONS for s in C.SEEDS]
    runs += [(cmd(m, s, 100, "results/B_extrap/%s/%s/seed%d" % (name, k, s), sx + list(ex) + lflags(FINAL, 100) + ADD_CURV, "%s/%s/const_seed%d.json" % (K, k, s)), MIN[FINAL])
             for k, sx in SUITE.items() for s in C.SEEDS]
    return runs


def smoke_runs():
    out = []
    for name in MODELS:
        line = main_runs(name, [25], [42], "results/smoke", 300)[0][0]
        out.append((line.replace("results/smoke/%s/25pct/seed42" % name, "results/smoke/%s" % name), 1.0))
    m, ex = MODELS[FINAL]
    out.append((cmd(m, 42, 100, "results/smoke/x_%s_group_shear" % FINAL, SUITE["group_shear"] + list(ex) + lflags(FINAL, 100),
                    "%s/group_shear/const_seed42.json" % K, 300), 1.0))
    out.append((cmd(m, 42, 5, "results/smoke/a_%s_add_zero_strain" % FINAL, ["--split", "symmetry"] + list(ex) + lflags(FINAL, 5) + ZS,
                    "%s/symmetry/const_seed42.json" % K, 300), 1.0))
    out.append((cmd(m, 42, 1, "results/smoke/lm_%s_1pct" % FINAL, ["--split", "symmetry"] + list(ex) + lflags(FINAL, 1),
                    "%s/1pct/const_seed42.json" % KLM, 300), 1.0))
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


def matched_runs():
    # the matched residual control only: all fractions, the held-out suite and the label-matched calibration
    out = main_runs("delta_phmterms") + heldout_runs("delta_phmterms")
    m, ex = MODELS["delta_phmterms"]
    out += [(cmd(m, s, p, "results/B_labelmatched/delta_phmterms/%dpct/seed%d" % (p, s), ["--split", "symmetry"] + list(ex) + lflags("delta_phmterms", p),
                 "%s/%dpct/const_seed%d.json" % (KLM, p, s)), MIN["delta_phmterms"]) for p in LM_FRACTIONS for s in C.SEEDS]
    return out


def resfix_runs():
    # improvement screen for the matched residual: 5 seeds, the fractions where each change can act (B_resfix)
    return sum([main_runs(n, RESFIX_LEVELS[n], RESFIX_SEEDS, root="results/B_resfix") for n in RESFIX_LEVELS if n not in RESFIX2 + RESFIX3 + RESFIX4], [])


def resfix2_runs():
    # round 2 of the screen: the isotropic correction and how it spreads (B_resfix, seeds 42-46)
    return sum([main_runs(n, RESFIX_LEVELS[n], RESFIX_SEEDS, root="results/B_resfix") for n in RESFIX2], [])


SYMBAL_CONTROLS = ["mlp_symlab_bal", "delta_symlab_bal"]


def symbal_confirm_runs():
    # full check of r15 (res_symlab_bal): 9 fractions x 17 seeds, held-out suite, label-matched calibration; and the two
    # fairness controls at 1-100% (at 0% they equal the MLP and Delta-learning: no labels to copy)
    name = "res_symlab_bal"; m, ex = MODELS[name]
    out = main_runs(name) + heldout_runs(name)
    out += [(cmd(m, s, p, "results/B_labelmatched/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + lflags(name, p),
                 "%s/%dpct/const_seed%d.json" % (KLM, p, s)), MIN[name]) for p in LM_FRACTIONS for s in C.SEEDS]
    out += sum([main_runs(c, [p for p in C.FRACTIONS if p > 0]) for c in SYMBAL_CONTROLS], [])
    return out


# ---- runs asked for by the independent review (6 Nov 2026); results/B_review, and new cells of results/B_labelmatched
# R2: the PHM with the same balanced copied labels as res_symlab_bal
# R3: SA-PINN and RBA-PINN with the agreement weight falling as for the PHM (2s on the gap, s on each band edge; the band-edge
#     terms stay off in the *_phmterms versions). NTK-PINN is left out: its rule replaces the base weights.
REVIEW_MODELS = {
    "phm_symlab_bal":         ("rba", GAP + ["--sym_labels", "2000", "--sym_labels_balanced"]),
    "sa_pinn_fade":           MODELS["sa_pinn"],
    "rba_pinn_fade":          MODELS["rba_pinn"],
    "sa_pinn_phmterms_fade":  MODELS["sa_pinn_phmterms"],
    "rba_pinn_phmterms_fade": MODELS["rba_pinn_phmterms"],
}
MODELS.update(REVIEW_MODELS)
LEVEL_FLAGS.update({n: [agree(9, 3)] for n in REVIEW_MODELS})
MIN.update({"phm_symlab_bal": 60.0, "sa_pinn_fade": 21.3, "rba_pinn_fade": 18.7, "sa_pinn_phmterms_fade": 30.0, "rba_pinn_phmterms_fade": 30.0})
KSOFT = "results/98_constants_2band_softmin"
# R1: label-matched constants for more models (1-10%) and at 25-75% (at 100% they are the constants of the main suite)
LM_NEW = ["delta_symlab_bal", "aphynity", "sa_pinn", "rba_pinn", "ntk_pinn", "sa_pinn_phmterms", "rba_pinn_phmterms", "ntk_pinn_phmterms"]
LM_OLD = [FINAL, "delta", "sdp_net", "delta_phmterms", "res_symlab_bal"]
# R5: 0% without the isotropic gaps (--drop_loss iso_eg), with validation selection and with the last iteration kept
NOISO_MODELS = [FINAL, "hyb_sdp_gap_fixed", "sdp_net", "delta_phmterms", "res_symlab_bal", "aphynity",
                "sa_pinn", "rba_pinn", "ntk_pinn", "sa_pinn_phmterms", "rba_pinn_phmterms", "ntk_pinn_phmterms"]
# R4: constants fitted in the soft-minimum form of training
SOFT_MODELS = ["delta", "delta_phmterms", "res_symlab_bal", "delta_symlab_bal", FINAL]
SOFT_LEVELS = [0, 1, 2, 5, 25, 100]
# R7: fresh seeds 59-75 (constants: calibrate_constants.py --two_band --seeds 59-75)
FRESH_SEEDS = list(range(59, 76))
FRESH_MODELS = [FINAL, "delta", "delta_phmterms", "res_symlab_bal", "delta_symlab_bal"]


def _lm(name, pcts):
    m, ex = MODELS[name]
    return [(cmd(m, s, p, "results/B_labelmatched/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + lflags(name, p),
                 "%s/%dpct/const_seed%d.json" % (KLM, p, s)), MIN[name]) for p in pcts for s in C.SEEDS]


def _levels(name, levels):
    # the copied-label models have nothing to copy at 0%
    return [p for p in levels if not (p == 0 and "symlab" in name)]


def review_runs():
    out = sum([_lm(n, LM_FRACTIONS) for n in LM_NEW], []) + sum([_lm(n, [25, 50, 75]) for n in LM_NEW + LM_OLD], [])
    out += main_runs("phm_symlab_bal", [p for p in C.FRACTIONS if p > 0], root="results/B_review/main")
    out += sum([main_runs(n, [p for p in C.FRACTIONS if p >= 2], root="results/B_review/main") for n in REVIEW_MODELS if n != "phm_symlab_bal"], [])
    for tag, fl in (("noiso", ["--drop_loss", "iso_eg"]), ("noiso_last", ["--drop_loss", "iso_eg", "--select_last"])):
        for name in NOISO_MODELS:
            m, ex = MODELS[name]
            out += [(cmd(m, s, 0, "results/B_review/%s/%s/0pct/seed%d" % (tag, name, s), ["--split", "symmetry"] + list(ex) + lflags(name, 0) + fl,
                         "%s/symmetry/const_seed%d.json" % (K, s)), MIN[name]) for s in C.SEEDS]
    for name in SOFT_MODELS:
        m, ex = MODELS[name]
        out += [(cmd(m, s, p, "results/B_review/softmin/%s/%dpct/seed%d" % (name, p, s), ["--split", "symmetry"] + list(ex) + lflags(name, p),
                     "%s/symmetry/const_seed%d.json" % (KSOFT, s)), MIN[name]) for p in _levels(name, SOFT_LEVELS) for s in C.SEEDS]
    return out


def review_fresh_runs():
    return sum([main_runs(n, _levels(n, C.FRACTIONS), FRESH_SEEDS, root="results/B_review/fresh") for n in FRESH_MODELS], [])


# ---- constants inside the 2-output model (user idea, 6 Nov 2026): band edges = analytic models with network-adjusted
# constants (as the PHM) + additive correction (as r15); screen on seeds 42-46, also r15 with a doubled iteration budget
CONST_MODELS = {
    "res_const":          ("rba", RES_ZERO + ["--const_head"]),
    "res_const_symlab_bal": ("rba", RES_ZERO + ["--const_head", "--sym_labels", "2000", "--sym_labels_balanced"]),
}
MODELS.update(CONST_MODELS)
LEVEL_FLAGS.update({n: [agree_s(RULES["cube"])] for n in CONST_MODELS})
MIN.update({"res_const": 60.0, "res_const_symlab_bal": 70.0})
CONST_LEVELS = [0, 1, 2, 5, 25, 100]


def consthead_runs():
    root = "results/B_consthead"
    out = main_runs("res_const", CONST_LEVELS, RESFIX_SEEDS, root=root) + main_runs("res_const_symlab_bal", CONST_LEVELS, RESFIX_SEEDS, root=root)
    for name in ("res_const_symlab_bal", "res_symlab_bal"):     # doubled budget: 48,000 iterations
        out += [(line.replace("%s/%s/" % (root, name), "%s/%s_e48k/" % (root, name)), 2 * mn)
                for line, mn in main_runs(name, CONST_LEVELS, RESFIX_SEEDS, root=root, epochs=48000)]
    return out


# ---- calmer training (user, 6 Nov 2026): one smooth step-size decay, + weight averaging, + early stop; seeds 42-46
SMOOTH_ARMS = {"smooth": ["--smooth_lr"], "smooth_ema": ["--smooth_lr", "--ema", "0.999"],
               "smooth_ema_stop": ["--smooth_lr", "--ema", "0.999", "--patience", "20"]}


def smooth_runs():
    out = []
    for arm, fl in SMOOTH_ARMS.items():
        root = "results/B_smooth/" + arm
        for name in ("phm", "res_symlab_bal", "res_const_symlab_bal"):
            out += [(line + " " + " ".join(fl), mn) for line, mn in main_runs(name, CONST_LEVELS, RESFIX_SEEDS, root=root)]
    return out


# ---- PHM (named 8 Nov 2026; formerly r15_phm). Key "phm_new", because the key "phm" is PHM old (the 13-output head).
# Constants inside (tanh-bounded) + additive correction + copied symmetry labels, gap agreement with weight 2s, band-edge
# level anchor with weight s, 200-iteration SOAP warm-up. Identical to 05_code_repository/phm/phm.py (checked).
PHM_NEW_BASE = RES_ZERO + ["--const_head", "--const_tanh", "--soap_warmup", "200"]
PHM_NEW_COPIES = ["--sym_labels", "2000", "--sym_labels_balanced"]


def midpoint_s(rule):
    # level-anchor weight s = rule(n), written exactly as phm.py computes it: (2 s rounded to 6 digits) / 2
    def f(p):
        n = round(903 * p / 100)
        return ["--w_midpoint", repr(float("%g" % (2 * rule(n))) / 2.0)]
    return f


PHM_NEW_MODELS = {
    "phm_new":                ("rba", PHM_NEW_BASE + PHM_NEW_COPIES),
    # ablations: one part removed each
    "phm_new_drop_sym":       ("rba", PHM_NEW_BASE + PHM_NEW_COPIES + ["--sym_colloc", "0"]),
    "phm_new_drop_iso_eg":    ("rba", PHM_NEW_BASE + PHM_NEW_COPIES + ["--drop_loss", "iso_eg"]),
    "phm_new_drop_agreement": ("rba", PHM_NEW_BASE + PHM_NEW_COPIES + ["--drop_loss", "cons"]),
    "phm_new_drop_anchor":    ("rba", PHM_NEW_BASE + PHM_NEW_COPIES),
    "phm_new_drop_copies":    ("rba", PHM_NEW_BASE),
}
MODELS.update(PHM_NEW_MODELS)
for _name in PHM_NEW_MODELS:
    if _name == "phm_new_drop_anchor":
        LEVEL_FLAGS[_name] = [agree_s(RULES["cube"])]
    else:
        LEVEL_FLAGS[_name] = [agree_s(RULES["cube"]), midpoint_s(RULES["cube"])]
    MIN[_name] = 70.0


def phm_new_runs():
    """Everything the manuscript needs for PHM: main (17 seeds x 9 fractions), the five ablations, the held-out suite
    (5 families + 3 strain cut-offs at 100% labels) and label-matched constants at 1/2/5/10%."""
    out = main_runs("phm_new")
    for name in PHM_NEW_MODELS:
        if name != "phm_new":
            out += main_runs(name, root="results/B_ablation")
    out += heldout_runs("phm_new")
    m, ex = MODELS["phm_new"]
    out += [(cmd(m, s, p, "results/B_labelmatched/phm_new/%dpct/seed%d" % (p, s), ["--split", "symmetry"] + list(ex) + lflags("phm_new", p),
                 "%s/%dpct/const_seed%d.json" % (KLM, p, s)), MIN["phm_new"]) for p in LM_FRACTIONS for s in C.SEEDS]
    return out


# ---- PHM, final version (8 Nov 2026). Key "phm2". The 17-seed ablations of phm_new showed that the symmetry loss costs
# 1.9 meV at 50% and 100% labels (the copied symmetry labels already carry that information) and the level anchor has
# no effect. Both are removed. The phm_new runs now serve as the "add back" checks:
#   phm_new_drop_anchor = phm2 + symmetry loss,  phm_new_drop_sym = phm2 + level anchor,  phm_new = phm2 + both.
PHM2_BASE = PHM_NEW_BASE + PHM_NEW_COPIES + ["--sym_colloc", "0"]
PHM2_MODELS = {
    "phm2":                ("rba", PHM2_BASE),
    "phm2_drop_iso_eg":    ("rba", PHM2_BASE + ["--drop_loss", "iso_eg"]),
    "phm2_drop_agreement": ("rba", PHM2_BASE + ["--drop_loss", "cons"]),
    "phm2_drop_copies":    ("rba", PHM_NEW_BASE + ["--sym_colloc", "0"]),
}
MODELS.update(PHM2_MODELS)
for _name in PHM2_MODELS:
    LEVEL_FLAGS[_name] = [agree_s(RULES["cube"])]
    MIN[_name] = 70.0


def phm2_runs():
    """Final PHM: main (17 seeds x 9 fractions), three ablations, held-out suite, label-matched constants at 1/2/5/10%."""
    out = main_runs("phm2")
    for name in PHM2_MODELS:
        if name != "phm2":
            out += main_runs(name, root="results/B_ablation")
    out += heldout_runs("phm2")
    m, ex = MODELS["phm2"]
    out += [(cmd(m, s, p, "results/B_labelmatched/phm2/%dpct/seed%d" % (p, s), ["--split", "symmetry"] + list(ex) + lflags("phm2", p),
                 "%s/%dpct/const_seed%d.json" % (KLM, p, s)), MIN["phm2"]) for p in LM_FRACTIONS for s in C.SEEDS]
    return out


# ---- the runs behind the paper (Communications Materials, 9 Nov 2026 version): 2,924 training runs
PAPER_MAIN = ["phm2", "res_symlab_bal", "sa_pinn", "rba_pinn", "ntk_pinn", "aphynity", "mlp"]
PAPER_ABLATION = ["phm2_drop_agreement", "phm2_drop_iso_eg", "phm2_drop_copies", "phm_new_drop_anchor", "phm_new_drop_sym"]


def paper_runs():
    """Every training run of the paper: the seven models at 9 labeled fractions (B_main), the five changes to PHM
    (B_ablation; phm_new_drop_anchor = PHM + symmetry term, phm_new_drop_sym = PHM + level anchor), the held-out suite of
    the seven models (B_extrap) and the label-matched constants of PHM and the residual model (B_labelmatched)."""
    out = sum([main_runs(n) for n in PAPER_MAIN], [])
    out += sum([main_runs(n, root="results/B_ablation") for n in PAPER_ABLATION], [])
    out += sum([heldout_runs(n) for n in PAPER_MAIN], [])
    out += _lm("phm2", LM_FRACTIONS) + _lm("res_symlab_bal", LM_FRACTIONS)
    return out


LISTS = {"smooth": smooth_runs, "phm2": phm2_runs, "paper": paper_runs, "main": lambda: sum([main_runs(n) for n in MODELS if n not in RESFIX_LEVELS and n not in SYMBAL_CONTROLS and n not in REVIEW_MODELS and n not in CONST_MODELS and n not in PHM_NEW_MODELS and n not in PHM2_MODELS], []),
         "heldout": lambda: sum([heldout_runs(n) for n in HELDOUT_MODELS], []),
         "labelmatched": labelmatched_runs, "ablation": ablation_runs, "residual_matched": matched_runs,
         "resfix_screen": resfix_runs, "resfix_screen2": resfix2_runs,
         "resfix_screen3": lambda: sum([main_runs(n, RESFIX_LEVELS[n], RESFIX_SEEDS, root="results/B_resfix") for n in RESFIX3], []),
         "symbal_confirm": symbal_confirm_runs, "review": review_runs, "review_fresh": review_fresh_runs, "consthead": consthead_runs, "phm_new": phm_new_runs,
         "resfix_screen4": lambda: sum([main_runs(n, RESFIX_LEVELS[n], RESFIX_SEEDS, root="results/B_resfix") for n in RESFIX4], [])}

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
