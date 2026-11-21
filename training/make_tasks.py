"""Task lists of the paper's training runs: one line per run, "<results folder>\t<command>", run from this folder.

  python make_tasks.py paper    tasks/paper.tsv    the 2,924 training runs of the paper
  python make_tasks.py smoke    tasks/smoke.tsv    one 300-iteration run per model and experiment (21 runs),
                                                   checked against reference_smoke/ by compare_smoke.py

Each line trains one network on one processor core. run_task.sh runs one line; submit.sh submits a list as SLURM job
arrays. The constants of the analytic models are included in constants/ (calibrate_*.py refits them).
"""

import os
import sys

import config as C

DATA = "../data/dft_hy_v9.csv"
PROBE = ["--grad_probe_every", "250"]  # gradient diagnostics of the main and ablation runs

# Options of each model. The keys are the folder names of the recorded results.
PINN = ["--slope_fix"]
RESIDUAL = [
    "--model",
    "rba",
    "--no_rba",
    "--slope_fix",
    "--residual_phys",
    "--zero_out",
    "--drop_loss",
    "cbm",
    "--drop_loss",
    "vbm",
    "--w_anchor",
    "0",
    "--w_pcbm",
    "0",
    "--w_pvbm",
    "0",
]
CONST_HEAD = ["--const_head", "--const_tanh", "--soap_warmup", "200"]
COPIES = ["--sym_labels", "2000", "--sym_labels_balanced"]
PHM = RESIDUAL + CONST_HEAD + COPIES
MODELS = {
    "phm2": PHM,  # PHM
    "res_symlab_bal": RESIDUAL + COPIES + ["--sym_colloc", "1.0"],  # residual model
    "sa_pinn": ["--model", "sa_paper"] + PINN,
    "rba_pinn": ["--model", "rba_paper"] + PINN,
    "ntk_pinn": ["--model", "ntk_paper"] + PINN,
    "aphynity": ["--model", "mlp", "--no_rba", "--aphynity", "--aph_tau2", "10"],
    "mlp": ["--model", "mlp", "--no_rba"],
}
ABLATIONS = {
    "phm2_drop_agreement": PHM + ["--drop_loss", "cons"],  # PHM without the agreement term
    "phm2_drop_iso_eg": PHM + ["--drop_loss", "iso_eg"],  # without the isotropic term
    "phm2_drop_copies": RESIDUAL + CONST_HEAD,  # without the copied symmetry labels
    "phm_new_drop_anchor": PHM + ["--sym_colloc", "1.0"],  # with a symmetry term added
    "phm_new_drop_sym": PHM,  # with a band-edge level anchor added
}
FALLING_WEIGHT = [
    "phm2",
    "res_symlab_bal",
    "phm2_drop_iso_eg",
    "phm2_drop_copies",
    "phm_new_drop_anchor",
    "phm_new_drop_sym",
]
HELDOUT = {
    "group_isotropic": ["--split", "group", "--holdout_family", "isotropic"],
    "group_uniaxial": ["--split", "group", "--holdout_family", "uniaxial"],
    "group_biaxial": ["--split", "group", "--holdout_family", "biaxial"],
    "group_triaxial": ["--split", "group", "--holdout_family", "triaxial"],
    "group_shear": ["--split", "group", "--holdout_family", "shear"],
    "magnitude_03": ["--split", "magnitude", "--magnitude_thresh", "0.03"],
    "magnitude_05": ["--split", "magnitude", "--magnitude_thresh", "0.05"],
    "magnitude_07": ["--split", "magnitude", "--magnitude_thresh", "0.07"],
}
LABELMATCHED_MODELS = ["phm2", "res_symlab_bal"]
LABELMATCHED_FRACTIONS = [1, 2, 5, 10]


def agreement_weight(pct):
    """s = min(1, (9/n)^3) for n labeled states, n = pct% of the 903 training states of a typical run."""
    n = round(903 * pct / 100)
    return 1.0 if n <= 9 else (9 / n) ** 3


def level_options(name, pct):
    """Options that depend on the labeled fraction: the agreement weight 2s and, for one ablation, the anchor weight s."""
    s = agreement_weight(pct)
    options = ["--w_cons", "%g" % (2 * s)] if name in FALLING_WEIGHT else []
    if name == "phm_new_drop_sym":
        options += ["--w_midpoint", repr(float("%g" % (2 * s)) / 2.0)]
    return options


def task(out, seed, pct, constants, options):
    return (
        "%s\tPHYSICS_CONST_JSON=constants/%s/const_seed%d.json python train.py --data_csv %s --results_dir %s "
        "--seed %d --dft_pct %d --device cpu %s" % (out, constants, seed, DATA, out, seed, pct, " ".join(options))
    )


def paper_runs():
    """B_main: 7 models x 9 fractions; B_ablation: 5 changes to PHM x 9 fractions; B_extrap: 7 models x 8 held-out
    tests; B_labelmatched: PHM and the residual model with label-matched constants x 4 fractions; 17 seeds each."""
    runs = []
    for folder, models in (("B_main", MODELS), ("B_ablation", ABLATIONS)):
        for name, options in models.items():
            runs += [
                task(
                    "results/%s/%s/%dpct/seed%d" % (folder, name, p, s),
                    s,
                    p,
                    "two_band/symmetry",
                    ["--split", "symmetry"] + options + level_options(name, p) + PROBE,
                )
                for p in C.FRACTIONS
                for s in C.SEEDS
            ]
    for name, options in MODELS.items():
        runs += [
            task(
                "results/B_extrap/%s/%s/seed%d" % (name, test, s),
                s,
                100,
                "two_band/" + test,
                split + options + level_options(name, 100),
            )
            for test, split in HELDOUT.items()
            for s in C.SEEDS
        ]
    for name in LABELMATCHED_MODELS:
        runs += [
            task(
                "results/B_labelmatched/%s/%dpct/seed%d" % (name, p, s),
                s,
                p,
                "labelmatched/%dpct" % p,
                ["--split", "symmetry"] + MODELS[name] + level_options(name, p),
            )
            for p in LABELMATCHED_FRACTIONS
            for s in C.SEEDS
        ]
    return runs


def smoke_runs():
    """Seed 42 of each experiment and model (25% labeled, or the first level), shortened to 300 iterations."""
    picked = {}
    for line in paper_runs():
        out = line.split("\t")[0]
        _, experiment, model, level, seed = out.split("/")
        if seed == "seed42" and ((experiment, model) not in picked or level == "25pct"):
            picked[(experiment, model)] = (out, line)
    return [
        line.replace(out, "results/smoke/%s_%s" % (experiment[2:].lower(), model)) + " --epochs 300"
        for (experiment, model), (out, line) in picked.items()
    ]


def write(name, runs):
    os.makedirs("tasks", exist_ok=True)
    folders = [line.split("\t")[0] for line in runs]
    assert len(folders) == len(set(folders)), "two runs share a results folder"
    with open("tasks/%s.tsv" % name, "w") as f:
        f.writelines(line + "\n" for line in runs)
    print("wrote tasks/%s.tsv: %d runs" % (name, len(runs)))


if __name__ == "__main__":
    os.chdir(C.HERE)
    lists = {"paper": paper_runs, "smoke": smoke_runs}
    if len(sys.argv) != 2 or sys.argv[1] not in lists:
        sys.exit(__doc__)
    write(sys.argv[1], lists[sys.argv[1]]())
