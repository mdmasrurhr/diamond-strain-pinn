"""Paths and settings shared by the training scripts."""

import importlib.util
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(os.path.dirname(HERE), "data", "dft_hy_v9.csv")
CONSTANTS = os.path.join(HERE, "constants")
RESULTS = os.path.join(HERE, "results")

STRAIN_FAMILIES = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]

# 17 runs per model and setting; the seed sets the data division, the labeled states and the initial weights.
SEEDS = list(range(42, 59))
FRACTIONS = [0, 1, 2, 5, 10, 25, 50, 75, 100]  # labeled share of the training states (%)

# Network and optimizer, fixed in preliminary studies (Supplementary Tables 8 and 9 of the paper).
TUNED = dict(
    width=64,
    depth=8,
    epochs=24000,  # 12,000 AdamW iterations, then 12,000 SOAP
    adamw_lr=2.636e-3,
    adamw_wd=1.793e-5,
    soap_lr=7.325e-3,
    soap_wd=1.224e-3,
    switch_frac=0.5,
    clip_norm=1.0,
    ckpt_interval=250,
)

# Base weights of the terms of the objective; each model sets its own where it differs.
LOSS_WEIGHTS = dict(data=5.0, cons=2.0, cbm=1.0, vbm=1.0, anchor=0.5, pcbm=0.5, pvbm=0.5, iso_eg=5.0)

SOFTMIN_BETA = 50.0  # sharpness of the soft minimum over the six conduction valleys (1/eV)


def import_trainer(*options):
    """Import train.py with the given options (it parses them on import) to reuse load_and_split and PINNModel."""
    argv = sys.argv
    sys.argv = ["train.py", "--data_csv", DATA_CSV, "--device", "cpu", *options]
    try:
        spec = importlib.util.spec_from_file_location("train", os.path.join(HERE, "train.py"))
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
    finally:
        sys.argv = argv
    return trainer


def load_trained_run(run):
    """Rebuild a finished run from its command.txt; returns the trainer module, the data division and the network."""
    import torch

    tokens = shlex.split(open(os.path.join(run, "command.txt")).read())
    for token in tokens:
        if token.startswith("PHYSICS_CONST_JSON="):
            os.environ["PHYSICS_CONST_JSON"] = os.path.join(HERE, token.split("=", 1)[1])
    sys.modules.pop("physics", None)  # load this run's constants
    start = next(i for i, t in enumerate(tokens) if t.endswith("train.py")) + 1
    trainer = import_trainer(*tokens[start:])
    a = trainer.args
    data = trainer.load_and_split(os.path.join(HERE, a.data_csv), a.seed, a.dft_pct)
    net = trainer.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"], scaler=data["scaler"])
    net.load_state_dict(torch.load(os.path.join(run, "best_model.pt"), map_location="cpu", weights_only=False))
    net.eval()
    return trainer, data, net
