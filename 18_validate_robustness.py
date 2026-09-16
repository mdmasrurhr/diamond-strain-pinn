"""Measure how predictions change under input noise, cubic permutations of the
normal strain components, and shear sign flips.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch

import config as C

NOISE_LEVELS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


# train_pinn is imported, not re-implemented: a check that tests its own copy of
# the model is not testing the model. Importing it also runs its argument parser
# with default values, which is why --data_csv has a default.
import train_pinn


def _load_trainer(extra_argv=()):
    """Return the trainer module. The argument is accepted but unused."""
    return train_pinn


def _load_model(T, data, run_dir, dev):
    model = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)
    w = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(w):
        raise SystemExit("no best_model.pt in %s -- train a model first" % run_dir)
    model.load_state_dict(torch.load(w, map_location=dev))
    model.eval()
    return model


def _predict(model, X, dev):
    with torch.no_grad():
        _, _, eg = model(torch.tensor(X, dtype=torch.float32, device=dev))
    return eg.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True,
                    help="a finished run directory containing best_model.pt")
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the seed the run was trained with")
    args = ap.parse_args()

    T = _load_trainer()
    data = T.load_and_split(C.DATA_CSV, seed=args.seed, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(T, data, args.run_dir, dev)

    X_te, eg_te = data["X_te"], data["eg_te"]
    rows = []

    # ---------------------------------------------------------------- noise
    # Noise is added in SCALED space, so a level is in units of the training
    # standard deviation and is comparable across the six components.
    print("INPUT NOISE  (perturbation in units of the training s.d.)")
    print("  %-12s %12s %12s" % ("sigma", "MAE meV", "x clean"))
    rng = np.random.default_rng(0)
    clean = None
    for s in NOISE_LEVELS:
        errs = []
        reps = 1 if s == 0 else 20
        for _ in range(reps):
            Xn = X_te + rng.normal(0.0, s, X_te.shape).astype(np.float32)
            errs.append(np.abs(_predict(model, Xn, dev) - eg_te).mean() * 1000.0)
        mae = float(np.mean(errs))
        clean = mae if s == 0 else clean
        print("  %-12.0e %12.2f %12.2f" % (s, mae, mae / clean))
        rows.append(dict(check="noise", level=s, mae_meV=round(mae, 2),
                         ratio_to_clean=round(mae / clean, 3)))
    print("\n  Read off the noise level at which the error doubles: that is the")
    print("  input precision the model requires to be worth using.")

    # ------------------------------------------------------------- symmetry
    # Permuting Exx, Eyy, Ezz maps a cubic crystal onto itself, so the gap must
    # not change. Nothing in the loss says so; this measures whether it learned it.
    print("\nCUBIC PERMUTATION SYMMETRY  (Exx, Eyy, Ezz permuted)")
    df = pd.read_csv(C.DATA_CSV)
    raw = df[C.STRAIN_COLS].values.astype(np.float32)[data["idx_te"]]
    base = _predict(model, data["scaler"].transform(raw).astype(np.float32), dev)
    drifts = []
    for perm in itertools.permutations(range(3)):
        if perm == (0, 1, 2):
            continue
        p = raw.copy()
        p[:, :3] = raw[:, list(perm)]
        got = _predict(model, data["scaler"].transform(p).astype(np.float32), dev)
        d = float(np.abs(got - base).mean() * 1000.0)
        drifts.append(d)
        print("  permute %s -> %8.2f meV drift" % (str(perm), d))
    mean_drift = float(np.mean(drifts))
    print("  mean drift %.2f meV, against a test MAE of %.2f meV"
          % (mean_drift, clean))
    print("  -> the model is %s to a symmetry it was never told about"
          % ("insensitive" if mean_drift < clean else "SENSITIVE"))
    rows.append(dict(check="permutation_symmetry", level=float("nan"),
                     mae_meV=round(mean_drift, 2),
                     ratio_to_clean=round(mean_drift / clean, 3)))

    # ------------------------------------------------------------ shear sign
    print("\nSHEAR SIGN  (Exy, Eyz, Ezx negated)")
    p = raw.copy()
    p[:, 3:] = -raw[:, 3:]
    got = _predict(model, data["scaler"].transform(p).astype(np.float32), dev)
    d = float(np.abs(got - base).mean() * 1000.0)
    print("  drift %.2f meV (%.2fx the test MAE)" % (d, d / clean))
    print("  Shear enters the physics through even powers, so this drift should")
    print("  be small. A large value means the network learned a sign dependence")
    print("  the physics does not have.")
    rows.append(dict(check="shear_sign", level=float("nan"), mae_meV=round(d, 2),
                     ratio_to_clean=round(d / clean, 3)))

    out = C.results_dir("validate_robustness")
    pd.DataFrame(rows).to_csv(os.path.join(out, "robustness.csv"), index=False)
    print("\nwrote %s/robustness.csv" % os.path.relpath(out, C.ROOT))


if __name__ == "__main__":
    main()
