"""Check a trained model against physics it was not fitted to: deformation-potential
slopes, cubic symmetry and smoothness.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

import config as C
import physics as P
# Imported for its model class and data pipeline, not to run a training job.
import train_pinn

# Strain directions probed, and the step used for the central difference.
MODES = {"hydrostatic": [1, 1, 1, 0, 0, 0],
         "uniaxial":    [1, 0, 0, 0, 0, 0],
         "shear":       [0, 0, 0, 1, 0, 0]}
DELTA = 0.002


def load_trainer():
    """Return the trainer module, for its network and data-pipeline code."""
    return train_pinn


def slopes(predict):
    """dEg/d(amplitude) at zero strain, eV per unit strain, per direction."""
    out = {}
    for name, d in MODES.items():
        v = np.array(d, dtype=np.float32)
        eg = predict(np.stack([DELTA * v, -DELTA * v]))
        out[name] = float((eg[0] - eg[1]) / (2 * DELTA))
    return out


def smoothness(predict, pool, n=400, eps=0.004, seed=0):
    rng = np.random.default_rng(seed)
    base = pool[rng.choice(len(pool), n)].astype(np.float32)
    step = rng.normal(size=base.shape).astype(np.float32)
    step = step / (np.linalg.norm(step, axis=1, keepdims=True) + 1e-9) * eps
    sens = np.abs(predict(base + step) - predict(base)) / eps
    return float(np.mean(sens))


def reference_slopes():
    """The same three slopes from the analytic model -- the target to match.

    The soft minimum (beta from config) is used, not the hard one, and the
    distinction matters here more than anywhere else. At zero strain all six
    conduction valleys are degenerate, so the two disagree exactly at the point
    we differentiate: hard-min gives a uniaxial slope of -2.36 eV per unit
    strain, soft-min -0.81. The soft minimum is what the physics loss uses
    during training, so it is what the network was actually asked to match.
    """
    def predict(x):
        t = torch.tensor(x, dtype=torch.float32)
        eg = P.physics_cbm(t, beta=C.SOFTMIN_BETA) - P.physics_vbm(t)
        return eg.detach().numpy()
    return slopes(predict)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pct", type=int, default=100)
    ap.add_argument("--study", default="label_sweep")
    args = ap.parse_args()

    T = load_trainer()
    ref = reference_slopes()
    print("analytic reference (eV per unit strain): " +
          "  ".join("%s %.2f" % (k, v) for k, v in ref.items()))

    out = C.results_dir("physics_validation")
    rows = []
    pattern = os.path.join(C.RESULTS, args.study, "*", "%dpct" % args.pct,
                           "seed*", "best_model.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print("no checkpoints under %s -- run 05 first" % pattern)
        return

    for ckpt in files:
        parts = ckpt.split(os.sep)
        model_name, seed = parts[-4], int(parts[-2].replace("seed", ""))

        # Rebuild this run's split and scaler; both are deterministic in the seed.
        data = T.load_and_split(C.DATA_CSV, seed, args.pct)
        scaler, net = data["scaler"], T.PINNModel(data["cbm_mean_tr"],
                                                  data["vbm_mean_tr"])
        net.load_state_dict(torch.load(ckpt, map_location="cpu",
                                       weights_only=False))
        net.eval()

        def predict(raw):
            x = torch.tensor(scaler.transform(raw).astype(np.float32))
            with torch.no_grad():
                return net(x)[2].numpy()          # (cbm, vbm, eg) -> eg

        s = slopes(predict)
        err = float(np.mean([abs(s[k] - ref[k]) for k in MODES]))
        rows.append(dict(model=model_name, seed=seed, dft_pct=args.pct,
                         slope_error=round(err, 3),
                         smoothness=round(smoothness(predict,
                                                     data["strain_tr"]), 2),
                         **{("slope_" + k): round(v, 3) for k, v in s.items()}))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "validity.csv"), index=False)
    agg = d.groupby("model").agg(
        n=("seed", "count"),
        slope_error_mean=("slope_error", "mean"),
        slope_error_sd=("slope_error", "std"),
        hydrostatic=("slope_hydrostatic", "mean")).round(3)
    print("\nslope error against the analytic reference (eV per unit strain):")
    print(agg.to_string())
    print("\nwrote %s/validity.csv" % out)


if __name__ == "__main__":
    main()
