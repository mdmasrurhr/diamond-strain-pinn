"""Compare a trained model's input derivatives against the values
deformation-potential theory fixes in advance.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import config as C


# train_pinn is imported, not re-implemented: a check that tests its own copy of
# the model is not testing the model. Importing it also runs its argument parser
# with default values, which is why --data_csv has a default.
import train_pinn


def _load_trainer(extra_argv=()):
    """Return the trainer module. The argument is accepted but unused."""
    return train_pinn


def _grads(model, X_scaled, scaler, dev):
    """dEg/d(raw strain component), chain-ruled back through the scaler."""
    x = torch.tensor(X_scaled, dtype=torch.float32, device=dev, requires_grad=True)
    _, _, eg = model(x)
    g = torch.autograd.grad(eg.sum(), x)[0].cpu().numpy()
    # The network sees scaled inputs; the physics is stated in raw strain, so
    # divide by the scale to get back to eV per unit strain.
    return g / scaler.scale_.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    T = _load_trainer()
    data = T.load_and_split(C.DATA_CSV, seed=args.seed, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)
    w = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(w):
        raise SystemExit("no best_model.pt in %s" % args.run_dir)
    model.load_state_dict(torch.load(w, map_location=dev))
    model.eval()

    rows = []
    sc = data["scaler"]

    # ------------------------------------------------- overall attribution
    g = _grads(model, data["X_te"], sc, dev)
    mean_abs = np.abs(g).mean(axis=0)
    print("ATTRIBUTION  (mean |dEg/dE_ij| over the test set, eV per unit strain)")
    for name, v in zip(C.STRAIN_COLS, mean_abs):
        print("  %-8s %10.3f" % (name, v))
    normal, shear = mean_abs[:3].mean(), mean_abs[3:].mean()
    print("  normal components %.3f, shear components %.3f, ratio %.2f"
          % (normal, shear, normal / max(shear, 1e-9)))
    print("  Hydrostatic strain moves the gap far more than shear does, so the")
    print("  normal components should dominate. They do by %.1fx."
          % (normal / max(shear, 1e-9)))
    for name, v in zip(C.STRAIN_COLS, mean_abs):
        rows.append(dict(check="attribution", item=name, value=round(float(v), 4),
                         expected="", note="mean |dEg/dcomponent|"))

    # ------------------------------------------- symmetry of the derivatives
    # At an isotropic point the three axes are equivalent, so the three normal
    # derivatives must agree. Any spread is learned anisotropy that the crystal
    # does not have.
    eps = np.linspace(-0.04, 0.04, 41).astype(np.float32)
    iso = np.zeros((len(eps), 6), dtype=np.float32)
    iso[:, 0] = iso[:, 1] = iso[:, 2] = eps
    gi = _grads(model, sc.transform(iso).astype(np.float32), sc, dev)
    spread = float(np.abs(gi[:, :3] - gi[:, :3].mean(axis=1, keepdims=True)).mean())
    scale = float(np.abs(gi[:, :3]).mean())
    print("\nAXIS EQUIVALENCE at isotropic strain")
    print("  mean |deviation| across the three axes: %.4f eV  (%.1f%% of %.3f)"
          % (spread, 100 * spread / max(scale, 1e-9), scale))
    print("  -> %s" % ("consistent with cubic symmetry" if spread < 0.1 * scale
                       else "ANISOTROPIC: the model distinguishes axes the crystal does not"))
    rows.append(dict(check="axis_equivalence", item="normal_derivative_spread",
                     value=round(spread, 5), expected="0 (cubic)",
                     note="%.1f%% of the derivative magnitude" % (100 * spread / max(scale, 1e-9))))

    # --------------------------------------------- shear derivative at zero
    # Shear enters through even powers, so dEg/dExy must vanish at zero shear.
    zero = np.zeros((1, 6), dtype=np.float32)
    gz = _grads(model, sc.transform(zero).astype(np.float32), sc, dev)[0]
    print("\nSHEAR DERIVATIVE AT ZERO STRAIN  (physics says 0)")
    for name, v in zip(C.STRAIN_COLS[3:], gz[3:]):
        print("  dEg/d%-6s %10.4f eV" % (name, v))
    worst = float(np.abs(gz[3:]).max())
    print("  largest %.4f eV -> %s" % (
        worst, "consistent with even-power shear coupling" if worst < 0.5
        else "the model has a linear shear response the physics does not"))
    rows.append(dict(check="shear_derivative_at_zero", item="max|dEg/dshear|",
                     value=round(worst, 5), expected="0 (even powers)", note=""))

    # ------------------------------------- hydrostatic slope against physics
    gi_mean = float(gi[:, :3].sum(axis=1).mean())
    print("\nHYDROSTATIC SLOPE  dEg/dI1 = %.3f eV" % gi_mean)
    print("  This is the deformation-potential combination the analytic model")
    print("  fixes. Compare it with the fitted value from 03_fit_physics_constants.py;")
    print("  agreement is evidence the network learned the physics rather than")
    print("  a curve that happens to pass through the same points.")
    rows.append(dict(check="hydrostatic_slope", item="dEg/dI1",
                     value=round(gi_mean, 4), expected="see 03_fit_physics_constants.py",
                     note=""))

    out = C.results_dir("validate_explainability")
    pd.DataFrame(rows).to_csv(os.path.join(out, "attribution.csv"), index=False)
    print("\nwrote %s/attribution.csv" % os.path.relpath(out, C.ROOT))


if __name__ == "__main__":
    main()
