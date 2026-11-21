"""Physical-consistency diagnostics of one finished run, from its saved network (no retraining); prints one JSON line.

  slope_hydrostatic, slope_uniaxial, slope_shear   slope of the predicted gap at zero strain along (t,t,t,0,0,0),
                       (t,0,0,0,0,0) and (0,0,0,t,0,0), by central differences with step 0.002 (the deformation
                       potential of the paper is slope_hydrostatic / 3, since I1 = 3t)
  ref_slope_*          the same slopes of the analytic models with the run's constants
  slope_error          mean |slope - ref_slope| of the three slopes
  axis_relabel_drift_meV   mean |change of the predicted gap| on the test states under the five axis permutations
  shear_sign_drift_meV     mean |change| when the shear components change sign (test states with at most two shear
                       components, where this is a symmetry operation)
  noise_rel_change     relative change of the test MAE when every strain component is perturbed by 1e-3 of its spread

Usage: python diag_run.py <run folder>     (run_diagnostics.py runs it on many runs)
"""

import itertools
import json
import sys

import numpy as np
import pandas as pd
import torch

import config as C

run = sys.argv[1]
T, d, net = C.load_trained_run(run)
import physics as P  # loaded with the run's constants by load_trained_run

seed, pct, scaler = T.args.seed, T.args.dft_pct, d["scaler"]
df = pd.read_csv(C.DATA_CSV)
RAW = df[T.STRAIN_COLUMNS].values.astype(np.float64)
MODES = {"hydrostatic": [1, 1, 1, 0, 0, 0], "uniaxial": [1, 0, 0, 0, 0, 0], "shear": [0, 0, 0, 1, 0, 0]}
STEP = 0.002


def predict(raw):
    with torch.no_grad():
        return net(torch.tensor(scaler.transform(raw).astype(np.float32)))[2].numpy().astype(np.float64)


def analytic(raw):
    t = torch.tensor(raw, dtype=torch.float32)
    return (P.physics_cbm(t, beta=C.SOFTMIN_BETA) - P.physics_vbm(t)).detach().numpy().astype(np.float64)


def slopes(f):
    return {
        k: float((f(np.array([STEP * np.array(v)]))[0] - f(np.array([-STEP * np.array(v)]))[0]) / (2 * STEP))
        for k, v in MODES.items()
    }


def permute_axes(raw, p):
    E = np.zeros((len(raw), 3, 3))
    E[:, 0, 0], E[:, 1, 1], E[:, 2, 2] = raw[:, 0], raw[:, 1], raw[:, 2]
    E[:, 0, 1] = E[:, 1, 0] = raw[:, 3]
    E[:, 1, 2] = E[:, 2, 1] = raw[:, 4]
    E[:, 2, 0] = E[:, 0, 2] = raw[:, 5]
    Pm = np.eye(3)[list(p)]
    Ep = Pm @ E @ Pm.T
    return np.stack([Ep[:, 0, 0], Ep[:, 1, 1], Ep[:, 2, 2], Ep[:, 0, 1], Ep[:, 1, 2], Ep[:, 2, 0]], 1)


sm, sr = slopes(predict), slopes(analytic)
te = RAW[np.array(d["idx_te"])]
y = df.Eg_eV.values[np.array(d["idx_te"])]
base = predict(te)
axis = np.mean(
    [
        np.abs(predict(permute_axes(te, p)) - base).mean() * 1000
        for p in itertools.permutations(range(3))
        if p != (0, 1, 2)
    ]
)
# Reversing every shear component is a cubic operation only when at most two are nonzero.
two = (np.abs(te[:, 3:]) > 1e-8).sum(1) <= 2
flipped = te.copy()
flipped[:, 3:] = -te[:, 3:]
shear = np.abs(predict(flipped[two]) - base[two]).mean() * 1000
rng = np.random.default_rng(seed)
noisy = te + rng.normal(size=te.shape) * 1e-3 * RAW.std(axis=0)
mae = np.abs(base - y).mean() * 1000
mae_noisy = np.abs(predict(noisy) - y).mean() * 1000
print(
    json.dumps(
        dict(
            run=run,
            seed=seed,
            pct=pct,
            test_mae_meV=mae,
            slope_error=float(np.mean([abs(sm[k] - sr[k]) for k in MODES])),
            **{"slope_" + k: sm[k] for k in MODES},
            **{"ref_slope_" + k: sr[k] for k in MODES},
            axis_relabel_drift_meV=axis,
            shear_sign_drift_meV=shear,
            noise_rel_change=(mae_noisy - mae) / mae
        )
    )
)
