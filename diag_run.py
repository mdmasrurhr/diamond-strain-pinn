#!/usr/bin/env python3
"""Physical-validity diagnostics of one trained run, computed after training from its saved model (no retraining).
Prints one JSON line with, among others:
  slope_hydrostatic   slope of the predicted gap along (t,t,t,0,0,0) at zero strain, by central differences with step
                      0.002 (the paper's deformation potential is this slope divided by 3, since I_1 = 3t)
  slope_shear         slope of the predicted gap along (0,0,0,t,0,0) at zero strain (zero by symmetry)
  ref_slope_*         the same slopes of the analytic models with this run's constants
  axis_relabel_drift_meV   mean |change of the predicted gap| over the test states under the five non-identity
                      permutations of the axes
  shear_sign_drift_meV     mean |change| when the signs of the shear components are reversed, on the test states with at
                      most two nonzero shear components (where that is a symmetry operation)
  noise_rel_change    relative change of the test MAE when every strain component is perturbed by 1e-3 of its spread
  slope_error         mean |model - analytic models| of the three slopes (not used in the paper)
Usage: python diag_run.py <run_dir>   (reads <run_dir>/command.txt; run from this folder). For many runs:
run_diagnostics.py."""
import os, sys, json, itertools, importlib.util
import numpy as np, pandas as pd, torch

run = sys.argv[1]
tok = open(os.path.join(run, "command.txt")).read().split()
for t in tok:
    if t.startswith("PHYSICS_CONST_JSON="): os.environ["PHYSICS_CONST_JSON"] = t.split("=", 1)[1]
i0 = next(i for i, t in enumerate(tok) if t.endswith("03_train_pinn.py")); argv = tok[i0:]
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
sys.argv = [os.path.join(here, "03_train_pinn.py")] + argv[1:] + ["--device", "cpu", "--results_dir", os.path.join(here, "results", "_diag_stub")]
spec = importlib.util.spec_from_file_location("trainer_diag", os.path.join(here, "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
import config as C, physics as P
seed = int(argv[argv.index("--seed") + 1]); pct = int(argv[argv.index("--dft_pct") + 1])
csv = os.path.join(here, argv[argv.index("--data_csv") + 1])
d = T.load_and_split(csv, seed, pct); sc = d["scaler"]
net = T.PINNModel(d["cbm_mean_tr"], d["vbm_mean_tr"], scaler=sc)
net.load_state_dict(torch.load(os.path.join(run, "best_model.pt"), map_location="cpu", weights_only=False)); net.eval()
df = pd.read_csv(csv); RAW = df[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values.astype(np.float64)

def predict(raw):
    with torch.no_grad():
        return net(torch.tensor(sc.transform(raw).astype(np.float32)))[2].numpy().astype(np.float64)

MODES = {"hydrostatic": [1, 1, 1, 0, 0, 0], "uniaxial": [1, 0, 0, 0, 0, 0], "shear": [0, 0, 0, 1, 0, 0]}; DELTA = 0.002
def slopes(f):
    return {k: float((f(np.array([DELTA * np.array(v)]))[0] - f(np.array([-DELTA * np.array(v)]))[0]) / (2 * DELTA)) for k, v in MODES.items()}
def ref(x):
    t = torch.tensor(x, dtype=torch.float32)
    return (P.physics_cbm(t, beta=C.SOFTMIN_BETA) - P.physics_vbm(t)).detach().numpy().astype(np.float64)
sm, sr = slopes(predict), slopes(ref)
te = RAW[np.array(d["idx_te"])]; y = df.Eg_eV.values[np.array(d["idx_te"])]; base = predict(te)
def perm(raw, p):
    E = np.zeros((len(raw), 3, 3)); E[:, 0, 0], E[:, 1, 1], E[:, 2, 2] = raw[:, 0], raw[:, 1], raw[:, 2]
    E[:, 0, 1] = E[:, 1, 0] = raw[:, 3]; E[:, 1, 2] = E[:, 2, 1] = raw[:, 4]; E[:, 2, 0] = E[:, 0, 2] = raw[:, 5]
    Pm = np.eye(3)[list(p)]; Ep = Pm @ E @ Pm.T
    return np.stack([Ep[:, 0, 0], Ep[:, 1, 1], Ep[:, 2, 2], Ep[:, 0, 1], Ep[:, 1, 2], Ep[:, 2, 0]], 1)
axis = np.mean([np.abs(predict(perm(te, p)) - base).mean() * 1000 for p in itertools.permutations(range(3)) if p != (0, 1, 2)])
# reversing every shear component is a cubic symmetry only when at most two are nonzero (a diagonal sign matrix
# flips two of the three); with all three nonzero it reverses eps_xy*eps_yz*eps_zx, a different strain state
two = (np.abs(te[:, 3:]) > 1e-8).sum(1) <= 2
q = te.copy(); q[:, 3:] = -te[:, 3:]; shear = np.abs(predict(q[two]) - base[two]).mean() * 1000
rng = np.random.default_rng(seed); noisy = te + rng.normal(size=te.shape) * 1e-3 * RAW.std(axis=0)
mae = np.abs(base - y).mean() * 1000; mae_n = np.abs(predict(noisy) - y).mean() * 1000
print(json.dumps(dict(run=run, seed=seed, pct=pct, test_mae_meV=mae, slope_error=float(np.mean([abs(sm[k] - sr[k]) for k in MODES])),
                      **{"slope_" + k: sm[k] for k in MODES}, **{"ref_slope_" + k: sr[k] for k in MODES},
                      axis_relabel_drift_meV=axis, shear_sign_drift_meV=shear, noise_rel_change=(mae_n - mae) / mae)))
