"""Why is shear the hardest family? Per-state analysis with the final model and the MLP at 100% labeled (17 runs)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root; run from there
os.makedirs(ROOT + "/results/shear_tests", exist_ok=True)

import glob, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
M = ROOT; R = M + "/results"
D = pd.read_csv(M + "/data/dft_hy_v9.csv"); SC = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
D["fam"] = D.strain_type.map(lambda x: "isotropic" if str(x) in ("unstrained", "isotropic") else str(x).split("_")[0])
X = D[SC].values; y = D.Eg_eV.values

def per_state(model):
    err = {}
    for f in glob.glob("%s/B_main/%s/100pct/seed*/test_results.csv" % (R, model)):
        t = pd.read_csv(f)
        for i, e in zip(t.sample_id.values, (t.Eg_pred_eV - t.Eg_true_eV).abs().values * 1000): err.setdefault(int(i), []).append(e)
    return pd.Series({i: np.mean(v) for i, v in err.items()})
D["err_ours"] = per_state("phm"); D["err_mlp"] = per_state("mlp")

# features from the labels and strains alone (no model)
tree = cKDTree(X); dist, idx = tree.query(X, k=16)
D["nn_dist"] = dist[:, 1:6].mean(1)                                   # mean distance to the 5 nearest states (strain units)
rough, grad = [], []
for i in range(len(D)):
    nb = idx[i, 1:]; A = np.c_[np.ones(len(nb)), X[nb] - X[i]]
    c, *_ = np.linalg.lstsq(A, y[nb], rcond=None)                     # local linear fit to the 15 neighbours, evaluated at state i
    rough.append(abs(y[i] - c[0]) * 1000); grad.append(np.linalg.norm(c[1:]))
D["rough_meV"] = rough; D["grad_eV"] = grad                           # departure from a local plane; slope of the gap
kp = D.cbm_kpt_index.values; D["kswitch"] = [(kp[idx[i, 1:7]] != kp[i]).mean() for i in range(len(D))]   # share of 6 nearest states with a different k-point
th = json.load(open(M + "/results/98_constants_v2/symmetry/const_seed42.json"))
import sys; sys.path.insert(0, M + ""); import os
os.environ["PHYSICS_CONST_JSON"] = M + "/results/98_constants_v2/symmetry/const_seed42.json"
import torch, physics as P
xt = torch.tensor(X, dtype=torch.float64); D["formula_err"] = np.abs((P.physics_cbm(xt) - P.physics_vbm(xt)).numpy() - y) * 1000

T = D[D.err_ours.notna()]
pd.set_option("display.width", 220)
print("A. by family (test states, mean over the runs in which a state was tested)")
print(T.groupby("fam")[["err_ours", "err_mlp", "formula_err", "rough_meV", "nn_dist", "grad_eV", "kswitch"]].mean().round(3).to_string())
print("\nB. shear by number of sheared planes")
S = T[T.fam == "shear"]; g = S.groupby("n_shear_planes"); print(g[["err_ours", "err_mlp", "formula_err", "rough_meV", "kswitch"]].mean().round(2).assign(n=g.size()).to_string())
print("\nC. Spearman correlation with the error of our model (shear states only; all states)")
for c in ["rough_meV", "nn_dist", "grad_eV", "kswitch", "formula_err"]:
    print("  %-12s shear %+.2f   all %+.2f" % (c, stats.spearmanr(S[c], S.err_ours)[0], stats.spearmanr(T[c], T.err_ours)[0]))
print("\nD. shear states: error of our model by k-point switching of the neighbourhood")
S = S.assign(sw=pd.cut(S.kswitch, [-0.01, 0.0, 0.34, 1.0], labels=["none", "1-2 of 6", "3+ of 6"]))
print(S.groupby("sw", observed=True)[["err_ours", "err_mlp", "rough_meV"]].mean().round(2).join(S.groupby("sw", observed=True).size().rename("n")).to_string())
print("\nE. does roughness explain the family gap? log(error) regression, shear dummy with and without controls")
import numpy.linalg as la
def fit(cols):
    A = np.c_[np.ones(len(T)), (T.fam == "shear").astype(float), T[cols].apply(lambda c: np.log(c + 1e-3)).values] if cols else np.c_[np.ones(len(T)), (T.fam == "shear").astype(float)]
    b, *_ = la.lstsq(A, np.log(T.err_ours.values + 1e-3), rcond=None); return b[1]
for cols in ([], ["rough_meV"], ["rough_meV", "nn_dist"], ["rough_meV", "nn_dist", "grad_eV"], ["rough_meV", "nn_dist", "grad_eV", "formula_err"]):
    print("  controls %-45s shear effect on log error: %+.2f (x%.1f)" % (cols, fit(cols), np.exp(fit(cols))))
D.to_csv(M + "/results/shear_tests/shear_difficulty_states.csv", index=False)
