"""Part 2: is the label roughness of multi-plane shear physical (conduction valley changes) or a k-grid effect?"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root; run from there
os.makedirs(ROOT + "/results/shear_tests", exist_ok=True)

import sys, numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
from scipy import stats
M = ROOT
os.environ["PHYSICS_CONST_JSON"] = M + "/results/98_constants_v2/symmetry/const_seed42.json"; sys.path.insert(0, M + "")
import physics as P
D = pd.read_csv(M + "/results/shear_tests/shear_difficulty_states.csv"); SC = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
X = D[SC].values; y = D.Eg_eV.values; xt = torch.tensor(X, dtype=torch.float64)
# analytic conduction valley that is lowest (0,1: z axis; 2,3: y; 4,5: x) -> the physical valley axis
I1 = xt[:, :3].sum(1)
def val(e): return P.XI_D * I1 + P.XI_U * e + P.XI_Q * e ** 2 + P.XI_C * I1 * e + P.XI_Q2 * I1 ** 2
d = torch.stack([val(xt[:, 2]) + 2 * P.XI_UP * xt[:, 3], val(xt[:, 2]) - 2 * P.XI_UP * xt[:, 3], val(xt[:, 1]) + 2 * P.XI_UP * xt[:, 5], val(xt[:, 1]) - 2 * P.XI_UP * xt[:, 5],
                 val(xt[:, 0]) + 2 * P.XI_UP * xt[:, 4], val(xt[:, 0]) - 2 * P.XI_UP * xt[:, 4]], 1)
D["valley"] = d.argmin(1).numpy(); D["axis"] = D.valley // 2
tree = cKDTree(X); dist, idx = tree.query(X, k=7); kp = D.cbm_kpt_index.values
D["axis_switch"] = [(D.axis.values[idx[i, 1:]] != D.axis.values[i]).mean() for i in range(len(D))]            # physical: another valley axis is lowest
D["valley_switch"] = [(D.valley.values[idx[i, 1:]] != D.valley.values[i]).mean() for i in range(len(D))]
D["grid_only"] = [((kp[idx[i, 1:]] != kp[i]) & (D.axis.values[idx[i, 1:]] == D.axis.values[i])).mean() for i in range(len(D))]   # grid point changes inside one valley axis
S = D[D.fam == "shear"]
print("Shear states, label roughness (departure from a local plane, meV) and error of our model, by sheared planes")
for n in (1, 2, 3):
    g = S[S.n_shear_planes == n]; print("  %d plane(s): roughness %.0f  error %.1f  | physical axis change among neighbours %.2f  grid-only change %.2f" % (n, g.rough_meV.mean(), g.err_ours.mean(), g.axis_switch.mean(), g.grid_only.mean()))
print("\nRoughness (meV) of shear states with 2-3 planes, by what changes in the neighbourhood")
G = S[S.n_shear_planes >= 2]
for name, m in [("nothing", (G.axis_switch == 0) & (G.grid_only == 0)), ("grid point only", (G.axis_switch == 0) & (G.grid_only > 0)), ("valley axis", G.axis_switch > 0)]:
    print("  %-16s n=%3d  roughness %.0f  error ours %.1f  error MLP %.1f" % (name, m.sum(), G[m].rough_meV.mean(), G[m].err_ours.mean(), G[m].err_mlp.mean()))
print("\nSpearman with roughness (2-3 plane shear): axis_switch %+.2f  grid_only %+.2f ; with error of our model: axis_switch %+.2f  grid_only %+.2f" % (
      stats.spearmanr(G.axis_switch, G.rough_meV)[0], stats.spearmanr(G.grid_only, G.rough_meV)[0], stats.spearmanr(G.axis_switch, G.err_ours)[0], stats.spearmanr(G.grid_only, G.err_ours)[0]))
# label jumps between nearest neighbours: change of gap per unit strain distance, same vs different grid point
print("\nNearest-neighbour label jumps, |dEg|/distance (eV per unit strain), shear states with 2-3 planes")
j = idx[:, 1]; dd = dist[:, 1]; same = kp == kp[j]; sameaxis = D.axis.values == D.axis.values[j]
slope = np.abs(y - y[j]) / dd; m23 = ((D.fam == "shear") & (D.n_shear_planes >= 2)).values
for name, m in [("same grid point", same), ("different grid point, same axis", ~same & sameaxis), ("different valley axis", ~sameaxis)]:
    mm = m & m23; print("  %-32s n=%3d  median slope %.0f  90th percentile %.0f" % (name, mm.sum(), np.median(slope[mm]), np.percentile(slope[mm], 90)))
D.to_csv(M + "/results/shear_tests/shear_difficulty_states.csv", index=False)
