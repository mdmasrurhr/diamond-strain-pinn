"""Test A: does a sharper conduction-valley minimum at evaluation change the error of trained final-model runs? (no retraining)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root; run from there
os.makedirs(ROOT + "/results/shear_tests", exist_ok=True)

import sys, glob, importlib.util, numpy as np, pandas as pd, torch
M = ROOT; REPO = M + ""; R = M + "/results"
S = pd.read_csv(M + "/results/shear_tests/shear_difficulty_states.csv")
rows = []
for pct in (25, 100):
    for run in sorted(glob.glob("%s/B_main/hyb_sdp_gap_ld9/%dpct/seed*" % (R, pct))):
        tok = open(run + "/command.txt").read().split()
        for t in tok:
            if t.startswith("PHYSICS_CONST_JSON="): os.environ["PHYSICS_CONST_JSON"] = REPO + "/" + t.split("=", 1)[1]
        argv = tok[next(i for i, t in enumerate(tok) if t.endswith("03_train_pinn.py")):]
        sys.argv = [REPO + "/03_train_pinn.py"] + argv[1:] + ["--device", "cpu", "--results_dir", "/tmp/_beta_stub", "--data_csv", REPO + "/data/dft_hy_v9.csv"]
        sys.path.insert(0, REPO); spec = importlib.util.spec_from_file_location("tb%d" % hash(run), REPO + "/03_train_pinn.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        seed = int(argv[argv.index("--seed") + 1]); d = T.load_and_split(REPO + "/data/dft_hy_v9.csv", seed, pct); sc = d["scaler"]
        net = T.PINNModel(d["cbm_mean_tr"], d["vbm_mean_tr"], scaler=sc); net.load_state_dict(torch.load(run + "/best_model.pt", map_location="cpu", weights_only=False)); net.eval()
        te = np.array(d["idx_te"]); X = torch.tensor(d["X_te"].astype(np.float32)); y = d["eg_te"].astype(np.float64)
        sub = S.iloc[te]; shear = (sub.fam == "shear").values; multi = shear & (sub.n_shear_planes >= 2).values; vc = multi & (sub.axis_switch > 0).values
        for b in (50, 100, 200, 500, 5000):
            T.args.beta = float(b)
            with torch.no_grad(): p = net(X)[2].numpy().astype(np.float64)
            e = np.abs(p - y) * 1000
            rows.append(dict(pct=pct, seed=seed, beta=b, all=e.mean(), shear=e[shear].mean(), multi=e[multi].mean(), valley_change=e[vc].mean() if vc.any() else np.nan, rest=e[~vc].mean()))
D = pd.DataFrame(rows); print(D.groupby(["pct", "beta"])[["all", "shear", "multi", "valley_change", "rest"]].mean().round(2).to_string())
