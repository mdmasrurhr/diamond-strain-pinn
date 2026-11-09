"""Test A: does a sharper conduction-valley minimum at evaluation change the error of the trained PHM runs? (no retraining)
Usage, from the repository root: python analysis/shear_tests/beta_test.py [results folder, default results]
Writes results/shear_tests/beta_test.csv (one row per run and beta)."""
import os, sys, glob, importlib.util, numpy as np, pandas as pd, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ROOT + "/results"
os.makedirs(ROOT + "/results/shear_tests", exist_ok=True)
DATA = pd.read_csv(ROOT + "/data/dft_hy_v9.csv")
rows = []
for pct in (25, 100):
    for run in sorted(glob.glob("%s/B_main/phm/%dpct/seed*" % (R, pct))):
        tok = open(run + "/command.txt").read().split()
        for t in tok:
            if t.startswith("PHYSICS_CONST_JSON="): os.environ["PHYSICS_CONST_JSON"] = ROOT + "/" + t.split("=", 1)[1]
        argv = tok[next(i for i, t in enumerate(tok) if t.endswith("03_train_pinn.py")):]
        sys.argv = [ROOT + "/03_train_pinn.py"] + argv[1:] + ["--device", "cpu", "--results_dir", "/tmp/_beta_stub", "--data_csv", ROOT + "/data/dft_hy_v9.csv"]
        sys.modules.pop("physics", None)   # physics.py reads the run's constants when imported
        sys.path.insert(0, ROOT); spec =importlib.util.spec_from_file_location("tb%d" % abs(hash(run)), ROOT + "/03_train_pinn.py")
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        seed = int(argv[argv.index("--seed") + 1]); d = T.load_and_split(ROOT + "/data/dft_hy_v9.csv", seed, pct)
        net = T.PINNModel(d["cbm_mean_tr"], d["vbm_mean_tr"], scaler=d["scaler"])
        net.load_state_dict(torch.load(run + "/best_model.pt", map_location="cpu", weights_only=False)); net.eval()
        te = np.array(d["idx_te"]); X = torch.tensor(d["X_te"].astype(np.float32)); y = d["eg_te"].astype(np.float64)
        sub = DATA.iloc[te]; shear = (sub.strain_type == "shear").values; multi = shear & (sub.n_shear_planes >= 2).values
        for b in (50, 100, 200, 500, 5000):
            T.args.beta = float(b)
            with torch.no_grad(): p = net(X)[2].numpy().astype(np.float64)
            e = np.abs(p - y) * 1000
            rows.append(dict(pct=pct, seed=seed, beta=b, all=e.mean(), shear=e[shear].mean(), multi=e[multi].mean(), rest=e[~shear].mean()))
D = pd.DataFrame(rows); D.to_csv(ROOT + "/results/shear_tests/beta_test.csv", index=False)
print(D.groupby(["pct", "beta"])[["all", "shear", "multi", "rest"]].mean().round(2).to_string())
