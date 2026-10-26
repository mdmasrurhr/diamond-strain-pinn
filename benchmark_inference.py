#!/usr/bin/env python3
"""Inference cost of a trained model: time to predict the band edges of all 1,292 strain states on one CPU thread.
Usage: python benchmark_inference.py <run_dir> [<run_dir> ...]   (reads <run_dir>/command.txt to rebuild the model)"""
import os, sys, time, shlex, importlib.util, numpy as np, torch, pandas as pd
torch.set_num_threads(1)
for run in sys.argv[1:]:
    cmd = open(os.path.join(run, "command.txt")).read().split()
    env = [c for c in cmd if c.startswith("PHYSICS_CONST_JSON=")]
    if env: os.environ["PHYSICS_CONST_JSON"] = env[0].split("=", 1)[1]
    argv = cmd[cmd.index("03_train_pinn.py"):]
    sys.argv = argv[:]; spec = importlib.util.spec_from_file_location("tr" + str(abs(hash(run))), "03_train_pinn.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    seed = int(argv[argv.index("--seed") + 1]); pct = int(argv[argv.index("--dft_pct") + 1])
    d = T.load_and_split("data/dft_hy_v9.csv", seed, pct)
    net = T.PINNModel(d["cbm_mean_tr"], d["vbm_mean_tr"], scaler=d["scaler"]); net.load_state_dict(torch.load(os.path.join(run, "best_model.pt"), map_location="cpu", weights_only=False)); net.eval()
    df = pd.read_csv("data/dft_hy_v9.csv"); X = torch.tensor(d["scaler"].transform(df[["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]].values).astype(np.float32))
    with torch.no_grad():
        net(X); t = []
        for _ in range(20):
            s = time.perf_counter(); net(X); t.append(time.perf_counter() - s)
        one = []
        for i in range(200):
            s = time.perf_counter(); net(X[i:i + 1]); one.append(time.perf_counter() - s)
    n_par = sum(p.numel() for p in net.parameters())
    print("%s: %d parameters | all 1,292 states in %.2f ms (%.2f us per state, batched) | single state %.3f ms (median)" % (run, n_par, 1000 * np.median(t), 1e6 * np.median(t) / len(X), 1000 * np.median(one)))
