"""Prediction time of trained networks for all 1,292 strain states on one CPU thread.

Usage: python benchmark_inference.py <run folder> [<run folder> ...]
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C

torch.set_num_threads(1)
for run in sys.argv[1:]:
    T, d, net = C.load_trained_run(run)
    df = pd.read_csv(C.DATA_CSV)
    X = torch.tensor(d["scaler"].transform(df[T.STRAIN_COLUMNS].values).astype(np.float32))
    with torch.no_grad():
        net(X)
        batched = []
        for _ in range(20):
            start = time.perf_counter()
            net(X)
            batched.append(time.perf_counter() - start)
        single = []
        for i in range(200):
            start = time.perf_counter()
            net(X[i : i + 1])
            single.append(time.perf_counter() - start)
    print(
        "%s: %d parameters | all %d states in %.2f ms (%.2f us per state) | one state %.3f ms (median)"
        % (
            run,
            sum(p.numel() for p in net.parameters()),
            len(X),
            1000 * np.median(batched),
            1e6 * np.median(batched) / len(X),
            1000 * np.median(single),
        )
    )
