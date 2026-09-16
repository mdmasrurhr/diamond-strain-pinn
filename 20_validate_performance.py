"""Measure inference time, memory use, training cost, and how many predictions
repay the cost of training.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C

# One HSE06 static calculation on this cell, in core-hours. This is the cost the
# surrogate is compared against. Replace it with the value read from the
# reference calculations' own timing output.
DFT_CORE_HOURS = 8.0
DFT_COST_IS_VERIFIED = False


# train_pinn is imported, not re-implemented: a check that tests its own copy of
# the model is not testing the model. Importing it also runs its argument parser
# with default values, which is why --data_csv has a default.
import train_pinn


def _load_trainer(extra_argv=()):
    """Return the trainer module. The argument is accepted but unused."""
    return train_pinn


def _time_forward(model, X, dev, reps=200):
    """Median forward-pass time. Median, not mean: the first calls are warm-up."""
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default=None,
                    help="a finished run; without it an untrained network is "
                         "timed, which is identical for latency purposes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    T = _load_trainer()
    data = T.load_and_split(C.DATA_CSV, seed=args.seed, dft_pct=100)
    rows = []

    for devname in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        dev = torch.device(devname)
        model = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)
        if args.run_dir:
            w = os.path.join(args.run_dir, "best_model.pt")
            if os.path.exists(w):
                model.load_state_dict(torch.load(w, map_location=dev))
        model.eval()

        print("\nINFERENCE on %s" % devname)
        for n in (1, 100, 1292):
            X = data["X_te"][:1].repeat(n, axis=0) if n > len(data["X_te"]) \
                else np.resize(data["X_te"], (n, data["X_te"].shape[1]))
            t = _time_forward(model, X, dev)
            per = t / n
            print("  batch %5d : %9.3f ms total, %9.4f ms per sample"
                  % (n, t * 1e3, per * 1e3))
            rows.append(dict(check="latency", device=devname, batch=n,
                             total_ms=round(t * 1e3, 4),
                             per_sample_ms=round(per * 1e3, 6)))

    # ------------------------------------------------------------- memory
    model_cpu = T.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"])
    n_par = sum(p.numel() for p in model_cpu.parameters())
    bytes_fp32 = n_par * 4
    print("\nFOOTPRINT")
    print("  trainable parameters %d" % n_par)
    print("  weights at fp32      %.1f kB" % (bytes_fp32 / 1024.0))
    print("  -> small enough to run anywhere; the deployment constraint is not")
    print("     this model's size.")
    rows.append(dict(check="footprint", device="", batch=0,
                     total_ms=float("nan"), per_sample_ms=float("nan")))
    rows[-1].update(n_params=n_par, weights_kB=round(bytes_fp32 / 1024.0, 1))

    # ----------------------------------------------------------- training
    df = decide_collect()
    if df is not None and len(df):
        per_run = float(df.elapsed_s.mean())
        total_h = float(df.elapsed_s.sum()) / 3600.0
        print("\nTRAINING COST  (from %d recorded runs)" % len(df))
        print("  mean per run   %.1f s" % per_run)
        print("  total recorded %.1f GPU-hours" % total_h)
        rows.append(dict(check="training", device="", batch=0,
                         total_ms=float("nan"), per_sample_ms=float("nan"),
                         mean_run_s=round(per_run, 1),
                         total_gpu_hours=round(total_h, 2)))

        # ------------------------------------------------------- payoff
        n_labels = 938
        dft_hours = n_labels * DFT_CORE_HOURS
        print("\nPAYOFF")
        print("  the labels cost   %.0f core-hours (%d HSE06 runs x %.1f h)"
              % (dft_hours, n_labels, DFT_CORE_HOURS))
        print("  one model costs   %.2f GPU-hours to train" % (per_run / 3600.0))
        print("  one prediction    %.4f ms" % (rows[0]["per_sample_ms"]))
        import math
        train_h = per_run / 3600.0
        n_break = max(1, math.ceil(train_h / DFT_CORE_HOURS))
        print("  -> training costs less than %d DFT calculation(s), so the model"
              % n_break)
        print("     repays its own training after %d prediction(s)." % n_break)
        print("     Including the dataset, it breaks even after %d predictions."
              % (n_labels + n_break))
        print("  The case is not that the model is cheap to build. It is that once")
        print("  built it answers in %.1f ms a question DFT charges %.0f core-hours for."
              % (rows[0]["per_sample_ms"], DFT_CORE_HOURS))
        if not DFT_COST_IS_VERIFIED:
            print("\n  NOTE: DFT_CORE_HOURS is a placeholder, see the top of this file.")

    out = C.results_dir("validate_performance")
    pd.DataFrame(rows).to_csv(os.path.join(out, "operational.csv"), index=False)
    print("\nwrote %s/operational.csv" % os.path.relpath(out, C.ROOT))


def decide_collect():
    import decide
    for sub in ("label_sweep",):
        df = decide.collect(os.path.join(C.RESULTS, sub), ["model", "pct", "seed"])
        if len(df):
            return df
    return None


if __name__ == "__main__":
    main()
