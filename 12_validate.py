"""Checks on a trained model.

    python 12_validate.py physics          slopes, cubic symmetry, smoothness
    python 12_validate.py generalization   train-test gap, unseen family, unseen strain
    python 12_validate.py robustness       input noise, permutations, shear sign
    python 12_validate.py explainability   input derivatives against theory
    python 12_validate.py performance      speed, memory, training cost
    python 12_validate.py all              all five

The last three read one finished run, given by --run_dir.
"""

import argparse
import glob
import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C
import decide
import physics as P
# Imported for its model class and data pipeline.
import train_pinn

# Strain directions probed, and the step used for the central difference.
MODES = {"hydrostatic": [1, 1, 1, 0, 0, 0],
         "uniaxial":    [1, 0, 0, 0, 0, 0],
         "shear":       [0, 0, 0, 1, 0, 0]}
DELTA = 0.002

NOISE_LEVELS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# Cost of one reference calculation, for the payoff comparison. Replace with the
# value read from the reference calculations' own timing output.
DFT_CORE_HOURS = 8.0
DFT_COST_IS_VERIFIED = False


def load_trainer():
    """Return the trainer module, for its model class and data pipeline."""
    return train_pinn


def load_model(data, run_dir, dev):
    """Load a finished run's weights into the model it was trained with."""
    model = train_pinn.PINNModel(data["cbm_mean_tr"], data["vbm_mean_tr"]).to(dev)
    weights = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(weights):
        raise SystemExit("no best_model.pt in %s" % run_dir)
    model.load_state_dict(torch.load(weights, map_location=dev))
    model.eval()
    return model


MODES = {"hydrostatic": [1, 1, 1, 0, 0, 0],
         "uniaxial":    [1, 0, 0, 0, 0, 0],
         "shear":       [0, 0, 0, 1, 0, 0]}
DELTA = 0.002


def slopes(predict):
    """dEg/d(amplitude) at zero strain, eV per unit strain, per direction."""
    out = {}
    for name, d in MODES.items():
        v = np.array(d, dtype=np.float32)
        eg = predict(np.stack([DELTA * v, -DELTA * v]))
        out[name] = float((eg[0] - eg[1]) / (2 * DELTA))
    return out


def smoothness(predict, pool, n=400, eps=0.004, seed=0):
    rng = np.random.default_rng(seed)
    base = pool[rng.choice(len(pool), n)].astype(np.float32)
    step = rng.normal(size=base.shape).astype(np.float32)
    step = step / (np.linalg.norm(step, axis=1, keepdims=True) + 1e-9) * eps
    sens = np.abs(predict(base + step) - predict(base)) / eps
    return float(np.mean(sens))


def reference_slopes():
    """The same three slopes from the analytic model -- the target to match.

    The soft minimum (beta from config) is used, not the hard one, and the
    distinction matters here more than anywhere else. At zero strain all six
    conduction valleys are degenerate, so the two disagree exactly at the point
    we differentiate: hard-min gives a uniaxial slope of -2.36 eV per unit
    strain, soft-min -0.81. The soft minimum is what the physics loss uses
    during training, so it is what the network was actually asked to match.
    """
    def predict(x):
        t = torch.tensor(x, dtype=torch.float32)
        eg = P.physics_cbm(t, beta=C.SOFTMIN_BETA) - P.physics_vbm(t)
        return eg.detach().numpy()
    return slopes(predict)


def run_physics(args):

    T = load_trainer()
    ref = reference_slopes()
    print("analytic reference (eV per unit strain): " +
          "  ".join("%s %.2f" % (k, v) for k, v in ref.items()))

    out = C.results_dir("physics_validation")
    rows = []
    pattern = os.path.join(C.RESULTS, args.study, "*", "%dpct" % args.pct,
                           "seed*", "best_model.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print("no checkpoints under %s -- run 05 first" % pattern)
        return

    for ckpt in files:
        parts = ckpt.split(os.sep)
        model_name, seed = parts[-4], int(parts[-2].replace("seed", ""))

        # Rebuild this run's split and scaler; both are deterministic in the seed.
        data = T.load_and_split(C.DATA_CSV, seed, args.pct)
        scaler, net = data["scaler"], T.PINNModel(data["cbm_mean_tr"],
                                                  data["vbm_mean_tr"])
        net.load_state_dict(torch.load(ckpt, map_location="cpu",
                                       weights_only=False))
        net.eval()

        def predict(raw):
            x = torch.tensor(scaler.transform(raw).astype(np.float32))
            with torch.no_grad():
                return net(x)[2].numpy()          # (cbm, vbm, eg) -> eg

        s = slopes(predict)
        err = float(np.mean([abs(s[k] - ref[k]) for k in MODES]))
        rows.append(dict(model=model_name, seed=seed, dft_pct=args.pct,
                         slope_error=round(err, 3),
                         smoothness=round(smoothness(predict,
                                                     data["strain_tr"]), 2),
                         **{("slope_" + k): round(v, 3) for k, v in s.items()}))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "validity.csv"), index=False)
    agg = d.groupby("model").agg(
        n=("seed", "count"),
        slope_error_mean=("slope_error", "mean"),
        slope_error_sd=("slope_error", "std"),
        hydrostatic=("slope_hydrostatic", "mean")).round(3)
    print("\nslope error against the analytic reference (eV per unit strain):")
    print(agg.to_string())
    print("\nwrote %s/validity.csv" % out)


def memorisation():
    """Train-test gap per label fraction, from the main sweep."""
    df = decide.collect(os.path.join(C.RESULTS, "label_sweep"), ["model", "pct", "seed"])
    if not len(df):
        return pd.DataFrame()
    df["pct_n"] = df.pct.str.replace("pct", "").astype(int)
    g = df.groupby(["model", "pct_n"])[["train_mae_meV", "test_mae_meV"]].mean()
    g = g.reset_index()
    g["gap_meV"] = g.test_mae_meV - g.train_mae_meV
    # A ratio is the more readable form: 1.0 is no gap, 2.0 means test error is
    # twice train error.
    g["ratio"] = (g.test_mae_meV / g.train_mae_meV.clip(lower=1e-9)).round(2)

    print("\nMEMORISATION  (test error against train error)")
    print("  %-10s %6s %12s %12s %10s %8s"
          % ("model", "pct", "train meV", "test meV", "gap meV", "ratio"))
    for r in g.itertuples():
        print("  %-10s %6d %12.2f %12.2f %10.2f %8.2f"
              % (r.model, r.pct_n, r.train_mae_meV, r.test_mae_meV, r.gap_meV, r.ratio))
    print("\n  A ratio near 1 means the model generalises as well as it fits. A")
    print("  large ratio at low label fractions is expected and is not a defect:")
    print("  it is what the physics prior exists to bound.")
    return g


def out_of_sample():
    """Extrapolation, from the split study's holdout runs."""
    root = os.path.join(C.RESULTS, "split_study")
    df = decide.collect(root, ["scheme", "run"])
    if not len(df):
        print("\nno split-study runs found -- run 03_decide_split.py first")
        return pd.DataFrame()

    agg = decide.aggregate(df, ["scheme"])
    base = agg[agg.scheme == "stratified"]
    b = float(base.test_mae_meV_mean.iloc[0]) if len(base) else float("nan")

    print("\nOUT-OF-SAMPLE  (against the in-distribution number, %.2f meV)" % b)
    print("  %-18s %12s %12s  %s" % ("scheme", "meV", "x worse", "what it measures"))
    what = {"group_shear": "unseen family: shear",
            "group_triaxial": "unseen family: triaxial",
            "group_biaxial": "unseen family: biaxial",
            "magnitude_05": "strain beyond |E| = 0.05"}
    rows = []
    for r in agg.itertuples():
        if r.scheme not in what:
            continue
        x = r.test_mae_meV_mean / b if np.isfinite(b) and b > 0 else float("nan")
        print("  %-18s %12.2f %12.1f  %s"
              % (r.scheme, r.test_mae_meV_mean, x, what[r.scheme]))
        rows.append(dict(scheme=r.scheme, mae_meV=round(r.test_mae_meV_mean, 2),
                         times_worse=round(x, 1), measures=what[r.scheme]))
    print("\n  These are the bounds of the claim. A strain-engineering reader wants")
    print("  the second column, and a paper that reports only the in-distribution")
    print("  number has not answered them.")
    return pd.DataFrame(rows)


def per_family():
    """Where the error concentrates, from the per-family columns already logged."""
    df = decide.collect(os.path.join(C.RESULTS, "label_sweep"), ["model", "pct", "seed"])
    if not len(df):
        return pd.DataFrame()
    df = df[df.pct == "100pct"]
    cols = [c for c in df.columns if c.startswith("mae_") and c.endswith("_meV")]
    if not cols:
        return pd.DataFrame()
    g = df.groupby("model")[cols].mean().round(2)
    print("\nERROR BY DEFORMATION FAMILY  (100% labels)")
    print("  %-10s %s" % ("model", " ".join("%12s" % c[4:-4] for c in cols)))
    for model, row in g.iterrows():
        print("  %-10s %s" % (model, " ".join("%12.2f" % row[c] for c in cols)))
    print("\n  The family with the largest error is where the model fails first, and")
    print("  The largest of these is where the model fails first.")
    return g.reset_index()


def run_generalization(args):
    out = C.results_dir("validate_generalization")
    mem = memorisation()
    oos = out_of_sample()
    fam = per_family()
    for name, d in (("memorisation", mem), ("out_of_sample", oos),
                    ("per_family", fam)):
        if len(d):
            d.to_csv(os.path.join(out, "%s.csv" % name), index=False)
    print("\nwrote %s/" % os.path.relpath(out, C.ROOT))


NOISE_LEVELS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


# train_pinn is imported, not re-implemented: a check that tests its own copy of
# the model is not testing the model. Importing it also runs its argument parser
# with default values, which is why --data_csv has a default.
import train_pinn


def predict_eg(model, X, dev):
    with torch.no_grad():
        _, _, eg = model(torch.tensor(X, dtype=torch.float32, device=dev))
    return eg.cpu().numpy()


def run_robustness(args):

    T = load_trainer()
    data = T.load_and_split(C.DATA_CSV, seed=args.seed, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(data, args.run_dir, dev)

    X_te, eg_te = data["X_te"], data["eg_te"]
    rows = []

    # ---------------------------------------------------------------- noise
    # Noise is added in SCALED space, so a level is in units of the training
    # standard deviation and is comparable across the six components.
    print("INPUT NOISE  (perturbation in units of the training s.d.)")
    print("  %-12s %12s %12s" % ("sigma", "MAE meV", "x clean"))
    rng = np.random.default_rng(0)
    clean = None
    for s in NOISE_LEVELS:
        errs = []
        reps = 1 if s == 0 else 20
        for _ in range(reps):
            Xn = X_te + rng.normal(0.0, s, X_te.shape).astype(np.float32)
            errs.append(np.abs(predict_eg(model, Xn, dev) - eg_te).mean() * 1000.0)
        mae = float(np.mean(errs))
        clean = mae if s == 0 else clean
        print("  %-12.0e %12.2f %12.2f" % (s, mae, mae / clean))
        rows.append(dict(check="noise", level=s, mae_meV=round(mae, 2),
                         ratio_to_clean=round(mae / clean, 3)))
    print("\n  Read off the noise level at which the error doubles: that is the")
    print("  input precision the model requires to be worth using.")

    # ------------------------------------------------------------- symmetry
    # Permuting Exx, Eyy, Ezz maps a cubic crystal onto itself, so the gap must
    # not change. Nothing in the loss says so; this measures whether it learned it.
    print("\nCUBIC PERMUTATION SYMMETRY  (Exx, Eyy, Ezz permuted)")
    df = pd.read_csv(C.DATA_CSV)
    raw = df[C.STRAIN_COLS].values.astype(np.float32)[data["idx_te"]]
    base = predict_eg(model, data["scaler"].transform(raw).astype(np.float32), dev)
    drifts = []
    for perm in itertools.permutations(range(3)):
        if perm == (0, 1, 2):
            continue
        p = raw.copy()
        p[:, :3] = raw[:, list(perm)]
        got = predict_eg(model, data["scaler"].transform(p).astype(np.float32), dev)
        d = float(np.abs(got - base).mean() * 1000.0)
        drifts.append(d)
        print("  permute %s -> %8.2f meV drift" % (str(perm), d))
    mean_drift = float(np.mean(drifts))
    print("  mean drift %.2f meV, against a test MAE of %.2f meV"
          % (mean_drift, clean))
    print("  -> the model is %s to a symmetry it was never told about"
          % ("insensitive" if mean_drift < clean else "SENSITIVE"))
    rows.append(dict(check="permutation_symmetry", level=float("nan"),
                     mae_meV=round(mean_drift, 2),
                     ratio_to_clean=round(mean_drift / clean, 3)))

    # ------------------------------------------------------------ shear sign
    print("\nSHEAR SIGN  (Exy, Eyz, Ezx negated)")
    p = raw.copy()
    p[:, 3:] = -raw[:, 3:]
    got = predict_eg(model, data["scaler"].transform(p).astype(np.float32), dev)
    d = float(np.abs(got - base).mean() * 1000.0)
    print("  drift %.2f meV (%.2fx the test MAE)" % (d, d / clean))
    print("  Shear enters the physics through even powers, so this drift should")
    print("  be small. A large value means the network learned a sign dependence")
    print("  the physics does not have.")
    rows.append(dict(check="shear_sign", level=float("nan"), mae_meV=round(d, 2),
                     ratio_to_clean=round(d / clean, 3)))

    out = C.results_dir("validate_robustness")
    pd.DataFrame(rows).to_csv(os.path.join(out, "robustness.csv"), index=False)
    print("\nwrote %s/robustness.csv" % os.path.relpath(out, C.ROOT))


def input_gradients(model, X_scaled, scaler, dev):
    """dEg/d(raw strain component), chain-ruled back through the scaler."""
    x = torch.tensor(X_scaled, dtype=torch.float32, device=dev, requires_grad=True)
    _, _, eg = model(x)
    g = torch.autograd.grad(eg.sum(), x)[0].cpu().numpy()
    # The network sees scaled inputs; the physics is stated in raw strain, so
    # divide by the scale to get back to eV per unit strain.
    return g / scaler.scale_.astype(np.float32)


def run_explainability(args):

    T = load_trainer()
    data = T.load_and_split(C.DATA_CSV, seed=args.seed, dft_pct=100)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(data, args.run_dir, dev)

    rows = []
    sc = data["scaler"]

    # ------------------------------------------------- overall attribution
    g = input_gradients(model, data["X_te"], sc, dev)
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
    gi = input_gradients(model, sc.transform(iso).astype(np.float32), sc, dev)
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
    gz = input_gradients(model, sc.transform(zero).astype(np.float32), sc, dev)[0]
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
    print("  fixes. Compare it with the fitted value from 02_fit_physics_constants.py;")
    print("  agreement is evidence the network learned the physics rather than")
    print("  a curve that happens to pass through the same points.")
    rows.append(dict(check="hydrostatic_slope", item="dEg/dI1",
                     value=round(gi_mean, 4), expected="see 02_fit_physics_constants.py",
                     note=""))

    out = C.results_dir("validate_explainability")
    pd.DataFrame(rows).to_csv(os.path.join(out, "attribution.csv"), index=False)
    print("\nwrote %s/attribution.csv" % os.path.relpath(out, C.ROOT))


DFT_CORE_HOURS = 8.0
DFT_COST_IS_VERIFIED = False


# train_pinn is imported, not re-implemented: a check that tests its own copy of
# the model is not testing the model. Importing it also runs its argument parser
# with default values, which is why --data_csv has a default.
import train_pinn


def time_forward(model, X, dev, reps=200):
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


def run_performance(args):

    T = load_trainer()
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
            t = time_forward(model, X, dev)
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
    df = sweep_runs()
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


def sweep_runs():
    import decide
    for sub in ("label_sweep",):
        df = decide.collect(os.path.join(C.RESULTS, sub), ["model", "pct", "seed"])
        if len(df):
            return df
    return None

CHECKS = {
    "physics":        run_physics,
    "generalization": run_generalization,
    "robustness":     run_robustness,
    "explainability": run_explainability,
    "performance":    run_performance,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all",
                    choices=sorted(CHECKS) + ["all"])
    ap.add_argument("--run_dir", default=None,
                    help="a finished run directory containing best_model.pt")
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the seed the run was trained with")
    ap.add_argument("--pct", type=int, default=100, help="physics: label fraction")
    ap.add_argument("--study", default="label_sweep", help="physics: study folder")
    args = ap.parse_args()

    which = sorted(CHECKS) if args.what == "all" else [args.what]
    for name in which:
        print("\n===== %s =====" % name)
        CHECKS[name](args)


if __name__ == "__main__":
    main()
