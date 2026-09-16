"""Train one purely data-driven network, reproduced from the published architecture,
and write its results.
"""
# --- imports and argument parsing ---
import argparse
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Repo root resolved by the path header (walks up to the .git dir)
HERE = Path(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser(description="Shi et al. (2019) ANN baseline.")
    ap.add_argument("--data_csv",    type=str, required=True,
                    help="HSE06 target dataset CSV.")
    ap.add_argument("--pbe_csv",     type=str, default=None,
                    help="GGA-PBE dataset CSV (required for --mode delta_ml).")
    ap.add_argument("--mode",        type=str, default="pbe_only",
                    choices=["pbe_only", "delta_ml"],
                    help="pbe_only: train on HSE06 directly. "
                         "delta_ml: two-stage PBE+correction pipeline.")
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--dft_pct", type=int, default=100,
                    choices=[0, 1, 2, 5, 10, 25, 50, 75, 100])
    ap.add_argument("--epochs",  type=int, default=6000)
    ap.add_argument("--device",  type=str, default="auto")
    ap.add_argument("--cv_fold",   type=int, default=None)
    ap.add_argument("--cv_nfolds", type=int, default=5)
    return ap.parse_args()


def _model_name_from_mode(mode):
    return "Shi_ANN_DeltaML" if mode == "delta_ml" else "Shi_ANN"


# --- reproducibility setup ---

def setup_reproducibility(seed, device_str, results_dir, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))

    results_dir.mkdir(parents=True, exist_ok=True)
    model_name = _model_name_from_mode(args.mode)
    run_info = (
        f"script:    {Path(__file__).name}\n"
        f"model:     {model_name}\n"
        f"mode:      {args.mode}\n"
        f"seed:      {seed}\n"
        f"dft_pct:   {args.dft_pct}\n"
        f"epochs:    {args.epochs}\n"
        f"device:    {device}\n"
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"data_csv:  {args.data_csv}\n"
        f"pbe_csv:   {args.pbe_csv}\n"
    )
    (results_dir / "run_info.txt").write_text(run_info)
    print(run_info.strip())
    return device


# --- data loading and feature construction ---

def _norm_strain_type(s):
    s = str(s).lower().strip()
    if s.startswith("uniaxial"):  return "uniaxial"
    if s.startswith("biaxial"):   return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):  return s
    return "isotropic"


STRAIN_COLS = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]


def load_and_split(data_csv, pbe_csv, seed, dft_pct, results_dir,
                   cv_fold=None, cv_nfolds=5):
    """
    Load HSE06 (target) and optionally PBE (Stage 1) datasets.
    Both share identical strain states, so the same train/val/test split
    indices apply to both.
    Returns a data dict with HSE06 targets and, if pbe_csv given, PBE targets.
    """
    df_hse = pd.read_csv(data_csv)
    df_pbe = pd.read_csv(pbe_csv) if pbe_csv else None

    X_all     = df_hse[STRAIN_COLS].values.astype(np.float32)
    eg_hse    = df_hse["Eg_eV"].values.astype(np.float32)
    cbm_all   = df_hse["CBM_eV"].values.astype(np.float32)
    vbm_all   = df_hse["VBM_eV"].values.astype(np.float32)
    shear_all = df_hse["shear_planes"].values if "shear_planes" in df_hse.columns \
                else np.full(len(df_hse), "", dtype=object)
    idx_all   = np.arange(len(df_hse))

    stype_all = np.array([_norm_strain_type(s) for s in df_hse["strain_type"].values])

    if cv_fold is None:
        idx_trval, idx_te = train_test_split(
            idx_all, test_size=0.20, random_state=seed, stratify=stype_all
        )
        idx_tr, idx_va = train_test_split(
            idx_trval, test_size=0.125, random_state=seed, stratify=stype_all[idx_trval]
        )
    else:
        # Stratified k-fold (test = fold cv_fold); used only by the CV driver.
        skf = StratifiedKFold(n_splits=cv_nfolds, shuffle=True, random_state=seed)
        folds = list(skf.split(idx_all, stype_all))
        idx_trval, idx_te = folds[cv_fold][0], folds[cv_fold][1]
        idx_tr, idx_va = train_test_split(
            idx_trval, test_size=0.125, random_state=seed,
            stratify=stype_all[idx_trval]
        )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[idx_tr]).astype(np.float32)
    X_va = scaler.transform(X_all[idx_va]).astype(np.float32)
    X_te = scaler.transform(X_all[idx_te]).astype(np.float32)

    pd.DataFrame({
        "global_idx": np.concatenate([idx_tr, idx_va, idx_te]),
        "split": (["train"] * len(idx_tr) + ["val"] * len(idx_va)
                  + ["test"] * len(idx_te)),
    }).to_csv(results_dir / "split_indices.csv", index=False)

    # Same RNG offset as PINN -- ensures identical labeled subsets for fair comparison
    rng = np.random.default_rng(seed + 9999)
    if dft_pct == 0:
        dft_idx = np.array([], dtype=np.int64)
    else:
        n_lab   = max(1, round(len(idx_tr) * dft_pct / 100.0))
        dft_idx = rng.choice(len(idx_tr), size=n_lab, replace=False)

    data = dict(
        X_tr=X_tr, X_va=X_va, X_te=X_te,
        eg_hse_tr=eg_hse[idx_tr],
        eg_hse_va=eg_hse[idx_va],
        eg_hse_te=eg_hse[idx_te],
        cbm_te=cbm_all[idx_te],
        vbm_te=vbm_all[idx_te],
        stype_te=stype_all[idx_te],
        shear_te=shear_all[idx_te],
        idx_te=idx_te,
        dft_idx=dft_idx,
        eg_hse_mean_tr=float(np.mean(eg_hse[idx_tr])),
        N_tr=len(idx_tr),
    )

    if df_pbe is not None:
        eg_pbe = df_pbe["Eg_eV"].values.astype(np.float32)
        data["eg_pbe_tr"] = eg_pbe[idx_tr]
        data["eg_pbe_va"] = eg_pbe[idx_va]
        data["eg_pbe_te"] = eg_pbe[idx_te]
        data["eg_pbe_mean_tr"] = float(np.mean(eg_pbe[idx_tr]))

    return data


# --- neural network architecture ---

def build_model(output_mean, device):
    """
    Shi et al. (2019) SI Appendix Note S2: 512->256->256->256, LeakyReLU(0.01),
    Dropout(0.1), orthogonal init (Saxe et al. 2013), single scalar output.
    Used for both Stage 1 (predicts Eg_PBE) and Stage 2 (predicts Delta = HSE06-PBE).
    """
    layers = [
        nn.Linear(6, 512),   nn.LeakyReLU(0.01), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.LeakyReLU(0.01), nn.Dropout(0.1),
        nn.Linear(256, 256), nn.LeakyReLU(0.01), nn.Dropout(0.1),
        nn.Linear(256, 256), nn.LeakyReLU(0.01), nn.Dropout(0.1),
        nn.Linear(256, 1),
    ]
    net = nn.Sequential(*layers)

    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    with torch.no_grad():
        net[-1].bias[0] = output_mean

    class _ANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = _ANN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params={n_params:,}")
    return model


# ============================================================
# SECTION 5: TRAINING LOOP (shared for Stage 1 and Stage 2)
# ============================================================

def _gpu_mem_mb(device):
    return torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0


def _cpu_percent():
    return psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0


def _ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6 if HAS_PSUTIL else 0.0


def _train_one_stage(model, X_tr_t, y_tr_t, X_va_t, y_va_np,
                     train_idx_t, epochs, log_path, t0, device, label,
                     X_te_t=None, y_te_np=None):
    """
    Train model to predict y_tr_t[train_idx_t] with MSE loss.
    Appends to log_path. Returns (best_state, gpu_peak, best_val_mae).
    train_idx_t=None means use the full X_tr_t (Stage 1 uses all PBE samples).
    X_te_t/y_te_np, if given, log a held-out test MAE per epoch for the
    learning-curve figure; they never influence checkpoint selection.
    """
    opt      = optim.Adam(model.parameters(), lr=1e-3)
    best_val = np.inf
    best_ckpt_epoch = 0
    va_mae = float("nan")     # until the first checkpoint measures it
    best_state = None
    gpu_peak   = 0.0

    def _val_mae():
        model.eval()
        with torch.no_grad():
            p = model(X_va_t).cpu().numpy()
        model.train()
        return float(mean_absolute_error(y_va_np, p))

    def _test_mae():
        if X_te_t is None or y_te_np is None:
            return float("nan")
        model.eval()
        with torch.no_grad():
            p = model(X_te_t).cpu().numpy()
        model.train()
        return float(mean_absolute_error(y_te_np, p))

    print(f"\n  {label}  ({epochs} epochs, Adam lr=1e-3)")
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        preds = model(X_tr_t)
        if train_idx_t is not None:
            loss = nn.functional.mse_loss(preds[train_idx_t], y_tr_t[train_idx_t])
            tr_mae = float(mean_absolute_error(
                y_tr_t[train_idx_t].cpu().numpy(),
                preds[train_idx_t].detach().cpu().numpy()
            ))
        else:
            loss   = nn.functional.mse_loss(preds, y_tr_t)
            tr_mae = float(mean_absolute_error(
                y_tr_t.cpu().numpy(), preds.detach().cpu().numpy()
            ))
        loss.backward()
        opt.step()

        cur_lr = opt.param_groups[0]["lr"]
        gm     = _gpu_mem_mb(device)
        gpu_peak = max(gpu_peak, gm)

        # Validation on the checkpoint cadence, matching the PINN trainer. It used
        # to be measured every epoch while selection still happened every 250, so
        # the logged column moved on a different clock from the kept model.
        if ep % 250 == 0:
            with torch.no_grad():
                va_mae = _val_mae()
            if va_mae < best_val:
                best_val   = va_mae
                best_ckpt_epoch = ep
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

        te_mae = _test_mae()
        # val_mae_eV = as last measured; val_best_eV = the kept model. Two names
        # for two quantities, identical to train_pinn.py.
        vb = f"{best_val:.6f}" if best_val < float("inf") else ""
        with open(log_path, "a") as fh:
            fh.write(f"{ep},{tr_mae:.6f},{va_mae:.6f},{vb},{te_mae:.6f},{cur_lr:.2e},"
                     f"{time.time()-t0:.1f},{gm:.1f},{_cpu_percent():.1f},{_ram_mb():.1f}\n")

        if ep % 500 == 0:
            print(f"    ep={ep:>5}  lr={cur_lr:.2e}  val={best_val*1000:.1f}meV")

    if best_state:
        model.load_state_dict(best_state)
    return best_state, gpu_peak, best_val


# ============================================================
# SECTION 6: MODE DISPATCH -- pbe_only vs delta_ml
# ============================================================

def run_pbe_only(data, args, results_dir, device):
    """
    Train directly on HSE06 targets. Identical to previous Shi_ANN behaviour.
    """
    def T(a):
        """Move a numpy array onto the training device as float32."""
        return torch.tensor(a, dtype=torch.float32, device=device)

    t0 = time.time()

    dft_idx = data["dft_idx"]
    if len(dft_idx) == 0:
        return None, 0.0, 0.0, float("nan")

    X_tr_t = T(data["X_tr"])
    X_va_t = T(data["X_va"])
    X_te_t = T(data["X_te"])
    y_tr_t = T(data["eg_hse_tr"])
    dft_idx_t = torch.tensor(dft_idx, dtype=torch.long, device=device)

    log_path = results_dir / "epoch_log.csv"
    with open(log_path, "w") as fh:
        fh.write("epoch,train_mae_eV,val_mae_eV,val_best_eV,test_mae_eV,lr,elapsed_s,gpu_mem_mb,cpu_percent,ram_mb\n")

    model = build_model(data["eg_hse_mean_tr"], device)
    print("  Architecture: Shi 512->256->256->256 (Eg direct)", end="")

    best_state, gpu_peak, best_val = _train_one_stage(
        model, X_tr_t, y_tr_t, X_va_t, data["eg_hse_va"],
        dft_idx_t, args.epochs, log_path, t0, device,
        label="Stage: direct HSE06 regression",
        X_te_t=X_te_t, y_te_np=data["eg_hse_te"]
    )

    elapsed = time.time() - t0
    torch.save(best_state or model.state_dict(), results_dir / "best_model.pt")
    return model, elapsed, gpu_peak, best_val


def run_delta_ml(data, args, results_dir, device):
    """
    Two-stage Delta-ML pipeline reproducing Shi et al.'s headline result:
      Stage 1 -- ANN trained on ALL PBE samples (no dft_pct restriction).
      Stage 2 -- ANN trained on dft_pct-subsetted HSE06 residuals Delta = HSE06 - PBE_pred.
      Final:    Eg_pred = ANN_PBE(eps) + ANN_Delta(eps)
    """
    if "eg_pbe_tr" not in data:
        raise ValueError("--pbe_csv is required for --mode delta_ml")

    def T(a):
        """Move a numpy array onto the training device as float32."""
        return torch.tensor(a, dtype=torch.float32, device=device)

    t0 = time.time()

    X_tr_t = T(data["X_tr"])
    X_va_t = T(data["X_va"])
    X_te_t = T(data["X_te"])

    # -- Stage 1: train on full PBE training set (all 938 samples, no dft_pct) -
    print("\n  Stage 1 -- PBE regression")
    print("  Architecture: Shi 512->256->256->256", end="")
    model_pbe = build_model(data["eg_pbe_mean_tr"], device)

    log_path = results_dir / "epoch_log.csv"
    with open(log_path, "w") as fh:
        fh.write("epoch,train_mae_eV,val_mae_eV,val_best_eV,test_mae_eV,lr,elapsed_s,gpu_mem_mb,cpu_percent,ram_mb\n")

    best_state_pbe, gpu_peak1, _ = _train_one_stage(
        model_pbe, X_tr_t, T(data["eg_pbe_tr"]), X_va_t, data["eg_pbe_va"],
        None,   # train on ALL PBE samples
        args.epochs, log_path, t0, device,
        label="Stage 1: PBE regression (all samples)",
        X_te_t=X_te_t, y_te_np=data["eg_pbe_te"]
    )
    model_pbe.eval()
    torch.save(best_state_pbe or model_pbe.state_dict(),
               results_dir / "best_model_stage1_pbe.pt")

    # -- Compute residuals Delta = HSE06_Eg - PBE_pred on train/val splits ---
    with torch.no_grad():
        pbe_pred_tr = model_pbe(X_tr_t).cpu().numpy()
        pbe_pred_va = model_pbe(X_va_t).cpu().numpy()
        pbe_pred_te = model_pbe(X_te_t).cpu().numpy()

    delta_tr = data["eg_hse_tr"] - pbe_pred_tr   # shape (N_tr,)
    delta_va = data["eg_hse_va"] - pbe_pred_va
    delta_te = data["eg_hse_te"] - pbe_pred_te

    print(f"\n  Stage 1 PBE train MAE: {mean_absolute_error(data['eg_pbe_tr'], pbe_pred_tr)*1000:.1f} meV")
    print(f"  Residual Delta (train): mean={delta_tr.mean()*1000:.1f}  std={delta_tr.std()*1000:.1f} meV")

    # -- Stage 2: train correction NN on dft_pct-subsetted HSE06 residuals ---
    dft_idx = data["dft_idx"]
    if len(dft_idx) == 0:
        print("  dft_pct=0 -- no HSE06 labels for Stage 2, writing NaN row.")
        return None, time.time() - t0, gpu_peak1, float("nan")

    dft_idx_t = torch.tensor(dft_idx, dtype=torch.long, device=device)
    y_delta_t = T(delta_tr)

    print("\n  Stage 2 -- Delta correction regression")
    print("  Architecture: Shi 512->256->256->256", end="")
    delta_mean = float(np.mean(delta_tr[dft_idx]))
    model_delta = build_model(delta_mean, device)

    log_path2 = results_dir / "epoch_log_stage2.csv"
    with open(log_path2, "w") as fh:
        fh.write("epoch,train_mae_eV,val_mae_eV,val_best_eV,test_mae_eV,lr,elapsed_s,gpu_mem_mb,cpu_percent,ram_mb\n")

    best_state_delta, gpu_peak2, best_val_delta = _train_one_stage(
        model_delta, X_tr_t, y_delta_t, X_va_t, delta_va,
        dft_idx_t, args.epochs, log_path2, t0, device,
        label=f"Stage 2: Delta correction ({len(dft_idx)} HSE06 labels)",
        X_te_t=X_te_t, y_te_np=delta_te
    )
    model_delta.eval()
    torch.save(best_state_delta or model_delta.state_dict(),
               results_dir / "best_model_stage2_delta.pt")

    elapsed  = time.time() - t0
    gpu_peak = max(gpu_peak1, gpu_peak2)

    # -- Combine: final Eg = PBE_pred + Delta_pred ---
    with torch.no_grad():
        delta_pred_va = model_delta(X_va_t).cpu().numpy()

    val_mae = float(mean_absolute_error(data["eg_hse_va"],
                                        pbe_pred_va + delta_pred_va))
    print(f"\n  Combined val MAE: {val_mae*1000:.2f} meV")

    # Return a combined model wrapper for evaluation
    class _CombinedModel:
        """Wraps Stage1 + Stage2; .predict(X_t) returns final Eg tensor."""
        def __init__(self, m1, m2):
            self.m1 = m1
            self.m2 = m2

        def predict(self, X_t):
            with torch.no_grad():
                return self.m1(X_t) + self.m2(X_t)

        def eval(self):
            self.m1.eval()
            self.m2.eval()

    combined = _CombinedModel(model_pbe, model_delta)
    return combined, elapsed, gpu_peak, val_mae


# --- evaluation on test set ---

def _per_strain_mae(eg_true, eg_pred, strain_types):
    groups = {"isotropic": [], "uniaxial": [], "biaxial": [],
              "triaxial": [], "shear": []}
    for i, st in enumerate(strain_types):
        s = str(st).lower()
        for g in groups:
            if g in s:
                groups[g].append(abs(eg_pred[i] - eg_true[i]))
                break
    return {g: (float(np.mean(v)) if v else float("nan")) for g, v in groups.items()}


def evaluate_and_write(model_or_combined, data, args, results_dir,
                        device, elapsed, gpu_peak, best_val, model_name):
    if model_or_combined is None:
        row = {c: float("nan") for c in [
            "model_name","seed","dft_pct","n_labeled",
            "test_mae_meV","test_rmse_meV","test_max_err_meV","test_p95_err_meV",
            "test_mape_pct","sign_accuracy_pct",
            "val_mae_meV","train_mae_meV","test_r2","cbm_mae_meV","vbm_mae_meV",
            "mae_isotropic_meV","mae_uniaxial_meV","mae_biaxial_meV",
            "mae_triaxial_meV","mae_shear_meV","total_epochs","elapsed_s","gpu_mem_peak_mb"
        ]}
        row.update({"model_name": model_name, "seed": args.seed,
                    "dft_pct": args.dft_pct, "n_labeled": 0,
                    "total_epochs": 0, "elapsed_s": 0.0})
        pd.DataFrame([row]).to_csv(results_dir / "metrics_summary.csv", index=False)
        return row

    def T(a):
        """Move a numpy array onto the training device as float32."""
        return torch.tensor(a, dtype=torch.float32, device=device)

    model_or_combined.eval()

    # Handle both plain model and combined wrapper
    def _predict(X_np):
        X_t = T(X_np)
        if hasattr(model_or_combined, "predict"):
            return model_or_combined.predict(X_t).cpu().numpy()
        with torch.no_grad():
            return model_or_combined(X_t).cpu().numpy()

    eg_te_p  = _predict(data["X_te"])
    eg_va_p  = _predict(data["X_va"])
    eg_tr_p  = _predict(data["X_tr"])

    eg_true   = data["eg_hse_te"]
    abs_err   = np.abs(eg_te_p - eg_true)
    test_mae  = float(abs_err.mean())
    test_rmse = float(np.sqrt((abs_err**2).mean()))
    test_max  = float(abs_err.max())
    test_p95  = float(np.percentile(abs_err, 95))
    test_mape = float((abs_err / np.abs(eg_true)).mean() * 100.0)
    # Unstrained Eg reference from dataset (row where all strain components ~ 0)
    eg_ref    = float(data["eg_hse_te"].mean())   # fallback; sign acc still meaningful
    sign_acc  = float(np.mean(np.sign(eg_te_p - eg_ref) == np.sign(eg_true - eg_ref)) * 100.0)

    test_r2   = float(r2_score(eg_true, eg_te_p))
    val_mae   = float(mean_absolute_error(data["eg_hse_va"], eg_va_p))
    train_mae = float(mean_absolute_error(data["eg_hse_tr"], eg_tr_p))
    strain_mae = _per_strain_mae(eg_true, eg_te_p, data["stype_te"])

    print(f"\n  Test  MAE: {test_mae*1000:.2f} meV   RMSE: {test_rmse*1000:.2f} meV   R^2={test_r2:.4f}")
    print(f"  max err  : {test_max*1000:.2f} meV   p95: {test_p95*1000:.2f} meV   MAPE: {test_mape:.2f}%")
    print(f"  sign acc : {sign_acc:.1f}%")
    print(f"  Val   MAE: {val_mae*1000:.2f} meV")

    nan_arr = np.full(len(eg_te_p), float("nan"))
    pd.DataFrame({
        "sample_id":       data["idx_te"],
        "strain_type":     data["stype_te"],
        "shear_planes":    data["shear_te"],
        "Eg_true_eV":      eg_true,
        "Eg_pred_eV":      eg_te_p,
        "CBM_true_eV":     data["cbm_te"],
        "CBM_pred_eV":     nan_arr,
        "VBM_true_eV":     data["vbm_te"],
        "VBM_pred_eV":     nan_arr,
        "abs_error_eV":    abs_err,
        "signed_error_eV": eg_te_p - eg_true,
        "seed":            args.seed,
        "dft_pct":         args.dft_pct,
    }).to_csv(results_dir / "test_results.csv", index=False)

    row = {
        "model_name":          model_name,
        "seed":                args.seed,
        "dft_pct":             args.dft_pct,
        "n_labeled":           len(data["dft_idx"]),
        "test_mae_meV":        round(test_mae  * 1000, 2),
        "test_rmse_meV":       round(test_rmse * 1000, 2),
        "test_max_err_meV":    round(test_max  * 1000, 2),
        "test_p95_err_meV":    round(test_p95  * 1000, 2),
        "test_mape_pct":       round(test_mape, 3),
        "sign_accuracy_pct":   round(sign_acc,  2),
        "val_mae_meV":         round(val_mae   * 1000, 2),
        "train_mae_meV":       round(train_mae * 1000, 2),
        "test_r2":             round(test_r2, 4),
        "cbm_mae_meV":         float("nan"),
        "vbm_mae_meV":         float("nan"),
        "mae_isotropic_meV":   round(strain_mae["isotropic"] * 1000, 2),
        "mae_uniaxial_meV":    round(strain_mae["uniaxial"]  * 1000, 2),
        "mae_biaxial_meV":     round(strain_mae["biaxial"]   * 1000, 2),
        "mae_triaxial_meV":    round(strain_mae["triaxial"]  * 1000, 2),
        "mae_shear_meV":       round(strain_mae["shear"]     * 1000, 2),
        "total_epochs":        args.epochs,
        "elapsed_s":           round(elapsed, 1),
        "gpu_mem_peak_mb":     round(gpu_peak, 1),
    }
    pd.DataFrame([row]).to_csv(results_dir / "metrics_summary.csv", index=False)
    print(f"  Saved -> {results_dir}/")
    return row


# --- resource logging ---

class ResourceLogger:
    def __init__(self, path, device):
        self.path   = path
        self.device = device
        self._stop  = threading.Event()
        with open(path, "w") as f:
            f.write("timestamp_s,gpu_mem_mb,cpu_percent,ram_mb\n")

    def _loop(self):
        t0 = time.time()
        while not self._stop.wait(30):
            with open(self.path, "a") as f:
                f.write(f"{time.time()-t0:.1f},{_gpu_mem_mb(self.device):.1f},"
                        f"{_cpu_percent():.1f},{_ram_mb():.1f}\n")

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


# --- readme logger call ---

def main():
    args = parse_args()

    model_name = _model_name_from_mode(args.mode)

    if args.results_dir is None:
        results_dir = HERE / "results" / model_name
    else:
        results_dir = Path(args.results_dir)

    if args.mode == "delta_ml" and args.pbe_csv is None:
        raise ValueError("--pbe_csv is required when --mode delta_ml")

    device = setup_reproducibility(args.seed, args.device, results_dir, args)

    print(f"\nLoading dataset: {args.data_csv}")
    data = load_and_split(
        args.data_csv, args.pbe_csv, args.seed, args.dft_pct, results_dir,
        cv_fold=args.cv_fold, cv_nfolds=args.cv_nfolds
    )
    print(f"  Split: {data['N_tr']} train / "
          f"{len(data['eg_hse_va'])} val / {len(data['eg_hse_te'])} test")
    print(f"  HSE06 labeled (dft_pct={args.dft_pct}%): "
          f"{len(data['dft_idx'])} / {data['N_tr']}")

    res_logger = ResourceLogger(results_dir / "resource_log.csv", device)
    res_logger.start()

    try:
        if args.mode == "delta_ml":
            model_out, elapsed, gpu_peak, best_val = run_delta_ml(
                data, args, results_dir, device
            )
        else:
            model_out, elapsed, gpu_peak, best_val = run_pbe_only(
                data, args, results_dir, device
            )
        row = evaluate_and_write(
            model_out, data, args, results_dir,
            device, elapsed, gpu_peak, best_val, model_name
        )
    finally:
        res_logger.stop()

        print("\n  wrote %s" % args.results_dir)


if __name__ == "__main__":
    main()
