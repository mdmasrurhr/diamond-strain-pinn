"""Test MAE (meV) per family of the four variants against the final model, mean of seeds 42-44 (needs the runs of variant_tasks.py)."""
import os, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = pd.read_csv(ROOT + "/data/dft_hy_v9.csv")
fam = D.strain_type.map(lambda x: "isotropic" if str(x) in ("unstrained", "isotropic") else str(x).split("_")[0]).values
def row(p):
    t = pd.read_csv(p + "/test_results.csv"); e = (t.Eg_pred_eV - t.Eg_true_eV).abs().values * 1000; fm = fam[t.sample_id.values.astype(int)]
    r = {"all": e.mean()}; r.update({k: e[fm == k].mean() for k in ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]}); return r
for pct in (5, 25):
    rows = {v: pd.DataFrame([row("%s/results/%s/seed%d" % (ROOT, ("B_main/hyb_sdp_gap_ld9/%dpct" % pct) if v == "final" else ("B_shear_variants/%s/%dpct" % (v, pct)), s)) for s in (42, 43, 44)]).mean()
            for v in ("final", "head", "sup", "focus", "both")}
    print("\n%d%% labeled, test MAE (meV)" % pct); print(pd.DataFrame(rows).T.round(1).to_string())
