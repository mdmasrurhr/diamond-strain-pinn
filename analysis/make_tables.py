# Copy of the manuscript's v3_numbers.py with DATA_MODE = "summary" (written by sync_code_repository.py; edit the
# manuscript script, not this file). Reads summary/, writes outputs/.
"""Numbers, statistics and tables of manuscript v3 (Communications Materials).

Reads the final LEAP2 campaign (17 seeds, symmetry-grouped division of the data) and writes tables/v3/:
  numbers_v3.json    "txt": every value quoted in the text, already formatted, keyed by the [[key]] placeholders of
                     v3_text.py; "raw": the underlying values; "checks": claims the text makes, each True or False;
                     "tests": every statistical test quoted in the text (also printed in Supplementary Table 8)
  runs_main.csv, runs_ablation.csv   one row per run with every metric (read by v3_figures.py)
  table1_dataset.tex, table2_metrics.tex, table3a_ablation.tex, table3b_ablation.tex, table4_constants.tex
  si_*.tex           Supplementary tables
Run: python v3_numbers.py"""
import glob, json, math, os, re, sys
import numpy as np, pandas as pd, torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as PD
# Data source: "raw" reads the campaign's run folders (the manuscript build); "summary" reads summary/ of the code
# repository (analysis/make_tables.py there is this file with DATA_MODE = "summary").
DATA_MODE = "summary"
if DATA_MODE == "raw":
    PD.configure("raw", master=os.path.abspath(os.path.join(HERE, "..", "..")))          # DiamondPINN_MASTER
    OUT = os.path.join(HERE, "..", "tables", "v3")
else:
    PD.configure("summary", repo=os.path.abspath(os.path.join(HERE, "..")))               # the code repository
    OUT = os.path.join(HERE, "..", "outputs", "tables")
os.makedirs(OUT, exist_ok=True)
import physics as P
import metrics_report as MR

# FINAL = PHM (results key "phm2", since 9 Nov 2026); the key "phm" in this folder is PHM old and is not used.
FINAL, RES, SA, RBA, MLP, SHI, GP = "phm2", "res_symlab_bal", "sa_pinn", "rba_pinn", "mlp", "shi", "GaussianProcess"
NTK, APH = "ntk_pinn", "aphynity"
LV = [0, 1, 2, 5, 10, 25, 50, 75, 100]; NL = dict(zip(LV, [0, 9, 18, 45, 90, 224, 448, 672, 896])); SEEDS = list(range(42, 59))
PCT = {p: "%d" % p for p in LV}   # labeled fraction of the training portion, in percent: the label used in every table
FAMS = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]; SC = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
NAME = {FINAL: "PHM", RES: "Residual model", SA: "SA-PINN", RBA: "RBA-PINN", NTK: "NTK-PINN", APH: "APHYNITY", MLP: "MLP", SHI: "Shi et al.~\\cite{shi2019deep}",
        GP: "Gaussian process", "formula": "Analytic models alone"}
DATA = pd.read_csv(PD.data_csv())
TXT, RAW, CHECKS, TESTS = {}, {}, {}, []


# ------------------------------------------------------------------ formatting
def num(x, dec=1):
    """A value for the text: thousands separators above 999, otherwise `dec` decimals."""
    if abs(x) >= 1000: return "{:,.0f}".format(x).replace(",", "{,}")
    if abs(x) >= 100 and dec <= 1: return "%.0f" % x
    return ("%." + str(dec) + "f") % x


def pval(p):
    if p < 0.001: return "$P < 0.001$"
    s = "%.2g" % p
    if "e" in s: s = "%.4f" % p
    return "$P = %s$" % s


def pmax(ps, adjective="\\leq"):
    """Largest P value of a set, as quoted for a claim that holds for every member."""
    m = max(ps)
    return "$P < 0.001$" if m < 0.001 else "$P %s %.2g$" % (adjective, m)


def ci(lo, hi, dec, unit):
    u = ("~" + unit) if unit else ""
    return "95\\%% CI $%s$ to $%s$%s" % (("%." + str(dec) + "f") % lo, ("%." + str(dec) + "f") % hi, u)


def mean_sd(m, sd):
    """Mean and standard deviation at the same precision: integers from 100 meV upwards, one decimal below."""
    return ("$%s \\pm %.0f$" % (num(m), sd)) if abs(m) >= 100 else ("$%.1f \\pm %.1f$" % (m, sd))


def put(key, value):
    assert key not in TXT, "duplicate text key " + key
    TXT[key] = value


def check(key, ok):
    CHECKS[key] = bool(ok)


# ------------------------------------------------------------------ statistics
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); out[i] = run
    return out


def compare(a, b, delta=None):
    """Paired comparison over seeds of series b against series a; the difference is b - a."""
    i = a.index.intersection(b.index); x, y = a[i].astype(float), b[i].astype(float); dl = (y - x).values; n = len(dl)
    r = dict(n=n, mean_a=x.mean(), sd_a=x.std(ddof=1), mean_b=y.mean(), sd_b=y.std(ddof=1), diff=dl.mean(), a_lower=int((x < y).sum()))
    if np.allclose(dl, 0):
        r.update(t=0.0, df=n - 1, p=1.0, d=0.0, lo=0.0, hi=0.0, p_w=1.0, p_sw=1.0); return r
    t = stats.ttest_rel(y, x); se = dl.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    r.update(t=float(t.statistic), df=n - 1, p=float(t.pvalue), d=float(dl.mean() / dl.std(ddof=1)), lo=dl.mean() - q * se, hi=dl.mean() + q * se,
             p_w=float(stats.wilcoxon(y, x).pvalue), p_sw=float(stats.shapiro(dl).pvalue))
    if delta is not None:
        r["p_tost"] = float(max(stats.t.sf((dl.mean() + delta) / se, n - 1), stats.t.cdf((dl.mean() - delta) / se, n - 1)))
    return r


def family(results):
    """Holm-adjust a family of comparisons in place (t-test and Wilcoxon P values separately)."""
    for r, h, hw in zip(results, holm([r["p"] for r in results]), holm([r["p_w"] for r in results])):
        r["p_adj"], r["p_w_adj"] = float(h), float(hw)
    return results


def statstr(r, unit="meV", dec=1, sign=1):
    """Full report of a paired test: t(df), P (Holm-adjusted where a family applies), Cohen's d, 95% CI of the difference.
    sign=-1 reports the difference the other way round, so that the CI is positive when the sentence's claim holds."""
    lo, hi = sorted([sign * r["lo"], sign * r["hi"]])
    return "$t(%d) = %.2f$, %s, $d = %.2f$, %s" % (r["df"], abs(r["t"]), pval(r.get("p_adj", r["p"])), abs(r["d"]), ci(lo, hi, dec, unit))


def register(key, what, r, unit, dec=1, sign=1):
    """Quote a test in the text and list it in Supplementary Table 8."""
    p = r.get("p_adj", r["p"]); pw = r.get("p_w_adj", r["p_w"])
    TESTS.append(dict(key=key, what=what, n=r["n"], t=abs(r["t"]), df=r["df"], p=p, d=abs(r["d"]), lo=min(sign * r["lo"], sign * r["hi"]),
                      hi=max(sign * r["lo"], sign * r["hi"]), unit=unit, p_w=pw, p_sw=r["p_sw"], agree=bool((p < 0.05) == (pw < 0.05))))
    put(key, statstr(r, unit, dec, sign))
    put(key + "_p", pval(p))


# ------------------------------------------------------------------ 1. reference values from the isotropic series
iso = DATA[DATA.strain_type == "isotropic"]; win = iso.E_xx.abs() <= 0.02
coef = np.polyfit(iso.E_xx[win], iso.Eg_eV[win], 3)
GAP0, DP_REF = float(coef[-1]), float(coef[-2] / 3.0)     # gap at zero strain (eV) and dE_g/dI_1 (eV)
MR.EG0 = GAP0                                             # sign accuracy is measured against this gap
RAW["iso_reference"] = dict(gap0=GAP0, dp=DP_REF, n=int(win.sum()))
put("gap0", "%.2f" % GAP0); put("dp_ref", "$%.2f$" % DP_REF); put("dp_ref_m", "%.2f" % DP_REF); put("n_iso_fit", "%d" % win.sum())
put("n_states", "1{,}292"); check("n_states", len(DATA) == 1292)
_i1 = DATA[["E_xx", "E_yy", "E_zz"]].sum(axis=1).abs() < 0.01                 # states the zero-strain slope terms act on
put("n_nz_states", "%d" % _i1.sum()); put("n_nz_shear", "%d" % (_i1 & (DATA.strain_type == "shear")).sum())


# band edges on the k-point grid (Supplementary Note 6)
fams_all = DATA.strain_type.map(lambda x: "isotropic" if str(x).lower() in ("unstrained", "isotropic") else str(x).lower().split("_")[0])
put("n_vbm_kpts", "%d" % DATA.vbm_kpt_index.nunique()); check("vbm_at_gamma", DATA.vbm_kpt_index.nunique() == 1 and int(DATA.vbm_kpt_index.iloc[0]) == 0)
put("n_cbm_kpts", "%d" % DATA.cbm_kpt_index.nunique())
for f in ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]:
    put("n_cbm_kpts_%s" % f, "%d" % DATA[fams_all == f].cbm_kpt_index.nunique())
put("n_cbm_at_gamma", "%d" % (DATA.cbm_kpt_index == 0).sum())
top = DATA.cbm_kpt_index.value_counts(); put("cbm_top3_share", "%.0f" % (100 * top.iloc[:3].sum() / len(DATA)))
RAW["cbm_kpoints"] = {int(k): int(v) for k, v in top.items()}

# ------------------------------------------------------------------ 2. per-run metrics
def fam_of(s):
    s = str(s).lower(); return "isotropic" if s in ("unstrained", "isotropic") else s.split("_")[0]


def load(kind):
    """Per-run metrics of the paper's runs (kind: main, ablation, labelmatched), with sign accuracy in percent."""
    out = PD.runs(kind, MR.metrics)
    out["sign"] = 100 * out.sign_acc
    return out


def analytic_bands(ids, const_path):
    for k, v in json.load(open(const_path)).items(): setattr(P, k, float(v))
    x = torch.tensor(DATA.loc[ids, SC].values, dtype=torch.float64)
    return P.physics_cbm(x).numpy(), P.physics_vbm(x).numpy()


M = load("main")
# ablations: one term of PHM removed (phm2_drop_*), or a term PHM does not use added back. The phm_new runs are PHM plus
# the symmetry term and the band-edge level anchor, so phm_new_drop_anchor = PHM + symmetry term and
# phm_new_drop_sym = PHM + level anchor.
ABL_FOLDER = {"drop_agreement": FINAL + "_drop_agreement", "drop_iso_eg": FINAL + "_drop_iso_eg", "drop_copies": FINAL + "_drop_copies",
              "add_sym": "phm_new_drop_anchor", "add_anchor": "phm_new_drop_sym"}
A = load("ablation")
A["model"] = A.model.map({f: t for t, f in ABL_FOLDER.items()})
NLAB = M[M.model == MLP].groupby("level").n_labeled.agg(["min", "max"])
RAW["n_labeled_range"] = {int(k): [int(v["min"]), int(v["max"])] for k, v in NLAB.iterrows()}
for p in (10, 25, 50, 75, 100): put("nlab_%d" % p, ("%d" % NLAB.loc[p, "min"]) if NLAB.loc[p, "min"] == NLAB.loc[p, "max"] else "%d--%d" % (NLAB.loc[p, "min"], NLAB.loc[p, "max"]))
check("label_counts_nominal", all(abs(NLAB.loc[p, "min"] - NL[p]) <= 4 and abs(NLAB.loc[p, "max"] - NL[p]) <= 4 for p in LV))
DG = PD.diagnostics()
DG["dp"] = DG.slope_hydrostatic / 3.0; DG["dp_err"] = (DG.dp - DP_REF).abs(); DG["shear_slope"] = DG.slope_shear.abs()
DG["dp_an"] = DG.ref_slope_hydrostatic / 3.0
DG = DG.rename(columns={"axis_relabel_drift_meV": "axis", "shear_sign_drift_meV": "shearsign", "noise_rel_change": "noise"})
DGC = ["run", "dp", "dp_err", "shear_slope", "axis", "shearsign", "noise", "dp_an"]
M = M.merge(DG[DGC], on="run", how="left"); A = A.merge(DG[DGC], on="run", how="left")
check("ablation_diagnostics_complete", A.dp.notna().all())

form = []
for s in SEEDS:
    t = PD.test_frame(s)
    cb, vb = analytic_bands(t.sample_id.values, PD.constants_file("2band", s))
    f = t.copy(); f["CBM_pred_eV"], f["VBM_pred_eV"], f["Eg_pred_eV"] = cb, vb, cb - vb
    r = MR.metrics(f); r.update(model="formula", seed=s, sign=100 * r["sign_acc"]); form.append(r)
FORM = pd.DataFrame(form)
form_lin = []
for s in SEEDS:
    t_ = PD.test_frame(s)
    P.DELTA_X = 0.0
    cb, vb = analytic_bands(t_.sample_id.values, PD.constants_file("linear", s))
    f_ = t_.copy(); f_["CBM_pred_eV"], f_["VBM_pred_eV"], f_["Eg_pred_eV"] = cb, vb, cb - vb
    r_ = MR.metrics(f_); r_.update(seed=s); form_lin.append(r_)
FORM_LIN = pd.DataFrame(form_lin)
put("mae_formula_lin", num(FORM_LIN.MAE_meV.mean())); put("mae_formula_lin_shear", num(FORM_LIN.MAE_shear_meV.mean())); put("mae_formula_shear", num(FORM.MAE_shear_meV.mean()))
r_ = compare(FORM.set_index("seed").MAE_meV, FORM_LIN.set_index("seed").MAE_meV); register("stat_formula_lin", "MAE, analytic models with the linear vs the two-band shear term", r_, "meV")
check("two_band_formula_better", r_["diff"] > 0 and r_["p"] < 0.05)
DP_AN = float(DG[DG.run.str.startswith("B_main/%s/" % FINAL)].dp_an.mean())
FORM["dp_err"] = abs(DP_AN - DP_REF); FORM["shear_slope"] = 0.0; FORM["axis"] = 0.0; FORM["shearsign"] = 0.0
RAW["dp_analytic"] = DP_AN; put("dp_an", "$%.2f$" % DP_AN)
CL = PD.classical().rename(columns={"dft_pct": "level", "test_mae_meV": "MAE_meV", "test_r2": "R2"})
M.to_csv(os.path.join(OUT, "runs_main.csv"), index=False); A.to_csv(os.path.join(OUT, "runs_ablation.csv"), index=False)
FORM.to_csv(os.path.join(OUT, "runs_analytic.csv"), index=False)


def series(model, level, col="MAE_meV", frame=None):
    f = M if frame is None else frame
    return f[(f.model == model) & (f.level == level)].set_index("seed")[col]


def mean(model, level, col="MAE_meV"):
    return float(series(model, level, col).mean())


print("main runs %d, ablation runs %d, classical runs %d" % (len(M), len(A), len(CL)))

# ------------------------------------------------------------------ 3. Table 1: dataset and its division
sys.argv = ["03_train_pinn.py", "--data_csv", PD.data_csv(), "--results_dir", os.path.join(PD.code_root(), "results", "_v3_stub"),
            "--device", "cpu", "--split", "symmetry"]
import importlib.util
spec = importlib.util.spec_from_file_location("trn", os.path.join(PD.code_root(), "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
fam_all = DATA.strain_type.map(fam_of).values; cnt = []
for s in SEEDS:
    d = T.load_and_split(PD.data_csv(), s, 100)
    te = np.array(d["idx_te"]); trf = np.array([fam_of(x) for x in d["stype_tr"]]); st = d["strain_tr"]
    iso_tr = int(((np.abs(st[:, 3:]).sum(1) < 1e-4) & (np.abs(st[:, :3] - st[:, :3].mean(1, keepdims=True)).max(1) < 1e-4)).sum())
    for f in FAMS + ["total"]:
        ntr = (trf == f).sum() if f != "total" else len(trf); nte = (fam_all[te] == f).sum() if f != "total" else len(te)
        ntot = (fam_all == f).sum() if f != "total" else len(DATA)
        cnt.append(dict(seed=s, family=f, train=ntr, test=nte, val=ntot - ntr - nte, iso_train=iso_tr))
stub = os.path.join(PD.code_root(), "results", "_v3_stub")
if os.path.isdir(stub) and not os.listdir(stub): os.rmdir(stub)
CNT = pd.DataFrame(cnt); tot = CNT[CNT.family == "total"]
rng_ = lambda c: ("%d" % c.min()) if c.min() == c.max() else ("%d--%d" % (c.min(), c.max()))
put("split_train", rng_(tot.train)); put("split_val", rng_(tot.val)); put("split_test", rng_(tot.test)); put("iso_train", rng_(CNT.iso_train))
DESC = {"isotropic": "equal along the three axes", "uniaxial": "along one axis", "biaxial": "along two axes",
        "triaxial": "independent along three axes", "shear": "in one, two or three planes"}
lines = ["\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{2.6cm}rrrrcc}", "\\toprule",
         "\\textbf{Family} & \\textbf{Strain applied} & \\textbf{States} & \\textbf{Training} & \\textbf{Validation} & \\textbf{Test} & "
         "\\textbf{Strain (\\%)} & \\textbf{Gap (eV)} \\\\", "\\midrule"]
for f in FAMS + ["total"]:
    c = CNT[CNT.family == f]; m_ = (fam_all == f) if f != "total" else np.ones(len(DATA), bool)
    sv = DATA.loc[m_, SC].values * 100; ev = DATA.Eg_eV.values[m_]
    srange = "$%+.1f$ to $%+.1f$" % (sv.min(), sv.max()) if f != "total" else ""
    if f == "total": lines.append("\\midrule")
    lines.append("%s & %s & %s & %s & %s & %s & %s & %.2f--%.2f \\\\" % (f.capitalize() if f != "total" else "Total", DESC.get(f, ""),
                 "{:,}".format(int(m_.sum())).replace(",", "{,}"),
                 rng_(c.train), rng_(c.val), rng_(c.test), srange, ev.min(), ev.max()))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table1_dataset.tex"), "w").write("\n".join(lines) + "\n")
for f in FAMS: put("n_%s" % f, "%d" % (fam_all == f).sum())

# ------------------------------------------------------------------ 4. key metrics (Table 2) and comparisons
KM = ["MAE_meV", "RMSE_meV", "P95_meV", "max_meV", "R2", "sign", "dp_err", "shear_slope", "axis", "shearsign"]
UNIT = {"MAE_meV": "meV", "RMSE_meV": "meV", "P95_meV": "meV", "max_meV": "meV", "R2": "", "sign": "percentage points", "dp_err": "eV",
        "shear_slope": "eV", "axis": "meV", "shearsign": "meV"}
DEC = {"MAE_meV": 1, "RMSE_meV": 1, "P95_meV": 1, "max_meV": 0, "R2": 3, "sign": 1, "dp_err": 2, "shear_slope": 2, "axis": 1, "shearsign": 1}
COMP = {}   # COMP[(comparator, metric)] = [result at 45, result at 896], each the comparator minus the PHM
COMPARATORS = [m for m in (RES, SA, RBA, NTK, APH, MLP) if (M.model == m).any()]
for comp in COMPARATORS:
    for met in KM:
        COMP[(comp, met)] = family([compare(series(FINAL, lv, met), series(comp, lv, met), delta=3.0 if met == "MAE_meV" else None) for lv in (5, 100)])
sw = np.array([r["p_sw"] for k, v in COMP.items() for r in v])
put("n_sw_reject", "%d" % (sw < 0.05).sum()); put("n_sw_total", "%d" % len(sw))

rows = ["\\begin{tabular}{lrrrrrrrrrrr}", "\\toprule",
        " & \\multicolumn{8}{c}{\\textbf{Accuracy}} & \\multicolumn{3}{c}{\\textbf{Physical validity}} \\\\",
        "\\cmidrule(lr){2-9}\\cmidrule(l){10-12}",
        "\\textbf{Model} & \\textbf{MAE} & \\textbf{MAPE} & \\textbf{Within 10 meV} & \\textbf{RMSE} & \\textbf{P95} & \\textbf{Max} & \\textbf{$R^2$} & \\textbf{Sign} & \\textbf{DP error} & "
        "\\textbf{Shear slope} & \\textbf{Axis drift} \\\\",
        " & (meV) & (\\%) & (\\%) & (meV) & (meV) & (meV) & & (\\%) & (eV) & (eV) & (meV) \\\\", "\\midrule"]
def t2row(label, g, phys=True):
    c = ["%.1f $\\pm$ %.1f" % (g.MAE_meV.mean(), g.MAE_meV.std(ddof=1)), "%.2f" % (100 * g.MAPE.mean()), "%.0f" % (100 * g.within_10meV.mean())]
    for col, fmt in [("RMSE_meV", "%.1f"), ("P95_meV", "%.1f"), ("max_meV", "%.0f"), ("R2", "%.3f"), ("sign", "%.1f")]:
        c.append("--" if col not in g or g[col].isna().all() else fmt % g[col].mean())
    for col, fmt in [("dp_err", "%.3f"), ("shear_slope", "%.2f"), ("axis", "%.1f")]:
        c.append(fmt % g[col].mean() if phys and col in g and g[col].notna().all() else "--")
    return "%s & %s \\\\" % (label, " & ".join(c))
rows.append(t2row(NAME["formula"], FORM)); rows.append("\\addlinespace")
for lv in (5, 100):
    rows.append("\\multicolumn{12}{l}{\\textit{%s}} \\\\" % ("5\\% of the training states labeled (45 calculations)" if lv == 5 else "100\\%% of the training states labeled (%s calculations)" % rng_(NLAB.loc[100, ["min", "max"]].astype(int))))
    for m in [FINAL] + COMPARATORS:
        rows.append(t2row(NAME[m], M[(M.model == m) & (M.level == lv)]))
    g = CL[(CL.model == GP) & (CL.level == lv)]
    rows.append("%s & %.1f $\\pm$ %.1f & -- & -- & -- & -- & -- & %.3f & -- & -- & -- & -- \\\\" % (NAME[GP], g.MAE_meV.mean(), g.MAE_meV.std(ddof=1), g.R2.mean()))
    if lv == 5: rows.append("\\addlinespace")
rows += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table2_metrics.tex"), "w").write("\n".join(rows) + "\n")

# values quoted in the text
for m, key in [(FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (NTK, "ntk"), (RES, "res"), (APH, "aph"), (MLP, "mlp")]:
    if not (M.model == m).any(): continue
    for lv in LV:
        g = series(m, lv)
        if len(g): put("mae_%s_%d" % (key, NL[lv]), num(g.mean())); put("mae_%s_%d_sd" % (key, NL[lv]), mean_sd(g.mean(), g.std(ddof=1)))
    for lv in (5, 100):
        for met in ["RMSE_meV", "P95_meV", "max_meV", "R2", "sign", "dp_err", "shear_slope", "axis", "shearsign"]:
            v = series(m, lv, met)
            if v.notna().all() and len(v): put("%s_%s_%d" % (met.replace("_meV", "").lower(), key, NL[lv]), ("%." + str(DEC[met]) + "f") % v.mean())
for lv in LV:
    g = CL[(CL.model == GP) & (CL.level == lv)].MAE_meV
    if len(g): put("mae_gp_%d" % NL[lv], num(g.mean()))
put("mae_formula", num(FORM.MAE_meV.mean())); put("mae_formula_sd", mean_sd(FORM.MAE_meV.mean(), FORM.MAE_meV.std(ddof=1)))
put("mae_ours_896_pct", "%.2f" % (100 * mean(FINAL, 100) / (1000 * GAP0)))   # relative to the computed unstrained gap
# percentage metrics (error relative to the reference gap of each state) and the share of states within 10 meV
for m, key in [(FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (NTK, "ntk"), (RES, "res"), (APH, "aph"), (MLP, "mlp")]:
    if not (M.model == m).any(): continue
    for lv in (5, 50, 100):
        put("mape_%s_%d" % (key, NL[lv]), "%.2f" % (100 * mean(m, lv, "MAPE"))); put("within10_%s_%d" % (key, NL[lv]), "%.0f" % (100 * mean(m, lv, "within_10meV")))
put("mape_formula", "%.2f" % (100 * FORM.MAPE.mean()))
put("gap_lo_896", "%.3f" % (GAP0 - mean(FINAL, 100) / 1000)); put("gap_hi_896", "%.3f" % (GAP0 + mean(FINAL, 100) / 1000))
put("gap0_3", "%.3f" % GAP0); put("tol10_pct", "%.2f" % (100 * 0.010 / GAP0))
r = compare(series(FINAL, 50), series(MLP, 100)); register("stat_half_labels", "MAE of the MLP with all training states labeled minus that of the PHM with half labeled", r, "meV")
TXT["stat_half_labels"] = TXT["stat_half_labels"].replace("95\\% CI", "95\\% confidence interval")   # first use in the text
check("half_labels_better", r["diff"] > 0 and r["p"] < 0.05)
put("rmse_ratio_ours_896", "%.1f" % (mean(FINAL, 100, "RMSE_meV") / mean(FINAL, 100)))
dd = max(mean(m, lv, "dp_err") for m in (FINAL, RES, MLP) for lv in (5, 100)); put("dp_err_datadriven_max", "%.2f" % dd)
for m, key in [(FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (RES, "res"), (MLP, "mlp")]:
    put("dp_%s_896" % key, "$%.2f$" % mean(m, 100, "dp"))
check("dp_datadriven_close", dd < 0.1); check("dp_pinn_far", min(mean(m, lv, "dp_err") for m in (SA, RBA) for lv in (5, 100)) > 0.3)
check("dp_pinns_near_analytic", all(abs(mean(m, 100, "dp") - DP_AN) < abs(mean(m, 100, "dp") - DP_REF) for m in (SA, RBA)))

# accuracy at 896
r = COMP[(MLP, "MAE_meV")][1]; register("stat_mlp_896", "MAE, MLP vs the PHM, 100% labeled", r, "meV")
put("diff_mlp_896", "%.1f" % r["diff"])
pen = [mean(m, 100) - mean(MLP, 100) for m in (SA, RBA)]; put("pinn_penalty_896_range", "%s to %s" % (num(min(pen)), num(max(pen))))
for m in (SA, RBA):
    rr = COMP[(m, "MAE_meV")][1]; register("stat_%s_896" % ("sa" if m == SA else "rba"), "MAE, %s vs the PHM, 100%% labeled" % NAME[m], rr, "meV")
r = COMP[(RES, "MAE_meV")][1]; put("tost_res_896", "two one-sided tests, %s" % pval(r["p_tost"]).replace("P", "P_{\\mathrm{TOST}}"))
check("res_equivalent_896", r["p_tost"] < 0.05)
r = COMP[(MLP, "P95_meV")][1]; register("stat_p95_mlp_896", "95th-percentile error, MLP vs the PHM, 100% labeled", r, "meV")
r = COMP[(MLP, "max_meV")][1]; register("stat_max_mlp_896", "maximum error, MLP vs the PHM, 100% labeled", r, "meV", 0)
check("max_mlp_896_ns", r["p_adj"] >= 0.05)
check("r2_networks_896", all(mean(m, 100, "R2") > 0.99 for m in (FINAL, RES, SA, RBA, MLP)))
# families at 896, the PHM vs MLP, Holm over the five families
fr = family([compare(series(FINAL, 100, "MAE_%s_meV" % f), series(MLP, 100, "MAE_%s_meV" % f)) for f in FAMS])
FR = {f: r for f, r in zip(FAMS, fr) if f != "shear"}                        # diff = MLP minus the PHM
better = [f for f, r in FR.items() if r["p_adj"] < 0.05 and r["diff"] < 0]       # families where the MLP is more accurate
equal = [f for f, r in FR.items() if r["p_adj"] >= 0.05]
join = lambda xs: " and ".join(xs)
ours_b = [f for f, r in FR.items() if r["p_adj"] < 0.05 and r["diff"] > 0]       # families where the PHM is more accurate
RAW["families_896"] = {f: dict(ours=mean(FINAL, 100, "MAE_%s_meV" % f), mlp=mean(MLP, 100, "MAE_%s_meV" % f), p_adj=FR[f]["p_adj"]) for f in FR}
def _vals(fs): return " and ".join("%.1f against %.1f" % (mean(FINAL, 100, "MAE_%s_meV" % f), mean(MLP, 100, "MAE_%s_meV" % f)) for f in fs) + "~meV"
_parts = []
if ours_b: _parts.append("The PHM is also more accurate on %s states (%s; %s)" % (join(ours_b), _vals(ours_b), pmax([FR[f]["p_adj"] for f in ours_b])))
if better: _parts.append("the MLP is more accurate on %s states (%s; %s)" % (join(better), _vals(better), pmax([FR[f]["p_adj"] for f in better])))
if equal: _parts.append("on %s states we found no credible evidence of a difference ($P \\geq %.2f$)" % (join(equal), min(FR[f]["p_adj"] for f in equal)))
_s = "; ".join(_parts) + "."; put("fam_nonshear_text", _s[0].upper() + _s[1:])
register("stat_shear_896", "shear-family MAE, MLP vs the PHM, 100% labeled", fr[FAMS.index("shear")], "meV")
put("mae_shear_ours_896", "%.1f" % mean(FINAL, 100, "MAE_shear_meV")); put("mae_shear_mlp_896", "%.1f" % mean(MLP, 100, "MAE_shear_meV"))
check("shear_hardest_ours_896", max(FAMS, key=lambda f: mean(FINAL, 100, "MAE_%s_meV" % f)) == "shear")
put("fam_max_nonshear_ours_896", "%.1f" % max(mean(FINAL, 100, "MAE_%s_meV" % f) for f in FAMS if f != "shear"))
# physical validity
ps = [COMP[(c, "shear_slope")][1]["p_adj"] for c in (RBA, MLP)]
put("shear_slope_vs_pmax", pmax(ps)); put("shear_slope_eq_p", "$P = %.2f$" % COMP[(SA, "shear_slope")][1]["p_adj"])
check("shear_slope_split", all(COMP[(c, "shear_slope")][1]["diff"] > 0 for c in (RBA, MLP)) and max(ps) < 0.05 and COMP[(SA, "shear_slope")][1]["p_adj"] >= 0.05)
ps = [COMP[(c, "axis")][i]["p_adj"] for c in (SA, RBA, MLP) for i in (0, 1)]
put("axis_pmax", pmax(ps)); check("axis_lowest_both", all(COMP[(c, "axis")][i]["diff"] > 0 for c in (SA, RBA, MLP) for i in (0, 1)) and max(ps) < 0.05)
r = COMP[(SA, "shearsign")][0]; register("stat_shearsign_sa_45", "shear-sign drift, SA-PINN vs the PHM, 5% labeled", r, "meV", 1)
ps = [COMP[(c, "shearsign")][i]["p_adj"] for c in (SA, RBA, MLP) for i in (0, 1)]; put("shearsign_pmax", pmax(ps))
check("shearsign_lowest_both", all(COMP[(c, "shearsign")][i]["diff"] > 0 for c in (SA, RBA, MLP) for i in (0, 1)) and max(ps) < 0.05)
check("shearsign_lowest_896", all(COMP[(c, "shearsign")][1]["diff"] > 0 and COMP[(c, "shearsign")][1]["p_adj"] < 0.05 for c in (SA, RBA, MLP)))
ps = [COMP[(c, "dp_err")][i]["p_adj"] for c in (SA, RBA) for i in (0, 1)]; put("dp_pinn_pmax", pmax(ps)); check("dp_pinn_sig", max(ps) < 0.05)

# PHM against the residual model at every labeled fraction (residual minus PHM; Holm over the nine fractions)
VR = dict(zip(LV, family([compare(series(FINAL, lv), series(RES, lv)) for lv in LV])))
RAW["res_vs_phm"] = {NL[lv]: dict(diff=r["diff"], p_adj=r["p_adj"]) for lv, r in VR.items()}
_rb = [lv for lv in LV if VR[lv]["p_adj"] < 0.05 and VR[lv]["diff"] > 0]     # PHM more accurate
_rw = [lv for lv in LV if VR[lv]["p_adj"] < 0.05 and VR[lv]["diff"] < 0]     # residual model more accurate
put("res_phm_better_levels", " and ".join("%d" % lv for lv in _rb) + "\\%"); put("res_phm_worse_levels", " and ".join("%d" % lv for lv in _rw) + "\\%")
for lv in (0, 1, 2, 10, 100): put("res_diff_%d" % NL[lv], "%.1f" % abs(VR[lv]["diff"]))
register("stat_res_0", "MAE, residual model vs the PHM, 0% labeled", VR[0], "meV")
register("stat_res_9", "MAE, residual model vs the PHM, 1% labeled", VR[1], "meV")
register("stat_res_90", "MAE, residual model vs the PHM, 10% labeled", VR[10], "meV", 1, sign=-1)
put("res_ns_pmin", "$P \\geq %.2f$" % min(VR[lv]["p_adj"] for lv in LV if lv not in _rb + _rw))
register("stat_res_18", "MAE, residual model vs the PHM, 2% labeled", VR[2], "meV")
put("res_phm_better_levels_range", "%d--%d\\%%" % (min(_rb), max(_rb)))
check("res_pattern", _rb == [0, 1, 2] and _rw == [10])
# symmetry drift: PHM and the residual model against the other models
_others = (SA, RBA, NTK, APH, MLP)
_ps = [COMP[(c, k)][i]["p_adj"] for c in _others for k in ("axis", "shearsign") for i in (0, 1)]
put("sym_others_pmax", pmax(_ps)); check("sym_phm_below_others", all(COMP[(c, k)][i]["diff"] > 0 for c in _others for k in ("axis", "shearsign") for i in (0, 1)) and max(_ps) < 0.05)
r = COMP[(RES, "axis")][0]; register("stat_axis_res_45", "axis-relabelling drift, residual model vs the PHM, 5% labeled", r, "meV", 1, sign=-1)
put("axis_res_p_896", pval(COMP[(RES, "axis")][1]["p_adj"]))
for lv in (5, 100):
    _v = [mean(c, lv, "axis") for c in _others]; put("axis_others_%d_range" % NL[lv], "%.1f to %.0f" % (min(_v), max(_v)))
check("axis_res_lower_45", r["diff"] < 0 and r["p_adj"] < 0.05)

# ------------------------------------------------------------------ 5. scarce data
r = COMP[(MLP, "MAE_meV")][0]; register("stat_mlp_45", "MAE, MLP vs the PHM, 5% labeled", r, "meV")
ratio45 = mean(MLP, 5) / mean(FINAL, 5); put("ratio_mlp_45", "%.1f" % ratio45); put("ratio_mlp_45_int", "%.0f" % ratio45); check("ratio_mlp_45_int", "%.0f" % ratio45 == TXT["ratio_mlp_45_int"])
r = COMP[(MLP, "sign")][0]; register("stat_sign_mlp_45", "sign accuracy, the PHM vs MLP, 5% labeled", r, "percentage points", 1, sign=-1)
check("sign_ours_above_mlp_45", r["diff"] < 0 and r["p_adj"] < 0.05)
check("sign_physics_models_high", all(mean(m, lv, "sign") > 99 for m in (FINAL, RES, SA, RBA) for lv in (5, 100)))
x0 = series(FINAL, 0); y0 = FORM.set_index("seed").MAE_meV
r = compare(x0, y0); r["p_adj"], r["p_w_adj"] = r["p"], r["p_w"]
register("stat_formula_0", "MAE, analytic models vs the PHM, 0% labeled", r, "meV")
put("ours_beats_formula_0", "%d" % r["a_lower"]); check("formula_0_ns", r["p"] >= 0.05)

# ------------------------------------------------------------------ 6. mechanism
vsm = {m: family([compare(series(m, lv), series(MLP, lv)) for lv in LV]) for m in (SA, RBA, NTK, FINAL)}   # no physics minus model
ratios = [mean(MLP, lv) / mean(m, lv) for m in (SA, RBA, NTK) for lv in (0, 1, 2, 5, 10)]
put("pinn_ratio_range", "%.1f to %.0f" % (min(ratios), max(ratios))); check("pinns_better_low", all(vsm[m][LV.index(lv)]["diff"] > 0 and vsm[m][LV.index(lv)]["p_adj"] < 0.05 for m in (SA, RBA, NTK) for lv in (0, 1, 2, 5, 10)))
for m, key in [(SA, "sa"), (RBA, "rba"), (NTK, "ntk")]:
    worse = [lv for lv, r in zip(LV, vsm[m]) if r["diff"] < 0 and r["p_adj"] < 0.05]
    first = min(worse); check("%s_worse_from_first" % key, worse == [lv for lv in LV if lv >= first])
    put("%s_cross" % key, "%d" % NL[first]); put("%s_cross_pct" % key, "%d" % first); put("mae_%s_cross" % key, num(mean(m, first))); put("mae_mlp_%s_cross" % key, num(mean(MLP, first)))
    register("stat_%s_cross" % key, "MAE, %s vs MLP, %s%% labeled" % (NAME[m], PCT[first]), vsm[m][LV.index(first)], "meV", 1, sign=-1)
put("ours_vs_mlp_pmax", pmax([r["p_adj"] for r in vsm[FINAL]])); check("ours_better_every_volume", all(r["diff"] > 0 and r["p_adj"] < 0.05 for r in vsm[FINAL]))
# values for the summarized Results text
_m3 = DATA[SC].abs().max(axis=1) <= 0.03 + 1e-9
put("gap_shift_03", "%.1f" % (GAP0 - DATA.Eg_eV[_m3].min())); check("gap_shift_03_above_1eV", GAP0 - DATA.Eg_eV[_m3].min() > 1.0)
put("fam_nonshear_pmin", "$P \\geq %.2f$" % min(r["p_adj"] for r in FR.values())); RAW["nonshear_equal"] = equal
_ro = [mean(m, 5) / mean(FINAL, 5) for m in (SA, RBA, NTK, APH)]; put("ratio_others_45_range", "%.1f to %.1f" % (min(_ro), max(_ro)))
put("others_45_pmax", pmax([COMP[(m, "MAE_meV")][0]["p_adj"] for m in (SA, RBA, NTK, APH)]))
check("ours_best_45_except_res", all(COMP[(m, "MAE_meV")][0]["diff"] > 0 and COMP[(m, "MAE_meV")][0]["p_adj"] < 0.05 for m in (SA, RBA, NTK, APH)))
put("ratio_aph_45", "%.1f" % (mean(APH, 5) / mean(FINAL, 5)))
put("res_45_p", pval(COMP[(RES, "MAE_meV")][0]["p_adj"])); check("res_45_ns", COMP[(RES, "MAE_meV")][0]["p_adj"] >= 0.05)
_ps = [COMP[(c, k)][i]["p_adj"] for c in (SA, RBA, NTK, APH, MLP) for k in ("axis", "shearsign") for i in (0, 1)]
put("sym_all_pmax", pmax(_ps)); check("sym_lowest_all_models", all(COMP[(c, k)][i]["diff"] > 0 for c in (SA, RBA, NTK, APH, MLP) for k in ("axis", "shearsign") for i in (0, 1)) and max(_ps) < 0.05)
check("pinn_cross_same", len({TXT["sa_cross_pct"], TXT["rba_cross_pct"], TXT["ntk_cross_pct"]}) == 1)
put("pinn_cross_pmax", pmax([vsm[m][LV.index(lv)]["p_adj"] for m in (SA, RBA, NTK) for lv in LV if lv >= int(TXT["sa_cross_pct"])]))


def traj(model, pct):
    floor = PD.checkpoint_floor(model, pct)
    return np.mean(floor.best.values), np.mean(floor.train.values)
for m, key in [(SA, "sa"), (RBA, "rba"), (MLP, "mlp"), (FINAL, "ours")]:
    b, t = traj(m, 50); put("floor_%s" % key, "%.0f" % b); put("train_%s" % key, "%.0f" % t)
RAW["formula_family"] = {f: float(FORM["MAE_%s_meV" % f].mean()) for f in FAMS}
pen = {m: [mean(m, 50, "MAE_%s_meV" % f) - mean(MLP, 50, "MAE_%s_meV" % f) for f in FAMS] for m in (SA, RBA)}
put("pen_sa_range", "%.0f to %.0f" % (min(pen[SA]), max(pen[SA]))); put("pen_rba_range", "%.0f to %.0f" % (min(pen[RBA]), max(pen[RBA])))
check("penalty_every_family", all(v > 0 for m in (SA, RBA) for v in pen[m]))
gapf = [FORM["MAE_%s_meV" % f].mean() - mean(MLP, 50, "MAE_%s_meV" % f) for f in FAMS]
put("mlp_vs_formula_range", "%.0f to %.0f" % (min(gapf), max(gapf))); check("mlp_beats_formula_448_every_family", min(gapf) > 0)
LAM = PD.sa_weights(); med = LAM.groupby(["level", "term"]).lam2.median()
put("sa_lambda2_range", "%s to %s" % (num(med.min()), num(med.max()))); RAW["sa_lambda2_medians"] = {"%s_%s" % k: float(v) for k, v in med.items()}
GR = PD.gradients_at_checkpoint()
G = GR[GR.model.isin([SA, RBA, FINAL, RES]) & GR.level.isin([1, 5, 25, 100])].reset_index(drop=True); G.to_csv(os.path.join(OUT, "gradients.csv"), index=False)
gm = G.groupby(["model", "level"]).agg(ratio=("ratio", "median"), cos=("cos", "mean"))
put("sa_ratio_range", "%.0f to %.0f" % (gm.loc[SA].ratio.min(), gm.loc[SA].ratio.max()))
put("rba_ratio_range", "%.1f to %.1f" % (gm.loc[RBA].ratio.min(), gm.loc[RBA].ratio.max()))
put("rba_cos_9", "$%.2f$" % gm.loc[(RBA, 1)].cos); put("rba_cos_896", "$%.2f$" % gm.loc[(RBA, 100)].cos)
check("rba_cos_negative_lowest_896", (gm.loc[RBA].cos < 0).all() and gm.loc[RBA].cos.idxmin() == 100)
put("ours_ratio_range", "%.0f to %.0f\\%%" % (100 * gm.loc[FINAL].ratio.min(), 100 * gm.loc[FINAL].ratio.max()))
put("ours_cos_min", "$%.2f$" % G[G.model == FINAL].groupby("level").cos.mean().min()); put("ours_cos_max", "$%+.2f$" % G[G.model == FINAL].groupby("level").cos.mean().max())
check("ours_cos_weak", G[G.model == FINAL].groupby("level").cos.mean().abs().max() < 0.3 and G[G.model == FINAL].groupby("level").cos.mean().abs().max() < abs(gm.loc[(RBA, 100)].cos))
G2 = GR[GR.model == NTK].reset_index(drop=True); G2.to_csv(os.path.join(OUT, "gradients_ntk.csv"), index=False)   # NTK-PINN, every fraction
gn = G2[G2.model == NTK].groupby("level").agg(ratio=("ratio", "median"), cos=("cos", "mean")) if (G2.model == NTK).any() else None
if gn is not None:
    _g = gn.loc[[l for l in (1, 5, 25, 100) if l in gn.index]]
    put("ntk_ratio_range", "%.0f to %.0f" % (_g.ratio.min(), _g.ratio.max())); put("ntk_cos_896", "$%.2f$" % gn.loc[100].cos)
    _gl = gn.loc[[1, 5, 25]]; put("ntk_ratio_low_range", "%.1f to %.1f" % (_gl.ratio.min(), _gl.ratio.max())); put("ntk_ratio_896", "%.0f" % gn.loc[100].ratio)
    check("ntk_physics_dominates_896", gn.loc[100].ratio > 1)
RAW["gradients"] = json.loads(gm.reset_index().to_json(orient="records", double_precision=15))

# ------------------------------------------------------------------ 7. ablation over every key metric (Table 3, Supplementary Table 7)
# Three terms of PHM removed, and the two terms PHM does not use (symmetry term, band-edge level anchor) added back.
TERMS = ["drop_agreement", "drop_iso_eg", "drop_copies", "add_sym", "add_anchor"]
TERMS_MAIN = TERMS
TNAME = {"drop_agreement": "Agreement term removed", "drop_iso_eg": "Isotropic term removed", "drop_copies": "Copied symmetry labels removed",
         "add_sym": "Symmetry term added", "add_anchor": "Band-edge level anchor added"}
ABL = {}   # ABL[(term, metric)] = {level: result}; the difference is the changed model minus the PHM
for term in TERMS:
    lvls = sorted(A[A.model == term].level.unique())
    for met in KM:
        res = family([compare(series(FINAL, lv, met), A[(A.model == term) & (A.level == lv)].set_index("seed")[met],
                              delta=3.0 if met == "MAE_meV" else None) for lv in lvls])
        ABL[(term, met)] = dict(zip(lvls, res))
check("ablation_runs_complete", all(len(A[A.model == t]) == 153 for t in TERMS))
abl_rows = [dict(term=t, metric=m, level=NL[lv], **{k: v for k, v in r.items()}) for (t, m), d in ABL.items() for lv, r in d.items()]
pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "ablation_all.csv"), index=False)
def cell(r, dec=1):
    v = round(r["diff"], dec); txt = ("%." + str(dec) + "f") % abs(v)
    txt = txt if v == 0 else ("+" if v > 0 else "-") + txt
    return "$%s%s$" % (txt, "^{*}" if r["p_adj"] < 0.05 else "")
lv6 = [0, 1, 2, 5, 10, 25, 100]
lv3a = LV   # Table 3a shows every labeled fraction; Supplementary Table 6 the seven of lv6
lines = ["\\begin{tabular}{l" + "r" * len(lv3a) + "}", "\\toprule", " & \\multicolumn{%d}{c}{\\textbf{Labeled training states (\\%%)}} \\\\" % len(lv3a), "\\cmidrule(l){2-%d}" % (len(lv3a) + 1), "\\textbf{Change to the PHM} & " + " & ".join("\\textbf{%s}" % PCT[p] for p in lv3a) + " \\\\",
         "\\midrule", "None (PHM) & " + " & ".join("%.1f" % mean(FINAL, p) for p in lv3a) + " \\\\", "\\addlinespace"]
for term in TERMS_MAIN:
    if term == "add_sym": lines.append("\\addlinespace")
    lines.append("%s & %s \\\\" % (TNAME[term], " & ".join(cell(ABL[(term, "MAE_meV")][p]) if p in ABL[(term, "MAE_meV")] else "--" for p in lv3a)))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table3a_ablation.tex"), "w").write("\n".join(lines) + "\n")
B_MET = [("RMSE_meV", "RMSE", "meV", 1), ("P95_meV", "P95", "meV", 1), ("max_meV", "Max", "meV", 0), ("dp_err", "DP\\\\error", "eV", 3),
         ("shear_slope", "Shear\\\\slope", "eV", 2), ("axis", "Axis\\\\drift", "meV", 1), ("shearsign", "Shear-sign\\\\drift", "meV", 1)]
def head2(h):
    return "\\begin{tabular}[b]{@{}c@{}}\\textbf{%s}\\end{tabular}" % h.replace("\\\\", "}\\\\\\textbf{")
lines = ["\\begin{tabular}{l" + "r" * len(B_MET) + "}", "\\toprule",
         "\\textbf{Change to the PHM} & " + " & ".join(head2(h) for _, h, _, _ in B_MET) + " \\\\",
         " & " + " & ".join("(%s)" % u for _, _, u, _ in B_MET) + " \\\\", "\\midrule",
         "None (PHM) & " + " & ".join(("%." + str(dc) + "f") % mean(FINAL, 5, c) for c, _, _, dc in B_MET) + " \\\\", "\\addlinespace"]
for term in TERMS_MAIN:
    if term == "add_sym": lines.append("\\addlinespace")
    lines.append("%s & %s \\\\" % (TNAME[term], " & ".join(cell(ABL[(term, c)][5], dc) for c, _, _, dc in B_MET)))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table3b_ablation.tex"), "w").write("\n".join(lines) + "\n")

# values quoted in the text
sig = lambda term, met="MAE_meV", worse=True: [lv for lv, r in ABL[(term, met)].items() if r["p_adj"] < 0.05 and ((r["diff"] > 0) == worse)]
# agreement term: needed with few labels only
ag = ABL[("drop_agreement", "MAE_meV")]
for lv in (0, 1, 2): put("abl_agree_%d" % NL[lv], "%.1f" % ag[lv]["diff"])
put("abl_agree_pmax", pmax([ag[lv]["p_adj"] for lv in (0, 1)])); check("agreement_needed_low", all(ag[lv]["p_adj"] < 0.05 and ag[lv]["diff"] > 0 for lv in (0, 1)))
register("stat_abl_agree_9", "MAE, the PHM without vs with the agreement term, 1% labeled", ag[1], "meV")
put("abl_agree_high_pmin", "$P \\geq %.2f$" % min(ag[lv]["p_adj"] for lv in LV if lv >= 2)); check("agreement_ns_high", all(ag[lv]["p_adj"] >= 0.05 for lv in LV if lv >= 2))
put("mae_ours_45_int", "%.0f" % mean(FINAL, 5))
check("sign_networks_896_above_99", all(mean(m, 100, "sign") > 99 for m in (FINAL, RES, SA, RBA, MLP)))
# isotropic term
ii = ABL[("drop_iso_eg", "MAE_meV")]
put("iso_drop_0", "%.1f" % ii[0]["diff"]); put("iso_drop_9", "%.1f" % ii[1]["diff"])
register("stat_iso_drop_0", "MAE, the PHM without vs with the isotropic term, 0% labeled", ii[0], "meV")
check("iso_needed_0_1_only", sig("drop_iso_eg") == [0, 1] and not sig("drop_iso_eg", worse=False))
for lv, key in [(0, "0"), (5, "45")]:
    put("abl_iso_dp_%s" % key, "%.2f" % ABL[("drop_iso_eg", "dp_err")][lv]["diff"])
register("stat_abl_iso_dp_0", "deformation-potential error, the PHM without vs with the isotropic term, 0% labeled", ABL[("drop_iso_eg", "dp_err")][0], "eV", 2)
RAW["iso_dp_sig"] = [lv for lv in (0, 5) if ABL[("drop_iso_eg", "dp_err")][lv]["p_adj"] < 0.05]
# copied symmetry labels: the largest effect from 1% labels on
cp = ABL[("drop_copies", "MAE_meV")]
for lv in (1, 2, 5, 10, 25, 50, 75, 100): put("abl_copies_%d" % NL[lv], "%.1f" % cp[lv]["diff"])
_cs = sig("drop_copies"); put("abl_copies_sig_range", "%d--%d\\%%" % (min(_cs), max(_cs))); put("abl_copies_pmax", pmax([cp[lv]["p_adj"] for lv in _cs]))
put("abl_copies_max", "%.1f" % max(cp[lv]["diff"] for lv in _cs))
register("stat_abl_copies_45", "MAE, the PHM without vs with the copied symmetry labels, 5% labeled", cp[5], "meV")
check("copies_never_better", not sig("drop_copies", worse=False))
check("copies_sig_contiguous", _cs == [lv for lv in LV if min(_cs) <= lv <= max(_cs)])
r = ABL[("drop_copies", "axis")][5]; put("abl_copies_axis_45", "%.1f" % r["diff"]); register("stat_abl_copies_axis_45", "axis-relabelling drift, the PHM without vs with the copied symmetry labels, 5% labeled", r, "meV")
put("axis_nocopies_45", "%.1f" % (mean(FINAL, 5, "axis") + r["diff"]))
# symmetry term added: no accuracy gain; lower symmetry drift
sy = ABL[("add_sym", "MAE_meV")]
put("abl_addsym_worse_levels", ", ".join("%d" % lv for lv in sig("add_sym")) + "\\%"); put("abl_addsym_worse_pmax", pmax([sy[lv]["p_adj"] for lv in sig("add_sym")]) if sig("add_sym") else "--")
put("abl_addsym_896", "%.1f" % sy[100]["diff"]); register("stat_abl_addsym_896", "MAE, the PHM with vs without the symmetry term, 100% labeled", sy[100], "meV")
check("addsym_never_better", not sig("add_sym", worse=False))
for lv in (5, 100):
    r = ABL[("add_sym", "axis")][lv]; put("abl_addsym_axis_%d" % NL[lv], "%.1f" % -r["diff"]); put("axis_addsym_%d" % NL[lv], "%.1f" % r["mean_b"])
register("stat_abl_addsym_axis_45", "axis-relabelling drift, the PHM with vs without the symmetry term, 5% labeled", ABL[("add_sym", "axis")][5], "meV", 1, sign=-1)
check("addsym_lowers_axis", all(ABL[("add_sym", "axis")][lv]["diff"] < 0 and ABL[("add_sym", "axis")][lv]["p_adj"] < 0.05 for lv in (5, 100)))
# band-edge level anchor added: no effect on any metric at any fraction
an_sig = [(c, lv) for c in KM for lv, r in ABL[("add_anchor", c)].items() if r["p_adj"] < 0.05]; RAW["anchor_significant"] = an_sig
_amin = min(r["p_adj"] for r in ABL[("add_anchor", "MAE_meV")].values())
put("abl_anchor_pmin", "$P = 1.00$ at every fraction" if _amin > 0.995 else "$P \\geq %.2f$" % _amin)
check("anchor_no_mae_effect", all(r["p_adj"] >= 0.05 for r in ABL[("add_anchor", "MAE_meV")].values()))

# ------------------------------------------------------------------ 8. calibration cost and label-matched constants
L = load("labelmatched")
lmres = family([compare(series(FINAL, lv), L[(L.model == FINAL) & (L.level == lv)].set_index("seed").MAE_meV) for lv in (1, 2, 5, 10)])
for lv, r in zip((1, 2, 5, 10), lmres): put("lm_%d" % NL[lv], "%.1f" % r["diff"])
register("stat_lm_18", "MAE, the PHM with label-matched vs training-portion constants, 2% labeled", lmres[1], "meV")
register("stat_lm_45", "MAE, the PHM with label-matched vs training-portion constants, 5% labeled", lmres[2], "meV")
register("stat_lm_90", "MAE, the PHM with label-matched vs training-portion constants, 10% labeled", lmres[3], "meV")
check("lm_weak_18_45_90", lmres[1]["diff"] > 0 and lmres[2]["diff"] > 0 and all(lmres[i]["p_adj"] >= 0.05 for i in (1, 2, 3)))
put("lm_9_median", "%.0f" % L[(L.model == FINAL) & (L.level == 1)].MAE_meV.median())
lmr = [mean(MLP, lv) / L[(L.model == FINAL) & (L.level == lv)].MAE_meV.mean() for lv in (2, 5, 10)]
put("lm_ratio_range", "%.1f to %.1f" % (min(lmr), max(lmr)))
LMT = []
for m in [FINAL, RES]:
    res = family([compare(series(m, lv), L[(L.model == m) & (L.level == lv)].set_index("seed").MAE_meV) for lv in (1, 2, 5, 10)])
    for lv, r in zip((1, 2, 5, 10), res):
        LMT.append(dict(model=m, level=lv, lm=r["mean_b"], pool=r["mean_a"], diff=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p_adj"], nophys=mean(MLP, lv)))
LMT = pd.DataFrame(LMT); LMT.to_csv(os.path.join(OUT, "labelmatched.csv"), index=False)
cal = []
for p in (1, 2, 5, 10):
    v = []
    for s in SEEDS:
        t = PD.test_frame(s)
        cb_, vb_ = analytic_bands(t.sample_id.values, PD.constants_file("labelmatched", s, p))
        v.append(1000 * np.abs(cb_ - vb_ - t.Eg_true_eV.values).mean())
    v = np.array(v); cal.append(dict(n=NL[p], mean=v.mean(), median=np.median(v), q1=np.percentile(v, 25), q3=np.percentile(v, 75), failed=int((v > 1000).sum())))
CAL = pd.DataFrame(cal); CAL.to_csv(os.path.join(OUT, "calibration_cost.csv"), index=False)
for _, c in CAL.iterrows(): put("cal_%d_mean" % c.n, num(c["mean"])); put("cal_%d_median" % c.n, num(c["median"]))
put("cal_9_failed", "%d" % CAL[CAL.n == 9].failed.iloc[0]); check("cal_9_one_failure", CAL[CAL.n == 9].failed.iloc[0] == 1)
put("cal_90_gap", "%.0f" % (CAL[CAL.n == 90]["mean"].iloc[0] - FORM.MAE_meV.mean()))

# ------------------------------------------------------------------ 9. beyond the training data
HS = PD.heldout_suite()      # the analytic models alone and the baselines; PHM and the residual model from their run folders
new = pd.concat([PD.heldout_runs(FINAL), PD.heldout_runs(RES)], ignore_index=True)
check("heldout_runs_complete", len(new) == 2 * 136); HS = pd.concat([HS, new], ignore_index=True)
hs = HS.groupby(["model", "level"]).agg(mae=("MAE_meV", "mean"), sd=("MAE_meV", "std"), r2=("R2", "mean"), n=("seed", "count")).reset_index()
hs.to_csv(os.path.join(OUT, "heldout.csv"), index=False)
H = lambda m, l: float(hs[(hs.model == m) & (hs.level == l)].mae.iloc[0])
for m, key in [("formula", "formula"), (FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (MLP, "mlp"), (RES, "res")]:
    for l, lk in [("magnitude_03", "03"), ("magnitude_05", "05"), ("magnitude_07", "07"), ("group_triaxial", "tri"), ("group_isotropic", "iso"),
                  ("group_uniaxial", "uni"), ("group_biaxial", "bi")]:
        put("ho_%s_%s" % (key, lk), ("%.0f" % H(m, l)) if lk in ("03", "05", "07", "tri") else ("%.1f" % H(m, l)))   # one precision per sentence
put("ho_ours_05_r2", "%.2f" % hs[(hs.model == FINAL) & (hs.level == "magnitude_05")].r2.iloc[0])
rr = [H("formula", l) / H(FINAL, l) for l in ("group_isotropic", "group_uniaxial", "group_biaxial")]
put("ho_ratio_range", "%.0f to %.0f" % (min(rr), max(rr)))
sh = [H(m, "group_shear") for m in hs[hs.level == "group_shear"].model.unique()]
put("ho_shear_range", "%.2f to %.2f" % (min(sh) / 1000, max(sh) / 1000))
_hf = max(H(FINAL, l) / H("formula", l) for l in ("magnitude_03", "magnitude_05", "magnitude_07", "group_triaxial"))
put("ho_ours_factor_max", "%.1f" % (math.ceil(_hf * 10) / 10)); check("ho_factor_upper_bound", math.ceil(_hf * 10) / 10 >= _hf)   # an upper bound, rounded up
RAW["ho_ours_factors"] = {l: H(FINAL, l) / H("formula", l) for l in ("magnitude_03", "magnitude_05", "magnitude_07", "group_triaxial")}
_of = [H(m, "magnitude_05") / H("formula", "magnitude_05") for m in (SA, RBA, NTK)]; put("ho_obj_factor_range", "%.0f to %.0f" % (min(_of), max(_of)))
check("ho_ours_worse_03_05_better_07", H(FINAL, "magnitude_03") > H("formula", "magnitude_03") and H(FINAL, "magnitude_05") > H("formula", "magnitude_05") and H(FINAL, "magnitude_07") < H("formula", "magnitude_07"))
check("ho_ours_within_2x", max(H(FINAL, l) / H("formula", l) for l in ("magnitude_03", "magnitude_05", "magnitude_07", "group_triaxial")) < 2)
check("ho_pinns_worse_than_formula_05", H(SA, "magnitude_05") > H("formula", "magnitude_05") and H(RBA, "magnitude_05") > H("formula", "magnitude_05"))
nz = M[(M.model == FINAL) & M.level.isin([5, 100])].noise.abs().max() * 100
put("noise_max", "%d" % math.ceil(nz)); RAW["noise_max_pct"] = float(nz)

# ------------------------------------------------------------------ 10. Table 4: constants of the analytic models
CON = pd.DataFrame([json.load(open(PD.constants_file("2band", s))) for s in SEEDS])
SYM = [("CBM_0", "$E^{0}_{\\mathrm{CBM}}$", "(\\ref{eq:cbm})", "offset"), ("XI_D", "$\\Xi_d$", "(\\ref{eq:cbm})", "linear"),
       ("XI_U", "$\\Xi_u$", "(\\ref{eq:cbm})", "linear"), ("XI_UP", "$\\Xi_u'$", "(\\ref{eq:cbm})", "linear"),
       ("XI_Q", "$\\Xi_Q$", "(\\ref{eq:cbm})", "quadratic"), ("XI_C", "$\\Xi_C$", "(\\ref{eq:cbm})", "quadratic"),
       ("XI_Q2", "$\\Xi_{Q2}$", "(\\ref{eq:cbm})", "quadratic"), ("DELTA_X", "$\\Delta_X$", "(\\ref{eq:shear})", "band separation"), ("VBM_0", "$E^{0}_{\\mathrm{VBM}}$", "(\\ref{eq:vbm})", "offset"),
       ("AV", "$a_v$", "(\\ref{eq:vbm})", "linear"), ("AV2", "$a_{v2}$", "(\\ref{eq:vbm})", "quadratic"), ("B_BP", "$b$", "(\\ref{eq:vbm})", "linear"),
       ("B2_BP", "$b_2$", "(\\ref{eq:vbm})", "quadratic"), ("D_BP", "$d_v$", "(\\ref{eq:vbm})", "linear")]
lines = ["\\begin{tabular}{lllrr}", "\\toprule", "\\textbf{Constant} & \\textbf{Equation} & \\textbf{Order in strain} & \\textbf{Mean (eV)} & \\textbf{s.d. (eV)} \\\\", "\\midrule"]
for k, sym, eq, kind in SYM:
    lines.append("%s & %s & %s & $%.3f$ & %.3f \\\\" % (sym, eq, kind, CON[k].mean(), CON[k].std(ddof=1)))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table4_constants.tex"), "w").write("\n".join(lines) + "\n")
lin = ["XI_D", "XI_U", "XI_UP", "AV", "B_BP", "D_BP"]; put("const_linear_sd_max", "%.2f" % max(CON[k].std(ddof=1) for k in lin))
put("delta_x", "%.2f" % CON.DELTA_X.mean()); put("delta_x_range", "%.2f--%.2f" % (CON.DELTA_X.min(), CON.DELTA_X.max()))
# separation of the two lowest conduction bands along Gamma-X in the unstrained HSE06 calculation, at the k-point nearest the valley minimum
_gx = PD.gamma_x_bands(); _near = _gx[-1]   # the sampled point closest to X
put("sep_unstrained", "%.2f" % (_near[2] - _near[1])); put("sep_k", "%.2f" % _near[0]); RAW["gamma_x_bands"] = _gx
put("gx_points", ", ".join("%.2f" % g[0] for g in _gx[:-1]) + " and %.2f" % _gx[-1][0])
check("delta_x_close_to_dft", abs(CON.DELTA_X.mean() - (_near[2] - _near[1])) < 0.3)
# smooth minimum and maximum against the exact ones, mean constants, all states of the dataset
for k in CON.columns: setattr(P, k, float(CON[k].mean()))
_x = torch.tensor(DATA[SC].values, dtype=torch.float64)
put("soft_cbm_dev", "%.1f" % (1000 * (P.physics_cbm(_x) - P.physics_cbm(_x, beta=50.0)).abs().mean()))
_th = torch.tensor([float(CON[k].mean()) for k in T.THETA_KEYS], dtype=torch.float64).expand(len(_x), -1)
put("soft_vbm_dev", "%.1f" % (1000 * (T.vbm_from_theta(_x, _th) - P.physics_vbm(_x)).abs().mean()))
_ac = PD.aphynity_choice()
if _ac is not None: put("aph_tau2", "%g" % _ac.sort_values("val_mae_meV").tau2.iloc[0])
put("const_linear_range_max", "%.2f" % max(CON[k].max() - CON[k].min() for k in lin))

# per-state error of the PHM and of the MLP with all training states labeled, mean over the runs in which the state was tested
_se = {}
_SER = PD.state_errors([FINAL, MLP], 100)
for _m, _col in ((FINAL, "err_ours"), (MLP, "err_mlp")):
    for _i, _e in zip(_SER[_SER.model == _m].sample_id.values, _SER[_SER.model == _m].error_meV.values): _se.setdefault((int(_i), _col), []).append(_e)
SE = pd.DataFrame({c: pd.Series({i: np.mean(v) for (i, cc), v in _se.items() if cc == c}) for c in ("err_ours", "err_mlp")}).dropna()

# ------------------------------------------------------------------ 10b. why shear is the hardest family (Supplementary Note 8)
from scipy.spatial import cKDTree
SEFAM = DATA.strain_type.map(fam_of).values; XS = DATA[SC].values; I1s = XS[:, :3].sum(1); cm = CON.mean()
def _val(e): return cm["XI_D"] * I1s + cm["XI_U"] * e + cm["XI_Q"] * e ** 2 + cm["XI_C"] * I1s * e + cm["XI_Q2"] * I1s ** 2
# valley energy per axis (the two valleys on one axis are degenerate), mean calibrated constants, two-band shear term
_d = np.stack([_val(XS[:, 2]) + P.shear_shift(XS[:, 3], cm["XI_UP"], cm["DELTA_X"], np), _val(XS[:, 1]) + P.shear_shift(XS[:, 5], cm["XI_UP"], cm["DELTA_X"], np),
               _val(XS[:, 0]) + P.shear_shift(XS[:, 4], cm["XI_UP"], cm["DELTA_X"], np)], 1)
_axis = _d.argmin(1); _srt = np.sort(_d, 1); _margin = _srt[:, 1] - _srt[:, 0]          # axis of the lowest valleys; gap to the next axis (eV)
_dist, _idx = cKDTree(XS).query(XS, k=7); _kp = DATA.cbm_kpt_index.values
SH = pd.DataFrame(dict(fam=SEFAM, planes=DATA.n_shear_planes.values, margin=_margin, nn=_dist[:, 1:6].mean(1),
                       axis_sw=[(_axis[_idx[i, 1:]] != _axis[i]).mean() for i in range(len(DATA))],
                       grid_only=[((_kp[_idx[i, 1:]] != _kp[i]) & (_axis[_idx[i, 1:]] == _axis[i])).mean() for i in range(len(DATA))]))
SH = SH.join(SE, how="inner")                                                                     # states that were in a test portion at least once
sh = SH[SH.fam == "shear"]; mu = sh[sh.planes >= 2]; vc = mu[mu.axis_sw > 0]; nv = mu[mu.axis_sw == 0]
sq = lambda d: float((d.err_ours ** 2).sum())
for k in (1, 2, 3):
    g = sh[sh.planes == k]; put("sh_n_%d" % k, "%d" % len(g)); put("sh_err_%d" % k, "%.1f" % g.err_ours.mean()); put("sh_mlp_%d" % k, "%.1f" % g.err_mlp.mean())
put("sh_multi_share", "%.1f" % (100 * sq(mu) / sq(sh)))
put("sh_vc_frac", "%.0f" % (100 * len(vc) / len(mu))); put("sh_vc_err", "%.1f" % vc.err_ours.mean()); put("sh_nvc_err", "%.1f" % nv.err_ours.mean())
put("sh_vc_mlp", "%.1f" % vc.err_mlp.mean()); put("sh_vc_share", "%.1f" % (100 * sq(vc) / sq(sh)))
go = mu[(mu.axis_sw == 0) & (mu.grid_only > 0)]; put("sh_grid_err", "%.1f" % go.err_ours.mean()); put("sh_grid_n", "%d" % len(go))
near = mu[mu.margin < 0.3]; far = mu[mu.margin >= 0.3]
put("sh_near_err", "%.1f" % near.err_ours.mean()); put("sh_far_err", "%.1f" % far.err_ours.mean()); put("sh_near_n", "%d" % len(near)); put("sh_far_n", "%d" % len(far))
terc = pd.qcut(vc.nn, 3, labels=["dense", "middle", "sparse"]); te = vc.groupby(terc, observed=True).err_ours.mean()
put("sh_dense_err", "%.1f" % te["dense"]); put("sh_mid_err", "%.1f" % te["middle"]); put("sh_sparse_err", "%.1f" % te["sparse"])
put("sh_nn_ratio", "%.1f" % (sh[sh.planes == 3].nn.mean() / sh[sh.planes == 1].nn.mean()))
check("shear_concentrated", sq(mu) / sq(sh) > 0.95 and sh[sh.planes == 1].err_ours.mean() < 5 and sh[sh.planes == 3].err_ours.mean() > 20)
check("shear_valley_crossings", vc.err_ours.mean() > 3 * nv.err_ours.mean() and sq(vc) / sq(sh) > 0.9)
check("shear_grid_not_cause", go.err_ours.mean() < 8 and go.err_ours.mean() < vc.err_ours.mean() / 3)
check("shear_density_trend", te["dense"] < te["middle"] < te["sparse"] and sh[sh.planes == 3].nn.mean() > 2 * sh[sh.planes == 1].nn.mean())
SH.assign(valley_change=SH.axis_sw > 0).to_csv(os.path.join(OUT, "shear_states.csv"))

# ------------------------------------------------------------------ 10c. tests aimed at the shear error (Supplementary Note 8)
FT = PD.shear_formula_test().groupby("group")[["linear", "calibrated", "extended"]].mean()
for g, k in [("all", "all"), ("other families", "other"), ("shear 1 plane", "s1"), ("shear 2 planes", "s2"), ("shear 3 planes", "s3")]:
    for c, ck in [("linear", "lin"), ("calibrated", "two"), ("extended", "ext")]:
        put("ft_%s_%s" % (ck, k), "%.1f" % FT.loc[g, c])
_gain = lambda g: FT.loc[g, "linear"] - FT.loc[g, "calibrated"]
check("ft_two_band_gain_smallest_three_planes", _gain("shear 3 planes") < _gain("shear 1 plane"))
check("ft_extended_no_gain", all(FT.loc[g, "extended"] >= FT.loc[g, "calibrated"] for g in FT.index))

# ------------------------------------------------------------------ 11. cost
# cost of one reference calculation: wall time x cores of every HSE06 step (OUTCAR), median over the states
_ch = PD.hse_core_hours()
RAW["hse_core_hours"] = dict(n=len(_ch), median=float(np.median(_ch)), q1=float(np.percentile(_ch, 25)), q3=float(np.percentile(_ch, 75)))
put("hse_core_hours", "%.0f" % np.median(_ch)); put("hse_n_timed", "{:,}".format(len(_ch)).replace(",", "{,}"))
check("hse_timed_most_states", len(_ch) > 1000)
inf = PD.inference_text()
put("inference_ms", "%.2f" % float(re.search(r"single state ([0-9.]+) ms", inf).group(1)))
check("inference_under_1ms", float(TXT["inference_ms"]) < 1.0)   # Conclusions: "under a millisecond"
def minutes(model):
    return float(np.mean(PD.run_minutes(model)))
put("min_ours", "%.0f" % minutes(FINAL)); put("min_sa", "%.0f" % minutes(SA)); put("min_rba", "%.0f" % minutes(RBA))

# ------------------------------------------------------------------ 12. Supplementary tables
def si_table(name, header, rows, colspec):
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(["\\begin{tabular}{%s}" % colspec, "\\toprule", header + " \\\\", "\\midrule"] + [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]) + "\n")
# S1: every model at every data volume (two stacked panels)
allm = pd.concat([M[["model", "level", "seed", "MAE_meV"]], CL[["model", "level", "seed", "MAE_meV"]]], ignore_index=True)
s1 = allm.groupby(["model", "level"]).MAE_meV.agg(["mean", "std"])
ORDER = [(FINAL, "PHM"), (RES, "Residual model"), (SA, "SA-PINN"), (RBA, "RBA-PINN"), (NTK, "NTK-PINN"), (APH, "APHYNITY"),
         (MLP, "MLP"), (GP, "Gaussian process"), ("RandomForest", "Random forest"), ("GBDT", "Gradient boosting")]
def s1panel(levels):
    out = ["\\begin{tabular}{l" + "r" * len(levels) + "}", "\\toprule", " & \\multicolumn{%d}{c}{\\textbf{Labeled training states (\\%%)}} \\\\" % len(levels), "\\cmidrule(l){2-%d}" % (len(levels) + 1), "\\textbf{Model} & " + " & ".join("\\textbf{%s}" % PCT[p] for p in levels) + " \\\\", "\\midrule"]
    for m, lab in ORDER:
        out.append("%s & %s \\\\" % (lab, " & ".join(mean_sd(*s1.loc[(m, p)]) if (m, p) in s1.index else "--" for p in levels)))
    return "\n".join(out + ["\\bottomrule", "\\end{tabular}"]) + "\n"
open(os.path.join(OUT, "si_all_models.tex"), "w").write(s1panel([0, 1, 2, 5, 10]) + "\n\\vspace{1.5mm}\n" + s1panel([25, 50, 75, 100]))
# S2: held-out suite
SCH = [("group_isotropic", "Isotropic"), ("group_uniaxial", "Uniaxial"), ("group_biaxial", "Biaxial"), ("group_triaxial", "Triaxial"), ("group_shear", "Shear"),
       ("magnitude_03", "0.03"), ("magnitude_05", "0.05"), ("magnitude_07", "0.07")]
rows = []
for m, lab in [("formula", "Analytic models alone"), (FINAL, "PHM"), (RES, "Residual model"), (SA, "SA-PINN"), (RBA, "RBA-PINN"), (NTK, "NTK-PINN"),
               (APH, "APHYNITY"), (MLP, "MLP")]:
    cells = []
    for l, _ in SCH:
        g = hs[(hs.model == m) & (hs.level == l)]
        cells.append(("%.0f $\\pm$ %.0f" % (g.mae.iloc[0], g.sd.iloc[0])) if len(g) else "--")
    rows.append("%s & %s" % (lab, " & ".join(cells)))
si_table("si_heldout", " & \\multicolumn{5}{c}{\\textbf{Family left out}} & \\multicolumn{3}{c}{\\textbf{Strain beyond $|\\varepsilon|$}} \\\\\n"
         "\\cmidrule(lr){2-6}\\cmidrule(l){7-9}\n\\textbf{Model} & " + " & ".join("\\textbf{%s}" % h for _, h in SCH), rows, "l" + "r" * len(SCH))
# S3: label-matched constants
rows = []
for _, r in LMT.iterrows():
    lab = {FINAL: "PHM", RES: "Residual model"}[r.model]
    rows.append("%s & %s & %.1f & %.1f & $%+.1f$ & %s & %.1f" % (lab, PCT[r.level], r.lm, r.pool, r["diff"], ("$<$0.001" if r.p < 0.001 else "%.2g" % r.p), r.nophys))
for _, c in CAL.iterrows():
    rows.append("Analytic models alone & %s & %.1f (median %.1f) & %.1f & -- & -- & --" % ({9: "1", 18: "2", 45: "5", 90: "10"}[int(c.n)], c["mean"], c["median"], FORM.MAE_meV.mean()))
si_table("si_labelmatched", "\\textbf{Model} & %s & %s & %s & \\textbf{Difference} & \\textbf{$P$} & %s" % (head2("Labeled\\\\(\\%)"), head2("Label-matched\\\\constants"), head2("Training-portion\\\\constants"), head2("MLP, same\\\\fraction")), rows, "lrrrrrr")
# S4: conventional regressors
rows = []
for m, lab in [(GP, "Gaussian process"), ("RandomForest", "Random forest"), ("GBDT", "Gradient boosting")]:
    g = CL[CL.model == m].groupby("level").MAE_meV.agg(["mean", "std"])
    rows.append("%s & %s" % (lab, " & ".join(("%.0f $\\pm$ %.0f" % tuple(g.loc[p])) if p in g.index else "--" for p in LV[1:])))
si_table("si_classical", "\\textbf{Model} & " + " & ".join("\\textbf{%s\\%%}" % PCT[p] for p in LV[1:]), rows, "l" + "r" * 8)
# S6: gradients
fr_ = lambda x: ("%.0f" % x) if x >= 10 else ("%.1f" % x if x >= 1 else ("%.2f" % x if x >= 0.005 else "$<$0.01"))
rows = ["%s & %s" % (NAME[m], " & ".join("%s & $%+.2f$" % (fr_(gm.loc[(m, p)].ratio), gm.loc[(m, p)].cos) for p in (1, 5, 25, 100))) for m in (FINAL, RES, SA, RBA)]
si_table("si_gradients", " & " + " & ".join("\\multicolumn{2}{c}{\\textbf{%s\\%% labeled}}" % PCT[p] for p in (1, 5, 25, 100)) + " \\\\\n"
         + "".join("\\cmidrule(lr){%d-%d}" % (2 + 2 * k, 3 + 2 * k) for k in range(4)) + "\n\\textbf{Model}" + " & \\textbf{Ratio} & \\textbf{Cosine}" * 4, rows, "l" + "rr" * 4)
# S7: ablation, every metric and data volume (difference with 95% CI and Holm-adjusted P)
rows = []
for term in TERMS:
    rows.append("\\multicolumn{%d}{l}{\\textit{%s}}" % (len(lv6) + 1, TNAME[term]))
    for met, lab, unit, dec in [("MAE_meV", "MAE", "meV", 1)] + B_MET:
        cells = []
        for p in lv6:
            r = ABL[(term, met)].get(p)
            cells.append("--" if r is None else "%s (%s)" % (cell(r, dec).replace("^{*}", ""), "$<$0.001" if r["p_adj"] < 0.001 else "%.2g" % r["p_adj"]))
        rows.append("\\quad %s (%s) & %s" % (lab.replace("\\\\", " "), unit, " & ".join(cells)))
    rows.append("\\addlinespace")
open(os.path.join(OUT, "si_ablation_rows.tex"), "w").write("\n".join(r + " \\\\" if not r.startswith("\\addlinespace") else r for r in rows[:-1]) + "\n")

# S9 and S10: preliminary studies that fixed the settings (stratified random division, three runs per variant)
cap = PD.preliminary("capacity_decision")
cap["depth"] = cap.variant.str.extract(r"d(\d+)")[0].astype(int); cap["width"] = cap.variant.str.extract(r"w(\d+)")[0].astype(int)
pm, ps_, pn = (cap.pivot(index="depth", columns="width", values=v) for v in ("test_mae_meV_mean", "test_mae_meV_std", "n_params"))
_cap = lambda d, w: ("\\textbf{%.1f $\\pm$ %.1f}" if (d, w) == (8, 64) else "%.1f $\\pm$ %.1f") % (pm.loc[d, w], ps_.loc[d, w])   # the selected network in bold
rows = ["%d & %s" % (d, " & ".join(_cap(d, w) for w in pm.columns)) for d in pm.index]
rows += ["\\midrule\n\\multicolumn{%d}{l}{\\textit{Trainable parameters}}" % (len(pm.columns) + 1)]
rows += ["%d & %s" % (d, " & ".join("{:,}".format(int(pn.loc[d, w])).replace(",", "{,}") for w in pm.columns)) for d in pm.index]
si_table("si_capacity", "\\textbf{Hidden layers} & " + " & ".join("\\textbf{Width %d}" % w for w in pm.columns), rows, "l" + "r" * len(pm.columns))
check("capacity_selected", pn.loc[8, 64] == 29698)
LAB = {"ref": "Reference: 5 layers $\\times$ 100 units", "widen": "Wider layers (156{,}250 parameters)", "layernorm": "Layer normalization",
       "residual": "Skip connections", "taper": "Tapered widths", "residual_norm": "Skip connections and layer normalization",
       "dropout_005": "Dropout, rate 0.05", "dropout_010": "Dropout, rate 0.10", "dropout_020": "Dropout, rate 0.20",
       "two_phase": "Reference: AdamW, then SOAP", "soap_only": "SOAP only", "adamw_only": "AdamW only",
       "switch_050": "Reference: switch at 50\\% of training", "switch_025": "Switch at 25\\%", "switch_075": "Switch at 75\\%",
       "clip_10": "Reference: gradient clipping at 1.0", "clip_off": "No gradient clipping", "clip_50": "Clipping at 5.0", "clip_05": "Clipping at 0.5",
       "clip_01": "Clipping at 0.1"}
fmt = lambda r: "%.1f $\\pm$ %.1f" % (r.test_mae_meV_mean, r.test_mae_meV_std)
rows = []
el = PD.preliminary("elements_decision")
for v in ["ref", "widen", "layernorm", "residual", "taper", "residual_norm", "dropout_005", "dropout_010", "dropout_020"]:
    r = el[el.variant == v].iloc[0]; rows.append("%s & %s & %s & %d" % ("Architecture" if v == "ref" else "", LAB[v], fmt(r), r.n_seeds))
rows.append("\\addlinespace")
op = PD.preliminary("optimizer_decision"); op = op[op.group != "sa_clamp"]   # the bound on SA weights concerns a model not in this paper
first = True
for grp, order in [("arms", ["two_phase", "soap_only", "adamw_only"]), ("switch", ["switch_050", "switch_025", "switch_075"]), ("clip", ["clip_10", "clip_off", "clip_50", "clip_05", "clip_01"])]:
    for v in order:
        r = op[op.variant == v].iloc[0]; rows.append("%s & %s & %s & %d" % ("Optimizer" if first else "", LAB[v], fmt(r), r.n_seeds)); first = False
rows.append("\\addlinespace")
ep = PD.preliminary("epoch_decision"); first = True
for _, r in ep.iterrows():
    it = int(str(r.budget).replace("ep", "")); bind = str(r.budget_binding).replace("%", "\\%")
    rows.append("%s & %s iterations; final checkpoint retained in %s of runs%s & %s & %d" % ("Training length" if first else "", "{:,}".format(it).replace(",", "{,}"), bind,
                " (reference)" if it == 6000 else "", fmt(r), r.n_seeds)); first = False
pa = PD.preliminary("patience_decision"); first = True
for _, r in pa.iterrows():
    rows.append("%s & Early stopping, patience %s (stopped after %s iterations) & %s & %d" % ("" , r.variant[2:], "{:,}".format(int(r.epochs_used)).replace(",", "{,}"), fmt(r), r.n_seeds))
rows.append("\\addlinespace")
sp = PD.preliminary("split_decision"); first = True
for v, lab in [("stratified", "Reference: stratified by family"), ("random", "Random"), ("kfold", "Five-fold cross-validation")]:
    r = sp[sp.scheme == v].iloc[0]; rows.append("%s & %s & %s & %d" % ("Data division" if first else "", lab, fmt(r), r.n_seeds)); first = False
si_table("si_settings", "\\textbf{Study} & \\textbf{Variant} & \\textbf{Test MAE (meV)} & \\textbf{Runs}", rows, "llrr")

# ------------------------------------------------------------------ 13. Supplementary Table 8: every test quoted in the text, and output
TT = pd.DataFrame(TESTS); TT.to_csv(os.path.join(OUT, "tests.csv"), index=False)
put("n_test_agree", "%d of the %d" % (TT.agree.sum(), len(TT))); RAW["wilcoxon_disagree"] = list(TT[~TT.agree].key)
rows = []
for _, r in TT.iterrows():
    unit = "" if r.unit in ("", "percentage points") else "~" + r.unit
    dc = 2 if r.unit == "eV" else 1
    pf = lambda q: "$<$0.001" if q < 0.001 else ("%.3f" % q if 0.995 <= q < 1 else "%.2g" % q)   # 0.9966 must not print as 1
    rows.append("%s & %.2f & %s & %.2f & $%.*f$ to $%.*f$%s & %s & %s" % ((r.what[0].upper() + r.what[1:]).replace("%", "\\%"), r.t, pf(r.p), r.d, dc, r.lo, dc, r.hi, unit,
                                                                   pf(r.p_w), pf(r.p_sw)))
si_table("si_tests", "\\textbf{Comparison} & \\textbf{$t(16)$} & \\textbf{$P$} & \\textbf{$d$} & \\textbf{95\\% CI} & \\textbf{Wilcoxon $P$} & \\textbf{Shapiro--Wilk $P$}", rows,
         ">{\\raggedright\\arraybackslash}p{6.2cm}rrrrrr")
json.dump(dict(txt=TXT, raw=RAW, checks=CHECKS, tests=TESTS), open(os.path.join(OUT, "numbers_v3.json"), "w"), indent=1, default=float)
bad = [k for k, v in CHECKS.items() if not v]
print("wrote %d text values, %d tests, %d checks (%d failed: %s)" % (len(TXT), len(TESTS), len(CHECKS), len(bad), ", ".join(bad) or "none"))
