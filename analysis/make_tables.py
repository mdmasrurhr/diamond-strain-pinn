"""Stage 2 of the paper's analysis: every number, statistical test and table of the paper, from summary/.

Reads summary/ (written by export_summary.py), the dataset and the constants in results/, and writes outputs/tables/:
  numbers.json       "txt": every value quoted in the text, formatted as printed; "raw": underlying values;
                     "checks": the claims the text makes, each True or False; "tests": every statistical test quoted in
                     the text (Supplementary Table 8)
  table1_dataset.tex, table2_metrics.tex, table3a_ablation.tex, table3b_ablation.tex, table4_constants.tex
  si_*.tex           Supplementary Tables 1-12
  ablation_all.csv, labelmatched.csv, calibration_cost.csv, heldout.csv, tests.csv   intermediate tables
Ends with the number of failed checks; every check must pass for the text to hold.
Usage: python analysis/make_tables.py"""
import json, math, os, re, shutil, sys
import tempfile
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
S = os.path.join(ROOT, "summary")
sys.path.insert(0, ROOT)
import metrics_report as MR
OUT = os.path.join(ROOT, "outputs", "tables"); os.makedirs(OUT, exist_ok=True)

FINAL, SA, RBA, MLP, DELTA, GP = "hyb_sdp_gap_ld9", "sa_pinn", "rba_pinn", "mlp", "delta", "GaussianProcess"
LV = [0, 1, 2, 5, 10, 25, 50, 75, 100]; NL = dict(zip(LV, [0, 9, 18, 45, 90, 224, 448, 672, 896])); SEEDS = list(range(42, 59))
PCT = {p: "%d" % p for p in LV}   # labeled fraction of the training portion, in percent: the label used in every table
FAMS = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]; SC = ["E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_zx"]
NAME = {FINAL: "Our model", SA: "SA-PINN", RBA: "RBA-PINN", DELTA: "$\\Delta$-learning", MLP: "MLP", GP: "Gaussian process",
        "formula": "Analytic models alone"}
DATA = pd.read_csv(os.path.join(ROOT, "data", "dft_hy_v9.csv"))
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


M = pd.read_csv(os.path.join(S, "runs_main.csv")); A = pd.read_csv(os.path.join(S, "runs_ablation.csv"))
NLAB = M[M.model == MLP].groupby("level").n_labeled.agg(["min", "max"])
RAW["n_labeled_range"] = {int(k): [int(v["min"]), int(v["max"])] for k, v in NLAB.iterrows()}
for p in (10, 25, 50, 75, 100): put("nlab_%d" % p, ("%d" % NLAB.loc[p, "min"]) if NLAB.loc[p, "min"] == NLAB.loc[p, "max"] else "%d--%d" % (NLAB.loc[p, "min"], NLAB.loc[p, "max"]))
check("label_counts_nominal", all(abs(NLAB.loc[p, "min"] - NL[p]) <= 4 and abs(NLAB.loc[p, "max"] - NL[p]) <= 4 for p in LV))
M["dp_err"] = (M.dp - DP_REF).abs(); A["dp_err"] = (A.dp - DP_REF).abs()
check("ablation_diagnostics_complete", A.dp.notna().all())

FORM = pd.read_csv(os.path.join(S, "runs_analytic.csv"))
DP_AN = float(M[M.model == FINAL].dp_an.mean())
FORM["dp_err"] = abs(DP_AN - DP_REF); FORM["shear_slope"] = 0.0; FORM["axis"] = 0.0; FORM["shearsign"] = 0.0
RAW["dp_analytic"] = DP_AN; put("dp_an", "$%.2f$" % DP_AN)
CL = pd.read_csv(os.path.join(S, "classical_runs.csv")).rename(columns={"dft_pct": "level", "test_mae_meV": "MAE_meV", "test_r2": "R2"})


def series(model, level, col="MAE_meV", frame=None):
    f = M if frame is None else frame
    return f[(f.model == model) & (f.level == level)].set_index("seed")[col]


def mean(model, level, col="MAE_meV"):
    return float(series(model, level, col).mean())


print("main runs %d, ablation runs %d, classical runs %d" % (len(M), len(A), len(CL)))

# ------------------------------------------------------------------ 3. Table 1: dataset and its division
STUB = tempfile.mkdtemp()
sys.argv = ["03_train_pinn.py", "--data_csv", os.path.join(ROOT, "data/dft_hy_v9.csv"), "--results_dir", STUB,
            "--device", "cpu", "--split", "symmetry"]
import importlib.util
spec = importlib.util.spec_from_file_location("trn", os.path.join(ROOT, "03_train_pinn.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
fam_all = DATA.strain_type.map(fam_of).values; cnt = []
for s in SEEDS:
    d = T.load_and_split(os.path.join(ROOT, "data/dft_hy_v9.csv"), s, 100)
    te = np.array(d["idx_te"]); trf = np.array([fam_of(x) for x in d["stype_tr"]]); st = d["strain_tr"]
    iso_tr = int(((np.abs(st[:, 3:]).sum(1) < 1e-4) & (np.abs(st[:, :3] - st[:, :3].mean(1, keepdims=True)).max(1) < 1e-4)).sum())
    for f in FAMS + ["total"]:
        ntr = (trf == f).sum() if f != "total" else len(trf); nte = (fam_all[te] == f).sum() if f != "total" else len(te)
        ntot = (fam_all == f).sum() if f != "total" else len(DATA)
        cnt.append(dict(seed=s, family=f, train=ntr, test=nte, val=ntot - ntr - nte, iso_train=iso_tr))
shutil.rmtree(STUB, ignore_errors=True)
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
COMP = {}   # COMP[(comparator, metric)] = [result at 45, result at 896], each the comparator minus our model
for comp in [SA, RBA, DELTA, MLP]:
    for met in KM:
        COMP[(comp, met)] = family([compare(series(FINAL, lv, met), series(comp, lv, met), delta=3.0 if met == "MAE_meV" else None) for lv in (5, 100)])
sw = np.array([r["p_sw"] for k, v in COMP.items() for r in v])
put("n_sw_reject", "%d" % (sw < 0.05).sum()); put("n_sw_total", "%d" % len(sw))

rows = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
        " & \\multicolumn{6}{c}{\\textbf{Accuracy}} & \\multicolumn{3}{c}{\\textbf{Physical validity}} \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(l){8-10}",
        "\\textbf{Model} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{P95} & \\textbf{Max} & \\textbf{$R^2$} & \\textbf{Sign} & \\textbf{DP error} & "
        "\\textbf{Shear slope} & \\textbf{Axis drift} \\\\",
        " & (meV) & (meV) & (meV) & (meV) & & (\\%) & (eV) & (eV) & (meV) \\\\", "\\midrule"]
def t2row(label, g, phys=True):
    c = ["%.1f $\\pm$ %.1f" % (g.MAE_meV.mean(), g.MAE_meV.std(ddof=1))]
    for col, fmt in [("RMSE_meV", "%.1f"), ("P95_meV", "%.1f"), ("max_meV", "%.0f"), ("R2", "%.3f"), ("sign", "%.1f")]:
        c.append("--" if col not in g or g[col].isna().all() else fmt % g[col].mean())
    for col, fmt in [("dp_err", "%.3f"), ("shear_slope", "%.2f"), ("axis", "%.1f")]:
        c.append(fmt % g[col].mean() if phys and col in g and g[col].notna().all() else "--")
    return "%s & %s \\\\" % (label, " & ".join(c))
rows.append(t2row(NAME["formula"], FORM)); rows.append("\\addlinespace")
for lv in (5, 100):
    rows.append("\\multicolumn{10}{l}{\\textit{%s}} \\\\" % ("5\\% of the training states labeled (45 calculations)" if lv == 5 else "100\\%% of the training states labeled (%s calculations)" % rng_(NLAB.loc[100, ["min", "max"]].astype(int))))
    for m in [FINAL, SA, RBA, DELTA, MLP]:
        rows.append(t2row(NAME[m], M[(M.model == m) & (M.level == lv)]))
    g = CL[(CL.model == GP) & (CL.level == lv)]
    rows.append("%s & %.1f $\\pm$ %.1f & -- & -- & -- & %.3f & -- & -- & -- & -- \\\\" % (NAME[GP], g.MAE_meV.mean(), g.MAE_meV.std(ddof=1), g.R2.mean()))
    if lv == 5: rows.append("\\addlinespace")
rows += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table2_metrics.tex"), "w").write("\n".join(rows) + "\n")

# values quoted in the text
for m, key in [(FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (DELTA, "delta"), (MLP, "mlp")]:
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
put("rmse_ratio_ours_896", "%.1f" % (mean(FINAL, 100, "RMSE_meV") / mean(FINAL, 100)))
dd = max(mean(m, lv, "dp_err") for m in (FINAL, DELTA, MLP) for lv in (5, 100)); put("dp_err_datadriven_max", "%.2f" % dd)
for m, key in [(FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (DELTA, "delta"), (MLP, "mlp")]:
    put("dp_%s_896" % key, "$%.2f$" % mean(m, 100, "dp"))
check("dp_datadriven_close", dd < 0.1); check("dp_pinn_far", min(mean(m, lv, "dp_err") for m in (SA, RBA) for lv in (5, 100)) > 0.3)
check("dp_pinns_near_analytic", all(abs(mean(m, 100, "dp") - DP_AN) < abs(mean(m, 100, "dp") - DP_REF) for m in (SA, RBA)))

# accuracy at 896
r = COMP[(MLP, "MAE_meV")][1]; register("stat_mlp_896", "MAE, MLP vs our model, 100% labeled", r, "meV")
put("diff_mlp_896", "%.1f" % r["diff"])
pen = [mean(m, 100) - mean(MLP, 100) for m in (SA, RBA)]; put("pinn_penalty_896_range", "%s to %s" % (num(min(pen)), num(max(pen))))
for m in (SA, RBA):
    rr = COMP[(m, "MAE_meV")][1]; register("stat_%s_896" % ("sa" if m == SA else "rba"), "MAE, %s vs our model, 100%% labeled" % NAME[m], rr, "meV")
r = COMP[(DELTA, "MAE_meV")][1]; put("tost_delta_896", "two one-sided tests, %s" % pval(r["p_tost"]).replace("P", "P_{\\mathrm{TOST}}"))
check("delta_equivalent_896", r["p_tost"] < 0.05)
r = COMP[(MLP, "P95_meV")][1]; register("stat_p95_mlp_896", "95th-percentile error, MLP vs our model, 100% labeled", r, "meV")
r = COMP[(MLP, "max_meV")][1]; register("stat_max_mlp_896", "maximum error, MLP vs our model, 100% labeled", r, "meV", 0)
check("max_mlp_896_ns", r["p_adj"] >= 0.05)
check("r2_networks_896", all(mean(m, 100, "R2") > 0.99 for m in (FINAL, SA, RBA, DELTA, MLP)))
# families at 896, our model vs MLP, Holm over the five families
fr = family([compare(series(FINAL, 100, "MAE_%s_meV" % f), series(MLP, 100, "MAE_%s_meV" % f)) for f in FAMS])
FR = {f: r for f, r in zip(FAMS, fr) if f != "shear"}                        # diff = MLP minus our model
better = [f for f, r in FR.items() if r["p_adj"] < 0.05 and r["diff"] < 0]       # families where the MLP is more accurate
equal = [f for f, r in FR.items() if r["p_adj"] >= 0.05]
join = lambda xs: " and ".join(xs)
put("fam_mlp_better_names", join(better))
put("fam_mlp_better_vals", " and ".join("%.1f against %.1f" % (mean(FINAL, 100, "MAE_%s_meV" % f), mean(MLP, 100, "MAE_%s_meV" % f)) for f in better) + "~meV")
put("fam_mlp_better_p", pmax([FR[f]["p_adj"] for f in better])); put("fam_mlp_better_max", "%.1f" % max(-FR[f]["diff"] for f in better))
put("fam_equal_names", join(equal)); put("fam_equal_p", "$P \\geq %.2f$" % min(FR[f]["p_adj"] for f in equal))
check("fam_nonshear_split", set(better) == {"isotropic", "biaxial"} and set(equal) == {"uniaxial", "triaxial"} and max(-FR[f]["diff"] for f in better) < 1.5)
register("stat_shear_896", "shear-family MAE, MLP vs our model, 100% labeled", fr[FAMS.index("shear")], "meV")
put("mae_shear_ours_896", "%.1f" % mean(FINAL, 100, "MAE_shear_meV")); put("mae_shear_mlp_896", "%.1f" % mean(MLP, 100, "MAE_shear_meV"))
check("shear_hardest_ours_896", max(FAMS, key=lambda f: mean(FINAL, 100, "MAE_%s_meV" % f)) == "shear")
put("fam_max_nonshear_ours_896", "%.1f" % max(mean(FINAL, 100, "MAE_%s_meV" % f) for f in FAMS if f != "shear"))
# physical validity
ps = [COMP[(c, "shear_slope")][1]["p_adj"] for c in (SA, DELTA)]
put("shear_slope_vs_pmax", pmax(ps)); put("shear_slope_eq_p", "$P \\geq %.2f$" % min(COMP[(c, "shear_slope")][1]["p_adj"] for c in (RBA, MLP)))
check("shear_slope_split", all(COMP[(c, "shear_slope")][1]["diff"] > 0 for c in (SA, DELTA)) and max(ps) < 0.05 and all(COMP[(c, "shear_slope")][1]["p_adj"] >= 0.05 for c in (RBA, MLP)))
ps = [COMP[(c, "axis")][i]["p_adj"] for c in (SA, RBA, DELTA, MLP) for i in (0, 1)]
put("axis_pmax", pmax(ps)); check("axis_lowest_both", all(COMP[(c, "axis")][i]["diff"] > 0 for c in (SA, RBA, DELTA, MLP) for i in (0, 1)) and max(ps) < 0.05)
r = COMP[(SA, "shearsign")][0]; register("stat_shearsign_sa_45", "shear-sign drift, SA-PINN vs our model, 5% labeled", r, "meV", 1, sign=-1)
check("sa_shearsign_lower_45", r["diff"] < 0 and r["p_adj"] < 0.05)
check("shearsign_lowest_896", all(COMP[(c, "shearsign")][1]["diff"] > 0 and COMP[(c, "shearsign")][1]["p_adj"] < 0.05 for c in (SA, RBA, DELTA, MLP)))
ps = [COMP[(c, "dp_err")][i]["p_adj"] for c in (SA, RBA) for i in (0, 1)]; put("dp_pinn_pmax", pmax(ps)); check("dp_pinn_sig", max(ps) < 0.05)

# ------------------------------------------------------------------ 5. scarce data
r = COMP[(MLP, "MAE_meV")][0]; register("stat_mlp_45", "MAE, MLP vs our model, 5% labeled", r, "meV")
ratio45 = mean(MLP, 5) / mean(FINAL, 5); put("ratio_mlp_45", "%.1f" % ratio45); check("ratio_mlp_45_nine", round(ratio45) == 9)
r = COMP[(MLP, "sign")][0]; register("stat_sign_mlp_45", "sign accuracy, our model vs MLP, 5% labeled", r, "percentage points", 1, sign=-1)
check("sign_ours_above_mlp_45", r["diff"] < 0 and r["p_adj"] < 0.05)
check("sign_physics_models_high", all(mean(m, lv, "sign") > 99 for m in (FINAL, SA, RBA, DELTA) for lv in (5, 100)))
x0 = series(FINAL, 0); y0 = FORM.set_index("seed").MAE_meV
r = compare(x0, y0); r["p_adj"], r["p_w_adj"] = r["p"], r["p_w"]
register("stat_formula_0", "MAE, analytic models vs our model, 0% labeled", r, "meV")
put("ours_beats_formula_0", "%d" % r["a_lower"])
# isotropic term in the runs without labels
r = compare(series(FINAL, 0), A[(A.model == FINAL + "_drop_iso_eg") & (A.level == 0)].set_index("seed").MAE_meV)
fi = family([r] + [compare(series(FINAL, lv), A[(A.model == FINAL + "_drop_iso_eg") & (A.level == lv)].set_index("seed").MAE_meV) for lv in (5, 25, 100)])
register("stat_iso_drop_0", "MAE, our model without vs with the isotropic term, 0% labeled", fi[0], "meV")
put("iso_drop_0", "%.1f" % fi[0]["diff"])

# ------------------------------------------------------------------ 6. mechanism
vsm = {m: family([compare(series(m, lv), series(MLP, lv)) for lv in LV]) for m in (SA, RBA, FINAL)}   # no physics minus model
ratios = [mean(MLP, lv) / mean(m, lv) for m in (SA, RBA) for lv in (0, 1, 2, 5, 10)]
put("pinn_ratio_range", "%.1f to %.0f" % (min(ratios), max(ratios))); check("pinns_better_low", all(vsm[m][LV.index(lv)]["diff"] > 0 and vsm[m][LV.index(lv)]["p_adj"] < 0.05 for m in (SA, RBA) for lv in (0, 1, 2, 5, 10)))
for m, key in [(SA, "sa"), (RBA, "rba")]:
    worse = [lv for lv, r in zip(LV, vsm[m]) if r["diff"] < 0 and r["p_adj"] < 0.05]
    first = min(worse); check("%s_worse_from_first" % key, worse == [lv for lv in LV if lv >= first])
    put("%s_cross" % key, "%d" % NL[first]); put("%s_cross_pct" % key, "%d" % first); put("mae_%s_cross" % key, num(mean(m, first))); put("mae_mlp_%s_cross" % key, num(mean(MLP, first)))
    register("stat_%s_cross" % key, "MAE, %s vs MLP, %s%% labeled" % (NAME[m], PCT[first]), vsm[m][LV.index(first)], "meV", 1, sign=-1)
put("ours_vs_mlp_pmax", pmax([r["p_adj"] for r in vsm[FINAL]])); check("ours_better_every_volume", all(r["diff"] > 0 and r["p_adj"] < 0.05 for r in vsm[FINAL]))


def traj(model, pct):
    g = M[(M.model == model) & (M.level == pct)].sort_values("seed")
    return np.mean(g.val_best_final.values), np.mean(g.train_at_best.values)
for m, key in [(SA, "sa"), (RBA, "rba"), (MLP, "mlp"), (FINAL, "ours")]:
    b, t = traj(m, 50); put("floor_%s" % key, "%.0f" % b); put("train_%s" % key, "%.0f" % t)
RAW["formula_family"] = {f: float(FORM["MAE_%s_meV" % f].mean()) for f in FAMS}
pen = {m: [mean(m, 50, "MAE_%s_meV" % f) - mean(MLP, 50, "MAE_%s_meV" % f) for f in FAMS] for m in (SA, RBA)}
put("pen_sa_range", "%.0f to %.0f" % (min(pen[SA]), max(pen[SA]))); put("pen_rba_range", "%.0f to %.0f" % (min(pen[RBA]), max(pen[RBA])))
check("penalty_every_family", all(v > 0 for m in (SA, RBA) for v in pen[m]))
gapf = [FORM["MAE_%s_meV" % f].mean() - mean(MLP, 50, "MAE_%s_meV" % f) for f in FAMS]
put("mlp_vs_formula_range", "%.0f to %.0f" % (min(gapf), max(gapf))); check("mlp_beats_formula_448_every_family", min(gapf) > 0)
lam = pd.read_csv(os.path.join(S, "sa_weights.csv")).assign(lam2=lambda w: w["mean"] ** 2)[["level", "term", "lam2"]]
LAM = pd.DataFrame(lam); med = LAM.groupby(["level", "term"]).lam2.median()
put("sa_lambda2_range", "%s to %s" % (num(med.min()), num(med.max()))); RAW["sa_lambda2_medians"] = {"%s_%s" % k: float(v) for k, v in med.items()}
G = pd.read_csv(os.path.join(S, "gradients_checkpoint.csv")); G = G[G.level.isin([1, 5, 25, 100])]
gm = G.groupby(["model", "level"]).agg(ratio=("ratio", "median"), cos=("cos", "mean"))
put("sa_ratio_range", "%.0f to %.0f" % (gm.loc[SA].ratio.min(), gm.loc[SA].ratio.max()))
put("rba_ratio_range", "%.0f to %.0f" % (gm.loc[RBA].ratio.min(), gm.loc[RBA].ratio.max()))
put("rba_cos_9", "$%.2f$" % gm.loc[(RBA, 1)].cos); put("rba_cos_896", "$%.2f$" % gm.loc[(RBA, 100)].cos)
check("rba_cos_negative_lowest_896", (gm.loc[RBA].cos < 0).all() and gm.loc[RBA].cos.idxmin() == 100)
put("ours_ratio_range", "%.0f to %.0f\\%%" % (100 * gm.loc[FINAL].ratio.min(), 100 * gm.loc[FINAL].ratio.max()))
put("ours_cos_min", "$%.2f$" % G[G.model == FINAL].groupby("level").cos.mean().min()); put("ours_cos_max", "$%+.2f$" % G[G.model == FINAL].groupby("level").cos.mean().max())
check("ours_cos_weak", G[G.model == FINAL].groupby("level").cos.mean().abs().max() < 0.3 and G[G.model == FINAL].groupby("level").cos.mean().abs().max() < abs(gm.loc[(RBA, 100)].cos))
put("fixed_224", "%.1f" % mean("hyb_fixed", 25)); put("falling_224", "%.1f" % mean("hyb_ld9", 25))
RAW["gradients"] = json.loads(gm.reset_index().to_json(orient="records", double_precision=15))

# ------------------------------------------------------------------ 7. ablation over every key metric (Table 3, Supplementary Table 7)
TERMS = ["drop_agreement", "drop_sym", "drop_curv", "drop_zero_strain", "drop_iso_eg"]
TNAME = {"drop_agreement": "Agreement term", "drop_sym": "Symmetry term", "drop_curv": "Curvature term", "drop_zero_strain": "Zero-strain terms", "drop_iso_eg": "Isotropic term"}
ABL = {}   # ABL[(term, metric)] = {level: result}; the difference is without minus with the term
for term in TERMS:
    lvls = sorted(A[A.model == FINAL + "_" + term].level.unique())
    for met in KM:
        res = family([compare(series(FINAL, lv, met), A[(A.model == FINAL + "_" + term) & (A.level == lv)].set_index("seed")[met],
                              delta=3.0 if met == "MAE_meV" else None) for lv in lvls])
        ABL[(term, met)] = dict(zip(lvls, res))
abl_rows = [dict(term=t, metric=m, level=NL[lv], **{k: v for k, v in r.items()}) for (t, m), d in ABL.items() for lv, r in d.items()]
pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "ablation_all.csv"), index=False)
def cell(r, dec=1):
    v = round(r["diff"], dec); txt = ("%." + str(dec) + "f") % abs(v)
    txt = txt if v == 0 else ("+" if v > 0 else "-") + txt
    return "$%s%s$" % (txt, "^{*}" if r["p_adj"] < 0.05 else "")
lv6 = [0, 1, 2, 5, 25, 100]
lines = ["\\begin{tabular}{l" + "r" * len(lv6) + "}", "\\toprule", " & \\multicolumn{%d}{c}{\\textbf{Labeled training states (\\%%)}} \\\\" % len(lv6), "\\cmidrule(l){2-%d}" % (len(lv6) + 1), "\\textbf{Term removed} & " + " & ".join("\\textbf{%s}" % PCT[p] for p in lv6) + " \\\\",
         "\\midrule", "None (complete model) & " + " & ".join("%.1f" % mean(FINAL, p) for p in lv6) + " \\\\", "\\addlinespace"]
for term in TERMS:
    lines.append("%s & %s \\\\" % (TNAME[term], " & ".join(cell(ABL[(term, "MAE_meV")][p]) if p in ABL[(term, "MAE_meV")] else "--" for p in lv6)))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table3a_ablation.tex"), "w").write("\n".join(lines) + "\n")
B_MET = [("RMSE_meV", "RMSE", "meV", 1), ("P95_meV", "P95", "meV", 1), ("max_meV", "Max", "meV", 0), ("dp_err", "DP\\\\error", "eV", 2),
         ("shear_slope", "Shear\\\\slope", "eV", 2), ("axis", "Axis\\\\drift", "meV", 1), ("shearsign", "Shear-sign\\\\drift", "meV", 1)]
def head2(h):
    return "\\begin{tabular}[b]{@{}c@{}}\\textbf{%s}\\end{tabular}" % h.replace("\\\\", "}\\\\\\textbf{")
lines = ["\\begin{tabular}{l" + "r" * len(B_MET) + "}", "\\toprule",
         "\\textbf{Term removed} & " + " & ".join(head2(h) for _, h, _, _ in B_MET) + " \\\\",
         " & " + " & ".join("(%s)" % u for _, _, u, _ in B_MET) + " \\\\", "\\midrule",
         "None (complete model) & " + " & ".join(("%." + str(dc) + "f") % mean(FINAL, 5, c) for c, _, _, dc in B_MET) + " \\\\", "\\addlinespace"]
for term in TERMS:
    lines.append("%s & %s \\\\" % (TNAME[term], " & ".join(cell(ABL[(term, c)][5], dc) for c, _, _, dc in B_MET)))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table3b_ablation.tex"), "w").write("\n".join(lines) + "\n")
# values quoted in the text
ag = ABL[("drop_agreement", "MAE_meV")]
for lv in (0, 1, 2): put("abl_agree_%d" % NL[lv], "%.1f" % ag[lv]["diff"])
put("abl_agree_pmax", pmax([ag[lv]["p_adj"] for lv in (0, 1, 2)])); check("agreement_needed_low", all(ag[lv]["p_adj"] < 0.05 and ag[lv]["diff"] > 0 for lv in (0, 1, 2)))
register("stat_abl_agree_9", "MAE, our model without vs with the agreement term, 1% labeled", ag[1], "meV")
put("mae_ours_45_int", "%.0f" % mean(FINAL, 5))
check("sign_networks_896_above_99", all(mean(m, 100, "sign") > 99 for m in (FINAL, SA, RBA, DELTA, MLP)))
hi_lv = [5, 25, 100]
put("abl_agree_high_pmin", "$P \\geq %.2f$" % min(ABL[(t, "MAE_meV")][lv]["p_adj"] for t in ("drop_agreement",) for lv in hi_lv))
check("agreement_ns_high", all(ABL[(t, "MAE_meV")][lv]["p_adj"] >= 0.05 for t in ("drop_agreement",) for lv in hi_lv))
r = ABL[("drop_agreement", "axis")][5]; put("abl_agree_axis_45", "%.1f" % -r["diff"]); register("stat_abl_agree_axis_45", "axis-relabelling drift, our model with vs without the agreement term, 5% labeled", r, "meV", 1, sign=-1)
r2 = ABL[("drop_agreement", "shearsign")][5]; put("abl_agree_shearsign_45", "%.1f" % -r2["diff"]); register("stat_abl_agree_shearsign_45", "shear-sign drift, our model with vs without the agreement term, 5% labeled", r2, "meV", 1, sign=-1)
check("agreement_cost_45", all(x["diff"] < 0 and x["p_adj"] < 0.05 for x in (r, r2)))
sy = ABL[("drop_sym", "MAE_meV")]
put("abl_sym_45", "%.1f" % sy[5]["diff"]); put("abl_sym_224", "%.1f" % sy[25]["diff"])
register("stat_abl_sym_45", "MAE, our model without vs with the symmetry term, 5% labeled", sy[5], "meV")
check("sym_worse_all_accuracy", all(ABL[("drop_sym", c)][lv]["p_adj"] < 0.05 and (ABL[("drop_sym", c)][lv]["diff"] > 0 if c != "R2" else ABL[("drop_sym", c)][lv]["diff"] < 0)
                                    for c in ("MAE_meV", "RMSE_meV", "P95_meV", "max_meV", "R2") for lv in (5, 25)))
r = ABL[("drop_sym", "axis")][100]; put("abl_sym_axis_896", "%.1f" % r["diff"]); register("stat_abl_sym_axis_896", "axis-relabelling drift, our model without vs with the symmetry term, 100% labeled", r, "meV")
check("sym_mae_ns_896", ABL[("drop_sym", "MAE_meV")][100]["p_adj"] >= 0.05)
for lv, key in [(0, "0"), (5, "45")]:
    r = ABL[("drop_iso_eg", "dp_err")][lv]; put("abl_iso_dp_%s" % key, "%.2f" % r["diff"])
register("stat_abl_iso_dp_0", "deformation-potential error, our model without vs with the isotropic term, 0% labeled", ABL[("drop_iso_eg", "dp_err")][0], "eV", 2)
check("iso_dp", all(ABL[("drop_iso_eg", "dp_err")][lv]["diff"] > 0 and ABL[("drop_iso_eg", "dp_err")][lv]["p_adj"] < 0.05 for lv in (0, 5)))
curv_sig = [(c, lv) for c in KM for lv, r in ABL[("drop_curv", c)].items() if r["p_adj"] < 0.05]
zs_sig = [(c, lv) for c in KM for lv, r in ABL[("drop_zero_strain", c)].items() if r["p_adj"] < 0.05]
RAW["curv_significant"] = curv_sig; RAW["zero_strain_significant"] = zs_sig
check("curv_no_effect", len(curv_sig) == 0)
check("zero_strain_no_effect", len(zs_sig) == 0)
ii = ABL[("drop_iso_eg", "MAE_meV")]; check("iso_mae_only_at_0", ii[0]["p_adj"] < 0.05 and all(ii[lv]["p_adj"] >= 0.05 for lv in (5, 25, 100)))

# ------------------------------------------------------------------ 8. calibration cost and label-matched constants
L = pd.read_csv(os.path.join(S, "runs_labelmatched.csv"))
lmres = family([compare(series(FINAL, lv), L[(L.model == FINAL) & (L.level == lv)].set_index("seed").MAE_meV) for lv in (1, 2, 5, 10)])
for lv, r in zip((1, 2, 5, 10), lmres): put("lm_%d" % NL[lv], "%.1f" % r["diff"])
register("stat_lm_18", "MAE, our model with label-matched vs training-portion constants, 2% labeled", lmres[1], "meV")
register("stat_lm_45", "MAE, our model with label-matched vs training-portion constants, 5% labeled", lmres[2], "meV")
register("stat_lm_90", "MAE, our model with label-matched vs training-portion constants, 10% labeled", lmres[3], "meV")
check("lm_cost_18_45_90", all(lmres[i]["p_adj"] < 0.05 and lmres[i]["diff"] > 0 for i in (1, 2, 3)))
put("lm_9_median", "%.0f" % L[(L.model == FINAL) & (L.level == 1)].MAE_meV.median())
lmr = [mean(MLP, lv) / L[(L.model == FINAL) & (L.level == lv)].MAE_meV.mean() for lv in (2, 5, 10)]
put("lm_ratio_range", "%.1f to %.1f" % (min(lmr), max(lmr)))
LMT = []
for m in [FINAL, DELTA, "sdp_net"]:
    res = family([compare(series(m, lv), L[(L.model == m) & (L.level == lv)].set_index("seed").MAE_meV) for lv in (1, 2, 5, 10)])
    for lv, r in zip((1, 2, 5, 10), res):
        LMT.append(dict(model=m, level=lv, lm=r["mean_b"], pool=r["mean_a"], diff=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p_adj"], nophys=mean(MLP, lv)))
LMT = pd.DataFrame(LMT); LMT.to_csv(os.path.join(OUT, "labelmatched.csv"), index=False)
CL1 = pd.read_csv(os.path.join(S, "calibration_labelmatched.csv"))
cal = []
for p in (1, 2, 5, 10):
    v = CL1[CL1.level == p].sort_values("seed").MAE_meV.values
    cal.append(dict(n=NL[p], mean=v.mean(), median=np.median(v), q1=np.percentile(v, 25), q3=np.percentile(v, 75), failed=int((v > 1000).sum())))
CAL = pd.DataFrame(cal); CAL.to_csv(os.path.join(OUT, "calibration_cost.csv"), index=False)
for _, c in CAL.iterrows(): put("cal_%d_mean" % c.n, num(c["mean"])); put("cal_%d_median" % c.n, num(c["median"]))
put("cal_9_failed", "%d" % CAL[CAL.n == 9].failed.iloc[0]); check("cal_9_one_failure", CAL[CAL.n == 9].failed.iloc[0] == 1)
check("cal_saturates_90", abs(CAL[CAL.n == 90]["median"].iloc[0] - FORM.MAE_meV.median()) < 2)

# ------------------------------------------------------------------ 9. beyond the training data
HS = pd.read_csv(os.path.join(S, "heldout_runs.csv"))
check("heldout_runs_complete", (HS.model == FINAL).sum() == 136)
hs = HS.groupby(["model", "level"]).agg(mae=("MAE_meV", "mean"), sd=("MAE_meV", "std"), r2=("R2", "mean"), n=("seed", "count")).reset_index()
hs.to_csv(os.path.join(OUT, "heldout.csv"), index=False)
H = lambda m, l: float(hs[(hs.model == m) & (hs.level == l)].mae.iloc[0])
for m, key in [("formula", "formula"), (FINAL, "ours"), (SA, "sa"), (RBA, "rba"), (MLP, "mlp"), (DELTA, "delta")]:
    for l, lk in [("magnitude_03", "03"), ("magnitude_05", "05"), ("magnitude_07", "07"), ("group_triaxial", "tri"), ("group_isotropic", "iso"),
                  ("group_uniaxial", "uni"), ("group_biaxial", "bi")]:
        put("ho_%s_%s" % (key, lk), ("%.0f" % H(m, l)) if lk in ("03", "05", "07", "tri") else ("%.1f" % H(m, l)))   # one precision per sentence
put("ho_ours_05_r2", "%.2f" % hs[(hs.model == FINAL) & (hs.level == "magnitude_05")].r2.iloc[0])
rr = [H("formula", l) / H(FINAL, l) for l in ("group_isotropic", "group_uniaxial", "group_biaxial")]
put("ho_ratio_range", "%.0f to %.0f" % (min(rr), max(rr)))
sh = [H(m, "group_shear") for m in hs[hs.level == "group_shear"].model.unique()]
put("ho_shear_range", "%.2f to %.2f" % (min(sh) / 1000, max(sh) / 1000))
check("ho_pinns_worse_than_formula_05", H(SA, "magnitude_05") > H("formula", "magnitude_05") and H(RBA, "magnitude_05") > H("formula", "magnitude_05"))
nz = M[(M.model == FINAL) & M.level.isin([5, 100])].noise.abs().max() * 100
put("noise_max", "%d" % math.ceil(nz)); RAW["noise_max_pct"] = float(nz)

# ------------------------------------------------------------------ 10. Table 4: constants of the analytic models
CON = pd.DataFrame([json.load(open(os.path.join(ROOT, "results", "98_constants_v2", "symmetry", "const_seed%d.json" % s))) for s in SEEDS])
SYM = [("CBM_0", "$E^{0}_{\\mathrm{CBM}}$", "(\\ref{eq:cbm})", "offset"), ("XI_D", "$\\Xi_d$", "(\\ref{eq:cbm})", "linear"),
       ("XI_U", "$\\Xi_u$", "(\\ref{eq:cbm})", "linear"), ("XI_UP", "$\\Xi_u'$", "(\\ref{eq:cbm})", "linear"),
       ("XI_Q", "$\\Xi_Q$", "(\\ref{eq:cbm})", "quadratic"), ("XI_C", "$\\Xi_C$", "(\\ref{eq:cbm})", "quadratic"),
       ("XI_Q2", "$\\Xi_{Q2}$", "(\\ref{eq:cbm})", "quadratic"), ("VBM_0", "$E^{0}_{\\mathrm{VBM}}$", "(\\ref{eq:vbm})", "offset"),
       ("AV", "$a_v$", "(\\ref{eq:vbm})", "linear"), ("AV2", "$a_{v2}$", "(\\ref{eq:vbm})", "quadratic"), ("B_BP", "$b$", "(\\ref{eq:vbm})", "linear"),
       ("B2_BP", "$b_2$", "(\\ref{eq:vbm})", "quadratic"), ("D_BP", "$d$", "(\\ref{eq:vbm})", "linear")]
lines = ["\\begin{tabular}{lllrr}", "\\toprule", "\\textbf{Constant} & \\textbf{Equation} & \\textbf{Order in strain} & \\textbf{Mean (eV)} & \\textbf{s.d. (eV)} \\\\", "\\midrule"]
for k, sym, eq, kind in SYM:
    lines.append("%s & %s & %s & $%.3f$ & %.3f \\\\" % (sym, eq, kind, CON[k].mean(), CON[k].std(ddof=1)))
lines += ["\\bottomrule", "\\end{tabular}"]; open(os.path.join(OUT, "table4_constants.tex"), "w").write("\n".join(lines) + "\n")
lin = ["XI_D", "XI_U", "XI_UP", "AV", "B_BP", "D_BP"]; put("const_linear_sd_max", "%.2f" % max(CON[k].std(ddof=1) for k in lin))
put("const_linear_range_max", "%.2f" % max(CON[k].max() - CON[k].min() for k in lin))

# ------------------------------------------------------------------ 11. cost
inf = open(os.path.join(S, "inference_times.txt")).read()
put("inference_ms", "%.2f" % float(re.search(r"single state ([0-9.]+) ms", inf).group(1)))
cost = pd.read_csv(os.path.join(S, "cost.csv")).set_index("item").value
put("min_ours", "%.0f" % cost["train_minutes_" + FINAL]); put("min_sa", "%.0f" % cost["train_minutes_" + SA]); put("min_rba", "%.0f" % cost["train_minutes_" + RBA])

# ------------------------------------------------------------------ 12. Supplementary tables
def si_table(name, header, rows, colspec):
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(["\\begin{tabular}{%s}" % colspec, "\\toprule", header + " \\\\", "\\midrule"] + [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]) + "\n")
# S1: every model at every data volume (two stacked panels)
allm = pd.concat([M[["model", "level", "seed", "MAE_meV"]], CL[["model", "level", "seed", "MAE_meV"]]], ignore_index=True)
s1 = allm.groupby(["model", "level"]).MAE_meV.agg(["mean", "std"])
ORDER = [(FINAL, "Our model"), ("hyb_sdp_ld45", "Variant, weight falling from 45"), ("hyb_ld9", "Variant on $\\Delta$-learning, weight falling from 9"),
         ("hyb_ld45", "Variant on $\\Delta$-learning, weight falling from 45"), ("hyb_ld45_col", "Variant on $\\Delta$-learning, falling from 45, collocation"),
         ("hyb_fixed", "Variant on $\\Delta$-learning, fixed weight"), (SA, "SA-PINN"), (RBA, "RBA-PINN"), (DELTA, "$\\Delta$-learning"),
         ("delta_zs", "$\\Delta$-learning, zero-strain terms"), ("delta_zs_sym", "$\\Delta$-learning, zero-strain and symmetry terms"),
         ("sdp_net", "Our model without the agreement term"), (MLP, "MLP"), (GP, "Gaussian process"),
         ("RandomForest", "Random forest"), ("GBDT", "Gradient boosting")]
def s1panel(levels):
    out = ["\\begin{tabular}{l" + "r" * len(levels) + "}", "\\toprule", " & \\multicolumn{%d}{c}{\\textbf{Labeled training states (\\%%)}} \\\\" % len(levels), "\\cmidrule(l){2-%d}" % (len(levels) + 1), "\\textbf{Model} & " + " & ".join("\\textbf{%s}" % PCT[p] for p in levels) + " \\\\", "\\midrule"]
    for m, lab in ORDER:
        out.append("%s & %s \\\\" % (lab, " & ".join(("%.1f $\\pm$ %.1f" % tuple(s1.loc[(m, p)])) if (m, p) in s1.index else "--" for p in levels)))
    return "\n".join(out + ["\\bottomrule", "\\end{tabular}"]) + "\n"
open(os.path.join(OUT, "si_all_models.tex"), "w").write(s1panel([0, 1, 2, 5, 10]) + "\n\\vspace{3mm}\n" + s1panel([25, 50, 75, 100]))
# S2: held-out suite
SCH = [("group_isotropic", "Isotropic"), ("group_uniaxial", "Uniaxial"), ("group_biaxial", "Biaxial"), ("group_triaxial", "Triaxial"), ("group_shear", "Shear"),
       ("magnitude_03", "$>0.03$"), ("magnitude_05", "$>0.05$"), ("magnitude_07", "$>0.07$")]
rows = []
for m, lab in [("formula", "Analytic models alone"), (FINAL, "Our model"), (SA, "SA-PINN"), (RBA, "RBA-PINN"), (DELTA, "$\\Delta$-learning"),
               ("delta_zs", "$\\Delta$-learning, zero-strain terms"), ("delta_zs_sym", "$\\Delta$-learning, zero-strain and symmetry terms"),
               ("sdp_net", "Our model without the agreement term"), (MLP, "MLP")]:
    cells = []
    for l, _ in SCH:
        g = hs[(hs.model == m) & (hs.level == l)]
        cells.append(("%.0f $\\pm$ %.0f" % (g.mae.iloc[0], g.sd.iloc[0])) if len(g) else "--")
    rows.append("%s & %s" % (lab, " & ".join(cells)))
si_table("si_heldout", "\\textbf{Model} & " + " & ".join("\\textbf{%s}" % h for _, h in SCH), rows, "l" + "r" * len(SCH))
# S3: label-matched constants
rows = []
for _, r in LMT.iterrows():
    lab = {FINAL: "Our model", DELTA: "$\\Delta$-learning", "sdp_net": "Our model without the agreement term"}[r.model]
    rows.append("%s & %s & %.1f & %.1f & $%+.1f$ & %s & %.1f" % (lab, PCT[r.level], r.lm, r.pool, r["diff"], ("$<$0.001" if r.p < 0.001 else "%.2g" % r.p), r.nophys))
for _, c in CAL.iterrows():
    rows.append("Analytic models alone & %s & %.1f (median %.1f) & %.1f & -- & -- & --" % ({9: "1", 18: "2", 45: "5", 90: "10"}[int(c.n)], c["mean"], c["median"], FORM.MAE_meV.mean()))
si_table("si_labelmatched", "\\textbf{Model} & %s & %s & %s & \\textbf{Change} & \\textbf{$P$} & \\textbf{MLP}" % (head2("Labeled\\\\(\\%)"), head2("Label-matched\\\\constants"), head2("Training-portion\\\\constants")), rows, "lrrrrrr")
# S4: conventional regressors
rows = []
for m, lab in [(GP, "Gaussian process"), ("RandomForest", "Random forest"), ("GBDT", "Gradient boosting")]:
    g = CL[CL.model == m].groupby("level").MAE_meV.agg(["mean", "std"])
    rows.append("%s & %s" % (lab, " & ".join(("%.0f $\\pm$ %.0f" % tuple(g.loc[p])) if p in g.index else "--" for p in LV[1:])))
si_table("si_classical", "\\textbf{Model} & " + " & ".join("\\textbf{%s\\%%}" % PCT[p] for p in LV[1:]), rows, "l" + "r" * 8)
# S5: selection of the weighting rule
HC = pd.read_csv(os.path.join(S, "hybrid_choice.csv"), index_col=0); vcols = [c for c in HC.columns if c.startswith("val_mae_")]
VAR = {"hyb_sdp_ld9": "Weight falling from 9 (selected)", "hyb_sdp_ld45": "Weight falling from 45", "hyb_ld9": "On $\\Delta$-learning, falling from 9",
       "hyb_ld45": "On $\\Delta$-learning, falling from 45", "hyb_ld45_col": "On $\\Delta$-learning, falling from 45, collocation", "hyb_fixed": "On $\\Delta$-learning, fixed weight"}
rows = ["%s & %s & %.1f & %.2f & %.1f" % (VAR.get(m, m), " & ".join("%.1f" % HC.loc[m, c] for c in vcols), HC.loc[m, "mean_over_levels"], HC.loc[m, "mean_rank_over_levels"], HC.loc[m, "geomean_over_levels"]) for m in HC.index]
si_table("si_selection", "\\textbf{Variant} & " + " & ".join("\\textbf{%s\\%%}" % c.split("_")[-1][:-3] for c in vcols) + " & \\textbf{Mean} & \\textbf{Rank} & \\textbf{Geo.\\ mean}", rows, "l" + "r" * (len(vcols) + 3))
check("selected_lowest_mean", HC.mean_over_levels.idxmin() == "hyb_sdp_ld9")
# S6: gradients
fr_ = lambda x: ("%.0f" % x) if x >= 10 else ("%.1f" % x if x >= 1 else "%.2f" % x)
rows = ["%s & %s" % (NAME[m], " & ".join("%s & $%+.2f$" % (fr_(gm.loc[(m, p)].ratio), gm.loc[(m, p)].cos) for p in (1, 5, 25, 100))) for m in (SA, RBA, FINAL)]
si_table("si_gradients", "\\textbf{Model} & " + " & ".join("\\textbf{%s\\%%: ratio} & \\textbf{cosine}" % PCT[p] for p in (1, 5, 25, 100)), rows, "l" + "rr" * 4)
# S7: ablation, every metric and data volume (difference with 95% CI and Holm-adjusted P)
rows = []
for term in TERMS:
    rows.append("\\multicolumn{%d}{l}{\\textit{%s removed}}" % (len(lv6) + 1, TNAME[term]))
    for met, lab, unit, dec in [("MAE_meV", "MAE", "meV", 1)] + B_MET:
        cells = []
        for p in lv6:
            r = ABL[(term, met)].get(p)
            cells.append("--" if r is None else "%s (%s)" % (cell(r, dec).replace("^{*}", ""), "$<$0.001" if r["p_adj"] < 0.001 else "%.2g" % r["p_adj"]))
        rows.append("\\quad %s (%s) & %s" % (lab.replace("\\\\", " "), unit, " & ".join(cells)))
    rows.append("\\addlinespace")
open(os.path.join(OUT, "si_ablation_rows.tex"), "w").write("\n".join(r + " \\\\" if not r.startswith("\\addlinespace") else r for r in rows[:-1]) + "\n")

# S9 and S10: preliminary studies that fixed the settings (stratified random division, three runs per variant)
RP = os.path.join(S, "preliminary")   # made in September 2026 with the code tagged thesis-2026-09
cap = pd.read_csv(os.path.join(RP, "capacity_decision.csv"))
cap["depth"] = cap.variant.str.extract(r"d(\d+)")[0].astype(int); cap["width"] = cap.variant.str.extract(r"w(\d+)")[0].astype(int)
pm, ps_, pn = (cap.pivot(index="depth", columns="width", values=v) for v in ("test_mae_meV_mean", "test_mae_meV_std", "n_params"))
rows = ["%d & %s" % (d, " & ".join("%.1f $\\pm$ %.1f" % (pm.loc[d, w], ps_.loc[d, w]) for w in pm.columns)) for d in pm.index]
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
el = pd.read_csv(os.path.join(RP, "elements_decision.csv"))
for v in ["ref", "widen", "layernorm", "residual", "taper", "residual_norm", "dropout_005", "dropout_010", "dropout_020"]:
    r = el[el.variant == v].iloc[0]; rows.append("%s & %s & %s & %d" % ("Architecture" if v == "ref" else "", LAB[v], fmt(r), r.n_seeds))
rows.append("\\addlinespace")
op = pd.read_csv(os.path.join(RP, "optimizer_decision.csv")); op = op[op.group != "sa_clamp"]   # the bound on SA weights concerns a model not in this paper
first = True
for grp, order in [("arms", ["two_phase", "soap_only", "adamw_only"]), ("switch", ["switch_050", "switch_025", "switch_075"]), ("clip", ["clip_10", "clip_off", "clip_50", "clip_05", "clip_01"])]:
    for v in order:
        r = op[op.variant == v].iloc[0]; rows.append("%s & %s & %s & %d" % ("Optimizer" if first else "", LAB[v], fmt(r), r.n_seeds)); first = False
rows.append("\\addlinespace")
ep = pd.read_csv(os.path.join(RP, "epoch_decision.csv")); first = True
for _, r in ep.iterrows():
    it = int(str(r.budget).replace("ep", "")); bind = str(r.budget_binding).replace("%", "\\%")
    rows.append("%s & %s iterations; final checkpoint retained in %s of runs%s & %s & %d" % ("Training length" if first else "", "{:,}".format(it).replace(",", "{,}"), bind,
                " (reference)" if it == 6000 else "", fmt(r), r.n_seeds)); first = False
pa = pd.read_csv(os.path.join(RP, "patience_decision.csv")); first = True
for _, r in pa.iterrows():
    rows.append("%s & Early stopping, patience %s (stopped after %s iterations) & %s & %d" % ("" , r.variant[2:], "{:,}".format(int(r.epochs_used)).replace(",", "{,}"), fmt(r), r.n_seeds))
rows.append("\\addlinespace")
sp = pd.read_csv(os.path.join(RP, "split_decision.csv")); first = True
for v, lab in [("stratified", "Reference: stratified by family"), ("random", "Random"), ("kfold", "Five-fold cross-validation")]:
    r = sp[sp.scheme == v].iloc[0]; rows.append("%s & %s & %s & %d" % ("Data division" if first else "", lab, fmt(r), r.n_seeds)); first = False
si_table("si_settings", "\\textbf{Study} & \\textbf{Variant} & \\textbf{Test MAE (meV)} & \\textbf{Runs}", rows, "llrr")

# ------------------------------------------------------------------ 13. Supplementary Table 8: every test quoted in the text, and output
TT = pd.DataFrame(TESTS); TT.to_csv(os.path.join(OUT, "tests.csv"), index=False)
check("wilcoxon_agrees_all", TT.agree.all())
rows = []
for _, r in TT.iterrows():
    unit = "" if r.unit in ("", "percentage points") else "~" + r.unit
    dc = 2 if r.unit == "eV" else 1
    pf = lambda q: "$<$0.001" if q < 0.001 else ("%.3f" % q if 0.995 <= q < 1 else "%.2g" % q)   # 0.9966 must not print as 1
    rows.append("%s & %.2f & %s & %.2f & $%.*f$ to $%.*f$%s & %s & %s" % ((r.what[0].upper() + r.what[1:]).replace("%", "\\%"), r.t, pf(r.p), r.d, dc, r.lo, dc, r.hi, unit,
                                                                   pf(r.p_w), pf(r.p_sw)))
si_table("si_tests", "\\textbf{Comparison} & \\textbf{$t(16)$} & \\textbf{$P$} & \\textbf{$d$} & \\textbf{95\\% CI} & \\textbf{Wilcoxon $P$} & \\textbf{Shapiro--Wilk $P$}", rows,
         "p{6.2cm}rrrrrr")
json.dump(dict(txt=TXT, raw=RAW, checks=CHECKS, tests=TESTS), open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
bad = [k for k, v in CHECKS.items() if not v]
print("wrote %d text values, %d tests, %d checks (%d failed: %s)" % (len(TXT), len(TESTS), len(CHECKS), len(bad), ", ".join(bad) or "none"))
