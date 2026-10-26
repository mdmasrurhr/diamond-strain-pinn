"""Stage 3 of the paper's analysis: Figures 2-5 and Supplementary Figures 1-3 (Figure 1 is TikZ: analysis/fig1_workflow.tex).

Reads summary/ and outputs/tables/numbers.json (run make_tables.py first) and writes outputs/figures/ (PDF) and
outputs/figures/preview/ (PNG). Style follows the journal: Arial-type lettering (Liberation Sans, a TrueType font that
embeds cleanly; DejaVu Sans if it is not installed) at 8 pt or more at the printed size, no titles inside panels, bold
lowercase panel letters, colour plus a marker or hatch pattern for every model, and individual runs shown wherever bars
are drawn. Usage: python analysis/make_figures.py"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
S = os.path.join(ROOT, "summary"); TAB = os.path.join(ROOT, "outputs", "tables")
OUT = os.path.join(ROOT, "outputs", "figures"); OUTSI = OUT; PREV = os.path.join(OUT, "preview")
for d in (OUT, PREV): os.makedirs(d, exist_ok=True)
W = 6.5
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.75, "xtick.major.width": 0.75, "ytick.major.width": 0.75, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
                     "lines.linewidth": 1.3, "hatch.linewidth": 0.6, "savefig.dpi": 600, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
FINAL, SA, RBA, DELTA, MLP, GP = "hyb_sdp_ld9", "sa_pinn", "rba_pinn", "delta", "mlp", "GaussianProcess"
STY = {FINAL: dict(c="#009E73", m="o", ls="-", h="", lab="Our model"),
       SA: dict(c="#0072B2", m="s", ls="-", h="////", lab="SA-PINN"),
       RBA: dict(c="#56B4E9", m="^", ls="-", h="\\\\\\\\", lab="RBA-PINN"),
       DELTA: dict(c="#CC79A7", m="D", ls="--", h="xxx", lab="$\\Delta$-learning"),
       MLP: dict(c="#D55E00", m="v", ls=":", h="....", lab="MLP"),
       GP: dict(c="#E69F00", m="X", ls="-.", h="", lab="Gaussian process"),
       "formula": dict(c="#000000", m="", ls="-.", h="++", lab="Analytic models alone")}
FAMS = ["isotropic", "uniaxial", "biaxial", "triaxial", "shear"]
NL = {0: 0, 1: 9, 2: 18, 5: 45, 10: 90, 25: 224, 50: 448, 75: 672, 100: 896}
M = pd.read_csv(os.path.join(S, "runs_main.csv")); FORM = pd.read_csv(os.path.join(S, "runs_analytic.csv"))
N = json.load(open(os.path.join(TAB, "numbers.json")))
rng = np.random.default_rng(7)


def letter(ax, s, x=-0.15, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def plain_log(axis, ticks):
    axis.set_major_locator(mt.FixedLocator(ticks)); axis.set_major_formatter(mt.FixedFormatter(["%g" % t for t in ticks]))
    axis.set_minor_formatter(mt.NullFormatter())


def save(fig, name, si=False):
    fig.savefig(os.path.join(OUTSI if si else OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(PREV, name + ".png"), bbox_inches="tight", dpi=150); plt.close(fig); print(name)


def bars(ax, x, values, model, width, face=None):
    """A bar at the mean with a hatch pattern, plus the individual runs as dots."""
    st = STY[model]; fc = st["c"] if face is None else face
    ax.bar(x, np.mean(values), width, color=fc, edgecolor="white" if model != "formula" else "black", hatch=st["h"], linewidth=0.0 if model != "formula" else 0.6, zorder=2)
    ax.scatter(x + rng.uniform(-0.28, 0.28, len(values)) * width, values, s=3.5, color="black", alpha=0.55, linewidths=0, zorder=4)


def legend_handles(models, kind="bar"):
    out = []
    for m in models:
        st = STY[m]
        if kind == "bar":
            out.append(Patch(facecolor=st["c"] if m != "formula" else "white", edgecolor="white" if m != "formula" else "black", hatch=st["h"], label=st["lab"]))
        else:
            out.append(Line2D([], [], color=st["c"], marker=st["m"], ls=st["ls"], ms=4, lw=1.2, label=st["lab"]))
    return out


# ------------------------------------------------------------------ Figure 2: accuracy against data; calibration cost
X0 = 0.42   # position of the zero-label point on the logarithmic axis (labeled fraction in percent)
xpos = lambda lv: X0 if lv == 0 else lv
fig, ax = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"width_ratios": [1.55, 1]})
a = ax[0]
CL = pd.read_csv(os.path.join(S, "classical_runs.csv")).rename(columns={"dft_pct": "level", "test_mae_meV": "MAE_meV"})
for m in [GP, MLP, DELTA, SA, RBA, FINAL]:
    src = CL[CL.model == GP] if m == GP else M[M.model == m]
    g = src.groupby("level").MAE_meV; mu = g.mean(); ci = 1.96 * g.std() / np.sqrt(g.count())
    xs = np.array([xpos(l) for l in mu.index]); st = STY[m]
    a.errorbar(xs, mu.values, yerr=ci.values, color=st["c"], marker=st["m"], ls=st["ls"], ms=3.8, lw=1.2, capsize=1.8, elinewidth=0.8, label=st["lab"], zorder=3 if m == FINAL else 2)
a.axhline(FORM.MAE_meV.mean(), color="black", lw=0.9, ls="-.", zorder=1)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.34, 128); a.set_ylim(4, 2000)
a.xaxis.set_major_locator(mt.FixedLocator([X0, 1, 2, 5, 10, 25, 50, 100])); a.xaxis.set_major_formatter(mt.FixedFormatter(["0", "1", "2", "5", "10", "25", "50", "100"]))
a.xaxis.set_minor_locator(mt.NullLocator()); plain_log(a.yaxis, [5, 10, 30, 100, 300, 1000])
for xx in (0.56, 0.63):   # break marks between the zero-label point and the logarithmic axis
    a.plot([xx * 0.97, xx * 1.03], [3.75, 4.25], color="black", lw=0.75, clip_on=False, transform=a.transData)
a.set_xlabel("Labeled training states (%)"); a.set_ylabel("Bandgap MAE (meV)"); letter(a, "a", x=-0.13)
CAL = pd.read_csv(os.path.join(TAB, "calibration_cost.csv"))
b = ax[1]
CAL["pct"] = CAL.n.map({9: 1, 18: 2, 45: 5, 90: 10})
b.errorbar(CAL.pct, CAL["median"], yerr=[CAL["median"] - CAL.q1, CAL.q3 - CAL["median"]], color="black", marker="o", ms=3.8, lw=1.2, capsize=2, elinewidth=0.8)
b.axhline(FORM.MAE_meV.median(), color="black", ls="--", lw=0.9)
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(0.75, 14); plain_log(b.xaxis, [1, 2, 5, 10]); plain_log(b.yaxis, [40, 50, 60, 80, 100, 150])
b.set_ylim(40, 160); b.set_xlabel("Training states used to fit the constants (%)"); b.set_ylabel("MAE of the analytic models (meV)"); letter(b, "b", x=-0.2)
h = legend_handles([FINAL, SA, RBA, DELTA, MLP, GP], "line") + [Line2D([], [], color="black", ls="-.", lw=0.9, label="Analytic models alone"),
                                                            Line2D([], [], color="black", ls="--", lw=0.9, label="Analytic models, whole training portion")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03), handlelength=2.2, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.2); save(fig, "fig2_accuracy")

# ------------------------------------------------------------------ Figure 3: error by family
mods = [FINAL, SA, RBA, MLP]; wd = 0.19
fig, ax = plt.subplots(1, 2, figsize=(W, 3.0))
for a, lv, lab in zip(ax, (5, 100), ("a", "b")):
    for j, m in enumerate(mods):
        for i, f in enumerate(FAMS):
            bars(a, i + (j - 1.5) * wd, M[(M.model == m) & (M.level == lv)]["MAE_%s_meV" % f].values, m, wd * 0.92)
    a.set_yscale("log"); a.set_xticks(range(5)); a.set_xticklabels(FAMS, rotation=25, ha="right", rotation_mode="anchor")
    a.set_ylabel("Bandgap MAE, %d%% labeled (meV)" % lv); letter(a, lab, x=-0.17)
plain_log(ax[0].yaxis, [1, 10, 100, 1000]); ax[0].set_ylim(0.6, 2500); plain_log(ax[1].yaxis, [0.01, 0.1, 1, 10, 100]); ax[1].set_ylim(0.004, 200)
fig.legend(handles=legend_handles(mods), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.6)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.5); save(fig, "fig3_family")

# ------------------------------------------------------------------ Figure 4: mechanism
fig, ax = plt.subplots(2, 2, figsize=(W, 5.0)); ax = ax.ravel()
CUR = pd.read_csv(os.path.join(S, "curves.csv"))
def trajectory(model, pct):
    c = CUR[(CUR.model == model) & (CUR.level == pct)].dropna(subset=["val_best_mean_complete"])
    return c.epoch.values, c.val_best_mean_complete.values
for m in [SA, RBA, MLP, FINAL]:
    e, mu = trajectory(m, 50); ax[0].plot(e, mu, color=STY[m]["c"], ls=STY[m]["ls"], lw=1.3)
ax[0].set_yscale("log"); ax[0].set_xlim(0, 24000); ax[0].set_xticks([0, 6000, 12000, 18000, 24000]); ax[0].set_xticklabels(["0", "6,000", "12,000", "18,000", "24,000"])
plain_log(ax[0].yaxis, [5, 10, 20, 50, 100, 200]); ax[0].set_xlabel("Training iteration"); ax[0].set_ylabel("Lowest validation MAE (meV)"); letter(ax[0], "a", x=-0.2)
wd = 0.34
for j, m in enumerate([SA, RBA]):
    for i, f in enumerate(FAMS):
        a_ = M[(M.model == m) & (M.level == 50)].set_index("seed")["MAE_%s_meV" % f]; b_ = M[(M.model == MLP) & (M.level == 50)].set_index("seed")["MAE_%s_meV" % f]
        d = (a_ - b_).dropna().values; x = i + (j - 0.5) * wd
        ax[1].scatter(x + rng.uniform(-0.09, 0.09, len(d)), d, s=6, color=STY[m]["c"], marker=STY[m]["m"], alpha=0.8, linewidths=0, zorder=3)
        ax[1].plot([x - 0.12, x + 0.12], [d.mean()] * 2, color="black", lw=1.4, zorder=4)
ax[1].axhline(0, color="black", lw=0.75); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(FAMS, rotation=25, ha="right", rotation_mode="anchor")
ax[1].set_ylabel("MAE difference to the MLP (meV)"); letter(ax[1], "b", x=-0.24)
G = pd.read_csv(os.path.join(S, "gradients_checkpoint.csv")); wd = 0.26
for j, m in enumerate([SA, RBA, FINAL]):
    for i, lv in enumerate([1, 5, 25, 100]):
        v = G[(G.model == m) & (G.level == lv)].ratio.values; x = i + (j - 1) * wd
        ax[2].scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=STY[m]["c"], marker=STY[m]["m"], alpha=0.8, linewidths=0, zorder=3)
        ax[2].plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color="black", lw=1.4, zorder=4)
ax[2].axhline(1, color="black", lw=0.75, ls="--"); ax[2].set_yscale("log"); plain_log(ax[2].yaxis, [0.01, 0.1, 1, 10, 100, 1000])
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["1", "5", "25", "100"]); ax[2].set_xlabel("Labeled training states (%)")
ax[2].set_ylabel("Physics gradient / data gradient"); letter(ax[2], "c", x=-0.2)
DPR, DPA = N["raw"]["iso_reference"]["dp"], N["raw"]["dp_analytic"]
dm = [FINAL, DELTA, MLP, SA, RBA]
for i, m in enumerate(dm):
    v = M[(M.model == m) & (M.level == 100)].dp.values
    ax[3].scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=STY[m]["c"], marker=STY[m]["m"], alpha=0.85, linewidths=0, zorder=3)
    ax[3].plot([i - 0.2, i + 0.2], [v.mean()] * 2, color="black", lw=1.4, zorder=4)
ax[3].axhline(DPR, color="black", lw=0.9, ls="-"); ax[3].axhline(DPA, color="black", lw=0.9, ls="--")
ax[3].set_xlim(-0.5, 4.5); ax[3].set_xticks(range(len(dm))); ax[3].set_xticklabels([STY[m]["lab"] for m in dm], rotation=25, ha="right", rotation_mode="anchor")
ax[3].set_ylabel("d$E_g$/d$I_1$ at zero strain (eV)"); letter(ax[3], "d", x=-0.2)
h4 = legend_handles([FINAL, SA, RBA, DELTA, MLP], "line") + [Line2D([], [], color="black", lw=0.9, ls="-", label="Reference calculations (d)"),
                                                       Line2D([], [], color="black", lw=0.9, ls="--", label="Analytic models (d)")]
fig.legend(handles=h4, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.6, w_pad=2.2); save(fig, "fig4_mechanism")

# ------------------------------------------------------------------ Figure 5: predictions within and beyond the sampled deformations
FCOL = {"isotropic": "#332288", "uniaxial": "#CC6677", "biaxial": "#117733", "triaxial": "#882255", "shear": "#44AA99"}
FMK = {"isotropic": "o", "uniaxial": "s", "biaxial": "D", "triaxial": "^", "shear": "v"}
fig, ax = plt.subplots(1, 2, figsize=(W, 3.1), gridspec_kw={"width_ratios": [1, 1.55]})
tr = pd.read_csv(os.path.join(S, "parity_median_run.csv")); fam = tr.strain_type.str.split("_").str[0].replace({"unstrained": "isotropic"})
for f in FAMS:
    g = tr[fam == f]; ax[0].scatter(g.Eg_true_eV, g.Eg_pred_eV, s=9, marker=FMK[f], color=FCOL[f], linewidths=0, alpha=0.9, label=f, zorder=3)
lo, hi = tr.Eg_true_eV.min() - 0.15, tr.Eg_true_eV.max() + 0.15
ax[0].plot([lo, hi], [lo, hi], color="black", lw=0.75, ls="--", zorder=1); ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi); ax[0].set_aspect("equal")
ax[0].set_xlabel("Reference bandgap (eV)"); ax[0].set_ylabel("Predicted bandgap (eV)")
ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, handletextpad=0.2, columnspacing=0.8, markerscale=1.6)
letter(ax[0], "a", x=-0.22)
HS = pd.read_csv(os.path.join(S, "heldout_runs.csv"))
SCH = [("group_isotropic", "isotropic"), ("group_uniaxial", "uniaxial"), ("group_biaxial", "biaxial"), ("group_triaxial", "triaxial"), ("group_shear", "shear"),
       ("magnitude_03", "0.03"), ("magnitude_05", "0.05"), ("magnitude_07", "0.07")]
hm = ["formula", FINAL, SA, RBA, MLP]; wd = 0.16
for j, m in enumerate(hm):
    for i, (s, _) in enumerate(SCH):
        bars(ax[1], i + (j - 2) * wd, HS[(HS.level == s) & (HS.model == m)].MAE_meV.values, m, wd * 0.92, face="white" if m == "formula" else None)
ax[1].set_yscale("log"); plain_log(ax[1].yaxis, [1, 10, 100, 1000]); ax[1].set_ylim(1, 3000); ax[1].axvline(4.5, color="#999999", lw=0.6, ls=":")
ax[1].set_xticks(range(len(SCH))); ax[1].set_xticklabels([l for _, l in SCH], rotation=25, ha="right", rotation_mode="anchor")
ax[1].text(2.0, -0.36, "family left out", transform=ax[1].get_xaxis_transform(), ha="center", va="top", fontsize=8)
ax[1].text(6.0, -0.36, "strains beyond $|\\varepsilon|$", transform=ax[1].get_xaxis_transform(), ha="center", va="top", fontsize=8)
ax[1].set_ylabel("MAE on held-out states (meV)"); letter(ax[1], "b", x=-0.13)
ax[1].legend(handles=legend_handles(hm), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2), handlelength=1.6, columnspacing=1.0)
fig.tight_layout(w_pad=1.5); save(fig, "fig5_extrapolation")

# ------------------------------------------------------------------ Supplementary Figure 1: every run
sm = [FINAL, SA, RBA, DELTA, MLP]; LV = [0, 1, 2, 5, 10, 25, 50, 75, 100]
fig, ax = plt.subplots(figsize=(W, 3.0))
for j, m in enumerate(sm):
    for i, lv in enumerate(LV):
        v = M[(M.model == m) & (M.level == lv)].MAE_meV.values
        if len(v): ax.scatter(i + (j - 2) * 0.15 + rng.uniform(-0.035, 0.035, len(v)), v, s=4, color=STY[m]["c"], marker=STY[m]["m"], linewidths=0, alpha=0.85)
ax.set_yscale("log"); plain_log(ax.yaxis, [3, 10, 30, 100, 300, 1000]); ax.set_xticks(range(len(LV))); ax.set_xticklabels([str(l) for l in LV])
ax.set_xlabel("Labeled training states (%)"); ax.set_ylabel("Bandgap MAE (meV)")
fig.legend(handles=[Line2D([], [], color=STY[m]["c"], marker=STY[m]["m"], ls="", ms=4, label=STY[m]["lab"]) for m in sm], frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "sfig1_spread", si=True)

# ------------------------------------------------------------------ Supplementary Figure 2: training curves with 896 labels
fig, ax = plt.subplots(1, 2, figsize=(W, 2.8))
for m in [SA, RBA, MLP, FINAL]:
    c = CUR[(CUR.model == m) & (CUR.level == 100)]
    ax[0].plot(c.epoch, c.train_mean, color=STY[m]["c"], ls=STY[m]["ls"], lw=1.0, label=STY[m]["lab"])
    ax[1].plot(c.epoch, c.val_best_mean, color=STY[m]["c"], ls=STY[m]["ls"], lw=1.2, label=STY[m]["lab"])
for a, lab, yl, tk in zip(ax, ("a", "b"), ("Training MAE (meV)", "Lowest validation MAE (meV)"), ([0.3, 1, 3, 10, 30, 100, 300, 1000], [5, 10, 20, 50, 100, 200])):
    a.set_yscale("log"); plain_log(a.yaxis, tk); a.set_xlim(0, 24000); a.set_xticks([0, 12000, 24000]); a.set_xticklabels(["0", "12,000", "24,000"]); a.set_xlabel("Training iteration"); a.set_ylabel(yl); letter(a, lab, x=-0.2)
fig.legend(handles=legend_handles([FINAL, SA, RBA, MLP], "line"), frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0); save(fig, "sfig2_training", si=True)

# ------------------------------------------------------------------ Supplementary Figure 3: symmetry drift with 896 labels
sm = [FINAL, SA, RBA, DELTA, MLP]
fig, ax = plt.subplots(1, 2, figsize=(W, 2.8))
for a, col, lab, yl in zip(ax, ("axis", "shearsign"), ("a", "b"), ("Change under axis relabelling (meV)", "Change under shear reversal (meV)")):
    for i, m in enumerate(sm):
        bars(a, i, M[(M.model == m) & (M.level == 100)][col].values, m, 0.7)
    a.set_xticks(range(len(sm))); a.set_xticklabels([STY[m]["lab"] for m in sm], rotation=25, ha="right", rotation_mode="anchor"); a.set_ylabel(yl); a.set_ylim(0, None); letter(a, lab, x=-0.2)
fig.tight_layout(w_pad=2.0); save(fig, "sfig3_symmetry", si=True)
