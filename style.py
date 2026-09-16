"""Shared appearance settings for figures.

apply() sets the plotting defaults. series(name) returns the colour, marker and
line style for a model, keyed by name so a model looks the same in every figure.
The palette is the Okabe-Ito set, which stays distinguishable under the common
forms of colour blindness and in greyscale.
"""

# ----------------------------------------------------------------- palette
# Okabe-Ito, the standard colour-vision-deficiency-safe qualitative set, with
# lightness deliberately spread so the series separate in greyscale too.
_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"
_SKY    = "#56B4E9"
_YELLOW = "#F0E442"
_GREY   = "#666666"

# One entry per model. The two physics-informed variants take contrasting warm
# and cool colours; the baseline takes grey.
MODEL = {
    "SA_PINN":       dict(color=_BLUE,   marker="o", linestyle="-",  label="SA-PINN"),
    "Dia_RBA_PINN":  dict(color=_ORANGE, marker="s", linestyle="-",  label="RBA-PINN"),
    "Shi_ANN":       dict(color=_GREY,   marker="^", linestyle="--", label="Shi ANN"),
    "Shi_ANN_DeltaML": dict(color=_PURPLE, marker="v", linestyle="--", label="Shi ANN ($\\Delta$-ML)"),
    "Pure_NN":       dict(color=_RED,    marker="D", linestyle=":",  label="No physics"),
    "Physics_Only":  dict(color=_GREEN,  marker="*", linestyle="-.", label="Physics only"),
    "Linear_DP":     dict(color=_SKY,    marker="x", linestyle=":",  label="Linear DP"),
}
# Short aliases used by the study scripts.
MODEL["sa"]  = MODEL["SA_PINN"]
MODEL["rba"] = MODEL["Dia_RBA_PINN"]
MODEL["shi"] = MODEL["Shi_ANN"]
MODEL["mlp"] = MODEL["Pure_NN"]

# Strain families. Ordered by how much they deform the cell, and coloured on a
# single sequential ramp so that the ordering is visible rather than arbitrary --
# these are ordered categories, not unrelated ones.
FAMILY = {
    "isotropic": "#FEE6CE",
    "uniaxial":  "#FDAE6B",
    "biaxial":   "#F16913",
    "triaxial":  "#D94801",
    "shear":     "#7F2704",
}

# Sequential and diverging ramps, named so a script never picks one by taste.
#   SEQUENTIAL  a magnitude with a meaningful zero (error, density)
#   DIVERGING   a signed quantity where zero is the reference (residual, bias)
# viridis is perceptually uniform and greyscale-safe; RdBu is the standard
# diverging choice and must ALWAYS be centred on zero or it misleads.
SEQUENTIAL = "viridis"
DIVERGING = "RdBu_r"

# Semantic accents, so "this is the adopted one" is never left to position.
HIGHLIGHT = _BLUE
MUTED = "#BBBBBB"
REFERENCE = "#333333"      # analytic / ground-truth lines


def apply(context="paper"):
    """Set the look once. context: paper | slides."""
    import matplotlib
    scale = 1.0 if context == "paper" else 1.35
    matplotlib.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Serif to match the document body; a figure that uses the page's font
        # reads as part of the argument rather than as an inserted object.
        "font.family": "serif",
        "font.size": 10 * scale,
        "axes.titlesize": 11 * scale,
        "axes.labelsize": 10 * scale,
        "xtick.labelsize": 9 * scale,
        "ytick.labelsize": 9 * scale,
        "legend.fontsize": 9 * scale,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,      # two spines carry the axes; four is noise
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "errorbar.capsize": 2.5,
    })


def series(model, **over):
    """Plot kwargs for one model, keyed by NAME so the colour never moves."""
    d = dict(MODEL.get(model, dict(color=_GREY, marker="o", linestyle="-",
                                   label=str(model))))
    d.update(over)
    return d


def family_color(name):
    """Colour for a strain family, defaulting to grey for anything unlisted."""
    return FAMILY.get(str(name).lower().split("_")[0], MUTED)


def diverging_norm(values, center=0.0):
    """A diverging colour scale centred on `center`.

    A diverging map that is not centred reads a small positive bias as a large
    one, so the centring is not cosmetic. Returns a Normalize to pass as `norm`.
    """
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    half = max(abs(v.max() - center), abs(center - v.min()), 1e-12)
    return TwoSlopeNorm(vmin=center - half, vcenter=center, vmax=center + half)


def annotate_n(ax, n_seeds, loc="lower left"):
    """State the seed count on the figure itself.

    An error bar with no n is not interpretable, and the caption is often
    separated from the figure when it is reused in a talk.
    """
    ax.text(0.02 if "left" in loc else 0.98,
            0.02 if "lower" in loc else 0.98,
            "n = %d seeds" % n_seeds, transform=ax.transAxes,
            ha="left" if "left" in loc else "right",
            va="bottom" if "lower" in loc else "top",
            fontsize=8, color=REFERENCE, alpha=0.8)
