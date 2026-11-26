"""Figures 1 to 5 and Extended Data Figs. 1 to 5 of the paper, drawn from the tables in results/ into figures/. 600 dpi
PNG and vector PDF, Arial at 6.5 to 7 pt (panel letters 8 pt), at most 165 mm wide (the text width, so they are placed
at true size), no grid lines or minor ticks, panel labels a to d.

Regimes keep one hue and one marker throughout: compound blue circles, open shell aqua triangles, polymorph orange
squares and intermetallic plum diamonds. Blue, orange, aqua, plum and brown pass the all-pairs colorblind and
normal-vision checks of the dataviz palette validator; aqua sits below 3:1 contrast on white, so marker shape and
legends always carry identity too. Temperature runs blue to red, lowest blue. Text, reference lines and hatching
stay black or grey. No titles, captions or provenance on the images.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibration_regimes import decisions  # noqa: E402

matplotlib.use("Agg")
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
    "axes.linewidth": 0.6, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.visible": False, "ytick.minor.visible": False, "lines.markersize": 4,
    "legend.fontsize": 7, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.5,
})
BLUE, ORANGE, AQUA, PLUM, BROWN = "#2a78d6", "#eb6834", "#1baf7a", "#a3449e", "#7f3b08"
COOL, WARM = "#4f8fd6", "#e0413a"  # temperature series, low to high
INK, MUTED, REF = "#0b0b0b", "#52514e", "#9a9893"
EDGE = "white"  # a thin ring keeps overlapping markers apart
WHITE_FILL = "#ffffff"
REGIME = {"compound": (BLUE, "o", "Closed shell compounds"), "open-shell": (AQUA, "^", "Open shell oxides"),
          "polymorph": (ORANGE, "s", "Polymorphs"), "intermetallic": (PLUM, "D", "Intermetallics")}


NO_TOOL_TAGS = {"png": {"Software": None}, "pdf": {"Creator": None, "Producer": None}}  # submissions carry no tool strings


def save(fig, name, tight=True):
    """tight=False keeps the canvas as drawn, for a figure laid out in millimetres at the 180 mm limit."""
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight" if tight else None, metadata=NO_TOOL_TAGS[ext])
    plt.close(fig)


def label(ax, letter):
    ax.text(-0.02, 1.04, letter.lower(), transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def tex(f):
    """Mg3Al2(SiO4)3 -> mathtext with subscripts."""
    import re
    return re.sub(r"(\d+)", r"$_{\1}$", f)


def no_minor_log(ax, axis="x"):
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(matplotlib.ticker.NullLocator())


def figure1():
    """The procedure as a flow map: calculation and calorimetry feed the trust scale, its uses, and the measurement loop."""
    from matplotlib.patches import FancyBboxPatch
    W, H = 165, 92  # the text width of a Letter page, so the figure is placed at true size
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    calc, expt, use, test = ("#e3eefb", BLUE), ("#fde6dc", ORANGE), ("#ddf3ea", AQUA), ("#f1f0ee", REF)

    def box(x, y, w, h, title, body, tone, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2", fc=tone[0], ec=tone[1], lw=lw))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=7, fontweight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + (h - 6) / 2, body, ha="center", va="center", fontsize=7, color=INK, linespacing=1.3)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, lw=0.8, color=INK, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7))

    box(2, 64, 38, 22, "Calculated energies", "DFT and machine learning\n(Materials Project, GNoME)", calc)
    box(47, 64, 38, 22, r"Stability margin $\Delta E$", "target vs. competing\nphases, or one precursor\npair vs. another",
        calc)
    box(2, 32, 38, 22, "Calorimetry", "oxide melt solution,\ndirect synthesis, liquid\nmetal solution", expt)
    box(47, 32, 38, 22, r"Uncertainty $\sigma$", "by class: closed shell\noxides, open shell oxides,\nintermetallics",
        expt)
    box(92, 38, 36, 36, "Trust scale",
        r"$p = |\Delta E|\,/\,\sigma$" + "\n\n" + r"$\Phi(p)$, the probability" + "\nthat the calculated\nsign is correct",
        (WHITE_FILL, INK), lw=1.2)
    box(92, 77, 36, 14, "Tested against", "measured relations, holdout,\nmetastable compounds", test)
    for y, title in ((66, "Screen known and\npredicted compounds"), (49, "Choose targets\nand precursors"),
                     (32, "Rank the next\nmeasurement")):
        ax.add_patch(FancyBboxPatch((134, y), 30, 13, boxstyle="round,pad=0,rounding_size=2", fc=use[0], ec=use[1], lw=0.8))
        ax.text(149, y + 6.5, title, ha="center", va="center", fontsize=7, fontweight="bold", color=INK, linespacing=1.25)
        arrow(128, 56, 134, y + 6.5)
    ax.text(149, 85.5, "the researcher sets the\nprobability required", ha="center", va="center", fontsize=7,
            style="italic", color=MUTED)
    box(92, 4, 72, 18, "Calorimetric measurement",
        "fixes every relation the compound enters\nand scores the predictions deposited before it", expt)
    ax.text(56, 14.5, "joins the calorimetric reference", ha="center", va="bottom", fontsize=7, style="italic",
            color=MUTED)
    arrow(40, 75, 47, 75)
    arrow(40, 43, 47, 43)
    arrow(85, 75, 92, 64)
    arrow(85, 43, 92, 48)
    arrow(110, 74, 110, 77, style="<|-|>")
    arrow(149, 32, 149, 22)
    ax.plot([92, 21], [13, 13], color=INK, lw=0.8)
    arrow(21, 13, 21, 32)
    save(fig, "fig1_workflow", tight=False)


def figure2():
    """Calorimetric benchmark, machine-learned potentials, and the blind holdout."""
    b = (pd.read_csv(RES / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
         .drop_duplicates(["compound", "measured_meV_atom"]))
    m = pd.read_csv(RES / "mlip_vs_measured.csv")
    h = pd.read_csv(RES / "holdout_test.csv").sort_values("z")
    fams = {"spinels": ("o", "Spinels", BLUE), "RE titanate pyrochlores": ("D", "RE titanate pyrochlores", ORANGE),
            "titanates and zirconates": ("v", "Titanates and zirconates", AQUA),
            "La perovskites": ("s", "La perovskites", PLUM), "RE orthophosphates": ("^", "RE orthophosphates", BROWN),
            "Li cathode oxides": ("P", "LiMnO$_2$", MUTED)}
    fig = plt.figure(figsize=(150 * MM, 140 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1.1, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]

    ax = axes[0]
    x, y = b.measured_meV_atom.values, b.error_meV_atom.values
    k, c = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 20, 40, 50)
    ax.axhline(0, color=REF, lw=0.7)
    ax.plot(xs, k * xs + c, color=INK, lw=0.9, ls="--", label=f"Trend (slope {1 + k:.2f})")
    for fam, (mk, lab, colr) in fams.items():
        s = b[b.family == fam]
        ax.scatter(s.measured_meV_atom, s.error_meV_atom, marker=mk, s=16, color=colr, edgecolor=EDGE, lw=0.4,
                   label=lab, zorder=3)
    ax.set_xlabel("Measured enthalpy of formation from oxides (meV/atom)")
    ax.set_ylabel("Difference between calculated\nand measured (meV/atom)")
    ax.set_ylim(-125, 140)
    ax.legend(loc="lower left", fontsize=7, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    label(ax, "A")

    ax = axes[1]
    order = ["spinels", "RE titanate pyrochlores", "titanates and zirconates", "La perovskites", "RE orthophosphates"]
    methods = [("dft_error", "DFT", "o", BLUE, BLUE), ("orbinf_relaxed", "Orb-v3", "s", "white", ORANGE),
               ("mace_relaxed", "MACE-MP-0", "^", "white", PLUM)]
    yp = np.arange(len(order))[::-1]
    for j, (col, lab, mk, face, edge) in enumerate(methods):
        vals = [(m[m.family == f][col] if col == "dft_error" else m[m.family == f][col] - m[m.family == f].measured_meV_atom).mean()
                for f in order]
        ax.scatter(vals, yp + (j - 1) * 0.24, marker=mk, s=16, facecolor=face, edgecolor=edge, lw=0.8, label=lab, zorder=3)
    ax.axvline(0, color=REF, lw=0.7)
    ax.set_yticks(yp)
    ax.set_yticklabels([fams[f][1] for f in order])
    ax.set_xlabel("Mean error against calorimetry (meV/atom)")
    ax.legend(loc="upper right")
    label(ax, "B")

    ax = axes[2]
    xx = np.arange(len(h))
    for v, ls in ((1, "--"), (2, ":")):
        for sgn in (-1, 1):
            ax.axhline(sgn * v, color=REF, lw=0.7, ls=ls, zorder=1)
    for reg, sel in (("compound", ~h.open_shell), ("open-shell", h.open_shell)):
        col, mk, lab = REGIME[reg]
        ax.scatter(xx[sel.values], h.z[sel], marker=mk, s=18, color=col, edgecolor=EDGE, lw=0.4, zorder=3,
                   label=lab)
    ax.axhline(0, color=REF, lw=0.7)
    ax.set_xticks(xx)
    ax.set_xticklabels([tex(s) for s in h.compound], fontsize=7, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, len(h) - 0.3)
    ax.set_ylabel(r"Held-out error (units of $\sigma$)")
    ax.set_ylim(-3.2, 3.2)
    ax.legend(loc="upper left", ncol=2)
    label(ax, "C")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig2_calorimetry")


def figure3():
    """Calibration by regime, the polymorph density bias and its r2SCAN correction, and the sigma each regime allows."""
    d = decisions()
    reg = pd.read_csv(RES / "calibration_regimes.csv")
    reg = reg[reg.labels == "audited"].set_index("regime")
    poly = pd.read_csv(RES / "polymorph_r2scan.csv")
    fig = plt.figure(figsize=(150 * MM, 125 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 0.8])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]

    ax = axes[0]
    ax.plot([0.5, 1], [0.5, 1], color=INK, lw=0.8, ls="--", label="Calibrated")
    for r, nb in (("compound", 6), ("open-shell", 3), ("polymorph", 3)):
        g = d[d.regime == r].sort_values("pred")
        col, mk, lab = REGIME[r]
        for i, idx in enumerate(np.array_split(np.arange(len(g)), nb)):
            chunk = g.iloc[idx]
            k, n = int(chunk.q_audited.sum()), len(chunk)
            ci = binomtest(k, n).proportion_ci(method="wilson")
            ax.errorbar(chunk.pred.mean(), k / n, yerr=[[k / n - ci.low], [ci.high - k / n]], fmt=mk, ms=4, color=col,
                        mec=EDGE, mew=0.4, ecolor=col, elinewidth=0.8, capsize=1.5, label=lab if i == 0 else None,
                        zorder=3)
    # Intermetallic decisions from direct-synthesis calorimetry, p at the oxide sigma (calibration_intermetallic.py).
    from calibration_intermetallic import decisions as intermetallic_decisions
    g = intermetallic_decisions().sort_values("pred")
    for i, idx in enumerate(np.array_split(np.arange(len(g)), 6)):
        chunk = g.iloc[idx]
        k, n = int(chunk.q.sum()), len(chunk)
        ci = binomtest(k, n).proportion_ci(method="wilson")
        ax.errorbar(chunk.pred.mean(), k / n, yerr=[[k / n - ci.low], [ci.high - k / n]], fmt="D", ms=3.6,
                    color=PLUM, mec=EDGE, mew=0.4, ecolor=PLUM, elinewidth=0.8, capsize=1.5,
                    label=REGIME["intermetallic"][2] if i == 0 else None, zorder=4)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Predicted probability correct, $\Phi(p)$")
    ax.set_ylabel("Share confirmed by measurement")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7, columnspacing=0.8)
    label(ax, "A")

    ax = axes[1]
    poly = poly.dropna(subset=["gga_meV_atom", "measured_meV_atom"])  # ground states known by sign only have no error
    err_gga = poly.gga_meV_atom - poly.measured_meV_atom
    has = poly.r2scan_meV_atom.notna()
    err_r2 = poly.r2scan_meV_atom - poly.measured_meV_atom
    # CoO (GGA -229, r2SCAN -284 meV/atom) lies far below the others; it is drawn at the floor and labelled.
    floor = -74
    off = (err_gga < floor) | (err_r2 < floor)
    for i in np.where(off)[0]:
        ax.annotate(f"{poly.formula.iloc[i]} {err_gga.iloc[i]:.0f}, {err_r2.iloc[i]:.0f}".replace("-", "−"),
                    xy=(poly.denser_stable_pct.iloc[i], floor), xytext=(-4, -1), textcoords="offset points", fontsize=6.5,
                    color=INK, ha="right", va="center")
    err_gga, err_r2 = err_gga.clip(lower=floor), err_r2.clip(lower=floor)
    for i in np.where(has & ~off)[0]:
        ax.annotate("", xy=(poly.denser_stable_pct.iloc[i], err_r2.iloc[i]), xytext=(poly.denser_stable_pct.iloc[i], err_gga.iloc[i]),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, alpha=0.8, shrinkA=2, shrinkB=2), zorder=2)
    ax.scatter(poly.denser_stable_pct[~off], err_gga[~off], marker="s", s=16, color=ORANGE, edgecolor=EDGE, lw=0.4,
               zorder=3, label="GGA/GGA+U")
    ax.scatter(poly.denser_stable_pct[has & ~off], err_r2[has & ~off], marker="s", s=16, facecolor="white",
               edgecolor=ORANGE, lw=0.9, zorder=3, label="r2SCAN, same materials")
    ax.scatter(poly.denser_stable_pct[off], err_gga[off], marker="v", s=18, color=ORANGE, edgecolor=EDGE, lw=0.4,
               zorder=3)
    ax.axhline(0, color=REF, lw=0.7)
    ax.axvline(0, color=REF, lw=0.7, ls=":")
    wrong = np.sign(poly.gga_meV_atom) != np.sign(poly.measured_meV_atom)
    ax.fill_between([0, 52], -80, 0, facecolor="none", edgecolor="#c9c7c2", hatch="//", lw=0, zorder=0)
    ax.set_xlabel("Stable polymorph denser than metastable (%)")
    ax.set_ylabel("Difference between calculated\nand measured (meV/atom)")
    ax.set_ylim(-80, 95)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    label(ax, "B")

    ax = axes[2]
    from trust_scale import gaussian, heavy
    margin = np.linspace(0, 100, 401)
    for model, colr, ls, lab in ((gaussian(22.5), BLUE, "-", "Closed shell compounds, σ = 22.5"),
                                 (gaussian(34.0), AQUA, "-", "Open shell, σ = 34 (upper bound)"),
                                 (heavy(), PLUM, "-", "Intermetallics, Student's $t$")):
        ax.plot(margin, model["cdf"](margin), color=colr, lw=1.3, ls=ls, label=lab, zorder=3)
    # polymorphs get no probability; their observed shares, in equal-count bins of margin, show why
    g = d[d.regime == "polymorph"].assign(margin=lambda x: x.p * x.sigma).sort_values("margin")
    for i, idx in enumerate(np.array_split(np.arange(len(g)), 3)):
        chunk = g.iloc[idx]
        k, n = int(chunk.q_audited.sum()), len(chunk)
        ci = binomtest(k, n).proportion_ci(method="wilson")
        if chunk.margin.mean() <= 100:
            ax.errorbar(chunk.margin.mean(), k / n, yerr=[[k / n - ci.low], [ci.high - k / n]], fmt="s", ms=4, color=ORANGE,
                        mec=EDGE, mew=0.4, ecolor=ORANGE, elinewidth=0.8, capsize=1.5, zorder=4,
                        label="Polymorphs, observed" if i == 0 else None)
    ax.axhline(0.5, color=REF, lw=0.7, ls=":", zorder=1)
    # the three worked examples of Table 2
    for name, m, prob, colr, dx in (("Ca$_3$SiO$_5$", 20.1, 0.814, BLUE, 2), ("ZnFe$_2$O$_4$", 14.0, 0.660, AQUA, -2),
                                    ("TiSi$_2$", 17.7, 0.626, PLUM, 2)):
        ax.scatter(m, prob, s=34, marker="o", color=colr, edgecolor=INK, lw=0.8, zorder=5)
        ax.annotate(name, (m, prob), xytext=(dx * 3, -9 if name.startswith("TiSi") else 6), textcoords="offset points",
                    fontsize=7, ha="left" if dx > 0 else "right", color=INK)
    for v, text in ((10, "10"), (15, "15")):
        ax.axvline(v, color=REF, lw=0.7, ls="--", zorder=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Calculated margin (meV/atom)")
    ax.set_ylabel("Probability that the sign is correct")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    label(ax, "C")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig3_calibration")


def figure4():
    """Precision across known and machine-learning-predicted oxides."""
    mp = pd.read_csv(RES / "precision_mp_observed.csv")
    core = mp.oxide.astype(bool) & mp.competitor.isin(["fewer-element phases only", "same-order neighbour"])
    ox = mp[core].assign(right=lambda x: norm.cdf(x.margin_meV.abs() / x.sigma))
    obs = ox.groupby("family").agg(resolved=("resolved", "mean"), right=("right", "mean"), n=("right", "size"))
    gs = pd.read_csv(RES / "gnome_precision_summary.csv")
    gf = gs[gs.subset.str.startswith("family: ")].assign(family=lambda x: x.subset.str[8:]).set_index("family")
    fams = [f for f in obs.sort_values("resolved").index if f in gf.index]
    fig = plt.figure(figsize=(150 * MM, 120 * MM))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.05])
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]

    ax = axes[0]
    y = np.arange(len(fams))
    ax.scatter(100 * obs.loc[fams, "resolved"], y + 0.15, marker="o", s=16, color=BLUE, edgecolor=EDGE, lw=0.4,
               label="Observed oxides", zorder=3)
    ax.scatter(100 * gf.loc[fams, "resolved_share"], y - 0.15, marker="s", s=16, color=ORANGE, edgecolor=EDGE,
               lw=0.4, label="GNoME oxides", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f.capitalize() for f in fams])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relations resolved, $p$ > 1 (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7)
    label(ax, "A")

    ax = axes[1]
    snap = gs[(gs.hull == "MP snapshot") & gs.subset.isin(["all scored", "compound", "open-shell"])]
    rows = [("All GNoME oxides", "all scored", INK, "D"), ("Compound", "compound", BLUE, "o"),
            ("Open shell", "open-shell", AQUA, "^")]
    allrange = gs[gs.subset.isin(["all scored", "compound", "open-shell"])]
    for i, (lab, sub, col, mk) in enumerate(rows):
        yy = len(rows) - 1 - i
        a = allrange[allrange.subset == sub]
        lo, hi = 100 * min(a.expected_right_flat.min(), a.expected_right_eb.min()), 100 * a.expected_right_flat.max()
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.0, alpha=0.6)
        base = snap[(snap.subset == sub) & snap.sigma.str.startswith("calorimetric")].iloc[0]
        ax.scatter(100 * base.expected_right_flat, yy + 0.12, marker=mk, s=18, color=col, edgecolor=EDGE, lw=0.4,
                   zorder=3, label="Face value" if i == 0 else None)
        ax.scatter(100 * base.expected_right_eb, yy - 0.12, marker=mk, s=18, facecolor="white", edgecolor=col, lw=0.9,
                   zorder=3, label="Selection-corrected" if i == 0 else None)
    ax.axvline(100 * ox.right.mean(), color=REF, lw=0.8, ls="--")
    ax.text(100 * ox.right.mean() + 1, -0.55, "observed\noxides", fontsize=7, color=MUTED, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 3.1)
    ax.set_xlabel("Calculated signs expected correct (%)")
    ax.legend(loc="upper left", fontsize=7)
    label(ax, "B")

    ax = axes[2]
    chk = pd.read_csv(RES / "gnome_precision_r2scan.csv")
    mids = {"(-0.001, 0.5]": 0.25, "(0.5, 1.0]": 0.75, "(1.0, 2.0]": 1.5, "(2.0, 4.0]": 3.0, "(4.0, inf]": 6.0}
    for r in ("compound", "open-shell"):
        c = chk[chk.regime == r].assign(x=lambda x: x.bin.map(mids))
        col, mk, lab = REGIME[r]
        ax.plot(c.x, c.kept_r2scan, marker=mk, ms=4, color=col, lw=0.9, mec=EDGE, mew=0.4, label=f"{lab}, kept")
        ax.plot(c.x, c.predicted, color=col, lw=0.7, ls="--")
        ax.plot(c.x, c.predicted_eb, color=col, lw=0.7, ls=":")
    ax.plot([], [], color=MUTED, lw=0.7, ls="--", label=r"$\Phi(p)$")
    ax.plot([], [], color=MUTED, lw=0.7, ls=":", label="Selection-corrected")
    ax.set_xscale("log")
    no_minor_log(ax)
    ax.set_xticks([0.25, 0.75, 1.5, 3, 6])
    ax.set_xticklabels(["<0.5", "0.5–1", "1–2", "2–4", ">4"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Precision of GNoME claim, $p$")
    ax.set_ylabel("Share")
    ax.legend(loc="lower right", fontsize=7)
    label(ax, "C")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig4_discovery")


def figure5():
    """The two automated campaigns, the test of the measurement ranking, and the deposited predictions."""
    from sigma_model import open_shell
    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[(alab.dataset == "A-Lab2023") & ~alab.database_inconsistency.astype(bool)]
    col = "p_stability_measured_scatter"
    curves = pd.read_csv(RES / "measurement_ladder_curves.csv")
    pre = pd.read_csv(RES / "preregistered_decisions.csv", comment="#")
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(150 * MM, 125 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.8, 1])
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    ax = axes[0]
    groups = [("Chen et al., precursor selection", chen, "p_selection_measured_scatter"),
              ("Chen et al., target stability", chen, col), ("A-Lab, target stability", alab, col)]
    for i, (lab, df, c) in enumerate(groups):
        y0 = len(groups) - 1 - i
        for reg in ("compound", "open-shell"):
            sel = df.target.map(open_shell) == (reg == "open-shell")
            vals = df.loc[sel, c].clip(lower=0.003)
            colr, mk, _ = REGIME[reg]
            ax.scatter(vals, y0 + rng.uniform(-0.25, 0.25, len(vals)), marker=mk, s=16, color=colr, edgecolor=EDGE,
                       lw=0.4, zorder=3, label=REGIME[reg][2] if i == 0 else None)
    ax.axvline(1, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    no_minor_log(ax)
    ax.set_xlim(0.003, 70)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0].replace(", ", ",\n") for g in groups][::-1])
    ax.set_xlabel("Metastability precision, $p$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7)
    label(ax, "A")

    ax = axes[1]
    noisy = curves[curves.rung == "5/10"]
    exact = curves[curves.rung == "3/10"]
    for df, strategy, style, lab in (
            (noisy, "method", dict(color=BLUE, lw=1.3), "Method"),
            (noisy, "closest calls first", dict(color=ORANGE, lw=0.9, ls="--"), "Smallest margins first"),
            (noisy, "most connected first", dict(color=AQUA, lw=1.0, ls=":"), "Most connected first"),
            (exact, "random (mean)", dict(color=MUTED, lw=0.8, ls="-."), "Random, exact measurements"),
            (exact, "oracle", dict(color=REF, lw=0.7), "Oracle, exact measurements")):
        c = df[df.strategy == strategy]
        ax.plot(c.K, c.right, label=lab, **style)
    ax.axhline(118, color=REF, lw=0.6, ls=":")
    ax.set_xlim(0, 40)
    ax.set_ylim(103, 119)
    ax.set_xlabel("Measurements made")
    ax.set_ylabel("Known answer relations correct (of 118)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.6, columnspacing=1.0)
    label(ax, "B")

    ax = axes[2]
    flip = pre.predicted_sign != np.where(pre.computed_margin_meV_atom < 0, "negative", "positive")
    lim = (-80, 50)
    ax.set_xscale("symlog", linthresh=10, linscale=1.5)
    ax.plot(np.linspace(*lim, 400), np.linspace(*lim, 400), color=REF, lw=0.7, ls="--", zorder=1)
    ax.axhline(0, color=REF, lw=0.6, zorder=1)
    ax.axvline(0, color=REF, lw=0.6, zorder=1)
    for sel, face, edge, lab in ((~flip, "white", BLUE, "Calculated sign expected to hold"),
                                 (flip, ORANGE, INK, "Predicted to reverse")):
        ax.errorbar(pre.computed_margin_meV_atom[sel], pre.predicted_true_margin_meV_atom[sel],
                    yerr=pre.predicted_sd[sel], fmt="o", ms=3.5, mfc=face, mec=edge, mew=0.6, ecolor=REF,
                    elinewidth=0.5, capsize=0, zorder=3 if face == ORANGE else 2, label=lab)
    ax.set_xlim(*lim)
    ax.set_ylim(-150, 110)
    ticks = [-50, -10, -5, 0, 5, 10, 50]
    ax.set_xticks(ticks, [str(v).replace("-", "−") for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Calculated margin (meV/atom)")
    ax.set_ylabel("Predicted measured margin (meV/atom)")
    ax.legend(loc="upper left")
    label(ax, "C")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig5_campaigns")


# Extended Data Figs. 1 to 5.
MEV_PER_J_MOL = 1000 / 96485.33
OPEN_SHELL_3D = {"Cr2O3", "MnO", "Mn2O3", "FeO", "Fe2O3", "CoO", "NiO", "CuO", "Cu2O"}  # d0 CrO3, V2O5, TiO2 stay out


def scatter_band(ax, axis="x", v=22.5):
    """The calorimetric scatter as dashed lines."""
    line = ax.axvline if axis == "x" else ax.axhline
    for sgn in (-1, 1):
        line(sgn * v, color=REF, lw=0.7, ls="--", zorder=1)


def figure_s1():
    """GGA/GGA+U and r2SCAN against calorimetry, without and with an open-shell 3d cation."""
    b = pd.read_csv(RES / "benchmark_r2scan_calorimetric.csv")
    b["tm"] = b.oxides.map(lambda s: any(p.split(" ", 1)[1] in OPEN_SHELL_3D for p in s.split("+")))
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 70 * MM), sharey=True)
    for ax, (sub, letter) in zip(axes, [(b[~b.tm], "A"), (b[b.tm], "B")]):
        x = sub.measured_meV_atom.values
        ax.axhline(0, color=REF, lw=0.7, zorder=1)
        for col, lab, mk, colr, ls in (("error_meV_atom", "GGA/GGA+U", "o", BLUE, ":"),
                                       ("r2scan_error_meV_atom", "r2SCAN", "s", ORANGE, "--")):
            y = sub[col].values
            k, c = np.polyfit(x, y, 1)
            sd = np.std(y - (k * x + c), ddof=2)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, k * xs + c, color=colr, lw=1.0, ls=ls, zorder=2)
            ax.scatter(x, y, marker=mk, s=16, color=colr, edgecolor=EDGE, lw=0.4, zorder=3,
                       label=f"{lab}, slope {1 + k:.2f}, SD {sd:.0f} meV/atom")
        ax.legend(loc="lower left", borderaxespad=0.2)
        label(ax, letter)
    axes[0].set_ylabel("Difference between calculated\nand measured (meV/atom)")
    axes[0].set_ylim(-275, 145)
    fig.supxlabel("Measured enthalpy of formation from oxides (meV/atom)", fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "extended_data_fig1_r2scan")


def measured_reaction_entropy(ax):
    d = pd.read_csv(RES / "reaction_entropy_measured.csv").sort_values("TdS_1000K_meV_atom")
    y = np.arange(len(d))
    scatter_band(ax)
    ax.axvline(0, color=REF, lw=0.7, zorder=1)
    for i in y:
        ax.plot([d.TdS_1000K_meV_atom.iloc[i], d.TdS_1300K_meV_atom.iloc[i]], [i, i], color=REF, lw=0.6, zorder=2)
    ax.plot(d.TdS_1000K_meV_atom, y, ls="none", marker="o", ms=4, mfc=COOL, mec=EDGE, mew=0.4, zorder=3,
            label="1000 K")
    ax.plot(d.TdS_1300K_meV_atom, y, ls="none", marker="s", ms=4, mfc=WARM, mec=EDGE, mew=0.4, zorder=3,
            label="1300 K")
    ax.set_yticks(y, [tex(c) for c in d.compound], fontsize=6.5)
    ax.set_ylim(-0.8, len(d) - 0.2)
    span = 1.08 * max(d.TdS_1300K_meV_atom.abs().max(), 30)
    ax.set_xlim(-span, span)
    ax.set_xlabel("$T$ × entropy of formation (meV/atom)")
    ax.legend(loc="lower right")


def decision_margins_at_temperature(ax):
    d = pd.read_csv(RES / "precision_temperature.csv")
    lin = 30.0
    ax.set_xscale("symlog", linthresh=lin, linscale=2.0)
    ax.set_yscale("symlog", linthresh=lin, linscale=2.0)
    lim = (-400, 60)
    # Hatched: the quadrants where a decision changes sign at temperature. Dashed: the 22.5 meV/atom scatter.
    ax.fill_between([lim[0], 0], 0, lim[1], facecolor="none", edgecolor="#c9c7c2", hatch="//", lw=0, zorder=0)
    ax.fill_between([0, lim[1]], lim[0], 0, facecolor="none", edgecolor="#c9c7c2", hatch="//", lw=0, zorder=0)
    scatter_band(ax, "x")
    scatter_band(ax, "y")
    ax.axhline(0, color=REF, lw=0.6, zorder=1)
    ax.axvline(0, color=REF, lw=0.6, zorder=1)
    ax.plot(lim, lim, color=INK, lw=0.7, zorder=1)
    groups = [(("Chen2024", "selection"), dict(marker="^", mfc=BLUE, label="Chen et al., selection")),
              (("Chen2024", "stability"), dict(marker="o", mfc=AQUA, label="Chen et al., stability")),
              (("A-Lab2023", "stability"), dict(marker="s", mfc=ORANGE, label="A-Lab, stability"))]
    for (ds, kind), st in groups:
        g = d[(d.dataset == ds) & (d.decision == kind)]
        ax.plot(g.margin_0K, g.margin_final, ls="none", ms=4, mec=EDGE, mew=0.4, zorder=3, **st)
    md = d[d.entropy_source == "MD"]
    ax.plot(md.margin_0K, md.margin_final, ls="none", marker="o", ms=7, mfc="none", mec=INK, mew=0.7, zorder=2,
            label="Entropy from MD")
    ticks = [-300, -30, -10, 0, 10, 30]
    ax.set_xticks(ticks, [str(t).replace("-", "−") for t in ticks])
    ax.set_yticks(ticks, [str(t).replace("-", "−") for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlabel("Margin at 0 K (meV/atom)")
    ax.set_ylabel("Margin at the firing temperature (meV/atom)")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2, fontsize=7, framealpha=1.0, frameon=True,
              edgecolor="none")


def figure_s2():
    """Measured and computed entropies, and the campaign margins at the firing temperature."""
    h = pd.read_csv(RES / "hermes_vs_measured.csv")
    rx = pd.read_csv(RES / "hermes_reaction_entropies.csv").dropna(subset=["TdS_rxn_measured_meV_atom"])
    hp = pd.read_csv(RES / "harmonic_vs_measured.csv").dropna(subset=["S_measured"])
    hr = pd.read_csv(RES / "harmonic_reaction_entropy.csv")
    rx["target"] = rx.reaction.str.split(" from ").str[0]
    fig = plt.figure(figsize=(150 * MM, 170 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.8, 1], width_ratios=[1, 1.15])
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_c, ax_d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    measured_reaction_entropy(ax_a)
    label(ax_a, "A")

    temps = {298.15: (COOL, "o", "298 K"), 1000.0: (WARM, "s", "1000 K")}
    ax = ax_c
    hi = 1.05 * h[["S_hermes", "S_measured"]].max().max()
    ax.plot([0, hi], [0, hi], color=INK, lw=0.7, zorder=1)
    for t, (col, mk, lab) in temps.items():
        s = h[h.T_K.round(2) == t]
        ax.plot(s.S_measured, s.S_hermes, ls="none", marker=mk, ms=4, mfc=col, mec=EDGE, mew=0.4, zorder=3, label=lab)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlabel("Measured entropy (J K$^{-1}$ mol$^{-1}$)")
    ax.set_ylabel("Hermes entropy (J K$^{-1}$ mol$^{-1}$)")
    ax.legend(loc="upper left")
    label(ax, "C")

    ax = ax_b
    series = {("md", 298.15): (-0.27, dict(marker="o", mfc="white", mec=COOL, label="MD, 298 K")),
              ("harmonic", 298.15): (-0.09, dict(marker="^", mfc="white", mec=COOL, label="Harmonic, 298 K")),
              ("md", 1000.0): (0.09, dict(marker="o", mfc=WARM, mec=WARM, label="MD, 1000 K")),
              ("harmonic", 1000.0): (0.27, dict(marker="^", mfc=WARM, mec=WARM, label="Harmonic, 1000 K"))}
    order = h[h.T_K.round(2) == 298.15].sort_values("S_measured").compound.tolist()
    y = {c: i for i, c in enumerate(order)}
    scatter_band(ax)
    ax.axvline(0, color=REF, lw=0.7, zorder=1)
    labelled = set()

    def mark(x, rows, key):
        dy, st = series[key]
        if key in labelled:
            st = {k: v for k, v in st.items() if k != "label"}
        labelled.add(key)
        ax.plot(x, [r + dy for r in rows], ls="none", ms=4, mew=0.8, zorder=3, **st)

    for t in temps:
        s = h[h.T_K.round(2) == t]
        mark(s.TdS_meV_atom, [y[c] for c in s.compound], ("md", t))
        s = hp[(hp.T_K.round(2) == t) & hp.phase.isin(y)]
        mark(s.TdS_meV_atom, [y[c] for c in s.phase], ("harmonic", t))
    meas298 = dict(zip(hr.compound, hr.measured_dS298))
    for c, g in rx[rx.T_K.round(2) == 298.15].groupby("target"):
        meas298.setdefault(c, g.dS_rxn_measured.iloc[0])
    targets = sorted(set(rx.target) | set(hr.compound), key=lambda c: meas298[c])
    yr = {c: len(order) + 0.4 + i for i, c in enumerate(targets)}
    ax.axhline(len(order) - 0.3, color=INK, lw=0.6, zorder=1)
    for t in temps:
        s = rx[rx.T_K.round(2) == t]
        mark(s.TdS_rxn_hermes_meV_atom - s.TdS_rxn_measured_meV_atom, [yr[c] for c in s.target], ("md", t))
    for t, col in ((298.15, "error_dS298"), (1000.0, "error_dS1000")):
        s = hr.dropna(subset=[col])
        mark(t * s[col] * MEV_PER_J_MOL, [yr[c] for c in s.compound], ("harmonic", t))
    ax.set_yticks(list(range(len(order))) + list(yr.values()),
                  [tex(c) for c in order] + [tex(c) + " from oxides" for c in targets], fontsize=6.5)
    ax.set_ylim(-0.6, max(yr.values()) + 0.6)
    span = max(30.0, 1.1 * h.TdS_meV_atom.abs().max())
    ax.set_xlim(-1.6 * span, span)  # room on the empty negative side for the legend
    ax.set_xlabel("$T$ × entropy error (meV/atom)")
    ax.legend(loc="lower left", fontsize=7, borderaxespad=0.2, handletextpad=0.4)
    label(ax, "B")
    decision_margins_at_temperature(ax_d)
    label(ax_d, "D")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save(fig, "extended_data_fig4_entropy")


def figure_s3():
    """Chen et al. decisions recomputed on the current release against the published values."""
    d = pd.read_csv(RES / "chen_rerun_today.csv").dropna(subset=["today_rxn_meV"])
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 76 * MM))
    for ax, (pub, now, name), letter in zip(axes, [("pub_rxn_meV", "today_rxn_meV", "reaction energy"),
                                                   ("pub_invE_meV", "today_invE_meV", "inverse hull energy")], "AB"):
        x, y = -d[pub], -d[now]
        ok = (x > 0) & (y > 0)
        lo, hi = 0.05, 400
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.7, zorder=1)
        top = d.today_robot_pair_is_top.astype(bool)
        sel = d.today_robot_pair_still_selected.astype(bool)
        ax.scatter(x[ok & top], y[ok & top], s=16, marker="o", color=BLUE, edgecolor=EDGE, lw=0.4, zorder=3,
                   label="Tested pair still ranked first")
        ax.scatter(x[ok & sel & ~top], y[ok & sel & ~top], s=18, marker="^", facecolor="white", edgecolor=ORANGE,
                   lw=1.0, zorder=4, label="Still selected, no longer first")
        ax.scatter(x[ok & ~sel], y[ok & ~sel], s=18, marker="x", color=WARM, lw=1.2, zorder=4, label="No longer selected")
        ax.set_xscale("log")
        ax.set_yscale("log")
        no_minor_log(ax, "x")
        no_minor_log(ax, "y")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Published {name} (meV/atom)")
        ax.set_ylabel(f"Recalculated {name} (meV/atom)")
        ax.set_aspect("equal")
        label(ax, letter)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "extended_data_fig3_reproducibility")


def figure_s4():
    """DFT error against compiled enthalpies of formation from the binary oxides, 120 compounds."""
    b = pd.read_csv(RES / "benchmark_compiled.csv")
    fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
    ax.axhline(0, color=REF, lw=0.7, zorder=1)
    ax.scatter(b.measured_meV_atom, b.error_meV_atom, s=16, marker="o", color=BLUE, edgecolor=EDGE, lw=0.4,
               zorder=2, label=f"Compiled values ($n$ = {len(b)})")
    edges = [-10000, -400, -200, -100, -50, 10000]
    g = b.groupby(pd.cut(b.measured_meV_atom, edges), observed=True).agg(
        x=("measured_meV_atom", "median"), m=("error_meV_atom", "mean"),
        se=("error_meV_atom", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
    ax.errorbar(g.x, g.m, yerr=g.se, fmt="s", ms=4.5, color=ORANGE, mec=INK, mew=0.5, ecolor=INK, capsize=3, lw=1.0,
                zorder=4, label="Bin mean with standard error")
    ax.set_xlabel("Measured enthalpy of formation from oxides (meV/atom)")
    ax.set_ylabel("Difference between calculated\nand measured (meV/atom)")
    ax.legend(loc="lower left")
    save(fig, "extended_data_fig2_compiled")


def figure_s5():
    """How far cation disorder can move each A-Lab margin at the synthesis temperature."""
    d = pd.read_csv(RES / "configurational_ordering.csv").sort_values("stability_margin_meV")
    fig, ax = plt.subplots(figsize=(120 * MM, 140 * MM))
    y = np.arange(len(d))
    scatter_band(ax)
    scatter_band(ax, v=34.0)
    for line in ax.lines[-2:]:
        line.set_linestyle(":")
    ax.axvline(0, color=REF, lw=0.7, zorder=1)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.stability_margin_meV, r.margin_at_T_best], [i, i], color=REF, lw=0.7, zorder=2)
    ax.plot(d.stability_margin_meV, y, ls="none", marker="o", ms=4, mfc="white", mec=BLUE, mew=1.0, zorder=4,
            label="0 K")
    ax.plot(d.margin_at_T_worst, y, ls="none", marker="s", ms=4, mfc=ORANGE, mec=EDGE, mew=0.4, zorder=3,
            label="With the certain disorder gain")
    ax.plot(d.margin_at_T_best, y, ls="none", marker="|", ms=4.5, mew=1.2, color=INK, zorder=3,
            label="With the largest possible gain")
    ax.set_yticks(y, [tex(c) for c in d.target], fontsize=6.5)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("Energy relative to the hull of competing phases (meV/atom)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3, fontsize=7, handletextpad=0.3, columnspacing=1.0)
    save(fig, "extended_data_fig5_disorder")


if __name__ == "__main__":
    for f in (figure1, figure2, figure3, figure4, figure5, figure_s1, figure_s2, figure_s3, figure_s4, figure_s5):
        f()
    print("written: fig1 to fig5, extended_data_fig1 to extended_data_fig5")
