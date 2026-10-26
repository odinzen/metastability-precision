"""Figures 1 to 5 and the SI figures from the result tables, greyscale house style, 600 dpi PNG and vector PDF.

No titles, captions or provenance lines on the images: the document's figure legends carry that text.
Identity is carried by marker shape and grey shade together, never by shade alone.
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
    "axes.linewidth": 0.6, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 7, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 600,
})
GRID = dict(color="0.88", linewidth=0.4, zorder=0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, letter):
    # Above the top-left corner of the axes, clear of the y-axis label whatever its length.
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def figure1():
    """Chen et al. decisions recomputed on today's Materials Project release against the published values."""
    d = pd.read_csv(RES / "chen_rerun_today.csv").dropna(subset=["today_rxn_meV"])
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 80 * MM))
    for ax, (pub, now, label), letter in zip(
            axes, [("pub_rxn_meV", "today_rxn_meV", "reaction energy"),
                   ("pub_invE_meV", "today_invE_meV", "inverse hull energy")], "ab"):
        x, y = -d[pub], -d[now]  # plotted as positive margins on log axes
        ok = (x > 0) & (y > 0)
        lo, hi = 0.05, 400
        ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.8, zorder=1)
        top = d.today_robot_pair_is_top.astype(bool)
        sel = d.today_robot_pair_still_selected.astype(bool)
        ax.scatter(x[ok & top], y[ok & top], s=22, marker="o", facecolor="0.2", edgecolor="white", lw=0.5,
                   zorder=3, label="tested pair still ranked first")
        ax.scatter(x[ok & sel & ~top], y[ok & sel & ~top], s=26, marker="^", facecolor="white", edgecolor="0.1",
                   lw=0.8, zorder=4, label="still selected, no longer first")
        ax.scatter(x[ok & ~sel], y[ok & ~sel], s=30, marker="x", color="0.1", lw=1.0, zorder=4,
                   label="no longer selected")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Published {label} (meV per atom)")
        ax.set_ylabel(f"Recomputed {label} (meV per atom)")
        ax.grid(True, which="major", **GRID)
        ax.set_aspect("equal")
        panel_label(ax, letter)
    axes[0].legend(loc="upper left")
    save(fig, "fig1_database_reproducibility")


FAMILY_STYLE = {  # family: (marker, face grey, label)
    "spinels": ("o", "0.85", "Spinels"),
    "RE titanate pyrochlores": ("s", "0.55", "RE titanate pyrochlores"),
    "titanates and zirconates": ("D", "0.35", "Alkaline-earth titanates/zirconates"),
    "La perovskites": ("^", "0.15", "La perovskites"),
    "RE orthophosphates": ("v", "white", "RE orthophosphates"),
    "Li cathode oxides": ("P", "0.6", "LiMnO$_2$"),
}


def figure2():
    """(a) DFT error against the measured enthalpy of formation from the oxides; (b) family mean offset by method."""
    b = (pd.read_csv(RES / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
         .drop_duplicates(["compound", "measured_meV_atom"]))
    m = pd.read_csv(RES / "mlip_vs_measured.csv")
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 82 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    x, y = b.measured_meV_atom.values, b.error_meV_atom.values
    k, c = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 20, 60, 50)
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.plot(xs, k * xs + c, color="0.1", lw=1.0, ls="--", zorder=2,
            label=f"trend: DFT recovers {100 * (1 + k):.0f}% of measured")
    for fam, (mk, face, lab) in FAMILY_STYLE.items():
        s = b[b.family == fam]
        ax.scatter(s.measured_meV_atom, s.error_meV_atom, marker=mk, s=24, facecolor=face, edgecolor="0.05",
                   lw=0.6, zorder=3, label=lab)
    ax.set_xlabel("Measured enthalpy of formation from oxides (meV per atom)")
    ax.set_ylabel("DFT minus measured (meV per atom)")
    ax.grid(True, **GRID)
    ax.set_ylim(-60, 140)
    # Below the axes: inside, the legend landed on the titanate and LiMnO2 markers.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.3, columnspacing=1.2)
    panel_label(ax, "a")

    ax = axes[1]
    fams = [f for f in ["spinels", "RE titanate pyrochlores", "titanates and zirconates", "La perovskites",
                        "RE orthophosphates"] if f in set(m.family)]
    methods = [("dft_error", "DFT", "o", "0.1"), ("orbinf_relaxed", "Orb-v3", "s", "0.5"),
               ("mace_relaxed", "MACE-MP-0", "^", "white")]
    ypos = np.arange(len(fams))[::-1]
    for j, (col, lab, mk, face) in enumerate(methods):
        vals = []
        for f in fams:
            g = m[m.family == f]
            err = g[col] if col == "dft_error" else g[col] - g.measured_meV_atom
            vals.append(err.mean())
        ax.scatter(vals, ypos + (j - 1) * 0.22, marker=mk, s=26, facecolor=face, edgecolor="0.05", lw=0.6,
                   zorder=3, label=lab)
    ax.axvline(0, color="0.55", lw=0.8, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([FAMILY_STYLE[f][2] for f in fams])
    ax.set_xlabel("Mean error against calorimetry (meV per atom)")
    ax.grid(True, axis="x", **GRID)
    ax.set_xlim(-25, 105)
    ax.legend(loc="upper right")
    panel_label(ax, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_calorimetry_benchmark")


def figure3():
    """Phase-purity gain of each selection against the computed margin it rested on."""
    d = pd.read_csv(RES / "chen_rerun_today.csv")
    band = 22.5
    margin = d.pub_invE_meV.abs().clip(lower=0.1)
    fig, ax = plt.subplots(figsize=(89 * MM, 80 * MM))
    ax.axvspan(0.1, band, color="0.9", zorder=0, lw=0)
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    inside = margin <= band
    ax.scatter(margin[inside], d.score_diff[inside], marker="o", s=22, facecolor="white", edgecolor="0.05",
               lw=0.7, zorder=3, label=f"margin within {band:g} meV per atom (n = {int(inside.sum())})")
    ax.scatter(margin[~inside], d.score_diff[~inside], marker="s", s=20, facecolor="0.25", edgecolor="0.05",
               lw=0.5, zorder=3, label=f"margin beyond (n = {int((~inside).sum())})")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 400)
    ax.set_ylim(-0.3, 0.9)
    ax.set_xlabel("Inverse hull energy of the selected pair (meV per atom)")
    ax.set_ylabel("Phase-purity gain over conventional recipe")
    ax.grid(True, which="major", **GRID)
    # Legend below the axes: every corner of the plot holds data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    rho = spearmanr(margin, d.score_diff).statistic
    print(f"figure 3 Spearman rho = {rho:.2f}")
    save(fig, "fig3_synthesis_outcome_vs_margin")


def figure5():
    """(a) Metastability precision of every decision in both datasets; (b) A-Lab outcome against precision;
    (c) calibration against measured stability decisions; (d) observed oxides resolved and expected right, by family."""
    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[~alab.database_inconsistency.astype(bool)]  # scored as in precision_analysis.py
    col = "p_stability_measured_scatter"
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(183 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1.05], hspace=0.5)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax = axes[0]
    groups = [("Chen et al., precursor pair", chen.p_selection_measured_scatter, "o", "0.2"),
              ("Chen et al., target stability", chen[col], "s", "0.6"),
              ("A-Lab, target stability", alab[col], "^", "white")]
    for i, (lab, vals, mk, face) in enumerate(groups):
        y = len(groups) - 1 - i + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(vals.clip(lower=0.003), y, marker=mk, s=18, facecolor=face, edgecolor="0.05", lw=0.5, zorder=3)
    ax.axvline(1, color="0.1", lw=0.9, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 70)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g[0]}\n{int((g[1] > 1).sum())} of {len(g[1])} above 1" for g in groups][::-1])
    ax.set_xlabel("Metastability precision (margin / 22.5 meV per atom)")
    ax.grid(True, axis="x", which="major", **GRID)
    panel_label(ax, "a")

    ax = axes[1]
    order = [("obtained", "o", "0.2"), ("partial", "D", "0.55"), ("inconclusive", "s", "0.85"),
             ("not obtained", "v", "white")]
    for i, (out, mk, face) in enumerate(order):
        vals = alab[alab.outcome == out][col]
        y = len(order) - 1 - i + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(vals.clip(lower=0.003), y, marker=mk, s=18, facecolor=face, edgecolor="0.05", lw=0.5, zorder=3)
        ax.plot([vals.median()] * 2, [len(order) - 1 - i - 0.32, len(order) - 1 - i + 0.32], color="0.1", lw=1.4,
                zorder=4)
    ax.axvline(1, color="0.1", lw=0.9, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 70)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{o[0]} (n = {int((alab.outcome == o[0]).sum())})" for o in order][::-1])
    ax.set_xlabel("Metastability precision of target stability")
    ax.grid(True, axis="x", which="major", **GRID)
    panel_label(ax, "b")

    # c: does precision predict whether measurement confirms the computed decision? (precision_calibration.py)
    cal = pd.read_csv(RES / "precision_calibration_bins.csv")
    ax = axes[2]
    ax.plot([0, 1.0], [0, 1.0], color="0.5", lw=0.8, zorder=1)
    ax.errorbar(cal.predicted, cal.observed, yerr=[cal.observed - cal.ci_low, cal.ci_high - cal.observed], fmt="o",
                ms=5, mfc="white", mec="0.05", ecolor="0.3", elinewidth=0.8, capsize=2.5, zorder=3)
    ax.set_xlim(0, 1.02)  # PNAS: numerical axes start at zero
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Predicted probability that the decision is right")
    ax.set_ylabel("Share confirmed by measurement")
    ax.grid(True, **GRID)
    panel_label(ax, "c")

    # d: observed oxides by family, share resolved (bars) and share expected right (diamonds)
    from scipy.stats import norm
    mp = pd.read_csv(RES / "precision_mp_observed.csv")
    ox = mp[mp.oxide.astype(bool)].assign(right=lambda d: norm.cdf(d.margin_meV.abs() / 22.5))
    stats = (ox.groupby("family").agg(n=("p", "size"), pct=("resolved", lambda s: 100 * s.mean()),
                                      right=("right", lambda s: 100 * s.mean())).sort_values("pct"))
    ax = axes[3]
    y = np.arange(len(stats))
    ax.barh(y, stats.pct, height=0.62, color="0.6", edgecolor="0.1", lw=0.5, zorder=3, label="resolved, p > 1")
    ax.plot(stats.right, y, ls="none", marker="D", ms=4.5, mfc="white", mec="0.05", mew=0.7, zorder=4,
            label="expected right")
    ax.axvline(100 * ox.resolved.mean(), color="0.1", lw=0.9, ls="--", zorder=2)
    ax.axvline(100 * ox.right.mean(), color="0.1", lw=0.9, ls=":", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n.capitalize()} ({int(r.n):,})" for n, r in stats.iterrows()], fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(stats) - 0.4)
    ax.set_xlabel("Observed oxides (%)")
    ax.grid(True, axis="x", **GRID)
    ax.legend(loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, fontsize=6.5)
    panel_label(ax, "d")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.08, wspace=0.75)
    save(fig, "fig5_metastability_precision")


def tex_formula(f):
    """Mg3(PO4)2 -> Mg$_3$(PO$_4$)$_2$ for mathtext labels."""
    return re.sub(r"(\d+)", r"$_{\1}$", f)


MEV_PER_J_MOL = 1000 / 96485.33  # T dS in J/mol atoms to meV per atom


def figure4(path=RES / "hermes_vs_measured.csv"):
    """Hermes vibrational entropy from the Sol MD against measured entropy, and the error as T dS, with the harmonic
    phonons of the same potential beside it."""
    if not path.exists():
        print("figure 4 skipped: no", path.name)
        return
    h = pd.read_csv(path)
    styles = {298.15: dict(marker="o", mfc="white", mec="0.1", label="298 K"),
              1000.0: dict(marker="s", mfc="0.25", mec="0.1", label="1000 K")}
    # Panel b: MD and harmonic at each temperature, so each row holds up to four markers at fixed offsets.
    series = {("md", 298.15): (-0.27, dict(styles[298.15], label="MD, 298 K")),
              ("harmonic", 298.15): (-0.09, dict(marker="^", mfc="white", mec="0.1", label="harmonic, 298 K")),
              ("md", 1000.0): (0.09, dict(styles[1000.0], label="MD, 1000 K")),
              ("harmonic", 1000.0): (0.27, dict(marker="v", mfc="0.25", mec="0.1", label="harmonic, 1000 K"))}
    rx_path = path.with_name("hermes_reaction_entropies.csv")
    hp_path, hr_path = RES / "harmonic_vs_measured.csv", RES / "harmonic_reaction_entropy.csv"
    rx = pd.read_csv(rx_path).dropna(subset=["TdS_rxn_measured_meV_atom"]) if rx_path.exists() else None
    hp = pd.read_csv(hp_path).dropna(subset=["S_measured"]) if hp_path.exists() else None
    hr = pd.read_csv(hr_path) if hr_path.exists() else None
    targets = set()
    if rx is not None:
        rx["target"] = rx.reaction.str.split(" from ").str[0]
        targets |= set(rx.target)
    if hr is not None:
        targets |= set(hr.compound)
    # a and b are row charts (42 oxides, then each phase and reaction); c and d are square panels beneath them.
    fig = plt.figure(figsize=(183 * MM, 222 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.75, 1], width_ratios=[1, 1.15])
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_c, ax_d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    measured_reaction_entropy(ax_a)
    panel_label(ax_a, "a")

    ax = ax_c
    lo, hi = 0.0, 1.05 * h[["S_hermes", "S_measured"]].max().max()  # PNAS: numerical axes start at zero
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, zorder=1)
    for t, st in styles.items():
        s = h[h.T_K.round(2) == t]
        ax.plot(s.S_measured, s.S_hermes, ls="none", ms=5, mew=0.6, zorder=3, **st)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_anchor("N")  # level with panel d beside it
    ax.set_xlabel("Measured entropy (J K$^{-1}$ mol$^{-1}$ of atoms)")
    ax.set_ylabel("Hermes entropy (J K$^{-1}$ mol$^{-1}$ of atoms)")
    ax.grid(True, **GRID)
    ax.legend(loc="upper left")
    panel_label(ax, "c")

    ax = ax_b
    # By the 298 K value, which every phase has (La2Zr2O7 has no measured 1000 K entropy).
    order = h[h.T_K.round(2) == 298.15].sort_values("S_measured").compound.tolist()
    y = {c: i for i, c in enumerate(order)}
    ax.axvspan(-22.5, 22.5, color="0.9", zorder=0, lw=0)
    ax.axvline(0, color="0.5", lw=0.8, zorder=1)

    labelled = set()

    def mark(x, rows, key):
        dy, st = series[key]
        if key in labelled:
            st = {k: v for k, v in st.items() if k != "label"}
        labelled.add(key)
        ax.plot(x, [r + dy for r in rows], ls="none", ms=4, mew=0.6, zorder=3, **st)

    for t in styles:
        s = h[h.T_K.round(2) == t]
        mark(s.TdS_meV_atom, [y[c] for c in s.compound], ("md", t))
        if hp is not None:
            s = hp[(hp.T_K.round(2) == t) & hp.phase.isin(y)]
            mark(s.TdS_meV_atom, [y[c] for c in s.phase], ("harmonic", t))
    labels = [tex_formula(c) for c in order]
    if targets:
        # The quantity a stability decision uses: the entropy of reaction, where much of the per-phase error cancels.
        meas298 = {}
        if hr is not None:
            meas298.update(zip(hr.compound, hr.measured_dS298))
        if rx is not None:
            for c, g in rx[rx.T_K.round(2) == 298.15].groupby("target"):
                meas298.setdefault(c, g.dS_rxn_measured.iloc[0])
        targets = sorted(targets, key=lambda c: meas298[c])
        yr = {c: len(order) + 0.4 + i for i, c in enumerate(targets)}
        ax.axhline(len(order) - 0.3, color="0.3", lw=0.6, zorder=1)
        if rx is not None:
            for t in styles:
                s = rx[rx.T_K.round(2) == t]
                mark(s.TdS_rxn_hermes_meV_atom - s.TdS_rxn_measured_meV_atom, [yr[c] for c in s.target], ("md", t))
        if hr is not None:
            for t, col in ((298.15, "error_dS298"), (1000.0, "error_dS1000")):
                s = hr.dropna(subset=[col])  # no 1000 K measured value for La2Zr2O7 or Na2SiO3
                mark(t * s[col] * MEV_PER_J_MOL, [yr[c] for c in s.compound], ("harmonic", t))
        ax.set_yticks(list(range(len(order))) + list(yr.values()),
                      labels + [tex_formula(c) + " from oxides" for c in targets])
        ax.set_ylim(-0.6, max(yr.values()) + 0.6)
    else:
        ax.set_yticks(range(len(order)), labels)
        ax.set_ylim(-0.6, len(order) - 0.4)
    span = max(30.0, 1.1 * h.TdS_meV_atom.abs().max())  # symmetric, so both edges of the scatter band show
    ax.set_xlim(-span, span)
    ax.set_xlabel("T × (computed minus measured entropy) (meV per atom)")
    ax.grid(True, axis="x", **GRID)
    ax.legend(loc="lower left")  # the phase rows hold no negative errors beyond a few meV per atom
    panel_label(ax, "b")
    decision_margins_at_temperature(ax_d)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save(fig, "fig4_finite_temperature_entropy")


def decision_margins_at_temperature(ax, path=RES / "precision_temperature.csv"):
    """Panel d: each campaign decision's margin at its firing temperature against its 0 K margin."""
    if not path.exists():
        ax.set_visible(False)
        return
    d = pd.read_csv(path)
    lin = 30.0  # linear within +-30 meV per atom, where decisions can change sign; logarithmic beyond
    ax.set_xscale("symlog", linthresh=lin, linscale=2.0)
    ax.set_yscale("symlog", linthresh=lin, linscale=2.0)
    lim = (-400, 60)
    # Shaded: the quadrants where the decision changes sign at temperature. Dashed: the 22.5 meV per atom scatter.
    ax.fill_between([lim[0], 0], 0, lim[1], color="0.9", lw=0, zorder=0)
    ax.fill_between([0, lim[1]], lim[0], 0, color="0.9", lw=0, zorder=0)
    for v in (-22.5, 22.5):
        ax.axvline(v, color="0.45", lw=0.6, ls="--", zorder=1)
        ax.axhline(v, color="0.45", lw=0.6, ls="--", zorder=1)
    ax.axhline(0, color="0.5", lw=0.6, zorder=1)
    ax.axvline(0, color="0.5", lw=0.6, zorder=1)
    ax.plot(lim, lim, color="0.5", lw=0.8, zorder=1)
    groups = [(("Chen2024", "selection"), dict(marker="^", mfc="white", mec="0.1", label="Chen et al., selection")),
              (("Chen2024", "stability"), dict(marker="o", mfc="0.65", mec="0.1", label="Chen et al., stability")),
              (("A-Lab2023", "stability"), dict(marker="s", mfc="0.2", mec="0.1", label="A-Lab, stability"))]
    for (ds, kind), st in groups:
        g = d[(d.dataset == ds) & (d.decision == kind)]
        ax.plot(g.margin_0K, g.margin_final, ls="none", ms=4, mew=0.5, zorder=3, **st)
    md = d[d.entropy_source == "MD"]
    ax.plot(md.margin_0K, md.margin_final, ls="none", marker="o", ms=8, mfc="none", mec="0.1", mew=0.6, zorder=2,
            label="entropy from MD")
    ticks = [-300, -30, -10, 0, 10, 30]
    ax.set_xticks(ticks, [str(t).replace("-", "−") for t in ticks])
    ax.set_yticks(ticks, [str(t).replace("-", "−") for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlabel("Margin at 0 K (meV per atom)")
    ax.set_ylabel("Margin at the firing temperature (meV per atom)")
    ax.grid(True, **GRID)
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.3)
    panel_label(ax, "d")


FORMERS = ["Al", "W", "Mo", "Ti", "Si", "B", "P", "V", "Nb", "Ge"]  # after the other cations: Fe3Al2Si3O12, PbMoO4


def plain_formula(f):
    """KAl(SiO3)2 -> KAlSi2O6, written as mineral formulas are: other cations, then oxyanion formers, then O."""
    from pymatgen.core import Composition
    amounts = Composition(f).reduced_composition.get_el_amt_dict()
    cations = sorted((e for e in amounts if e != "O" and e not in FORMERS),
                     key=lambda e: Composition(e).elements[0].X)
    formers = [e for e in FORMERS if e in amounts]
    return "".join(e + (f"{amounts[e]:g}" if amounts[e] != 1 else "") for e in [*cations, *formers, "O"] if e in amounts)


def measured_reaction_entropy(ax):
    """Measured entropy of formation from the binary oxides, times temperature, for 42 oxides (Fig. 4a)."""
    d = pd.read_csv(RES / "reaction_entropy_measured.csv").sort_values("TdS_1000K_meV_atom")
    y = np.arange(len(d))
    ax.axvspan(-22.5, 22.5, color="0.9", zorder=0, lw=0)
    ax.axvline(0, color="0.5", lw=0.8, zorder=1)
    for i in y:
        ax.plot([d.TdS_1000K_meV_atom.iloc[i], d.TdS_1300K_meV_atom.iloc[i]], [i, i], color="0.6", lw=0.6, zorder=2)
    ax.plot(d.TdS_1000K_meV_atom, y, ls="none", marker="o", ms=4.5, mfc="white", mec="0.1", mew=0.6, zorder=3,
            label="1000 K")
    ax.plot(d.TdS_1300K_meV_atom, y, ls="none", marker="s", ms=4, mfc="0.25", mec="0.1", mew=0.6, zorder=3,
            label="1300 K")
    ax.set_yticks(y, [tex_formula(plain_formula(c)) for c in d.compound], fontsize=6.5)
    ax.set_ylim(-0.8, len(d) - 0.2)
    span = 1.08 * max(d.TdS_1300K_meV_atom.abs().max(), 30)
    ax.set_xlim(-span, span)
    ax.set_xlabel("T × entropy of formation from the oxides (meV per atom)")
    ax.grid(True, axis="x", **GRID)
    ax.legend(loc="lower right")


def extended_data_1():
    """DFT error against compiled experimental enthalpies of formation from the binary oxides, 120 compounds."""
    b = pd.read_csv(RES / "benchmark_compiled.csv")
    fig, ax = plt.subplots(figsize=(120 * MM, 85 * MM))
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.scatter(b.measured_meV_atom, b.error_meV_atom, s=14, marker="o", facecolor="0.8", edgecolor="0.3", lw=0.4,
               zorder=2, label=f"compiled values (n = {len(b)})")
    edges = [-10000, -400, -200, -100, -50, 10000]
    cats = pd.cut(b.measured_meV_atom, edges)
    g = b.groupby(cats, observed=True).agg(x=("measured_meV_atom", "median"), m=("error_meV_atom", "mean"),
                                           se=("error_meV_atom", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
    ax.errorbar(g.x, g.m, yerr=g.se, fmt="s", ms=6, color="0.05", mfc="0.05", capsize=3, lw=1.0, zorder=4,
                label="bin mean with standard error")
    ax.set_xlabel("Measured enthalpy of formation from oxides (meV per atom)")
    ax.set_ylabel("DFT minus measured (meV per atom)")
    ax.grid(True, **GRID)
    ax.legend(loc="lower left")  # upper right holds the two largest positive errors
    save(fig, "extended_data_fig1_compiled_benchmark")


# Binary oxides whose cation has an open 3d shell; CrO3, V2O5 and TiO2 are d0 and stay out.
OPEN_SHELL_3D = {"Cr2O3", "MnO", "Mn2O3", "FeO", "Fe2O3", "CoO", "NiO", "CuO", "Cu2O"}


def extended_data_3():
    """GGA/GGA+U and r2SCAN against calorimetry, split by whether the compound holds an open-shell 3d metal."""
    b = pd.read_csv(RES / "benchmark_r2scan_calorimetric.csv")
    b["tm"] = b.oxides.map(lambda s: any(p.split(" ", 1)[1] in OPEN_SHELL_3D for p in s.split("+")))
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 80 * MM), sharey=True)
    for ax, (sub, letter) in zip(axes, [(b[~b.tm], "a"), (b[b.tm], "b")]):
        x = sub.measured_meV_atom.values
        ax.axhline(0, color="0.55", lw=0.8, zorder=1)
        for col, lab, mk, face, ls in (("error_meV_atom", "GGA/GGA+U", "o", "0.8", ":"),
                                       ("r2scan_error_meV_atom", "r2SCAN", "s", "0.1", "--")):
            y = sub[col].values
            k, c = np.polyfit(x, y, 1)
            sd = np.std(y - (k * x + c), ddof=2)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, k * xs + c, color="0.1", lw=0.9, ls=ls, zorder=2)
            ax.scatter(x, y, marker=mk, s=20, facecolor=face, edgecolor="0.05", lw=0.5, zorder=3,
                       label=f"{lab}: recovers {100 * (1 + k):.0f}%, scatter {sd:.0f}")
        ax.set_xlabel("Measured enthalpy of formation from oxides (meV per atom)")
        ax.grid(True, **GRID)
        ax.legend(loc="lower left")  # the upper left of panel a holds the largest GGA errors
        panel_label(ax, letter)
    axes[0].set_ylabel("Computed minus measured (meV per atom)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "extended_data_fig2_r2scan")


def extended_data_4():
    """How far cation disorder can move each A-Lab margin at the synthesis temperature (Orb-v3 ordering energies)."""
    d = pd.read_csv(RES / "configurational_ordering.csv").sort_values("stability_margin_meV")
    fig, ax = plt.subplots(figsize=(120 * MM, 150 * MM))
    y = np.arange(len(d))
    ax.axvspan(-22.5, 22.5, color="0.9", zorder=0, lw=0)
    ax.axvline(0, color="0.5", lw=0.8, zorder=1)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.stability_margin_meV, r.margin_at_T_best], [i, i], color="0.55", lw=0.7, zorder=2)
    ax.plot(d.stability_margin_meV, y, ls="none", marker="o", ms=4.5, mfc="white", mec="0.1", mew=0.6, zorder=4,
            label="0 K")
    ax.plot(d.margin_at_T_worst, y, ls="none", marker="s", ms=4, mfc="0.25", mec="0.1", mew=0.6, zorder=3,
            label="with the certain disorder gain")
    ax.plot(d.margin_at_T_best, y, ls="none", marker="|", ms=6, mew=1.0, color="0.1", zorder=3,
            label="with the largest possible gain")
    ax.set_yticks(y, [tex_formula(plain_formula(c)) for c in d.target], fontsize=6.5)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("Energy relative to the hull of competing phases (meV per atom)")
    ax.grid(True, axis="x", **GRID)
    ax.legend(loc="lower right")  # every margin is near or below the hull, so the right edge is clear
    save(fig, "extended_data_fig4_configurational")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    extended_data_1()
    extended_data_3()
    extended_data_4()
    print("written:", sorted(p.name for p in OUT.glob("fig*.png")))
