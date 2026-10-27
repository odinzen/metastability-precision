"""Is metastability precision a calibrated confidence? The checks behind the statistics section.

1. sigma: the scatter of DFT about the calorimetric trend (69 values), how far it moves when each family is left
   out, and whether the residuals behave as a Gaussian of that width (coverage of one and two sigma; the error of the
   compiled reactions against their size).
2. Calibration against measured truth: for the decomposition reactions of the compiled benchmark
   (sigma_by_reaction_type.py) and the measured-metastable polymorph pairs (polymorph_decisions.py), the sign of the
   DFT reaction energy is the computed stability decision and the sign of the measured one the truth. With an unbiased Gaussian error of width sigma, the decision is right with probability
   Phi(p), p = |margin| / sigma, sigma from sigma_model.py (22.5 closed-shell, 43 open-shell). Reported: agreement by
   bin and by chemistry, discrimination (AUC, Fisher test of p > 1 against p <= 1), the always-stable baseline, and the
   maximum-likelihood sigma with its profile interval. The compiled data played no part in setting 22.5; the
   open-shell 43 is the lower end of the interval fitted on these decisions, so their agreement is not an
   independent test.
3. Expected correct decisions, the sum of Phi(p), for the observed oxides and the two synthesis datasets.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_model import SIGMA_CLOSED, open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BINS = [0, 0.5, 1, 2, 4, np.inf]
SIZE_BINS = [0, 25, 75, 150, np.inf]
SIGMA_GRID = np.arange(2.0, 2000.0, 0.1)
rng = np.random.default_rng(20261024)


def trend_residuals(b):
    k, c = np.polyfit(b.measured_meV_atom, b.error_meV_atom, 1)
    return b.error_meV_atom - (k * b.measured_meV_atom + c)


def trend_sigma(b):
    return float(np.std(trend_residuals(b), ddof=2))


def calorimetric_values():
    b = pd.read_csv(RES / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    return b.drop_duplicates(["compound", "measured_meV_atom"])  # as figure 2: one LaFeO3 value is listed twice; 69 values


def sigma_by_family():
    b = calorimetric_values()
    rows = [dict(left_out="none", n=len(b), sigma=trend_sigma(b))]
    for fam in sorted(b.family.unique()):
        rest = b[b.family != fam]
        rows.append(dict(left_out=fam, n=len(rest), sigma=trend_sigma(rest)))
    out = pd.DataFrame(rows).round(2)
    out.to_csv(RES / "precision_sigma_by_family.csv", index=False)
    print(out.to_string(index=False))
    return out


def error_model(d):
    """Coverage of the calorimetric residuals, by chemistry, and the compiled errors against reaction size."""
    b = calorimetric_values()
    r = trend_residuals(b)
    s = trend_sigma(b)
    b_open = b.compound.map(open_shell)
    rows = []
    for name, mask in (("calorimetric, all", b_open | ~b_open), ("calorimetric, closed-shell", ~b_open),
                       ("calorimetric, open-shell", b_open)):
        x = r[mask]
        rows.append(dict(set=name, n=len(x), sd=float(np.std(x, ddof=1)), mean=float(x.mean()),
                         within_1sigma=float((x.abs() <= s).mean()), within_2sigma=float((x.abs() <= 2 * s).mean())))
    for iv, g in d.groupby(pd.cut(d.measured_meV_atom.abs(), SIZE_BINS, right=False), observed=True):
        rows.append(dict(set=f"compiled, |measured| {iv.left:g} to {iv.right:g}", n=len(g),
                         sd=float(g.error_meV_atom.std()), mean=float(g.error_meV_atom.mean()),
                         within_1sigma=float((g.error_meV_atom.abs() <= SIGMA_CLOSED).mean()),
                         within_2sigma=float((g.error_meV_atom.abs() <= 2 * SIGMA_CLOSED).mean()),
                         wrong=int((g.right == 0).sum())))
    out = pd.DataFrame(rows).round(3)
    out.to_csv(RES / "precision_error_model.csv", index=False)
    print(out.to_string(index=False))


def ml_sigma(g):
    """Maximum-likelihood sigma from the right/wrong outcomes, with the interval where the log-likelihood is within
    1.92 of its maximum (95%)."""
    m = g.dft_meV_atom.abs().values[:, None]
    P = np.clip(norm.cdf(m / SIGMA_GRID[None, :]), 1e-12, 1 - 1e-12)
    y = g.right.values[:, None]
    ll = (y * np.log(P) + (1 - y) * np.log(1 - P)).sum(0)
    ok = SIGMA_GRID[ll >= ll.max() - 1.92]
    return SIGMA_GRID[ll.argmax()], ok.min(), ok.max()


def auc(g):
    right, wrong = g[g.right == 1].p, g[g.right == 0].p
    return mannwhitneyu(right, wrong).statistic / (len(right) * len(wrong))


def calibration():
    d = pd.read_csv(RES / "sigma_by_reaction_type.csv")
    d = d[d.measured_meV_atom != 0].copy()  # a measured value exactly on the hull has no sign to recover
    d["source"] = "compiled"
    # Measured-metastable polymorphs balance a set in which compilations list mostly stable compounds
    # (polymorph_decisions.py); only pairs whose measured difference fixes the sign enter.
    poly = pd.read_csv(RES / "polymorph_decisions.csv")
    poly = poly[poly.sign_established].assign(source="polymorph")
    d = pd.concat([d.assign(formula=d.compound), poly[[c for c in poly.columns if c in d.columns or c in ("source", "formula")]]],
                  ignore_index=True)
    d["open_shell"] = d.formula.map(open_shell)
    d["sigma"] = d.formula.map(sigma_for)
    d["p"] = d.dft_meV_atom.abs() / d.sigma
    d["predicted"] = norm.cdf(d.p)
    d["p_single_sigma"] = d.dft_meV_atom.abs() / SIGMA_CLOSED
    d["right"] = (np.sign(d.dft_meV_atom) == np.sign(d.measured_meV_atom)).astype(int)
    d.to_csv(RES / "precision_calibration_decisions.csv", index=False)

    rows = []
    comp = d[d.source == "compiled"]
    for name, g in (("all", d), ("closed-shell", d[~d.open_shell]), ("compiled, all", comp),
                    ("compiled, closed-shell", comp[~comp.open_shell])):
        for iv, b in g.groupby(pd.cut(g.p, BINS), observed=True):
            ci = binomtest(int(b.right.sum()), len(b)).proportion_ci(method="wilson")
            rows.append(dict(set=name, p_low=iv.left, p_high=iv.right, n=len(b), observed=b.right.mean(), ci_low=ci.low,
                             ci_high=ci.high, predicted=b.predicted.mean(), p_mean=b.p.mean()))
    bins = pd.DataFrame(rows).round(3)
    bins.to_csv(RES / "precision_calibration_bins.csv", index=False)
    print(bins.to_string(index=False))

    summary = []
    for name, g in (("all", d), ("closed-shell", d[~d.open_shell]), ("open-shell", d[d.open_shell]),
                    ("compiled, all", comp), ("compiled, closed-shell", comp[~comp.open_shell]),
                    ("compiled, open-shell", comp[comp.open_shell]),
                    ("against binary oxides", d[d.kind == "binary oxides only"]),
                    ("between compounds", d[d.kind == "includes a ternary or higher compound"]),
                    ("polymorph pairs", d[d.kind == "polymorph"]),
                    ("polymorph pairs, closed-shell", d[(d.kind == "polymorph") & ~d.open_shell])):
        lo, hi = g[g.p <= 1], g[g.p > 1]
        ci_lo = binomtest(int(lo.right.sum()), len(lo)).proportion_ci(method="wilson") if len(lo) else None
        ci_hi = binomtest(int(hi.right.sum()), len(hi)).proportion_ci(method="wilson") if len(hi) else None
        fisher = (fisher_exact([[hi.right.sum(), (hi.right == 0).sum()], [lo.right.sum(), (lo.right == 0).sum()]]).pvalue
                  if len(lo) and len(hi) else np.nan)
        s_ml, s_lo, s_hi = ml_sigma(g)
        aucs = []
        for _ in range(5000):
            b = g.iloc[rng.integers(0, len(g), len(g))]
            if 0 < b.right.sum() < len(b):
                aucs.append(auc(b))
        summary.append(dict(
            set=name, n=len(g), right=int(g.right.sum()), measured_stable=int((g.measured_meV_atom < 0).sum()),
            n_low=len(lo), right_low=int(lo.right.sum()), predicted_low=lo.predicted.sum(),
            ci_low_lo=ci_lo.low if ci_lo else np.nan, ci_low_hi=ci_lo.high if ci_lo else np.nan,
            n_high=len(hi), right_high=int(hi.right.sum()), predicted_high=hi.predicted.sum(),
            ci_high_lo=ci_hi.low if ci_hi else np.nan, ci_high_hi=ci_hi.high if ci_hi else np.nan,
            predicted_total=g.predicted.sum(), fisher_p=fisher, auc=auc(g) if 0 < g.right.sum() < len(g) else np.nan,
            auc_lo=np.percentile(aucs, 2.5) if aucs else np.nan, auc_hi=np.percentile(aucs, 97.5) if aucs else np.nan,
            brier=((g.predicted - g.right) ** 2).mean(), brier_base=((g.right.mean() - g.right) ** 2).mean(),
            sigma_ml=s_ml, sigma_ml_lo=s_lo, sigma_ml_hi=s_hi))
    summary = pd.DataFrame(summary)
    summary.to_csv(RES / "precision_calibration_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))
    print("  wrong at p > 1:", ", ".join(d[(d.p > 1) & (d.right == 0)].compound))
    print("  open-shell:", ", ".join(d[d.open_shell].compound))
    error_model(d[d.source == "compiled"])


def expected_correct():
    rows = []
    ox = pd.read_csv(RES / "precision_mp_observed.csv")
    ox = ox[ox.oxide].copy()
    ox["sigma"] = ox.formula.map(sigma_for)
    core = ox.competitor.isin(["fewer-element phases only", "same-order neighbour"])  # the decisions the calibration covers
    for name, g in (("observed oxides", ox), ("observed oxides, against other compounds", ox[core])):
        P = norm.cdf(g.margin_meV.abs() / g.sigma)
        res = g.margin_meV.abs() > g.sigma
        rows.append(dict(set=name, n=len(g), open_shell=int((g.sigma > SIGMA_CLOSED).sum()), resolved=int(res.sum()),
                         expected_right=P.sum(), expected_right_resolved=P[res].mean(),
                         expected_right_unresolved=P[~res].mean(), expected_wrong=(1 - P).sum(),
                         expected_wrong_in_unresolved=(1 - P)[~res].sum()))
    for comp, g in ox.groupby("competitor"):
        Pg = norm.cdf(g.margin_meV.abs() / g.sigma)
        rows.append(dict(set=f"observed oxides, {comp}", n=len(g), open_shell=int((g.sigma > SIGMA_CLOSED).sum()),
                         resolved=int((g.margin_meV.abs() > g.sigma).sum()), expected_right=Pg.sum(),
                         expected_wrong=(1 - Pg).sum()))
    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[(alab.dataset == "A-Lab2023") & (alab.database_inconsistency != True)]
    for name, df, col in (("Chen et al. selections", chen, "selection_margin_meV"),
                          ("Chen et al. stability", chen, "stability_margin_meV"),
                          ("A-Lab stability", alab, "stability_margin_meV")):
        g = df.dropna(subset=[col])
        m, s = g[col].abs(), g.target.map(sigma_for)
        Pm = norm.cdf(m / s)
        rows.append(dict(set=name, n=len(g), open_shell=int((s > SIGMA_CLOSED).sum()), resolved=int((m > s).sum()),
                         expected_right=Pm.sum(), expected_wrong=(1 - Pm).sum()))
    out = pd.DataFrame(rows).round(3)
    out.to_csv(RES / "precision_expected_correct.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.to_string(index=False))


if __name__ == "__main__":
    print("-- sigma about the calorimetric trend, each family left out in turn")
    sigma_by_family()
    print("\n-- calibration against measured stability decisions")
    calibration()
    print("\n-- expected correct decisions")
    expected_correct()
