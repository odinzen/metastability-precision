"""Is metastability precision a calibrated confidence? Three checks.

1. sigma: the scatter of DFT about the calorimetric trend (69 values), and how far it moves when each family is left
   out of the fit.
2. Calibration against measured truth: for the 120 decomposition reactions of the compiled benchmark
   (sigma_by_reaction_type.py), the sign of the DFT reaction energy is the computed stability decision and the sign of
   the measured one the truth. With an unbiased Gaussian error of width sigma, the decision is right with probability
   Phi(p), p = |margin| / sigma; observed agreement is compared with that prediction bin by bin. The compiled data
   played no part in setting sigma.
3. Expected correct decisions, the sum of Phi(p), for the observed oxides and the two synthesis datasets.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SIGMA = 22.5
BINS = [0, 0.5, 1, 2, 4, np.inf]
rng = np.random.default_rng(20261024)


def trend_sigma(b):
    k, c = np.polyfit(b.measured_meV_atom, b.error_meV_atom, 1)
    return float(np.std(b.error_meV_atom - (k * b.measured_meV_atom + c), ddof=2))


def sigma_by_family():
    b = pd.read_csv(RES / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    b = b.drop_duplicates(["compound", "measured_meV_atom"])  # as figure 2: one LaFeO3 value is listed twice; 69 values
    rows = [dict(left_out="none", n=len(b), sigma=trend_sigma(b))]
    for fam in sorted(b.family.unique()):
        rest = b[b.family != fam]
        rows.append(dict(left_out=fam, n=len(rest), sigma=trend_sigma(rest)))
    out = pd.DataFrame(rows).round(2)
    out.to_csv(RES / "precision_sigma_by_family.csv", index=False)
    print(out.to_string(index=False))
    return out


def calibration():
    d = pd.read_csv(RES / "sigma_by_reaction_type.csv")
    d = d[d.measured_meV_atom != 0].copy()  # a measured value exactly on the hull has no sign to recover
    d["p"] = d.dft_meV_atom.abs() / SIGMA
    d["predicted"] = norm.cdf(d.p)
    d["right"] = (np.sign(d.dft_meV_atom) == np.sign(d.measured_meV_atom)).astype(int)
    d.to_csv(RES / "precision_calibration_decisions.csv", index=False)
    rows = []
    for iv, g in d.groupby(pd.cut(d.p, BINS), observed=True):
        ci = binomtest(int(g.right.sum()), len(g)).proportion_ci(method="wilson")
        rows.append(dict(p_low=iv.left, p_high=iv.right, n=len(g), observed=g.right.mean(), ci_low=ci.low, ci_high=ci.high,
                         predicted=g.predicted.mean(), p_mean=g.p.mean()))
    bins = pd.DataFrame(rows).round(3)
    bins.to_csv(RES / "precision_calibration_bins.csv", index=False)
    print(bins.to_string(index=False))
    brier = ((d.predicted - d.right) ** 2).mean()
    base = ((d.right.mean() - d.right) ** 2).mean()
    lo, hi = d[d.p <= 1], d[d.p > 1]
    gaps = []
    for _ in range(20000):
        i = rng.integers(0, len(d), len(d))
        a, pp = d.right.values[i], d.p.values[i]
        if (pp <= 1).any() and (pp > 1).any():
            gaps.append(a[pp > 1].mean() - a[pp <= 1].mean())
    ci = lambda g: binomtest(int(g.right.sum()), len(g)).proportion_ci(method="wilson")
    print(f"\n{len(d)} decisions: right {d.right.mean():.3f} overall; p <= 1 {lo.right.mean():.3f} "
          f"({ci(lo).low:.2f}-{ci(lo).high:.2f}, n={len(lo)}), p > 1 {hi.right.mean():.3f} "
          f"({ci(hi).low:.2f}-{ci(hi).high:.2f}, n={len(hi)}); gap 95% interval {np.percentile(gaps, 2.5):.2f} to "
          f"{np.percentile(gaps, 97.5):.2f}; Brier {brier:.3f} against {base:.3f} for the base rate")
    for kind, g in d.groupby("kind"):
        print(f"  {kind:40s} n={len(g):3d} right at p <= 1 {g.right[g.p <= 1].mean():.2f} (n={int((g.p <= 1).sum())}),"
              f" p > 1 {g.right[g.p > 1].mean():.2f} (n={int((g.p > 1).sum())})")
    print("  wrong at p > 1:", ", ".join(hi[hi.right == 0].compound))


def expected_correct():
    rows = []
    ox = pd.read_csv(RES / "precision_mp_observed.csv")
    ox = ox[ox.oxide]
    P = norm.cdf(ox.margin_meV.abs() / SIGMA)
    res = ox.margin_meV.abs() > SIGMA
    rows.append(dict(set="observed oxides", n=len(ox), resolved=int(res.sum()), expected_right=P.sum(),
                     expected_right_resolved=P[res].mean(), expected_right_unresolved=P[~res].mean(),
                     expected_wrong=(1 - P).sum(), expected_wrong_in_unresolved=(1 - P)[~res].sum()))
    for comp, g in ox.groupby("competitor"):
        Pg = norm.cdf(g.margin_meV.abs() / SIGMA)
        rows.append(dict(set=f"observed oxides, {comp}", n=len(g), resolved=int((g.margin_meV.abs() > SIGMA).sum()),
                         expected_right=Pg.sum(), expected_wrong=(1 - Pg).sum()))
    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[(alab.dataset == "A-Lab2023") & (alab.database_inconsistency != True)]
    for name, m in (("Chen et al. selections", chen.selection_margin_meV), ("Chen et al. stability", chen.stability_margin_meV),
                    ("A-Lab stability", alab.stability_margin_meV)):
        m = m.dropna().abs()
        Pm = norm.cdf(m / SIGMA)
        rows.append(dict(set=name, n=len(m), resolved=int((m > SIGMA).sum()), expected_right=Pm.sum(),
                         expected_wrong=(1 - Pm).sum()))
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
