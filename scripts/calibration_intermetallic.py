"""The intermetallic regime, scored like the oxide regimes of calibration_regimes.py: decisions from the direct-synthesis
calorimetry of Kim et al. 2017 (results/benchmark_nonoxide_calorimetric*.csv, each compound against the measured hull
of the other measured phases of its system), p at sigma 22.5, outcome = the measured sign. Also scored at the fitted sigma,
and with only decisions measured at least 2u from zero (u: the stated uncertainty of the compound, times sqrt 2 for
the product side). Writes results/calibration_intermetallic.csv, results/calibration_intermetallic_t.csv and the summary table behind Supplementary Table 6."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_regimes import bootstrap, sigma_ml, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SIGMA = 22.5


def decisions(sigma=SIGMA):
    d = pd.read_csv(RES / "benchmark_nonoxide_calorimetric_neighbours.csv")
    d = d[d.measured_meV_atom != 0].copy()
    b = pd.read_csv(RES / "benchmark_nonoxide_calorimetric.csv")
    sd = b.groupby("formula").sd_meV_atom.median()
    d["u"] = d.formula.map(sd).fillna(sd.median()) * np.sqrt(2)
    d["system"] = d.formula.map(lambda f: "-".join(sorted(str(e) for e in Composition(f).elements)))
    d["p"] = d.dft_meV_atom.abs() / sigma
    d["pred"] = norm.cdf(d.p)
    d["q"] = (np.sign(d.dft_meV_atom) == np.sign(d.measured_meV_atom)).astype(float)
    d["certain"] = d.measured_meV_atom.abs() >= 2 * d.u
    return d


def row(g, label):
    s = stats(g, "q")
    s.update(labels=label, regime="intermetallic", systems=g.system.nunique())
    s["sigma_ml"], s["sigma_ml_lo"], s["sigma_ml_hi"] = sigma_ml(g.q.values, g.dft_meV_atom.abs().values)
    return s


def heavy_tail(d):
    """The Gaussian sigma is stretched by a few large misses (robust SD 49 against SD 102). A Student-t forecast,
    P = T_nu(|margin| / scale), fitted by maximum likelihood on a grid of nu, shows how much of sigma is tail."""
    from scipy.stats import t as student
    q, m = d.q.values, d.dft_meV_atom.abs().values
    grid = np.arange(5.0, 400.0, 1.0)
    rows = []
    for nu in (1, 1.5, 2, 3, 5, 10, 30, np.inf):
        P = norm.cdf(m[None, :] / grid[:, None]) if np.isinf(nu) else student.cdf(m[None, :] / grid[:, None], nu)
        ll = np.log(np.clip(np.where(q > 0.5, P, 1 - P), 1e-12, 1)).sum(1)
        scale = grid[ll.argmax()]
        inside = grid[ll >= ll.max() - 1.92]
        pb = P[ll.argmax()]
        low = m / scale <= 1
        rows.append(dict(nu=nu, scale=scale, scale_lo=inside.min(), scale_hi=inside.max(), log_likelihood=ll.max(),
                         n_low=int(low.sum()), share_low=q[low].mean(), pred_low=pb[low].mean(),
                         share_high=q[~low].mean(), pred_high=pb[~low].mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "calibration_intermetallic_t.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


def paired():
    """Compounds measured both ways: how far the two calorimetric methods disagree on the same phase."""
    b = pd.read_csv(RES / "benchmark_nonoxide_calorimetric.csv")
    b = b[~b.method.str.contains("Acetic")]
    b["reduced"] = b.formula.map(lambda f: Composition(f).reduced_formula)
    b["how"] = np.where(b.method == "Direct synthesis", "direct", "metal")
    w = b.groupby(["reduced", "space_group", "how"]).measured_meV_atom.mean().unstack().dropna()
    diff = w.metal - w.direct
    print(f"measured both ways: {len(w)} compounds, liquid-metal minus direct mean {diff.mean():+.1f}, "
          f"sd {diff.std(ddof=1):.1f}, median |diff| {diff.abs().median():.1f} meV/atom")


def main():
    d = decisions()
    rows = []
    s = row(d, "audited")
    bt = bootstrap(d, "q")
    for k in ("share_low", "share_high", "auc", "sigma_ml"):
        s[f"{k}_ci"] = f"{bt[k].quantile(0.025):.2f} to {bt[k].quantile(0.975):.2f}"
    rows.append(s)
    rows.append(row(d[d.certain], "|truth| >= 2u"))
    for lab, g in (("no Cr, Mn, Fe, Co, Ni", d[~d.magnetic]), ("with Cr, Mn, Fe, Co or Ni", d[d.magnetic]),
                   ("compounds only among the products", d[~d.with_element])):
        rows.append(row(g, lab))
    # Does the fitted sigma depend on how the compound was measured? Solute-solvent drop and the metal baths dissolve
    # the sample in a liquid metal; the three acetic-acid values belong to neither group.
    direct = d.method == "Direct synthesis"
    metal = d.method.str.startswith("Solute") | (d.method.str.startswith("Solution") & ~d.method.str.contains("Acetic"))
    for lab, g in (("direct synthesis", d[direct]), ("liquid-metal solution", d[metal]),
                   ("direct synthesis, every term", d[direct & d.one_method]),
                   ("liquid-metal solution, every term", d[metal & d.one_method]),
                   ("direct synthesis, systems with a liquid-metal value", d[direct & d.system.isin(d[metal].system)])):
        rows.append(row(g, lab))
    paired()
    heavy_tail(d)
    fitted = rows[0]["sigma_ml"]
    at_fit = decisions(sigma=fitted)
    s = stats(at_fit, "q")
    s.update(labels=f"p at the fitted sigma ({fitted:.0f})", regime="intermetallic", systems=at_fit.system.nunique())
    rows.append(s)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "calibration_intermetallic.csv", index=False)
    cols = ["labels", "n", "systems", "n_low", "share_low", "pred_low", "n_high", "share_high", "pred_high", "auc",
            "cox_slope", "spiegelhalter_z", "sigma_ml", "sigma_ml_lo", "sigma_ml_hi"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(out[[c for c in cols if c in out]].to_string(index=False))
        print(out[[c for c in out.columns if c.endswith("_ci")]].iloc[:1].to_string(index=False))


if __name__ == "__main__":
    main()
