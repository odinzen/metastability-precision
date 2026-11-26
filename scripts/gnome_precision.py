"""Metastability precision of the oxides that GNoME predicts stable (Merchant et al. 2023, released summaries), by regime.

A GNoME claim is that a new crystal lies below the convex hull of the known phases. Its margin is the released
decomposition energy, against the Materials Project snapshot hull ("Decomposition Energy Per Atom") or against the
Materials Project plus OQMD ("... MP OQMD"). Claims with a margin at or above zero against that hull are counted but not
scored. Oxides are compounds with O and none of C, N, S, H, Se, Te or a halogen, as in precision_mp_observed.py.

Regimes (sigma_model.py): a composition with an experimentally observed Materials Project entry makes the claim a new
polymorph below a known one, a polymorph decision with the systematic GGA density bias and no Gaussian probability.
The others are compound decisions (sigma 22.5) or open-shell decisions (sigma 34).

Three probabilities per scored claim. The flat-prior one is Phi(|margin| / sigma). GNoME kept candidates whose
computed margin came out negative, so a computed margin may overstate the true one (the winner's curse), and the true
margin is the computed one plus a shift s toward the hull: P = Phi(-(margin + s) / sigma).

The exponential-prior shift takes s from Tweedie's formula, sigma^2 times the slope of the log density of computed
margins, which below zero falls off nearly exponentially (rate lambda = 1 / mean |margin|), so s = sigma^2 lambda.

GNoME's own r2SCAN recalculations test that shift. If the computed margins near the hull were overstated by s, the
r2SCAN margins of the same claims would sit above them by s times the share of the GGA error that r2SCAN does not
share, (sigma_g^2 - cov) / sigma_g^2, which the two r2SCAN benchmarks give from the residuals of both functionals about
their trends. The measured r2SCAN-minus-GGA difference at the hull (intercept of a line in the margin, claims within
60 meV/atom of the hull, bootstrap 95% interval) is first corrected for the offset the two functionals show at the
hull with no selection involved (gnome_functional_offset.py, from the known-answer decisions of the compiled
benchmark): r2SCAN binds ternary oxides more strongly, so it sits below GGA there. What remains, divided by each
share, is the shift the recalculations imply, which is set against the exponential-prior s.

Data: gs://gdm_materials_discovery/gnome_data (CC BY-NC 4.0) in data/raw/gnome; observed MP oxide formulas in
data/raw/mp/mp_oxide_formulas_2026.09.28.csv.gz. Writes results/gnome_precision.csv (row level, not tracked),
results/gnome_precision_summary.csv, results/gnome_precision_r2scan.csv and results/gnome_selection_check.csv.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precision_mp_observed import NON_OXIDE_ANIONS, family  # noqa: E402
from gnome_functional_offset import intercept, offsets  # noqa: E402
from sigma_model import SIGMA_CLOSED, SIGMA_OPEN, open_shell  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gnome"
HULLS = {"MP snapshot": "Decomposition Energy Per Atom", "MP + OQMD": "Decomposition Energy Per Atom MP OQMD"}
SIGMA_SETS = {"calorimetric (22.5, 34)": (SIGMA_CLOSED, SIGMA_OPEN), "compound fit (13.6, 34)": (13.6, SIGMA_OPEN),
              "hold-out (27, 34)": (27.0, SIGMA_OPEN), "open-shell fit (22.5, 66)": (SIGMA_CLOSED, 65.8),
              "one sigma (22.5, 22.5)": (SIGMA_CLOSED, SIGMA_CLOSED),
              "open-shell fit without CeCrO3 (22.5, 25)": (SIGMA_CLOSED, 25.2)}
BINS = [0, 0.5, 1, 2, 4, np.inf]
NEAR_HULL = -60.0  # meV/atom; window of the r2SCAN shift fit
R2SCAN_BENCHMARKS = {"calorimetric": ("benchmark_r2scan_calorimetric.csv", "error_meV_atom", "r2scan_error_meV_atom"),
                     "compiled": ("benchmark_r2scan_compiled.csv", "gga_error", "r2scan_error")}


def oxides():
    cols = ["MaterialId", "Reduced Formula", "Elements", "NSites"] + list(HULLS.values())
    g = pd.read_csv(RAW / "stable_materials_summary.csv", usecols=cols, dtype={"MaterialId": str})
    els = g.Elements.map(ast.literal_eval)
    keep = els.map(lambda e: "O" in e and not set(e) & NON_OXIDE_ANIONS)
    g = g[keep].copy()
    g["family"] = els[keep].map(family)
    mp = pd.read_csv(ROOT / "data" / "raw" / "mp" / "mp_oxide_formulas_2026.09.28.csv.gz")
    observed = set(mp.loc[~mp.theoretical, "formula"])
    g["open_shell"] = g["Reduced Formula"].map(open_shell)
    g["regime"] = np.where(g["Reduced Formula"].isin(observed), "polymorph",
                           np.where(g.open_shell, "open-shell", "compound"))
    for h, col in HULLS.items():
        g[f"margin {h}"] = 1000 * g[col]
    return g


def decay_rate(margins):
    """Rate of the exponential fall-off of the computed margins below zero (maximum likelihood: 1 / mean |margin|)."""
    return 1 / np.abs(margins[margins < 0]).mean()


def visible_share():
    """(sigma_g^2 - cov) / sigma_g^2 per regime and benchmark: the part of the GGA error a GGA-to-r2SCAN difference sees."""
    out = {}
    for name, (f, gcol, rcol) in R2SCAN_BENCHMARKS.items():
        b = pd.read_csv(ROOT / "results" / f).dropna(subset=[gcol, rcol])
        b["regime"] = np.where(b.compound.map(open_shell), "open-shell", "compound")
        for reg, x in b.groupby("regime"):
            m = x.measured_meV_atom.values
            eg, er = (x[c].values - np.polyval(np.polyfit(m, x[c].values, 1), m) for c in (gcol, rcol))
            out[(reg, name)] = dict(n=len(x), sigma_gga=eg.std(ddof=2), sigma_r2scan=er.std(ddof=2),
                                    corr=np.corrcoef(eg, er)[0, 1], share=1 - np.cov(eg, er)[0, 1] / eg.var(ddof=1))
    return out


def r2scan_shift(m, d, n_boot=2000, seed=0):
    """r2SCAN-minus-GGA margin at the hull: intercept of d on m for claims within NEAR_HULL, with a bootstrap interval."""
    near = m > NEAR_HULL
    m, d = m[near], d[near]
    rng = np.random.default_rng(seed)
    boot = [np.polyfit(m[i], d[i], 1)[1] for i in (rng.integers(0, len(m), len(m)) for _ in range(n_boot))]
    return len(m), np.polyfit(m, d, 1)[1], *np.percentile(boot, [2.5, 97.5])


def score(g, hull, sig_c, sig_o):
    m = g[f"margin {hull}"]
    claim = (m < 0) & (g.regime != "polymorph")
    sigma = np.where(g.regime == "open-shell", sig_o, sig_c)
    p = (m.abs() / sigma).where(claim)
    flat = pd.Series(norm.cdf(p), index=g.index).where(claim)
    eb = pd.Series(np.nan, index=g.index)
    for reg, s in (("compound", sig_c), ("open-shell", sig_o)):
        sel = claim & (g.regime == reg)
        eb[sel] = norm.cdf(-(m[sel] + s ** 2 * decay_rate(m[sel].values)) / s)
    return claim, p, flat, eb


def selection_check(g):
    """The winner's-curse shift that GNoME's r2SCAN recalculations imply once the functional offset is removed."""
    r = pd.read_csv(RAW / "stable_materials_r2scan.csv", usecols=["MaterialId", "Decomposition Energy Per Atom"],
                    dtype={"MaterialId": str})
    x = g[(g["margin MP snapshot"] < 0) & (g.regime != "polymorph")].merge(
        r.rename(columns={"Decomposition Energy Per Atom": "r2"}), on="MaterialId")
    share = visible_share()
    off = offsets()
    rows = []
    for reg in ("compound", "open-shell"):
        y = x[x.regime == reg]
        m, d = y["margin MP snapshot"].values, 1000 * y.r2.values - y["margin MP snapshot"].values
        n, c, lo, hi = r2scan_shift(m, d)
        lam = decay_rate(g.loc[(g.regime == reg) & (g["margin MP snapshot"] < 0), "margin MP snapshot"].values)
        sig = SIGMA_OPEN if reg == "open-shell" else SIGMA_CLOSED
        shares = {k[1]: v["share"] for k, v in share.items() if k[0] == reg}
        o, o_lo, o_hi = intercept(off[off.open_shell == (reg == "open-shell")])
        visible = c - o
        rows.append(dict(regime=reg, n_r2scan=len(y), n_near_hull=n, shift_at_hull=round(c, 2), shift_lo95=round(lo, 2),
                         shift_hi95=round(hi, 2), exponential_shift=round(sig ** 2 * lam, 1),
                         exponential_visible_lo=round(sig ** 2 * lam * min(shares.values()), 1),
                         exponential_visible_hi=round(sig ** 2 * lam * max(shares.values()), 1),
                         visible_share_calorimetric=round(shares["calorimetric"], 2),
                         visible_share_compiled=round(shares["compiled"], 2), offset_no_selection=round(o, 1),
                         offset_lo95=round(o_lo, 1), offset_hi95=round(o_hi, 1), visible_after_offset=round(visible, 1),
                         implied_shift_lo=round(visible / max(shares.values()), 1),
                         implied_shift_hi=round(visible / min(shares.values()), 1)))
    bench = pd.DataFrame([dict(regime=k[0], benchmark=k[1], **{a: round(b, 3) for a, b in v.items()})
                          for k, v in share.items()])
    return pd.DataFrame(rows), bench


def main():
    g = oxides()
    sel_check, bench = selection_check(g)
    rows = []
    for hull in HULLS:
        m = g[f"margin {hull}"]
        for sname, (sc, so) in SIGMA_SETS.items():
            claim, p, flat, eb = score(g, hull, sc, so)
            for subset, sel in (("all scored", claim), ("compound", claim & (g.regime == "compound")),
                                ("open-shell", claim & (g.regime == "open-shell"))):
                rows.append(dict(hull=hull, sigma=sname, subset=subset, n=int(sel.sum()),
                                 resolved_share=round((p[sel] > 1).mean(), 3),
                                 expected_right_flat=round(flat[sel].mean(), 3), expected_right_eb=round(eb[sel].mean(), 3)))
            if sname.startswith("calorimetric"):
                g[f"p {hull}"], g[f"flat {hull}"], g[f"eb {hull}"] = p, flat, eb
        rows.append(dict(hull=hull, sigma="", subset="counts", n=len(g),
                         resolved_share=np.nan, expected_right_flat=np.nan, expected_right_eb=np.nan,
                         polymorph=int((g.regime == "polymorph").sum()),
                         not_below_hull=int(((m >= 0) & (g.regime != "polymorph")).sum())))
    s = pd.DataFrame(rows)
    fam = (g[g["flat MP snapshot"].notna()].groupby("family")
           .agg(n=("flat MP snapshot", "size"), resolved=("p MP snapshot", lambda x: round((x > 1).mean(), 3)),
                expected_right=("flat MP snapshot", lambda x: round(x.mean(), 3)),
                expected_right_eb=("eb MP snapshot", lambda x: round(x.mean(), 3))).reset_index())
    fam["hull"], fam["sigma"], fam["subset"] = "MP snapshot", "calorimetric (22.5, 34)", "family: " + fam.family
    s = pd.concat([s, fam.rename(columns={"resolved": "resolved_share", "expected_right": "expected_right_flat"})
                   .drop(columns="family")], ignore_index=True)
    s.to_csv(ROOT / "results" / "gnome_precision_summary.csv", index=False)
    g.to_csv(ROOT / "results" / "gnome_precision.csv", index=False)
    sel_check.to_csv(ROOT / "results" / "gnome_selection_check.csv", index=False)

    # GNoME's own r2SCAN recalculations: share of scored claims that keep their sign, by bin of p (consistency check).
    r = pd.read_csv(RAW / "stable_materials_r2scan.csv", usecols=["MaterialId", "Decomposition Energy Per Atom"],
                    dtype={"MaterialId": str})
    x = g[g["flat MP snapshot"].notna()].merge(r.rename(columns={"Decomposition Energy Per Atom": "r2"}), on="MaterialId")
    x["kept"] = x.r2 < 0
    x["bin"] = pd.cut(x["p MP snapshot"], BINS, include_lowest=True)
    chk = (x.groupby(["regime", "bin"], observed=True)
           .agg(n=("kept", "size"), kept_r2scan=("kept", "mean"), predicted=("flat MP snapshot", "mean"),
                predicted_eb=("eb MP snapshot", "mean"))
           .round(3).reset_index())
    chk.to_csv(ROOT / "results" / "gnome_precision_r2scan.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(s.to_string(index=False))
        print(f"\nr2SCAN consistency: {len(x)} scored claims recomputed by GNoME, {x.kept.mean():.3f} keep their sign")
        print(chk.to_string(index=False))
        print("\nSelection check (meV/atom):")
        print(sel_check.to_string(index=False))
        print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
