"""How well DFT recognizes measured-metastable compounds, scored from the measured side.

Every other calibration in this repository conditions on the computed margin c and asks how often its sign holds.
Here the condition is the measured margin t: a compound measured t above its competing phases (metastable, t > 0) or
below them (stable, t < 0). If the computed margin is t plus an error of spread sigma and no bias, DFT gives the
measured sign with probability Phi(|t| / sigma), whatever the share of stable compounds in the set, so the base rate
that confounds the flat-prior forecast drops out. Each family is scored on both sides: decisions, computed signs
right, the share Phi(|t| / sigma) expects, the mean error DFT minus measured, and sigma fitted to that side alone.
A gap between the two sides' errors is a bias toward one sign (DFT stabilizing metastable compounds, for example),
which a symmetric error cannot produce.

Families: oxide compound and open-shell decisions (calibration_base_rate.compiled), intermetallic decisions
(calibration_intermetallic.decisions; Gaussian sigma and the Student-t of heavy_tail), chalcogenide and nitride
reactions from the binaries (calibration_nonoxide.csv), the non-intermetallic compiled non-oxides
(calibration_compiled_nonoxide.csv, formulas not already in the intermetallic set), and any verified additions from the
literature search in data/curated/metastable_additions.csv. Writes results/metastable_side.csv, results/metastable_side_bias.csv and results/metastable_threshold_sensitivity.csv.

The measured-metastable intermetallics are also split by where assessed phase diagrams place them
(data/curated/intermetallic_phase_diagrams.csv). A compound that is an equilibrium phase lies below every combination
of the other phases of its system, so a measured margin above them is a measurement error, not a DFT miss; only the
phases stable at high temperature alone carry a measured label the diagram supports. Writes
results/metastable_intermetallic_diagrams.csv (one row per compound) and results/metastable_intermetallic_split.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import norm, t as student, ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_base_rate import compiled  # noqa: E402
from calibration_intermetallic import decisions as intermetallic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ADDITIONS = ROOT / "data" / "curated" / "metastable_additions.csv"
DIAGRAMS = ROOT / "data" / "curated" / "intermetallic_phase_diagrams.csv"
GRID = np.arange(2.0, 400.0, 0.5)
T_NU, T_SCALE = 1.5, 46.0


def red(f):
    return Composition(f).reduced_formula


def families():
    """One frame per family: formula, measured margin t, computed margin c, per-decision sigma, measurement u."""
    out = {}
    ox = compiled()
    a = pd.read_csv(RES / "truth_audit.csv").set_index("compound")
    ox["u"] = ox.compound.map(a.truth_u)
    for reg in ("compound", "open-shell"):
        g = ox[ox.regime == reg]
        out[f"oxide, {reg}"] = pd.DataFrame(dict(formula=g.formula, t=g.truth, c=g.dft_meV_atom, sigma=g.sigma, u=g.u))
    im = intermetallic()
    out["intermetallic"] = pd.DataFrame(dict(formula=im.formula, t=im.measured_meV_atom, c=im.dft_meV_atom,
                                             sigma=138.4, u=im.u))
    nx = pd.read_csv(RES / "calibration_nonoxide.csv")
    out["chalcogenide and nitride"] = pd.DataFrame(dict(formula=nx.formula, t=nx.measured_meV_atom, c=nx.dft_meV_atom,
                                                        sigma=22.5, u=np.nan))
    cn = pd.read_csv(RES / "calibration_compiled_nonoxide.csv")
    seen = set(im.formula.map(red))
    cn = cn[(cn.family != "intermetallic") & ~cn.compound.map(red).isin(seen)]
    out["other compiled non-oxide"] = pd.DataFrame(dict(formula=cn.compound, t=cn.measured_meV_atom,
                                                        c=cn.dft_meV_atom, sigma=22.5, u=cn.u_meas_meV_atom))
    if ADDITIONS.exists():
        # The search found measured-metastable compounds only; each joins the family whose stable side it shares.
        ad = pd.read_csv(ADDITIONS)
        home = {"chalcogenide": "chalcogenide and nitride", "carbide, nitride or hydride": "other compiled non-oxide",
                "halide": "other compiled non-oxide", "intermetallic": "intermetallic"}
        for _, x in ad.iterrows():
            fam = (("oxide, open-shell" if x.sigma > 22.5 else "oxide, compound") if x.family == "oxide"
                   else home[x.family])
            row = pd.DataFrame([dict(formula=x.formula, t=x.measured_meV_atom, c=x.dft_meV_atom, sigma=x.sigma,
                                     u=x.u_meV_atom, source="literature search")])
            out[fam] = pd.concat([out[fam], row], ignore_index=True)
    for k, v in out.items():
        if "source" not in v:
            v["source"] = "benchmark"
        v["source"] = v.source.fillna("benchmark")
    return {k: v[(v.t.abs() >= 0.5) & v.c.notna()].reset_index(drop=True) for k, v in out.items()}


def sigma_side(t, right):
    """Maximum-likelihood sigma for one side, P(right) = Phi(|t| / sigma)."""
    P = norm.cdf(np.abs(t)[None, :] / GRID[:, None])
    ll = np.log(np.clip(np.where(right[None, :], P, 1 - P), 1e-12, 1)).sum(1)
    inside = GRID[ll >= ll.max() - 1.92]
    hi = inside.max() if inside.max() < GRID[-1] - 1 else np.inf
    return GRID[ll.argmax()], inside.min(), hi


def side(name, g, label, heavy=False):
    right = np.sign(g.c) == np.sign(g.t)
    e = g.c - g.t
    expected = (student.cdf(g.t.abs() / T_SCALE, T_NU) if heavy else norm.cdf(g.t.abs() / g.sigma)).mean()
    s, lo, hi = sigma_side(g.t.values, right.values)
    certain = g.u.notna() & (g.t.abs() >= 2 * g.u)
    return dict(family=name, side=label, forecast="Student t" if heavy else "Gaussian", n=len(g),
                right=int(right.sum()), share_right=right.mean(), expected=expected,
                median_measured=g.t.median(), mean_error=e.mean(), median_error=e.median(),
                sigma_side=s, sigma_side_lo=lo, sigma_side_hi=hi,
                n_certain=int(certain.sum()), right_certain=int(right[certain].sum()))


def noise_null(g, draw_error, n_sim=4000, seed=1):
    """Could measurement noise alone make DFT look biased on the metastable side? Conditioning on a noisy measured
    margin t > 0 selects measurements that erred upward, so c - t leans negative even for an unbiased DFT. Null: the
    measured margins as truth, fresh measurement noise of each decision's u, DFT error from the family's model; the
    observed share of measured-metastable compounds DFT calls metastable, and the gap in mean error between the two
    sides, are set against their null distributions."""
    rng = np.random.default_rng(seed)
    u = g.u.fillna(g.u.median()).fillna(0).values
    t = g.t.values
    meta = g.t > 0
    obs_rec = ((g.c > 0) & meta).sum() / meta.sum()
    obs_gap = (g.c - g.t)[meta].mean() - (g.c - g.t)[~meta].mean()
    recs, gaps = [], []
    for _ in range(n_sim):
        tm = t + rng.normal(0, u)
        c = t + draw_error(rng, len(t))
        m = tm > 0
        if m.sum() < 2 or (~m).sum() < 2:
            continue
        recs.append(((c > 0) & m).sum() / m.sum())
        gaps.append((c - tm)[m].mean() - (c - tm)[~m].mean())
    recs, gaps = np.array(recs), np.array(gaps)
    return dict(recognized=obs_rec, recognized_null=np.median(recs), recognized_null_lo=np.percentile(recs, 2.5),
                recognized_null_hi=np.percentile(recs, 97.5), p_recognized=np.mean(recs <= obs_rec),
                gap=obs_gap, gap_null=np.median(gaps), gap_null_lo=np.percentile(gaps, 2.5),
                gap_null_hi=np.percentile(gaps, 97.5), p_gap=np.mean(gaps <= obs_gap))


def thresholds(fams):
    """How the counts move with the definition of a miss: labels at all, beyond 1u, beyond 2u; a call counted metastable
    at a computed margin above 0, -10 and -22.5 meV/atom."""
    rows = []
    for name, g in fams.items():
        u = g.u.fillna(g.u.median()).fillna(10).values
        meta = (g.t > 0).values
        for lab, k in (("all", 0), ("beyond 1u", 1), ("beyond 2u", 2)):
            sel = meta & (g.t.values >= k * u)
            if sel.any():
                rows.append(dict(family=name, labels=lab, n=int(sel.sum()),
                                 **{f"called_above_{tau:g}": int((g.c.values[sel] > -tau).sum()) for tau in (0, 10, 22.5)}))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "metastable_threshold_sensitivity.csv", index=False)
    return out


ERROR_MODELS = {"oxide, compound": lambda r, n: r.normal(0, 22.5, n),
                "oxide, open-shell": lambda r, n: r.normal(0, 34.0, n),
                "intermetallic": lambda r, n: T_SCALE * r.standard_t(T_NU, n),
                "chalcogenide and nitride": lambda r, n: r.normal(0, 22.5, n),
                "other compiled non-oxide": lambda r, n: r.normal(0, 22.5, n)}


def diagram_split(fams):
    """Recognition of the measured-metastable intermetallics by their status on assessed phase diagrams."""
    g = fams["intermetallic"]
    d = pd.read_csv(DIAGRAMS)
    d["key"] = d.formula.map(red)
    m = g[g.t > 0].assign(key=lambda x: x.formula.map(red))
    m = m.merge(d.drop(columns="formula"), on="key", how="left", validate="one_to_one")
    assert m.status.notna().all() and len(m) == len(d), set(m.key) ^ set(d.key)
    u = m.u.fillna(g.u.median())
    m["clear_cut"] = m.t >= 2 * u
    m.drop(columns="key").to_csv(RES / "metastable_intermetallic_diagrams.csv", index=False)
    model = ERROR_MODELS["intermetallic"]
    groups = {"all": m.index, "equilibrium phase": m.index[m.status == "equilibrium phase"],
              "not an equilibrium phase": m.index[m.status != "equilibrium phase"],
              "stable only at high temperature": m.index[m.status.str.startswith("high-temperature")]}
    rows = []
    for name, idx in groups.items():
        sub = pd.concat([g[g.t <= 0], m.loc[idx, g.columns]])
        r = noise_null(sub, model)
        n = len(idx)
        rows.append(dict(group=name, n=n, called=int((m.loc[idx].c > 0).sum()),
                         clear_cut=int(m.loc[idx].clear_cut.sum()), expected=r["recognized_null"] * n,
                         expected_lo=r["recognized_null_lo"] * n, expected_hi=r["recognized_null_hi"] * n,
                         p_recognized=r["p_recognized"]))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "metastable_intermetallic_split.csv", index=False)
    return out


def main():
    rows, bias = [], []
    for name, g in families().items():
        meta, stab = g[g.t > 0], g[g.t < 0]
        for label, x in (("measured metastable", meta), ("measured stable", stab)):
            if len(x):
                rows.append(side(name, x, label))
                if name == "intermetallic":
                    rows.append(side(name, x, label, heavy=True))
        if len(meta) >= 3 and len(stab) >= 3:
            em, es = meta.c - meta.t, stab.c - stab.t
            model = ERROR_MODELS.get(name, lambda r, n: r.normal(0, float(g.sigma.median()), n))
            bias.append(dict(family=name, n_metastable=len(meta), mean_error_metastable=em.mean(),
                             mean_error_stable=es.mean(), welch_p=ttest_ind(em, es, equal_var=False).pvalue,
                             **noise_null(g, model)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "metastable_side.csv", index=False)
    thresholds(families())
    split = diagram_split(families())
    pd.DataFrame(bias).to_csv(RES / "metastable_side_bias.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(out.to_string(index=False))
        print(pd.DataFrame(bias).to_string(index=False))
        print(split.to_string(index=False))


if __name__ == "__main__":
    main()
