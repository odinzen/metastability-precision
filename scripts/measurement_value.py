"""Which calorimetric measurements would make the most computed decisions right?

A decision's margin is the energy of a reaction among its phases: a stability decision sets the target against its
decomposition products on the current hull, a precursor selection sets the target against its two precursors. The
binary oxides are anchored already (the database's fitted corrections are fit to their measured formation enthalpies),
so the error of the margin sits with the phases of three or more elements, e = sum_f w_f eps_f, with signed reaction
weights w (target +1, products and precursors minus their atom fractions). Their errors share a part within the
decision, as in the robustness draws (chen_ranking_robustness.py):

    cov(eps_f, eps_g) = s^2 (rho + (1 - rho) [f == g]),   rho = 0.65,

with s set so that var(e) is the decision's sigma (sigma_model.py). A measurement of compound c returns
y_c = eps_c + noise of sigma_cal. Before it is made its outcome is unknown, so the value of a set of measurements is
the expected probability, over the outcomes, that the decision is right once they are in:

    E[P_right] = E over m' ~ N(m, var(e) - v) of Phi(|m'| / sqrt(v)),

with m the computed margin, v the variance of e left after conditioning on the measured y, and var(e) - v the spread
of the updated margin. The expected number of resolved decisions is P(|m'| > sqrt(v)) summed the same way.
Measurements are chosen one at a time for the largest gain in expected right decisions. The compounds the compiled
literature has already measured are entered the same way, as measurements with sigma_cal.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_model import sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SIGMA_CAL, RHO = 5.0, 0.65
N_NEW = 30
# Expectation over a standard normal on a fine grid; Gauss-Hermite nodes miss the kink of |m'| at zero.
NODES = np.linspace(-8.0, 8.0, 4001)
WEIGHTS = norm.pdf(NODES) / norm.pdf(NODES).sum()


def red(f):
    return Composition(f).reduced_formula


def stability_reaction(chemsys, target):
    entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
    t = red(target)
    others = PhaseDiagram([e for e in entries if e.composition.reduced_formula != t])
    weights = {t: 1.0}
    for e, frac in others.get_decomposition(Composition(target)).items():
        f = e.composition.reduced_formula
        weights[f] = weights.get(f, 0.0) - float(frac)
    return weights


def decisions():
    today = pd.read_csv(RES / "mp_targets_today.csv").set_index(["dataset", "target"])
    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[(alab.dataset == "A-Lab2023") & (alab.database_inconsistency != True)]
    out = []
    for ds, df in (("Chen2024", chen), ("A-Lab2023", alab)):
        for _, r in df.dropna(subset=["stability_margin_meV"]).iterrows():
            w = stability_reaction(today.loc[(ds, r.target), "chemsys"], r.target)
            out.append(dict(label=f"{ds} stability {r.target}", target=r.target, margin=abs(r.stability_margin_meV),
                            weights=w))
    for _, r in chen.dropna(subset=["selection_margin_meV"]).iterrows():
        w = {red(r.target): 1.0}
        for f in r.robot_pair.split("+"):
            w[red(f)] = w.get(red(f), 0.0) - 0.5
        out.append(dict(label=f"Chen2024 selection {r.target}", target=r.target, margin=abs(r.selection_margin_meV),
                        weights=w))
    for d in out:  # only phases of three or more elements carry the uncertainty
        d["weights"] = {f: x for f, x in d["weights"].items() if len(Composition(f).elements) >= 3}
        d["sigma"] = sigma_for(d["target"])
    return out


def expected(dec, measured, sigma_cal=SIGMA_CAL, rho=RHO):
    """Expected P(right) and P(resolved) of one decision after the measurements in `measured`."""
    names = list(dec["weights"])
    m, sig = dec["margin"], dec["sigma"]
    if not names:
        p = m / sig
        return norm.cdf(p), float(p > 1)
    w = np.array([dec["weights"][f] for f in names])
    corr = rho + (1 - rho) * np.eye(len(names))
    var_unit = w @ corr @ w
    if var_unit <= 0:
        p = m / sig
        return norm.cdf(p), float(p > 1)
    C = corr * sig ** 2 / var_unit  # per-phase covariance scaled so that var(e) = sigma^2
    idx = [i for i, f in enumerate(names) if f in measured]
    if not idx:
        p = m / sig
        return norm.cdf(p), float(p > 1)
    cov_ey = (w @ C)[idx]
    S = C[np.ix_(idx, idx)] + sigma_cal ** 2 * np.eye(len(idx))
    v = max(sig ** 2 - cov_ey @ np.linalg.solve(S, cov_ey), 1e-9)
    spread = np.sqrt(max(sig ** 2 - v, 0.0))
    updated = m + spread * NODES
    p_right = float(WEIGHTS @ norm.cdf(np.abs(updated) / np.sqrt(v)))
    p_resolved = float(norm.sf((np.sqrt(v) - m) / spread) + norm.cdf((-np.sqrt(v) - m) / spread)) if spread > 0 else float(m > np.sqrt(v))
    return p_right, p_resolved


def score(decs, measured, **kw):
    pr = [expected(d, measured, **kw) for d in decs]
    return sum(p for p, _ in pr), sum(r for _, r in pr)


def main():
    decs = decisions()
    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    literature = {red(r[0]) for r in d["data"]} | {red(c) for c in pd.read_csv(RES / "benchmark_dft_vs_measured.csv").compound}
    phases = {f for x in decs for f in x["weights"]}
    known = phases & literature

    rows = []
    base_right, base_res = score(decs, set())
    rows.append(dict(step=0, measured="none (as computed)", expected_right=base_right, expected_resolved=base_res))
    lit_right, lit_res = score(decs, known)
    rows.append(dict(step=0, measured=f"enter the {len(known)} already measured", expected_right=lit_right,
                     expected_resolved=lit_res))
    chosen = set(known)
    candidates = sorted(phases - known)
    touches = {c: [i for i, x in enumerate(decs) if c in x["weights"]] for c in candidates}
    current = [expected(x, chosen)[0] for x in decs]
    for k in range(1, N_NEW + 1):
        # Only the decisions a compound enters change when it is measured.
        gains = {c: sum(expected(decs[i], chosen | {c})[0] - current[i] for i in touches[c])
                 for c in candidates if c not in chosen}
        best = max(gains, key=gains.get)
        chosen.add(best)
        for i in touches[best]:
            current[i] = expected(decs[i], chosen)[0]
        right, res = score(decs, chosen)
        n_dec = len(touches[best])
        rows.append(dict(step=k, measured=best, decisions_entered=n_dec, gain=gains[best], expected_right=right,
                         expected_resolved=res))
    out = pd.DataFrame(rows).round(2)
    out.to_csv(RES / "measurement_value.csv", index=False)
    print(f"{len(decs)} decisions; {len(phases)} phases of three or more elements, {len(known)} already measured")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))
    sens = []
    for name, kw in (("sigma_cal 3", dict(sigma_cal=3.0)), ("sigma_cal 10", dict(sigma_cal=10.0)),
                     ("rho 0", dict(rho=0.0))):
        r0, s0 = score(decs, set(), **kw)
        r_lit, s_lit = score(decs, known, **kw)
        r_all, s_all = score(decs, chosen, **kw)
        sens.append(dict(case=name, right_computed=r0, right_literature=r_lit, right_with_new=r_all,
                         resolved_computed=s0, resolved_literature=s_lit, resolved_with_new=s_all))
    sens = pd.DataFrame(sens).round(2)
    sens.to_csv(RES / "measurement_value_sensitivity.csv", index=False)
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
