"""Which calorimetric measurement would make the most computed decisions right?

A decision's margin is the energy of a reaction among its phases: a stability decision sets the target against its
decomposition products on the current hull, a precursor selection sets the target against its two precursors. The
binary oxides are anchored already (the database's fitted corrections are fit to their measured formation enthalpies),
so the uncertainty of the margin sits with the phases of three or more elements. It is shared among them by squared
reaction weight w (atom fraction; the target has weight 1):

    sigma_decision^2 = sigma^2 * U + sigma_cal^2 * (1 - U),   U = sum(w^2 over unmeasured) / sum(w^2 over all)

so an unmeasured decision keeps sigma = 22.5 meV/atom, the calibrated value, and a fully measured one falls to the
calorimetric uncertainty sigma_cal. The probability that the decision is right is Phi(|margin| / sigma_decision)
(precision_calibration.py). Two gains are counted: anchoring the compounds the compiled literature has already
measured, and new measurements chosen one at a time, each the compound that adds the most expected right decisions.

This is an approximation: it ignores correlation between the errors of related compounds, and sigma_cal is a typical
oxide-melt calorimetric uncertainty rather than one per compound.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SIGMA, SIGMA_CAL = 22.5, 5.0
N_NEW = 30


def red(f):
    return Composition(f).reduced_formula


def stability_reaction(chemsys, target):
    entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
    t = red(target)
    others = PhaseDiagram([e for e in entries if e.composition.reduced_formula != t])
    weights = {t: 1.0}
    for e, frac in others.get_decomposition(Composition(target)).items():
        f = e.composition.reduced_formula
        weights[f] = weights.get(f, 0.0) + float(frac)
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
            out.append(dict(label=f"{ds} stability {r.target}", margin=abs(r.stability_margin_meV), weights=w))
    for _, r in chen.dropna(subset=["selection_margin_meV"]).iterrows():
        w = {red(r.target): 1.0}
        for f in r.robot_pair.split("+"):
            w[red(f)] = w.get(red(f), 0.0) + 0.5
        out.append(dict(label=f"Chen2024 selection {r.target}", margin=abs(r.selection_margin_meV), weights=w))
    for d in out:  # only phases of three or more elements carry the uncertainty
        d["weights"] = {f: x for f, x in d["weights"].items() if len(Composition(f).elements) >= 3}
    return out


def p_right(dec, measured):
    w2 = {f: x * x for f, x in dec["weights"].items()}
    total = sum(w2.values())
    u = sum(v for f, v in w2.items() if f not in measured) / total if total else 0.0
    sigma = np.sqrt(SIGMA ** 2 * u + SIGMA_CAL ** 2 * (1 - u))
    return norm.cdf(dec["margin"] / sigma), dec["margin"] > sigma


def score(decs, measured):
    pr = [p_right(d, measured) for d in decs]
    return sum(p for p, _ in pr), sum(r for _, r in pr)


def main():
    decs = decisions()
    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    literature = {red(r[0]) for r in d["data"]} | {red(c) for c in pd.read_csv(RES / "benchmark_dft_vs_measured.csv").compound}
    phases = {f for x in decs for f in x["weights"]}
    known = phases & literature

    rows = []
    base_right, base_res = score(decs, set())
    rows.append(dict(step=0, measured="none (as computed)", expected_right=base_right, resolved=base_res))
    lit_right, lit_res = score(decs, known)
    rows.append(dict(step=0, measured=f"anchor the {len(known)} already measured", expected_right=lit_right,
                     resolved=lit_res))
    chosen = set(known)
    candidates = sorted(phases - known)
    for k in range(1, N_NEW + 1):
        cur, _ = score(decs, chosen)
        gains = {c: score(decs, chosen | {c})[0] - cur for c in candidates if c not in chosen}
        best = max(gains, key=gains.get)
        chosen.add(best)
        right, res = score(decs, chosen)
        n_dec = sum(best in x["weights"] for x in decs)
        rows.append(dict(step=k, measured=best, decisions_entered=n_dec, gain=gains[best], expected_right=right,
                         resolved=res))
    out = pd.DataFrame(rows).round(2)
    out.to_csv(RES / "measurement_value.csv", index=False)
    print(f"{len(decs)} decisions; {len(phases)} phases of three or more elements, {len(known)} already measured")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))
    for s in (3.0, 10.0):  # sensitivity to the calorimetric uncertainty
        global SIGMA_CAL
        keep, SIGMA_CAL = SIGMA_CAL, s
        r_lit, _ = score(decs, known)
        r_all, res_all = score(decs, chosen)
        print(f"sigma_cal {s:4.1f}: literature anchors {r_lit:.1f}, plus the {N_NEW} new {r_all:.1f} right, {res_all} resolved")
        SIGMA_CAL = keep


if __name__ == "__main__":
    main()
