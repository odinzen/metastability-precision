"""Metastability precision across every experimentally observed (ICSD) phase in the current Materials Project, the
population of Sun et al. 2016 on today's release: each phase's stability margin divided by the measured DFT scatter.
Stable phases use the decomposition enthalpy (depth below the hull of all other phases); metastable phases use the
energy above the hull. GGA/GGA+U thermo, as in the rest of the paper.

Oxides take sigma from sigma_model.py (22.5 meV/atom, or 43 with an open-shell 3d or 4f cation); for other
chemistries 22.5 is an extrapolation, reported separately. `--rescore` reapplies sigma to the saved table without
querying the database again, so the numbers stay on the release the paper used. The family offset is not added: within one phase diagram the target and its competitors share a
chemistry, so the proportional offset largely cancels between them while the random scatter does not.

Each phase is also labelled by what it competes with on the hull (its decomposition products): its own polymorph,
phases with fewer elements only (the reaction type the scatter was measured on), at least one neighbour with as many
elements (compound against compound; sigma_by_reaction_type.py measures that error), or a reaction involving an
element such as O2.
"""
import sys
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key  # noqa: E402
from sigma_model import SIGMA_CLOSED, sigma_for  # noqa: E402
from mp_api.client import MPRester  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "precision_mp_observed.csv"
ACID = {"P": "phosphates", "B": "borates", "Si": "silicates", "Ti": "titanates", "Ge": "germanates",
        "V": "vanadates", "Nb": "niobates", "W": "tungstates", "Mo": "molybdates"}
NON_OXIDE_ANIONS = {"C", "N", "S", "H", "Se", "Te", "F", "Cl", "Br", "I"}


def family(elements):
    els = set(elements)
    if "O" not in els:
        return "non-oxides"
    if els & NON_OXIDE_ANIONS:
        return "mixed-anion oxides"
    for el, name in ACID.items():
        if el in els:
            return name
    return "other oxides"


def competitor(formula, n_el, products):
    if not products:
        return "unknown"
    comps = [Composition(p.formula) for p in products]
    if len(comps) == 1 and comps[0].reduced_formula == Composition(formula).reduced_formula:
        return "own polymorph"
    if any(len(c.elements) == 1 for c in comps):
        return "involves an element"
    if all(len(c.elements) < n_el for c in comps):
        return "fewer-element phases only"
    return "same-order neighbour"


def main():
    with MPRester(api_key()) as mpr:
        observed = mpr.materials.summary.search(theoretical=False, fields=["material_id", "elements", "formula_pretty"])
        meta = {str(d.material_id): ([str(e) for e in d.elements], d.formula_pretty) for d in observed}
        ids = sorted(meta)
        thermo = []
        for i in range(0, len(ids), 1000):
            thermo += mpr.materials.thermo.search(
                material_ids=ids[i:i + 1000], thermo_types=["GGA_GGA+U"],
                fields=["material_id", "energy_above_hull", "is_stable", "decomposition_enthalpy", "decomposes_to",
                        "decomposition_enthalpy_decomposes_to"])
    rows = []
    for t in thermo:
        els, formula = meta[str(t.material_id)]
        if len(els) < 2:
            continue
        if t.is_stable and t.decomposition_enthalpy is not None:
            margin = 1000 * t.decomposition_enthalpy
            products = t.decomposition_enthalpy_decomposes_to
        else:
            margin = 1000 * (t.energy_above_hull or 0.0)
            products = t.decomposes_to
        rows.append(dict(mpid=str(t.material_id), formula=formula, nelements=len(els), stable=bool(t.is_stable),
                         margin_meV=margin, family=family(els), oxide="O" in els and not set(els) & NON_OXIDE_ANIONS,
                         competitor=competitor(formula, len(els), products)))
    df = pd.DataFrame(rows).drop_duplicates("mpid")
    report(score(df))


def score(df):
    df["margin_meV"] = df.margin_meV.round(4)  # rescoring rereads the table; fixed digits keep the file stable
    df["sigma"] = [sigma_for(f) if ox else SIGMA_CLOSED for f, ox in zip(df.formula, df.oxide)]
    df["p"] = (df.margin_meV.abs() / df.sigma).round(6)
    df["resolved"] = df.p > 1
    df.to_csv(OUT, index=False)
    return df


def report(df):
    print(f"ICSD-observed phases with GGA/GGA+U thermo (2+ elements): {len(df)} (Sun et al. 2016: 29,902)")
    for lab, g in (("all", df), ("oxides (calibrated)", df[df.oxide]), ("other (extrapolated)", df[~df.oxide])):
        print(f"{lab:22s} n={len(g):6d} resolved {100 * g.resolved.mean():4.0f}% | stable {int(g.stable.sum())} "
              f"resolved {100 * g[g.stable].resolved.mean():.0f}% | metastable resolved {100 * g[~g.stable].resolved.mean():.0f}%"
              f" | median |margin| {g.margin_meV.abs().median():.1f}")
    print(df[df.oxide].groupby("family").agg(n=("p", "size"), resolved_pct=("resolved", lambda s: round(100 * s.mean())),
                                             median_abs_margin=("margin_meV", lambda s: round(s.abs().median(), 1))).to_string())
    print(df[df.oxide].groupby("competitor").agg(
        n=("p", "size"), resolved_pct=("resolved", lambda s: round(100 * s.mean())),
        stable=("stable", "sum"), median_abs_margin=("margin_meV", lambda s: round(s.abs().median(), 1))).to_string())
    print(df.groupby("nelements").agg(n=("p", "size"), resolved_pct=("resolved", lambda s: round(100 * s.mean()))).to_string())


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        report(score(pd.read_csv(OUT)))
    else:
        main()
