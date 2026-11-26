"""Audit of the measured outcome of each compiled decision (the "truth" of the calibration).

The compiled decisions (sigma_by_reaction_type.py) take their measured sign from differences of compiled formation
enthalpies. Each reaction here gets three things:
- the compiled value and its uncertainty, propagated from the per-compound uncertainties the compilation lists;
- the value from primary calorimetry of formation from the oxides, wherever the literature compilation holds it for
  the compound and every ternary or higher product (records named in data/curated/primary_truth.csv, chosen by
  polymorph; their values, sources and pages are in results/primary_truth_raw.csv);
- the probability that the true reaction enthalpy is negative given that value and uncertainty (a soft label).
A stable phase on an assessed phase diagram (data/curated/phase_diagram_labels.csv) need not be stable in enthalpy, as
Coughlin's C3A shows, so those labels enter only as a sensitivity. Writes results/truth_audit.csv.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.optimize import nnls
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
MEV_PER_KJ_MOL = 1000 / 96.48533


def compiled_uncertainty():
    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    df = pd.DataFrame(d["data"], columns=d["columns"]).drop_duplicates("formula")
    return {Composition(f).reduced_formula: 1000 * u for f, u in zip(df.formula, df.uncertainty) if pd.notna(u)}


def primary_records(ids):
    """The named records as dicts shaped like the compilation's, from results/primary_truth_raw.csv."""
    d = pd.read_csv(ROOT / "results" / "primary_truth_raw.csv")
    out = {}
    for r in d[d.record.isin(ids)].itertuples():
        u = None if pd.isna(r.uncertainty) else dict(value=r.uncertainty, unit=r.uncertainty_unit)
        out[r.record] = dict(value=r.value, unit=r.unit, uncertainty=u, conditions=dict(composition=r.compound))
    return out


def per_atom_mev(a):
    """kJ/mol (or J/mol) per formula unit of the record's composition -> meV per atom, value and uncertainty."""
    scale = 1e-3 if a["unit"].replace(" ", "") == "J/mol" else 1.0
    n = Composition(a["conditions"]["composition"]).num_atoms
    u = (a.get("uncertainty") or {}).get("value")
    uscale = 1e-3 if ((a.get("uncertainty") or {}).get("unit") or "").replace(" ", "") == "J/mol" else 1.0
    return a["value"] * scale / n * MEV_PER_KJ_MOL, (None if u is None else u * uscale / n * MEV_PER_KJ_MOL)


def fractions(target, products):
    """Atoms of each product per atom of the target, from the element balance."""
    els = sorted({str(e) for p in [target] + products for e in Composition(p).elements})
    vec = lambda f: np.array([Composition(f).get_atomic_fraction(e) for e in els])
    x, res = nnls(np.column_stack([vec(p) for p in products]), vec(target))
    assert res < 1e-6, (target, products, res)
    return dict(zip(products, x))


def main():
    d = pd.read_csv(ROOT / "results" / "sigma_by_reaction_type.csv")
    d = d[d.measured_meV_atom != 0].copy()
    u_comp = compiled_uncertainty()
    u_median = float(np.median(list(u_comp.values())))
    primary = pd.read_csv(ROOT / "data" / "curated" / "primary_truth.csv")
    rec = primary_records(set(primary.record))
    missing = set(primary.record) - set(rec)
    assert not missing, missing
    prim = {Composition(f).reduced_formula: (per_atom_mev(rec[i]), i) for f, i in zip(primary.formula, primary.record)}
    pdl = pd.read_csv(ROOT / "data" / "curated" / "phase_diagram_labels.csv")
    pd_stable = {Composition(f).reduced_formula for f, s in zip(pdl.formula, pdl.stable_on_phase_diagram) if s}

    rows = []
    for _, r in d.iterrows():
        target = Composition(r.compound).reduced_formula
        prods = [Composition(p).reduced_formula for p in r.products.split("+")]
        x = fractions(target, prods)
        terms = [(target, 1.0)] + [(p, x[p]) for p in prods]
        u = np.sqrt(sum((w * u_comp.get(f, u_median)) ** 2 for f, w in terms))
        row = dict(compound=r.compound, products=r.products, dft_meV_atom=r.dft_meV_atom,
                   measured_compiled=r.measured_meV_atom, u_compiled=round(u, 1),
                   u_from_median=any(f not in u_comp for f, _ in terms))
        ternary = [f for f, _ in terms if len(Composition(f).elements) > 2]
        if all(f in prim for f in ternary):
            val = prim[target][0][0] - sum(x[p] * prim[p][0][0] for p in prods if p in prim)
            us = [w * prim[f][0][1] for f, w in terms if f in prim and prim[f][0][1] is not None]
            row.update(measured_primary=round(val, 1), u_primary=round(float(np.sqrt(np.sum(np.square(us)))), 1) if us else np.nan,
                       primary_records=" ".join(prim[f][1] for f in ternary))
        rows.append(row)
    a = pd.DataFrame(rows)
    a["truth_meV_atom"] = a.measured_primary.fillna(a.measured_compiled)
    a["truth_u"] = np.where(a.measured_primary.notna(), a.u_primary.fillna(u_median), a.u_compiled)
    a["truth_basis"] = np.where(a.measured_primary.notna(), "primary calorimetry", "compiled")
    a["p_stable"] = norm.cdf(-a.truth_meV_atom / a.truth_u).round(4)
    a["phase_diagram_stable"] = a.compound.map(lambda c: Composition(c).reduced_formula in pd_stable)
    a.to_csv(ROOT / "results" / "truth_audit.csv", index=False)

    flip = a[np.sign(a.truth_meV_atom) != np.sign(a.measured_compiled)]
    print(f"{len(a)} compiled decisions; {int(a.measured_primary.notna().sum())} recomputed from primary calorimetry; "
          f"{len(flip)} change sign:")
    print(flip[["compound", "products", "measured_compiled", "measured_primary", "u_primary", "dft_meV_atom"]].to_string(index=False))
    pc = a[a.measured_primary.notna()]
    print("\nprimary against compiled, meV/atom:")
    print(pc[["compound", "products", "measured_compiled", "u_compiled", "measured_primary", "u_primary"]].to_string(index=False))
    print(f"\ncompiled reaction uncertainty: median {a.u_compiled.median():.1f}, {int(a.u_from_median.sum())} use the median "
          f"compound uncertainty; |truth| < 2u for {int((a.truth_meV_atom.abs() < 2 * a.truth_u).sum())} decisions")


if __name__ == "__main__":
    main()
