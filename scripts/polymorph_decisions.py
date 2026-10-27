"""Computed decisions for measured metastable polymorphs (results/polymorph_measured.csv).

Each pair is a decision whose true sign is known: the metastable polymorph lies above the stable one by the measured
enthalpy. The computed margin is the Materials Project energy difference of the two polymorphs, each the
lowest-energy entry of the formula that the database matches to an experimental (ICSD) structure and that lies in the
listed space groups (symmetry found with spglib at 0.1 A). Two polymorphs share their space group with another observed
phase of lower energy and are named by entry (NAMED). The database's corrections cancel between two entries of one
composition. Writes results/polymorph_decisions.csv in the column
layout of precision_calibration_decisions.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from emmet.core.mpid import AlphaID
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key, load_or_pull, mp_number  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
# The lowest observed C2/c SiO2 entry is moganite (mp-12787, 2.43 g/cm3); coesite is mp-11684 (2.80, against 2.91 from
# its measured molar volume of 20.64 cm3/mol). The lowest observed P2_1/n Ca2SiO4 entry has 56 atoms per cell; larnite,
# Z = 4, is mp-4180 with 28.
NAMED = {("SiO2", "coesite"): 11684, ("Ca2SiO4", "beta (larnite)"): 4180}


def chemsys(formula):
    return "-".join(sorted(str(e) for e in Composition(formula).elements))


def observed_ids(mpr, formula):
    docs = mpr.materials.summary.search(formula=Composition(formula).reduced_formula, theoretical=False,
                                        fields=["material_id"])
    return {int(AlphaID(str(d.material_id))) for d in docs}


def pick(entries, formula, groups, observed, named=None):
    red = Composition(formula).reduced_formula
    found = []
    for e in entries:
        if e.composition.reduced_formula != red or mp_number(e) not in observed:
            continue
        if named is not None and mp_number(e) != named:
            continue
        sg = SpacegroupAnalyzer(e.structure, symprec=0.1).get_space_group_number()
        if sg in groups:
            found.append((e.energy_per_atom, e, sg))
    if not found:
        return None
    energy, e, sg = min(found, key=lambda x: x[0])
    return e, sg


def main():
    m = pd.read_csv(ROOT / "results" / "polymorph_measured.csv")
    rows = []
    with MPRester(api_key()) as mpr:
        for _, r in m.iterrows():
            entries = load_or_pull(mpr, chemsys(r.formula))
            observed = observed_ids(mpr, r.formula)
            meta = pick(entries, r.formula, {int(x) for x in str(r.sg_metastable).split()}, observed,
                        NAMED.get((r.formula, r.metastable)))
            stable = pick(entries, r.formula, {int(x) for x in str(r.sg_stable).split()}, observed)
            if meta is None or stable is None:
                print(f"{r.formula} {r.metastable}/{r.stable}: no entry in the listed space groups "
                      f"({'metastable' if meta is None else 'stable'})")
                continue
            (em, sgm), (es, sgs) = meta, stable
            dft = 1000 * (em.energy_per_atom - es.energy_per_atom)  # meV per atom, positive = metastable above
            rows.append(dict(compound=f"{r.formula} ({r.metastable})", formula=r.formula, n_elements=len(Composition(r.formula)),
                             products=f"{r.formula} ({r.stable})", kind="polymorph", measured_meV_atom=r.dH_meV_atom,
                             dft_meV_atom=round(dft, 2), error_meV_atom=round(dft - r.dH_meV_atom, 2),
                             sign_established=bool(r.sign_established),
                             mp_metastable=f"mp-{mp_number(em)}", sg_metastable=sgm,
                             mp_stable=f"mp-{mp_number(es)}", sg_stable=sgs))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "polymorph_decisions.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.drop(columns=["products", "n_elements"]).to_string(index=False))
    ok = out[out.sign_established]
    print(f"\n{len(ok)} pairs with an established sign; computed sign right for "
          f"{int((np.sign(ok.dft_meV_atom) == np.sign(ok.measured_meV_atom)).sum())}")


if __name__ == "__main__":
    main()
