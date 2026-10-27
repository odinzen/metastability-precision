"""Computed decisions for measured metastable polymorphs (results/polymorph_measured.csv).

Each pair is a decision whose true sign is known: the metastable polymorph lies above the stable one by the measured
enthalpy. The computed margin is the Materials Project energy difference of the two polymorphs, each the
lowest-energy entry of the formula in the listed space groups (symmetry found with spglib at 0.1 A). The database's
corrections cancel between two entries of one composition. Writes results/polymorph_decisions.csv in the column
layout of precision_calibration_decisions.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key, load_or_pull, mp_number  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def chemsys(formula):
    return "-".join(sorted(str(e) for e in Composition(formula).elements))


def pick(entries, formula, groups):
    red = Composition(formula).reduced_formula
    found = []
    for e in entries:
        if e.composition.reduced_formula != red:
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
            meta = pick(entries, r.formula, {int(x) for x in str(r.sg_metastable).split()})
            stable = pick(entries, r.formula, {int(x) for x in str(r.sg_stable).split()})
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
