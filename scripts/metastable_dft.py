"""Computed margins for the measured-metastable candidates of the literature search: each compound against the
competing phases its measurement uses (data/curated/metastable_reactions.csv), from Materials Project GGA/GGA+U
entries of release 2026.09.28 with the database's corrections, elements at zero. The competing phases take the lowest-energy entry of their formula. The
compound takes the lowest-energy entry in the structure that was measured: taking the lowest polymorph instead would
pull a metastable phase toward the hull, the very error being scored. A disordered phase with no entry in its own
space group takes the lowest entry as an ordered approximant (flagged); a compound whose measured structure the
database lacks is left out. Also records the compound's own energy above the database hull, how many entries the
database holds at that composition, and the space group used. Writes results/metastable_dft.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_nonoxide import pick, spacegroup  # noqa: E402
from pull_mp_systems import api_key, load_or_pull  # noqa: E402
from sigma_model import open_shell, sigma_for  # noqa: E402
from truth_audit import fractions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REACTIONS = ROOT / "data" / "curated" / "metastable_reactions.csv"


def entry_id(e):
    i = e.entry_id
    if isinstance(i, dict):
        return f"{i['identifier']}-{i['suffix']}"
    return f"{i.identifier}-{i.suffix}" if hasattr(i, "identifier") else str(i)


def pick_structure(entries, formula, structures, disordered):
    red = Composition(formula).reduced_formula
    same = [e for e in entries if e.composition.reduced_formula == red]
    if not same:
        return None, "no Materials Project entry"
    allowed = [x for x in str(structures).split(";") if x and x != "nan"]
    if not allowed:
        return min(same, key=lambda e: e.energy_per_atom), "lowest entry, measured structure not stated"
    match = [e for e in same if spacegroup(e) in allowed]
    if match:
        return min(match, key=lambda e: e.energy_per_atom), "measured structure"
    if disordered:
        return min(same, key=lambda e: e.energy_per_atom), "ordered approximant of a disordered phase"
    return None, "measured structure not in the database (" + ", ".join(sorted({spacegroup(e) for e in same})) + ")"


def main():
    r = pd.read_csv(REACTIONS)
    rows = []
    with MPRester(api_key()) as mpr:
        for x in r.itertuples():
            comp = Composition(x.compound)
            products = x.products.split("+")
            els = sorted({str(e) for f in [x.compound] + products for e in Composition(f).elements})
            entries = load_or_pull(mpr, "-".join(els))
            row = dict(compound=x.compound, family=x.family, products=x.products, note=x.note)
            target, status = pick_structure(entries, x.compound, x.structures, x.disordered)
            if target is None:
                rows.append(dict(row, status=status))
                continue
            refs = {f: pick(entries, f, match_reference=False)[0] for f in products if len(Composition(f).elements) > 1}
            if any(e is None for e in refs.values()):
                rows.append(dict(row, status="competing phase missing: " + ", ".join(f for f, e in refs.items() if e is None)))
                continue
            pdg = PhaseDiagram(entries)
            frac = fractions(comp.reduced_formula, [Composition(f).reduced_formula for f in products])
            e_form = lambda e: pdg.get_form_energy_per_atom(e)
            margin = e_form(target) - sum(frac[Composition(f).reduced_formula] * e_form(refs[f])
                                          for f in products if f in refs)
            same = [e for e in entries if e.composition.reduced_formula == comp.reduced_formula]
            rows.append(dict(row, status=status, mp_id=entry_id(target), mp_sg=spacegroup(target),
                             polymorphs=len(same), dft_meV_atom=round(1000 * margin, 1),
                             mp_e_above_hull_meV=round(1000 * pdg.get_e_above_hull(target), 1),
                             atoms_per_fu=comp.num_atoms, open_shell=open_shell(comp.reduced_formula),
                             sigma=sigma_for(comp.reduced_formula)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "metastable_dft.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(out.drop(columns=["note"]).to_string(index=False))


if __name__ == "__main__":
    main()
