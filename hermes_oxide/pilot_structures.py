"""Write the pilot structures from the cached Materials Project entries: rocksalt MgO and cubic SrTiO3 (Pm-3m, the
phase that exists from 105 K up, so the one present at 298 and 1000 K), as conventional cells."""
import gzip
import json
from pathlib import Path

from monty.json import MontyDecoder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "pilot"
WANT = {"MgO": ("Al-Mg-O", "Fm-3m"), "SrTiO3": ("O-Sr-Ti", "Pm-3m")}


def main():
    OUT.mkdir(exist_ok=True)
    for formula, (chemsys, sg) in WANT.items():
        entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
        cands = []
        for e in entries:
            if e.composition.reduced_formula != formula or not hasattr(e, "structure"):
                continue
            sga = SpacegroupAnalyzer(e.structure, symprec=0.05)
            if sga.get_space_group_symbol() == sg:
                cands.append((e.energy_per_atom, sga.get_conventional_standard_structure(), e))
        energy, conv, entry = min(cands, key=lambda c: c[0])
        conv.to(filename=str(OUT / f"{formula}_conventional.cif"))
        eid = entry.entry_id
        ident = eid.get("identifier") if isinstance(eid, dict) else getattr(eid, "identifier", str(eid))
        print(formula, sg, ident, "a =", round(conv.lattice.a, 4), "atoms", len(conv), "E/atom", round(energy, 4),
              "| other polymorphs in cache:", len([1 for e in entries if e.composition.reduced_formula == formula]))


if __name__ == "__main__":
    main()
