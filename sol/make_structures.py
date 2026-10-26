"""Write the test structures for the Sol MD sweep and the run list (one row per compound and temperature).

Each structure is the lowest-energy Materials Project entry in the polymorph the measured entropy refers to, as a
conventional cell, repeated along its shortest axis until the supercell has at least 400 atoms.
Runs on Windows with the project venv (pymatgen); the CIFs and run list go to Sol.
"""
import gzip
import json
from pathlib import Path

from monty.json import MontyDecoder
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MIN_ATOMS = 400
TEMPERATURES = (298.15, 1000.0)
# compound: (cached chemical system, space group of the measured polymorph)
COMPOUNDS = {
    "MgO": ("Al-Mg-O", "Fm-3m"), "Al2O3": ("Al-Mg-O", "R-3c"), "TiO2": ("Mg-O-Ti", "P4_2/mnm"),
    "Y2O3": ("O-P-Y", "Ia-3"), "Sc2O3": ("O-P-Sc", "Ia-3"), "La2O3": ("La-O-P", "P-3m1"),
    "Lu2O3": ("Lu-O-P", "Ia-3"), "SrTiO3": ("O-Sr-Ti", "Pm-3m"), "MgTiO3": ("Mg-O-Ti", "R-3"),
    "Mg3(PO4)2": ("Mg-O-P-Ti", "P2_1/c"),
    # Added 2026-10-24 so that more entropies of formation from the oxides close with measured values at 298 K.
    # No cached system: taken from the Materials Project directly. Rows 1-20 above are unchanged.
    "CaO": (None, "Fm-3m"), "ZrO2": (None, "P2_1/c"), "SiO2": (None, "P3_121"), "CaTiO3": (None, "Pnma"),
    "CaZrO3": (None, "Pnma"), "Mg2SiO4": (None, "Pnma"), "ZrSiO4": (None, "I4_1/amd"), "La2Zr2O7": (None, "Fd-3m"),
}
SAME_PHASE = {"P3_121": {"P3_121", "P3_221"}}  # alpha-quartz is enantiomorphic


def reps_for(lattice_lengths, n_atoms):
    if max(lattice_lengths) - min(lattice_lengths) < 1e-3:
        # Cubic cells keep a cubic supercell (SrTiO3 would otherwise get 5 x 4 x 4).
        n = 1
        while n_atoms * n ** 3 < MIN_ATOMS:
            n += 1
        return [n, n, n]
    reps = [1, 1, 1]
    while n_atoms * reps[0] * reps[1] * reps[2] < MIN_ATOMS:
        i = min(range(3), key=lambda k: lattice_lengths[k] * reps[k])
        reps[i] += 1
    return reps


def candidates(formula, chemsys):
    """(energy per atom, structure, identifier) for every entry of the formula."""
    red = Composition(formula).reduced_formula
    if chemsys:
        entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
        out = []
        for e in entries:
            if hasattr(e, "structure") and e.composition.reduced_formula == red:
                eid = e.entry_id
                ident = eid.get("identifier") if isinstance(eid, dict) else getattr(eid, "identifier", str(eid))
                out.append((e.energy_per_atom, e.structure, ident))
        return out
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from pull_mp_systems import api_key
    from mp_api.client import MPRester
    with MPRester(api_key()) as mpr:
        docs = mpr.materials.summary.search(formula=red, fields=["material_id", "structure", "energy_per_atom"])
    return [(d.energy_per_atom, d.structure, str(d.material_id)) for d in docs]


def main():
    (OUT / "structures").mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    rows = ["index\tcompound\ttemperature_K\tcif\treps\tn_atoms\tmp_id\tspacegroup"]
    i = 0
    for formula, (chemsys, sg) in COMPOUNDS.items():
        cands = candidates(formula, chemsys)
        matched = []
        for energy, structure, ident in cands:
            sga = SpacegroupAnalyzer(structure, symprec=0.05)
            if sga.get_space_group_symbol() in SAME_PHASE.get(sg, {sg}):
                matched.append((energy, sga.get_conventional_standard_structure(), ident))
        if not matched:
            raise SystemExit(f"{formula}: no {sg} entry; polymorphs present: "
                             f"{sorted({SpacegroupAnalyzer(s, symprec=0.05).get_space_group_symbol() for _, s, _ in cands})}")
        _, conv, ident = min(matched, key=lambda m: m[0])
        safe = formula.replace("(", "").replace(")", "")
        cif = f"structures/{safe}_{sg.replace('/', '').replace('-', 'm')}.cif"
        conv.to(filename=str(OUT / cif))
        reps = reps_for(conv.lattice.abc, len(conv))
        n = len(conv) * reps[0] * reps[1] * reps[2]
        for t in TEMPERATURES:
            i += 1
            rows.append(f"{i}\t{formula}\t{t}\t{cif}\t{'x'.join(map(str, reps))}\t{n}\t{ident}\t{sg}")
        print(f"{formula:10s} {sg:9s} {ident} conventional {len(conv):3d} atoms, reps {reps}, supercell {n} atoms")
    (OUT / "runs" / "run_list.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(len(rows) - 1, "runs written")


if __name__ == "__main__":
    main()
