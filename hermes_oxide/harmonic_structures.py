"""Structures for the harmonic reaction-entropy test: the measured polymorph of each phase, taken from the Materials
Project as the lowest-energy entry with the stated space group, written as primitive cells to sol/phonon_structures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pull_mp_systems import api_key  # noqa: E402
from mp_api.client import MPRester  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "sol" / "phonon_structures"
# formula: space group of the phase the tabulated entropy refers to
PHASES = {"MgO": "Fm-3m", "CaO": "Fm-3m", "Na2O": "Fm-3m", "TiO2": "P4_2/mnm", "ZrO2": "P2_1/c", "SiO2": "P3_121",
          "La2O3": "P-3m1", "MgTiO3": "R-3", "CaTiO3": "Pnma", "CaZrO3": "Pnma", "Mg2SiO4": "Pnma",
          "ZrSiO4": "I4_1/amd", "Na2SiO3": "Cmc2_1", "La2Zr2O7": "Fd-3m"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with MPRester(api_key()) as mpr:
        for formula, sg in PHASES.items():
            docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "structure", "symmetry",
                                                                         "energy_above_hull", "theoretical"])
            # alpha-quartz is enantiomorphic; P3_221 is the same phase
            ok = [d for d in docs if d.symmetry.symbol == sg or (formula == "SiO2" and d.symmetry.symbol == "P3_221")]
            if not ok:
                print(f"{formula}: no {sg} entry; found {sorted({d.symmetry.symbol for d in docs})}")
                continue
            d = min(ok, key=lambda d: d.energy_above_hull)
            prim = SpacegroupAnalyzer(d.structure, symprec=0.01).get_primitive_standard_structure()
            prim.to(filename=str(OUT / f"{formula}.cif"))
            print(f"{formula:9s} {d.material_id} {sg:9s} e_hull {1000 * d.energy_above_hull:6.1f} meV/atom, "
                  f"{len(prim)} atoms, observed {not d.theoretical}")


if __name__ == "__main__":
    main()
