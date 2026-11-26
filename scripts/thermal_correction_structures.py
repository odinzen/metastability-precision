"""Structures for the 0 K to 298 K enthalpy check: the Materials Project entries behind the three worked examples,
their reactions from the binary oxides or the elements, and the element and binary ground states. Writes data/thermal_correction/<tag>.cif and phases.csv; thermal_correction.py runs the phonons.
"""
import sys
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "thermal_correction"
# the entries used for these compounds by the decisions and the calorimetric benchmarks
FIXED = {"Ca3SiO5": "mp-641754", "Ca2SiO4": "mp-4481", "CaO": "mp-2605", "ZnFe2O4": "mp-19313", "ZnO": "mp-2133",
         "Fe2O3": "mp-19770", "TiSi2": "mp-2582", "TiSi": "mp-7092",
         "Ti": "mp-46"}  # hcp titanium, the reference of the measurements; the database ground state is omega-Ti
GROUND = ["SiO2", "Si"]  # lowest-energy entry of the database


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with MPRester(api_key()) as mpr:
        for tag, mpid in FIXED.items():
            s = mpr.get_structure_by_material_id(mpid, conventional_unit_cell=False)
            rows.append(dict(tag=tag, mp_id=mpid, atoms=len(s), formula=s.composition.reduced_formula))
            s.to(filename=str(OUT / f"{tag}.cif"))
        for f in GROUND:
            docs = mpr.materials.summary.search(formula=f, fields=["material_id", "energy_above_hull", "structure"])
            best = min(docs, key=lambda d: d.energy_above_hull)
            s = best.structure.get_primitive_structure()
            rows.append(dict(tag=f, mp_id=str(best.material_id), atoms=len(s), formula=f))
            s.to(filename=str(OUT / f"{f}.cif"))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "phases.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
