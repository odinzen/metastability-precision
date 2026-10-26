"""Export every structure the calorimetry benchmark needs (each anchor compound and the reference polymorph of each
binary oxide it is formed from) as extended-XYZ, so the MLIP relaxations in WSL use exactly the Materials Project
structures the DFT side used. Picks entries with the same rules as benchmark_dft_vs_measured.py."""
import sys
from pathlib import Path

import pandas as pd
from ase.io import write
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dft_vs_measured import lowest, oxide_split  # noqa: E402
from pull_mp_systems import load_or_pull  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mlip_benchmark" / "structures.extxyz"


def main():
    bench = pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    wanted = {}  # (formula, role) -> entry
    for formula in sorted(bench.compound.unique()):
        chemsys = "-".join(sorted(str(e) for e in Composition(formula).elements))
        entries = load_or_pull(None, chemsys)  # cached; no API call
        wanted[(formula, "target")] = lowest(entries, formula)
        for ox in oxide_split(formula):
            if (ox, "reference") not in wanted:
                wanted[(ox, "reference")] = lowest(entries, ox, match_reference=True)
    images = []
    for (formula, role), e in wanted.items():
        atoms = AseAtomsAdaptor.get_atoms(e.structure)
        atoms.info.update(formula=formula, role=role, mp_energy_per_atom=float(e.energy_per_atom))
        images.append(atoms)
    OUT.parent.mkdir(exist_ok=True)
    write(OUT, images, format="extxyz")
    print(len(images), "structures;", sum(len(a) for a in images), "atoms in total; largest", max(len(a) for a in images))


if __name__ == "__main__":
    main()
