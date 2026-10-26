"""Harmonic phonon baseline for the Hermes oxide pilot (WSL, phonopy + the same MLIP): relax at 0 K, finite
displacements in a supercell, quantum harmonic entropy at the test temperatures. Reports imaginary modes rather than
hiding them, since cubic SrTiO3 is unstable at 0 K and its harmonic entropy is then not defined.

Usage: python pilot_phonons.py MODEL CIF SUPERCELL OUT.json
"""
import json
import sys

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pilot_md import calculator  # noqa: E402

model, cif, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
calc, label = calculator(model, precision="float64")  # finite differences of 0.01 A need double precision forces
atoms = read(cif)
atoms.calc = calc
FIRE(FrechetCellFilter(atoms, hydrostatic_strain=True), logfile=None).run(fmax=0.001, steps=1000)
unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array, scaled_positions=atoms.get_scaled_positions())
ph = Phonopy(unit, supercell_matrix=np.eye(3) * n, primitive_matrix="P")  # phonopy 4 defaults to the primitive cell; keep the given cell so per-cell units match n_atoms
ph.generate_displacements(distance=0.01)
forces = []
for sc in ph.supercells_with_displacements:
    s = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
    s.calc = calc
    forces.append(s.get_forces())
ph.forces = forces
ph.produce_force_constants()
ph.run_mesh([24, 24, 24], with_eigenvectors=False)
freqs = ph.mesh.frequencies
ph.run_thermal_properties(temperatures=[298.15, 600.0, 1000.0])
tp = ph.get_thermal_properties_dict()
n_atoms = len(atoms)
result = dict(model=label, cif=cif, supercell=n, a_relaxed=float(np.cbrt(atoms.get_volume() / n_atoms)),
              lattice_a=float(atoms.cell.lengths()[0]), n_displacements=len(forces),
              min_frequency_THz=float(freqs.min()), imaginary_fraction=float((freqs < -0.05).mean()),
              # phonopy gives J/(K mol of unit cells); divide by atoms per cell for J/(K mol atoms)
              S_harmonic_per_mol_atoms={str(t): float(s) / n_atoms for t, s in zip(tp["temperatures"], tp["entropy"])})
json.dump(result, open(out, "w"), indent=1)
print(json.dumps(result))
