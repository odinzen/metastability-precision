"""Quantum harmonic entropy of one phase with Orb-v3 (the potential of the Sol MD), for the reaction-entropy test.

Orb-v3 is only approximately rotation-equivariant: on a perfectly symmetric crystal it leaves residual forces of a
few 0.01 eV/A where symmetry demands zero (Na2O, 0.025), comparable to the forces a 0.01 A finite displacement
produces. Every force here is therefore averaged over rotated copies of the cell, the residual of the undisplaced
supercell is subtracted, the net drift is removed and the force constants are symmetrized. The phase is computed at
two displacement amplitudes and, for small cells, also in a larger supercell; it is marked converged only if the
entropies agree within TOL at both temperatures.

Usage: python phonon_run.py PHASE OUT_DIR    (PHASE is a file stem in phonon_structures/)
"""
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
TEMPS = (298.15, 1000.0)
GRID = tuple(float(x) for x in range(300, 1601, 50))  # synthesis temperatures, for the temperature-aware precision
N_ROT = 12
TOL = 0.2  # J/(K mol atoms)
ROTS = Rotation.random(N_ROT, random_state=20261024).as_matrix()


MODEL = os.environ.get("PHONON_MODEL", "orb_v3_conservative_20_omat")
DEVICE = os.environ.get("PHONON_DEVICE", "cpu")  # "cuda" on the A100 nodes; double precision either way


def load_calc():
    loader = getattr(pretrained, MODEL)
    url = inspect.signature(loader).parameters["weights_path"].default
    local = HERE / "weights" / url.rsplit("/", 1)[1]
    kw = dict(weights_path=str(local)) if local.exists() else {}  # local weights: compute nodes are offline
    return ORBCalculator(loader(device=DEVICE, precision="float64", compile=False, **kw), device=DEVICE)


def averaged_forces(atoms, calc):
    out = np.zeros((len(atoms), 3))
    for R in ROTS:
        r = atoms.copy()
        r.set_cell(atoms.cell.array @ R.T, scale_atoms=True)  # rotates cell and positions together
        r.calc = calc
        out += r.get_forces() @ R  # back to the original frame
    return out / len(ROTS)


def phonons(atoms, calc, min_width, disp):
    n_sc = [math.ceil(min_width / x) for x in atoms.cell.lengths()]
    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())
    # primitive_matrix "P" keeps the given cell, so per-cell values divide by its atom count
    ph = Phonopy(unit, supercell_matrix=np.diag(n_sc), primitive_matrix="P")
    ph.generate_displacements(distance=disp)

    def as_atoms(sc):
        return Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)

    perfect = as_atoms(ph.supercell)
    f0 = averaged_forces(perfect, calc)
    perfect.calc = calc
    raw_residual = float(np.abs(perfect.get_forces()).max())
    forces = []
    for sc in ph.supercells_with_displacements:
        f = averaged_forces(as_atoms(sc), calc) - f0
        forces.append(f - f.mean(axis=0))
    ph.forces = forces
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    ph.run_mesh(100.0, with_eigenvectors=False)
    freqs = ph.mesh.frequencies
    temps = list(TEMPS) + list(GRID)
    ph.run_thermal_properties(temperatures=temps)
    entropy = ph.thermal_properties.entropy  # J/(K mol of cells)
    n = len(atoms)
    k = len(TEMPS)
    return dict(supercell=n_sc, displacement=disp, n_displacements=len(forces),
                residual_force_raw=raw_residual, residual_force_averaged=float(np.abs(f0).max()),
                min_frequency_THz=float(freqs.min()), imaginary_fraction=float((freqs < -0.05).mean()),
                S_per_mol_atoms={str(t): float(s) / n for t, s in zip(TEMPS, entropy[:k])},
                S_grid_per_mol_atoms={str(t): float(s) / n for t, s in zip(GRID, entropy[k:])})


def main():
    phase, out_dir = sys.argv[1], Path(sys.argv[2])
    out = out_dir / f"{phase}.json"
    # a result from before the temperature grid was added is redone, so every phase has S at synthesis temperature
    if out.exists() and "S_grid_per_mol_atoms" in json.load(open(out)):
        print(f"{out} exists, skipping", flush=True)
        return
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    t0 = time.time()
    calc = load_calc()
    atoms = read(HERE / "phonon_structures" / f"{phase}.cif")
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=0.001, steps=2000)
    atoms.set_constraint()
    print(f"[{(time.time() - t0) / 60:6.1f} min] relaxed in {opt.nsteps} steps, {len(atoms)} atoms", flush=True)

    settings = {"d0.01": (10.0, 0.01), "d0.03": (10.0, 0.03)}
    if len(atoms) <= 6:
        settings["d0.01_wide"] = (14.0, 0.01)
    runs = {}
    for name, (width, disp) in settings.items():
        runs[name] = phonons(atoms, calc, width, disp)
        print(f"[{(time.time() - t0) / 60:6.1f} min] {name}: S {runs[name]['S_per_mol_atoms']}, "
              f"imaginary {runs[name]['imaginary_fraction']:.4f}", flush=True)
    # The reported entropy is the widest supercell's where a small cell has one (BaO moves 0.27 from 10 to 14 A).
    best = "d0.01_wide" if "d0.01_wide" in runs else "d0.01"
    ref = runs[best]["S_per_mol_atoms"]
    spread = max(abs(r["S_per_mol_atoms"][t] - ref[t]) for r in runs.values() for t in ref)
    result = dict(phase=phase, model=f"{MODEL} float64", n_rotations=N_ROT, n_atoms=len(atoms),
                  relax_steps=opt.nsteps, volume_per_atom=atoms.get_volume() / len(atoms), runs=runs,
                  reference_run=best, S_per_mol_atoms=ref, S_grid_per_mol_atoms=runs[best]["S_grid_per_mol_atoms"],
                  spread=spread, converged=bool(spread <= TOL), tolerance=TOL,
                  minutes=(time.time() - t0) / 60, host=platform.node())
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    json.dump(result, open(tmp, "w"), indent=1)
    os.replace(tmp, out)
    print(f"RESULT {phase}: {'CONVERGED' if result['converged'] else 'NOT CONVERGED'} (spread {spread:.3f})", flush=True)


if __name__ == "__main__":
    main()
