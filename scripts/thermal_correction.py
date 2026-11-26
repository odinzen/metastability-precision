"""How far a 0 K energy is from a 298 K enthalpy, for the reactions of the three worked examples.

DFT gives the energy of a static lattice. The enthalpy at 298 K adds the zero point energy and the heat content from
0 to 298 K, both from harmonic phonons of Orb-v3 with the settings of sol/phonon_run.py. Each phase is relaxed with
its symmetry held, and H(T) = F(T) + T S(T), so H(0) is the zero point energy. Small cells average every force over
12 rotations, as for the entropies. The 162-atom alite cell uses one orientation and one-sided displacements, which
is enough for an enthalpy dominated by the zero point energy.

The silicide reactions use MACE-MPA-0 (medium, MIT licence) instead. Orb-v3 leaves 0.02-0.03 eV/A on perfect
diamond Si and gives 29-43% imaginary modes whatever the cell, displacement sign or averaging, so its Si heat content
is about half the JANAF value. Of the MIT MACE models, MP-0 makes hcp Ti far too soft (heat content 6.0 against JANAF
4.8 kJ/mol) and MPA-0 gives all four silicide phases real modes with both element heat contents within 13% of JANAF.
MACE is exactly equivariant, so it needs one orientation. The element heat contents are printed against JANAF as a
check. Run in the mlip-mace environment (the MACE checkpoint does not load with the e3nn of mlip-screen):
    python thermal_correction.py            (all phases, then the reactions)
Writes results/thermal_correction/<tag>.json (Orb), results/thermal_correction/<mlip>/<tag>.json and
results/thermal_correction.csv.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sol"))
import phonon_run as pr  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.constraints import FixSymmetry  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.io import read  # noqa: E402
from ase.optimize import FIRE  # noqa: E402
from phonopy import Phonopy  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402

STRUCT = ROOT / "data" / "thermal_correction"
OUT = ROOT / "results" / "thermal_correction"
EV_PER_KJ_MOL = 1 / 96.485332
T_ROOM = 298.15
LARGE = 60  # atoms; above this one orientation and one-sided displacements
ALL_ROTS = pr.ROTS
# H(298) - H(0), kJ/mol, JANAF tables (Chase 1998)
JANAF_HEAT_CONTENT = {"Si": 3.217, "Ti": 4.824}
# target: products, as stoichiometric coefficients per formula unit
REACTIONS = {"Ca3SiO5 from 3 CaO + SiO2": ("orb", {"Ca3SiO5": 1, "CaO": -3, "SiO2": -1}),
             "Ca3SiO5 from Ca2SiO4 + CaO": ("orb", {"Ca3SiO5": 1, "Ca2SiO4": -1, "CaO": -1}),
             "ZnFe2O4 from ZnO + Fe2O3": ("orb", {"ZnFe2O4": 1, "ZnO": -1, "Fe2O3": -1}),
             "TiSi2 from Ti + 2 Si": ("mace-mpa-0", {"TiSi2": 1, "Ti": -1, "Si": -2}),
             "TiSi2 from TiSi + Si": ("mace-mpa-0", {"TiSi2": 1, "TiSi": -1, "Si": -1})}
# the MIT MACE models, all run on the silicides to show how much the choice matters
MACE_MODELS = {"mace-mp-0": "medium", "mace-mp-0b3": "medium-0b3", "mace-mpa-0": "medium-mpa-0"}


def load_calc(mlip):
    if mlip == "orb":
        return pr.load_calc()
    from mace.calculators import mace_mp
    return mace_mp(model=MACE_MODELS[mlip], default_dtype="float64", device="cpu")


def run_phase(tag, mlip, calc):
    out = OUT / f"{tag}.json" if mlip == "orb" else OUT / mlip / f"{tag}.json"
    if out.exists():
        return json.load(open(out))
    atoms = read(STRUCT / f"{tag}.cif")
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.001, steps=2000)
    atoms.set_constraint()
    e_static = atoms.get_potential_energy() / len(atoms)
    large = len(atoms) > LARGE
    pr.ROTS = np.eye(3)[None] if mlip != "orb" else ALL_ROTS[:1] if large else ALL_ROTS
    n_sc = [math.ceil(10.0 / x) for x in atoms.cell.lengths()]
    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=np.diag(n_sc), primitive_matrix="P")
    ph.generate_displacements(distance=0.01, is_plusminus=False if large else "auto")

    def as_atoms(sc):
        return Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)

    f0 = pr.averaged_forces(as_atoms(ph.supercell), calc)
    forces = []
    for sc in ph.supercells_with_displacements:
        f = pr.averaged_forces(as_atoms(sc), calc) - f0
        forces.append(f - f.mean(axis=0))
    ph.forces = forces
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    ph.run_mesh(100.0, with_eigenvectors=False)
    freqs = ph.mesh.frequencies
    ph.run_thermal_properties(temperatures=[0.0, T_ROOM])
    tp = ph.thermal_properties
    n = len(atoms)
    # phonopy: free energy in kJ/mol of cells, entropy in J/(K mol of cells)
    h = [(tp.free_energy[i] + T * tp.entropy[i] / 1000) / n for i, T in enumerate([0.0, T_ROOM])]
    res = dict(tag=tag, mlip=mlip, atoms=n, supercell=n_sc, n_displacements=len(forces), orientations=len(pr.ROTS),
               e_static_eV_atom=float(e_static), zpe_kJ_mol_atoms=float(h[0]),
               heat_content_kJ_mol_atoms=float(h[1] - h[0]), min_frequency_THz=float(freqs.min()),
               imaginary_fraction=float((freqs < -0.05).mean()))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, res, flush=True)
    return res


def reactions(ph, reacts, path):
    from pymatgen.core import Composition
    import pandas as pd
    rows = []
    for name, (mlip, nu) in reacts.items():
        target = next(k for k, v in nu.items() if v > 0)
        n_atoms = Composition(target).num_atoms
        def delta(key, scale=1.0):
            return sum(v * Composition(k).num_atoms * ph[mlip, k][key] * scale for k, v in nu.items()) / n_atoms
        rows.append(dict(reaction=name, mlip=mlip,
                         max_imaginary_fraction=max(ph[mlip, k]["imaginary_fraction"] for k in nu),
                         static_meV_atom=1000 * delta("e_static_eV_atom"),
                         zpe_meV_atom=1000 * delta("zpe_kJ_mol_atoms", EV_PER_KJ_MOL),
                         heat_content_meV_atom=1000 * delta("heat_content_kJ_mol_atoms", EV_PER_KJ_MOL)))
    t = pd.DataFrame(rows)
    t["correction_298_meV_atom"] = t.zpe_meV_atom + t.heat_content_meV_atom
    t.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(t.to_string(index=False))


def main():
    silicide = {n: nu for n, (m, nu) in REACTIONS.items() if m.startswith("mace")}
    compare = {f"{n} ({m})": (m, nu) for m in MACE_MODELS for n, nu in silicide.items()}
    ph = {}
    for mlip in sorted({m for m, _ in REACTIONS.values()} | set(MACE_MODELS)):
        calc = load_calc(mlip)
        tags = sorted({k for m, nu in [*REACTIONS.values(), *compare.values()] if m == mlip for k in nu},
                      key=lambda t: len(read(STRUCT / f"{t}.cif")))
        ph.update({(mlip, t): run_phase(t, mlip, calc) for t in tags})
    for (mlip, t), r in ph.items():
        if t in JANAF_HEAT_CONTENT:
            print(f"{t} ({mlip}) H298-H0 {r['heat_content_kJ_mol_atoms']:.3f} kJ/mol, "
                  f"JANAF {JANAF_HEAT_CONTENT[t]:.3f}, imaginary {r['imaginary_fraction']:.3f}")
    reactions(ph, REACTIONS, ROOT / "results" / "thermal_correction.csv")
    reactions(ph, compare, ROOT / "results" / "thermal_correction_mace_models.csv")


if __name__ == "__main__":
    main()
