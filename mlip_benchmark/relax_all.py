"""Relax every benchmark structure with one MLIP (WSL) and record its energy per atom, at the Materials Project
geometry and after a full cell and position relaxation.

Usage: python relax_all.py {orb20|orbinf|mace} structures.extxyz out.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hermes_oxide"))
from pilot_md import calculator  # noqa: E402

model, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
calc, label = calculator(model)
results = []
for atoms in read(src, index=":"):
    info = dict(atoms.info)
    atoms.calc = calc
    n = len(atoms)
    e_mp_geometry = atoms.get_potential_energy() / n
    v0 = atoms.get_volume() / n
    t0 = time.time()
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = bool(opt.run(fmax=0.01, steps=600))
    results.append(dict(formula=info["formula"], role=info["role"], n_atoms=n, model=label,
                        e_mp_geometry=float(e_mp_geometry), e_relaxed=float(atoms.get_potential_energy() / n),
                        volume_mp=float(v0), volume_relaxed=float(atoms.get_volume() / n), converged=converged,
                        max_force=float(np.abs(atoms.get_forces()).max()), seconds=round(time.time() - t0, 1),
                        mp_energy_per_atom=info.get("mp_energy_per_atom")))
    print(f"{info['formula']:12s} {info['role']:9s} {n:3d} E={results[-1]['e_relaxed']:.4f} conv={converged} "
          f"dV={100 * (results[-1]['volume_relaxed'] / v0 - 1):+.1f}%", flush=True)
json.dump(results, open(out, "w"), indent=1)
