"""Relax every supercell of runs/run_list.tsv at 0 K, locally, and save it to relaxed/ for md_run.py to start from.

Same model, settings and supercell as the relaxation inside md_run.py (it is the same function), so the Sol jobs
start from identical cells; on Sol's GPU nodes that step alone ran for over 90 minutes. Writes relaxed/relax_log.json
with the steps, the largest residual force and the time of each.

Usage: python prerelax.py [--device cuda]
"""
import argparse
import csv
import json
import os
import time

import numpy as np
from ase.io import read, write
from orb_models.forcefield.calculator import ORBCalculator

from md_run import HERE, RELAX_FMAX, RELAX_STEPS, load_model, relax, relaxed_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    with open(os.path.join(HERE, "runs", "run_list.tsv"), newline="") as f:
        cells = sorted({(r["cif"], r["reps"]) for r in csv.DictReader(f, delimiter="\t")})
    os.makedirs(os.path.join(HERE, "relaxed"), exist_ok=True)
    model, source = load_model(a.device)
    calc = ORBCalculator(model, device=a.device)
    log_path = os.path.join(HERE, "relaxed", "relax_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    for cif, reps in cells:
        out = relaxed_path(cif, reps)
        if os.path.exists(out):
            print(f"{os.path.basename(out)} exists, skipping", flush=True)
            continue
        atoms = read(os.path.join(HERE, cif)).repeat(tuple(int(x) for x in reps.split("x")))
        atoms.calc = calc
        t0 = time.time()
        steps, fmax = relax(atoms)
        atoms.calc = None
        write(out, atoms)
        log[os.path.basename(out)] = dict(cif=cif, reps=reps, n_atoms=len(atoms), steps=steps, max_steps=RELAX_STEPS,
                                          fmax_target=RELAX_FMAX, fmax_final=round(fmax, 5),
                                          volume_per_atom=round(atoms.get_volume() / len(atoms), 5),
                                          seconds=round(time.time() - t0, 1), weights_from=source)
        json.dump(log, open(log_path, "w"), indent=1)
        print(f"{os.path.basename(out)}: {steps} steps, max force {fmax:.4f} eV/A, "
              f"{np.cbrt(atoms.get_volume() / len(atoms)):.4f} A per atom, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
