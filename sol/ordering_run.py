"""Ordering energy of one A-Lab target with Orb-v3 (one GPU): the Materials Project's ordered arrangement and N_RANDOM
random arrangements of the named mixing cations over their own sites, each relaxed at 0 K (cell and positions).

The mean energy of the random arrangements above the ordered one, against T times the ideal mixing entropy, bounds the
free energy of the disordered crystal from above (Gibbs-Bogoliubov with a uniform trial distribution). The analysis is
local (scripts/configurational_ordering.py).

Usage: python ordering_run.py TAG OUT_DIR   (TAG is a row of ordering_structures/targets.tsv)
A finished TAG.json is never redone; an unfinished one resumes after its last arrangement.
"""
import csv
import inspect
import json
import math
import os
import sys
import time

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("ORDERING_MODEL", "orb_v3_conservative_inf_omat")
N_RANDOM = int(os.environ.get("ORDERING_N_RANDOM", "24"))
FMAX, STEPS = 0.02, 600
SEED = 20261024


def load_calc(device):
    url = inspect.signature(getattr(pretrained, MODEL)).parameters["weights_path"].default
    local = os.path.join(HERE, "weights", url.rsplit("/", 1)[1])
    weights = local if os.path.exists(local) else url  # compute nodes are offline; setup_env.sh saves the weights
    return ORBCalculator(getattr(pretrained, MODEL)(weights_path=weights, device=device, precision="float32-high",
                                                    compile=False), device=device)


def relax(atoms):
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=FMAX, steps=STEPS)
    return opt.nsteps, float(np.abs(atoms.get_forces()).max())


def ideal_entropy(symbols, groups):
    """ln(number of distinct arrangements) per atom, exact for this cell (no Stirling)."""
    s = 0.0
    for g in groups:
        counts = [symbols.count(e) for e in g]
        s += math.lgamma(sum(counts) + 1) - sum(math.lgamma(c + 1) for c in counts)
    return s / len(symbols)


def arrangement(symbols, groups, rng):
    """A random arrangement: within each group, the group's atoms shuffled over the group's own sites."""
    new = list(symbols)
    for g in groups:
        idx = [j for j, s in enumerate(symbols) if s in g]
        for j, k in zip(idx, rng.permutation(idx)):
            new[j] = symbols[k]
    return new


def main():
    tag, out_dir = sys.argv[1], sys.argv[2]
    with open(os.path.join(HERE, "ordering_structures", "targets.tsv"), newline="") as f:
        row = next(r for r in csv.DictReader(f, delimiter="\t") if r["tag"] == tag)
    groups = [g.split("/") for g in row["groups"].split(";")]
    out = os.path.join(out_dir, f"{tag}.json")
    res = json.load(open(out)) if os.path.exists(out) else None
    if res and res.get("complete"):
        print(f"{tag}: already complete")
        return
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = load_calc(device)
    base = read(os.path.join(HERE, "ordering_structures", f"{tag}.cif"))
    symbols = base.get_chemical_symbols()
    if res is None:
        res = dict(tag=tag, target=row["target"], groups=row["groups"], T_K=float(row["T_K"]), atoms=len(base),
                   model=MODEL, device=device, fmax_target=FMAX, max_steps=STEPS, seed=SEED,
                   ideal_entropy_kB_per_atom=ideal_entropy(symbols, groups), ordered=None, random=[], complete=False)

    def run(atoms, label):
        t0 = time.time()
        atoms.calc = calc
        e_start = atoms.get_potential_energy() / len(atoms)
        steps, fmax = relax(atoms)
        rec = dict(label=label, E_per_atom=atoms.get_potential_energy() / len(atoms), E_unrelaxed_per_atom=e_start,
                   steps=steps, fmax=fmax, converged=fmax <= FMAX, volume_per_atom=atoms.get_volume() / len(atoms),
                   seconds=round(time.time() - t0, 1))
        print(f"{tag} {label}: E {rec['E_per_atom']:.5f} eV/atom, {steps} steps, fmax {fmax:.3f}", flush=True)
        return rec

    if res["ordered"] is None:
        res["ordered"] = run(base.copy(), "ordered")
        json.dump(res, open(out, "w"), indent=1)
    rng = np.random.default_rng(SEED)
    for i in range(N_RANDOM):
        new = arrangement(symbols, groups, rng)  # drawn even when skipped, so a resumed run continues the sequence
        if i < len(res["random"]):
            continue
        atoms = base.copy()
        atoms.set_chemical_symbols(new)
        res["random"].append(run(atoms, f"random_{i + 1:02d}"))
        json.dump(res, open(out, "w"), indent=1)
    e0 = res["ordered"]["E_per_atom"]
    d = np.array([r["E_per_atom"] for r in res["random"]]) - e0
    res["complete"] = True
    json.dump(res, open(out, "w"), indent=1)
    print(f"RESULT {tag}: random minus ordered {1000 * d.mean():+.1f} meV/atom (min {1000 * d.min():+.1f}); "
          f"T S_ideal {res['T_K'] * 8.617333e-2 * res['ideal_entropy_kB_per_atom']:.1f} meV/atom")


if __name__ == "__main__":
    main()
