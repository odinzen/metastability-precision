"""MD supercells for the decisions whose harmonic entropy of reaction is unreliable.

Phonopy drops imaginary modes, so a decision with a phase that has many of them carries a large extra scatter at its
synthesis temperature (precision_temperature.py, imaginary_term_meV). Molecular dynamics at temperature does not need
the crystal to be harmonically stable. The MD entropy of a phase sits a few J/(K mol atoms) above measurement, an
offset that cancels within a reaction only when every phase of it comes from MD, so every phase of each such decision
is run, not only the unstable ones. The selections still waiting for phonons of their line neighbors are included, so
that one Sol round covers them whatever those phonons show.

Each phase is the conventional cell of its phonon structure (the lowest-energy Materials Project entry), repeated along
its thinnest direction until the supercell has at least 400 atoms and is at least 12 A wide, and runs at two
temperatures spanning the firing temperatures of the two campaigns (923 to 1273 K). Writes sol/structures/<tag>_<sg>.cif
and appends the missing rows to sol/runs/run_list.tsv; rows already there are kept as they are.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precision_temperature import build_decisions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "sol"
TEMPERATURES = (950.0, 1250.0)
IMAG_TERM = 5.0  # meV per atom
MIN_ATOMS, MIN_WIDTH = 400, 12.0


def widths(lattice, reps):
    m = lattice.matrix * np.array(reps)[:, None]
    vol = abs(np.linalg.det(m))
    return [vol / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3])) for i in range(3)]


def supercell(s):
    reps = [1, 1, 1]
    while len(s) * np.prod(reps) < MIN_ATOMS or min(widths(s.lattice, reps)) < MIN_WIDTH:
        reps[int(np.argmin(widths(s.lattice, reps)))] += 1
    return reps


def main():
    with contextlib.redirect_stdout(io.StringIO()):  # Chen's code prints every reaction
        decisions, _, _ = build_decisions()
    done = pd.read_csv(ROOT / "results" / "precision_temperature.csv")
    flagged = set(zip(done.dataset, done.decision, done.target)) - set(
        zip(*done.loc[done.imaginary_term_meV <= IMAG_TERM, ["dataset", "decision", "target"]].T.values))
    pending = {(ds, kind, target) for ds, kind, target, *_ in decisions} - set(zip(done.dataset, done.decision, done.target))
    phases = set()
    for ds, kind, target, t, products, *_ in decisions:
        if (ds, kind, target) in flagged | pending:
            phases |= {t, *products}
    print(f"{len(flagged)} decisions with an imaginary-mode term above {IMAG_TERM} meV per atom, {len(pending)} pending; "
          f"{len(phases)} phases")

    tags = pd.read_csv(SOL / "phonon_structures" / "phases.tsv", sep="\t").set_index("formula")
    runs_path = SOL / "runs" / "run_list.tsv"
    runs = pd.read_csv(runs_path, sep="\t")
    have = set(zip(runs.compound, runs.temperature_K))
    new = []
    for f in sorted(phases):
        tag = tags.loc[f, "tag"]
        prim = Structure.from_file(str(SOL / "phonon_structures" / f"{tag}.cif"))
        sga = SpacegroupAnalyzer(prim, symprec=0.01)
        conv = sga.get_conventional_standard_structure()
        sg = sga.get_space_group_symbol()
        cif = f"structures/{tag}_{sg.replace('-', 'm').replace('/', '')}.cif"
        if not (SOL / cif).exists():
            conv.to(filename=str(SOL / cif))
        reps = supercell(conv)
        for T in TEMPERATURES:
            if (f, T) not in have:
                new.append(dict(compound=f, temperature_K=T, cif=cif, reps="x".join(map(str, reps)),
                                n_atoms=len(conv) * int(np.prod(reps)), mp_id=tags.loc[f, "mp_id"], spacegroup=sg))
    if new:
        add = pd.DataFrame(new)
        add.insert(0, "index", range(int(runs["index"].max()) + 1, int(runs["index"].max()) + 1 + len(add)))
        pd.concat([runs, add]).to_csv(runs_path, sep="\t", index=False, lineterminator="\n")
    print(f"{len(new)} MD runs added, {len(runs) + len(new)} rows in run_list.tsv")
    if new:
        print(f"supercell atoms: median {add.n_atoms.median():.0f}, max {add.n_atoms.max()}")


if __name__ == "__main__":
    main()
