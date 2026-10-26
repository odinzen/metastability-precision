"""Supercells for the Orb-v3 ordering-energy runs on Sol (sol/ordering_run.py): the Materials Project structure each
A-Lab stability margin was computed from, for every target whose re-refinement names a cation mixing group.

Writes sol/ordering_structures/<tag>.cif and sol/ordering_structures/targets.tsv (tag, target, mixing groups,
synthesis temperature). Vacancy disorder is left out (the notes give no vacancy count), as in configurational_bound.py.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import mp_number  # noqa: E402
from emmet.core.mpid import AlphaID  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sol" / "ordering_structures"
MIN_ATOMS, MIN_MIXING_SITES, MAX_ATOMS = 80, 12, 260
# Only part of a species mixes here, on a site the Materials Project cell does not mark, so the arrangements cannot be
# drawn without guessing which atoms; its entropy comes from the refined occupancies alone.
SKIP = {"KNaP6(PbO3)8": "2 of 8 Pb share the K/Na site"}


def target_entry(chemsys, formula, paper_mpid):
    """The entry precision_analysis.signed_hull_distance uses: the paper's id if present, else the lowest energy."""
    entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
    red = Composition(formula).reduced_formula
    cands = [e for e in entries if e.composition.reduced_formula == red]
    if isinstance(paper_mpid, str):
        cands = [e for e in cands if mp_number(e) == int(AlphaID(paper_mpid))] or cands
    return min(cands, key=lambda e: e.energy_per_atom)


def supercell(structure, mixing):
    """Repeat along the shortest repeated axis until the cell has enough atoms and mixing sites to sample disorder."""
    scale = np.array([1, 1, 1])
    abc = np.array(structure.lattice.abc)
    n_mix = sum(1 for s in structure if s.specie.symbol in mixing)
    while len(structure) * scale.prod() < MIN_ATOMS or n_mix * scale.prod() < MIN_MIXING_SITES:
        trial = scale.copy()
        trial[np.argmin(abc * scale)] += 1
        if len(structure) * trial.prod() > MAX_ATOMS:
            break
        scale = trial
    return structure * scale, scale


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bound = pd.read_csv(ROOT / "results" / "configurational_bound.csv")
    today = pd.read_csv(ROOT / "results" / "mp_targets_today.csv")
    today = today[today.dataset == "A-Lab2023"].assign(key=lambda d: d.target.map(lambda f: Composition(f).reduced_formula))
    rows = []
    for _, r in bound.iterrows():
        if r.target in SKIP:
            print(f"{r.target:22s} skipped: {SKIP[r.target]}")
            continue
        groups = [set(g.split("/")) - {"vacancy"} for g in str(r.mixing).split(" ; ") if g and g != "nan"]
        groups = [g for g in groups if len(g) > 1]
        if not groups:
            print(f"{r.target:22s} skipped: no cation mixing group ({r.mixing})")
            continue
        t = today[today.key == Composition(r.target).reduced_formula].iloc[0]
        e = target_entry(t.chemsys, r.target, t.paper_mpid)
        mixing = set().union(*groups)
        sc, scale = supercell(e.structure, mixing)
        tag = r.target.replace("(", "").replace(")", "")
        sc.to(filename=str(OUT / f"{tag}.cif"))
        n_mix = sum(1 for s in sc if s.specie.symbol in mixing)
        rows.append(dict(tag=tag, target=r.target, groups=";".join("/".join(sorted(g)) for g in groups),
                         T_K=r.T_synthesis_K, atoms=len(sc), mixing_sites=n_mix, mp_number=mp_number(e)))
        print(f"{r.target:22s} mp-{mp_number(e):<9d} x{scale.tolist()} {len(sc):4d} atoms, {n_mix:3d} mixing sites")
    pd.DataFrame(rows).to_csv(OUT / "targets.tsv", sep="\t", index=False, lineterminator="\n")  # read on Sol
    print(f"{len(rows)} targets -> {OUT}")


if __name__ == "__main__":
    main()
