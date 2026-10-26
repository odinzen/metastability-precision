"""Phases whose harmonic entropy a temperature-aware precision needs: every target of the two campaigns, the products of
its decomposition on the current hull and, for a Chen et al. selection, the two precursors and the phases its inverse
hull energy is measured against. Each is the lowest-energy
Materials Project entry of its formula in the target's chemical system, written as its primitive cell to
sol/phonon_structures/<tag>.cif, with sol/phonon_structures/phases.tsv mapping tag to formula. Phases already in that
folder are kept. Elements and gases cannot take phonons; they are listed separately for tabulated entropies.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from precision_temperature import line_reaction  # noqa: E402
from pull_mp_systems import mp_number  # noqa: E402
from rerun_chen_today import norm  # noqa: E402
OUT = ROOT / "sol" / "phonon_structures"


def tag(formula):
    return re.sub(r"[()]", "", formula)


def lowest(entries):
    best = {}
    for e in entries:
        f = e.composition.reduced_formula
        if f not in best or e.energy_per_atom < best[f].energy_per_atom:
            best[f] = e
    return best


def main():
    today = pd.read_csv(ROOT / "results" / "mp_targets_today.csv").set_index(["dataset", "target"])
    chen = pd.read_csv(ROOT / "results" / "precision_chen.csv")
    alab = pd.read_csv(ROOT / "results" / "precision_alab.csv")
    alab = alab[(alab.dataset == "A-Lab2023") & (alab.database_inconsistency != True)]
    wanted = {}  # formula -> (entry, roles)
    for ds, df in (("Chen2024", chen), ("A-Lab2023", alab)):
        for _, r in df.dropna(subset=["stability_margin_meV"]).iterrows():
            cs = today.loc[(ds, r.target), "chemsys"]
            entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{cs}.json.gz")))
            best = lowest(entries)
            t = Composition(r.target).reduced_formula
            others = PhaseDiagram([e for e in entries if e.composition.reduced_formula != t])
            names = {t: "target"}
            for e in others.get_decomposition(Composition(r.target)):
                names.setdefault(e.composition.reduced_formula, "competitor")
            if ds == "Chen2024":
                for f in r.robot_pair.split("+"):
                    names.setdefault(Composition(f).reduced_formula, "precursor")
                # the phases behind the inverse hull energy, which a selection's entropy of reaction needs
                products, _ = line_reaction(r.target, [norm(x) for x in r.robot_pair.split("+")])
                for f in products or {}:
                    names.setdefault(f, "line neighbor")
            for f, role in names.items():
                if f in best:
                    entry, roles = wanted.get(f, (best[f], set()))
                    wanted[f] = (entry, roles | {role})
    OUT.mkdir(exist_ok=True)
    rows, gases = [], []
    for f, (entry, roles) in sorted(wanted.items()):
        if len(entry.composition.elements) == 1:
            gases.append(f)
            continue
        prim = entry.structure.get_primitive_structure()
        name = tag(f)
        path = OUT / f"{name}.cif"
        # A phase computed earlier for another purpose may be a different polymorph from the hull's entry (alpha-quartz
        # against the lowest-energy SiO2); the decisions need the hull's, so it gets its own tag.
        if path.exists() and not StructureMatcher().fit(Structure.from_file(str(path)), prim):
            name = f"{name}_hull"
            path = OUT / f"{name}.cif"
        if not path.exists():
            prim.to(filename=str(path))
        rows.append(dict(tag=name, formula=f, mp_id=f"mp-{mp_number(entry)}", atoms=len(prim), roles=";".join(sorted(roles))))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "phases.tsv", sep="\t", index=False, lineterminator="\n")
    print(f"{len(d)} phases with phonons ({len(list(OUT.glob('*.cif')))} CIFs in the folder); elements or gases: {gases}")
    print(f"primitive cell atoms: median {d.atoms.median():.0f}, 90th percentile {d.atoms.quantile(0.9):.0f}, "
          f"above 60: {(d.atoms > 60).sum()}, above 120: {(d.atoms > 120).sum()}")
    print(d.sort_values("atoms").tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
