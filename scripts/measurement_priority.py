"""Which measurements would resolve the most decisions? For every decision of the two synthesis datasets that the
energies do not resolve (margin within sigma = 22.5 meV/atom), list the compounds of three or more elements it rests
on: the target, its competitors on the hull (the decomposition products against all other phases), and for a Chen et
al. selection the two precursors. Compounds are ranked by the number of unresolved decisions they enter and marked as
measured when the compiled dataset (Wang et al. 2021) or the calorimetric benchmark holds an enthalpy of formation.
"""
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
SIGMA = 22.5


def red(f):
    return Composition(f).reduced_formula


def competitors(chemsys, target):
    entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
    t = red(target)
    others = PhaseDiagram([e for e in entries if e.composition.reduced_formula != t])
    return [e.composition.reduced_formula for e in others.get_decomposition(Composition(target))]


def main():
    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    measured = {red(r[0]) for r in d["data"]}
    measured |= {red(c) for c in pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").compound}

    today = pd.read_csv(ROOT / "results" / "mp_targets_today.csv").set_index(["dataset", "target"])
    chen = pd.read_csv(ROOT / "results" / "precision_chen.csv")
    alab = pd.read_csv(ROOT / "results" / "precision_alab.csv")
    alab = alab[~alab.database_inconsistency]

    decisions = []  # (label, [phases])
    for ds, df in (("Chen2024", chen), ("A-Lab2023", alab)):
        for _, r in df[df.stability_margin_meV.abs() <= SIGMA].iterrows():
            phases = [r.target] + competitors(today.loc[(ds, r.target), "chemsys"], r.target)
            decisions.append((f"{ds} stability {r.target}", phases))
    for _, r in chen[chen.selection_margin_meV <= SIGMA].iterrows():
        decisions.append((f"Chen2024 selection {r.target}", [r.target] + r.robot_pair.split("+")))

    count, where = defaultdict(int), defaultdict(list)
    for label, phases in decisions:
        for p in {red(x) for x in phases if len(Composition(x).elements) >= 3}:
            count[p] += 1
            where[p].append(label)
    out = pd.DataFrame([dict(compound=p, unresolved_decisions=n, measured=p in measured,
                             role="target" if any(lab.endswith(" " + p) for lab in where[p]) else "competitor or precursor",
                             decisions="; ".join(sorted(where[p]))) for p, n in count.items()])
    out = out.sort_values(["unresolved_decisions", "compound"], ascending=[False, True])
    out.to_csv(ROOT / "results" / "measurement_priority.csv", index=False)
    print(f"{len(decisions)} unresolved decisions rest on {len(out)} compounds of three or more elements, "
          f"{int(out.measured.sum())} of them measured")
    multi = out[out.unresolved_decisions > 1]
    print(f"{len(multi)} compounds enter more than one decision ({int(multi.measured.sum())} measured); together they "
          f"enter {len({l for ws in multi.decisions for l in ws.split('; ')})} decisions")
    print(out.head(30).drop(columns="decisions").to_string(index=False))


if __name__ == "__main__":
    main()
