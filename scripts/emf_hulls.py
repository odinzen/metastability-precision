"""Measured-metastable chalcogenides from the EMF papers of the literature compilation, decided against the other
phases each paper measured.

An EMF study usually measures every phase of its system on one basis, so each phase can be placed against the
measured hull of the others, as the intermetallic decisions are (results/benchmark_nonoxide_calorimetric_neighbours.csv).
The hull of a paper holds its own line-compound formation enthalpies (confidence not low, no solid-solution flag), the
elements at zero, and, for a binary the paper does not print, the evaluated-table value the compilation holds (TKV,
Robie and Hemingway, Barin, NEA; calibration_nonoxide.compiled_binaries). Every phase of three or more elements with at least one
competing compound gives a decision: its measured margin above that hull and the computed margin of the same reaction
from Materials Project GGA/GGA+U entries (lowest entry of each formula, elements at zero). A phase several papers
measured enters from each, and the measured-metastable ones are listed with the paper behind them.
Writes results/emf_hull_decisions.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_nonoxide import pick, spacegroup  # noqa: E402
from calibration_nonoxide import compiled_binaries, emf_records  # noqa: E402
from pull_mp_systems import api_key, load_or_pull  # noqa: E402
from sigma_model import sigma_for  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
KJ_PER_EV = 96.485


def per_atom_mev(kj, formula):
    return 1000 * kj / KJ_PER_EV / Composition(formula).num_atoms


def paper_hull(g, binaries):
    """{reduced formula: (meV/atom from the elements, source)} for one paper's system."""
    phases = {r.compound: (per_atom_mev(r.kJ, r.compound), "paper") for r in g.itertuples()}
    els = sorted({str(e) for f in phases for e in Composition(f).elements})
    for f, (kj, src) in binaries.items():
        red = Composition(f).reduced_formula
        if red not in phases and {str(e) for e in Composition(f).elements} <= set(els):
            phases[red] = (per_atom_mev(kj, red), src)
    return phases, els


def main():
    df = emf_records()
    df = df[df.line & (df.confidence != "low")]
    df = df.groupby(["paper", "compound"], as_index=False).agg(kJ=("kJ", "mean"), u_kJ=("u_kJ", "mean"))
    binaries = compiled_binaries()
    rows = []
    with MPRester(api_key()) as mpr:
        for paper, g in df.groupby("paper"):
            phases, els = paper_hull(g, binaries)
            for target in g.compound:
                tc = Composition(target)
                if len(tc.elements) < 3:
                    continue
                own = {str(e) for e in tc.elements}
                others = {f: v for f, v in phases.items() if f != target and {str(e) for e in Composition(f).elements} <= own}
                if not any(len(Composition(f).elements) > 1 for f in others):
                    continue
                entries = [PDEntry(Composition(e), 0.0, name=e) for e in own]
                entries += [PDEntry(Composition(f), v / 1000 * Composition(f).num_atoms, name=f) for f, (v, _) in others.items()]
                pdm = PhaseDiagram(entries)
                t_entry = PDEntry(tc, phases[target][0] / 1000 * tc.num_atoms, name=target)
                decomp, e_meas = pdm.get_decomp_and_e_above_hull(t_entry, allow_negative=True)
                products = {p.name: f for p, f in decomp.items()}
                mp = load_or_pull(mpr, "-".join(sorted(own)))
                pdg = PhaseDiagram(mp)
                tgt, _ = pick(mp, target, match_reference=False)
                refs = {f: pick(mp, f, match_reference=False)[0] for f in products if f not in own}
                row = dict(paper=paper, compound=target, products="+".join(sorted(products)),
                           measured_meV_atom=round(1000 * e_meas, 1),
                           u_meV_atom=round(per_atom_mev(g.set_index("compound").u_kJ.get(target, np.nan), target), 1),
                           borrowed=";".join(sorted(f for f in products if f in others and others[f][1] != "paper")))
                if tgt is None or any(v is None for v in refs.values()):
                    rows.append(dict(row, status="missing from the database"))
                    continue
                dft = pdg.get_form_energy_per_atom(tgt) - sum(frac * pdg.get_form_energy_per_atom(refs[f])
                                                              for f, frac in products.items() if f not in own)
                same = [e for e in mp if e.composition.reduced_formula == tc.reduced_formula]
                rows.append(dict(row, status="ok", dft_meV_atom=round(1000 * dft, 1), mp_sg=spacegroup(tgt),
                                 polymorphs=len(same), sigma=sigma_for(target)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "emf_hull_decisions.csv", index=False)
    ok = out[out.status == "ok"]
    meta = ok[ok.measured_meV_atom > 0.5]
    print(f"{len(out)} decisions, {len(ok)} with DFT; measured metastable {len(meta)}, called metastable {(meta.dft_meV_atom > 0).sum()}")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(meta.sort_values("measured_meV_atom", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
