"""Does the DFT error shrink when a compound competes with other compounds instead of its binary oxides?

For every ternary-or-higher oxide of the compiled benchmark (benchmark_compiled.py), the measured phase diagram of its
chemical system is built from the compiled enthalpies alone: its binary oxides plus every other compiled compound of
that system or its subsystems that uses the same binary oxides. The compound's decomposition reaction into the
lowest-enthalpy measured combination of the other phases is then evaluated twice, with the measured values and with
Materials Project GGA/GGA+U energies of the assigned material ids. Reactions whose products are binary oxides only are
the type the calorimetric scatter was measured on; reactions with at least one ternary or higher product are the
compound-against-compound decisions that make up most of a convex hull.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from emmet.core.mpid import AlphaID
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_compiled import compiled, reactions  # noqa: E402
from pull_mp_systems import api_key  # noqa: E402
from mp_api.client import MPRester  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def mp_energies(ids):
    out = {}
    with MPRester(api_key()) as mpr:
        for chunk in [ids[i:i + 200] for i in range(0, len(ids), 200)]:
            for doc in mpr.materials.thermo.search(material_ids=chunk, thermo_types=["GGA_GGA+U"],
                                                   fields=["material_id", "energy_per_atom"]):
                out[int(AlphaID(str(doc.material_id)))] = doc.energy_per_atom
    return out


def main():
    table = compiled().set_index("formula")
    rx = [r for r in reactions(compiled()) if isinstance(r["mpid"], str)]
    # A phase is its formula, measured formation enthalpy (eV/atom from the elements) and material id.
    phases = {}
    for r in rx:
        phases[r["compound"]] = (r["mpid"], {f: v[0] for f, v in r["oxides"].items()})
        for f, (_, mid, _) in r["oxides"].items():
            phases.setdefault(f, (mid, {f: 1.0}))
    ids = sorted({m for m, _ in phases.values() if isinstance(m, str)})
    e_dft = mp_energies(ids)

    def dft(formula):
        mid = phases[formula][0]
        return e_dft.get(int(AlphaID(mid))) if isinstance(mid, str) else None

    def meas(formula):
        return float(table.loc[formula, "expt_form_e"])

    rows = []
    for r in rx:
        x = r["compound"]
        if dft(x) is None:
            continue
        binaries = set(r["oxides"])
        cations = {e for e in Composition(x).elements if e.symbol != "O"}
        # Same binary oxides only, so every candidate shares the compound's oxidation states.
        cands = [f for f, (_, ox) in phases.items() if f != x and set(ox) <= binaries
                 and {e for e in Composition(f).elements if e.symbol != "O"} <= cations and dft(f) is not None]
        elements = [PDEntry(Composition({el: 1}), 0.0) for el in list(cations) + [Element("O")]]
        entries = {f: PDEntry(Composition(f), meas(f) * Composition(f).num_atoms, name=f) for f in cands}
        pd_meas = PhaseDiagram(elements + list(entries.values()))
        target = PDEntry(Composition(x), meas(x) * Composition(x).num_atoms, name=x)
        decomp, e_meas = pd_meas.get_decomp_and_e_above_hull(target, allow_negative=True)
        if any(len(p.composition.elements) == 1 for p in decomp):
            continue  # an element in the products means the oxide set did not close
        e_calc = dft(x) - sum(frac * dft(p.name) for p, frac in decomp.items())
        products = sorted(p.name for p in decomp)
        kind = ("binary oxides only" if all(len(Composition(p).elements) == 2 for p in products)
                else "includes a ternary or higher compound")
        rows.append(dict(compound=x, n_elements=len(Composition(x).elements), products="+".join(products), kind=kind,
                         measured_meV_atom=round(1000 * e_meas, 1), dft_meV_atom=round(1000 * e_calc, 1),
                         error_meV_atom=round(1000 * (e_calc - e_meas), 1)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "sigma_by_reaction_type.csv", index=False)
    print(f"{len(out)} decomposition reactions")
    for kind, g in out.groupby("kind"):
        e, m = g.error_meV_atom, g.measured_meV_atom
        k, c = np.polyfit(m, e, 1) if len(g) > 2 else (np.nan, np.nan)
        res = e - (k * m + c)
        print(f"{kind:40s} n={len(g):3d} mean {e.mean():+6.1f} SD {e.std():5.1f} MAE {e.abs().mean():5.1f} "
              f"| median |measured| {m.abs().median():6.1f} | slope {k:+.3f} residual SD {res.std(ddof=2):5.1f}")
    # Same-size comparison: compounds that appear in both classes cannot, so compare against the from-oxides errors of
    # exactly the compounds whose decomposition includes a compound.
    bench = pd.read_csv(ROOT / "results" / "benchmark_compiled.csv").set_index("compound")
    sub = out[out.kind.str.startswith("includes")]
    ox = bench.loc[sub.compound, "error_meV_atom"]
    print(f"\nsame {len(sub)} compounds, error of the reaction from binary oxides: mean {ox.mean():+.1f} SD {ox.std():.1f}"
          f" | error of the decomposition into compounds: mean {sub.error_meV_atom.mean():+.1f} "
          f"SD {sub.error_meV_atom.std():.1f}")
    print(sub.sort_values("error_meV_atom").to_string(index=False))


if __name__ == "__main__":
    main()
