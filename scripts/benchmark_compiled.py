"""Wider test of the offset law: DFT against compiled experimental enthalpies of formation from the binary oxides,
for every ternary-or-higher oxide in the Wang/Kingsbury compilation (Kim 2017, Kubaschewski, NIST-JANAF; 2,135
solids) whose binary oxides are also in it.

Measured: dH_ox = dH_f(compound) - sum of nu * dH_f(binary oxides), all from the compilation.
DFT: the same reaction from Materials Project GGA/GGA+U entries, using the material ids the compilation assigns
(so polymorphs match). Compounds with a gas-forming acid oxide (C, N, S, H, halogens, Se) are excluded, and so is any
compound whose oxygen can be balanced by more than one choice of binary oxides (ambiguous oxidation states).
"""
import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key  # noqa: E402
from mp_api.client import MPRester  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAS_FORMING = {"C", "N", "S", "H", "Se", "F", "Cl", "Br", "I"}
EV_TO_KJ = 96.4853


def compiled():
    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    df = pd.DataFrame(d["data"], columns=d["columns"]).drop_duplicates("formula")
    df["comp"] = df.formula.map(Composition)
    df["els"] = df.comp.map(lambda c: frozenset(str(e) for e in c.elements))
    return df[df.els.map(lambda e: "O" in e)]


def reactions(df):
    binaries = df[df.els.map(len) == 2]
    by_cation = {}
    for _, r in binaries.iterrows():
        cation = next(e for e in r.els if e != "O")
        by_cation.setdefault(cation, []).append(r)
    out = []
    for _, r in df[(df.els.map(len) >= 3) & df.els.map(lambda e: not (e & GAS_FORMING))].iterrows():
        cations = [e for e in r.els if e != "O"]
        if not all(c in by_cation for c in cations):
            continue
        balanced = []
        for combo in itertools.product(*(by_cation[c] for c in cations)):
            o_sum = sum(r.comp[c] / b.comp[c] * b.comp["O"] for c, b in zip(cations, combo))
            if abs(o_sum - r.comp["O"]) < 1e-6:
                balanced.append(combo)
        if len(balanced) != 1:
            continue  # none, or ambiguous oxidation states
        combo = balanced[0]
        nu = {b.formula: r.comp[c] / b.comp[c] for c, b in zip(cations, combo)}
        n_atoms = r.comp.num_atoms
        # compilation values are eV per atom from the elements
        meas = (r.expt_form_e * n_atoms - sum(nu[b.formula] * b.expt_form_e * b.comp.num_atoms for b in combo)) / n_atoms
        out.append(dict(compound=r.formula, mpid=r.likely_mpid, n_atoms=n_atoms, measured_eV_atom=meas,
                        oxides={b.formula: (nu[b.formula], b.likely_mpid, b.comp.num_atoms) for b in combo}))
    return out


def main():
    rx = reactions(compiled())
    ids = sorted(i for i in {r["mpid"] for r in rx} | {o[1] for r in rx for o in r["oxides"].values()}
                 if isinstance(i, str))  # entries without an assigned material id come through as NaN
    energies = {}
    with MPRester(api_key()) as mpr:
        for chunk in [ids[i:i + 200] for i in range(0, len(ids), 200)]:
            docs = mpr.materials.thermo.search(material_ids=chunk, thermo_types=["GGA_GGA+U"],
                                               fields=["material_id", "energy_per_atom", "formula_pretty"])
            for doc in docs:
                energies[int(__import__("emmet.core.mpid", fromlist=["AlphaID"]).AlphaID(str(doc.material_id)))] = doc.energy_per_atom

    def e_of(mpid):
        if not isinstance(mpid, str):
            return None
        return energies.get(int(__import__("emmet.core.mpid", fromlist=["AlphaID"]).AlphaID(mpid)))

    rows = []
    for r in rx:
        e_c = e_of(r["mpid"])
        e_ox = [(nu, e_of(mid), n) for nu, mid, n in r["oxides"].values()]
        if e_c is None or any(e is None for _, e, _ in e_ox):
            continue
        dft = (e_c * r["n_atoms"] - sum(nu * e * n for nu, e, n in e_ox)) / r["n_atoms"]
        rows.append(dict(compound=r["compound"], measured_meV_atom=round(1000 * r["measured_eV_atom"], 1),
                         dft_meV_atom=round(1000 * dft, 1), error_meV_atom=round(1000 * (dft - r["measured_eV_atom"]), 1),
                         oxides="+".join(f"{nu:g} {f}" for f, (nu, _, _) in sorted(r["oxides"].items())), mpid=r["mpid"]))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "benchmark_compiled.csv", index=False)
    x, y = out.measured_meV_atom.values, out.error_meV_atom.values
    k, c = np.polyfit(x, y, 1)
    res = y - (k * x + c)
    print(f"reactions built {len(rx)}, with MP energies {len(out)}")
    print(f"slope {k:.3f} (DFT recovers {100 * (1 + k):.0f}%), intercept {c:.1f}, r {np.corrcoef(x, y)[0, 1]:.2f}, "
          f"residual SD {res.std(ddof=2):.1f}, mean abs error {np.abs(y).mean():.1f} meV/atom")
    bins = pd.cut(out.measured_meV_atom, [-1e9, -400, -200, -100, -50, 1e9])
    print(out.groupby(bins, observed=True).error_meV_atom.agg(["count", "mean", "median"]).round(1).to_string())


if __name__ == "__main__":
    main()
