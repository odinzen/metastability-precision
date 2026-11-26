"""Where does the Materials Project probably get the polymorph ground state wrong?

GGA/GGA+U favours the less dense structure: all eight measured polymorph misses put the less dense form lower, by 16
to 71 meV/atom (polymorph_decisions.py). For every oxide with two or more experimentally observed polymorphs
(matched to the ICSD, theoretical = False; oxygen the only anion), this lists the pairs in which an observed polymorph
is denser than the one GGA/GGA+U places lowest, by at least 3%, and lies above it by less than 75 meV/atom. These are
the candidates whose computed ground state the density bias could have inverted; the literature decides each.

Writes results/polymorph_density_candidates.csv (one row per formula, its GGA-lowest observed polymorph and the
densest candidate against it, with every other candidate listed).
"""
import sys
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key  # noqa: E402
from sigma_model import open_shell  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NON_OXIDE = {"C", "N", "S", "H", "Se", "Te", "F", "Cl", "Br", "I", "P"}  # P kept out: phosphides; PO4 handled below
MIN_DENSER, MAX_GAP = 3.0, 75.0


def main():
    with MPRester(api_key()) as mpr:
        docs = mpr.materials.summary.search(theoretical=False, elements=["O"], num_elements=(1, 6),
                                            fields=["material_id", "formula_pretty", "density", "symmetry",
                                                    "elements", "nsites", "database_IDs"])
        rows = []
        for d in docs:
            els = {str(e) for e in d.elements}
            if els & {"C", "N", "S", "H", "Se", "Te", "F", "Cl", "Br", "I"}:
                continue
            rows.append(dict(mpid=str(d.material_id), formula=d.formula_pretty, density=d.density,
                             spacegroup=d.symmetry.symbol, sg_number=d.symmetry.number, nsites=d.nsites,
                             icsd=";".join(str(i) for i in (d.database_IDs or {}).get("icsd", [])[:5])))
        obs = pd.DataFrame(rows)
        multi = obs.groupby("formula").filter(lambda g: g.sg_number.nunique() >= 2)
        ids = multi.mpid.tolist()
        energies = {}
        for i in range(0, len(ids), 500):
            for t in mpr.materials.thermo.search(material_ids=ids[i:i + 500], thermo_types=["GGA_GGA+U"],
                                                 fields=["material_id", "energy_per_atom"]):
                energies[str(t.material_id)] = t.energy_per_atom
    multi = multi.assign(e=multi.mpid.map(energies)).dropna(subset=["e"])
    # One entry per space group: the lowest-energy observed entry of that space group.
    multi = multi.sort_values("e").drop_duplicates(["formula", "sg_number"])
    out = []
    for f, g in multi.groupby("formula"):
        if g.sg_number.nunique() < 2:
            continue
        low = g.iloc[0]
        rest = g.iloc[1:].assign(gap=lambda x: 1000 * (x.e - low.e),
                                 denser=lambda x: 100 * (x.density / low.density - 1))
        cand = rest[(rest.denser >= MIN_DENSER) & (rest.gap < MAX_GAP)]
        if cand.empty:
            continue
        top = cand.sort_values("denser", ascending=False).iloc[0]
        out.append(dict(formula=f, open_shell=open_shell(f), n_observed_polymorphs=int(g.sg_number.nunique()),
                        gga_lowest=low.mpid, gga_lowest_sg=low.spacegroup, gga_lowest_density=round(low.density, 3),
                        denser=top.mpid, denser_sg=top.spacegroup, denser_density=round(top.density, 3),
                        denser_pct=round(top.denser, 1), gap_meV=round(top.gap, 1),
                        other_candidates="; ".join(f"{r.spacegroup} {r.mpid} +{r.gap:.0f} meV, {r.denser:.0f}% denser"
                                                   for _, r in cand.iloc[1:].iterrows() if r.mpid != top.mpid),
                        icsd_lowest=low.icsd, icsd_denser=top.icsd))
    res = pd.DataFrame(out).sort_values(["gap_meV"])
    res.to_csv(ROOT / "results" / "polymorph_density_candidates.csv", index=False)
    known = ["SiO2", "TiO2", "GeO2", "Al2SiO5", "Mg2GeO4", "Al2O3", "ZrO2", "HfO2"]
    print(f"{len(obs)} observed oxide entries, {multi.formula.nunique()} formulas with two or more observed space "
          f"groups; {len(res)} candidate formulas")
    print("known cases recovered:", {k: k in set(res.formula) for k in known})
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 60):
        print(res.drop(columns=["icsd_lowest", "icsd_denser", "other_candidates"]).to_string(index=False))


if __name__ == "__main__":
    main()
