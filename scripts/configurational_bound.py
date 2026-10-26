"""Configurational entropy the 0 K stability margins leave out, for the A-Lab targets whose cation ordering the 2026
re-refinement calls ambiguous or likely disordered.

Two numbers per target, both ideal (random) mixing, as T S at the synthesis temperature in meV per atom:
- bound: every atom of each named mixing group spread at random over all of that group's sites, read from the
  re-refinement notes, e.g. "(Mg/Ni) disorder". It is an upper bound for that disorder; vacancy disorder has no count in
  the notes and is left out (flagged).
- refined: the site occupancies of the A-Lab's own refined structure (Supplementary Data CIFs), where that model is
  disordered.
Ideal mixing ignores the energy of disordering and assumes the competing phases stay ordered; both make it an upper
bound on how far disorder can move a margin. The ordering energies that tighten it come from Orb-v3 on Sol.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ALAB = ROOT / "data" / "raw" / "alab2023" / "unzipped"
K_B_MEV = 8.617333e-2  # meV per K
SIGMA = 22.5
# Where the notes' wording would mix the wrong atoms, the group is taken from the refined model, with counts per
# formula unit (checked against the occupancies of the A-Lab CIFs).
OVERRIDES = {
    "NaCaMgFe(SiO3)4": ([{"Na": 1, "Ca": 1}, {"Mg": 1, "Fe": 1}],
                        "manual model, Na/Ca and Mg/Fe on separate sites (the automated one mixed all four)"),
    "Ba9Ca3La4(Fe4O15)2": ([{"Ca": 3, "La": 4}], "refined model; the Ba/Ca alternative also tested is not counted"),
    "KNaP6(PbO3)8": ([{"K": 1, "Na": 1, "Pb": 2}], "one 4f site holds K, Na and 2 of the 8 Pb"),
}
GROUP = re.compile(r"\(((?:[A-Z][a-z]?|vacancy)(?:\s*[/-]\s*(?:[A-Z][a-z]?|vacancy))+)\)")


def ordering_status(conclusion):
    c = conclusion.lower()
    if re.search(r"unlikely", c):
        return "ordered"
    if "ambiguous" in c:
        return "ambiguous"
    if "likely" in c:
        return "likely disordered"
    return None


def mixing_groups(notes):
    """Named mixing groups, merged where they share a species: (Ca/Na) and (Ca/Fe) are one Ca/Na/Fe group."""
    groups = []
    for m in GROUP.finditer(notes):
        g = set(re.split(r"\s*[/-]\s*", m.group(1)))
        for other in [o for o in groups if o & g]:
            groups.remove(other)
            g |= other
        groups.append(g)
    return groups


def bound_entropy(formula, groups):
    """Ideal mixing entropy in units of k_B per atom, and whether a vacancy group was dropped. A group is a set of
    species (all their atoms mix) or a dict of species to the number of atoms per formula unit that mix."""
    comp = Composition(formula)
    s, dropped = 0.0, False
    for g in groups:
        if isinstance(g, dict):
            n = np.array(list(g.values()), dtype=float)
            s -= float((n * np.log(n / n.sum())).sum())
            continue
        if "vacancy" in g:
            dropped = True
            g = g - {"vacancy"}
            if len(g) < 2:
                continue
        n = np.array([comp[e] for e in g])
        x = n / n.sum()
        s -= float((n * np.log(x)).sum())
    return s / comp.num_atoms, dropped


def refined_entropy(target):
    """Ideal entropy of the refined site occupancies (k_B per atom), or None without a CIF."""
    cif = ALAB / "Automated_Refinement_Results" / target / f"{target}.cif"
    if not cif.exists():
        return None
    st = Structure.from_file(str(cif))
    s, atoms = 0.0, 0.0
    for site in st:
        occ = np.array(list(site.species.as_dict().values()))
        vac = 1 - occ.sum()
        x = np.append(occ, vac) if vac > 1e-6 else occ
        s -= float((x * np.log(x)).sum())
        atoms += occ.sum()
    return s / atoms


def main():
    ref = pd.read_csv(ROOT / "results" / "alab_refinement_conclusions.csv").dropna(subset=["manual_notes"])
    recipes = pd.read_csv(ALAB / "20230502 Synthesis Results with Recipes.csv", encoding="latin-1")
    recipes["key"] = recipes.Target.map(lambda f: Composition(f).reduced_formula)
    prec = pd.read_csv(ROOT / "results" / "precision_alab.csv")
    prec = prec[prec.dataset == "A-Lab2023"].assign(key=lambda d: d.target.map(lambda f: Composition(f).reduced_formula))

    rows = []
    for _, r in ref.iterrows():
        status = ordering_status(str(r.conclusion))
        if status not in ("ambiguous", "likely disordered"):
            continue
        key = Composition(r.target).reduced_formula
        rec = recipes[recipes.key == key].iloc[0]
        t = float(rec["Temperature (C)"]) + 273.15
        groups, note = OVERRIDES.get(r.target, (mixing_groups(r.manual_notes), ""))
        s_b, dropped = bound_entropy(r.target, groups)
        s_r = refined_entropy(rec.Target)
        margin = prec[prec.key == key].stability_margin_meV
        rows.append(dict(
            target=r.target, ordering=status, mixing=" ; ".join("/".join(sorted(g)) for g in groups), note=note,
            offline=rec.Result.endswith("(offline)"),
            vacancy_left_out=dropped, T_synthesis_K=t, S_bound_kB_per_atom=round(s_b, 4),
            TS_bound_meV_atom=round(t * K_B_MEV * s_b, 1),
            S_refined_kB_per_atom=None if s_r is None else round(s_r, 4),
            TS_refined_meV_atom=None if s_r is None else round(t * K_B_MEV * s_r, 1),
            stability_margin_meV=float(margin.iloc[0]) if len(margin) else np.nan))
    d = pd.DataFrame(rows).sort_values("TS_bound_meV_atom", ascending=False)
    d.to_csv(ROOT / "results" / "configurational_bound.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 30):
        print(d.drop(columns=["S_bound_kB_per_atom", "S_refined_kB_per_atom"]).to_string(index=False))
    b = d.TS_bound_meV_atom
    print(f"\n{len(d)} targets ({(d.ordering == 'ambiguous').sum()} ambiguous, "
          f"{(d.ordering == 'likely disordered').sum()} likely disordered); T S bound median {b.median():.1f}, "
          f"range {b.min():.1f}-{b.max():.1f} meV per atom; above sigma {(b > SIGMA).sum()}; "
          f"above |margin| {(b > d.stability_margin_meV.abs()).sum()} of {d.stability_margin_meV.notna().sum()}")


if __name__ == "__main__":
    main()
