"""Measured enthalpy of formation from the binary oxides versus the same reaction in today's Materials Project
(GGA/GGA+U, MP2020 corrections), for the calorimetric values in results/anchor_measured_raw.csv.

Only single-lab calorimetric values measured from oxides go in. Mixed reactions with O2, doped or
off-stoichiometric samples, and compilation values are left out so the comparison is like for like.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key, load_or_pull  # noqa: E402
from mp_api.client import MPRester  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KJ_PER_EV = 96.4853

# Binary oxide each cation is referenced to, at the oxidation state it holds in these compounds.
OXIDE = {"Li": "Li2O", "Mg": "MgO", "Ca": "CaO", "Sr": "SrO", "Ba": "BaO", "Al": "Al2O3", "Ga": "Ga2O3",
         "In": "In2O3", "Sc": "Sc2O3", "Y": "Y2O3", "Ti": "TiO2", "Zr": "ZrO2", "Ge": "GeO2", "P": "P2O5",
         "Cr": "Cr2O3", "Mn": "Mn2O3", "Fe": "Fe2O3", "Co": "CoO", "Ni": "NiO", "Cu": "CuO", "Zn": "ZnO",
         "Cd": "CdO"}
RE = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
OXIDE.update({r: f"{r}2O3" for r in RE})

FROM_OXIDE_PAPERS = ("ushakov_2001", "helean_2004", "navrotsky_kleppa", "la2zr2o7_pyrochlore", "sahu_navrotsky",
                     "cheng_navrotsky", "goncalves_2014", "wang_navrotsky_2005")
FAMILY = {"ushakov_2001": "RE orthophosphates", "helean_2004": "RE titanate pyrochlores",
          "la2zr2o7": "RE titanate pyrochlores", "navrotsky_kleppa": "spinels", "cheng_navrotsky": "La perovskites",
          "sahu_navrotsky": "titanates and zirconates", "goncalves_2014": "titanates and zirconates",
          "wang_navrotsky_2005": "Li cathode oxides"}


def family_of(paper):
    return next(v for k, v in FAMILY.items() if k in paper)


def anchors() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "anchor_measured_raw.csv")
    df = df[(df.source_class == "measured_fact") & df.paper.str.contains("|".join(FROM_OXIDE_PAPERS))]
    basis = df.basis.fillna("")
    keep = ~basis.str.contains("elements|O2\\)|YBZ|Zr0\\.|La1\\.|La2\\.0", regex=True)
    keep &= ~df.compound.str.contains(r"\.", regex=True)
    keep &= ~((df.reduced == "La2Ti2O7") & df.table.str.contains("text"))  # quoted in prose, not Table 1
    df = df[keep].copy()
    # LiMnO2 is reported per Li2Mn2O4; the compilation also holds LiFeO2 with composition "LiFeO5" (logged as a data error).
    df.loc[df.reduced == "LiFeO5", "reduced"] = "LiFeO2"
    df["formula_units"] = np.where(df.basis.str.contains("Li2Mn2O4"), 2, 1)
    df["family"] = df.paper.map(family_of)
    return df


def oxide_split(formula):
    comp = Composition(formula)
    parts = {}
    for el, amt in comp.items():
        if el.symbol == "O":
            continue
        ox = Composition(OXIDE[el.symbol])
        parts[OXIDE[el.symbol]] = amt / ox[el]
    o_sum = sum(Composition(f)["O"] * n for f, n in parts.items())
    if abs(o_sum - comp["O"]) > 1e-6:
        raise ValueError(f"{formula}: oxygen not balanced by {parts}")
    return parts


# Polymorph the calorimetry actually referenced, where the source states it or the convention is fixed.
# Ushakov 2001: P2O5 XRD matched the NBS pattern "for the H form of P4O10 (R3c)"; heavy RE2O3 annealed to the C form.
REFERENCE_SG = {"P2O5": "R3c", "TiO2": "P4_2/mnm", "La2O3": "P-3m1", "Nd2O3": "P-3m1", "Pr2O3": "P-3m1",
                "Al2O3": "R-3c", "Cr2O3": "R-3c", "Fe2O3": "R-3c", "ZrO2": "P2_1/c", "Ga2O3": "C2/m",
                "Sc2O3": "Ia-3", "Y2O3": "Ia-3", "In2O3": "Ia-3", "Mn2O3": "Ia-3",
                **{f"{r}2O3": "Ia-3" for r in ["Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]}}


def spacegroup(entry):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    return SpacegroupAnalyzer(entry.structure, symprec=0.1).get_space_group_symbol()


def lowest(entries, formula, match_reference=False):
    red = Composition(formula).reduced_formula
    cands = [e for e in entries if e.composition.reduced_formula == red]
    if not cands:
        return None
    if match_reference and red in REFERENCE_SG:
        same = [e for e in cands if hasattr(e, "structure") and spacegroup(e) == REFERENCE_SG[red]]
        if same:
            return min(same, key=lambda e: e.energy_per_atom)
    return min(cands, key=lambda e: e.energy_per_atom)


def main():
    df = anchors()
    rows = []
    with MPRester(api_key()) as mpr:
        for _, r in df.iterrows():
            comp = Composition(r.reduced)
            chemsys = "-".join(sorted(str(e) for e in comp.elements))
            entries = load_or_pull(mpr, chemsys)
            parts = oxide_split(r.reduced)
            target = lowest(entries, r.reduced)
            refs = {f: lowest(entries, f, match_reference=True) for f in parts}
            refs_low = {f: lowest(entries, f) for f in parts}
            if target is None or any(v is None for v in refs.values()):
                rows.append(dict(family=r.family, compound=r.reduced, note="missing in MP"))
                continue
            pd_ = PhaseDiagram(entries)
            e_fu = target.energy_per_atom * comp.num_atoms

            def e_oxides(rr):
                return sum(n * rr[f].energy_per_atom * Composition(f).num_atoms for f, n in parts.items())

            dft_kj = (e_fu - e_oxides(refs)) * KJ_PER_EV
            dft_lowest_kj = (e_fu - e_oxides(refs_low)) * KJ_PER_EV
            meas_kj = r.value / r.formula_units
            rows.append(dict(
                family=r.family, compound=r.reduced, paper=r.paper, page=r.page, table=r.table, T_K=r.T_K,
                atoms_per_fu=comp.num_atoms, measured_kJ=round(meas_kj, 2), dft_kJ=round(dft_kj, 2),
                error_kJ=round(dft_kj - meas_kj, 2),
                error_meV_atom=round((dft_kj - meas_kj) / KJ_PER_EV / comp.num_atoms * 1000, 1),
                measured_meV_atom=round(meas_kj / KJ_PER_EV / comp.num_atoms * 1000, 1),
                error_meV_atom_lowest_polymorph_refs=round(
                    (dft_lowest_kj - meas_kj) / KJ_PER_EV / comp.num_atoms * 1000, 1),
                mp_e_above_hull_meV=round(1000 * pd_.get_e_above_hull(target), 1),
                oxides="+".join(f"{n:g} {f}" for f, n in parts.items()),
            ))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv", index=False)
    ok = out.dropna(subset=["error_meV_atom"])
    summary = ok.groupby("family").error_meV_atom.agg(
        n="count", mean_signed="mean", mean_abs=lambda s: s.abs().mean(), max_abs=lambda s: s.abs().max()).round(1)
    allrow = pd.DataFrame([dict(n=len(ok), mean_signed=round(ok.error_meV_atom.mean(), 1),
                                mean_abs=round(ok.error_meV_atom.abs().mean(), 1),
                                max_abs=round(ok.error_meV_atom.abs().max(), 1))], index=["ALL"])
    summary = pd.concat([summary, allrow])
    summary.to_csv(ROOT / "results" / "benchmark_summary.csv")
    print(summary.to_string())
    print("\nmissing:", out[out.get("note").notna()].compound.tolist() if "note" in out else [])


if __name__ == "__main__":
    main()
