"""Measured enthalpies of ternary sulfides, nitrides and selenides against today's Materials Project (GGA/GGA+U,
MP2020 corrections), scored with the oxide error model frozen.

Two comparisons, kept apart. The first is the like-for-like one of the oxide benchmark: the reaction from the binary
compounds exactly as each paper measured it (its own reactants, elements or N2 included where the paper wrote them),
against the same reaction in the Materials Project. The second is formation from the elements against the corrected
Materials Project formation enthalpy, which also tests the fitted sulfide, selenide and nitride corrections; selenides
have only this one. The measured values, each with its source, table and page (page-checked 2026-11-02), are the
rows of results/nonoxide_measured_raw.csv marked calorimetric_benchmark, taken from the literature compilation.

Left out: McHale 1996 (superseded by McHale 1997), Popovich 2021 (sign error in the thermochemical cycle), and
non-stoichiometric samples (violarite, carrollite). High-temperature forms quenched to 298 K, derived values (Bryndzia
1989 "calculated") and literature values quoted by another paper are listed but kept out of the summary. Chalcopyrite
from the elements is reported as the spread of its four values, not one choice. A compound with several measured
reactions enters the summary once, as its mean error.

Writes results/benchmark_nonoxide.csv, results/benchmark_nonoxide_summary.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dft_vs_measured import KJ_PER_EV, spacegroup  # noqa: E402
from pull_mp_systems import api_key, load_or_pull  # noqa: E402
from sigma_model import open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / "results" / "nonoxide_measured_raw.csv"
SYSTEMS = ["Sulfides", "Nitrides", "Selenides"]

# Reactants of each measured reaction, per formula unit of product; a negative amount is a co-product (N2 given off).
REACTIONS = {
    "meas:cemic_1987:dHr_pentlandite_from_sulfides": {"FeS": 4, "NiS": 4, "Fe": 0.5, "Ni": 0.5},
    "meas:cemic_1988:dHr_CuFeS2_from_CuS_FeS": {"CuS": 1, "FeS": 1},
    "meas:cemic_1988:dHr_CuFeS2_from_Cu_FeS2": {"Cu": 1, "FeS2": 1},
    "meas:bryndzia_1988a:dHr_beta_AgBiS2_from_sulfides": {"Ag2S": 0.5, "Bi2S3": 0.5},
    "meas:bryndzia_1988a:dHr_alpha_AgBiS2_from_sulfides": {"Ag2S": 0.5, "Bi2S3": 0.5},
    "meas:bryndzia_1988a:dHr_AgBi3S5_from_sulfides": {"Ag2S": 0.5, "Bi2S3": 1.5},
    "meas:bryndzia_1988b:dHr_beta_AgSbS2_from_sulfides": {"Ag2S": 0.5, "Sb2S3": 0.5},
    "meas:bryndzia_1988b:dHr_alpha_AgSbS2_from_sulfides": {"Ag2S": 0.5, "Sb2S3": 0.5},
    "meas:bryndzia_1988b:dHr_Ag3SbS3_from_sulfides": {"Ag2S": 1.5, "Sb2S3": 0.5},
    "meas:bryndzia_1988c:dHr_CuSbS2_from_sulfides": {"Cu2S": 0.5, "Sb2S3": 0.5},
    "meas:bryndzia_1988c:dHr_Cu3SbS3_from_sulfides": {"Cu2S": 1.5, "Sb2S3": 0.5},
    "meas:stolyarova_2018:dHr_R1_stannite": {"Cu2S": 1, "FeS2": 1, "SnS": 1},
    "meas:stolyarova_2018:dHr_R2_stannite": {"CuS": 2, "FeS": 1, "SnS": 1},
    "meas:stolyarova_2019:dHr_kesterite_from_sulfides": {"CuS": 2, "ZnS": 1, "SnS": 1},
    "meas:stolyarova_2020:dHr_mohite_from_sulfides": {"CuS": 2, "SnS": 1},
    "meas:mchale_1997_nitrides:dHr_binaries_LiCaN_table3": {"Li3N": 1 / 3, "Ca3N2": 1 / 3},
    "meas:mchale_1997_nitrides:dHr_binaries_LiCaN_text": {"Li3N": 1 / 3, "Ca3N2": 1 / 3},
    "meas:mchale_1997_nitrides:dHr_binaries_LiZnN": {"Li3N": 1 / 3, "Zn3N2": 1 / 3},
    "meas:mchale_1997_nitrides:dHr_binaries_Ca2ZnN2": {"Ca3N2": 2 / 3, "Zn3N2": 1 / 3},
    "meas:mchale_1997_nitrides:dHr_binaries_Sr2ZnN2": {"Zn3N2": 1 / 3, "Sr2N": 2 / 3, "SrN": 2 / 3},
    "meas:mchale_1997_nitrides:dHr_binaries_CaTaN2_avg": {"Ca3N2": 1 / 3, "Ta3N5": 1 / 6, "TaN": 0.5},
    "meas:mchale_1997_nitrides:dHr_binaries_CaTaN2_eq72": {"Ca3N2": 1 / 3, "Ta3N5": 1 / 3, "N2": -1 / 6},
    "meas:mchale_1997_nitrides:dHr_binaries_CaTaN2_eq73": {"Ca3N2": 1 / 3, "TaN": 1, "N2": 1 / 6},
    "meas:elder_1993_limon2:dHr_binaries_LiMoN2": {"Mo2N": 0.5, "Li3N": 1 / 3, "N2": 7 / 12},
    # The paper writes 2 CeN + 1/3 Mn3N1.95 and leaves out the N2 that balances it (zero enthalpy of formation);
    # Mn3N1.95 is taken as Mn3N2, which moves the reaction by about 1 kJ/mol against a 15 kJ/mol uncertainty.
    "meas:tessier_2001_cemnn:dHr_binaries_Ce2MnN3": {"CeN": 2, "Mn3N2": 1 / 3, "N2": 1 / 6},
    "meas:ternary_nitride_1999:reaction_enthalpy_Li3BN2": {"Li3N": 1, "BN": 1},
    "meas:ternary_nitride_1999:reaction_enthalpy_Li3AlN2": {"Li3N": 1, "AlN": 1},
    "meas:ternary_nitride_1999:reaction_enthalpy_Ca3B2N4": {"Ca3N2": 1, "BN": 2},
}

# Binary reactants whose measured polymorph the papers state or convention fixes; anything else takes the
# lowest-energy entry. Cemic used the high-temperature NiS (NiAs type), not millerite.
REFERENCE_SG = {"FeS2": "Pa-3", "NiS": "P6_3/mmc", "CuS": "P6_3/mmc", "ZnS": "F-43m", "SnS": "Pnma",
                "Ag2S": "P2_1/c", "Sb2S3": "Pnma", "Bi2S3": "Pnma", "BN": "P6_3/mmc", "AlN": "P6_3mc",
                "CeN": "Fm-3m", "Li3N": "P6/mmm", "Ca3N2": "Ia-3", "Zn3N2": "Ia-3"}

# Reactions in which a formal oxidation state changes or N2 or a metal takes part. The oxide benchmark holds only
# reactions between binaries at the oxidation state of the product, so these are summarised apart.
REDOX = {
    "dHr_pentlandite_from_sulfides": "metals oxidised",
    "dHr_CuFeS2_from_CuS_FeS": "Fe(II) to Fe(III), persulfide of CuS reduced",
    "dHr_CuFeS2_from_Cu_FeS2": "Cu metal oxidised, persulfide reduced",
    "dHr_R1_stannite": "Sn(II) to Sn(IV), persulfide reduced",
    "dHr_R2_stannite": "Sn(II) to Sn(IV), persulfide of CuS reduced",
    "dHr_kesterite_from_sulfides": "Sn(II) to Sn(IV), persulfide of CuS reduced",
    "dHr_mohite_from_sulfides": "Sn(II) to Sn(IV), persulfide of CuS reduced",
    "dHr_binaries_Sr2ZnN2": "subnitride Sr2N and diazenide SrN as the Sr reference",
    "CaTaN2_eq72": "Ta(V) to Ta(IV), N2 given off",
    "CaTaN2_eq73": "Ta(III) to Ta(IV), N2 taken up",
    "CaTaN2_avg": "Ta(V) and Ta(III) to Ta(IV)",
    "dHr_binaries_LiMoN2": "Mo oxidised, N2 taken up",
    "dHr_binaries_Ce2MnN3": "Mn oxidised, N2 taken up",
}

EXCLUDED = {"mchale_1996": "superseded by McHale 1997", "popovich_2021": "sign error in the thermochemical cycle",
            "violarite": "non-stoichiometric natural sample", "carrollite": "non-stoichiometric natural sample",
            "dHr_pseudo_Na3WN3": "reaction from W2N and NaNH2, not from binary nitrides"}
# Listed, kept out of the summary.
NOT_SUMMARY = {"alpha_AgBiS2": "high-T form quenched to 298 K", "beta_AgSbS2": "high-T form quenched to 298 K",
               "smithite": "high-T form quenched to 298 K", "_calc": "derived from the paper's reaction value",
               "lit_dHf": "literature value quoted by Brichkina 2026", "LiCaN_text": "same measurement as Table 3",
               "CaTaN2_avg": "mean of eq. 72 and 73"}
CHALCOPYRITE = "CuFeS2"


def measured():
    return pd.read_csv(MEASURED, keep_default_na=False, na_values=[""], dtype={"page": str})


def records():
    m = measured()
    rows = []
    for system in SYSTEMS:
        for a in m[m.calorimetric_benchmark & (m.system == system)].itertuples():
            assert a.unit == "J/mol", a.record
            rows.append(dict(system=system, record=a.record, compound=a.compound, phase=a.phase,
                             quantity=a.quantity, kJ=a.value / 1000, u_kJ=(a.uncertainty or np.nan) / 1000,
                             basis=a.basis, page=a.page, reference_state=a.reference_state,
                             ref="binaries" if a.record in REACTIONS else "elements"))
    df = pd.DataFrame(rows)
    df["excluded"] = df.record.map(lambda r: next((v for k, v in EXCLUDED.items() if k in r), ""))
    unmapped = df[(df.excluded == "") & (df.ref == "elements") & (df.reference_state != "elements")]
    assert unmapped.empty, f"reactions with no reactants mapped: {unmapped.record.tolist()}"
    df["not_in_summary"] = df.record.map(lambda r: next((v for k, v in NOT_SUMMARY.items() if k in r), ""))
    df["redox"] = df.record.map(lambda r: next((v for k, v in REDOX.items() if k in r), ""))
    df.loc[df.ref == "binaries", "ref"] = np.where(df.loc[df.ref == "binaries", "redox"] == "",
                                                   "binaries, isovalent", "binaries, redox")
    return df


def pick(entries, formula, match_reference=True):
    red = Composition(formula).reduced_formula
    cands = [e for e in entries if e.composition.reduced_formula == red]
    if not cands:
        return None, ""
    if match_reference and red in REFERENCE_SG:
        same = [e for e in cands if spacegroup(e) == REFERENCE_SG[red]]
        if same:
            return min(same, key=lambda e: e.energy_per_atom), REFERENCE_SG[red]
        return min(cands, key=lambda e: e.energy_per_atom), f"no {REFERENCE_SG[red]} entry, lowest used"
    return min(cands, key=lambda e: e.energy_per_atom), "lowest"


def check_balance(target, reactants):
    left = {}
    for f, n in reactants.items():
        for el, amt in Composition(f).items():
            left[el.symbol] = left.get(el.symbol, 0) + n * amt
    right = {el.symbol: amt for el, amt in Composition(target).items()}
    for el in set(left) | set(right):
        if abs(left.get(el, 0) - right.get(el, 0)) > 1e-6:
            raise ValueError(f"{target}: reaction {reactants} not balanced in {el}")


def main():
    df = records()
    rows = []
    with MPRester(api_key()) as mpr:
        for _, r in df.iterrows():
            row = r.to_dict()
            if r.excluded:
                rows.append(row)
                continue
            comp = Composition(r.compound)
            chemsys = "-".join(sorted(str(e) for e in comp.elements))
            entries = load_or_pull(mpr, chemsys)
            target, _ = pick(entries, r.compound, match_reference=False)
            if target is None:
                rows.append(dict(row, note="no Materials Project entry at this composition"))
                continue
            pd_ = PhaseDiagram(entries)
            n_at = comp.num_atoms
            e_fu = target.energy_per_atom * n_at
            if r.record in REACTIONS:
                reactants = REACTIONS[r.record]
                check_balance(r.compound, reactants)
                refs = {f: pick(entries, f) for f in reactants}
                if any(e is None for e, _ in refs.values()):
                    rows.append(dict(row, note="binary reactant missing in Materials Project"))
                    continue
                dft_kj = (e_fu - sum(n * refs[f][0].energy_per_atom * Composition(f).num_atoms
                                     for f, n in reactants.items())) * KJ_PER_EV
                row["reaction"] = " + ".join(f"{n:.3g} {f}" for f, n in reactants.items())
                row["reference_polymorphs"] = "; ".join(f"{f} {how}" for f, (_, how) in refs.items()
                                                        if how not in ("lowest",))
            else:
                dft_kj = pd_.get_form_energy(target) / target.composition.num_atoms * n_at * KJ_PER_EV
            meas = r.kJ / KJ_PER_EV / n_at * 1000
            dft = dft_kj / KJ_PER_EV / n_at * 1000
            row.update(atoms_per_fu=n_at, mp_id=str(target.entry_id), mp_sg=spacegroup(target),
                       mp_e_above_hull_meV=round(1000 * pd_.get_e_above_hull(target), 1),
                       measured_meV_atom=round(meas, 1), dft_meV_atom=round(dft, 1),
                       error_meV_atom=round(dft - meas, 1),
                       u_meV_atom=round(r.u_kJ / KJ_PER_EV / n_at * 1000, 1),
                       open_shell=open_shell(comp.reduced_formula), sigma=sigma_for(comp.reduced_formula))
            rows.append(row)
    out = pd.DataFrame(rows)

    # Oxide error model, frozen: trend from the 69-value calorimetric benchmark, sigma from sigma_model.py.
    b = pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    b = b.drop_duplicates(["compound", "measured_meV_atom"])
    k, c = np.polyfit(b.measured_meV_atom, b.error_meV_atom, 1)
    out["z_raw"] = (out.error_meV_atom / out.sigma).round(2)
    out["z_trend"] = ((out.error_meV_atom - (k * out.measured_meV_atom + c)) / out.sigma).round(2)
    out.to_csv(ROOT / "results" / "benchmark_nonoxide.csv", index=False)

    scored = out.dropna(subset=["error_meV_atom"])
    usable = scored[(scored.excluded == "") & (scored.not_in_summary == "")]
    usable = usable[~((usable.ref == "elements") & (usable.compound == CHALCOPYRITE))]
    per_compound = usable.groupby(["ref", "system", "compound"]).agg(
        error=("error_meV_atom", "mean"), measured=("measured_meV_atom", "mean"), z_raw=("z_raw", "mean"),
        z_trend=("z_trend", "mean"), n_values=("record", "count")).reset_index()

    def stats(g, label):
        e = g.error
        return dict(group=label, n=len(g), mean_signed=round(e.mean(), 1), mean_abs=round(e.abs().mean(), 1),
                    sd=round(e.std(ddof=1), 1) if len(e) > 1 else np.nan, max_abs=round(e.abs().max(), 1),
                    within_1sigma_raw=round(np.mean(g.z_raw.abs() <= 1), 2),
                    within_2sigma_raw=round(np.mean(g.z_raw.abs() <= 2), 2),
                    within_1sigma_trend=round(np.mean(g.z_trend.abs() <= 1), 2),
                    within_2sigma_trend=round(np.mean(g.z_trend.abs() <= 2), 2))

    summ = [stats(g, f"{ref}: {sys_}") for (ref, sys_), g in per_compound.groupby(["ref", "system"])]
    summ += [stats(g, f"{ref}: all") for ref, g in per_compound.groupby("ref")]
    summ.append(stats(per_compound[per_compound.ref.str.startswith("binaries")], "binaries, isovalent and redox: all"))
    cp = scored[(scored.compound == CHALCOPYRITE) & (scored.ref == "elements")]
    summ.append(dict(group="chalcopyrite from the elements (spread, not in the rows above)", n=len(cp),
                     mean_signed=round(cp.error_meV_atom.mean(), 1),
                     max_abs=round(cp.error_meV_atom.abs().max(), 1),
                     sd=round(cp.error_meV_atom.std(ddof=1), 1)))
    summary = pd.DataFrame(summ)
    summary.to_csv(ROOT / "results" / "benchmark_nonoxide_summary.csv", index=False)

    cols = ["ref", "compound", "record", "measured_meV_atom", "dft_meV_atom", "error_meV_atom", "u_meV_atom",
            "z_raw", "z_trend", "mp_sg", "mp_e_above_hull_meV", "not_in_summary", "excluded", "note",
            "reference_polymorphs"]
    with pd.option_context("display.width", 300, "display.max_rows", 200, "display.max_colwidth", 48):
        print(out[[c_ for c_ in cols if c_ in out]].to_string(index=False))
        print()
        print(per_compound.round(1).to_string(index=False))
        print()
        print(summary.to_string(index=False))
    print(f"\nfrozen oxide trend: error = {k:+.3f} x measured {c:+.1f} meV/atom (from {len(b)} values)")
    print("chalcopyrite from the elements, measured kJ/mol:", cp[["record", "kJ"]].to_dict("records"))


if __name__ == "__main__":
    main()
