"""Computed decisions for measured metastable polymorphs (results/polymorph_measured.csv).

Each pair is a decision whose true sign is known: the metastable polymorph lies above the stable one by the measured
enthalpy. The computed margin is the Materials Project energy difference of the two polymorphs, each the
lowest-energy entry of the formula that the database matches to an experimental (ICSD) structure and that lies in the
listed space groups (symmetry found with spglib at 0.1 A). Three polymorphs share their space group with another observed
phase of lower energy and are named by entry (NAMED). The database's corrections cancel between two entries of one
composition. Ground states settled from the literature (GROUND_STATE) enter the same way, most with the sign alone
(truth_sign). Writes results/polymorph_decisions.csv in the column layout of precision_calibration_decisions.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from emmet.core.mpid import AlphaID
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import api_key, load_or_pull, mp_number  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
# The lowest observed C2/c SiO2 entry is moganite (mp-12787, 2.43 g/cm3); coesite is mp-11684 (2.80, against 2.91 from
# its measured molar volume of 20.64 cm3/mol). The lowest observed P2_1/n Ca2SiO4 entry has 56 atoms per cell; larnite,
# Z = 4, is mp-4180 with 28. Walstromite and wollastonite are both triclinic; walstromite is mp-4428, 5.2% denser than
# wollastonite mp-4429 as the measured molar volumes are (37.94 against 39.93 cm3/mol, Akaogi et al. 2004).
NAMED = {("SiO2", "coesite"): 11684, ("Ca2SiO4", "beta (larnite)"): 4180, ("CaSiO3", "walstromite"): 4428}

# Ground states settled from the full papers (results/polymorph_adjudication.csv, verdict MP right or MP wrong): the
# structure the database puts lowest against the denser observed structure, or against the stable one where the
# database is wrong. Space-group numbers as in polymorph_measured.csv. A measured enthalpy is given only where it
# belongs to this very pair (kJ per formula unit and its uncertainty); the rest carry the sign alone.
GROUND_STATE = [
    ("CoO", "zinc blende", "216", "rock salt", "225", (37.0, 10.0)),
    ("Mn2O3", "cubic bixbyite", "206", "orthorhombic bixbyite", "61", None),
    ("MnO2", "hollandite", "87", "pyrolusite", "136", None),
    ("V2O3", "bixbyite type", "206", "corundum", "167", None),
    ("V2O5", "C2/c", "15", "orthorhombic", "59", None),
    ("ZrO2", "tetragonal", "137", "baddeleyite", "14", (10.61, 0.57)),
    ("ThSiO4", "huttonite", "14", "thorite", "141", (6.7, 2.5)),
    ("Y2O3", "B type", "12", "C type", "206", None),
    ("La2O3", "C type", "206", "A type", "164", None),
    ("Ce2O3", "C type", "206", "A type", "164", None),
    ("Pr2O3", "C type", "206", "A type", "164", None),
    ("Eu2O3", "A type", "164", "C type", "206", None),
    ("Gd2O3", "A type", "164", "C type", "206", None),
    ("Tb2O3", "B type", "12", "C type", "206", None),
    ("Ho2O3", "B type", "12", "C type", "206", None),
    ("Er2O3", "B type", "12", "C type", "206", None),
    ("PrO2", "Pnma", "62", "fluorite", "225", None),
    ("AlPO4", "I-4", "82", "berlinite", "152", None),
    ("Ca3(PO4)2", "alpha", "14", "beta (whitlockite type)", "146", None),
    ("MgGeO3", "clinopyroxene", "15", "ilmenite", "148", None),
]
# No beta-Ca3(PO4)2 entry is matched to ICSD. mp-531132 is its 91-atom primitive cell (Z = 7 of the R3c cell), with the
# partly occupied Ca sites ordered, which lowers the symmetry to R3.
GROUND_NAMED = {("Ca3(PO4)2", "beta (whitlockite type)"): 531132}
KJ_PER_EV = 96.4853


def chemsys(formula):
    return "-".join(sorted(str(e) for e in Composition(formula).elements))


def observed_ids(mpr, formula):
    docs = mpr.materials.summary.search(formula=Composition(formula).reduced_formula, theoretical=False,
                                        fields=["material_id"])
    return {int(AlphaID(str(d.material_id))) for d in docs}


def pick(entries, formula, groups, observed, named=None):
    red = Composition(formula).reduced_formula
    found = []
    for e in entries:
        if e.composition.reduced_formula != red or mp_number(e) not in observed:
            continue
        if named is not None and mp_number(e) != named:
            continue
        sg = SpacegroupAnalyzer(e.structure, symprec=0.1).get_space_group_number()
        if sg in groups:
            found.append((e.energy_per_atom, e, sg))
    if not found:
        return None
    energy, e, sg = min(found, key=lambda x: x[0])
    return e, sg


def main():
    m = pd.read_csv(ROOT / "results" / "polymorph_measured.csv")
    rows = []
    with MPRester(api_key()) as mpr:
        for _, r in m.iterrows():
            entries = load_or_pull(mpr, chemsys(r.formula))
            observed = observed_ids(mpr, r.formula)
            meta = pick(entries, r.formula, {int(x) for x in str(r.sg_metastable).split()}, observed,
                        NAMED.get((r.formula, r.metastable)))
            stable = pick(entries, r.formula, {int(x) for x in str(r.sg_stable).split()}, observed)
            if meta is None or stable is None:
                print(f"{r.formula} {r.metastable}/{r.stable}: no entry in the listed space groups "
                      f"({'metastable' if meta is None else 'stable'})")
                continue
            (em, sgm), (es, sgs) = meta, stable
            dft = 1000 * (em.energy_per_atom - es.energy_per_atom)  # meV per atom, positive = metastable above
            rows.append(dict(compound=f"{r.formula} ({r.metastable})", formula=r.formula, n_elements=len(Composition(r.formula)),
                             products=f"{r.formula} ({r.stable})", kind="polymorph", measured_meV_atom=r.dH_meV_atom,
                             dft_meV_atom=round(dft, 2), error_meV_atom=round(dft - r.dH_meV_atom, 2),
                             sign_established=bool(r.sign_established), truth_sign=1, basis="measured pair",
                             mp_metastable=f"mp-{mp_number(em)}", sg_metastable=sgm,
                             mp_stable=f"mp-{mp_number(es)}", sg_stable=sgs))
        verdicts = pd.read_csv(ROOT / "results" / "polymorph_adjudication.csv").set_index("formula").final_verdict
        for formula, meta_name, sg_meta, stable_name, sg_stable, measured in GROUND_STATE:
            entries = load_or_pull(mpr, chemsys(formula))
            observed = observed_ids(mpr, formula)
            meta = pick(entries, formula, {int(sg_meta)}, observed)
            named = GROUND_NAMED.get((formula, stable_name))
            stable = pick(entries, formula, {int(sg_stable)}, observed | {named} if named else observed, named)
            if meta is None or stable is None:
                raise ValueError(f"{formula}: no observed entry in space group {sg_meta if meta is None else sg_stable}")
            (em, sgm), (es, sgs) = meta, stable
            dft = 1000 * (em.energy_per_atom - es.energy_per_atom)
            if (dft > 0) != (verdicts[formula] == "MP right"):
                raise ValueError(f"{formula}: computed sign {dft:+.1f} disagrees with the verdict {verdicts[formula]}")
            n_at = Composition(formula).num_atoms
            dh = 1000 * measured[0] / n_at / KJ_PER_EV if measured else np.nan
            rows.append(dict(compound=f"{formula} ({meta_name})", formula=formula, n_elements=len(Composition(formula)),
                             products=f"{formula} ({stable_name})", kind="polymorph",
                             measured_meV_atom=round(dh, 2) if measured else np.nan, dft_meV_atom=round(dft, 2),
                             error_meV_atom=round(dft - dh, 2) if measured else np.nan,
                             sign_established=True, truth_sign=1, basis="ground state",
                             mp_metastable=f"mp-{mp_number(em)}", sg_metastable=sgm,
                             mp_stable=f"mp-{mp_number(es)}", sg_stable=sgs))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "polymorph_decisions.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.drop(columns=["products", "n_elements"]).to_string(index=False))
    ok = out[out.sign_established]
    for basis, g in ok.groupby("basis"):
        print(f"\n{basis}: {len(g)} decisions with an established sign; computed sign right for "
              f"{int((g.dft_meV_atom > 0).sum())}")


if __name__ == "__main__":
    main()
