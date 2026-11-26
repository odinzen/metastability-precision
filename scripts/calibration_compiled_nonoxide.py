"""Calibration of metastability precision on non-oxide decisions from the compiled enthalpies (Kim/Kingsbury, the
same compilation as the 120 oxide decisions).

For every ternary or higher non-oxide of the compilation (intermetallics; carbides, borides, silicides, phosphides
and pnictides; halides; nitrides), the measured phase diagram of its chemical system is built from the compiled
enthalpies of every other compiled phase of that system and its subsystems, plus the elements. The compound's
decomposition into the lowest-enthalpy measured combination of the other phases gives the measured sign, and the same
reaction with Materials Project GGA/GGA+U energies of the assigned material ids gives the computed sign and its p, as
in sigma_by_reaction_type.py. Unlike oxides, intermetallics compete with their elements, so decompositions that
include an element are kept and reported apart. Every decision takes sigma = 22.5 meV/atom: the open-shell regime is
defined by oxidation states, which metals do not have. Decisions whose measured value is exactly zero have no sign.
Writes results/calibration_compiled_nonoxide.csv and ..._summary.csv.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from emmet.core.mpid import AlphaID
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_regimes import sigma_ml  # noqa: E402
from pull_mp_systems import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMA = 22.5
HALOGEN = {"F", "Cl", "Br", "I"}
CHALC = {"S", "Se", "Te"}
LIGHT = {"C", "B", "Si", "P", "H", "As", "Sb", "Ge"}


def family(els):
    if "O" in els:
        return "oxide"
    if els & HALOGEN:
        return "halide"
    if els & CHALC:
        return "chalcogenide"
    if "N" in els:
        return "nitride"
    if els & LIGHT:
        return "carbide, boride, silicide, phosphide or pnictide"
    return "intermetallic"


def compiled():
    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    df = pd.DataFrame(d["data"], columns=d["columns"]).drop_duplicates("formula")
    df["comp"] = df.formula.map(Composition)
    df["els"] = df.comp.map(lambda c: frozenset(str(e) for e in c.elements))
    df["family"] = df.els.map(family)
    return df[df.family != "oxide"]


def mp_energies(mpr, ids):
    out = {}
    for chunk in [ids[i:i + 200] for i in range(0, len(ids), 200)]:
        for doc in mpr.materials.thermo.search(material_ids=chunk, thermo_types=["GGA_GGA+U"],
                                               fields=["material_id", "energy_per_atom"]):
            out[int(AlphaID(str(doc.material_id)))] = doc.energy_per_atom
    return out


def element_energies(mpr, elements):
    out = {}
    for el in elements:
        docs = mpr.materials.thermo.search(chemsys=el, thermo_types=["GGA_GGA+U"], fields=["energy_per_atom"])
        if docs:
            out[el] = min(d.energy_per_atom for d in docs)
    return out


def main():
    df = compiled()
    df = df[df.likely_mpid.map(lambda m: isinstance(m, str))]
    with MPRester(api_key()) as mpr:
        e_dft = mp_energies(mpr, sorted(set(df.likely_mpid)))
        e_el = element_energies(mpr, sorted({e for s in df.els for e in s}))
    df["e_dft"] = df.likely_mpid.map(lambda m: e_dft.get(int(AlphaID(m))))
    df = df.dropna(subset=["e_dft"])
    rows = []
    for _, r in df[df.els.map(len) >= 3].iterrows():
        sys_els = r.els
        if not all(e in e_el for e in sys_els):
            continue
        others = df[(df.formula != r.formula) & df.els.map(lambda s: s <= sys_els)]
        entries = [PDEntry(Composition({e: 1}), 0.0, name=e) for e in sys_els]
        entries += [PDEntry(o.comp, o.expt_form_e * o.comp.num_atoms, name=o.formula) for _, o in others.iterrows()]
        pd_meas = PhaseDiagram(entries)
        target = PDEntry(r.comp, r.expt_form_e * r.comp.num_atoms, name=r.formula)
        decomp, e_meas = pd_meas.get_decomp_and_e_above_hull(target, allow_negative=True)

        def e_of(name):  # DFT energy per atom of a product relative to the elements
            if name in sys_els:
                return 0.0
            o = df.set_index("formula").loc[name]
            c = o.comp
            return o.e_dft - sum(c.get_atomic_fraction(el) * e_el[str(el)] for el in c.elements)

        e_t = r.e_dft - sum(r.comp.get_atomic_fraction(el) * e_el[str(el)] for el in r.comp.elements)
        e_calc = e_t - sum(frac * e_of(p.name) for p, frac in decomp.items())
        names = sorted(p.name for p in decomp)
        kind = ("includes an element" if any(n in sys_els for n in names)
                else "binaries only" if all(len(Composition(n).elements) == 2 for n in names)
                else "includes a ternary or higher compound")
        rows.append(dict(compound=r.formula, family=r.family, n_elements=len(sys_els), products="+".join(names),
                         kind=kind, measured_meV_atom=round(1000 * e_meas, 1), dft_meV_atom=round(1000 * e_calc, 1),
                         error_meV_atom=round(1000 * (e_calc - e_meas), 1),
                         u_meas_meV_atom=round(1000 * r.uncertainty, 1) if pd.notna(r.uncertainty) else np.nan,
                         mpid=r.likely_mpid, reference=r.reference))
    out = pd.DataFrame(rows)
    out = out[out.measured_meV_atom != 0].copy()
    out["p"] = out.dft_meV_atom.abs() / SIGMA
    out["pred"] = norm.cdf(out.p)
    out["right"] = (np.sign(out.dft_meV_atom) == np.sign(out.measured_meV_atom)).astype(int)
    out["certain"] = out.measured_meV_atom.abs() >= 2 * out.u_meas_meV_atom.fillna(0)
    out.to_csv(ROOT / "results" / "calibration_compiled_nonoxide.csv", index=False)

    def stats(g, label):
        lo, hi = g[g.p <= 1], g[g.p > 1]
        sml = sigma_ml(g.right.values.astype(float), g.dft_meV_atom.abs().values) if 0 < g.right.sum() < len(g) \
            else (np.nan,) * 3
        return dict(group=label, n=len(g), measured_stable=int((g.measured_meV_atom < 0).sum()),
                    right=int(g.right.sum()), share_right=round(g.right.mean(), 3),
                    n_low=len(lo), right_low=int(lo.right.sum()), pred_low=round(lo.pred.mean(), 3) if len(lo) else np.nan,
                    n_high=len(hi), right_high=int(hi.right.sum()), pred_high=round(hi.pred.mean(), 3) if len(hi) else np.nan,
                    error_mean=round(g.error_meV_atom.mean(), 1), error_sd=round(g.error_meV_atom.std(ddof=1), 1),
                    sigma_ml=sml[0], sigma_ml_lo=sml[1], sigma_ml_hi=sml[2])

    groups = [(out, "all non-oxide")] + [(g, f"family: {f}") for f, g in out.groupby("family")]
    groups += [(g, f"kind: {k}") for k, g in out.groupby("kind")]
    groups += [(out[out.certain], "all, measured at least 2u from zero")]
    im = out[out.family == "intermetallic"]
    groups += [(im[im.kind != "includes an element"], "intermetallic, no element in the products")]
    summary = pd.DataFrame([stats(g, lab) for g, lab in groups])
    summary.to_csv(ROOT / "results" / "calibration_compiled_nonoxide_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
