"""Which calorimetric measurements would make the most stability relations of the known oxides correct?

measurement_value.py ranks measurements for the two synthesis campaigns. This runs the same value of a measurement
over the 6,869 observed oxide relations of precision_mp_observed.py (each observed oxide against phases of fewer
elements or of as many): the relation is the target's decomposition reaction on the GGA/GGA+U hull, its error sits on
the phases of three or more elements, and a measurement of one compound conditions every relation that compound enters.
Candidates are observed (ICSD-matched) compounds of three or more elements that the compiled literature and the
benchmark have not measured, since only a compound that has been made can be measured.

For the compounds ranked first it writes the enthalpy of formation from the binary oxides that the error model of the
deposited predictions (preregistration_model.py) expects calorimetry to give, and for each relation they enter the
probability that measurement confirms the calculated sign. Outputs are hashed when written; the earlier deposited
files are not touched.

Usage: python measurement_value_observed.py [--pull] [--n=30]   (--pull queries the Materials Project again)
"""
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_design_ladder as L  # noqa: E402
import measurement_value as mv  # noqa: E402
import preregistration_model as P  # noqa: E402
from sigma_model import open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = ROOT / "data" / "raw" / "mp" / "observed_oxides_thermo_GGA.json.gz"
FORMULAS = ROOT / "data" / "raw" / "mp" / "lowest_formation_energy_GGA.json.gz"
RELEASE = "2026.09.28"
N_PICK = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 30))
ELSEWHERE = ROOT / "data" / "curated" / "measured_elsewhere.csv"
EXCLUDED = {"Hg", "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Pa", "Np", "Pu", "Am", "Cm"}
FIELDS = ["material_id", "formula_pretty", "is_stable", "energy_above_hull", "decomposition_enthalpy",
          "decomposes_to", "decomposition_enthalpy_decomposes_to", "formation_energy_per_atom"]


def red(f):
    return Composition(f).reduced_formula


def relations_table():
    # sodium nitride's formula is "NaN", which pandas reads as missing by default
    obs = pd.read_csv(RES / "precision_mp_observed.csv", keep_default_na=False, na_values={"margin_meV": [""]})
    rel = obs[obs.oxide & obs.competitor.isin(["fewer-element phases only", "same-order neighbour"])]
    return obs, rel


def pull(ids):
    from mp_api.client import MPRester
    from pull_mp_systems import api_key
    with MPRester(api_key()) as mpr:
        assert mpr.get_database_version() == RELEASE, "the paper uses release " + RELEASE
        docs = []
        for i in range(0, len(ids), 1000):
            docs += mpr.materials.thermo.search(material_ids=ids[i:i + 1000], thermo_types=["GGA_GGA+U"], fields=FIELDS)
    rows = []
    for t in docs:
        prods = t.decomposition_enthalpy_decomposes_to if t.is_stable else t.decomposes_to
        rows.append(dict(mpid=str(t.material_id), formula=red(t.formula_pretty), stable=bool(t.is_stable),
                         margin=1000 * (t.decomposition_enthalpy if t.is_stable else (t.energy_above_hull or 0.0)),
                         ef=t.formation_energy_per_atom,
                         products=[(red(p.formula), float(p.amount)) for p in (prods or [])]))
    json.dump(dict(release=RELEASE, thermo_type="GGA_GGA+U", rows=rows), gzip.open(RAW, "wt", encoding="utf-8"))


def lowest_ef(formulas):
    """Lowest GGA/GGA+U formation energy (eV/atom) of each reduced formula, cached across runs."""
    cache = json.load(gzip.open(FORMULAS, "rt", encoding="utf-8")) if FORMULAS.exists() else {}
    need = sorted(set(formulas) - set(cache))
    if need:
        from mp_api.client import MPRester
        from pull_mp_systems import api_key
        with MPRester(api_key()) as mpr:
            assert mpr.get_database_version() == RELEASE
            for i in range(0, len(need), 200):
                docs = mpr.materials.thermo.search(formula=need[i:i + 200], thermo_types=["GGA_GGA+U"],
                                                   fields=["formula_pretty", "formation_energy_per_atom"])
                for d in docs:
                    f = red(d.formula_pretty)
                    cache[f] = min(cache.get(f, 1e9), d.formation_energy_per_atom)
        for f in need:
            cache.setdefault(f, None)
        json.dump(cache, gzip.open(FORMULAS, "wt", encoding="utf-8"))
    return cache


def measurable(f):
    """A candidate needs a formation reaction from binary oxides (integer oxidation states) and no element that oxide
    melt solution calorimetry does not handle: Hg, whose oxide decomposes below the solvent temperature, and the
    radioactive elements other than Th and U."""
    els = {e.symbol for e in Composition(f).elements}
    return L.oxides_of(f) is not None and not els & EXCLUDED


def dft_from_oxides(f, ef):
    """Enthalpy of formation from the binary oxides (meV/atom) from lowest-energy formation energies, the same
    reaction as measurement_design_ladder.dft_from_oxides."""
    oxs = L.oxides_of(f)
    if not oxs or ef.get(f) is None or any(ef.get(o) is None for o in oxs.values()):
        return None
    comp = Composition(f)
    total = 0.0
    for el, o in oxs.items():
        oc = Composition(o)
        total += comp[el] / oc[el] * ef[o] * oc.num_atoms
    return 1000 * (ef[f] * comp.num_atoms - total) / comp.num_atoms


def main():
    obs, rel = relations_table()
    if "--pull" in sys.argv or not RAW.exists():
        pull(sorted(rel.mpid))
    raw = json.load(gzip.open(RAW, "rt", encoding="utf-8"))
    assert raw["release"] == RELEASE
    pulled = {r["mpid"]: r for r in raw["rows"]}
    # Same release, same margins as the paper's table, or the ranking would describe different relations.
    dm = np.array([abs(pulled[m]["margin"] - x) for m, x in zip(rel.mpid, rel.margin_meV) if m in pulled])
    print(f"{len(pulled)} of {len(rel)} relations pulled; largest margin difference from the paper's table "
          f"{dm.max():.4f} meV/atom")

    decs = []
    for r in pulled.values():
        w = {r["formula"]: 1.0}
        for f, a in r["products"]:
            w[f] = w.get(f, 0.0) - a
        w = {f: x for f, x in w.items() if len(Composition(f).elements) >= 3 and abs(x) > 1e-9}
        decs.append(dict(label=f"{r['formula']} ({r['mpid']})", target=r["formula"], margin=abs(r["margin"]),
                         margin_dft=r["margin"], weights=w, sigma=sigma_for(r["formula"])))

    d = json.load(gzip.open(ROOT / "data" / "raw" / "measured" / "expt_formation_enthalpy_kingsbury.json.gz"))
    literature = {red(x[0]) for x in d["data"]} | {red(c) for c in pd.read_csv(RES / "benchmark_dft_vs_measured.csv").compound}
    for t in ("holdout_test.csv",):
        literature |= {red(c) for c in pd.read_csv(RES / t).compound}
    for t, col in (("metastable_additions.csv", "formula"), ("primary_truth.csv", "formula")):
        literature |= {red(c) for c in pd.read_csv(ROOT / "data" / "curated" / t)[col]}
    # values found by the literature check of the ranked compounds (measured_elsewhere.csv): entered as measured
    elsewhere = pd.read_csv(ELSEWHERE, comment="#") if ELSEWHERE.exists() else pd.DataFrame(columns=["formula"])
    literature |= {red(c) for c in elsewhere.formula}
    phases = {f for x in decs for f in x["weights"]}
    known = phases & literature
    observed = {red(f) for f in obs.formula if f}
    candidates = sorted(c for c in (phases & observed) - known if measurable(c))
    touches = {c: [i for i, x in enumerate(decs) if c in x["weights"]] for c in candidates}
    print(f"{len(decs)} relations; {len(phases)} phases of three or more elements, {len(known)} measured, "
          f"{len(candidates)} observed and unmeasured candidates")

    chosen = set(known)
    current = [mv.expected(x, chosen)[0] for x in decs]
    base = sum(mv.expected(x, set())[0] for x in decs)
    lit = sum(current)
    gains = {c: sum(mv.expected(decs[i], chosen | {c})[0] - current[i] for i in touches[c]) for c in candidates}
    rows = [dict(rank=0, compound="none (as calculated)", expected_correct=round(base, 1)),
            dict(rank=0, compound=f"enter the {len(known)} already measured", expected_correct=round(lit, 1))]
    picks = []
    for k in range(1, N_PICK + 1):
        best = max(gains, key=gains.get)
        g = gains.pop(best)
        chosen.add(best)
        picks.append(best)
        affected = touches[best]
        for i in affected:
            current[i] = mv.expected(decs[i], chosen)[0]
        # only candidates sharing a relation with the new measurement change their gain
        for c in {c for i in affected for c in decs[i]["weights"] if c in gains}:
            gains[c] = sum(mv.expected(decs[i], chosen | {c})[0] - current[i] for i in touches[c])
        rows.append(dict(rank=k, compound=best, relations_entered=len(affected), gain=round(g, 2),
                         expected_correct=round(sum(current), 1)))
    ladder = pd.DataFrame(rows)

    # random orders of the same number of measurements, for scale
    rng = np.random.default_rng(20261114)
    rand = []
    for _ in range(50):
        pick = set(rng.choice(candidates, N_PICK, replace=False))
        s = set(known) | pick
        idx = {i for c in pick for i in touches[c]}
        rand.append(sum(mv.expected(decs[i], s)[0] - mv.expected(decs[i], set(known))[0] for i in idx))
    print(f"expected correct relations: as calculated {base:.1f}, with the literature {lit:.1f}, after {N_PICK} ranked "
          f"measurements {sum(current):.1f} (+{sum(current) - lit:.1f}); {N_PICK} random unmeasured compounds "
          f"+{np.mean(rand):.2f} (range {np.min(rand):.2f} to {np.max(rand):.2f})")

    # predictions for the ranked compounds and the relations they enter
    model = P.fit(P.measured_values())
    involved = {f for i in {i for c in picks for i in touches[c]} for f in decs[i]["weights"]}
    oxides = {o for f in involved | set(picks) for o in (L.oxides_of(f) or {}).values()}
    ef = lowest_ef(involved | set(picks) | oxides)
    meas = P.measured_values().set_index("compound")
    check = [(f, dft_from_oxides(f, ef), float(meas.loc[f, "dft"])) for f in involved if f in meas.index]
    diffs = np.array([a - b for _, a, b in check if a is not None])
    print(f"pipeline check, formation-energy DFT minus fitted DFT for {len(diffs)} measured compounds: median "
          f"{np.median(diffs):.1f}, max |{np.max(np.abs(diffs)):.1f}| meV/atom")

    comp_rows = []
    lad = ladder.set_index("compound")
    for rank, f in enumerate(picks, 1):
        dft = dft_from_oxides(f, ef)
        row = dict(rank=rank, compound=f, family=P.fam(f), open_shell=open_shell(f),
                   relations_entered=int(lad.loc[f, "relations_entered"]), expected_gain=float(lad.loc[f, "gain"]),
                   binary_oxides="+".join(sorted((L.oxides_of(f) or {}).values())))
        if dft is not None:
            m, sd = P.predict_measured(model, dft, P.fam(f), use_family=True)
            row.update(family_in_training=P.fam(f) in model["offsets"], dft_meV_atom=round(dft, 1),
                       predicted_measured_meV_atom=round(m, 1), interval_68=f"{m - sd:.1f} to {m + sd:.1f}",
                       interval_95=f"{m - 1.96 * sd:.1f} to {m + 1.96 * sd:.1f}")
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)

    def m_of(f):
        if f in meas.index:
            return float(meas.loc[f, "measured"]), 0.0
        dft = dft_from_oxides(f, ef)
        return P.predict_measured(model, dft, P.fam(f), use_family=True) if dft is not None else (None, None)

    dec_rows = []
    pickset = set(picks)
    for x in decs:
        hit = set(x["weights"]) & pickset
        if not hit:
            continue
        if any(m_of(f)[0] is None for f in x["weights"]):
            dec_rows.append(dict(relation=x["label"], note="a phase without a calculated enthalpy from oxides"))
            continue
        mu, sd = P.decision_prediction(model, x, m_of)
        dec_rows.append(dict(relation=x["label"], compounds_to_measure="+".join(sorted(hit)),
                             calculated_margin_meV_atom=round(x["margin_dft"], 1),
                             calculated_sign="stable (negative)" if x["margin_dft"] < 0 else "metastable (positive)",
                             predicted_true_margin_meV_atom=round(mu, 1), predicted_sd=round(sd, 1),
                             p_calculated_sign_confirmed=round(float(norm.cdf(np.sign(x["margin_dft"]) * mu / sd)), 3),
                             predicted_sign="negative" if mu < 0 else "positive"))
    decp = pd.DataFrame(dec_rows)

    ladder.to_csv(RES / "measurement_value_observed.csv", index=False)
    stamp = date.today().isoformat()
    h1 = P.write_hashed(RES / "forward_predictions_observed_oxides.csv",
                        f"# Forward predictions, written {stamp} before any of these compounds is measured. The {N_PICK} "
                        f"observed, unmeasured oxides whose calorimetric enthalpy would make the most of the {len(decs)} "
                        "stability relations of the known oxides correct (scripts/measurement_value_observed.py), with the "
                        "enthalpy of formation from the binary oxides (meV/atom, 298 K) that the error model of "
                        "preregistration_model.py expects calorimetry to give. Materials Project GGA/GGA+U, release "
                        f"{RELEASE}.\n", comp)
    h2 = P.write_hashed(RES / "forward_predictions_observed_relations.csv",
                        f"# Forward predictions, written {stamp}. For each stability relation of the known oxides that a "
                        "ranked compound enters: the calculated margin, the margin the model expects calorimetry to give, "
                        "and the probability that measurement confirms the calculated sign.\n", decp)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(ladder.to_string(index=False))
        print(comp.to_string(index=False))
        ok = decp.dropna(subset=["p_calculated_sign_confirmed"]) if "p_calculated_sign_confirmed" in decp else decp
        print(f"{len(decp)} relations, {len(ok)} predicted; expected confirmations "
              f"{ok.p_calculated_sign_confirmed.sum():.1f}; predicted to reverse "
              f"{int((ok.p_calculated_sign_confirmed < 0.5).sum())}")
        print(ok[ok.p_calculated_sign_confirmed < 0.5].to_string(index=False))
    print(f"\nsha256 compounds {h1}\nsha256 relations {h2}")


if __name__ == "__main__":
    main()
