"""Does the oxide calibration carry over to sulfides, selenides, tellurides and nitrides?

Measured reactions from the binaries come from two sources kept apart: the calorimetric non-oxide benchmark
(benchmark_nonoxide.py, isovalent rows) and the EMF papers extracted on 2026-11-02 (results/nonoxide_measured_raw.csv,
rows from papers marked emf_paper, each with its source, table and page). For an EMF compound the
reaction is built from the binary values the same paper prints (quoted or measured). Most EMF papers measure the
potential of their electrode metal (Ag, Cu, Tl) directly and print no binary for it; that one binary is taken from an
evaluated table (TKV, then Robie and Hemingway, Barin, NEA), which carries no experimental error of its own,
and a reaction that would need two such values is left out. Only isovalent reactions enter: every
cation keeps the oxidation state pymatgen assigns it in the product, as in the oxide benchmark. Confidence-low records
and phases their extractors flagged as solid-solution or variable-composition are left out, and a compound measured
in several papers enters once (mean).

Each reaction is scored twice: its error against the oxide model frozen (sigma 22.5, or 34 for an open-shell
compound), and as a decision, whether the product is stable against those binaries, with p = |DFT|/sigma and
P_right = Phi(p). The maximum-likelihood sigma is fitted to the decision outcomes as in calibration_regimes.py.
Writes results/calibration_nonoxide.csv and results/calibration_nonoxide_summary.csv.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dft_vs_measured import KJ_PER_EV  # noqa: E402
from benchmark_nonoxide import measured, pick  # noqa: E402
from calibration_regimes import sigma_ml  # noqa: E402
from pull_mp_systems import api_key, load_or_pull  # noqa: E402
from sigma_model import open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["Sulfides", "Selenides", "Tellurides"]
ANIONS = ["S", "Se", "Te", "N", "Cl", "Br", "I"]
# Phases the extractors flagged as not line compounds (solid-solution end-members, saturated or variable-composition
# phases) carry line_compound False and stay out like the non-stoichiometric calorimetric samples.


def nan_to_none(x):
    return None if isinstance(x, float) and np.isnan(x) else x


def emf_records():
    m = measured()
    rows = []
    for system in SYSTEMS:
        for a in m[(m.system == system) & m.emf_paper].itertuples():
            if a.quantity != "formation_enthalpy" or a.unit != "J/mol":
                continue
            try:
                c = Composition(a.compound)
            except Exception:
                continue
            rows.append(dict(system=system, record=a.record, paper=a.paper, compound=c.reduced_formula,
                             formula=a.compound, n_el=len(c.elements), kJ=a.value / 1000,
                             u_kJ=(a.uncertainty or np.nan) / 1000, source=a.source_class,
                             confidence=nan_to_none(a.confidence), ref=nan_to_none(a.reference_state),
                             line=bool(a.line_compound)))
    return pd.DataFrame(rows)


def cation_states(formula):
    """Oxidation state of every non-anion element in the most likely assignment, or None."""
    c = Composition(formula)
    try:
        g = c.oxi_state_guesses(max_sites=-50)
    except Exception:
        return None
    if not g:
        return None
    return {el: round(v, 3) for el, v in g[0].items() if el not in ANIONS}


def split(product, binaries):
    """Unique positive combination of the binaries that balances the product, with every cation at the product's
    oxidation state. Returns (reactants, kind) where kind is isovalent, redox, ambiguous or none."""
    target = Composition(product)
    els = sorted(str(e) for e in target.elements)
    cands = [b for b in binaries if {str(e) for e in Composition(b).elements} <= set(els)]
    states = cation_states(product)
    found = []
    for k in range(2, min(len(cands), len(els)) + 1):
        for combo in itertools.combinations(cands, k):
            A = np.array([[Composition(b)[e] for b in combo] for e in els])
            nu, *_ = np.linalg.lstsq(A, np.array([target[e] for e in els]), rcond=None)
            if np.all(nu > 1e-9) and np.allclose(A @ nu, [target[e] for e in els], atol=1e-6):
                iso = states is not None and all(
                    (cation_states(b) or {}).get(el) == states.get(el) for b in combo for el in (cation_states(b) or {}))
                found.append(({b: float(n) for b, n in zip(combo, nu)}, iso))
        if found:
            break
    iso = [r for r, ok in found if ok]
    if len(iso) == 1:
        return iso[0], "isovalent"
    if len(iso) > 1:
        return None, "ambiguous"
    return (found[0][0], "redox") if found else (None, "none")


# Evaluated single-substance values, in order of preference, for a cation whose binary the EMF paper does not print
# (typically the Ag, Cu or Tl of its electrode, whose potential it measures directly against the metal).
COMPILED = ["paper:tkv", "paper:rh", "paper:robie_hemingway_1995", "paper:barin1995", "paper:nea_tin_2012"]


def compiled_binaries():
    best = {}
    m = measured()
    for system in SYSTEMS:
        for a in m[m.system == system].itertuples():
            ref = nan_to_none(a.reference_state)
            if (a.paper not in COMPILED or a.quantity != "formation_enthalpy"
                    or ref not in ("elements", None) or a.source_class == "modeling_result"):
                continue
            comp = nan_to_none(a.compound)
            try:
                c = Composition(comp)
            except Exception:
                continue
            scale = {"J/mol": 1e-3, "kJ/mol": 1.0}.get(nan_to_none(a.unit))
            if len(c.elements) != 2 or scale is None:
                continue
            rank = COMPILED.index(a.paper)
            if comp not in best or rank < best[comp][2]:
                best[comp] = (a.value * scale, a.paper, rank)
    return {f: (v, p) for f, (v, p, _) in best.items()}


def pair(f):
    """(cation, anion) of a binary; a paper covers the pair if it prints any binary of it."""
    els = [str(e) for e in Composition(f).elements]
    return next((e for e in els if e not in ANIONS), None), next((e for e in els if e in ANIONS), None)


def emf_reactions(df):
    compiled = compiled_binaries()
    out = []
    for paper, g in df.groupby("paper"):
        binaries = g[(g.n_el == 2) & (g.confidence != "low")].drop_duplicates("formula")
        bval = dict(zip(binaries.formula, binaries.kJ))
        covered = {pair(f) for f in bval}
        borrowed = {f: v for f, (v, _) in compiled.items() if pair(f) not in covered and f not in bval}
        bval.update(borrowed)
        for _, r in g[(g.n_el >= 3) & (g.source == "measured_fact")].iterrows():
            row = dict(method="EMF", paper=paper, record=r.record, compound=r.compound, formula=r.formula,
                       system=r.system, u_kJ=r.u_kJ)
            if r.confidence == "low" or not r.line or r.ref not in ("elements", None):
                out.append(dict(row, kind="left out", note=f"confidence {r.confidence}, line {r.line}, ref {r.ref}"))
                continue
            reactants, kind = split(r.formula, list(bval))
            # Borrowing the electrode cation's binary leaves the paper's own measurement in the reaction; borrowing two
            # makes the reaction depend on which evaluated values were picked (Aspiala 2016 AgSbS2: -103 meV/atom
            # against -11 from Bryndzia's calorimetry), so such reactions are left out.
            if reactants and sum(b in borrowed for b in reactants) > 1:
                kind = "two borrowed binaries"
            row.update(kind=kind)
            if reactants:
                c = Composition(r.formula)
                meas_kj = r.kJ - sum(n * bval[b] for b, n in reactants.items())
                row.update(reactants=reactants, measured_meV_atom=meas_kj / KJ_PER_EV / c.num_atoms * 1000,
                           borrowed="; ".join(f"{b} ({compiled[b][1].replace('paper:', '')})"
                                              for b in reactants if b in borrowed))
            out.append(row)
    return pd.DataFrame(out)


def dft_reaction(mpr, formula, reactants):
    comp = Composition(formula)
    entries = load_or_pull(mpr, "-".join(sorted(str(e) for e in comp.elements)))
    target, _ = pick(entries, formula, match_reference=False)
    refs = {b: pick(entries, b)[0] for b in reactants}
    if target is None or any(v is None for v in refs.values()):
        return None
    e = target.energy_per_atom * comp.num_atoms - sum(n * refs[b].energy_per_atom * Composition(b).num_atoms
                                                      for b, n in reactants.items())
    hull = PhaseDiagram(entries).get_e_above_hull(target)
    return 1000 * e / comp.num_atoms, 1000 * hull, str(target.entry_id)


def main():
    emf = emf_reactions(emf_records())
    cal = pd.read_csv(ROOT / "results" / "benchmark_nonoxide.csv")
    cal = cal[(cal.ref == "binaries, isovalent") & cal.excluded.isna() & cal.not_in_summary.isna()
              & cal.error_meV_atom.notna()]
    rows = []
    with MPRester(api_key()) as mpr:
        for _, r in emf[emf.kind == "isovalent"].iterrows():
            res = dft_reaction(mpr, r.formula, r.reactants)
            if res is None:
                rows.append(dict(r.drop("reactants"), note="no Materials Project entry"))
                continue
            dft, hull, mpid = res
            rows.append(dict(r.drop("reactants"), dft_meV_atom=dft, mp_e_above_hull_meV=hull, mp_id=mpid,
                             reaction=" + ".join(f"{n:.3g} {b}" for b, n in r.reactants.items())))
    emf_scored = pd.DataFrame(rows)
    cal_rows = pd.DataFrame(dict(method="calorimetry", paper=cal.record.str.split(":").str[1], record=cal.record,
                                 compound=cal.compound.map(lambda f: Composition(f).reduced_formula),
                                 formula=cal.compound, system=cal.system, kind="isovalent",
                                 measured_meV_atom=cal.measured_meV_atom, dft_meV_atom=cal.dft_meV_atom,
                                 mp_e_above_hull_meV=cal.mp_e_above_hull_meV, reaction=cal.reaction))
    allr = pd.concat([cal_rows, emf_scored], ignore_index=True)
    allr["error_meV_atom"] = allr.dft_meV_atom - allr.measured_meV_atom
    allr.round(2).to_csv(ROOT / "results" / "calibration_nonoxide.csv", index=False)

    s = allr.dropna(subset=["error_meV_atom"])
    per = s.groupby("compound").agg(measured=("measured_meV_atom", "mean"), dft=("dft_meV_atom", "mean"),
                                    hull=("mp_e_above_hull_meV", "first"),
                                    methods=("method", lambda x: "+".join(sorted(set(x)))),
                                    n=("record", "count")).reset_index()
    per["error"] = per.dft - per.measured
    per["anion"] = per.compound.map(lambda f: "+".join(a for a in ANIONS if a in {str(e) for e in Composition(f).elements}))
    per["open_shell"] = per.compound.map(open_shell)
    per["sigma"] = per.compound.map(sigma_for)
    b = pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    b = b.drop_duplicates(["compound", "measured_meV_atom"])
    k, c = np.polyfit(b.measured_meV_atom, b.error_meV_atom, 1)
    per["z"] = per.error / per.sigma
    per["z_trend"] = (per.error - (k * per.measured + c)) / per.sigma
    per["p"] = per.dft.abs() / per.sigma
    per["pred"] = norm.cdf(per.p)
    per["right"] = (np.sign(per.dft) == np.sign(per.measured)).astype(int)
    per.round(3).to_csv(ROOT / "results" / "calibration_nonoxide_compounds.csv", index=False)

    def stats(g, label):
        lo, hi = g[g.p <= 1], g[g.p > 1]
        sml = sigma_ml(g.right.values.astype(float), g.dft.abs().values) if 0 < g.right.sum() < len(g) else (np.nan,) * 3
        return dict(group=label, n=len(g), mean_error=round(g.error.mean(), 1), sd_error=round(g.error.std(ddof=1), 1),
                    rms_error=round(np.sqrt((g.error ** 2).mean()), 1), within_1sigma=round((g.z.abs() <= 1).mean(), 2),
                    within_2sigma=round((g.z.abs() <= 2).mean(), 2),
                    within_1sigma_trend=round((g.z_trend.abs() <= 1).mean(), 2),
                    right=int(g.right.sum()), n_low=len(lo), right_low=int(lo.right.sum()),
                    pred_low=round(lo.pred.mean(), 2) if len(lo) else np.nan, n_high=len(hi),
                    right_high=int(hi.right.sum()), pred_high=round(hi.pred.mean(), 3) if len(hi) else np.nan,
                    sigma_ml=sml[0], sigma_ml_lo=sml[1], sigma_ml_hi=sml[2])

    groups = [(per, "all")] + [(g, f"anion {a}") for a, g in per.groupby("anion")]
    groups += [(per[~per.open_shell], "closed shell"), (per[per.open_shell], "open shell")]
    groups += [(g, f"method {m}") for m, g in per.groupby("methods")]
    summary = pd.DataFrame([stats(g, lab) for g, lab in groups])
    summary.to_csv(ROOT / "results" / "calibration_nonoxide_summary.csv", index=False)

    with pd.option_context("display.width", 260, "display.max_rows", 300, "display.max_colwidth", 40):
        print(emf.kind.value_counts().to_string())
        print(emf[emf.kind != "isovalent"][["paper", "formula", "kind"]].to_string(index=False))
        print(per.round(1).sort_values("error").to_string(index=False))
        print(summary.to_string(index=False))
    print(f"frozen oxide trend: error = {k:+.3f} x measured {c:+.1f}")


if __name__ == "__main__":
    main()
