"""Metastability precision at the synthesis temperature (step E).

For every stability decision the reaction is the target against its decomposition products on the current hull. For a
Chen et al. selection it is the reaction behind the inverse hull energy, the target against the two points of the
precursor line that bound it on the line's hull, each resolved into its phases; these come from Chen's own code run on
the current release (rerun_chen_today.py) and are checked by recomputing the inverse hull energy from them. The harmonic entropies of every phase come from the
Sol phonon runs (sol/phonon_run.py, S_grid_per_mol_atoms) through sol/phonon_structures/phases.tsv; the entropy of
reaction per mole of atoms at the synthesis temperature T then shifts the margin:

    margin(T) = margin(0 K) - T * dS_rxn(T),   dS_rxn = S(target) - sum_j f_j S(product j)

with f_j the atom fractions of the products. The scatter at T adds the error of a computed entropy of reaction, 0.6 J
per K per mole of atoms (the spread between MD and phonons; 0.23 against measurement as the low case), to the sigma of
the energy (sigma_model.py). T is the A-Lab recipe temperature, or the median of the temperatures at which Chen et al. fired the tested
pair (MOESM7 file names). Targets with no recorded temperature (the failed A-Lab targets, whose recipes the
supplementary table leaves blank, and LiNbWO6) take the median of their campaign and are also evaluated across its
full range of firing temperatures. Decisions with any phase still missing are reported and skipped.

Usage: python precision_temperature.py [PHONON_DIR ...]    (later folders override earlier ones per phase)
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram
from pymatgen.core import Composition
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# Error of a computed entropy of reaction, J/(K mol atoms): 0.6, the spread between MD and phonons for the same phases
# (the reaction errors of the five validation reactions, 0.23 for phonons and 0.24 for MD, are the low case).
S_ERR, S_ERR_LOW_PHONON, S_ERR_LOW_MD = 0.6, 0.23, 0.24
MEV_PER_J_MOL = 1000 / 96485.33
# Phonopy leaves imaginary modes out of the entropy. A phase with a fraction f of them loses 3 f modes per atom, each
# worth k (1 + ln(kT / h nu)) at a typical |nu| of 2 THz, about 83 f J/(K mol atoms) at 1000 K. That loss, weighted by
# each phase's share of the reaction, is carried as a further term of the scatter at T; 0.005 is about 4 meV per atom.
IMAG_MAX = 0.005
IMAG_TERM_MAX = 5.0  # meV per atom; above it the MD entropy of reaction is used where every phase has one
# MD entropies (md_entropy_step_e.py): the spectrum route, 0.24 J/(K mol atoms) RMS against measured entropies of
# reaction at 1000 K in the round-2 validation; only runs that stayed vibrating crystals count.
MD_TEMPS = (950.0, 1250.0)
NU_IMAG, NU_SENSITIVITY = 2e12, (1e12, 4e12)  # Hz
K_OVER_H = 1.380649e-23 / 6.62607015e-34
R_GAS = 8.314462618


def imaginary_loss(f, T, nu=NU_IMAG):
    """Entropy (J/(K mol atoms)) a phase with a fraction f of imaginary modes is missing at T."""
    return 3 * f * R_GAS * (1 + np.log(K_OVER_H * T / nu))


sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_chen_today as chen_code  # noqa: E402
from sigma_model import sigma_for  # noqa: E402


def red(f):
    return Composition(f).reduced_formula


def chen_temperatures():
    """Firing temperatures (K) of each tested pair, from the MOESM7 column names, e.g. "['BaO', 'LiBO2']700C.txt"."""
    wb = openpyxl.load_workbook(ROOT / "data/raw/chen2024/si/MOESM7_ESM.xlsx", read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        temps = {}
        for h in next(ws.iter_rows(values_only=True)):
            m = re.search(r"\[(.*?)\](\d+)C", str(h))
            if m:
                pair = tuple(sorted(red(x.strip(" '")) for x in m.group(1).split(",")))
                temps.setdefault(pair, set()).add(float(m.group(2)) + 273.15)
        out[red(ws.title)] = temps
    return out


def entropy_table(phonon_dirs):
    tags = pd.read_csv(ROOT / "sol" / "phonon_structures" / "phases.tsv", sep="\t").set_index("tag")
    s = {}
    for path in (p for d in phonon_dirs for p in sorted(Path(d).glob("*.json"))):
        r = json.load(open(path))
        if path.stem in tags.index and "S_grid_per_mol_atoms" in r:
            grid = {float(k): v for k, v in r["S_grid_per_mol_atoms"].items()}
            imag = r["runs"][r.get("reference_run", "d0.01")]["imaginary_fraction"]
            s[tags.loc[path.stem, "formula"]] = (np.array(sorted(grid)), np.array([grid[k] for k in sorted(grid)]),
                                                 r["converged"], imag)
    return s


def line_reaction(target, pair):
    """Atom fractions of the phases the target is measured against in its inverse hull energy, and that energy (meV/atom).

    Chen's inverse hull energy compares the target with the entries that stay stable on the precursor line's hull while
    the target is on it. Those entries are kinks of the line, each a mixture of hull phases. Terminal compositions are
    normalized, so one transformed unit is one atom and the decomposition fractions are atom fractions.
    """
    chen_code.sp_mod.getOrigStableEntriesList = chen_code.today_stable
    sp = chen_code.sp_mod.SynthesisPathways(target, exclude_reactants=["O2"], selected_reactions_to_csv=False)
    want = tuple(sorted(pair))
    for r in sp.all_pairwise_reactions:
        if tuple(sorted(chen_code.norm(e.name) for e in r.reactants)) != want:
            continue
        cpd = CompoundPhaseDiagram(r.all_entries, [e.composition for e in r.reactants])
        tgt = next(e for e in cpd.stable_entries if chen_code.norm(e.name) == chen_code.norm(target))
        mod = PhaseDiagram([e for e in cpd.stable_entries if e.name != tgt.name], cpd.species_mapping.values())
        dec, inv_e = mod.get_decomp_and_e_above_hull(tgt, allow_negative=True)
        products = {}
        for nbr, f in dec.items():
            for e, g in sp.pd.get_decomposition(nbr.original_entry.composition).items():
                products[red(e.composition)] = products.get(red(e.composition), 0.0) + f * g
        check = tgt.original_entry.energy_per_atom - sum(f * n.original_entry.energy_per_atom for n, f in dec.items())
        assert abs(check - inv_e) < 1e-6, (target, check, inv_e)
        return products, 1000 * inv_e
    return None, np.nan


def S_at(s, f, T):
    temps, vals = s[f][:2]
    return float(np.interp(T, temps, vals))


def build_decisions():
    """Every decision as (dataset, kind, target, reduced target, product atom fractions, margin, T), with the campaign
    firing temperatures and the selections whose tested pair is not a computed reaction."""
    today = pd.read_csv(RES / "mp_targets_today.csv").set_index(["dataset", "target"])
    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[(alab.dataset == "A-Lab2023") & (alab.database_inconsistency != True)]
    recipes = pd.read_csv(ROOT / "data/raw/alab2023/unzipped/20230502 Synthesis Results with Recipes.csv", encoding="latin-1")
    recipes["key"] = recipes.Target.map(red)
    t_alab = recipes.groupby("key")["Temperature (C)"].first().astype(float) + 273.15
    t_chen = chen_temperatures()
    span = {"A-Lab2023": t_alab.dropna().values,
            "Chen2024": np.array([T for pairs in t_chen.values() for ts in pairs.values() for T in ts])}

    unbalanced, decisions = [], []
    for ds, df in (("Chen2024", chen), ("A-Lab2023", alab)):
        for _, r in df.dropna(subset=["stability_margin_meV"]).iterrows():
            cs = today.loc[(ds, r.target), "chemsys"]
            entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{cs}.json.gz")))
            t = red(r.target)
            dec = PhaseDiagram([e for e in entries if e.composition.reduced_formula != t]).get_decomposition(Composition(r.target))
            products = {}
            for e, frac in dec.items():
                products[e.composition.reduced_formula] = products.get(e.composition.reduced_formula, 0.0) + float(frac)
            if ds == "A-Lab2023":
                T = t_alab.get(t, np.nan)
            else:
                pair = tuple(sorted(red(x) for x in r.robot_pair.split("+")))
                T = float(np.median(sorted(t_chen.get(t, {}).get(pair, {np.nan}))))
            decisions.append((ds, "stability", r.target, t, products, r.stability_margin_meV, T))
    for _, r in chen.dropna(subset=["selection_margin_meV"]).iterrows():
        t = red(r.target)
        pair = [chen_code.norm(x) for x in r.robot_pair.split("+")]
        products, inv_e = line_reaction(r.target, pair)
        if products is None:  # the tested pair is not a computed reaction (Pr6O11 in place of Pr2O3 for Li3Pr2(BO3)3)
            unbalanced.append(r.target)
            continue
        assert abs(inv_e - r.today_invE_meV) < 0.1, (r.target, inv_e, r.today_invE_meV)
        T = float(np.median(sorted(t_chen.get(t, {}).get(tuple(sorted(red(x) for x in pair)), {np.nan}))))
        decisions.append(("Chen2024", "selection", r.target, t, products, -abs(r.selection_margin_meV), T))
    return decisions, span, unbalanced


def evaluate(s, md, decisions, span, s_err=S_ERR, s_err_md=S_ERR, nu=NU_IMAG):
    rows, missing = [], set()
    for ds, kind, target, t, products, margin, T in decisions:
        need = {t, *products}
        lack = [f for f in need if f not in s]
        if lack:
            missing |= set(lack)
            continue
        recorded = bool(np.isfinite(T))
        temps = [T] if recorded else [float(np.median(span[ds])), span[ds].min(), span[ds].max()]
        sigma = sigma_for(target)
        weights = {t: 1.0, **products}

        def at(T):
            TdS = T * (S_at(s, t, T) - sum(f * S_at(s, p, T) for p, f in sorted(products.items()))) * MEV_PER_J_MOL
            imag = T * sum(w * imaginary_loss(s[p][3], T, nu) for p, w in sorted(weights.items())) * MEV_PER_J_MOL
            return TdS, margin - TdS, np.sqrt(sigma ** 2 + (T * s_err * MEV_PER_J_MOL) ** 2 + imag ** 2), imag

        TdS, m_T, sig_T, imag_T = at(temps[0])
        p_range = [abs(m) / sg for _, m, sg, _ in map(at, temps)]

        # MD entropy of reaction, where every phase stayed a vibrating crystal at both MD temperatures; linear in T
        # between them (and a little beyond, 923 to 1273 K). MD samples every mode, so it carries no imaginary term.
        TdS_md = m_T_md = p_md = np.nan
        p_range_md = []
        if all((p, T_md) in md for p in need for T_md in MD_TEMPS):
            def at_md(T):
                ds_md = [md[(t, T_md)] - sum(f * md[(p, T_md)] for p, f in sorted(products.items())) for T_md in MD_TEMPS]
                slope = (ds_md[1] - ds_md[0]) / (MD_TEMPS[1] - MD_TEMPS[0])
                TdS = T * (ds_md[0] + slope * (T - MD_TEMPS[0])) * MEV_PER_J_MOL  # linear, extrapolated at the ends
                return TdS, margin - TdS, np.hypot(sigma, T * s_err_md * MEV_PER_J_MOL)

            TdS_md, m_T_md, sig_md = at_md(temps[0])
            p_md = abs(m_T_md) / sig_md
            p_range_md = [abs(m) / sg for _, m, sg in map(at_md, temps)]
        # The harmonic entropy stands where it is reliable; where imaginary modes make it unreliable the MD value
        # replaces it when there is one.
        use_md = imag_T > IMAG_TERM_MAX and np.isfinite(p_md)
        m_final, p_final = (m_T_md, p_md) if use_md else (m_T, abs(m_T) / sig_T)
        rows.append(dict(dataset=ds, decision=kind, target=target, T_K=temps[0], T_recorded=recorded,
                         sigma_0K=sigma, margin_0K=margin, TdS_rxn_meV=round(TdS, 1), margin_T=round(m_T, 1),
                         p_0K=abs(margin) / sigma, sigma_T=round(sig_T, 1), imaginary_term_meV=round(imag_T, 1),
                         p_T=abs(m_T) / sig_T, p_T_min_over_range=min(p_range), p_T_max_over_range=max(p_range),
                         flips=bool(np.sign(m_T) != np.sign(margin)), all_converged=all(s[f][2] for f in need),
                         max_imaginary_fraction=max(s[f][3] for f in need),
                         harmonic_ok=all(s[f][2] and s[f][3] <= IMAG_MAX for f in need),
                         P_right_T=norm.cdf(abs(m_T) / sig_T),
                         TdS_rxn_md_meV=round(TdS_md, 1), margin_T_md=round(m_T_md, 1), p_T_md=p_md,
                         p_T_md_min_over_range=min(p_range_md, default=np.nan),
                         entropy_source="MD" if use_md else "harmonic", margin_final=round(m_final, 1), p_final=p_final,
                         flips_final=bool(np.sign(m_final) != np.sign(margin)), P_right_final=norm.cdf(p_final)))
    out = pd.DataFrame(rows)
    # Chen's hull code sums in an order that varies between runs, which moves the last digits of the selection rows.
    p_cols = [c for c in out.columns if c.startswith("p_")]
    out[p_cols] = out[p_cols].round(4)
    out[["P_right_T", "P_right_final", "max_imaginary_fraction"]] = out[["P_right_T", "P_right_final", "max_imaginary_fraction"]].round(6)
    return out, missing


def counts(out):
    """Resolved at 0 K and at T, sign changes and expected right, per dataset and decision type."""
    rows = []
    for (ds, kind), g in out.groupby(["dataset", "decision"]):
        rows.append(dict(dataset=ds, decision=kind, n=len(g), resolved_0K=int((g.p_0K > 1).sum()),
                         resolved_T=int((g.p_final > 1).sum()), sign_changes=int(g.flips_final.sum()),
                         sign_changes_resolved_0K=int((g.flips_final & (g.p_0K > 1)).sum()),
                         expected_right_0K=round(float(norm.cdf(g.p_0K).sum()), 1),
                         expected_right_T=round(float(g.P_right_final.sum()), 1)))
    return pd.DataFrame(rows)


def main(phonon_dirs):
    s = entropy_table(phonon_dirs)
    md = {}
    if (RES / "md_entropy_step_e.csv").exists():
        m = pd.read_csv(RES / "md_entropy_step_e.csv")
        md = {(red(r.compound), r.T_K): r.S_vdos for r in m[m.crystal].itertuples()}
    decisions, span, unbalanced = build_decisions()
    out, missing = evaluate(s, md, decisions, span)
    out.to_csv(RES / "precision_temperature.csv", index=False)
    print(f"{len(out)} of {len(decisions)} decisions computed; phases still missing: {len(missing)}; "
          f"selections whose pair does not balance to the target: {unbalanced}")
    if missing:
        print("missing phases:", sorted(missing))
    print(counts(out).to_string(index=False))
    for (ds, kind), g in out.groupby(["dataset", "decision"]):
        print(f"{ds} {kind}: median |T dS| {g.TdS_rxn_meV.abs().median():.1f} meV/atom; sigma at T median "
              f"{g.sigma_T.median():.1f}; imaginary term above {IMAG_TERM_MAX} meV for "
              f"{(g.imaginary_term_meV > IMAG_TERM_MAX).sum()}, MD used for {(g.entropy_source == 'MD').sum()}")
    flips = out[out.flips_final]
    print("sign changes:", ", ".join(f"{r.target} ({r.decision}, p_0K {r.p_0K:.2f})" for r in flips.itertuples()))
    # Sensitivity: the low entropy error of the validation reactions, and the frequency assumed for imaginary modes.
    cases = [("central", S_ERR, S_ERR, NU_IMAG), ("validation error 0.23/0.24", S_ERR_LOW_PHONON, S_ERR_LOW_MD, NU_IMAG)]
    cases += [(f"nu {nu / 1e12:g} THz", S_ERR, S_ERR, nu) for nu in NU_SENSITIVITY]
    sens = pd.concat([counts(evaluate(s, md, decisions, span, a, b, nu)[0]).assign(case=name)
                      for name, a, b, nu in cases])
    sens.to_csv(RES / "precision_temperature_sensitivity.csv", index=False)
    print(sens.to_string(index=False))
    both = out[np.isfinite(out.TdS_rxn_md_meV) & (out.imaginary_term_meV <= IMAG_TERM_MAX)]
    diff = both.TdS_rxn_md_meV - both.TdS_rxn_meV
    print(f"cross-check, decisions with both a reliable harmonic and an MD entropy (n={len(both)}): MD minus harmonic "
          f"T dS median {diff.median():+.1f}, RMS {np.sqrt((diff ** 2).mean()):.1f}, max |{diff.abs().max():.1f}| meV/atom")


if __name__ == "__main__":
    main(sys.argv[1:] or [ROOT / "sol" / "phonons"])
