"""Apply the calorimetric correction to the decisions themselves and count what changes.

The benchmark trend gives DFT = slope * measured + intercept for the enthalpy of formation from the binary oxides.
Inverting it, every oxide entry with three or more elements has its energy relative to the lower-order hull (binary
phases and elements, so the binary oxides for an oxide) replaced by (E_DFT - intercept) / slope; binary and elemental
entries are unchanged. The Chen et al. selection code and the stability margins are then rerun on the corrected
entries. If the offset cancels between a target and its competitors, as the precision analysis assumes, few decisions
should change; a second variant with the slope alone shows how much of any change comes from the intercept.
"""
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from emmet.core.mpid import AlphaID
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_chen_today as chen_rerun  # noqa: E402
from precision_analysis import INCONCLUSIVE  # noqa: E402
from pull_mp_systems import mp_number  # noqa: E402
from sigma_model import sigma_for  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def fit():
    d = pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    d = d.drop_duplicates(["compound", "measured_meV_atom"])  # the 69 values of Fig. 2 and of sigma
    k, c = np.polyfit(d.measured_meV_atom, d.error_meV_atom, 1)
    return 1 + k, c / 1000


def raw(chemsys):
    return MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))


def corrected(entries, slope, intercept):
    """Plain ComputedEntry list with the oxide reaction energies from the lower-order hull rescaled."""
    lower = PhaseDiagram([e for e in entries if len(e.composition.elements) <= 2])
    out = []
    for i, e in enumerate(entries):
        epa = e.energy_per_atom
        if len(e.composition.elements) >= 3 and Element("O") in e.composition:
            base = lower.get_hull_energy_per_atom(e.composition)
            epa = base + (epa - base - intercept) / slope
        out.append(ComputedEntry(e.composition, epa * e.composition.num_atoms, correction=0, entry_id=str(i),
                                 data={"mp": mp_number(e)}))
    return out


def stability_margin(entries, formula, paper_mpid=None):
    red = Composition(formula).reduced_formula
    cands = [e for e in entries if e.composition.reduced_formula == red]
    if not cands:
        return np.nan
    if isinstance(paper_mpid, str):
        cands = [e for e in cands if e.data["mp"] == int(AlphaID(paper_mpid))] or cands
    target = min(cands, key=lambda e: e.energy_per_atom)
    others = PhaseDiagram([e for e in entries if e.composition.reduced_formula != red])
    return 1000 * others.get_e_above_hull(target, allow_negative=True)


def main():
    slope, intercept = fit()
    print(f"benchmark fit: DFT = {slope:.4f} x measured + {1000 * intercept:.1f} meV/atom")
    variants = {"none": (1.0, 0.0), "full": (slope, intercept), "slope_only": (slope, 0.0)}
    today = pd.read_csv(ROOT / "results" / "mp_targets_today.csv")
    cache = {}

    def entries_for(chemsys, variant):
        key = (chemsys, variant)
        if key not in cache:
            cache[key] = corrected(raw(chemsys), *variants[variant])
        return cache[key]

    rows = []
    for _, r in today.iterrows():
        row = dict(dataset=r.dataset, target=r.target)
        for v in variants:
            row[f"stability_{v}"] = round(stability_margin(entries_for(r.chemsys, v), r.target, r.paper_mpid), 1)
        rows.append(row)
    st = pd.DataFrame(rows)

    pub = chen_rerun.chen_published()
    chen_rows = []
    for target, p in pub.items():
        robot = [tuple(sorted(chen_rerun.norm(x) for x in s)) for s in p["robot_pairs"]]
        pair = robot[0] if len(robot) == 1 else None
        row = dict(target=target, robot_pair="+".join(pair) if pair else None)
        for v in variants:
            def loader(els, v=v):
                return list(PhaseDiagram(entries_for("-".join(sorted(els)), v)).stable_entries)
            ranked, allp = chen_rerun.run(target, loader)
            row[f"selected_{v}"] = bool(pair) and any(x[0] == pair for x in ranked)
            row[f"top_{v}"] = bool(ranked) and ranked[0][0] == pair
            row[f"invE_{v}"] = round(allp[pair][1], 1) if pair in allp else np.nan
        chen_rows.append(row)
    ch = pd.DataFrame(chen_rows)

    alab = st[st.dataset == "A-Lab2023"]
    alab = alab[alab.stability_none.notna() & (alab.stability_none.abs() <= 200)]
    chs = st[st.dataset == "Chen2024"]
    ch.to_csv(ROOT / "results" / "correction_chen.csv", index=False)
    st.to_csv(ROOT / "results" / "correction_stability.csv", index=False)

    for v in ("full", "slope_only"):
        print(f"\n== correction: {v}")
        sel = ch.dropna(subset=["invE_none", f"invE_{v}"])
        d = sel[f"invE_{v}"] - sel.invE_none
        s_sel = sel.target.map(sigma_for)
        print(f"Chen selection, n={len(sel)}: tested pair selected {int(ch[f'selected_{v}'].sum())} "
              f"(uncorrected {int(ch.selected_none.sum())}), ranked first {int(ch[f'top_{v}'].sum())} "
              f"(uncorrected {int(ch.top_none.sum())}); inverse hull energy change median {d.median():+.1f}, "
              f"range {d.min():+.1f} to {d.max():+.1f} meV/atom; resolved {int((sel[f'invE_{v}'].abs() > s_sel).sum())} "
              f"(uncorrected {int((sel.invE_none.abs() > s_sel).sum())})")
        for lab, g in (("Chen stability", chs), ("A-Lab stability", alab)):
            g = g.dropna(subset=["stability_none", f"stability_{v}"])
            a, b = g.stability_none, g[f"stability_{v}"]
            sg = g.target.map(sigma_for)
            flip = ((a < 0) != (b < 0)).sum()
            cross = ((a.abs() > sg) != (b.abs() > sg)).sum()
            print(f"{lab}, n={len(g)}: change median {(b - a).median():+.1f}, range {(b - a).min():+.1f} to "
                  f"{(b - a).max():+.1f}; below hull {int((a < 0).sum())} -> {int((b < 0).sum())} (sign flips {flip}); "
                  f"resolved {int((a.abs() > sg).sum())} -> {int((b.abs() > sg).sum())} (crossings {cross})")
    print("\nChen pairs whose selection changes under the full correction:")
    print(ch[(ch.selected_none != ch.selected_full) | (ch.top_none != ch.top_full)].to_string(index=False))


if __name__ == "__main__":
    main()
