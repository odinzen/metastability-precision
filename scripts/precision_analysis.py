"""Metastability precision of the two synthesis datasets: each decision's margin divided by the measured uncertainty
of the energies behind it, sigma = 22.5 meV per atom (DFT scatter about the calorimetric trend), with 24 (Hautier
et al.) and 27.3 (Orb-v3 scatter) as sensitivity values. A ratio above 1 means the energies resolve the decision.

Margins:
- Chen et al., selection margin: the published inverse hull energy of the tested pair (line between the precursors).
- Chen et al. and A-Lab, stability margin: the signed distance of the target from the hull formed by every other
  phase in its full chemical system, today's Materials Project release (negative = below the hull).
The family offset is left out of sigma here: target and competitors in one decision share a chemistry, so the
proportional offset largely cancels between them, while the random scatter does not.
"""
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_mp_systems import mp_number  # noqa: E402
from emmet.core.mpid import AlphaID  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SIGMAS = {"measured_scatter": 22.5, "hautier2012": 24.0, "orbv3_scatter": 27.3}
INCONCLUSIVE = {"CaGd2Zr(GaO3)4", "KBaGdWO6", "Mn7(P2O7)4", "Mg3MnNi3O8"}  # 2026 correction, manual re-analysis


def signed_hull_distance(chemsys, formula, paper_mpid=None):
    entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
    red = Composition(formula).reduced_formula
    cands = [e for e in entries if e.composition.reduced_formula == red]
    if not cands:
        return np.nan
    if isinstance(paper_mpid, str):
        by_id = [e for e in cands if mp_number(e) == int(AlphaID(paper_mpid))]
        cands = by_id or cands
    target = min(cands, key=lambda e: e.energy_per_atom)
    others = PhaseDiagram([e for e in entries if e.composition.reduced_formula != red])
    return 1000 * others.get_e_above_hull(target, allow_negative=True)


def main():
    today = pd.read_csv(ROOT / "results" / "mp_targets_today.csv")
    chen = pd.read_csv(ROOT / "results" / "chen_rerun_today.csv")
    alab_csv = pd.read_csv(ROOT / "data" / "raw" / "alab2023" / "unzipped" / "20230502 Synthesis Results with Recipes.csv",
                           encoding="latin-1")
    rows = []
    for _, r in today.iterrows():
        d = signed_hull_distance(r.chemsys, r.target, r.paper_mpid)
        rows.append(dict(dataset=r.dataset, target=r.target, stability_margin_meV=round(d, 1)))
    st = pd.DataFrame(rows)

    chen = chen.merge(st[st.dataset == "Chen2024"][["target", "stability_margin_meV"]], on="target", how="left")
    chen["selection_margin_meV"] = chen.pub_invE_meV.abs()
    alab = st[st.dataset == "A-Lab2023"].merge(alab_csv[["Target", "Result"]], left_on="target", right_on="Target")
    alab["outcome"] = np.where(alab.target.isin(INCONCLUSIVE), "inconclusive",
                               np.where(alab.Result.str.startswith("Success"), "obtained",
                                        np.where(alab.Result == "Partial", "partial", "not obtained")))
    # Margins beyond 200 meV/atom are database inconsistencies, not chemistry (BaGdCrFeO6 at -2440; La5Mn5O16 at
    # -220, where the A-Lab paper itself notes DFT puts LaMnO3 120 meV/atom above the hull). Reported, not scored.
    alab["database_inconsistency"] = alab.stability_margin_meV.abs() > 200
    alab = alab[alab.stability_margin_meV.notna()].copy()
    for name, s in SIGMAS.items():
        chen[f"p_selection_{name}"] = chen.selection_margin_meV / s
        chen[f"p_stability_{name}"] = chen.stability_margin_meV.abs() / s
        alab[f"p_stability_{name}"] = alab.stability_margin_meV.abs() / s
    chen.to_csv(ROOT / "results" / "precision_chen.csv", index=False)
    alab.to_csv(ROOT / "results" / "precision_alab.csv", index=False)

    s = "measured_scatter"
    scored = alab[~alab.database_inconsistency]
    print("A-Lab: scored", len(scored), "| flagged as database inconsistency:", alab[alab.database_inconsistency].target.tolist(),
          "| no MP match: YbMoO4")
    lines = []
    for name, sig in SIGMAS.items():
        lines.append(dict(
            sigma=name, sigma_meV=sig,
            chen_selection_resolved=int((chen[f"p_selection_{name}"] > 1).sum()),
            chen_stability_resolved=int((chen[f"p_stability_{name}"] > 1).sum()), chen_n=len(chen),
            alab_stability_resolved=int((scored[f"p_stability_{name}"] > 1).sum()), alab_n=len(scored)))
    alab = scored
    summary = pd.DataFrame(lines)
    summary.to_csv(ROOT / "results" / "precision_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("\nA-Lab stability margins (meV/atom, negative = below the hull of all other phases):")
    print(alab.stability_margin_meV.describe().round(1).to_string())
    print("\nA-Lab outcome by precision (sigma 22.5):")
    alab["resolved"] = alab[f"p_stability_{s}"] > 1
    print(pd.crosstab(alab.outcome, alab.resolved).to_string())
    ob = alab[alab.outcome == "obtained"][f"p_stability_{s}"]
    no = alab[alab.outcome == "not obtained"][f"p_stability_{s}"]
    print(f"median p: obtained {ob.median():.2f} (n={len(ob)}), not obtained {no.median():.2f} (n={len(no)}); "
          f"Mann-Whitney obtained > not: p = {mannwhitneyu(ob, no, alternative='greater').pvalue:.3f}")
    rho = spearmanr(chen.stability_margin_meV.abs(), chen.score_diff)
    print(f"Chen: purity gain vs |stability margin| Spearman rho = {rho.statistic:.2f} (p = {rho.pvalue:.2f})")
    print("\nChen stability margins:", chen.stability_margin_meV.describe().round(1).to_dict())


if __name__ == "__main__":
    main()
