"""How often does the precursor pair Chen et al. tested stay selected, and ranked first, when the computed energies
carry the scatter measured against calorimetry?

Each draw perturbs every entry of three or more elements in the target's chemical system by
sigma * (sqrt(rho) * eta + sqrt(1 - rho) * xi) per atom, sigma from sigma_model.py for that entry, with eta shared by the system and xi drawn per entry; binary
oxides and elements stay as computed (the database's fitted corrections anchor them to measurement). Their code then
rebuilds the hull, the pairwise reactions and the ranking from scratch (rerun_chen_today.run with a perturbed loader).
rho, the shared fraction of the variance, is set so that a compound-against-compound reaction keeps about 0.76 of the
scatter of a reaction from the binary oxides, as measured on the compiled data (sigma_by_reaction_type.py); rho = 0
is the uncorrelated case.

Usage: python chen_ranking_robustness.py [N_DRAWS]   (default 200)
"""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_chen_today as rc  # noqa: E402
from sigma_model import sigma_for  # noqa: E402

RHOS = (0.65, 0.0)
SEED = 20261024


def entries_for(els, cache={}):
    chemsys = "-".join(sorted(els))
    if chemsys not in cache:
        raw = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
        cache[chemsys] = [(e.composition, e.energy) for e in raw]
    return cache[chemsys]


def perturbed_loader(rng, rho):
    def load(els):
        base = entries_for(els)
        eta = rng.standard_normal()
        out = []
        for i, (comp, energy) in enumerate(base):
            if len(comp.elements) >= 3:
                sigma = sigma_for(comp.reduced_formula) / 1000  # eV per atom
                energy += comp.num_atoms * sigma * (np.sqrt(rho) * eta + np.sqrt(1 - rho) * rng.standard_normal())
            out.append(ComputedEntry(comp, energy, correction=0, entry_id=str(i)))
        return list(PhaseDiagram(out).stable_entries)
    return load


def one_target(args):
    target, robot_pair, n, rho, seed = args
    rng = np.random.default_rng(seed)
    selected = top = done = failed = 0
    for _ in range(n):
        try:
            ranked, _ = rc.run(target, perturbed_loader(rng, rho))
        except Exception:  # a draw can leave the target without a valid pathway; counted, and as not selected
            ranked = []
            failed += 1
        done += 1
        selected += any(r[0] == robot_pair for r in ranked)
        top += bool(ranked) and ranked[0][0] == robot_pair
    return dict(target=target, rho=rho, n=done, failed=failed, p_selected=selected / done, p_top=top / done)


def main(n):
    pub = rc.chen_published()
    jobs = []
    for k, (target, p) in enumerate(pub.items()):
        robot = [tuple(sorted(rc.norm(x) for x in s)) for s in p["robot_pairs"]]
        if len(robot) != 1:
            continue
        for rho in RHOS:
            jobs.append((target, robot[0], n if rho else n // 2, rho, SEED + 1000 * k + int(100 * rho)))
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(one_target, jobs))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "chen_ranking_robustness.csv", index=False)
    chen = pd.read_csv(ROOT / "results" / "precision_chen.csv")[["target", "p_selection_measured_scatter",
                                                                  "today_robot_pair_still_selected", "today_robot_pair_is_top"]]
    for rho, g in out.groupby("rho"):
        g = g.merge(chen, on="target")
        print(f"rho {rho}: {len(g)} targets; mean P(selected) {g.p_selected.mean():.2f}, mean P(first) {g.p_top.mean():.2f}; "
              f"P(selected) >= 0.9 for {(g.p_selected >= 0.9).sum()}, >= 0.5 for {(g.p_selected >= 0.5).sum()}; "
              f"P(first) >= 0.5 for {(g.p_top >= 0.5).sum()}")
        res = g.p_selection_measured_scatter > 1
        print(f"   resolved selections (p > 1, n={res.sum()}): mean P(selected) {g.p_selected[res].mean():.2f}; "
              f"unresolved (n={(~res).sum()}): {g.p_selected[~res].mean():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
