"""How far cation disorder can move each A-Lab stability margin at the synthesis temperature, from the Orb-v3 ordering
energies computed on Sol (sol/ordering_run.py).

With dE the mean energy of random arrangements above the ordered one and S the ideal mixing entropy, the disordered
crystal's free energy lies at or below dE - T S relative to the ordered one (Gibbs-Bogoliubov, uniform trial
distribution). It cannot lie below the lowest arrangement's energy minus T S, so disorder lowers the target by between
max(0, T S - dE) and T S - min(0, dE_min), all per atom, with vibrations left out and the competing phases taken as
ordered. Where an arrangement lies below the ordered structure, the true ground state may lie lower still, so that upper
limit is itself a lower estimate.

Usage: python configurational_ordering.py [ORDERING_DIR]   (default ../sol/ordering)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K_B_MEV = 8.617333e-2
SIGMA = 22.5


def main(ordering_dir):
    bound = pd.read_csv(ROOT / "results" / "configurational_bound.csv").set_index("target")
    rows = []
    for path in sorted(Path(ordering_dir).glob("*.json")):
        r = json.load(open(path))
        if not r.get("complete"):
            print(f"{path.name}: unfinished ({len(r['random'])} arrangements)", file=sys.stderr)
            continue
        e0 = r["ordered"]["E_per_atom"]
        d = 1000 * (np.array([x["E_per_atom"] for x in r["random"]]) - e0)
        # The crystal is macroscopic: its mixing entropy is the large-cell limit, not the supercell's exact count
        # (0.22 against 0.28 k_B per atom for NaCaMgFe(SiO3)4 in 80 atoms).
        ts = r["T_K"] * K_B_MEV * bound.loc[r["target"], "S_bound_kB_per_atom"]
        gain_min, gain_max = max(0.0, ts - d.mean()), ts - min(0.0, d.min())
        margin = bound.loc[r["target"], "stability_margin_meV"]
        relaxed = [r["ordered"], *r["random"]]
        rows.append(dict(
            target=r["target"], groups=r["groups"], T_K=r["T_K"], atoms=r["atoms"], n_random=len(d),
            dE_random_meV_atom=round(d.mean(), 1), dE_random_sd=round(d.std(ddof=1), 1), dE_random_min=round(d.min(), 1),
            TS_ideal_meV_atom=round(ts, 1), dF_disorder_upper_meV_atom=round(d.mean() - ts, 1),
            gain_min_meV_atom=round(gain_min, 1), gain_max_meV_atom=round(gain_max, 1),
            stability_margin_meV=margin, margin_at_T_best=round(margin - gain_max, 1),
            margin_at_T_worst=round(margin - gain_min, 1),
            ordered_not_lowest=bool(d.min() < 0), unconverged=sum(not x["converged"] for x in relaxed)))
    if not rows:
        print("no finished ordering runs in", ordering_dir)
        return
    df = pd.DataFrame(rows).sort_values("dF_disorder_upper_meV_atom")
    df.to_csv(ROOT / "results" / "configurational_ordering.csv", index=False)
    with pd.option_context("display.width", 220):
        print(df.drop(columns=["groups", "atoms", "n_random"]).to_string(index=False))
    sure = df.dF_disorder_upper_meV_atom < 0
    print(f"\n{len(df)} targets: disorder lowers the free energy for certain in {sure.sum()} "
          f"(by at least {df.gain_min_meV_atom[sure].median():.1f} meV/atom, median); certain gain above sigma in "
          f"{(df.gain_min_meV_atom > SIGMA).sum()}, above |0 K margin| in "
          f"{(df.gain_min_meV_atom > df.stability_margin_meV.abs()).sum()}; ordered not the lowest arrangement in "
          f"{df.ordered_not_lowest.sum()}; unconverged relaxations {df.unconverged.sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "sol" / "ordering")
