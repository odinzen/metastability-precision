"""A-Lab outcomes against metastability precision: the test with and without the two targets left unscored for a
database inconsistency, and its power if a target were obtained exactly when its computed sign is right.

The power draw makes each of the 46 obtained or not-obtained targets succeed with probability Phi(p), the model's own
chance that the sign is right, and counts how often a one-sided Mann-Whitney test at 0.05 would then see obtained
targets at higher p. Writes results/alab_outcome_power.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

ROOT = Path(__file__).resolve().parents[1]
DRAWS, SEED, ALPHA = 5000, 20261027, 0.05


def test(df):
    ob = df[df.outcome == "obtained"].p_stability_measured_scatter
    no = df[df.outcome == "not obtained"].p_stability_measured_scatter
    return dict(n_obtained=len(ob), n_not=len(no), median_obtained=ob.median(), median_not=no.median(),
                P_one_sided=mannwhitneyu(ob, no, alternative="greater").pvalue)


def power(p):
    rng = np.random.default_rng(SEED)
    prob = norm.cdf(p)
    hits = runs = 0
    for _ in range(DRAWS):
        ok = rng.random(len(p)) < prob
        if ok.all() or not ok.any():
            continue
        runs += 1
        hits += mannwhitneyu(p[ok], p[~ok], alternative="greater").pvalue < ALPHA
    return hits / runs


def main():
    alab = pd.read_csv(ROOT / "results" / "precision_alab.csv")
    scored = alab[~alab.database_inconsistency]
    rows = [dict(case="scored", **test(scored)), dict(case="with the two unscored", **test(alab))]
    p = scored[scored.outcome.isin(["obtained", "not obtained"])].p_stability_measured_scatter.to_numpy()
    rows[0]["power_if_obtained_iff_right"] = power(p)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "alab_outcome_power.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
