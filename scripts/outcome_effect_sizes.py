"""Effect sizes with 95% bootstrap intervals for the outcome tests, so a non-significant result says how large an
effect the data can exclude: Spearman rho for the Chen et al. purity gain against each margin, and for the A-Lab the
rank-biserial correlation and the difference in median precision between targets obtained and not obtained.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20261024)
N_BOOT = 20000


def boot_ci(stat, *samples):
    vals = []
    for _ in range(N_BOOT):
        vals.append(stat(*(s[RNG.integers(0, len(s), len(s))] for s in samples)))
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    chen = pd.read_csv(ROOT / "results" / "precision_chen.csv")
    pairs = np.column_stack([chen.score_diff.values, chen.selection_margin_meV.values, chen.stability_margin_meV.abs().values])
    for j, lab in ((1, "selection margin"), (2, "stability margin")):
        rho = spearmanr(pairs[:, 0], pairs[:, j])
        lo, hi = boot_ci(lambda s: spearmanr(s[:, 0], s[:, j]).statistic, pairs)
        print(f"Chen purity gain vs |{lab}|: rho {rho.statistic:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}), P {rho.pvalue:.2f}, "
              f"n {len(pairs)}")

    alab = pd.read_csv(ROOT / "results" / "precision_alab.csv")
    alab = alab[~alab.database_inconsistency]
    ob = alab[alab.outcome == "obtained"].p_stability_measured_scatter.values
    no = alab[alab.outcome == "not obtained"].p_stability_measured_scatter.values
    u = mannwhitneyu(ob, no, alternative="greater")
    rbc = lambda a, b: 2 * mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1  # P(a > b) - P(a < b)
    lo, hi = boot_ci(rbc, ob, no)
    dlo, dhi = boot_ci(lambda a, b: np.median(a) - np.median(b), ob, no)
    print(f"A-Lab p, obtained (n {len(ob)}) vs not obtained (n {len(no)}): medians {np.median(ob):.2f} / {np.median(no):.2f}, "
          f"difference {np.median(ob) - np.median(no):+.2f} (95% CI {dlo:+.2f} to {dhi:+.2f}); rank-biserial "
          f"{rbc(ob, no):+.2f} (95% CI {lo:+.2f} to {hi:+.2f}); one-sided P {u.pvalue:.2f}")


if __name__ == "__main__":
    main()
