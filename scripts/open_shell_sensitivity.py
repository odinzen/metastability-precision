"""How the headline numbers move with the sigma of open-shell decisions.

The adopted sigma_open is 34 meV/atom, the lowest value the 21 audited open-shell decisions allow. Their maximum-
likelihood fit is 66, and two confident misses drive it: CeCrO3 and CuAl2O4. Here sigma_open is set to 22.5 (one sigma
for every compound decision, the calorimetric value, under which the open-shell benchmark scatters as much as the rest),
to the fit without CeCrO3, to 34 and to the fit with all 21; closed-shell decisions keep 22.5 throughout. Reported for
each: the open-shell calibration (shares right at p <= 1 and p > 1 against Phi(p), Brier), the resolved and expected
right counts of the two campaigns, and the expected right share of the observed Materials Project oxides. GNoME runs
the same values in gnome_precision.py. Reads saved tables only; writes results/open_shell_sensitivity.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_regimes import decisions, sigma_ml  # noqa: E402
from sigma_model import SIGMA_CLOSED, SIGMA_OPEN, open_shell  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MP_COMPETING = {"fewer-element phases only", "same-order neighbour"}  # the 6,869 oxides decided against other compounds


def main():
    d = decisions()
    op = d[d.regime == "open-shell"].copy()
    q, margin = op.q_audited.values, op.dft_meV_atom.abs().values
    fit_all = sigma_ml(q, margin)
    keep = (op.formula != "CeCrO3").values
    fit_wo = sigma_ml(q[keep], margin[keep])
    print(f"open-shell decisions {len(op)}: ML sigma {fit_all[0]:.1f} ({fit_all[1]:.1f} to {fit_all[2]:.1f}); "
          f"without CeCrO3 {fit_wo[0]:.1f} ({fit_wo[1]:.1f} to {fit_wo[2]:.1f})")

    chen = pd.read_csv(RES / "precision_chen.csv")
    alab = pd.read_csv(RES / "precision_alab.csv")
    alab = alab[~alab.database_inconsistency]  # the two targets beyond 200 meV/atom, as in precision_analysis.py
    mp = pd.read_csv(RES / "precision_mp_observed.csv")
    mp = mp[mp.oxide & mp.competitor.isin(MP_COMPETING)]
    mp_open = mp.formula.map(open_shell).values

    def sig(is_open, s_open):
        return np.where(is_open, s_open, SIGMA_CLOSED)

    rows = []
    for label, s_open in (("one sigma", SIGMA_CLOSED), ("fit without CeCrO3", round(fit_wo[0], 1)),
                          ("adopted", SIGMA_OPEN), ("fit, all 21", round(fit_all[0], 1))):
        p = margin / s_open
        pred = norm.cdf(p)
        lo = p <= 1
        row = dict(sigma_open=s_open, label=label, open_n=len(op), open_n_low=int(lo.sum()),
                   open_right_low=round(q[lo].mean(), 3), open_pred_low=round(pred[lo].mean(), 3),
                   open_n_high=int((~lo).sum()), open_right_high=round(q[~lo].mean(), 3) if (~lo).any() else np.nan,
                   open_pred_high=round(pred[~lo].mean(), 3) if (~lo).any() else np.nan,
                   open_brier=round(np.mean((pred - q) ** 2), 4))
        for name, df, col in (("chen_selection", chen, "selection_margin_meV"), ("chen_stability", chen, "stability_margin_meV"),
                              ("alab_stability", alab, "stability_margin_meV")):
            pc = df[col].abs() / sig(df.open_shell.astype(bool).values, s_open)
            row[f"{name}_n"] = len(df)
            row[f"{name}_resolved"] = int((pc > 1).sum())
            row[f"{name}_expected_right"] = round(norm.cdf(pc).sum(), 1)
        pm = mp.margin_meV.abs().values / sig(mp_open, s_open)
        row.update(mp_n=len(mp), mp_resolved_share=round((pm > 1).mean(), 3),
                   mp_expected_right_share=round(norm.cdf(pm).mean(), 3))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "open_shell_sensitivity.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
