"""What the 45 pre-registered decisions can test. The deposited forecast (the family-aware error model, model B) is
scored by the Brier score of its probability that measurement confirms the computed sign, against three baselines
fixed now: the flat-prior forecast Phi(|margin| / sigma) at the calorimetric sigma (model A), the computed sign always
holding (probability 1), and no information (0.5). Outcomes are simulated from each forecast in turn to give the
chance that the deposited forecast scores better when it is the true one, and the chance that it does when a
baseline is. Writes results/preregistration_power.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_model import sigma_for  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
N_SIM = 20000
rng = np.random.default_rng(20261103)


def forecasts():
    d = pd.read_csv(RES / "preregistered_decisions.csv", comment="#")
    target = d.decision.str.split().str[-1]
    sigma = target.map(sigma_for)
    return pd.DataFrame({"deposited (model B)": d.p_computed_sign_confirmed.values,
                         "flat prior (model A)": norm.cdf(d.computed_margin_meV_atom.abs().values / sigma.values),
                         "computed sign holds": np.ones(len(d)), "no information": np.full(len(d), 0.5)})


def main():
    f = forecasts()
    rows = []
    for truth in ("deposited (model B)", "flat prior (model A)"):
        y = (rng.random((N_SIM, len(f))) < f[truth].values[None, :]).astype(float)
        brier = {k: ((v.values[None, :] - y) ** 2).mean(1) for k, v in f.items()}
        for alt in f.columns:
            if alt == "deposited (model B)":
                continue
            gap = brier[alt] - brier["deposited (model B)"]
            rows.append(dict(outcomes_drawn_from=truth, baseline=alt, n=len(f),
                             expected_brier_deposited=brier["deposited (model B)"].mean(),
                             expected_brier_baseline=brier[alt].mean(), share_deposited_better=(gap > 0).mean(),
                             gap_lo95=np.percentile(gap, 2.5), gap_hi95=np.percentile(gap, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "preregistration_power.csv", index=False)
    print(f"{len(f)} decisions; expected confirmed under the deposited forecast {f.iloc[:, 0].sum():.1f}")
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
