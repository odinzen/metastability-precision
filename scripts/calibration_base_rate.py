"""Is the compound regime conservative, or is that the base rate? Phi(|computed margin| / sigma) is the probability
that the computed sign is right only if every true margin is equally likely beforehand. Among known compounds most
true margins are negative (stable), so a computed-stable call is right more often than Phi says whatever sigma is.

The empirical-Bayes probability replaces the flat prior with the measured margins of the other chemical systems:
P(right | c) = sum_j phi((c - t_j) / sigma) [sign t_j = sign c] / sum_j phi((c - t_j) / sigma), over the compiled
decisions j outside the system of the decision scored. Both forecasts are scored on the audited outcomes of the
compound and open-shell regimes with the statistics of calibration_regimes.py, and sigma is refitted under each.
Writes results/calibration_base_rate.csv and results/calibration_base_rate_observed.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_regimes import GRID, N_BOOT, cox, decisions, spiegelhalter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
rng = np.random.default_rng(20261103)


def compiled():
    d = decisions()
    a = pd.read_csv(RES / "truth_audit.csv").set_index("compound")
    d = d[d.regime != "polymorph"].copy()
    d["truth"] = d.compound.map(a.truth_meV_atom)
    d["q"] = d.q_audited
    return d.dropna(subset=["truth"]).reset_index(drop=True)


def eb_prob(d, sigma):
    """Probability the computed sign is right, with the leave-system-out measured margins as prior."""
    sysv = d.system.to_numpy(dtype=object)
    return prior_prob(d.dft_meV_atom.values, sigma, d.truth.values, sysv[:, None] != sysv[None, :])


def prior_prob(c, sigma, t, keep=None):
    """P(sign of the true margin = sign of c) for computed margins c, prior = the measured margins t (log space, so a
    margin far from every prior point does not underflow)."""
    s = np.broadcast_to(sigma, c.shape) if np.ndim(sigma) == 0 else np.asarray(sigma)
    lw = norm.logpdf((c[:, None] - t[None, :]) / s[:, None])
    if keep is not None:
        lw = np.where(keep, lw, -np.inf)
    same = np.sign(t)[None, :] == np.sign(c)[:, None]
    return np.exp(logsumexp(np.where(same, lw, -np.inf), axis=1) - logsumexp(lw, axis=1))


def observed_oxides(t):
    """Expected right share of the observed Materials Project oxides that compete with other compounds (the 78% of
    the text), under the flat prior and under the measured-margin prior, across the fitted sigma range."""
    d = pd.read_csv(RES / "precision_mp_observed.csv")
    o = d[d.oxide & d.competitor.isin(["fewer-element phases only", "same-order neighbour"])]
    m, s, st = o.margin_meV.values, o.sigma.values, o.margin_meV.values < 0
    rows = []
    for label, k in (("sigma 22.5 / 34", 1.0), ("empirical-Bayes fit 16.4", 16.4 / 22.5),
                     ("fit lower 9.4", 9.4 / 22.5), ("fit upper 42.6", 42.6 / 22.5)):
        for prior, P in (("flat", norm.cdf(np.abs(m) / (s * k))), ("measured margins", prior_prob(m, s * k, t))):
            rows.append(dict(sigma=label, prior=prior, n=len(m), expected_right=P.mean(),
                             computed_stable=P[st].mean(), computed_unstable=P[~st].mean(), n_unstable=int((~st).sum())))
    return pd.DataFrame(rows)


def score(q, pred):
    b0, b1 = cox(q, pred)
    low = pred < 0.84  # p <= 1 under the flat prior
    return dict(n=len(q), right=int(q.sum()), mean_pred=pred.mean(), cox_intercept=b0, cox_slope=b1,
                spiegelhalter_z=spiegelhalter(q, pred), brier=np.mean((pred - q) ** 2),
                log_score=-np.mean(np.log(np.clip(np.where(q > 0.5, pred, 1 - pred), 1e-12, 1))),
                n_low=int(low.sum()), share_low=q[low].mean(), pred_low=pred[low].mean())


def sigma_fit(d, idx, scale_open=34 / 22.5):
    """Maximum-likelihood sigma under the empirical-Bayes forecast, open-shell decisions at the same ratio."""
    q = d.q.values[idx]
    ll = []
    for s in GRID[GRID <= 120]:
        sig = np.where(d.regime.to_numpy(dtype=object) == "open-shell", s * scale_open, s)
        P = eb_prob(d, sig)[idx]
        ll.append(np.log(np.clip(np.where(q > 0.5, P, 1 - P), 1e-12, 1)).sum())
    ll = np.array(ll)
    grid = GRID[GRID <= 120]
    inside = grid[ll >= ll.max() - 1.92]
    return grid[ll.argmax()], inside.min(), inside.max()


def main():
    d = compiled()
    d["eb"] = eb_prob(d, d.sigma.values)
    rows = []
    for reg in ("compound", "open-shell"):
        g = d[d.regime == reg]
        idx = np.flatnonzero(d.regime == reg)
        for name, col in (("flat prior, Phi(p)", "pred"), ("empirical-Bayes prior", "eb")):
            s = score(g.q.values, g[col].values)
            s.update(regime=reg, forecast=name, systems=g.system.nunique())
            if col == "eb" and reg == "compound":
                s["sigma_ml"], s["sigma_ml_lo"], s["sigma_ml_hi"] = sigma_fit(d, idx)
            rows.append(s)
        # cluster bootstrap of the two forecasts' Cox slopes and Brier difference
        systems = g.system.unique()
        draws = []
        for _ in range(N_BOOT):
            b = pd.concat([g[g.system == x] for x in rng.choice(systems, len(systems))])
            if b.q.nunique() < 2:
                continue
            draws.append(dict(slope_flat=cox(b.q.values, b.pred.values)[1], slope_eb=cox(b.q.values, b.eb.values)[1],
                              brier_gain=np.mean((b.pred - b.q) ** 2) - np.mean((b.eb - b.q) ** 2)))
        bt = pd.DataFrame(draws)
        for r in rows[-2:]:
            key = "slope_flat" if r["forecast"].startswith("flat") else "slope_eb"
            r["cox_slope_ci"] = f"{bt[key].quantile(0.025):.2f} to {bt[key].quantile(0.975):.2f}"
        rows[-1]["brier_gain_ci"] = f"{bt.brier_gain.quantile(0.025):.4f} to {bt.brier_gain.quantile(0.975):.4f}"
    out = pd.DataFrame(rows)
    out.to_csv(RES / "calibration_base_rate.csv", index=False)
    obs = observed_oxides(d.truth.values)
    obs.to_csv(RES / "calibration_base_rate_observed.csv", index=False)

    g = d[d.regime == "compound"]
    stable_c, stable_t = g.dft_meV_atom < 0, g.truth < 0
    print("compound regime, computed (rows) against measured (columns) stability:")
    print(pd.crosstab(np.where(stable_c, "computed stable", "computed unstable"),
                      np.where(stable_t, "measured stable", "measured unstable")))
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
        print(f"prior: {len(d)} measured margins, {(d.truth < 0).mean():.3f} stable")
        print(obs.to_string(index=False))


if __name__ == "__main__":
    main()
