"""Calibration of metastability precision by regime, with audited outcomes and cluster-bootstrap intervals.

Regimes (sigma_model.py): compound decisions (sigma 22.5), open-shell decisions (sigma 34), polymorph decisions (no
Gaussian probability; the GGA sign and the r2SCAN sign are reported). Outcomes are the audited labels of
truth_audit.py: primary calorimetry where the literature compilation covers the reaction, the compiled value otherwise.

The primary endpoint, stated before the run, is the calibration of the compound regime: the Cox slope and intercept
(logistic regression of the outcome on logit Phi(p); 1 and 0 for a calibrated forecast) and the Spiegelhalter z.
Secondary: shares right at p <= 1 and p > 1, the Brier score with its Murphy decomposition (reliability, resolution,
uncertainty; ten equal-count bins) against the base rate, the log score, discrimination (AUC) and the
maximum-likelihood sigma. Decisions in one chemical system share compounds, so every interval comes from a bootstrap
that resamples chemical systems (2,000 draws). Sensitivities: soft labels, labels at least 2u from zero, and
phase-diagram labels. Writes results/calibration_regimes.csv and results/calibration_regimes_bins.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_model import open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GRID = np.arange(2.0, 400.0, 0.2)
N_BOOT = 2000
rng = np.random.default_rng(20261027)


def decisions():
    d = pd.read_csv(RES / "precision_calibration_decisions.csv")
    a = pd.read_csv(RES / "truth_audit.csv").set_index("compound")
    comp = d.source == "compiled"
    d["regime"] = np.where(~comp, "polymorph", np.where(d.formula.map(open_shell), "open-shell", "compound"))
    d["system"] = d.formula.map(lambda f: "-".join(sorted(str(e) for e in Composition(f).elements)))
    d["sigma"] = d.formula.map(sigma_for)
    d["p"] = d.dft_meV_atom.abs() / d.sigma
    d["pred"] = norm.cdf(d.p)
    truth = d.truth_sign.where(~comp, d.compound.map(a.truth_meV_atom))
    agree = lambda t: (np.sign(d.dft_meV_atom) == np.sign(t)).astype(float)
    d["q_audited"] = agree(truth)
    ps = d.compound.map(a.p_stable)
    d["q_soft"] = np.where(comp, np.where(d.dft_meV_atom < 0, ps, 1 - ps), d.q_audited)
    d["certain"] = (truth.abs() >= 2 * d.compound.map(a.truth_u)) | ~comp
    pdst = d.compound.map(a.phase_diagram_stable).fillna(False).astype(bool)
    d["q_phase_diagram"] = np.where(comp & pdst, (d.dft_meV_atom < 0).astype(float), d.q_audited)
    d["truth_basis"] = d.compound.map(a.truth_basis).fillna("polymorph")
    return d


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cox(q, pred):
    x = logit(pred)

    def nll(b):
        z = np.clip(b[0] + b[1] * x, -30, 30)
        return -np.sum(q * -np.log1p(np.exp(-z)) + (1 - q) * -np.log1p(np.exp(z)))
    return minimize(nll, [0.0, 1.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000)).x


def spiegelhalter(q, pred):
    w = 1 - 2 * pred
    return np.sum((q - pred) * w) / np.sqrt(np.sum(w ** 2 * pred * (1 - pred)))


def sigma_ml(q, margin):
    P = norm.cdf(margin[None, :] / GRID[:, None])
    ll = np.log(np.clip(q[None, :] * P + (1 - q[None, :]) * (1 - P), 1e-12, 1)).sum(axis=1)
    inside = GRID[ll >= ll.max() - 1.92]
    return GRID[ll.argmax()], inside.min(), (inside.max() if inside.max() < GRID[-1] - 1 else np.inf)


def murphy(q, pred, nbins=10):
    """Brier = reliability - resolution + uncertainty over equal-count bins of the forecast."""
    order = np.argsort(pred)
    bins = np.array_split(order, min(nbins, len(q)))
    base = q.mean()
    rel = sum(len(b) * (pred[b].mean() - q[b].mean()) ** 2 for b in bins) / len(q)
    res = sum(len(b) * (q[b].mean() - base) ** 2 for b in bins) / len(q)
    return rel, res, base * (1 - base)


def auc(q, p):
    r, w = p[q > 0.5], p[q <= 0.5]
    if len(r) == 0 or len(w) == 0:
        return np.nan
    return (np.sum(r[:, None] > w[None, :]) + 0.5 * np.sum(r[:, None] == w[None, :])) / (len(r) * len(w))


def stats(g, qcol):
    q, pred, p = g[qcol].values, g.pred.values, g.p.values
    lo, hi = p <= 1, p > 1
    b0, b1 = cox(q, pred)
    rel, res, unc = murphy(q, pred)
    return dict(n=len(g), share_low=q[lo].mean() if lo.any() else np.nan, pred_low=pred[lo].mean() if lo.any() else np.nan,
                share_high=q[hi].mean() if hi.any() else np.nan, pred_high=pred[hi].mean() if hi.any() else np.nan,
                n_low=int(lo.sum()), n_high=int(hi.sum()), right=q.sum(),
                brier=np.mean((pred - q) ** 2), brier_base=np.mean((q.mean() - q) ** 2), reliability=rel, resolution=res,
                uncertainty=unc, log_score=-np.mean(np.log(np.clip(np.where(q > 0.5, pred, 1 - pred), 1e-12, 1))),
                cox_intercept=b0, cox_slope=b1, spiegelhalter_z=spiegelhalter(q, pred), auc=auc(q, p))


def bootstrap(g, qcol):
    systems = g.system.unique()
    groups = {s: g[g.system == s] for s in systems}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(systems, len(systems), replace=True)
        b = pd.concat([groups[s] for s in pick], ignore_index=True)
        if b[qcol].nunique() < 2:
            continue
        s = stats(b, qcol)
        s["sigma_ml"] = sigma_ml(b[qcol].values, b.dft_meV_atom.abs().values)[0]
        draws.append(s)
    return pd.DataFrame(draws)


def main():
    d = decisions()
    rows = []
    variants = [("audited", "q_audited", d), ("soft", "q_soft", d), ("|truth| >= 2u", "q_audited", d[d.certain]),
                ("phase diagram", "q_phase_diagram", d)]
    for label, qcol, x in variants:
        for reg in ("compound", "open-shell", "polymorph"):
            g = x[x.regime == reg]
            s = stats(g, qcol)
            s.update(labels=label, regime=reg, systems=g.system.nunique())
            s["sigma_ml"], s["sigma_ml_lo"], s["sigma_ml_hi"] = sigma_ml(g[qcol].values, g.dft_meV_atom.abs().values)
            if label == "audited" and reg != "polymorph":
                bt = bootstrap(g, qcol)
                for k in ("share_low", "share_high", "cox_slope", "cox_intercept", "spiegelhalter_z", "brier", "auc", "sigma_ml"):
                    s[f"{k}_ci"] = f"{bt[k].quantile(0.025):.2f} to {bt[k].quantile(0.975):.2f}"
            rows.append(s)
    out = pd.DataFrame(rows)
    front = ["labels", "regime", "n", "systems", "n_low", "share_low", "pred_low", "n_high", "share_high", "pred_high",
             "cox_slope", "cox_intercept", "spiegelhalter_z", "brier", "brier_base", "reliability", "resolution", "log_score",
             "auc", "sigma_ml", "sigma_ml_lo", "sigma_ml_hi"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(RES / "calibration_regimes.csv", index=False)

    # Reliability curve of the compound regime, ten equal-count bins, with cluster-bootstrap intervals of each bin.
    g = d[d.regime == "compound"].sort_values("pred").reset_index(drop=True)
    g["bin"] = pd.qcut(g.pred.rank(method="first"), 10, labels=False)
    curve = g.groupby("bin").agg(n=("pred", "size"), pred=("pred", "mean"), observed=("q_audited", "mean"),
                                 p_mean=("p", "mean")).reset_index()
    curve.to_csv(RES / "calibration_regimes_bins.csv", index=False)

    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(out[front].to_string(index=False))
        print("\ncluster-bootstrap 95% intervals (audited labels):")
        print(out[out.labels == "audited"][["regime"] + [c for c in out.columns if c.endswith("_ci")]].to_string(index=False))
        print("\ncompound regime, reliability by equal-count bin:")
        print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
