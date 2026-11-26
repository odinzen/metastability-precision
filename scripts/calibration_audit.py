"""How the calibration moves with the measured truth: compiled labels against labels audited with primary calorimetry
(truth_audit.py), with uncertain labels dropped, as soft labels, and with phase-diagram labels.

For each labelling and each set (compiled or polymorph; closed- or open-shell) it reports the shares right at p <= 1
and p > 1 against the prediction Phi(p), the maximum-likelihood sigma with its likelihood interval, proper scores
(Brier, log) against the base rate, the Cox calibration slope and intercept (logistic regression of the outcome on the
logit of the prediction) and the Spiegelhalter z. A soft label q is the probability that the measured sign agrees
with the computed one; the probability of the observed outcome is then q Phi(p) + (1 - q)(1 - Phi(p)).
Writes results/calibration_audit.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigma_model import sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GRID = np.arange(2.0, 400.0, 0.1)


def labels():
    """Decision table with the computed margin and one agreement probability q per labelling."""
    d = pd.read_csv(RES / "precision_calibration_decisions.csv")
    a = pd.read_csv(RES / "truth_audit.csv").set_index("compound")
    d["sigma"] = d.formula.map(sigma_for)
    d["p"] = d.dft_meV_atom.abs() / d.sigma
    agree = lambda truth: (np.sign(d.dft_meV_atom) == np.sign(truth)).astype(float)
    comp = d.source == "compiled"
    truth = d.truth_sign.where(~comp, d.compound.map(a.truth_meV_atom))
    d["q_compiled"] = agree(d.truth_sign)
    d["q_audited"] = agree(truth)
    ps = d.compound.map(a.p_stable)
    soft = np.where(d.dft_meV_atom < 0, ps, 1 - ps)
    d["q_soft"] = np.where(comp, soft, d.q_audited)
    certain = (truth.abs() >= 2 * d.compound.map(a.truth_u)) | ~comp
    d["certain"] = certain
    pdst = d.compound.map(a.phase_diagram_stable).fillna(False).astype(bool)
    d["q_phase_diagram"] = np.where(comp & pdst, (d.dft_meV_atom < 0).astype(float), d.q_audited)
    return d


def loglik(q, pred):
    return np.log(np.clip(q * pred + (1 - q) * (1 - pred), 1e-12, 1))


def ml_sigma(g, qcol):
    """Profile of the log-likelihood over a common sigma for the set; 95% interval from a drop of 1.92."""
    ll = np.array([loglik(g[qcol].values, norm.cdf(g.dft_meV_atom.abs().values / s)).sum() for s in GRID])
    best = GRID[ll.argmax()]
    inside = GRID[ll >= ll.max() - 1.92]
    hi = inside.max() if inside.max() < GRID[-1] - 1 else np.inf
    return best, inside.min(), hi


def cox(q, pred):
    """Calibration intercept and slope: logistic regression of the (soft) outcome on logit(pred)."""
    x = np.log(np.clip(pred, 1e-6, 1 - 1e-6) / np.clip(1 - pred, 1e-6, 1))
    nll = lambda b: -np.sum(q * np.log(1 / (1 + np.exp(-(b[0] + b[1] * x)))) + (1 - q) * np.log(1 - 1 / (1 + np.exp(-(b[0] + b[1] * x)))))
    r = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return r.x


def score(g, qcol, label, name):
    pred = norm.cdf(g.p.values)
    q = g[qcol].values
    lo, hi = g.p <= 1, g.p > 1
    s, s_lo, s_hi = ml_sigma(g, qcol)
    base = q.mean()
    b0, b1 = cox(q, pred)
    spieg = np.sum((q - pred) * (1 - 2 * pred)) / np.sqrt(np.sum((1 - 2 * pred) ** 2 * pred * (1 - pred)))
    return dict(labels=label, set=name, n=len(g), right=round(q.sum(), 1),
                n_low=int(lo.sum()), right_low=round(q[lo].sum(), 1), predicted_low=round(pred[lo].sum(), 1),
                n_high=int(hi.sum()), right_high=round(q[hi].sum(), 1), predicted_high=round(pred[hi].sum(), 1),
                sigma_ml=round(s, 1), sigma_lo=round(s_lo, 1), sigma_hi=round(s_hi, 1),
                brier=round(np.mean((pred - q) ** 2), 4), brier_base=round(np.mean((base - q) ** 2), 4),
                log_score=round(-loglik(q, pred).mean(), 4), log_base=round(-loglik(q, np.full_like(pred, base)).mean(), 4),
                cox_intercept=round(b0, 2), cox_slope=round(b1, 2), spiegelhalter_z=round(spieg, 2))


def main():
    d = labels()
    rows = []
    variants = [("compiled", "q_compiled", d), ("audited", "q_audited", d), ("audited, |truth| >= 2u", "q_audited", d[d.certain]),
                ("audited, soft", "q_soft", d), ("audited + phase diagram", "q_phase_diagram", d)]
    for label, qcol, x in variants:
        comp, poly = x.source == "compiled", x.source == "polymorph"
        for name, g in (("closed-shell, all", x[~x.open_shell]), ("closed-shell, compiled", x[comp & ~x.open_shell]),
                        ("closed-shell, polymorph", x[poly & ~x.open_shell]), ("open-shell, all", x[x.open_shell]),
                        ("open-shell, compiled", x[comp & x.open_shell]), ("all", x)):
            rows.append(score(g, qcol, label, name))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "calibration_audit.csv", index=False)
    cols = ["labels", "set", "n", "right_low", "n_low", "predicted_low", "right_high", "n_high", "predicted_high",
            "sigma_ml", "sigma_lo", "sigma_hi", "brier", "brier_base", "cox_slope", "cox_intercept", "spiegelhalter_z"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
