"""Is the calibration test passed for a trivial reason? Three checks on existing data, and planted failures.

A. The audited decisions (calibration_regimes.decisions()). Phi(p) scored as the probability that the computed sign is
   right, against a constant forecaster at the base rate (in sample, and fitted leaving each chemical system out), with
   a permutation test that shuffles p across decisions. The outcome is almost always "right" because compilations
   list compounds that exist, so a forecaster can look good by saying 0.92 every time; the permutation test asks
   whether p puts the errors where they are.
B. The outcome a discovery campaign needs: is the compound stable? 92 of the 98 compound decisions are measured
   stable, so "every compiled compound is stable" is a strong prior. The DFT margin is scored as a forecast of
   stability, Phi(-margin / sigma), against that prior, and a logistic model with the prior as intercept asks whether
   the margin adds information to it (likelihood ratio, and leave-one-system-out log score).
C. A two-sided test. At the composition of every compiled compound, every assemblage of the other compiled phases of
   its system that balances to that composition is found, and every pair of assemblages is a reaction whose measured
   sign is fixed by the compiled enthalpies alone. Neither side is the compound itself, so neither sign is favoured by
   how the compilation was built. The computed sign uses the Materials Project energies of the assigned material ids.
   Scored like A, with the label uncertainty propagated from the compiled uncertainties (missing ones take the
   median), against a chemist's heuristic: the assemblage holding more of its atoms in ternary or higher compounds is
   the lower one.
D. Planted failures: margins replaced by noise of the same size, and sigma set to 5 meV/atom. The tests must flag both.

Intervals come from bootstraps that resample chemical systems. Writes results/calibration_baselines.csv and
results/calibration_two_sided.csv.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_compiled import compiled, reactions  # noqa: E402
from calibration_regimes import cox, decisions, spiegelhalter  # noqa: E402
from sigma_model import SIGMA_CLOSED, SIGMA_OPEN, open_shell  # noqa: E402
from sigma_by_reaction_type import mp_energies  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT, N_PERM = 2000, 20000
rng = np.random.default_rng(20261028)
EPS = 1e-6


def brier(q, f):
    return float(np.mean((q - f) ** 2))


def logloss(q, f):
    f = np.clip(f, EPS, 1 - EPS)
    return float(-np.mean(q * np.log(f) + (1 - q) * np.log(1 - f)))


def auc(q, s):
    pos, neg = s[q == 1], s[q == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :]))
                 / (len(pos) * len(neg)))


def loso_constant(q, groups):
    """Base rate fitted without the decision's own chemical system."""
    out = np.empty(len(q))
    for g in np.unique(groups):
        m = groups == g
        out[m] = q[~m].mean() if (~m).any() else q.mean()
    return np.clip(out, EPS, 1 - EPS)


def logistic(y, x, offset=None):
    """Logistic regression of y on x (one slope and an intercept); returns the coefficients."""
    off = np.zeros(len(y)) if offset is None else offset

    def nll(b):
        z = np.clip(off + b[0] + b[1] * x, -30, 30)
        return -np.sum(y * -np.log1p(np.exp(-z)) + (1 - y) * -np.log1p(np.exp(z)))
    r = minimize(nll, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=5000))
    return r.x, r.fun


def loso_logistic(y, x, groups):
    out = np.empty(len(y))
    for g in np.unique(groups):
        m = groups == g
        b, _ = logistic(y[~m], x[~m])
        out[m] = 1 / (1 + np.exp(-np.clip(b[0] + b[1] * x[m], -30, 30)))
    return out


def cluster_boot(df, fn, n=N_BOOT):
    systems = df.system.unique()
    by = {s: df[df.system == s] for s in systems}
    vals = []
    for _ in range(n):
        pick = rng.choice(systems, len(systems))
        vals.append(fn(pd.concat([by[s] for s in pick], ignore_index=True)))
    return np.nanpercentile(vals, [2.5, 97.5])


def score_rows(name, q, f, groups, p=None):
    """Proper scores of a forecast of q, with the in-sample and held-out constant forecasters beside it."""
    base_in = np.full(len(q), q.mean())
    base_cv = loso_constant(q, groups)
    row = dict(test=name, n=len(q), events=int((q == 0).sum()), base_rate=q.mean(),
               brier=brier(q, f), brier_const_in=brier(q, base_in), brier_const_cv=brier(q, base_cv),
               log=logloss(q, f), log_const_in=logloss(q, base_in), log_const_cv=logloss(q, base_cv),
               cox_slope=cox(q, np.clip(f, EPS, 1 - EPS))[1], spiegelhalter_z=spiegelhalter(q, np.clip(f, EPS, 1 - EPS)))
    row["brier_skill_cv"] = 1 - row["brier"] / row["brier_const_cv"]
    row["log_skill_cv"] = 1 - row["log"] / row["log_const_cv"]
    if p is not None:
        row["auc"] = auc(q, p)
        # Permutation: how often does shuffling p across decisions score as well?
        hits_b = hits_a = 0
        for _ in range(N_PERM):
            ps = rng.permutation(p)
            fs = norm.cdf(ps) if f is not None else None
            hits_b += brier(q, fs) <= row["brier"]
            hits_a += auc(q, ps) >= row["auc"]
        row["perm_P_brier"] = (hits_b + 1) / (N_PERM + 1)
        row["perm_P_auc"] = (hits_a + 1) / (N_PERM + 1)
    return row


def part_a(d):
    rows = []
    for reg in ("compound", "open-shell"):
        g = d[d.regime == reg]
        q, p = g.q_audited.to_numpy(), g.p.to_numpy()
        row = score_rows(f"A: computed sign right, {reg}", q, norm.cdf(p), g.system.to_numpy(), p)
        def skill(x):
            ref = brier(x.q_audited.to_numpy(), np.full(len(x), x.q_audited.mean()))
            return np.nan if ref == 0 else 1 - brier(x.q_audited.to_numpy(), norm.cdf(x.p.to_numpy())) / ref
        lo, hi = cluster_boot(g, skill)
        row["brier_skill_in_ci"] = f"{lo:.2f} to {hi:.2f}"
        rows.append(row)
    return rows


def part_b(d):
    a = pd.read_csv(RES / "truth_audit.csv").set_index("compound")
    rows = []
    for reg in ("compound", "open-shell"):
        g = d[d.regime == reg].copy()
        g["truth"] = g.compound.map(a.truth_meV_atom)
        y = (g.truth < 0).astype(float).to_numpy()          # 1 = measured stable
        x = (-g.dft_meV_atom / g.sigma).to_numpy()           # signed precision, positive = computed stable
        groups = g.system.to_numpy()
        f_dft = norm.cdf(x)
        f_cv = loso_logistic(y, x, groups)
        prior = y.mean()
        off = np.full(len(y), np.log(prior / (1 - prior)))
        (b0, b1), nll1 = logistic(y, x, offset=off)
        nll0 = -np.sum(y * np.log(prior) + (1 - y) * np.log(1 - prior))
        # The intercept is free in the full model and fixed in the null, so the LR test has one degree for the slope
        # once the intercept is profiled; fit the null with a free intercept for a fair comparison.
        (c0, _), nll0b = logistic(y, np.zeros(len(y)))
        lr = 2 * (nll0b - nll1)
        for name, f in (("DFT sign as Phi(-margin/sigma)", f_dft), ("prior + margin, held-out logistic", f_cv)):
            row = score_rows(f"B: measured stable, {reg}, {name}", y, f, groups)
            row.update(events=int((y == 0).sum()), auc=auc(y, x), lr_stat=lr, lr_P=chi2.sf(lr, 1),
                       slope_on_margin=b1)
            rows.append(row)
    return rows


def assemblages(target_vec, cands, vecs, k):
    """Every set of up to k candidate phases whose nonnegative combination is the target composition."""
    found = []
    for size in range(1, k + 1):
        for combo in itertools.combinations(cands, size):
            M = np.array([vecs[c] for c in combo]).T
            coef, res = nnls(M, target_vec)
            if res < 1e-8 and np.all(coef > 1e-9):
                found.append(dict(zip(combo, coef)))
    return found


def part_c():
    table = compiled().set_index("formula")
    rx = [r for r in reactions(compiled()) if isinstance(r["mpid"], str)]
    phases = {}
    for r in rx:
        phases[r["compound"]] = (r["mpid"], {f: v[0] for f, v in r["oxides"].items()})
        for f, (_, mid, _) in r["oxides"].items():
            phases.setdefault(f, (mid, {f: 1.0}))
    ids = sorted({m for m, _ in phases.values() if isinstance(m, str)})
    e_dft = mp_energies(ids)
    from emmet.core.mpid import AlphaID
    from pymatgen.core import Composition

    def natoms(f):
        return Composition(f).num_atoms

    def dft_fu(f):
        mid = phases[f][0]
        e = e_dft.get(int(AlphaID(mid))) if isinstance(mid, str) else None
        return None if e is None else e * natoms(f)

    u_med = float(table.uncertainty.median())
    meas_fu = {f: float(table.loc[f, "expt_form_e"]) * natoms(f) for f in phases}
    u_fu = {f: (float(table.loc[f, "uncertainty"]) if pd.notna(table.loc[f, "uncertainty"]) else u_med) * natoms(f)
            for f in phases}
    rows, seen = [], set()
    for r in rx:
        x = r["compound"]
        binaries = sorted(r["oxides"])
        cations = {e for e in Composition(x).elements if e.symbol != "O"}
        cands = [f for f, (_, ox) in phases.items() if f != x and set(ox) <= set(binaries)
                 and {e for e in Composition(f).elements if e.symbol != "O"} <= cations and dft_fu(f) is not None]
        vecs = {f: np.array([phases[f][1].get(b, 0.0) for b in binaries]) for f in cands}
        target = np.array([phases[x][1][b] for b in binaries])
        found = assemblages(target, cands, vecs, len(binaries))
        system = "-".join(sorted(str(e) for e in cations))
        n_at = natoms(x)
        for A, B in itertools.combinations(found, 2):
            key = (system, frozenset(A), frozenset(B))
            if key in seen or (system, frozenset(B), frozenset(A)) in seen:
                continue
            seen.add(key)
            # reaction A -> B, per atom of the target composition
            d_meas = (sum(c * meas_fu[f] for f, c in B.items()) - sum(c * meas_fu[f] for f, c in A.items())) / n_at
            d_dft = (sum(c * dft_fu(f) for f, c in B.items()) - sum(c * dft_fu(f) for f, c in A.items())) / n_at
            u = np.sqrt(sum((c * u_fu[f]) ** 2 for side in (A, B) for f, c in side.items())) / n_at
            tern = lambda S: sum(c * natoms(f) for f, c in S.items() if len(Composition(f).elements) >= 3)
            open_any = any(open_shell(f) for side in (A, B) for f in side)
            rows.append(dict(system=system, point=x, A="+".join(f"{c:.3g} {f}" for f, c in A.items()),
                             B="+".join(f"{c:.3g} {f}" for f, c in B.items()),
                             measured_meV=1000 * d_meas, dft_meV=1000 * d_dft, u_meV=1000 * u,
                             ternary_atoms_A=tern(A), ternary_atoms_B=tern(B), open_shell=open_any))
    c = pd.DataFrame(rows)
    c = c[c.measured_meV.abs() > 0]
    c["sigma"] = np.where(c.open_shell, SIGMA_OPEN, SIGMA_CLOSED)
    c["p"] = c.dft_meV.abs() / c.sigma
    c["right"] = (np.sign(c.dft_meV) == np.sign(c.measured_meV)).astype(float)
    # Soft label: probability the true sign matches the computed one, given the compiled uncertainties.
    pm = norm.cdf(-c.measured_meV / c.u_meV)          # probability the measured reaction is negative
    c["right_soft"] = np.where(c.dft_meV < 0, pm, 1 - pm)
    # Heuristic: the side holding more atoms in ternary or higher compounds is lower (reaction toward it negative).
    h = np.sign(c.ternary_atoms_B - c.ternary_atoms_A)          # +1: B richer, so predict A -> B negative
    c["heuristic_right"] = np.where(h == 0, np.nan, (np.sign(-c.measured_meV) == h).astype(float))
    c["certain"] = c.measured_meV.abs() >= 2 * c.u_meV
    c.to_csv(RES / "calibration_two_sided.csv", index=False)
    return c


def part_c_rows(c):
    rows = []
    for label, g in (("all", c), ("measured sign beyond 2u", c[c.certain]), ("closed shell", c[~c.open_shell]),
                     ("closed shell, beyond 2u", c[~c.open_shell & c.certain])):
        q, p = g.right.to_numpy(), g.p.to_numpy()
        row = score_rows(f"C: two-sided reactions, {label}", q, norm.cdf(p), g.system.to_numpy(), p)
        lo_hi = cluster_boot(g, lambda x: x.right.mean(), n=1000)
        hh = g.heuristic_right.dropna()
        low, high = g[g.p <= 1], g[g.p > 1]
        row.update(systems=g.system.nunique(), measured_negative_share=float((g.measured_meV < 0).mean()),
                   dft_right=g.right.mean(), dft_right_ci=f"{lo_hi[0]:.2f} to {lo_hi[1]:.2f}",
                   heuristic_right=hh.mean() if len(hh) else np.nan, heuristic_n=len(hh),
                   right_soft=g.right_soft.mean(),
                   n_low=len(low), right_low=low.right.mean(), pred_low=norm.cdf(low.p).mean(),
                   n_high=len(high), right_high=high.right.mean(), pred_high=norm.cdf(high.p).mean())
        rows.append(row)
    return rows


def part_d(d, c):
    """The same tests on forecasts known to be wrong: noise margins, and sigma far too small."""
    rows = []
    g = d[d.regime == "compound"]
    for label, data, qcol, mcol, sig in (("A compound", g, "q_audited", "dft_meV_atom", "sigma"),
                                         ("C two-sided", c, "right", "dft_meV", "sigma")):
        q = data[qcol].to_numpy()
        m = data[mcol].abs().to_numpy()
        s = data[sig].to_numpy()
        noise_p = rng.permutation(m) / s                       # margins unrelated to which calls are right
        rows.append(score_rows(f"D: planted, {label}, margins shuffled", q, norm.cdf(noise_p),
                               data.system.to_numpy(), noise_p))
        tight = m / 5.0
        rows.append(score_rows(f"D: planted, {label}, sigma 5 meV/atom", q, norm.cdf(tight), data.system.to_numpy(),
                               tight))
    return rows


def main():
    d = decisions()
    c = part_c()
    rows = part_a(d) + part_b(d) + part_c_rows(c) + part_d(d, c)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "calibration_baselines.csv", index=False)
    cols = ["test", "n", "events", "base_rate", "brier", "brier_const_cv", "brier_skill_cv", "log", "log_const_cv",
            "log_skill_cv", "auc", "perm_P_auc", "perm_P_brier", "cox_slope", "spiegelhalter_z"]
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 60):
        print(out[cols].round(3).to_string(index=False))
        extra = [k for k in ("lr_stat", "lr_P", "slope_on_margin", "brier_skill_in_ci", "systems",
                             "measured_negative_share", "dft_right", "dft_right_ci", "heuristic_right", "heuristic_n",
                             "right_soft", "n_low", "right_low", "pred_low", "n_high", "right_high", "pred_high")
                 if k in out]
        print(out[["test"] + extra].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
