"""Does the measurement ranking work? The method of measurement_value.py tested on decisions whose answers are known,
at rising difficulty, then on the real campaigns.

Test bed. For every compound of the compiled benchmark (benchmark_compiled.py), its decomposition into the
lowest-enthalpy measured combination of the other compiled phases of its system is a decision with a known answer,
as in the calibration. Energies are taken from the binary oxides, so the binary oxides carry no error (as the method
assumes, the database's corrections being fitted to them) and the error of a decision is sum_f w_f eps_f over its
phases of three or more elements, with eps_f = DFT minus measured enthalpy of formation from the oxides. CuAl2O4 and
Ca3Si2O7 are left out, because primary calorimetry shows their compiled values to be wrong (truth_audit.py).
"Measuring" a compound reveals eps_f, the decision's margin is updated by the conditional mean of its error under the
method's own covariance (measurement_value.py), and a decision counts as right when the updated sign matches the
measured one.

Strategies, each an order in which to measure: the method (greedy on expected right decisions, the code of the
paper); random orders; closest calls first (targets of the lowest-p decisions, then their products); most connected
first (largest total weight across decisions); and an oracle that sees the answers (an upper bound).

Rungs.
 1/10  synthetic errors drawn from the method's own model on the real decision structure; beat random.
 3/10  real errors; beat random.
 5/10  real errors, measurements with the compiled uncertainty as noise; beat the two heuristics, with intervals from
       a bootstrap over chemical systems.
 7/10  predicted against realised gain, and the method run with wrong parameters (rho 0, sigma halved or doubled).
 9/10  hostile: the out-of-sample shift of the blind hold-out (-14.6 meV/atom on every compound error) added to the
       errors, scored on the hard decisions (p <= 1) and on the open-shell decisions.
11/10  the real campaigns. (a) Blind prediction of the measured enthalpies of the campaign compounds the literature
       has measured, from the calorimetric trend and sigma, scored only on compounds that played no part in setting
       them. (b) Their real values entered into the 124 campaign decisions: realised against predicted shifts.
       (c) Pre-registered predictions for the 30 compounds the method would measure next, written with a SHA-256
       hash before any of them is measured.

Writes results/measurement_ladder.csv, results/measurement_ladder_curves.csv, results/measurement_blind_literature.csv,
results/measurement_real_entry.csv and results/preregistered_predictions.csv (+ .sha256).
"""
import gzip
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_value as mv  # noqa: E402
from benchmark_compiled import compiled  # noqa: E402
from sigma_model import open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
K_MAX = 40
N_RANDOM, N_SYNTH, N_BOOT, N_NOISE = 200, 100, 200, 20
HOLDOUT_SHIFT = -14.6           # mean residual of the blind hold-out (holdout_test.csv)
TREND_K, TREND_C = -0.143, 8.1  # calorimetric trend, error = k x measured + c (benchmark_dft_vs_measured.csv)
AUDIT_DROP = {"CuAl2O4", "Al2CuO4", "Ca3Si2O7"}
rng = np.random.default_rng(20261028)


# ---------------------------------------------------------------- test bed
def red(f):
    return Composition(f).reduced_formula


def test_bed():
    table = compiled().set_index("formula")
    bc = pd.read_csv(RES / "benchmark_compiled.csv")
    bc = bc[~bc.compound.map(red).isin({red(f) for f in AUDIT_DROP})]
    ox = {red(r.compound): dict(meas=r.measured_meV_atom, dft=r.dft_meV_atom,
                                binaries=[t.split(" ", 1)[1] for t in r.oxides.split("+")], formula=r.compound)
          for r in bc.itertuples()}
    u_med = float(table.uncertainty.median())
    unc = {red(f): 1000 * (float(table.loc[f, "uncertainty"]) if f in table.index and pd.notna(table.loc[f, "uncertainty"])
                           else u_med) for f in [v["formula"] for v in ox.values()]}
    decs = []
    for x, v in ox.items():
        binaries = set(v["binaries"])
        cations = {e for e in Composition(x).elements if e.symbol != "O"}
        cands = [f for f, w in ox.items() if f != x and set(w["binaries"]) <= binaries
                 and {e for e in Composition(f).elements if e.symbol != "O"} <= cations]
        # measured phase diagram from the compiled enthalpies of formation from the elements
        def e_el(f):
            return float(table.loc[ox[f]["formula"] if f in ox else f, "expt_form_e"]) * Composition(f).num_atoms
        entries = [PDEntry(Composition({el: 1}), 0.0) for el in list(cations) + [Element("O")]]
        entries += [PDEntry(Composition(b), float(table.loc[b, "expt_form_e"]) * Composition(b).num_atoms, name=b)
                    for b in binaries]
        entries += [PDEntry(Composition(f), e_el(f), name=f) for f in cands]
        pd_meas = PhaseDiagram(entries)
        target = PDEntry(Composition(x), e_el(x), name=x)
        decomp, e_hull = pd_meas.get_decomp_and_e_above_hull(target, allow_negative=True)
        if any(len(p.composition.elements) == 1 for p in decomp):
            continue
        weights = {x: 1.0}
        for p, frac in decomp.items():
            if len(p.composition.elements) >= 3:
                weights[red(p.name)] = weights.get(red(p.name), 0.0) - float(frac)
        m_meas = sum(w * ox[f]["meas"] for f, w in weights.items())
        m_dft = sum(w * ox[f]["dft"] for f, w in weights.items())
        assert abs(m_meas - 1000 * e_hull) < 0.5, (x, m_meas, 1000 * e_hull)  # from-oxides and from-elements agree
        decs.append(dict(label=x, target=x, weights=weights, margin_meas=m_meas, margin_dft=m_dft,
                         margin=abs(m_dft), sigma=sigma_for(x),
                         system="-".join(sorted(str(e) for e in cations))))
    eps = {f: v["dft"] - v["meas"] for f, v in ox.items()}
    return decs, eps, unc


# ---------------------------------------------------------------- the update the method implies
def cond_mean(dec, y, rho=mv.RHO, sigma_cal=mv.SIGMA_CAL, sigma=None):
    """Conditional mean of the decision's error given the measured eps (dict) of some of its phases."""
    names = list(dec["weights"])
    idx = [i for i, f in enumerate(names) if f in y]
    if not idx:
        return 0.0
    w = np.array([dec["weights"][f] for f in names])
    corr = rho + (1 - rho) * np.eye(len(names))
    sig = dec["sigma"] if sigma is None else sigma
    C = corr * sig ** 2 / (w @ corr @ w)
    cov_ey = (w @ C)[idx]
    S = C[np.ix_(idx, idx)] + sigma_cal ** 2 * np.eye(len(idx))
    return float(cov_ey @ np.linalg.solve(S, np.array([y[names[i]] for i in idx])))


def realised_right(decs, y, subset=None, **kw):
    n = 0
    for i, d in enumerate(decs):
        if subset is not None and i not in subset:
            continue
        corrected = d["margin_dft"] - cond_mean(d, y, **kw)
        n += np.sign(corrected) == np.sign(d["margin_meas"])
    return int(n)


# ---------------------------------------------------------------- strategies
def method_order(decs, k_max=K_MAX, rho=mv.RHO, sigma_scale=1.0):
    local = [dict(d, sigma=d["sigma"] * sigma_scale) for d in decs]
    phases = sorted({f for d in local for f in d["weights"]})
    touches = {c: [i for i, d in enumerate(local) if c in d["weights"]] for c in phases}
    chosen, order = set(), []
    current = [mv.expected(d, chosen, rho=rho)[0] for d in local]
    predicted = [sum(current)]
    for _ in range(min(k_max, len(phases))):
        gains = {c: sum(mv.expected(local[i], chosen | {c}, rho=rho)[0] - current[i] for i in touches[c])
                 for c in phases if c not in chosen}
        best = max(gains, key=gains.get)
        chosen.add(best)
        order.append(best)
        for i in touches[best]:
            current[i] = mv.expected(local[i], chosen, rho=rho)[0]
        predicted.append(sum(current))
    return order, predicted


def closest_call_order(decs):
    order = []
    for d in sorted(decs, key=lambda d: d["margin"] / d["sigma"]):
        for f, _ in sorted(d["weights"].items(), key=lambda kv: -abs(kv[1])):
            if f not in order:
                order.append(f)
    return order


def connected_order(decs):
    tot = {}
    for d in decs:
        for f, w in d["weights"].items():
            tot[f] = tot.get(f, 0.0) + abs(w)
    return sorted(tot, key=lambda f: -tot[f])


def oracle_order(decs, eps, k_max=K_MAX):
    phases = sorted({f for d in decs for f in d["weights"]})
    chosen = {}
    order = []
    for _ in range(min(k_max, len(phases))):
        best = max((c for c in phases if c not in chosen),
                   key=lambda c: realised_right(decs, {**chosen, c: eps[c]}))
        chosen[best] = eps[best]
        order.append(best)
    return order


def curve(decs, order, y_of, k_max=K_MAX, subset=None, **kw):
    out, y = [realised_right(decs, {}, subset, **kw)], {}
    for c in order[:k_max]:
        y[c] = y_of(c)
        out.append(realised_right(decs, y, subset, **kw))
    out += [out[-1]] * (k_max + 1 - len(out))
    return np.array(out)


def summary(c):
    """Mean decisions fixed over budgets 1..K_MAX, relative to measuring nothing."""
    return float(np.mean(c[1:] - c[0]))


# ---------------------------------------------------------------- rungs 1-9
def run_strategies(decs, eps, y_of=None, subset=None, eval_kw=None, select_decs=None):
    """Curves for every strategy; selection uses select_decs (the method's view), scoring the real answers."""
    eval_kw = eval_kw or {}
    sel = select_decs or decs
    y_of = y_of or (lambda c: eps[c])
    m_order, predicted = method_order(sel)
    curves = {"method": curve(decs, m_order, y_of, subset=subset, **eval_kw),
              "closest calls first": curve(decs, closest_call_order(sel), y_of, subset=subset, **eval_kw),
              "most connected first": curve(decs, connected_order(sel), y_of, subset=subset, **eval_kw),
              "oracle": curve(decs, oracle_order(decs, eps), lambda c: eps[c], subset=subset, **eval_kw)}
    phases = sorted({f for d in sel for f in d["weights"]})
    rand = np.array([curve(decs, list(rng.permutation(phases)), y_of, subset=subset, **eval_kw)
                     for _ in range(N_RANDOM)])
    curves["random (mean)"] = rand.mean(axis=0)
    return curves, predicted, m_order, rand


def ladder(decs, eps, unc):
    rows, curve_rows = [], []

    def record(rung, label, curves, rand=None, extra=None):
        for name, c in curves.items():
            r = dict(rung=rung, test=label, strategy=name, fixed_mean=summary(c), right_K0=int(c[0]),
                     right_K10=float(c[10]), right_K40=float(c[K_MAX]), decisions=len(decs))
            if rand is not None and name == "method":
                r["share_of_random_orders_beaten"] = float(np.mean([summary(x) < summary(c) for x in rand]))
            r.update(extra or {})
            rows.append(r)
            for k, v in enumerate(c):
                curve_rows.append(dict(rung=rung, test=label, strategy=name, K=k, right=float(v)))

    # 1/10: errors drawn from the model itself (per system shared part), same decisions and DFT margins
    synth = []
    phases = sorted({f for d in decs for f in d["weights"]})
    for _ in range(N_SYNTH):
        s = 20.0
        eta = {}
        e_syn = {}
        for f in phases:
            sysk = "-".join(sorted(str(e) for e in Composition(f).elements if e.symbol != "O"))
            eta.setdefault(sysk, rng.normal())
            e_syn[f] = s * (np.sqrt(mv.RHO) * eta[sysk] + np.sqrt(1 - mv.RHO) * rng.normal())
        world = [dict(d, margin_meas=d["margin_dft"] - sum(w * e_syn[f] for f, w in d["weights"].items())) for d in decs]
        m_order, _ = method_order(world)
        cm = curve(world, m_order, lambda c: e_syn[c])
        cr = np.mean([curve(world, list(rng.permutation(phases)), lambda c: e_syn[c]) for _ in range(20)], axis=0)
        synth.append((summary(cm), summary(cr)))
    synth = np.array(synth)
    rows.append(dict(rung="1/10", test="synthetic errors from the model", strategy="method", fixed_mean=synth[:, 0].mean(),
                     share_of_worlds_beating_random=float(np.mean(synth[:, 0] > synth[:, 1])), decisions=len(decs)))
    rows.append(dict(rung="1/10", test="synthetic errors from the model", strategy="random (mean)",
                     fixed_mean=synth[:, 1].mean(), decisions=len(decs)))

    # 3/10: real errors, exact measurements
    curves, predicted, m_order, rand = run_strategies(decs, eps)
    record("3/10", "real errors", {k: curves[k] for k in ("method", "random (mean)", "oracle")}, rand)
    # 5/10: real errors, noisy measurements, against the heuristics, bootstrap over systems
    noisy = []
    for _ in range(N_NOISE):
        draw = {c: eps[c] + rng.normal(0, unc.get(c, 15.0)) for c in eps}
        noisy.append({k: curve(decs, o, lambda c: draw[c]) for k, o in
                      (("method", m_order), ("closest calls first", closest_call_order(decs)),
                       ("most connected first", connected_order(decs)))})
    mean_noisy = {k: np.mean([n[k] for n in noisy], axis=0) for k in noisy[0]}
    record("5/10", "real errors, measured with the compiled uncertainty", mean_noisy)
    systems = sorted({d["system"] for d in decs})
    diffs = {h: [] for h in ("closest calls first", "most connected first", "random")}
    for _ in range(N_BOOT):
        pick = rng.choice(systems, len(systems))
        sub = [dict(d) for s in pick for d in decs if d["system"] == s]
        mo, _ = method_order(sub)
        sub_ph = sorted({f for d in sub for f in d["weights"]})
        base = summary(curve(sub, mo, lambda c: eps[c]))
        diffs["closest calls first"].append(base - summary(curve(sub, closest_call_order(sub), lambda c: eps[c])))
        diffs["most connected first"].append(base - summary(curve(sub, connected_order(sub), lambda c: eps[c])))
        diffs["random"].append(base - summary(curve(sub, list(rng.permutation(sub_ph)), lambda c: eps[c])))
    for h, v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(rung="5/10", test=f"method minus {h}, bootstrap over systems", strategy="difference",
                         fixed_mean=float(np.mean(v)), ci=f"{lo:.2f} to {hi:.2f}", decisions=len(decs)))
    # 7/10: predicted against realised, and wrong parameters
    real = curves["method"]
    for k in (0, 5, 10, 20, 40):
        rows.append(dict(rung="7/10", test=f"predicted against realised right, K = {k}", strategy="method",
                         predicted=float(predicted[min(k, len(predicted) - 1)]), realised=float(real[k]),
                         decisions=len(decs)))
    for label, kw in (("rho 0", dict(rho=0.0)), ("sigma halved", dict(sigma_scale=0.5)),
                      ("sigma doubled", dict(sigma_scale=2.0))):
        o, _ = method_order(decs, **kw)
        c = curve(decs, o, lambda c: eps[c])
        record("7/10", f"method with wrong parameters: {label}", {"method": c})
    # 9/10: hostile. The hold-out shift on every compound error; scored on hard and on open-shell decisions.
    shifted = {f: e + HOLDOUT_SHIFT for f, e in eps.items()}
    world = [dict(d, margin_dft=d["margin_meas"] + sum(w * shifted[f] for f, w in d["weights"].items())) for d in decs]
    for d in world:
        d["margin"] = abs(d["margin_dft"])
    hard = {i for i, d in enumerate(world) if d["margin"] / d["sigma"] <= 1}
    openi = {i for i, d in enumerate(world) if open_shell(d["target"])}
    for label, subset in (("shifted errors, all decisions", None), ("shifted errors, hard decisions (p <= 1)", hard),
                          ("shifted errors, open-shell decisions", openi)):
        c9, _, _, rand9 = run_strategies(world, shifted, subset=subset)
        n_sub = len(world) if subset is None else len(subset)
        record("9/10", label, c9, rand9, extra=dict(scored=n_sub))
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


# ---------------------------------------------------------------- rung 11
def benchmark_compounds():
    b = pd.read_csv(RES / "benchmark_dft_vs_measured.csv")
    return {red(c) for c in b.compound}


def campaign_signed():
    """The 124 campaign decisions of measurement_value.py with their signed margins."""
    decs = mv.decisions()
    chen = pd.read_csv(RES / "precision_chen.csv").set_index("target")
    alab = pd.read_csv(RES / "precision_alab.csv").set_index("target")
    for d in decs:
        ds, kind, t = d["label"].split(" ", 2)
        src = chen if ds == "Chen2024" else alab
        # selection_margin_meV holds the magnitude; the signed published inverse hull energy is negative when favoured
        col = "pub_invE_meV" if kind == "selection" else "stability_margin_meV"
        d["margin_signed"] = float(src.loc[t, col])
        assert abs(abs(d["margin_signed"]) - d["margin"]) < 1e-6
    return decs


def oxides_of(formula):
    """Binary oxides of a formula from its most likely oxidation states."""
    try:
        guess = Composition(formula).oxi_state_guesses(max_sites=-50)
    except ValueError:  # a large cell the guesser cannot reduce
        guess = Composition(formula).oxi_state_guesses()
    if not guess:
        return None
    out = {}
    for el, ox in guess[0].items():
        if el == "O":
            continue
        if abs(ox - round(ox)) > 1e-6 or ox <= 0:
            return None
        a, b = (2, int(round(ox))) if int(round(ox)) % 2 else (1, int(round(ox)) // 2)
        out[el] = Composition({el: a, "O": b}).reduced_formula
    return out


def dft_from_oxides(formula, chemsys):
    """DFT enthalpy of formation from the binary oxides (meV/atom), lowest-energy Materials Project entries."""
    entries = MontyDecoder().process_decoded(json.load(gzip.open(ROOT / "data" / "raw" / "mp" / f"{chemsys}.json.gz")))
    best = {}
    for e in entries:
        f = e.composition.reduced_formula
        if f not in best or e.energy_per_atom < best[f].energy_per_atom:
            best[f] = e
    oxs = oxides_of(formula)
    if not oxs or red(formula) not in best or any(o not in best for o in oxs.values()):
        return None
    comp = Composition(formula)
    e_c = best[red(formula)].energy_per_atom * comp.num_atoms
    total = 0.0
    for el, o in oxs.items():
        oc = Composition(o)
        n = comp[el] / oc[el]
        total += n * best[o].energy_per_atom * oc.num_atoms
    return 1000 * (e_c - total) / comp.num_atoms


def rung11(eps_compiled):
    rows = []
    bench = benchmark_compounds()
    decs = campaign_signed()
    phases = sorted({f for d in decs for f in d["weights"]})
    known = [f for f in phases if f in eps_compiled]
    meas = pd.read_csv(RES / "benchmark_compiled.csv").assign(f=lambda x: x.compound.map(red)).set_index("f")
    # (a) blind prediction of the measured values, from the calorimetric trend: measured = (DFT - c) / (1 + k)
    for f in known:
        s = sigma_for(f)
        dft, real = meas.loc[f, "dft_meV_atom"], meas.loc[f, "measured_meV_atom"]
        pred = (dft - TREND_C) / (1 + TREND_K)
        sd = s / (1 + TREND_K)
        rows.append(dict(compound=f, in_benchmark=f in bench, open_shell=open_shell(f), dft=dft, predicted=pred,
                         sd=sd, measured=real, z_trend=(real - pred) / sd, z_raw=(real - dft) / (s / (1 + TREND_K))))
    blind = pd.DataFrame(rows)
    blind.to_csv(RES / "measurement_blind_literature.csv", index=False)
    # (b) enter the real values into the campaign decisions
    y = {f: eps_compiled[f] for f in known}
    entry = []
    for d in decs:
        names = [f for f in d["weights"] if f in y]
        if not names:
            continue
        shift = cond_mean(d, y)
        p0, _ = mv.expected(d, set())
        p_hyp, _ = mv.expected(d, set(names))
        # predicted spread of the updated margin
        w = np.array(list(d["weights"].values()))
        corr = mv.RHO + (1 - mv.RHO) * np.eye(len(w))
        C = corr * d["sigma"] ** 2 / (w @ corr @ w)
        idx = [i for i, f in enumerate(d["weights"]) if f in y]
        cov_ey = (w @ C)[idx]
        S = C[np.ix_(idx, idx)] + mv.SIGMA_CAL ** 2 * np.eye(len(idx))
        v = d["sigma"] ** 2 - cov_ey @ np.linalg.solve(S, cov_ey)
        spread = np.sqrt(max(d["sigma"] ** 2 - v, 0))
        new = d["margin_signed"] - shift
        entry.append(dict(decision=d["label"], measured_phases="+".join(names), margin=d["margin_signed"],
                          shift=-shift, predicted_spread=spread, z=(-shift / spread) if spread > 0 else np.nan,
                          updated=new, flips=np.sign(new) != np.sign(d["margin_signed"]),
                          resolved_before=abs(d["margin_signed"]) > d["sigma"],
                          resolved_after=abs(new) > np.sqrt(max(v, 1e-9)),
                          p_right_before=p0, p_right_expected=p_hyp, p_right_after=norm.cdf(abs(new) / np.sqrt(max(v, 1e-9)))))
    entry = pd.DataFrame(entry)
    entry.to_csv(RES / "measurement_real_entry.csv", index=False)
    # (c) pre-registration: the 30 compounds the method measures next, after the literature values are in
    mvals = pd.read_csv(RES / "measurement_value.csv")
    picks = [m for m in mvals.measured if isinstance(m, str) and not m.startswith(("none", "enter"))]
    today = pd.read_csv(RES / "mp_targets_today.csv")
    chemsys_of = {}
    for d in decs:
        ds = d["label"].split(" ")[0]
        cs = today[(today.dataset == ds) & (today.target == d["target"])].chemsys
        if len(cs):
            for f in d["weights"]:
                chemsys_of.setdefault(f, cs.iloc[0])
    pre = []
    for rank, f in enumerate(picks, 1):
        dft = dft_from_oxides(f, chemsys_of[f]) if f in chemsys_of else None
        s = sigma_for(f)
        sd = s / (1 + TREND_K)
        pred = None if dft is None else (dft - TREND_C) / (1 + TREND_K)
        decisions_in = [d["label"] for d in decs if f in d["weights"]]
        pre.append(dict(rank=rank, compound=f, binary_oxides="+".join(sorted((oxides_of(f) or {}).values())),
                        open_shell=open_shell(f), dft_meV_atom=None if dft is None else round(dft, 1),
                        predicted_measured_meV_atom=None if pred is None else round(pred, 1),
                        interval_68=None if pred is None else f"{pred - sd:.1f} to {pred + sd:.1f}",
                        interval_95=None if pred is None else f"{pred - 1.96 * sd:.1f} to {pred + 1.96 * sd:.1f}",
                        decisions=";".join(decisions_in)))
    pre = pd.DataFrame(pre)
    path = RES / "preregistered_predictions.csv"
    header = (f"# Pre-registered predictions, written {date.today().isoformat()} before any of these compounds is "
              "measured. Enthalpy of formation from the binary oxides (meV/atom) predicted from the Materials Project "
              "GGA/GGA+U energy through the calorimetric trend, measured = (DFT - 8.1) / 0.857, with sigma 22.5 "
              "(34 open-shell) divided by 0.857. Families outside the calorimetric benchmark (borates, most "
              "phosphates) are an extrapolation of the trend.\n")
    path.write_text(header + pre.to_csv(index=False), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (RES / "preregistered_predictions.csv.sha256").write_text(f"{digest}  preregistered_predictions.csv\n", encoding="utf-8")
    return blind, entry, pre, digest


def coverage(z):
    z = np.asarray(z, float)
    return dict(n=len(z), within1=float(np.mean(np.abs(z) <= 1)), within2=float(np.mean(np.abs(z) <= 2)),
                mean_z=float(np.mean(z)), rms_z=float(np.sqrt(np.mean(z ** 2))))


def main():
    decs, eps, unc = test_bed()
    if len(sys.argv) > 1 and sys.argv[1] == "11":
        return report11(*rung11(eps))
    hard = sum(d["margin"] / d["sigma"] <= 1 for d in decs)
    right0 = realised_right(decs, {})
    print(f"test bed: {len(decs)} decisions in {len({d['system'] for d in decs})} systems, "
          f"{len({f for d in decs for f in d['weights']})} compounds; right as computed {right0}; hard (p <= 1) {hard}")
    table, curves = ladder(decs, eps, unc)
    table.to_csv(RES / "measurement_ladder.csv", index=False)
    curves.to_csv(RES / "measurement_ladder_curves.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(table.round(3).to_string(index=False))
    report11(*rung11(eps))


def report11(blind, entry, pre, digest):
    print("\n11/10 (a) blind literature values (trend model):")
    for label, g in (("all", blind), ("outside the benchmark", blind[~blind.in_benchmark])):
        print(f"  {label}: trend {coverage(g.z_trend)} | raw DFT {coverage(g.z_raw)}")
    print("\n11/10 (b) real values entered into the campaign decisions:")
    print(f"  decisions touched {len(entry)}; shift / predicted spread: {coverage(entry.z.dropna())}")
    print(f"  sign flips {int(entry.flips.sum())}; resolved before {int(entry.resolved_before.sum())} -> after "
          f"{int(entry.resolved_after.sum())}; P(right) before {entry.p_right_before.sum():.1f}, expected "
          f"{entry.p_right_expected.sum():.1f}, after {entry.p_right_after.sum():.1f}")
    print(f"\n11/10 (c) pre-registered {len(pre)} predictions, {pre.predicted_measured_meV_atom.notna().sum()} with a "
          f"value; sha256 {digest}")
    print(pre.to_string(index=False))


if __name__ == "__main__":
    main()
