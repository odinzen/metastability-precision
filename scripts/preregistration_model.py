"""Error model for advance predictions: family-aware and error-size-aware, validated by leaving each family out.

The calorimetric trend (error = -0.143 x measured + 8.1, scatter 22.5) failed the blind test on literature values
outside the benchmark (measurement_design_ladder.py, rung 11): the scatter transferred but the offset did not, and
the errors of strongly reacting compounds were larger. So the error of a compound, DFT minus measured enthalpy of
formation from its binary oxides, is modelled as

    err = a_family + k x measured + eta,    eta ~ N(0, s0^2 + (s1 x measured)^2),    a_family ~ N(a0, tau^2),

with the family the first oxyanion former present (precision_mp_observed.family) and the family offsets pooled
(random effects, marginal likelihood). It is fitted on every measured value we hold: the 69 calorimetric values, the
compiled values (CuAl2O4 and Ca3Si2O7 out, their compiled values being wrong) and the blind hold-out.

Validation, before any prediction is written:
- leave one family out: each family is predicted from the others, with only the pooled offset;
- leave one compound out: each compound is predicted with its family's other members;
- the same two for the old trend model;
- decisions: the 118 known-answer decompositions of the ladder test bed, each predicted with the model refitted
  without its chemical system, scored against the old Phi(|margin| / sigma).
Then the pre-registration: the 30 compounds the measurement ranking would measure next, and every campaign decision
they enter, with the model fitted on all data. Written with a SHA-256 hash.

Writes results/preregistration_validation.csv, results/preregistration_model.json,
results/preregistered_predictions.csv and results/preregistered_decisions.csv (+ .sha256).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deposited  # noqa: E402
import measurement_design_ladder as L  # noqa: E402
import measurement_value as mv  # noqa: E402
from precision_mp_observed import family  # noqa: E402
from sigma_model import open_shell, sigma_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OLD_K, OLD_C = -0.143, 8.1


def red(f):
    return Composition(f).reduced_formula


def fam(f):
    return family([str(e) for e in Composition(f).elements])


def measured_values():
    rows = {}
    b = (pd.read_csv(RES / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
         .drop_duplicates(["compound", "measured_meV_atom"]))
    for r in b.itertuples():
        # Lowest-energy reference polymorphs, as the predictions use (dft_from_oxides), not the calorimetric ones.
        err = r.error_meV_atom_lowest_polymorph_refs if pd.notna(r.error_meV_atom_lowest_polymorph_refs) else r.error_meV_atom
        rows.setdefault(red(r.compound), dict(compound=red(r.compound), measured=r.measured_meV_atom,
                                              dft=r.measured_meV_atom + err, source="calorimetric benchmark"))
    for r in pd.read_csv(RES / "holdout_test.csv").itertuples():
        rows.setdefault(red(r.compound), dict(compound=red(r.compound), measured=r.measured_meV_atom, dft=r.dft_meV_atom,
                                              source="blind hold-out"))
    for r in pd.read_csv(RES / "benchmark_compiled.csv").itertuples():
        f = red(r.compound)
        if f in {red(x) for x in L.AUDIT_DROP}:
            continue
        rows.setdefault(f, dict(compound=f, measured=r.measured_meV_atom, dft=r.dft_meV_atom, source="compiled"))
    d = pd.DataFrame(rows.values())
    d["family"] = d.compound.map(fam)
    d["open_shell"] = d.compound.map(open_shell)
    d["err"] = d.dft - d.measured
    # Repeat values of one compound (several LaFeO3 or MgAl2O4 measurements) keep the first source only.
    return d


# ---------------------------------------------------------------- model
def unpack(theta):
    a0, k, ls0, ls1, ltau = theta
    return a0, k, np.exp(ls0), np.exp(ls1), np.exp(ltau)


def noise_sd(m, s0, s1):
    return np.sqrt(s0 ** 2 + (s1 * m) ** 2)


def nll(theta, d):
    a0, k, s0, s1, tau = unpack(theta)
    total = 0.0
    for _, g in d.groupby("family"):
        r = g.err.to_numpy() - a0 - k * g.measured.to_numpy()
        v = noise_sd(g.measured.to_numpy(), s0, s1) ** 2
        C = np.diag(v) + tau ** 2
        sign, logdet = np.linalg.slogdet(C)
        total += 0.5 * (logdet + r @ np.linalg.solve(C, r) + len(r) * np.log(2 * np.pi))
    return total


def fit(d):
    best = None
    for start in ([0.0, -0.1, np.log(15), np.log(0.05), np.log(15)], [10.0, -0.15, np.log(20), np.log(0.02), np.log(5)]):
        r = minimize(nll, start, args=(d,), method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    a0, k, s0, s1, tau = unpack(best.x)
    offsets = {}
    for f, g in d.groupby("family"):
        r = g.err.to_numpy() - a0 - k * g.measured.to_numpy()
        w = 1 / noise_sd(g.measured.to_numpy(), s0, s1) ** 2
        prec = 1 / tau ** 2 + w.sum()
        offsets[f] = (a0 + (w * r).sum() / prec, 1 / prec)
    return dict(a0=a0, k=k, s0=s0, s1=s1, tau=tau, offsets=offsets, nll=best.fun, n=len(d))


def offset_of(model, f, use_family=True):
    if use_family and f in model["offsets"]:
        return model["offsets"][f]
    return model["a0"], model["tau"] ** 2


def predict_measured(model, dft, f, use_family=True):
    """Measured enthalpy from oxides (mean, sd) for a compound with computed enthalpy dft in family f."""
    a, va = offset_of(model, f, use_family)
    k = model["k"]
    m = (dft - a) / (1 + k)
    for _ in range(20):  # the noise depends on the unknown measured value
        sd = np.sqrt(va + noise_sd(m, model["s0"], model["s1"]) ** 2) / (1 + k)
        m = (dft - a) / (1 + k)
    return m, sd


def old_predict(dft, f):
    s = sigma_for(f)
    return (dft - OLD_C) / (1 + OLD_K), s / (1 + OLD_K)


def coverage(z):
    z = np.asarray(z, float)
    return dict(n=len(z), within1=float(np.mean(np.abs(z) <= 1)), within2=float(np.mean(np.abs(z) <= 2)),
                mean_z=float(np.mean(z)), rms_z=float(np.sqrt(np.mean(z ** 2))))


def validate(d):
    rows, zs = [], {"new, family left out": [], "new, compound left out": [], "old trend": []}
    for f in sorted(d.family.unique()):
        train, test = d[d.family != f], d[d.family == f]
        mdl = fit(train)
        for r in test.itertuples():
            m, sd = predict_measured(mdl, r.dft, f, use_family=False)
            zs["new, family left out"].append((r.measured - m) / sd)
            mo, so = old_predict(r.dft, r.compound)
            zs["old trend"].append((r.measured - mo) / so)
            rows.append(dict(compound=r.compound, family=f, source=r.source, measured=r.measured, dft=r.dft,
                             pred_family_out=m, sd_family_out=sd, pred_old=mo, sd_old=so))
    for i, r in d.iterrows():
        mdl = fit(d.drop(index=i))
        m, sd = predict_measured(mdl, r.dft, r.family, use_family=True)
        zs["new, compound left out"].append((r.measured - m) / sd)
    out = pd.DataFrame(rows)
    out["z_family_out"] = zs["new, family left out"]
    out["z_old"] = zs["old trend"]
    loco = dict(zip(d.compound, zs["new, compound left out"]))
    out["z_compound_out"] = out.compound.map(loco)
    summary = {k: coverage(v) for k, v in zs.items()}
    by_family = (out.groupby("family").agg(n=("z_old", "size"), rms_old=("z_old", lambda z: np.sqrt(np.mean(z ** 2))),
                                           rms_family_out=("z_family_out", lambda z: np.sqrt(np.mean(z ** 2))),
                                           mean_old=("z_old", "mean"), mean_family_out=("z_family_out", "mean"))
                 .round(2).reset_index())
    return out, summary, by_family


# ---------------------------------------------------------------- decisions
def decision_prediction(model, dec, m_of, use_family=True):
    """Predicted true margin (mean, sd) of a decision from the model: error = sum w_f eps_f, with family offsets
    shared by the phases of one family."""
    mean_e, var_noise, fam_w = 0.0, 0.0, {}
    k = model["k"]
    for f, w in dec["weights"].items():
        m_f, _ = m_of(f)
        a, va = offset_of(model, fam(f), use_family)
        mean_e += w * (a + k * m_f)
        var_noise += (w * noise_sd(m_f, model["s0"], model["s1"])) ** 2
        fam_w.setdefault(fam(f), [0.0, va])[0] += w
    var = var_noise + sum(ws ** 2 * va for ws, va in fam_w.values())
    return dec["margin_dft"] - mean_e, np.sqrt(var)


def validate_decisions(d):
    decs, eps, _ = L.test_bed()
    meas = d.set_index("compound")
    rows = []
    for dec in decs:
        phases = set(dec["weights"])
        train = d[~d.compound.isin(phases)]
        mdl = fit(train)

        def m_of(f):
            return predict_measured(mdl, float(meas.loc[f, "dft"]), fam(f), use_family=True)
        mu, sd = decision_prediction(mdl, dec, m_of)
        p_right_new = norm.cdf(np.sign(dec["margin_dft"]) * mu / sd)
        right = float(np.sign(dec["margin_dft"]) == np.sign(dec["margin_meas"]))
        rows.append(dict(decision=dec["label"], system=dec["system"], margin_dft=dec["margin_dft"],
                         margin_meas=dec["margin_meas"], right=right, p_new=p_right_new,
                         p_old=norm.cdf(abs(dec["margin_dft"]) / dec["sigma"]), mu=mu, sd=sd,
                         new_sign_right=float(np.sign(mu) == np.sign(dec["margin_meas"]))))
    out = pd.DataFrame(rows)
    q = out.right.to_numpy()
    base = q.mean()
    res = dict(n=len(out), computed_sign_right=float(base), model_sign_right=float(out.new_sign_right.mean()),
               brier_new=float(np.mean((q - out.p_new) ** 2)), brier_old=float(np.mean((q - out.p_old) ** 2)),
               brier_constant=float(np.mean((q - base) ** 2)),
               log_new=float(-np.mean(q * np.log(np.clip(out.p_new, 1e-9, 1)) + (1 - q) * np.log(np.clip(1 - out.p_new, 1e-9, 1)))),
               log_old=float(-np.mean(q * np.log(np.clip(out.p_old, 1e-9, 1)) + (1 - q) * np.log(np.clip(1 - out.p_old, 1e-9, 1)))))
    return out, res


# ---------------------------------------------------------------- pre-registration
def preregister(d, model):
    decs = L.campaign_signed()
    mvals = pd.read_csv(RES / "measurement_value.csv")
    picks = [m for m in mvals.measured if isinstance(m, str) and not m.startswith(("none", "enter"))]
    today = pd.read_csv(RES / "mp_targets_today.csv")
    chemsys_of = {}
    for dec in decs:
        ds = dec["label"].split(" ")[0]
        cs = today[(today.dataset == ds) & (today.target == dec["target"])].chemsys
        if len(cs):
            for f in dec["weights"]:
                chemsys_of.setdefault(f, cs.iloc[0])
    meas = d.set_index("compound")
    dft_cache = {}

    def dft_of(f):
        if f not in dft_cache:
            dft_cache[f] = L.dft_from_oxides(f, chemsys_of[f]) if f in chemsys_of else None
        return dft_cache[f]

    # The pipeline check: for measured compounds in the campaign systems, the DFT value from the campaign entries
    # must match the one the model was fitted on, or the prediction mixes two energy scales.
    check = [(f, dft_of(f), float(meas.loc[f, "dft"])) for f in chemsys_of if f in meas.index and dft_of(f) is not None]
    diffs = np.array([a - b for _, a, b in check])

    comp_rows = []
    for rank, f in enumerate(picks, 1):
        dft = dft_of(f)
        if dft is None:
            comp_rows.append(dict(rank=rank, compound=f, family=fam(f)))
            continue
        m, sd = predict_measured(model, dft, fam(f), use_family=True)
        comp_rows.append(dict(rank=rank, compound=f, family=fam(f), family_in_training=fam(f) in model["offsets"],
                              open_shell=open_shell(f), binary_oxides="+".join(sorted((L.oxides_of(f) or {}).values())),
                              dft_meV_atom=round(dft, 1), predicted_measured_meV_atom=round(m, 1),
                              interval_68=f"{m - sd:.1f} to {m + sd:.1f}",
                              interval_95=f"{m - 1.96 * sd:.1f} to {m + 1.96 * sd:.1f}"))

    def m_of(f):
        if f in meas.index:  # already measured: the measured value, with its compiled scatter folded into the noise
            return float(meas.loc[f, "measured"]), 0.0
        return predict_measured(model, dft_of(f), fam(f), use_family=True)

    dec_rows = []
    pickset = set(picks)
    for dec in decs:
        if not set(dec["weights"]) & pickset:
            continue
        if any(f not in meas.index and dft_of(f) is None for f in dec["weights"]):
            dec_rows.append(dict(decision=dec["label"], note="a phase without a computed enthalpy from oxides"))
            continue
        local = dict(dec, margin_dft=dec["margin_signed"])
        mu, sd = decision_prediction(model, local, m_of)
        computed_sign = "favoured (negative)" if dec["margin_signed"] < 0 else "not favoured (positive)"
        dec_rows.append(dict(decision=dec["label"], phases_to_measure="+".join(sorted(set(dec["weights"]) & pickset)),
                             computed_margin_meV_atom=round(dec["margin_signed"], 1), computed_sign=computed_sign,
                             predicted_true_margin_meV_atom=round(mu, 1), predicted_sd=round(sd, 1),
                             p_computed_sign_confirmed=round(float(norm.cdf(np.sign(dec["margin_signed"]) * mu / sd)), 3),
                             predicted_sign="negative" if mu < 0 else "positive"))
    return pd.DataFrame(comp_rows), pd.DataFrame(dec_rows), diffs, check


write_hashed = deposited.write_hashed  # measurement_value_observed calls P.write_hashed


def main():
    d = measured_values()
    print(f"{len(d)} measured compounds: {d.source.value_counts().to_dict()}; families {d.family.value_counts().to_dict()}")
    model = fit(d)
    print("model:", {k: round(v, 4) if isinstance(v, float) else v for k, v in model.items() if k != "offsets"})
    print("family offsets (meV/atom, sd):", {f: (round(a, 1), round(np.sqrt(v), 1)) for f, (a, v) in model["offsets"].items()})
    val, summary, by_family = validate(d)
    val.to_csv(RES / "preregistration_validation.csv", index=False)
    print("\nvalidation by leaving out:")
    for k, v in summary.items():
        print(f"  {k:28s} {({a: round(b, 3) for a, b in v.items()})}")
    print(by_family.to_string(index=False))
    dval, dres = validate_decisions(d)
    dval.to_csv(RES / "preregistration_decision_validation.csv", index=False)
    print("\ndecisions (118 known answers, model refitted without each decision's phases):",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in dres.items()})
    json.dump(dict({k: v for k, v in model.items() if k != "offsets"},
                   offsets={f: dict(mean=a, sd=float(np.sqrt(v))) for f, (a, v) in model["offsets"].items()},
                   validation=summary, decisions=dres), open(RES / "preregistration_model.json", "w"), indent=1)
    comp, decs, diffs, check = preregister(d, model)
    print(f"\npipeline check, campaign DFT minus fitted DFT for {len(diffs)} measured compounds: median {np.median(diffs):.1f}, "
          f"max |{np.max(np.abs(diffs)):.1f}| meV/atom")
    stamp = date.today().isoformat()
    head = (f"# Pre-registered predictions, written {stamp} before any of these compounds is measured. Enthalpy of "
            "formation from the binary oxides, meV/atom, at 298 K. Model (scripts/preregistration_model.py): DFT minus "
            "measured = family offset + k x measured + noise growing with the size of the reaction, fitted on "
            f"{len(d)} measured compounds; intervals validated by leaving each family out "
            f"(within 1 sd {summary['new, family left out']['within1']:.2f}, within 2 sd "
            f"{summary['new, family left out']['within2']:.2f}).\n")
    h1 = write_hashed(RES / "preregistered_predictions.csv", head, comp)
    head2 = (f"# Pre-registered decision predictions, written {stamp}. For each campaign decision that a pre-registered "
             "compound enters: the computed margin (Materials Project GGA/GGA+U, release 2026.09.28), the margin the "
             "model predicts calorimetry will give, and the probability that measurement confirms the computed sign.\n")
    h2 = write_hashed(RES / "preregistered_decisions.csv", head2, decs)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 50):
        print(comp.to_string(index=False))
        print(decs.to_string(index=False))
    print(f"\nsha256 compounds {h1}\nsha256 decisions {h2}")


if __name__ == "__main__":
    main()
