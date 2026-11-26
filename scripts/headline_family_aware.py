"""The two headline shares under the deposited family-aware error model.

The shares of the text (78% of known-oxide calls, 57% of GNoME claims after the selection correction) use the
calorimetric sigma. The family-aware model of preregistration_model.py, validated on the 118 known-answer decisions,
gives those decisions a wider uncertainty (median 36.4 meV/atom) and centres their true margins below the computed ones
(median -11.5 meV/atom, the under-binding of ternary oxides). Rebuilding every observed and GNoME decision from its
decomposition is not needed for a sensitivity: the median width and shift are carried into the same shares, the
selection correction of gnome_precision.py recomputed at that width. Writes results/headline_family_aware.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnome_precision as G  # noqa: E402
import preregistration_model as P  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    out, _ = P.validate_decisions(P.measured_values())
    width, shift = float(out.sd.median()), float((out.mu - out.margin_dft).median())
    d = pd.read_csv(RES / "precision_mp_observed.csv")
    o = d[d.oxide & d.competitor.isin(["fewer-element phases only", "same-order neighbour"])]
    m = o.margin_meV.values
    rows = []
    for lab, sd, sh in (("calorimetric sigma", None, 0.0), ("family-aware width", width, 0.0),
                        ("family-aware width and shift", width, shift)):
        s = o.sigma.values if sd is None else np.where(o.sigma.values > 22.5, max(34.0, sd), sd)
        p = np.where(m < 0, norm.cdf(-(m + sh) / s), norm.cdf((m + sh) / s))
        rows.append(dict(set="known oxides", forecast=lab, n=len(m), expected_right=p.mean()))
    g = G.oxides()
    for lab, sc, so in (("calorimetric sigma", 22.5, 34.0), ("family-aware width", width, width)):
        claim, _, flat, eb = G.score(g, "MP snapshot", sc, so)
        rows.append(dict(set="GNoME, face value", forecast=lab, n=int(claim.sum()), expected_right=flat[claim].mean()))
        rows.append(dict(set="GNoME, selection-corrected", forecast=lab, n=int(claim.sum()), expected_right=eb[claim].mean()))
    t = pd.DataFrame(rows)
    t["width_meV_atom"], t["shift_meV_atom"] = round(width, 1), round(shift, 1)
    t.to_csv(RES / "headline_family_aware.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
