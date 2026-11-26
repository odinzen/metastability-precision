"""The trust scale: the probability that a calculated stability call is right at a given margin, by kind of decision.

Uses the calibrated models and nothing new. Closed shell compound decisions take sigma = 22.5 meV/atom, open shell
decisions 34 (an upper bound on the probability), oxides outside the calibrated families the family-aware width, and
intermetallic decisions the Student's t that fits their outcomes in both halves of p. Polymorph decisions get no
probability, because their error is systematic. Observed shares are from the calibration tables. Writes
results/trust_scale.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
T_NU, T_SCALE = 1.5, 46.0
MARGINS = (5, 10, 15, 25, 50, 100)


def gaussian(sigma):
    return dict(cdf=lambda m: norm.cdf(m / sigma), ppf=lambda q: sigma * norm.ppf(q))


def heavy():
    return dict(cdf=lambda m: student.cdf(m / T_SCALE, T_NU), ppf=lambda q: T_SCALE * student.ppf(q, T_NU))


def main():
    width = pd.read_csv(RES / "headline_family_aware.csv").width_meV_atom.iloc[0]
    models = {"compound": gaussian(22.5), "family-aware": gaussian(float(width)), "open-shell": gaussian(34.0),
              "intermetallic": heavy()}
    rows = []
    for name, m in models.items():
        row = dict(decision=name, margin_84=float(m["ppf"](0.84)), margin_98=float(m["ppf"](0.98)))
        row.update({f"P_{x}": float(m["cdf"](x)) for x in MARGINS})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "trust_scale.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
    print("gaussian at sigma 138:", {x: round(float(norm.cdf(x / 138.4)), 3) for x in MARGINS})


if __name__ == "__main__":
    main()
