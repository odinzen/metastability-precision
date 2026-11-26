"""Is the lean of closed-shell oxide errors toward stability the polymorph density bias in another form?

GGA/GGA+U favors the less dense of two polymorphs. If it also favors a compound that is less dense than the phases it
competes with, the computed margin of such a compound comes out too negative. For every closed-shell oxide compound
decision (the compiled benchmark and the measured-metastable oxides of the literature search), the volume change of
the reaction, the compound's volume per atom less the atom-weighted volume of its competing phases, is taken from the
same Materials Project entries that give the computed margin. The error (DFT minus measured margin) is regressed on
it, and the measured-metastable lean is compared with and without that term. Writes results/reaction_volume.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_nonoxide import pick  # noqa: E402
from calibration_base_rate import compiled  # noqa: E402
from pull_mp_systems import api_key, load_or_pull  # noqa: E402
from truth_audit import fractions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def volume_change(mpr, target, products):
    """Volume per atom of the target less that of its competing phases (Å^3/atom), lowest entries throughout."""
    els = sorted({str(e) for f in [target] + products for e in Composition(f).elements})
    entries = load_or_pull(mpr, "-".join(els))
    vol = {}
    for f in [target] + products:
        e, _ = pick(entries, f, match_reference=False)
        if e is None:
            return np.nan
        vol[f] = e.structure.volume / e.structure.num_sites
    frac = fractions(Composition(target).reduced_formula, [Composition(p).reduced_formula for p in products])
    return vol[target] - sum(frac[Composition(p).reduced_formula] * vol[p] for p in products)


def main():
    c = compiled()
    bench = c[c.regime == "compound"]
    rows = [dict(formula=x.formula, products=x.products.split("+"), t=x.truth, c=x.dft_meV_atom, source="benchmark")
            for x in bench.itertuples()]
    add = pd.read_csv(ROOT / "data" / "curated" / "metastable_additions.csv")
    react = pd.read_csv(ROOT / "data" / "curated" / "metastable_reactions.csv").set_index("compound")
    for x in add[(add.family == "oxide") & (add.sigma <= 22.5)].itertuples():
        rows.append(dict(formula=x.formula, products=react.loc[x.formula, "products"].split("+"), t=x.measured_meV_atom,
                         c=x.dft_meV_atom, source="literature"))
    with MPRester(api_key()) as mpr:
        for r in rows:
            r["dV"] = volume_change(mpr, r["formula"], r["products"])
    d = pd.DataFrame(rows).dropna(subset=["dV"])
    d = d[d.t.abs() >= 0.5]
    d["error"] = d.c - d.t
    d["metastable"] = d.t > 0
    d["products"] = d.products.map("+".join)
    d.to_csv(RES / "reaction_volume.csv", index=False)
    slope, icept, r, p, _ = stats.linregress(d.dV, d.error)
    print(f"{len(d)} decisions; error = {icept:.1f} + {slope:.1f} * dV (meV/atom per A^3/atom), r {r:.2f}, P {p:.3g}")
    for lab, g in d.groupby("metastable"):
        print(f"  measured {'metastable' if lab else 'stable'}: n {len(g)}, mean dV {g.dV.mean():+.2f}, mean error {g.error.mean():+.1f}")
    # the metastable-side lean with and without the volume term
    X = np.column_stack([np.ones(len(d)), d.metastable.astype(float), d.dV])
    beta, *_ = np.linalg.lstsq(X, d.error.values, rcond=None)
    res = d.error.values - X @ beta
    cov = np.linalg.inv(X.T @ X) * res.var(ddof=3)
    print(f"  lean (metastable minus stable) alone {d[d.metastable].error.mean() - d[~d.metastable].error.mean():+.1f}; "
          f"with the volume term {beta[1]:+.1f} +- {np.sqrt(cov[1, 1]):.1f}, volume slope {beta[2]:+.1f} +- {np.sqrt(cov[2, 2]):.1f}")


if __name__ == "__main__":
    main()
