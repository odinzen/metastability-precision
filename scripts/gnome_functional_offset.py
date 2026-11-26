"""How far r2SCAN and GGA/GGA+U differ at the hull with no selection involved, and what that does to the GNoME
selection check (gnome_precision.py). The check reads the r2SCAN-minus-GGA margin of GNoME's below-hull claims at the
hull as the visible part of a selection shift. But r2SCAN binds ternary oxides more strongly than GGA (it recovers all
of the measured enthalpy from the binary oxides, GGA about 86%), so the two functionals differ at the hull even for
decisions nobody selected. That offset is measured here on the known-answer decisions of the compiled benchmark
(measurement_design_ladder.test_bed), from the per-compound r2SCAN and GGA errors, and subtracted before the shift is
inferred. Writes results/gnome_functional_offset.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_design_ladder as L  # noqa: E402
from sigma_model import open_shell  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
NEAR_HULL = 60.0


def offsets():
    decs, _, _ = L.test_bed()
    r = pd.read_csv(RES / "benchmark_r2scan_compiled.csv")
    diff = dict(zip(r.compound.map(lambda f: Composition(f).reduced_formula), r.r2scan_error - r.gga_error))
    rows = [dict(label=d["label"], gga=d["margin_dft"], D=sum(w * diff[f] for f, w in d["weights"].items()),
                 open_shell=open_shell(d["label"]), binaries_only=len(d["weights"]) == 1)
            for d in decs if all(f in diff for f in d["weights"])]
    return pd.DataFrame(rows)


def intercept(g, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    boot = [np.polyfit(g.gga.values[i], g.D.values[i], 1)[1] for i in (rng.integers(0, len(g), len(g)) for _ in range(n_boot))]
    return np.polyfit(g.gga, g.D, 1)[1], *np.percentile(boot, [2.5, 97.5])


def main():
    x = offsets()
    check = pd.read_csv(RES / "gnome_selection_check.csv").set_index("regime")
    rows = []
    for reg, g in (("compound", x[~x.open_shell]), ("open-shell", x[x.open_shell])):
        for subset, h in (("all decisions", g), (f"GGA margin within {NEAR_HULL:.0f}", g[g.gga.abs() <= NEAR_HULL])):
            if len(h) < 5:
                continue
            c, lo, hi = intercept(h)
            k = check.loc[reg]
            visible = k.shift_at_hull - c
            shares = (k.visible_share_calorimetric, k.visible_share_compiled)
            rows.append(dict(regime=reg, subset=subset, n=len(h), offset_at_hull=c, offset_lo95=lo, offset_hi95=hi,
                             gnome_shift_at_hull=k.shift_at_hull, visible_after_offset=visible,
                             implied_shift_lo=visible / max(shares), implied_shift_hi=visible / min(shares),
                             exponential_shift=k.exponential_shift))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "gnome_functional_offset.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.1f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
