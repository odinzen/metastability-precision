"""Does r2SCAN remove the density bias of the polymorph decisions? Each scored pair of polymorph_decisions.py is
recomputed with the Materials Project r2SCAN energies of the same two materials (same material ids, r2SCAN-relaxed
structures), whose space groups are checked against the GGA picks. Energies are raw; corrections cancel between two
entries of one composition. Writes results/polymorph_r2scan.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_r2scan import r2scan_entries  # noqa: E402
from polymorph_decisions import chemsys  # noqa: E402
from pull_mp_systems import api_key, load_or_pull, mp_number  # noqa: E402
from sigma_model import sigma_for  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / "results" / "polymorph_decisions.csv")
    d = d[d.sign_established]
    rows = []
    with MPRester(api_key()) as mpr:
        for _, r in d.iterrows():
            ids = int(r.mp_metastable[3:]), int(r.mp_stable[3:])
            gga = {mp_number(e): e for e in load_or_pull(mpr, chemsys(r.formula)) if mp_number(e) in ids}
            r2 = {mp_number(e): e for e in r2scan_entries(mpr, chemsys(r.formula)) if mp_number(e) in ids}
            row = dict(compound=r.compound, formula=r.formula, measured_meV_atom=r.measured_meV_atom,
                       gga_meV_atom=r.dft_meV_atom,
                       denser_stable_pct=round(100 * (gga[ids[1]].structure.density / gga[ids[0]].structure.density - 1), 1))
            if all(i in r2 for i in ids):
                m, s = r2[ids[0]], r2[ids[1]]
                sg = [SpacegroupAnalyzer(e.structure, symprec=0.1).get_space_group_number() for e in (m, s)]
                row.update(r2scan_meV_atom=round(1000 * (m.energy_per_atom - s.energy_per_atom), 2),
                           sg_r2scan=f"{sg[0]}/{sg[1]}", sg_gga=f"{r.sg_metastable}/{r.sg_stable}")
            rows.append(row)
    out = pd.DataFrame(rows)
    out["sigma"] = [sigma_for(f) for f in out.formula]
    for f in ("gga", "r2scan"):
        right = np.sign(out[f"{f}_meV_atom"]) == np.sign(out.measured_meV_atom)
        out[f"{f}_right"] = right.astype(object).where(out[f"{f}_meV_atom"].notna())
        out[f"{f}_p"] = (out[f"{f}_meV_atom"].abs() / out.sigma).round(3)
    out.to_csv(ROOT / "results" / "polymorph_r2scan.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.drop(columns=["formula"]).to_string(index=False))
    both = out.dropna(subset=["r2scan_meV_atom"])
    print(f"\n{len(both)} of {len(out)} pairs with r2SCAN energies for both materials")
    for f in ("gga", "r2scan"):
        err = both[f"{f}_meV_atom"] - both.measured_meV_atom
        print(f"  {f:7s} right {int(both[f + '_right'].sum())}/{len(both)}, mean error {err.mean():+.1f}, "
              f"MAE {err.abs().mean():.1f}, expected right {norm.cdf(both[f + '_p']).sum():.1f}")
    dense = both[both.denser_stable_pct > 0]
    print(f"  stable form denser: {len(dense)} pairs, GGA right {int(dense.gga_right.sum())}, "
          f"r2SCAN right {int(dense.r2scan_right.sum())}")
    print(f"  stable form less dense: {len(both) - len(dense)} pairs, GGA right {int(both.gga_right.sum() - dense.gga_right.sum())}, "
          f"r2SCAN right {int(both.r2scan_right.sum() - dense.r2scan_right.sum())}")
    changed = both[both.sg_r2scan != both.sg_gga]
    if len(changed):
        print("  space group differs from the GGA pick:\n" + changed[["compound", "sg_gga", "sg_r2scan"]].to_string(index=False))


if __name__ == "__main__":
    main()
