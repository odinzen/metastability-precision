"""Is the offset a property of GGA or of DFT? The calorimetric and compiled benchmarks recomputed with the Materials
Project's r2SCAN energies, same compounds, same reference polymorphs, compared with GGA/GGA+U on the compounds both
cover. r2SCAN energies are used raw: the empirical corrections cancel in these balanced oxide reactions anyway.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from emmet.core.mpid import AlphaID
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dft_vs_measured import KJ_PER_EV, anchors, lowest, oxide_split  # noqa: E402
from pull_mp_systems import api_key  # noqa: E402
from mp_api.client import MPRester  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "mp_r2scan"


def r2scan_entries(mpr, chemsys):
    path = CACHE / f"{chemsys}.json.gz"
    if path.exists():
        return MontyDecoder().process_decoded(json.load(gzip.open(path)))
    entries = mpr.get_entries_in_chemsys(chemsys.split("-"), additional_criteria={"thermo_types": ["R2SCAN"]})
    for e in entries:
        if isinstance(e.data.get("oxidation_states"), dict):
            e.data["oxidation_states"] = {str(k): v for k, v in e.data["oxidation_states"].items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(entries, f, cls=MontyEncoder)
    return entries


def fit(meas, err):
    k, c = np.polyfit(meas, err, 1)
    return k, c, np.std(err - (k * meas + c), ddof=2)


def calorimetric(mpr):
    gga = pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
    gga = gga[gga.family != "Li cathode oxides"].drop_duplicates(["compound", "paper", "measured_kJ"])
    df = anchors()
    rows = []
    for _, r in df.iterrows():
        comp = Composition(r.reduced)
        entries = r2scan_entries(mpr, "-".join(sorted(str(e) for e in comp.elements)))
        parts = oxide_split(r.reduced)
        target = lowest(entries, r.reduced)
        refs = {f: lowest(entries, f, match_reference=True) for f in parts}
        if target is None or any(v is None for v in refs.values()):
            continue
        e_kj = (target.energy_per_atom * comp.num_atoms
                - sum(n * refs[f].energy_per_atom * Composition(f).num_atoms for f, n in parts.items())) * KJ_PER_EV
        meas_kj = r.value / r.formula_units
        rows.append(dict(compound=r.reduced, family=r.family, measured_kJ=round(meas_kj, 2),
                         r2scan_error_meV_atom=round((e_kj - meas_kj) / KJ_PER_EV / comp.num_atoms * 1000, 1)))
    r2 = pd.DataFrame(rows).drop_duplicates(["compound", "measured_kJ"])
    both = gga.merge(r2, on=["compound", "family", "measured_kJ"])
    both.to_csv(ROOT / "results" / "benchmark_r2scan_calorimetric.csv", index=False)
    print(f"calorimetric: {len(gga)} GGA values, {len(r2)} with r2SCAN entries, {len(both)} matched")
    m = both.measured_meV_atom.values
    for lab, col in (("GGA/GGA+U", "error_meV_atom"), ("r2SCAN", "r2scan_error_meV_atom")):
        k, c, sd = fit(m, both[col].values)
        print(f"  {lab:10s} recovers {100 * (1 + k):.0f}% (slope {k:+.3f}, intercept {c:+.1f}), scatter {sd:.1f}, "
              f"mean {both[col].mean():+.1f}, MAE {both[col].abs().mean():.1f} meV/atom")
    print(both.groupby("family")[["error_meV_atom", "r2scan_error_meV_atom"]].agg(["count", "mean"]).round(1).to_string())


def compiled(mpr):
    gga = pd.read_csv(ROOT / "results" / "benchmark_compiled.csv")
    oxides = {}
    for s in gga.oxides:
        for part in s.split("+"):
            oxides[part.split(" ", 1)[1]] = None
    # material ids of the binary oxides come from the compilation, as in benchmark_compiled.py
    from benchmark_compiled import compiled as table
    t = table().set_index("formula")
    ids = sorted(set(gga.mpid.dropna()) | {t.loc[f, "likely_mpid"] for f in oxides if isinstance(t.loc[f, "likely_mpid"], str)})
    e = {}
    for chunk in [ids[i:i + 200] for i in range(0, len(ids), 200)]:
        for doc in mpr.materials.thermo.search(material_ids=chunk, thermo_types=["R2SCAN"],
                                               fields=["material_id", "energy_per_atom"]):
            e[int(AlphaID(str(doc.material_id)))] = doc.energy_per_atom

    def epa(mid):
        return e.get(int(AlphaID(mid))) if isinstance(mid, str) else None

    rows = []
    for _, r in gga.iterrows():
        n = Composition(r.compound).num_atoms
        ec = epa(r.mpid)
        parts = [(float(p.split(" ", 1)[0]), p.split(" ", 1)[1]) for p in r.oxides.split("+")]
        eo = [(nu, epa(t.loc[f, "likely_mpid"]), Composition(f).num_atoms) for nu, f in parts]
        if ec is None or any(x is None for _, x, _ in eo):
            continue
        dft = 1000 * (ec * n - sum(nu * x * na for nu, x, na in eo)) / n
        rows.append(dict(compound=r.compound, measured_meV_atom=r.measured_meV_atom, gga_error=r.error_meV_atom,
                         r2scan_error=round(dft - r.measured_meV_atom, 1)))
    both = pd.DataFrame(rows)
    both.to_csv(ROOT / "results" / "benchmark_r2scan_compiled.csv", index=False)
    print(f"\ncompiled: {len(gga)} reactions, {len(both)} with r2SCAN energies for every phase")
    m = both.measured_meV_atom.values
    for lab, col in (("GGA/GGA+U", "gga_error"), ("r2SCAN", "r2scan_error")):
        k, c, sd = fit(m, both[col].values)
        print(f"  {lab:10s} recovers {100 * (1 + k):.0f}% (slope {k:+.3f}), scatter {sd:.1f}, mean {both[col].mean():+.1f}, "
              f"MAE {both[col].abs().mean():.1f}, positive {100 * (both[col] > 0).mean():.0f}%")


def main():
    with MPRester(api_key()) as mpr:
        calorimetric(mpr)
        compiled(mpr)


if __name__ == "__main__":
    main()
