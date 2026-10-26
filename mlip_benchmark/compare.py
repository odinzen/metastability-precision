"""MLIP enthalpy of formation from the binary oxides against calorimetry and against DFT, on the same 69 values and
the same reference polymorphs as results/benchmark_dft_vs_measured.csv.

The Materials Project's empirical corrections (oxide anion, GGA+U) cancel in a balanced reaction from binary oxides,
because oxygen and each transition metal appear in the same amount and oxidation state on both sides, so raw MLIP
energies compare directly with the corrected DFT ones. Energies are at the MLIP-relaxed geometry; the MP-geometry
single point is reported alongside to separate relaxation from the potential itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_dft_vs_measured import oxide_split  # noqa: E402

MODELS = {"orb20": "Orb-v3 cons. 20 A", "orbinf": "Orb-v3 cons. inf (LLZO model)", "mace": "MACE-MP-0 medium"}


def energies(model):
    path = ROOT / "mlip_benchmark" / f"relaxed_{model}.json"
    if not path.exists():
        return None
    rows = json.load(open(path))
    return {(r["formula"], r["role"]): r for r in rows}


def main():
    bench = (pd.read_csv(ROOT / "results" / "benchmark_dft_vs_measured.csv").dropna(subset=["error_meV_atom"])
             .drop_duplicates(["compound", "measured_meV_atom"]))
    out = bench[["family", "compound", "measured_meV_atom", "error_meV_atom"]].rename(
        columns={"error_meV_atom": "dft_error"}).copy()
    out["dft_meV_atom"] = out.measured_meV_atom + out.dft_error
    summary = []
    for model, label in MODELS.items():
        e = energies(model)
        if e is None:
            continue
        for geometry in ("e_relaxed", "e_mp_geometry"):
            vals = []
            for f in out.compound:
                comp = Composition(f)
                parts = oxide_split(f)
                try:
                    tgt = e[(f, "target")][geometry]
                    ref = sum(n * e[(ox, "reference")][geometry] * Composition(ox).num_atoms for ox, n in parts.items())
                    vals.append(1000 * (tgt * comp.num_atoms - ref) / comp.num_atoms)
                except KeyError:
                    vals.append(np.nan)
            col = f"{model}_{'relaxed' if geometry == 'e_relaxed' else 'mpgeom'}"
            out[col] = np.round(vals, 1)
            err = out[col] - out.measured_meV_atom
            ok = err.notna()
            x, y = out.measured_meV_atom[ok].values, err[ok].values
            k, c = np.polyfit(x, y, 1)
            summary.append(dict(model=label, geometry=geometry, n=int(ok.sum()), mean_signed=round(err.mean(), 1),
                                mean_abs=round(err.abs().mean(), 1), slope=round(k, 3),
                                share_recovered=round(1 + k, 3), residual_sd=round(np.std(y - (k * x + c), ddof=2), 1),
                                mean_abs_vs_dft=round((out[col] - out.dft_meV_atom).abs().mean(), 1)))
    x, y = out.measured_meV_atom.values, out.dft_error.values
    k, c = np.polyfit(x, y, 1)
    summary.insert(0, dict(model="DFT (MP GGA/GGA+U)", geometry="MP", n=len(out), mean_signed=round(y.mean(), 1),
                           mean_abs=round(np.abs(y).mean(), 1), slope=round(k, 3), share_recovered=round(1 + k, 3),
                           residual_sd=round(np.std(y - (k * x + c), ddof=2), 1), mean_abs_vs_dft=0.0))
    out.to_csv(ROOT / "results" / "mlip_vs_measured.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(ROOT / "results" / "mlip_vs_measured_summary.csv", index=False)
    print(s.to_string(index=False))
    fam_cols = [c for c in out.columns if c.endswith("_relaxed")]
    fam = out.groupby("family").apply(lambda g: pd.Series(
        {"n": len(g), "DFT": round(g.dft_error.mean(), 1),
         **{c.replace("_relaxed", ""): round((g[c] - g.measured_meV_atom).mean(), 1) for c in fam_cols}}))
    print("\nMean signed error by family (meV/atom, relaxed geometry):")
    print(fam.to_string())


if __name__ == "__main__":
    main()
