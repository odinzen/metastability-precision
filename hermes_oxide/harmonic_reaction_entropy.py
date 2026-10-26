"""Orb-v3 harmonic entropies from sol/phonon_run.py against measurement, per phase and for the reactions from the binary
oxides whose phases were all computed. At 298.15 K the reaction is compared with the survey value; at 1000 K with the
value integrated from tabulated heat capacities (reference_entropies.csv) where every phase has one. The 1000 K harmonic
value also shows how far the heat-capacity change of reaction moves the entropy, the term the measured survey neglects.

Writes results/harmonic_reaction_entropy.csv and results/harmonic_vs_measured.csv (per phase).

Usage: python harmonic_reaction_entropy.py [PHONON_DIR]   (default ../sol/phonons)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
MEV_PER_J_MOL = 1000 / 96485.33  # T dS in J/mol atoms to meV per atom
TEMPS = ("298.15", "1000.0")


def measured_per_atom():
    """Measured S per mole of atoms by (reduced formula, temperature string)."""
    ref = pd.read_csv(Path(__file__).resolve().parent / "reference_entropies.csv")
    out = {}
    for _, r in ref.iterrows():
        for t in TEMPS:
            if abs(r.T_K - float(t)) < 0.5:
                out[(Composition(r.compound).reduced_formula, t)] = r.S_per_mol_atoms
    return out


def imag_of(r):
    return r["runs"][r.get("reference_run", "d0.01")]["imaginary_fraction"]


def main(res):
    s = {p.stem: json.load(open(p)) for p in Path(res).glob("*.json")}
    s_meas = measured_per_atom()

    phases = []
    for key, r in sorted(s.items()):
        if not any((key, t) in s_meas for t in TEMPS):  # the step E folder also holds every phase of the campaigns
            continue
        for t in TEMPS:
            m = s_meas.get((key, t), np.nan)
            phases.append(dict(phase=r["phase"], T_K=float(t), S_harmonic=round(r["S_per_mol_atoms"][t], 3),
                               S_measured=m, dS=round(r["S_per_mol_atoms"][t] - m, 3),
                               TdS_meV_atom=round(float(t) * (r["S_per_mol_atoms"][t] - m) * MEV_PER_J_MOL, 2),
                               imaginary_fraction=imag_of(r), converged=r["converged"]))
    pd.DataFrame(phases).to_csv(ROOT / "results" / "harmonic_vs_measured.csv", index=False)

    meas = pd.read_csv(ROOT / "results" / "reaction_entropy_measured.csv")
    rows = []
    for _, r in meas.iterrows():
        comp = Composition(r.compound)
        key = comp.reduced_formula
        parts = [(float(x.split()[0]), x.split()[1]) for x in r.oxides.split(" + ")]
        names = [key] + [Composition(f).reduced_formula for _, f in parts]
        if not all(n in s for n in names):
            continue
        n_c = comp.num_atoms

        def reaction(value):
            return value(key) - sum(nu * Composition(f).num_atoms / n_c * value(Composition(f).reduced_formula)
                                    for nu, f in parts)

        dS = {t: reaction(lambda n: s[n]["S_per_mol_atoms"][t]) for t in TEMPS}
        dS1000_meas = reaction(lambda n: s_meas.get((n, "1000.0"), np.nan))
        imag = max(imag_of(s[n]) for n in names)
        converged = all(s[n]["converged"] for n in names)
        rows.append(dict(compound=r.compound, oxides=r.oxides, measured_dS298=r.dS298_per_atom,
                         measured_spread=r.source_spread_per_atom, harmonic_dS298=round(dS["298.15"], 3),
                         harmonic_dS1000=round(dS["1000.0"], 3), error_dS298=round(dS["298.15"] - r.dS298_per_atom, 3),
                         measured_dS1000=round(dS1000_meas, 3), error_dS1000=round(dS["1000.0"] - dS1000_meas, 3),
                         TdS1000_harmonic_meV=round(1000 * dS["1000.0"] * MEV_PER_J_MOL, 1),
                         TdS1000_dCp0_measured_meV=r.TdS_1000K_meV_atom, max_imaginary_fraction=imag,
                         all_phases_converged=converged))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "harmonic_reaction_entropy.csv", index=False)
    print(out.to_string(index=False))
    if len(out) > 1:
        for col, t in (("error_dS298", 298.15), ("error_dS1000", 1000.0)):
            e = out[col].dropna()
            print(f"n={len(e)}: harmonic minus measured dS at {t:g} K mean {e.mean():+.2f}, RMS "
                  f"{np.sqrt((e ** 2).mean()):.2f}, max |{e.abs().max():.2f}| J/(K mol atoms); as T dS, max "
                  f"{t * e.abs().max() * MEV_PER_J_MOL:.1f} meV per atom")
        dcp = out.TdS1000_harmonic_meV - out.TdS1000_dCp0_measured_meV
        print(f"heat-capacity change of reaction, harmonic T dS at 1000 K minus the dCp = 0 survey value: "
              f"{dcp.min():+.1f} to {dcp.max():+.1f} meV per atom")
    ph = pd.DataFrame(phases).dropna(subset=["S_measured"])
    for t, g in ph.groupby("T_K"):
        print(f"per phase, {t:g} K, n={len(g)}: harmonic minus measured mean {g.dS.mean():+.2f}, "
              f"range {g.dS.min():+.2f} to {g.dS.max():+.2f} J/(K mol atoms)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "sol" / "phonons")  # phonon_run.py outputs returned from Sol
