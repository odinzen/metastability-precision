"""Hermes entropies for the Sol MD sweep, set against measured entropies (Hermes from the checkout named by HERMES_DIR).

Reads every .npz of one sweep, takes S = classical Gaussian entropy + quantum correction (pilot_hermes.analyse), and joins
the measured value from reference_entropies.csv. The error is also given as T*dS in meV per atom, the currency of the
stability margins, so it can be read against the 22.5 meV per atom scatter of DFT against calorimetry. Reaction
entropies are formed where every phase of the reaction was run.

Usage: python analyse_sol.py [OUT_DIR [RESULTS_DIR]]   (defaults ../sol/out_inf and ../results)

sol/out_inf is the uncapped Orb-v3 sweep the paper reports. sol/out is the first sweep with the 20-neighbour model; its
tables get a _cap20 suffix so the two can be compared (compare_cap.py).
"""
import csv
import sys
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from pilot_hermes import analyse  # noqa: E402

J_MOL_PER_MEV_ATOM = 96.4853
SUFFIX = {"out_inf": "", "out": "_cap20"}
# Ternaries whose binary oxides are also in the sweep; per formula unit.
REACTIONS = {"MgTiO3": {"MgO": 1, "TiO2": 1}, "CaTiO3": {"CaO": 1, "TiO2": 1}, "CaZrO3": {"CaO": 1, "ZrO2": 1},
             "Mg2SiO4": {"MgO": 2, "SiO2": 1}, "ZrSiO4": {"ZrO2": 1, "SiO2": 1}, "La2Zr2O7": {"La2O3": 1, "ZrO2": 2}}


def run_index():
    with open(ROOT / "sol" / "runs" / "run_list.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # md tags drop parentheses and round the temperature, e.g. Mg3PO42_298K
    return {f"{r['compound'].replace('(', '').replace(')', '')}_{int(float(r['temperature_K']))}K":
            (r["compound"], float(r["temperature_K"])) for r in rows}


def main(out_dir, results_dir):
    suffix = SUFFIX[Path(out_dir).name]
    index = run_index()
    ref = pd.read_csv(HERE / "reference_entropies.csv")
    rows = []
    for path in sorted(Path(out_dir).glob("*.npz")):
        if path.stem not in index:
            print(f"skipping {path.name}: not in run_list.tsv", file=sys.stderr)
            continue
        compound, t = index[path.stem]
        try:
            r = analyse(str(path))
        except ValueError as e:  # e.g. a run too short or too disordered for Hermes to find the crystal
            print(f"FAILED {path.name}: {e}", file=sys.stderr)
            continue
        m = ref[(ref.compound == compound) & (abs(ref.T_K - t) < 0.5)]
        s_meas = float(m.S_per_mol_atoms.iloc[0]) if len(m) else float("nan")
        rows.append(dict(compound=compound, T_K=t, atoms_per_fu=Composition(compound).num_atoms,
                         S_hermes=r["S_hermes"], S_classical_gaussian=r["S_classical_gaussian"],
                         quantum_correction=r["quantum_correction"], S_harmonic_quantum_vdos=r["S_harmonic_quantum_vdos"],
                         S_measured=s_meas, dS=r["S_hermes"] - s_meas,
                         TdS_meV_atom=t * (r["S_hermes"] - s_meas) / J_MOL_PER_MEV_ATOM,
                         mean_T=r["mean_T"], volume_per_atom=r["volume_per_atom"],
                         frames=r["frames"], wall_hours=r["wall_hours"], hermes_commit=r["hermes_commit"], run=path.name))
    df = pd.DataFrame(rows)
    missing = sorted(set(index) - {Path(r["run"]).stem for r in rows})
    if missing:
        print(f"{len(missing)} runs not yet returned: {', '.join(missing)}", file=sys.stderr)
    if df.empty:
        print("no runs found", file=sys.stderr)
        return
    df.to_csv(Path(results_dir) / f"hermes_vs_measured{suffix}.csv", index=False)

    rx = []
    for target, parts in REACTIONS.items():
        for t in sorted(df[df.compound == target].T_K):
            sel = {c: df[(df.compound == c) & (df.T_K == t)] for c in [target, *parts]}
            if any(s.empty for s in sel.values()):
                continue
            n = sel[target].atoms_per_fu.iloc[0]

            def per_fu(c, col):
                return sel[c][col].iloc[0] * sel[c].atoms_per_fu.iloc[0]

            out = {}
            for col in ("S_hermes", "S_measured"):
                out[col] = (per_fu(target, col) - sum(nu * per_fu(c, col) for c, nu in parts.items())) / n
            rx.append(dict(reaction=target + " from " + " + ".join(parts), T_K=t, dS_rxn_hermes=out["S_hermes"],
                           dS_rxn_measured=out["S_measured"],
                           TdS_rxn_hermes_meV_atom=t * out["S_hermes"] / J_MOL_PER_MEV_ATOM,
                           TdS_rxn_measured_meV_atom=t * out["S_measured"] / J_MOL_PER_MEV_ATOM))
    if rx:
        pd.DataFrame(rx).to_csv(Path(results_dir) / f"hermes_reaction_entropies{suffix}.csv", index=False)

    with pd.option_context("display.width", 160):
        print(df[["compound", "T_K", "S_hermes", "S_measured", "dS", "TdS_meV_atom", "mean_T"]].round(2).to_string(index=False))
        if rx:
            print(pd.DataFrame(rx).round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "sol" / "out_inf", sys.argv[2] if len(sys.argv) > 2 else ROOT / "results")
