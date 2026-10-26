"""How much the 20-neighbour cap of the first Orb-v3 sweep moved the Hermes entropies: the uncapped sweep (the paper's
numbers) minus the capped one, per phase and per reaction, as T dS in meV per atom.

Reads results/hermes_vs_measured{,_cap20}.csv and results/hermes_reaction_entropies{,_cap20}.csv from analyse_sol.py;
writes results/hermes_cap_effect.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
J_MOL_PER_MEV_ATOM = 96.4853


def main(res=RES):
    keys = ["compound", "T_K"]
    a = pd.read_csv(res / "hermes_vs_measured.csv")
    b = pd.read_csv(res / "hermes_vs_measured_cap20.csv")
    ph = a[keys + ["S_hermes", "S_measured"]].merge(b[keys + ["S_hermes"]], on=keys, suffixes=("", "_cap20"))
    ph["kind"] = "phase"

    ra = pd.read_csv(res / "hermes_reaction_entropies.csv")
    rb = pd.read_csv(res / "hermes_reaction_entropies_cap20.csv")
    rx = (ra.rename(columns={"reaction": "compound", "dS_rxn_hermes": "S_hermes", "dS_rxn_measured": "S_measured"})
          [keys + ["S_hermes", "S_measured"]]
          .merge(rb.rename(columns={"reaction": "compound", "dS_rxn_hermes": "S_hermes_cap20"})[keys + ["S_hermes_cap20"]],
                 on=keys))
    rx["kind"] = "reaction"

    d = pd.concat([ph, rx], ignore_index=True)
    d["dS_cap"] = d.S_hermes - d.S_hermes_cap20
    d["TdS_cap_meV_atom"] = d.T_K * d.dS_cap / J_MOL_PER_MEV_ATOM
    for tag, col in (("", "S_hermes"), ("_cap20", "S_hermes_cap20")):
        d[f"TdS_error{tag}_meV_atom"] = d.T_K * (d[col] - d.S_measured) / J_MOL_PER_MEV_ATOM
    d.to_csv(res / "hermes_cap_effect.csv", index=False)

    with pd.option_context("display.width", 160):
        print(d[["kind", "compound", "T_K", "dS_cap", "TdS_cap_meV_atom", "TdS_error_cap20_meV_atom",
                 "TdS_error_meV_atom"]].round(2).to_string(index=False))
    for (kind, t), g in d.groupby(["kind", "T_K"]):
        e0, e1 = g.TdS_error_cap20_meV_atom.dropna(), g.TdS_error_meV_atom.dropna()
        print(f"{kind:8s} {t:7.2f} K  n={len(g):2d}  cap effect mean {g.TdS_cap_meV_atom.mean():+6.1f}, "
              f"max |{g.TdS_cap_meV_atom.abs().max():.1f}|  error RMS capped {np.sqrt((e0 ** 2).mean()):5.1f} -> "
              f"uncapped {np.sqrt((e1 ** 2).mean()):5.1f} meV per atom")


if __name__ == "__main__":
    main()
