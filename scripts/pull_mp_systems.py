"""Pull every Materials Project entry in each automated-synthesis target's chemical system and record, per target,
today's energy above the hull and the stable competing phases.

The key is read from MP_API_KEY, else from the LLZO project's untracked key file. It is never printed or copied.
"""
import gzip
import json
import os
from pathlib import Path

import pandas as pd
from emmet.core.mpid import AlphaID
from monty.json import MontyEncoder
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
MP_DIR = ROOT / "data" / "raw" / "mp"


def api_key() -> str:
    return os.environ["MP_API_KEY"]  # needed only to pull entries again; data/raw/mp holds release 2026.09.28


def targets() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "coverage_compounds.csv")
    t = df[df.role == "target"][["dataset", "formula"]].copy()
    alab = pd.read_csv(ROOT / "data" / "raw" / "alab2023" / "unzipped" / "20230502 Synthesis Results with Recipes.csv",
                       encoding="latin-1")
    t["paper_mpid"] = t.formula.map(dict(zip(alab.Target, alab["Materials Project ID"])))
    t["chemsys"] = t.formula.map(lambda f: "-".join(sorted(str(e) for e in Composition(f).elements)))
    return t


def mp_number(entry) -> int:
    """Integer MP id. MP now serves alpha ids (mp-aaaabphfn); the papers quote the old numeric form (mp-725491)."""
    eid = entry.entry_id
    ident = eid["identifier"] if isinstance(eid, dict) else getattr(eid, "identifier", str(eid))
    return int(AlphaID(ident))


def load_or_pull(mpr: MPRester, chemsys: str) -> list:
    path = MP_DIR / f"{chemsys}.json.gz"
    if path.exists():
        from monty.json import MontyDecoder
        return MontyDecoder().process_decoded(json.load(gzip.open(path)))
    entries = mpr.get_entries_in_chemsys(chemsys.split("-"), additional_criteria={"thermo_types": ["GGA_GGA+U"]})
    for e in entries:
        # MP returns oxidation_states keyed by Element objects, which JSON cannot hold.
        if isinstance(e.data.get("oxidation_states"), dict):
            e.data["oxidation_states"] = {str(k): v for k, v in e.data["oxidation_states"].items()}
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(entries, f, cls=MontyEncoder)
    tmp.replace(path)
    return entries


def main():
    MP_DIR.mkdir(parents=True, exist_ok=True)
    t = targets()
    rows = []
    with MPRester(api_key()) as mpr:
        db_version = mpr.db_version
        for chemsys in sorted(t.chemsys.unique()):
            entries = load_or_pull(mpr, chemsys)
            pd_ = PhaseDiagram(entries)
            for _, r in t[t.chemsys == chemsys].iterrows():
                red = Composition(r.formula).reduced_formula
                match = [e for e in entries if e.composition.reduced_formula == red]
                by_id = ([e for e in match if mp_number(e) == int(AlphaID(r.paper_mpid))]
                         if isinstance(r.paper_mpid, str) else [])
                best = by_id[0] if by_id else (min(match, key=pd_.get_e_above_hull) if match else None)
                stable_inside = [e.composition.reduced_formula for e in pd_.stable_entries if len(e.composition.elements) > 1]
                rows.append(dict(
                    dataset=r.dataset, target=r.formula, chemsys=chemsys, n_entries=len(entries),
                    mp_entry=f"mp-{mp_number(best)}" if best else None, paper_mpid=r.paper_mpid,
                    paper_mpid_found=bool(by_id),
                    n_polymorphs_in_mp=len(match),
                    e_above_hull_meV=round(1000 * pd_.get_e_above_hull(best), 2) if best else None,
                    stable_competing_phases=";".join(sorted(set(stable_inside) - {red})),
                ))
            print(f"{chemsys}: {len(entries)} entries", flush=True)
    out = pd.DataFrame(rows)
    out["mp_database_version"] = db_version
    out.to_csv(ROOT / "results" / "mp_targets_today.csv", index=False)
    print("MP database", db_version)


if __name__ == "__main__":
    main()
