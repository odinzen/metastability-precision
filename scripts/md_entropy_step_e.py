"""Vibrational entropies of the step E MD runs (950 and 1250 K), for the decisions whose harmonic phonons are unreliable.

Each run gives two estimates. On the round-2 validation (results/hermes_vs_measured.csv, five reactions against
measurement at 1000 K) the harmonic entropy of the MD velocity spectrum reproduced the entropy of reaction with an RMS
error of 0.24 J/(K mol atoms), the Gaussian route of Hermes with 0.42. The spectrum route was therefore chosen before any
step E run was analysed; both are written so the choice can be checked. The spectrum comes from the anharmonic motion
at temperature, so it needs no harmonically stable crystal and has no imaginary modes to drop.

Hermes is imported from a copy of the commit that analysed round 2 (211b91c), set by HERMES_DIR, so the numbers are
comparable with that validation.

Usage: HERMES_DIR=<hermes 211b91c> python md_entropy_step_e.py MD_DIR   (writes results/md_entropy_step_e.csv)
"""
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERMES = Path(os.environ["HERMES_DIR"])
sys.path.insert(0, str(HERMES))
from hermes.compression_entropy import gaussian_entropy  # noqa: E402
from hermes.quantum_correction import harmonic_entropy, quantum_correction, vibrational_density_of_states  # noqa: E402

COMMIT = (HERMES / "HERMES_COMMIT").read_text().strip()
# The spectrum route needs atoms that vibrate about their sites. Hermes's own check (rms displacement under a quarter of
# the nearest-neighbour distance) also rejects the large librations of rigid PO4, BO3 and SiO4 groups, so the test here is
# diffusion: no species faster than 2e-4 A^2/ps, just above the largest value among runs that pass Hermes's check. Set
# before any decision was scored.
D_CRYSTAL = 2e-4


def run_index():
    with open(ROOT / "sol" / "runs" / "run_list.tsv", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if int(r["index"]) > 36]
    return {f"{re.sub(r'[()]', '', r['compound'])}_{int(float(r['temperature_K']))}K": (r["compound"], float(r["temperature_K"]))
            for r in rows}


def fastest_diffusion(positions, numbers, masses, dt_ps):
    """Largest self-diffusion coefficient over the species, A^2/ps, from the slope of the mean squared displacement
    between 5 and 20 ps of the unwrapped positions, centre-of-mass drift removed."""
    x = positions - ((positions * masses[None, :, None]).sum(1) / masses.sum())[:, None, :]
    lags = np.arange(1, len(x) // 2)
    t = lags * dt_ps
    fit = (t >= 5) & (t <= 20)
    worst = 0.0
    for s in np.unique(numbers):
        xs = x[:, numbers == s]
        msd = np.array([((xs[lag:] - xs[:-lag]) ** 2).sum(-1).mean() for lag in lags[fit]])
        worst = max(worst, np.polyfit(t[fit], msd, 1)[0] / 6)
    return worst


def analyse(path):
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        t = meta["temperature"]
        diffusion = fastest_diffusion(d["positions"].astype(float), d["numbers"], d["masses"].astype(float),
                                      meta.get("pos_every_fs", 100.0) / 1000.0)
        nu, dos = vibrational_density_of_states(d["velocities"], meta["vel_every_fs"] / 1000.0, masses_amu=d["masses"],
                                                segment_ps=min(5.0, meta["prod_ps"] / 2))
        # Hermes refuses a run whose atoms wander too far for a crystal (melted or diffusing); its spectrum is not a
        # crystal's either, so the run is kept in the table but marked and left out of the decisions.
        try:
            s_gauss = gaussian_entropy(d["positions"], d["cell"], d["masses"], t, species=d["numbers"])["entropy"]
            s_gauss += quantum_correction(nu, dos, t)
            solid, note = True, ""
        except ValueError as e:
            s_gauss, solid, note = np.nan, False, str(e)
    return dict(S_vdos=harmonic_entropy(nu, dos, t), S_gaussian=s_gauss, solid=solid, note=note,
                D_max_A2_ps=diffusion, crystal=bool(diffusion <= D_CRYSTAL),
                mean_T=meta["mean_temperature"], n_atoms=meta["n_atoms"], volume_per_atom=meta["volume_mean_A3"] / meta["n_atoms"])


def main(md_dir):
    index = run_index()
    paths = [p for p in sorted(Path(md_dir).glob("*.npz")) if p.stem in index]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(analyse, paths))
    rows = [dict(compound=index[p.stem][0], T_K=index[p.stem][1], **r, hermes_commit=COMMIT, run=p.name)
            for p, r in zip(paths, results)]
    out = pd.DataFrame(rows).sort_values(["compound", "T_K"])
    out.to_csv(ROOT / "results" / "md_entropy_step_e.csv", index=False)
    missing = sorted(set(index) - {p.stem for p in paths})
    print(f"{len(out)} of {len(index)} runs analysed; missing {missing}")
    print(f"mean T within {(out.mean_T - out.T_K).abs().max():.1f} K of set; S_vdos minus S_gaussian "
          f"{(out.S_vdos - out.S_gaussian).median():.2f} median J/(K mol atoms)")
    print(f"Hermes solidity check passed by {out.solid.sum()}; vibrating crystal by diffusion {out.crystal.sum()} "
          f"(950 K {out[out.T_K == 950].crystal.sum()}, 1250 K {out[out.T_K == 1250].crystal.sum()}); largest D among "
          f"runs passing Hermes's check {out[out.solid].D_max_A2_ps.max():.1e} A^2/ps")


if __name__ == "__main__":
    main(sys.argv[1])
