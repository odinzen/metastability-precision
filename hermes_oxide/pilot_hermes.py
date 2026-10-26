"""Hermes entropy for the MD runs (Hermes 0.2.0, from the checkout named by HERMES_DIR).

S = classical Gaussian entropy of the crystal (gaussian_entropy) + quantum correction from the velocity spectrum
(quantum_correction). Diagnostics alongside: the quantum and classical harmonic entropy of the same spectrum, so the
anharmonic content of the Gaussian route can be seen against a purely harmonic reading of the same run.

Usage: python pilot_hermes.py RUN.npz [RUN.npz ...] > out.json
"""
import json
import os
import subprocess
import sys

import numpy as np

HERMES = os.environ["HERMES_DIR"]
sys.path.insert(0, HERMES)
from hermes.compression_entropy import gaussian_entropy  # noqa: E402
from hermes.quantum_correction import harmonic_entropy, quantum_correction, vibrational_density_of_states  # noqa: E402

COMMIT = subprocess.run(["git", "-C", HERMES, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def analyse(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    t = meta["temperature"]
    g = gaussian_entropy(d["positions"], d["cell"], d["masses"], t, species=d["numbers"])
    # 5 ps segments for production runs; a short smoke run gets half its length so the spectrum can still be taken.
    segment = min(5.0, meta["prod_ps"] / 2)
    nu, dos = vibrational_density_of_states(d["velocities"], meta["vel_every_fs"] / 1000.0, masses_amu=d["masses"],
                                            segment_ps=segment)
    corr = quantum_correction(nu, dos, t)
    return dict(run=path, hermes_commit=COMMIT, model=meta["model"], temperature=t, n_atoms=meta["n_atoms"],
                frames=int(len(d["positions"])), S_classical_gaussian=g["entropy"], quantum_correction=corr,
                S_hermes=g["entropy"] + corr, S_harmonic_quantum_vdos=harmonic_entropy(nu, dos, t),
                S_harmonic_classical_vdos=harmonic_entropy(nu, dos, t, quantum=False),
                mean_T=meta["mean_temperature"], volume_per_atom=meta["volume_mean_A3"] / meta["n_atoms"],
                wall_hours=meta["wall_hours"])


if __name__ == "__main__":
    print(json.dumps([analyse(p) for p in sys.argv[1:]], indent=1))
