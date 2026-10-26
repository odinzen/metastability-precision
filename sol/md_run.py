"""One MD run for the Hermes oxide sweep on Sol (one GPU): relax at 0 K keeping the space group, fix the volume at T
with isotropic MTK NPT, then Nose-Hoover chain NVT production saving positions (for Hermes's Gaussian entropy) and
velocities (for the quantum correction). Same output format as hermes_oxide/pilot_md.py, so the local Hermes
analysis reads both.

Usage: python md_run.py CIF REPS T OUT.npz [--npt-ps 10] [--equil-ps 5] [--prod-ps 40]
REPS is e.g. 4x4x4. A finished OUT.npz is never overwritten, so a resubmitted array skips completed runs.
"""
import argparse
import inspect
import json
import os
import socket
import time

import numpy as np
from ase import units
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# The first sweep used the 20-neighbour model. Its cap cuts through degenerate neighbour shells, which breaks the
# crystal's symmetry and makes the energy jump when a small displacement reorders neighbours; the uncapped model
# does neither (Na2O phonons: residual force 0.03 -> 0.0006 eV/A). sol_config.sh sets MD_MODEL for the rerun.
DEFAULT_MODEL = "orb_v3_conservative_20_omat"
MODEL = os.environ.get("MD_MODEL", DEFAULT_MODEL)
HERE = os.path.dirname(os.path.abspath(__file__))
RELAX_FMAX, RELAX_STEPS = 0.005, 800


def load_model(device, compile=False):
    """Orb-v3 from weights/ when setup_env.sh has put it there, else from its URL; returns (model, source)."""
    # Given the URL, orb-models checks it online on every load, which on Sol's offline compute nodes ran into
    # 90 minutes of timeouts.
    url = inspect.signature(getattr(pretrained, MODEL)).parameters["weights_path"].default
    local = os.path.join(HERE, "weights", url.rsplit("/", 1)[1])
    weights = local if os.path.exists(local) else url
    model = getattr(pretrained, MODEL)(weights_path=weights, device=device, precision="float32-high", compile=compile)
    return model, ("weights/" if weights == local else "the network")


def relaxed_path(cif, reps):
    # Cells relaxed with one model are not reused with another.
    folder = "relaxed" if MODEL == DEFAULT_MODEL else "relaxed_" + MODEL.split("_")[3]
    return os.path.join(HERE, folder, f"{os.path.splitext(os.path.basename(cif))[0]}_{reps}.extxyz")


def relax(atoms):
    """0 K relaxation of cell and positions in place; returns (steps, largest residual force in eV/A)."""
    # Keep the space group while relaxing the cell: float32 noise otherwise drifts a free cell off its symmetry
    # (MgO came back tetragonal in the pilot), and a volume-only relaxation is wrong for the non-cubic crystals.
    atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
    fmax = float(np.abs(atoms.get_forces()).max())
    atoms.set_constraint()
    return opt.nsteps, fmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("reps")
    ap.add_argument("temperature", type=float)
    ap.add_argument("out")
    ap.add_argument("--npt-ps", type=float, default=10.0)
    ap.add_argument("--equil-ps", type=float, default=5.0)
    ap.add_argument("--prod-ps", type=float, default=40.0)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--pos-every-fs", type=float, default=100.0)
    ap.add_argument("--vel-every-fs", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--device", default="cuda")
    # torch.compile is off by default: on Sol GPU nodes its compile workers died ("Attempting to use a closed pool").
    # Forces are the same either way; compiling only makes the steps faster.
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"{a.out} exists, skipping", flush=True)
        return

    t0 = time.time()

    def stage(name):
        print(f"[{(time.time() - t0) / 60:7.1f} min] {name}", flush=True)

    model, source = load_model(a.device, a.compile)
    stage(f"model loaded from {source}")
    calc = ORBCalculator(model, device=a.device)
    # The relaxation is done ahead of time (prerelax.py, same model and settings) when relaxed/ holds this supercell:
    # on Sol's GPU nodes this step alone ran for over 90 minutes, against about a minute on a laptop.
    pre = relaxed_path(a.cif, a.reps)
    if os.path.exists(pre):
        atoms = read(pre)
        atoms.calc = calc
        relaxed_from = os.path.relpath(pre, HERE)
    else:
        atoms = read(a.cif).repeat(tuple(int(x) for x in a.reps.split("x")))
        atoms.calc = calc
        relax(atoms)
        relaxed_from = "this run"
    relaxed_cell = atoms.cell.array.copy()
    a0 = float(np.cbrt(atoms.get_volume() / len(atoms)))
    stage(f"relaxed ({relaxed_from}), {len(atoms)} atoms, volume per atom {atoms.get_volume() / len(atoms):.4f} A^3")

    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=rng)
    Stationary(atoms)
    dt = a.dt_fs * units.fs
    Langevin(atoms, dt, temperature_K=a.temperature, friction=0.02 / units.fs, rng=rng).run(1000)
    stage("Langevin warm-up done")
    # Isotropic volume scaling: anisotropic thermal expansion of the non-cubic crystals is not followed, a small
    # effect on the entropy compared with the volume itself.
    npt = IsotropicMTKNPT(atoms, dt, temperature_K=a.temperature, pressure_au=0.0,
                          tdamp=100 * units.fs, pdamp=1000 * units.fs)
    volumes = []
    for _ in range(int(a.npt_ps * 1000 / a.dt_fs) // 10):
        npt.run(10)
        volumes.append(atoms.get_volume())
    v_mean = float(np.mean(volumes[len(volumes) // 2:]))
    atoms.set_cell(relaxed_cell * np.cbrt(v_mean / abs(np.linalg.det(relaxed_cell))), scale_atoms=True)
    stage(f"NPT done, mean volume per atom {v_mean / len(atoms):.4f} A^3")

    nvt = NoseHooverChainNVT(atoms, dt, temperature_K=a.temperature, tdamp=100 * units.fs)
    nvt.run(int(a.equil_ps * 1000 / a.dt_fs))
    stage("NVT equilibration done")
    positions, velocities, energies, temps = [], [], [], []

    def record_velocities():
        velocities.append(atoms.get_velocities().astype(np.float32))

    def record_positions():
        positions.append(atoms.get_positions(wrap=False).astype(np.float64))
        energies.append(atoms.get_potential_energy())
        temps.append(atoms.get_temperature())

    # Observers on one continuous run: short run() chunks repeat a force evaluation at each start.
    nvt.attach(record_velocities, interval=int(a.vel_every_fs / a.dt_fs))
    nvt.attach(record_positions, interval=int(a.pos_every_fs / a.dt_fs))
    nvt.run(int(a.prod_ps * 1000 / a.dt_fs))
    # attach() also fires at step 0; drop that frame so every sample is from production.
    velocities, positions, energies, temps = velocities[1:], positions[1:], energies[1:], temps[1:]
    stage("production done")

    meta = dict(model=MODEL + " float32-high", compiled=a.compile, cif=a.cif, reps=a.reps, temperature=a.temperature,
                a0_relaxed_per_atom=a0, relaxed_from=relaxed_from, volume_mean_A3=v_mean, n_atoms=len(atoms),
                dt_fs=a.dt_fs, pos_every_fs=a.pos_every_fs, vel_every_fs=a.vel_every_fs, npt_ps=a.npt_ps,
                equil_ps=a.equil_ps,
                prod_ps=a.prod_ps, mean_temperature=float(np.mean(temps)), wall_hours=(time.time() - t0) / 3600,
                seed=a.seed, host=socket.gethostname())
    tmp = a.out + ".partial.npz"
    np.savez_compressed(tmp, positions=np.array(positions), velocities=np.array(velocities),
                        cell=atoms.cell.array, masses=atoms.get_masses(), numbers=atoms.numbers,
                        energies=np.array(energies), volumes=np.array(volumes), meta=json.dumps(meta))
    os.replace(tmp, a.out)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
