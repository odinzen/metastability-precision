"""MD for the Hermes oxide pilot (runs in WSL, ASE plus one MLIP).

Relax the cell at 0 K, find the mean volume at T with an isotropic MTK NPT run, then run Nose-Hoover chain NVT at that
volume and save positions (for Hermes's Gaussian entropy) and velocities (for the quantum correction from the
vibrational spectrum). Nose-Hoover rather than Langevin in production because friction would broaden the spectrum.

Usage: python pilot_md.py MODEL CIF REPS T OUT.npz [--npt-ps 10] [--equil-ps 5] [--prod-ps 40]
"""
import argparse
import json
import time

import numpy as np
from ase import units
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE


def calculator(name, precision="float32-high"):
    if name == "mace":
        from mace.calculators import mace_mp
        # float32 only with cuEquivariance: a float64 cueq run lost the GPU on 2026-10-08.
        return mace_mp(model="medium", default_dtype="float32", device="cuda", enable_cueq=True), "MACE-MP-0 medium"
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    loader = {"orb20": pretrained.orb_v3_conservative_20_omat, "orbinf": pretrained.orb_v3_conservative_inf_omat}[name]
    return ORBCalculator(loader(device="cuda", precision=precision), device="cuda"), f"{loader.__name__} {precision}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cif")
    ap.add_argument("reps", type=int)
    ap.add_argument("temperature", type=float)
    ap.add_argument("out")
    ap.add_argument("--npt-ps", type=float, default=10.0)
    ap.add_argument("--equil-ps", type=float, default=5.0)
    ap.add_argument("--prod-ps", type=float, default=40.0)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--pos-every-fs", type=float, default=100.0)
    ap.add_argument("--vel-every-fs", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()

    calc, label = calculator(a.model)
    atoms = read(a.cif).repeat((a.reps,) * 3)
    atoms.calc = calc
    t0 = time.time()

    def stage(name):
        print(f"[{(time.time() - t0) / 60:6.1f} min] {name}", flush=True)

    # Volume only: float32 force noise let a free cell drift off cubic (MgO came back I4/mmm), and these test
    # crystals are cubic at every temperature studied.
    FIRE(FrechetCellFilter(atoms, hydrostatic_strain=True), logfile=None).run(fmax=0.005, steps=500)
    stage("relaxed")
    a0 = float(np.cbrt(atoms.get_volume() / len(atoms)))
    relaxed_cell = atoms.cell.array.copy()

    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=rng)
    Stationary(atoms)
    dt = a.dt_fs * units.fs
    Langevin(atoms, dt, temperature_K=a.temperature, friction=0.02 / units.fs, rng=rng).run(1000)
    stage("Langevin warm-up done")
    npt = IsotropicMTKNPT(atoms, dt, temperature_K=a.temperature, pressure_au=0.0,
                          tdamp=100 * units.fs, pdamp=1000 * units.fs)
    volumes = []
    n_npt = int(a.npt_ps * 1000 / a.dt_fs)
    for i in range(n_npt // 10):
        npt.run(10)
        volumes.append(atoms.get_volume())
        if i == 9:
            stage(f"NPT: first 100 steps done, T = {atoms.get_temperature():.0f} K")
    v_mean = float(np.mean(volumes[len(volumes) // 2:]))
    stage(f"NPT done, mean volume per atom {v_mean / len(atoms):.4f} A^3")
    atoms.set_cell(relaxed_cell * np.cbrt(v_mean / abs(np.linalg.det(relaxed_cell))), scale_atoms=True)

    nvt = NoseHooverChainNVT(atoms, dt, temperature_K=a.temperature, tdamp=100 * units.fs)
    nvt.run(int(a.equil_ps * 1000 / a.dt_fs))
    stage("NVT equilibration done")
    pos_every, vel_every = int(a.pos_every_fs / a.dt_fs), int(a.vel_every_fs / a.dt_fs)
    n_prod = int(a.prod_ps * 1000 / a.dt_fs)
    positions, velocities, energies, temps = [], [], [], []

    # Observers on one continuous run: calling run() in short chunks repeats a force evaluation at each start.
    def record_velocities():
        velocities.append(atoms.get_velocities().astype(np.float32))

    def record_positions():
        positions.append(atoms.get_positions(wrap=False).astype(np.float64))
        energies.append(atoms.get_potential_energy())
        temps.append(atoms.get_temperature())

    nvt.attach(record_velocities, interval=vel_every)
    nvt.attach(record_positions, interval=pos_every)
    nvt.run(n_prod)
    # attach() also fires at step 0 of the run; drop that frame so every sample is from production proper.
    velocities, positions, energies, temps = velocities[1:], positions[1:], energies[1:], temps[1:]
    meta = dict(model=label, cif=a.cif, reps=a.reps, temperature=a.temperature, a0_relaxed_per_atom=a0,
                volume_mean_A3=v_mean, n_atoms=len(atoms), dt_fs=a.dt_fs, pos_every_fs=a.pos_every_fs,
                vel_every_fs=a.vel_every_fs, npt_ps=a.npt_ps, equil_ps=a.equil_ps, prod_ps=a.prod_ps,
                mean_temperature=float(np.mean(temps)), wall_hours=(time.time() - t0) / 3600, seed=a.seed)
    np.savez_compressed(a.out, positions=np.array(positions), velocities=np.array(velocities),
                        cell=atoms.cell.array, masses=atoms.get_masses(), numbers=atoms.numbers,
                        energies=np.array(energies), volumes=np.array(volumes), meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
