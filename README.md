# metastability-precision

Code and data for *Calorimetry calibrates the precision of computed metastability in automated oxide synthesis*, by
Michael E. Bustamante and Kristina Lilova.

Metastability precision is the margin of a computed synthesis decision divided by the scatter of density functional
theory (DFT) against calorimetry, 22.5 meV per atom (43 for a target with an open-shell 3d or 4f cation). As a
probability it is calibrated for closed-shell oxides against decisions fixed by measured enthalpies, including measured
metastable polymorphs. This repository computes it for the precursor selections of Chen et al. (2024) and the target
stabilities of Chen et al. and the A-Lab (Szymanski et al., 2023), at 0 K and at the temperature each target was fired,
and for every experimentally observed oxide in the Materials Project.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Every analysis, from the Materials Project pull to the precision at the firing temperature |
| `hermes_oxide/` | Vibrational entropies from molecular dynamics (Hermes) and harmonic phonons, against measured entropies |
| `mlip_benchmark/` | Orb-v3 and MACE-MP-0 enthalpies of formation against calorimetry and DFT |
| `sol/` | Scripts, structures and run lists of the calculations run on the Sol supercomputer, with the phonon (`sol/phonons`) and cation-ordering (`sol/ordering`) results |
| `results/` | Every table the paper reports |
| `figures/` | The script that draws the figures from `results/` |
| `data/raw/` | Materials Project entries (release 2026.09.28), plus the third-party inputs that `fetch_inputs.py` downloads |

## Reproducing the analysis

Python 3.11 or later with the packages in `requirements.txt`. Then download the third-party inputs, each checked
against the checksum of the copy the paper used:

```
pip install -r requirements.txt
python fetch_inputs.py
```

The Materials Project entries the paper used ship in `data/raw/mp` and `data/raw/mp_r2scan`, so the analysis runs on
the same database release. `scripts/pull_mp_systems.py` pulls them again and would bring in a newer release. The
benchmark scripts, `scripts/sigma_by_reaction_type.py`, `scripts/polymorph_decisions.py` and `hermes_oxide/harmonic_structures.py` read the cached
entries but open a Materials Project connection, so they need an API key in `MP_API_KEY`;
`scripts/precision_mp_observed.py` queries every observed phase of the current release live.

The main steps and the tables they write:

| Step | Script | Writes |
|---|---|---|
| Reproduce the precursor selection on the current release | `scripts/rerun_chen_today.py` | `results/chen_rerun_today.csv` |
| DFT against calorimetry and against compiled enthalpies | `scripts/benchmark_dft_vs_measured.py`, `scripts/benchmark_compiled.py`, `scripts/benchmark_r2scan.py` | `results/benchmark_*.csv` |
| Machine-learned potentials against calorimetry | `mlip_benchmark/compare.py` | `results/mlip_vs_measured*.csv` |
| Metastability precision of the two campaigns, with sigma from `scripts/sigma_model.py` | `scripts/precision_analysis.py` | `results/precision_chen.csv`, `results/precision_alab.csv` |
| Measured metastable polymorphs against the database | `scripts/polymorph_decisions.py` | `results/polymorph_decisions.csv` |
| The same polymorph pairs with r2SCAN energies | `scripts/polymorph_r2scan.py` | `results/polymorph_r2scan.csv` |
| Calibration against measured decisions, and the error model | `scripts/precision_calibration.py` | `results/precision_calibration_*.csv`, `results/precision_error_model.csv`, `results/precision_sigma_by_family.csv`, `results/precision_expected_correct.csv` |
| Precision of every observed oxide (`--rescore` reapplies sigma to the saved release) | `scripts/precision_mp_observed.py` | `results/precision_mp_observed.csv` |
| Robustness of the precursor ranking | `scripts/chen_ranking_robustness.py` | `results/chen_ranking_robustness.csv` |
| Value of a calorimetric measurement | `scripts/measurement_value.py` | `results/measurement_value*.csv` |
| Cation disorder of the A-Lab targets | `scripts/configurational_bound.py`, `scripts/configurational_ordering.py` | `results/configurational_*.csv` |
| Entropies against measurement | `hermes_oxide/analyse_sol.py`, `hermes_oxide/harmonic_reaction_entropy.py` | `results/hermes_*.csv`, `results/harmonic_*.csv` |
| Molecular-dynamics entropies for step E | `scripts/md_entropy_step_e.py` | `results/md_entropy_step_e.csv` |
| Precision at the firing temperature | `scripts/precision_temperature.py` | `results/precision_temperature*.csv` |
| Figures | `figures/make_figures.py` | `figures/*.png`, `figures/*.pdf` |

The entropy scripts import Hermes (version 0.2.0, doi 10.5281/zenodo.23573417) from a checkout named by the
`HERMES_DIR` environment variable. They read the molecular-dynamics trajectories, which are deposited on Zenodo (about
16 GB; doi 10.5281/zenodo.23677025, https://zenodo.org/records/23677025) rather than kept here.

## Calculations on Sol

`sol/` holds what ran on the Sol supercomputer at Arizona State University: molecular dynamics (`md_run.py`, rows of
`runs/run_list.tsv`), harmonic phonons (`phonon_run.py`, `phonon_structures/`) and cation-ordering energies
(`ordering_run.py`, `ordering_structures/`), all with Orb-v3 (conservative, OMat24). `sol_config.sh` sets the SLURM
account and resources, `setup_env.sh` builds the environment from `env.yml`, and `submit.sh` submits only the runs
without a finished output.

## Measured data

The calorimetric enthalpies (`results/anchor_measured_raw.csv`) and the measured entropies
(`results/entropy_reference_raw.csv`, `results/reaction_entropy_measured.csv`, `hermes_oxide/reference_entropies.csv`)
and the measured polymorph enthalpies (`results/polymorph_measured.csv`) are values from the published literature. Each row names its source and, where the source is a table, the table and
page. They were extracted from the authors' provenance-tracked literature compilation, whose extraction scripts are not
part of this repository.

## Pre-registered predictions

Before any of them is measured, the enthalpies of formation of 30 oxides with no calorimetric measurement, and how
those measurements will settle 45 decisions of the two synthesis campaigns, were predicted and deposited on Zenodo on
2026-10-28 (doi 10.5281/zenodo.23741115, https://zenodo.org/records/23741115). The record's README explains how to
score the predictions against later measurements, and its `SHA256SUMS.txt` holds the checksums of the two prediction
files as written.

## Licenses

Code is under the MIT license (`LICENSE`). Data, results and figures are under CC BY 4.0 (`LICENSE-DATA`). The
Materials Project entries are distributed under the Materials Project's CC BY 4.0 terms; cite Jain et al. (2013) and
the database release. The inputs that `fetch_inputs.py` downloads keep the licenses of their sources.

## Citation

Please cite the article (details to follow on publication), the molecular-dynamics trajectories
(doi 10.5281/zenodo.23677025), the pre-registered predictions (doi 10.5281/zenodo.23741115) and, for the entropy
calculations, Hermes (doi 10.5281/zenodo.23573417).
