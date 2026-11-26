# metastability-precision

Code and data for *Integrating calorimetry and computation to quantify metastability in materials discovery*, by
Michael E. Bustamante and Kristina Lilova.

Metastability precision is the margin of a computed stability relation divided by the scatter of density functional
theory (DFT) against calorimetry. The scatter depends on the regime of the relation (`scripts/sigma_model.py`): 22.5
meV per atom for closed-shell compounds and 34 for a target with an open-shell 3d cation or an open 4f shell of Ce, Eu
or Gd. Relations between polymorphs of one composition carry a systematic error and get no Gaussian probability.
Intermetallic relations follow a Student's *t* distribution with nu = 1.5 and scale 46 meV per atom
(`scripts/trust_scale.py`). The regimes are calibrated against relations whose sign is fixed by measured enthalpies,
including measured metastable polymorphs and compounds, and tested on a blind holdout. The repository applies them to
the precursor selections of Chen et al. (2024) and the target stabilities of Chen et al. and the A-Lab (Szymanski et
al., 2023), at 0 K and at the temperature each target was fired, to every experimentally observed oxide in the
Materials Project and to the oxides GNoME predicts stable (Merchant et al., 2023), and it ranks the calorimetric
measurements that would settle the most relations.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Every analysis, from the Materials Project pull to the precision at the firing temperature |
| `hermes_oxide/` | Vibrational entropies from molecular dynamics (Hermes) and harmonic phonons, against measured entropies |
| `mlip_benchmark/` | Orb-v3 and MACE-MP-0 enthalpies of formation against calorimetry and DFT |
| `sol/` | Scripts, structures and run lists of the calculations run on the Sol supercomputer, with the phonon (`sol/phonons`) and cation-ordering (`sol/ordering`) results |
| `results/` | Every table the paper reports, and the measured values the analysis reads |
| `figures/` | `make_figures.py`, which draws Figs. 1 to 5 and Extended Data Figs. 1 to 5 from `results/` |
| `data/curated/` | Hand-curated inputs: reactions and phase-diagram status of the measured metastable compounds, the primary calorimetry that audits the compiled relations, and the compounds found to have published enthalpies |
| `data/thermal_correction/` | Materials Project structures for the 0 K to 298 K enthalpy check |
| `data/raw/` | Materials Project entries (release 2026.09.28), plus the third-party inputs that `fetch_inputs.py` downloads |

## Reproducing the analysis

Python 3.11 or later with the packages in `requirements.txt`. Then download the third-party inputs, each checked
against the checksum of the copy the paper used:

```
pip install -r requirements.txt
python fetch_inputs.py
```

The Materials Project entries the paper used ship in `data/raw/mp` and `data/raw/mp_r2scan`, so the analysis runs on
the same database release. `scripts/pull_mp_systems.py` pulls them again and would bring in a newer release. Every
script that opens a Materials Project connection needs an API key in `MP_API_KEY`, even where every entry it reads is
cached. These read only the cached entries: `benchmark_dft_vs_measured.py`, `benchmark_nonoxide.py`,
`calibration_nonoxide.py`, `emf_hulls.py`, `metastable_dft.py`, `reaction_volume.py` and `polymorph_r2scan.py`. These
also query the current release, so a rerun can move with it: `pull_mp_systems.py` (the database version),
`benchmark_compiled.py`, `benchmark_r2scan.py`, `sigma_by_reaction_type.py`, `calibration_baselines.py`,
`calibration_compiled_nonoxide.py`, `polymorph_decisions.py`, `polymorph_density_candidates.py`,
`thermal_correction_structures.py`, `measurement_value_observed.py`, `hermes_oxide/harmonic_structures.py` and
`sol/make_structures.py`. `precision_mp_observed.py` queries every observed phase of the current release, and
`--rescore` reapplies sigma to the saved table instead.

Set `PYTHONHASHSEED=0` before the scripts that rank precursors with the selection code of Chen et al.
(`rerun_chen_today.py`, then `precision_analysis.py`, `calorimetric_correction.py` and `chen_ranking_robustness.py`).
Precursor pairs that score the same are ranked in the hash order of the Python run, so without a fixed seed which pair
ranks first can change for LiMnPO4 and KTiNbO5. Seed 0 reproduces `chen_rerun_today.csv`, `precision_chen.csv` and `correction_chen.csv`. The
`p_top` column of `chen_ranking_robustness.csv` (the share of draws in which the tested pair ranks first) moves by up
to 0.09 between runs; its `p_selected` column does not.

The steps in the order they depend on each other, and the tables they write:

| Step | Script | Writes |
|---|---|---|
| Reproduce the precursor selection on the current release | `scripts/rerun_chen_today.py` | `results/chen_rerun_today.csv` |
| Targets, their chemical systems and the measured compounds in them | `scripts/pull_mp_systems.py`, `scripts/coverage_step1.py` | `results/mp_targets_today.csv`, `results/coverage_*.csv` |
| DFT against calorimetry, compiled enthalpies and r2SCAN | `scripts/benchmark_dft_vs_measured.py`, `scripts/benchmark_compiled.py`, `scripts/benchmark_r2scan.py` | `results/benchmark_dft_vs_measured.csv`, `results/benchmark_summary.csv`, `results/benchmark_compiled.csv`, `results/benchmark_r2scan_*.csv` |
| Machine-learned potentials against calorimetry | `mlip_benchmark/compare.py` | `results/mlip_vs_measured*.csv` |
| Compiled stability relations, and their audit with primary calorimetry | `scripts/sigma_by_reaction_type.py`, `scripts/truth_audit.py` | `results/sigma_by_reaction_type.csv`, `results/truth_audit.csv` |
| Metastability precision of the two campaigns | `scripts/precision_analysis.py` | `results/precision_chen.csv`, `results/precision_alab.csv`, `results/precision_summary.csv` |
| Measured metastable polymorphs against the database, with GGA/GGA+U and r2SCAN | `scripts/polymorph_decisions.py`, `scripts/polymorph_r2scan.py` | `results/polymorph_decisions.csv`, `results/polymorph_r2scan.csv` |
| Polymorph ground states the database probably gets wrong | `scripts/polymorph_density_candidates.py` | `results/polymorph_density_candidates.csv` |
| Precision of every observed oxide | `scripts/precision_mp_observed.py` | `results/precision_mp_observed.csv` |
| First calibration and the error model | `scripts/precision_calibration.py` | `results/precision_calibration_*.csv`, `results/precision_error_model.csv`, `results/precision_sigma_by_family.csv`, `results/precision_expected_correct.csv` |
| Calibration by regime with audited outcomes, and its checks | `scripts/calibration_regimes.py`, `scripts/calibration_audit.py`, `scripts/calibration_base_rate.py`, `scripts/calibration_baselines.py` | `results/calibration_regimes*.csv`, `results/calibration_audit.csv`, `results/calibration_base_rate*.csv`, `results/calibration_baselines.csv`, `results/calibration_two_sided.csv` |
| Sulfides, selenides, tellurides and nitrides | `scripts/benchmark_nonoxide.py`, `scripts/calibration_nonoxide.py`, `scripts/emf_hulls.py`, `scripts/calibration_compiled_nonoxide.py` | `results/benchmark_nonoxide*.csv`, `results/calibration_nonoxide*.csv`, `results/emf_hull_decisions.csv`, `results/calibration_compiled_nonoxide*.csv` |
| Intermetallic regime | `scripts/calibration_intermetallic.py` | `results/calibration_intermetallic.csv`, `results/calibration_intermetallic_t.csv` |
| Measured metastable compounds, scored from the measured side, and the reaction-volume test | `scripts/metastable_dft.py`, `scripts/metastable_side.py`, `scripts/reaction_volume.py` | `results/metastable_dft.csv`, `results/metastable_side*.csv`, `results/metastable_intermetallic_*.csv`, `results/metastable_threshold_sensitivity.csv`, `results/reaction_volume.csv` |
| Value of a calorimetric measurement for the campaigns, and its test on known answers | `scripts/measurement_value.py`, `scripts/measurement_priority.py`, `scripts/measurement_design_ladder.py` | `results/measurement_value*.csv`, `results/measurement_priority.csv`, `results/measurement_ladder*.csv`, `results/measurement_blind_literature.csv`, `results/measurement_real_entry.csv` |
| Class-dependent error model and the pre-registered predictions | `scripts/preregistration_model.py`, `scripts/preregistration_power.py` | `results/preregistration_model.json`, `results/preregistration_*validation.csv`, `results/preregistered_*.csv`, `results/preregistration_power.csv` |
| GNoME oxides and the selection check | `scripts/gnome_functional_offset.py`, `scripts/gnome_precision.py` (then `gnome_functional_offset.py` again) | `results/gnome_functional_offset.csv`, `results/gnome_precision_summary.csv`, `results/gnome_precision_r2scan.csv`, `results/gnome_selection_check.csv` |
| Headline shares under the class-dependent model, the trust scale, and the open-shell sensitivity | `scripts/headline_family_aware.py`, `scripts/trust_scale.py`, `scripts/open_shell_sensitivity.py` | `results/headline_family_aware.csv`, `results/trust_scale.csv`, `results/open_shell_sensitivity.csv` |
| Forward predictions for the known oxides | `scripts/measurement_value_observed.py` | `results/measurement_value_observed.csv`, `results/forward_predictions_observed_*.csv` |
| Cation disorder of the A-Lab targets | `scripts/configurational_bound.py`, `scripts/configurational_ordering.py` | `results/configurational_*.csv` |
| Entropies against measurement | `hermes_oxide/analyse_sol.py`, `hermes_oxide/compare_cap.py`, `hermes_oxide/harmonic_reaction_entropy.py` | `results/hermes_*.csv`, `results/harmonic_*.csv` |
| Molecular-dynamics entropies for the relations whose phonons are unreliable | `scripts/md_entropy_step_e.py` | `results/md_entropy_step_e.csv` |
| Precision at the firing temperature | `scripts/precision_temperature.py` | `results/precision_temperature*.csv` |
| Robustness of the precursor ranking, outcome tests and their power | `scripts/chen_ranking_robustness.py`, `scripts/outcome_effect_sizes.py`, `scripts/alab_outcome_power.py` | `results/chen_ranking_robustness.csv`, `results/alab_outcome_power.csv` |
| The database decisions with a calorimetric correction applied | `scripts/calorimetric_correction.py` | `results/correction_*.csv` |
| 0 K energy against 298 K enthalpy | `scripts/thermal_correction_structures.py`, `scripts/thermal_correction.py` | `data/thermal_correction/`, `results/thermal_correction*.csv`, `results/thermal_correction/` |
| Figures | `figures/make_figures.py` | `figures/*.png`, `figures/*.pdf` |

`measurement_design_ladder.py`, `preregistration_model.py` and `measurement_value_observed.py` also rewrite the
deposited prediction files (`results/preregistered_*.csv`, `results/forward_predictions_observed_*.csv`) and their
checksums. The copies here are the deposited ones; compare a rerun against them rather than committing it.

The entropy scripts import Hermes (version 0.2.0, doi 10.5281/zenodo.23573417) from a checkout named by the
`HERMES_DIR` environment variable. They read the molecular-dynamics trajectories, which are deposited on Zenodo (about
16 GB; doi 10.5281/zenodo.23677025, https://zenodo.org/records/23677025) rather than kept here.
`scripts/thermal_correction.py` runs harmonic phonons with Orb-v3, and with MACE-MPA-0 for the silicide reactions.

## Calculations on Sol

`sol/` holds what ran on the Sol supercomputer at Arizona State University: molecular dynamics (`md_run.py`, rows of
`runs/run_list.tsv`), harmonic phonons (`phonon_run.py`, `phonon_structures/`) and cation-ordering energies
(`ordering_run.py`, `ordering_structures/`), all with Orb-v3 (conservative, OMat24). `sol_config.sh` sets the SLURM
account and resources, `setup_env.sh` builds the environment from `env.yml`, and `submit.sh` submits only the runs
without a finished output.

## Measured data

The measured values the analysis reads are from the published literature, and each row names its source and, where
the source is a table, the table and page:

- calorimetric enthalpies of the oxide benchmark, `results/anchor_measured_raw.csv`
- enthalpies of sulfides, selenides, tellurides and nitrides, calorimetric and from EMF studies, with the evaluated
  binary values they are combined with, `results/nonoxide_measured_raw.csv`
- primary calorimetry used to audit the compiled relations, `results/primary_truth_raw.csv` (chosen by
  `data/curated/primary_truth.csv`)
- measured polymorph enthalpies, `results/polymorph_measured.csv`, and the literature verdicts on polymorph ground
  states, `results/polymorph_adjudication.csv`
- measured metastable compounds and their competing phases, `data/curated/metastable_additions.csv` and
  `data/curated/metastable_reactions.csv`
- measured entropies, `results/entropy_reference_raw.csv`, `results/reaction_entropy_measured.csv` and
  `hermes_oxide/reference_entropies.csv`

They were extracted from the authors' provenance-tracked literature compilation. Its extraction scripts are not part of
this repository, and neither are the three analyses that read the compilation directly; their tables ship instead.
These are the intermetallic benchmark on the data release of Kim et al. (2017)
(`results/benchmark_nonoxide_calorimetric*.csv`), the blind holdout (`results/holdout_test.csv`) and the measured
entropies of reaction (`results/reaction_entropy_measured.csv`).

## GNoME

The GNoME summaries are under CC BY-NC 4.0, so this repository does not redistribute them or any row-level table made
from them. `fetch_inputs.py` downloads them, `scripts/gnome_precision.py` writes the row-level table
`results/gnome_precision.csv` (not tracked), and only its aggregate summaries ship.

## Deposited predictions

Three records on Zenodo hold what is not kept here or was fixed before measurement:

- molecular-dynamics trajectories, doi 10.5281/zenodo.23677025
- pre-registered predictions for the two synthesis campaigns, deposited on 2026-10-28 before any of the 30 oxides was
  measured, doi 10.5281/zenodo.23741115
- forward predictions for 29 observed oxides with no published enthalpy of formation, doi 10.5281/zenodo.24237166

Each prediction record has a README that explains how to score the predictions against later measurements and a
`SHA256SUMS.txt` with the checksums of the prediction files as written. The same files are in `results/` with their
`.sha256` checksums, stored byte for byte (`.gitattributes`).

## Licenses

Code is under the MIT license (`LICENSE`). Data, results and figures are under CC BY 4.0 (`LICENSE-DATA`). The
Materials Project entries are distributed under the Materials Project's CC BY 4.0 terms; cite Jain et al. (2013) and
the database release. The inputs that `fetch_inputs.py` downloads keep the licenses of their sources.

## Citation

Please cite the article (details to follow on publication), the molecular-dynamics trajectories
(doi 10.5281/zenodo.23677025), the pre-registered predictions (doi 10.5281/zenodo.23741115), the forward predictions
(doi 10.5281/zenodo.24237166) and, for the entropy calculations, Hermes (doi 10.5281/zenodo.23573417).
