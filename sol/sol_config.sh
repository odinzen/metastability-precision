# Sol settings for the Hermes oxide MD sweep and the phonon test. Every script sources this file; nothing else needs editing.
export ACCOUNT="your_allocation"     # SLURM account of your group allocation
export PARTITION="public"            # post-May-2026 Sol: public holds the publicly available nodes, GPUs included
export QOS="public"
export GPU_GRES="gpu:a100:1"         # one A100 per run
export CPUS=8
export MEM="64G"
export TIME="16:00:00"               # generous; a run is about 56 ps of MD on 400-700 atoms, slower uncapped

# Harmonic phonon test: CPU only, one phase per task
# The 2026-10-24 test ran on CPUs. The temperature-aware precision needs 253 more phases, up to 312 atoms, so they
# run on one A100 each in double precision, same protocol (12 rotations, 0.01 and 0.03 A, 10 A supercells).
export PHONON_DEVICE="cuda"
export PHONON_GRES="$GPU_GRES"
export PHONON_CPUS=8
export PHONON_MEM="64G"
export PHONON_TIME="24:00:00"

# Rerun of 2026-10-24: the uncapped Orb-v3 model, outputs to their own folder so the first sweep stays for comparison.
export MD_MODEL="orb_v3_conservative_inf_omat"
export MD_OUT="out_inf"
export PHONON_MODEL="orb_v3_conservative_inf_omat"

# Ordering energies of the disordered A-Lab targets (one GPU per target, 25 relaxations of 80 to 160 atoms)
export ORDERING_MODEL="orb_v3_conservative_inf_omat"
export ORDERING_TIME="06:00:00"

export MAMBA_MODULE="mamba/latest"
export ENV_NAME="msp-orb"

# Production MD lengths (ps). Defaults only, so the smoke values submit.sh exports survive the re-source in the job.
export NPT_PS="${NPT_PS:-10}"
export EQUIL_PS="${EQUIL_PS:-5}"
export PROD_PS="${PROD_PS:-40}"
