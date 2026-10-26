#!/bin/bash
# SLURM array body for the harmonic phonons (on a GPU when sol_config.sh sets PHONON_DEVICE=cuda): phase number SLURM_ARRAY_TASK_ID of phonon_structures/,
# in sorted order. Submit with submit.sh.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source sol_config.sh
# Same interpreter as the MD jobs, recorded by setup_env.sh; no conda activation in the job.
unset PYTHONPATH PYTHONHOME
PY="$(cat env_python.txt 2>/dev/null || true)"
if [ -z "$PY" ] || ! "$PY" -c "import orb_models, phonopy" 2>/dev/null; then
    echo "ERROR: no working interpreter with phonopy in env_python.txt ('$PY'); rerun setup_env.sh on a login node"; exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cif="$(ls phonon_structures/*.cif | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")"
if [ -z "$cif" ]; then echo "no phase at index $SLURM_ARRAY_TASK_ID"; exit 1; fi
phase="$(basename "$cif" .cif)"
echo "== $phase | host=$(hostname) start=$(date -Is) | cpus=$OMP_NUM_THREADS | python=$PY"

"$PY" phonon_run.py "$phase" "$OUTDIR"
rc=$?
echo "end=$(date -Is) rc=$rc"
[ $rc -eq 0 ] && echo "STATUS $phase: JOB DONE" || echo "STATUS $phase: FAILED rc=$rc"
exit $rc
