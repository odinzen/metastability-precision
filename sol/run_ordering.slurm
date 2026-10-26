#!/bin/bash
# SLURM array body for the ordering energies (one GPU): target row SLURM_ARRAY_TASK_ID of
# ordering_structures/targets.tsv. Submit with submit.sh.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source sol_config.sh
unset PYTHONPATH PYTHONHOME
PY="$(cat env_python.txt 2>/dev/null || true)"
if [ -z "$PY" ] || ! "$PY" -c "import orb_models" 2>/dev/null; then
    echo "ERROR: no working interpreter in env_python.txt ('$PY'); rerun setup_env.sh on a login node"; exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

tag="$(awk -F'\t' -v r="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==r { print $1 }' ordering_structures/targets.tsv)"
if [ -z "$tag" ]; then echo "no target at row $SLURM_ARRAY_TASK_ID"; exit 1; fi
echo "== $tag | host=$(hostname) start=$(date -Is) | python=$PY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" ordering_run.py "$tag" "$OUTDIR"
rc=$?
echo "end=$(date -Is) rc=$rc"
[ $rc -eq 0 ] && echo "STATUS $tag: JOB DONE" || echo "STATUS $tag: FAILED rc=$rc"
exit $rc
