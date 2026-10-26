#!/bin/bash
# SLURM array body: one MD run per task, row SLURM_ARRAY_TASK_ID of runs/run_list.tsv. Submit with submit.sh.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source sol_config.sh
# No conda activation here: a job inherits the submitting shell's environment, and an env active there (another
# project's, from another conda install) broke activation for the whole first production array. setup_env.sh
# recorded the interpreter of the msp-orb env, and it is called directly.
unset PYTHONPATH PYTHONHOME
PY="$(cat env_python.txt 2>/dev/null || true)"
if [ -z "$PY" ] || ! "$PY" -c "import orb_models" 2>/dev/null; then
    echo "ERROR: no working interpreter in env_python.txt ('$PY'); rerun setup_env.sh on a login node"; exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

row=$((SLURM_ARRAY_TASK_ID + 1))   # skip the header
IFS=$'\t' read -r idx compound temp cif reps natoms mpid sg < <(awk -F'\t' -v r="$row" 'NR==r' runs/run_list.tsv)
if [ -z "${cif:-}" ]; then echo "no run at row $row"; exit 1; fi
tag="$(echo "$compound" | tr -d '()')_${temp%.*}K"
echo "== $tag | $cif $reps ($natoms atoms) T=$temp | host=$(hostname) start=$(date -Is) | python=$PY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" md_run.py "$cif" "$reps" "$temp" "$OUTDIR/${tag}.npz" \
    --npt-ps "$NPT_PS" --equil-ps "$EQUIL_PS" --prod-ps "$PROD_PS"
rc=$?
echo "end=$(date -Is) rc=$rc"
[ $rc -eq 0 ] && echo "STATUS $tag: JOB DONE" || echo "STATUS $tag: FAILED rc=$rc"
exit $rc
