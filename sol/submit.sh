#!/bin/bash
# Submit the runs as SLURM arrays, from the package folder on Sol. Only runs without a finished output are submitted.
#   bash submit.sh              # all: the missing MD runs (GPU), phonon runs (CPU) and ordering runs (GPU)
#   bash submit.sh md           # missing MD runs of runs/run_list.tsv
#   bash submit.sh md 3,7       # MD rows 3 and 7 regardless (e.g. a rerun after deleting their outputs)
#   bash submit.sh phonons      # missing phonon runs, one per file in phonon_structures/
#   bash submit.sh ordering     # missing ordering-energy runs, one per row of ordering_structures/targets.tsv
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
source sol_config.sh
# Results and logs live beside this folder, not in it, so replacing the whole package folder with a new upload never
# deletes finished runs (it did once, 2026-10-24).
results="${MSP_RESULTS:-$(dirname "$here")/msp-sol-results}"
mkdir -p "$results/logs" "$results/$MD_OUT" "$results/phonons" "$results/ordering"
mode="${1:-all}"
rows="${2:-}"

join() { local IFS=,; echo "$*"; }

missing_md() {  # row numbers whose .npz is not there yet; tags as run_batch.slurm makes them
    awk -F'\t' 'NR > 1 { gsub(/[()]/, "", $2); printf "%d %s_%dK\n", NR - 1, $2, $3 }' runs/run_list.tsv |
        while read -r i tag; do [ -f "$results/$MD_OUT/$tag.npz" ] || echo "$i"; done
}

missing_phonons() {
    ls phonon_structures/*.cif | sort | awk '{ print NR, $0 }' |
        while read -r i f; do
            j="$results/phonons/$(basename "$f" .cif).json"
            grep -q S_grid_per_mol_atoms "$j" 2>/dev/null || echo "$i"  # missing, or from before the temperature grid
        done
}

missing_ordering() {  # an interrupted run leaves an unfinished json, which is resubmitted and resumes
    awk -F'	' 'NR > 1 { print NR - 1, $1 }' ordering_structures/targets.tsv |
        while read -r i tag; do grep -qs '"complete": true' "$results/ordering/$tag.json" || echo "$i"; done
}

submit_md() {
    local array="${rows:-$(join $(missing_md))}"
    if [ -z "$array" ]; then echo "MD: every run already has an output in $results/$MD_OUT/"; return; fi
    local id
    id=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" --gres="$GPU_GRES" \
        --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --array="$array" \
        --job-name="msp-production" --output="$results/logs/production_%A_%a.out" \
        --export=ALL,OUTDIR="$results/$MD_OUT" run_batch.slurm | cut -d';' -f1)  # --parsable prints id;cluster
    echo "Submitted MD job $id (rows $array, $MD_MODEL) -> $results/$MD_OUT/"
}

submit_phonons() {
    local array
    array="$(join $(missing_phonons))"
    if [ -z "$array" ]; then echo "Phonons: every phase already has an output in $results/phonons/"; return; fi
    local id
    id=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
        ${PHONON_GRES:+--gres="$PHONON_GRES"} --cpus-per-task="$PHONON_CPUS" --mem="$PHONON_MEM" --time="$PHONON_TIME" --array="$array" \
        --job-name="msp-phonons" --output="$results/logs/phonons_%A_%a.out" \
        --export=ALL,OUTDIR="$results/phonons" run_phonons.slurm | cut -d';' -f1)
    echo "Submitted phonon job $id (phases $array) -> $results/phonons/"
}

submit_ordering() {
    local array
    array="$(join $(missing_ordering))"
    if [ -z "$array" ]; then echo "Ordering: every target already has a finished output in $results/ordering/"; return; fi
    local id
    id=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" --gres="$GPU_GRES"         --cpus-per-task="$CPUS" --mem="$MEM" --time="$ORDERING_TIME" --array="$array"         --job-name="msp-ordering" --output="$results/logs/ordering_%A_%a.out"         --export=ALL,OUTDIR="$results/ordering" run_ordering.slurm | cut -d';' -f1)
    echo "Submitted ordering job $id (targets $array) -> $results/ordering/"
}

case "$mode" in
    all) submit_md; submit_phonons; submit_ordering ;;
    md) submit_md ;;
    phonons) submit_phonons ;;
    ordering) submit_ordering ;;
    *) echo "use: all | md [rows] | phonons | ordering"; exit 1 ;;
esac
