#!/bin/bash
#SBATCH -J itercc-compare
#SBATCH -N 1
#SBATCH -c 56
#SBATCH -m block:block
#SBATCH --mem=480G
#SBATCH -t 12:00:00
# The comparison runs behind docs/benchmarks.md on one CPU socket of a
# cluster node (56 cores, never more: a run that leaves the socket is slower).
# Each run gets the allocated cores (THREADS, at most 56; NWChem as MPI
# ranks) unless the line sets --threads, with memory bound to their NUMA node
# (numactl). A line of RUNS holds compare.py run options. Lines starting with
# "@scaling" write to scaling.jsonl (thread-scaling records, kept apart from
# the comparison's). Blank lines and "#" comments are skipped. Resubmitting
# continues where the last job stopped.
#
# usage: sbatch -p PARTITION benchmarks/cluster_campaign.sh RUNS OUTDIR [THREADS]
# environment: PYTHON (itercc, pyscf, coupled-cluster-py, psutil importable),
# COMPARE (path of compare.py), PSI4_PYTHON, NWCHEM_EXE, NWCHEM_BASIS_LIBRARY,
# NWCHEM_MPIRUN (NWChem's launcher with its options, see compare.py),
# ITERCC_KERNEL_CACHE.
set -u
RUNS=$(realpath "$1")
OUTDIR=$(realpath "$2")
ALLOWED=$(nproc)
THREADS=${3:-$(( ALLOWED < 56 ? ALLOWED : 56 ))}
[ "$THREADS" -le 56 ] || { echo "at most 56 threads (one socket)" >&2; exit 1; }
# The NUMA node of the allocated cores, for their memory.
NODES=$(numactl --show | sed -n 's/^nodebind: *//p')
if [ "$(echo "$NODES" | wc -w)" -eq 1 ]; then
    BIND="--cpunodebind=$NODES --membind=$NODES"
else
    echo "warning: the allocated cores span NUMA nodes $NODES" >&2
    BIND="--interleave=all"
fi
mkdir -p "$OUTDIR"
QUEUE=$OUTDIR/queue
[ -e "$QUEUE" ] || echo 0 > "$QUEUE"
mapfile -t LINES < <(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$RUNS")
export PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY:-400000} NWCHEM_MPIRUN
# Workstation-style launches: without Slurm's variables, mpirun forks its ranks
# under the run's binding instead of starting a Slurm step.
UNSLURM=$(compgen -e | grep '^SLURM' | sed 's/^/-u /' | tr '\n' ' ')

run_line() {     # run_line OUTFILE LINE...
    local out=$1
    shift
    # shellcheck disable=SC2086
    env $UNSLURM numactl $BIND "$PYTHON" -P "$COMPARE" run --out "$out" --threads "$THREADS" \
        --memory-limit-gib 440 --nwchem-memory 7000 --psi4-memory 300000 $*
}

while true; do
    n=$(flock "$QUEUE.lock" bash -c 'n=$(cat "$1"); echo $((n + 1)) > "$1"; echo "$n"' _ "$QUEUE")
    [ "$n" -lt "${#LINES[@]}" ] || break
    line=${LINES[$n]}
    out=$OUTDIR/socket.jsonl
    if [[ $line == @scaling* ]]; then            # thread-scaling records stay apart
        line=${line#@scaling}
        out=$OUTDIR/scaling.jsonl
    fi
    echo "$(date +%T) line $n: $line" >> "$OUTDIR/progress.log"
    run_line "$out" "$line" >> "$OUTDIR/runs.log" 2>&1
done
echo "$(date +%T) finished" >> "$OUTDIR/progress.log"
