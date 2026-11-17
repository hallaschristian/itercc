#!/bin/bash
#SBATCH -J itercc-compare
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 12:00:00
# The comparison runs behind docs/benchmarks.md on one 56-core socket of a
# node held whole (on a shared node Slurm spreads 56 cores over both sockets):
# every run gets that socket's cores (THREADS, at most 56; NWChem as MPI
# ranks) unless the line sets --threads, and its memory (numactl). Never more
# than one socket: more threads only slowed the runs. Submitted with -c 56
# instead of --exclusive it binds to the allocated cores, interleaving memory
# with a warning when they span sockets (timings then are not one socket's).
# A line of RUNS holds compare.py run options. Lines starting with
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
# One socket: the first NUMA node whose cores are all allocated, else the
# allocated cores as they are.
words() { tr -s ' ' '\n' | sed '/^$/d' | sort; }
ALLOWED=$(nproc)
NODES=$(numactl --show | sed -n 's/^nodebind: *//p')
PHYS=$(numactl --show | sed -n 's/^physcpubind: *//p' | words)
CORES=$ALLOWED
BIND="--interleave=all"
for node in $NODES; do
    node_cpus=$(numactl --hardware | sed -n "s/^node $node cpus: *//p" | words)
    if [ -z "$(comm -23 <(echo "$node_cpus") <(echo "$PHYS"))" ]; then
        BIND="--cpunodebind=$node --membind=$node"
        CORES=$(echo "$node_cpus" | wc -l)
        break
    fi
done
[ "$BIND" != "--interleave=all" ] || echo "warning: no whole NUMA node allocated; memory interleaved" >&2
THREADS=${3:-$(( CORES < 56 ? CORES : 56 ))}
[ "$THREADS" -le 56 ] || { echo "at most 56 threads (one socket)" >&2; exit 1; }
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
