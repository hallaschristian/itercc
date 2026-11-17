#!/bin/bash
#SBATCH -J itercc-compare
#SBATCH -N 1
#SBATCH -c 112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 12:00:00
# The comparison runs behind docs/benchmarks.md on one two-socket cluster node.
# Runs take one socket each (numactl --cpunodebind=S --membind=S) and the
# socket's cores (THREADS; NWChem as MPI ranks) unless the line sets
# --threads; a line of RUNS holds compare.py run options. One run at a time by
# default: a program on one socket can slow one on the other many times over
# (NWChem's always-runnable ranks keep the kernel's load balancing busy on the
# idle socket), so WORKERS=2 (two runs at a time, one per socket) is for
# checks only. Lines starting with "@scaling" write to scalingS.jsonl
# (thread-scaling records, kept apart from the comparison's), and lines
# starting with "@node" run last, on the whole node (memory interleaved;
# node.jsonl). Blank lines and "#" comments are skipped. Resubmitting
# continues where the last job stopped.
#
# usage: sbatch -p PARTITION benchmarks/cluster_campaign.sh RUNS OUTDIR [THREADS]
# environment: PYTHON (itercc, pyscf, coupled-cluster-py, psutil importable),
# COMPARE (path of compare.py), PSI4_PYTHON, NWCHEM_EXE, NWCHEM_BASIS_LIBRARY,
# NWCHEM_MPIRUN (NWChem's launcher with its options, see compare.py),
# ITERCC_KERNEL_CACHE; optional WORKERS.
set -u
RUNS=$(realpath "$1")
OUTDIR=$(realpath "$2")
THREADS=${3:-56}
WORKERS=${WORKERS:-1}
mkdir -p "$OUTDIR"
QUEUE=$OUTDIR/queue
[ -e "$QUEUE" ] || echo 0 > "$QUEUE"
mapfile -t LINES < <(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$RUNS")
export PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY:-400000} NWCHEM_MPIRUN
# Workstation-style launches: without Slurm's variables, mpirun forks its ranks
# under the run's socket binding instead of starting a Slurm step.
UNSLURM=$(compgen -e | grep '^SLURM' | sed 's/^/-u /' | tr '\n' ' ')

run_line() {     # run_line BIND OUTFILE LINE...
    local bind=$1 out=$2
    shift 2
    # shellcheck disable=SC2086
    env $UNSLURM numactl $bind "$PYTHON" -P "$COMPARE" run --out "$out" --threads "$THREADS" \
        --memory-limit-gib 440 --nwchem-memory 7000 --psi4-memory 300000 $*
}

worker() {
    local socket=$1 n line out
    while true; do
        n=$(flock "$QUEUE.lock" bash -c 'n=$(cat "$1"); echo $((n + 1)) > "$1"; echo "$n"' _ "$QUEUE")
        [ "$n" -lt "${#LINES[@]}" ] || break
        line=${LINES[$n]}
        if [[ $line == @node* ]]; then          # whole-node lines run last
            echo "$n" >> "$OUTDIR/node-lines"
            continue
        fi
        out=$OUTDIR/socket$socket.jsonl
        if [[ $line == @scaling* ]]; then        # thread-scaling records stay apart
            line=${line#@scaling}
            out=$OUTDIR/scaling$socket.jsonl
        fi
        echo "$(date +%T) socket $socket line $n: $line" >> "$OUTDIR/progress.log"
        run_line "--cpunodebind=$socket --membind=$socket" "$out" "$line" >> "$OUTDIR/socket$socket.log" 2>&1
    done
}

for socket in $(seq 0 $((WORKERS - 1))); do
    worker "$socket" &
done
wait
if [ -e "$OUTDIR/node-lines" ]; then
    sort -n -u "$OUTDIR/node-lines" | while read -r n; do
        line=${LINES[$n]#@node}
        echo "$(date +%T) node line $n: $line" >> "$OUTDIR/progress.log"
        run_line "--interleave=all" "$OUTDIR/node.jsonl" "$line" >> "$OUTDIR/node.log" 2>&1
    done
    mv "$OUTDIR/node-lines" "$OUTDIR/node-lines.done"
fi
echo "$(date +%T) finished" >> "$OUTDIR/progress.log"
