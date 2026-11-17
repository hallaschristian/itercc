"""Time IterCC against other coupled-cluster programs: same molecule, basis,
frozen core, convergence thresholds, machine and core count.

Each calculation runs in its own process; the driver records the wall time
from a converged RHF to the converged correlated energy (integral
transformation and setup included), the time per iteration, the iteration
count, the energy and the peak memory of the whole process tree.

    python -P benchmarks/compare.py run --code itercc --method ccsdt --basis cc-pvdz --out results.jsonl

Codes: itercc (C2v symmetry), itercc-nosym, pyscf (pyscf.cc.rccsdt /
rccsdtq), pyscf-highm (their high-memory variants), ccpy
(coupled-cluster-py), nwchem (TCE, run with mpirun; set NWCHEM_BASIS_LIBRARY),
orca (AUTOCI; `orca` on PATH or ORCA_EXE, scratch in ORCA_SCRATCH; CCSDTQ runs on
its UHF code, the only one AUTOCI has for quadruples).
Methods: ccsdt, ccsdtq, eom-ccsdt (the lowest --roots singlet roots of the
irrep holding the lowest excitation, after a tight ground state).
`benchmarks/plot_compare.py` draws the figures in docs/images.
"""
import argparse
import contextlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

MOLECULES = {
    "water": dict(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", frozen=1),
}
ENERGY_TOL, RESIDUAL_TOL = 1e-8, 1e-6
EOM_TOL = 1e-5                      # residual norm of each EOM root
LIMIT = 200                         # iterations; --max-iter times a few iterations only


def _rhf(molecule, basis):
    from pyscf import gto, scf

    spec = MOLECULES[molecule]
    group = gto.M(atom=spec["atom"], basis=basis, symmetry=True, verbose=0).groupname
    mol = gto.M(atom=spec["atom"], basis=basis, symmetry=group, verbose=0)   # ccpy reads the name
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return mf, spec["frozen"]


def _labels(mf, frozen):
    import numpy as np
    from pyscf.scf import hf_symm

    return np.asarray(hf_symm.get_orbsym(mf.mol, mf.mo_coeff), dtype=np.int64)[frozen:]


def _irrep_name(mf, irrep):
    from pyscf import symm

    return symm.irrep_id2name(mf.mol.groupname, irrep)     # mol.irrep_name lists only those present


def _intervals(stamps):
    return [b - a for a, b in zip(stamps, stamps[1:])]


def run_itercc(mf, frozen, method, symmetry, roots, irrep, basis):
    from itercc import EOMSinglet, RestrictedCC, RestrictedCCSDTQ
    from itercc.pyscf_interface import from_rhf

    stamps, out = [], {}
    start = time.perf_counter()
    m = from_rhf(mf, frozen=frozen)
    labels = _labels(mf, frozen) if symmetry else None
    if method == "ccsdtq":
        cc = RestrictedCCSDTQ(m.h, m.eri, m.nocc, m.constant, orbital_irreps=labels, backend="cpu")
        result = cc.solve(energy_tol=ENERGY_TOL, residual_tol=RESIDUAL_TOL, max_iter=LIMIT,
                          callback=lambda n, record: stamps.append(time.perf_counter()))
    else:
        cc = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=labels,
                          symmetry_tolerance=1e-8 if symmetry else 0.0)
        tight = method == "eom-ccsdt"           # EOM audits a 1e-10 ground residual
        result = cc.solve(energy_tol=1e-12 if tight else ENERGY_TOL,
                          residual_tol=1e-10 if tight else RESIDUAL_TOL, max_iter=LIMIT,
                          callback=lambda record: stamps.append(time.perf_counter()))
    out.update(energy=float(result.energy), converged=bool(result.converged),
               iterations=len(result.iterations), solve_seconds=time.perf_counter() - start,
               iteration_seconds=_intervals(stamps))
    if method == "eom-ccsdt":
        start = time.perf_counter()
        eom = EOMSinglet(cc, result.amplitudes, orbital_irreps=labels, irrep=irrep if symmetry else 0)
        states = eom.solve(nroots=roots, residual_tol=EOM_TOL, max_space=max(12, 4 * roots))
        out.update(eom_seconds=time.perf_counter() - start, eom_converged=bool(states.converged),
                   excitation_energies=[float(e) for e in states.excitation_energies],
                   eom_actions=int(states.matvecs))
    return out


def run_pyscf(mf, frozen, method, highm):
    if method == "ccsdtq":
        from pyscf.cc import rccsdtq as module
        if highm:
            from pyscf.cc import rccsdtq_highm as module
        cls = module.RCCSDTQ
    else:
        from pyscf.cc import rccsdt as module
        if highm:
            from pyscf.cc import rccsdt_highm as module
        cls = module.RCCSDT
    stamps = []
    start = time.perf_counter()
    mycc = cls(mf, frozen=frozen)
    mycc.conv_tol, mycc.conv_tol_normt, mycc.max_cycle, mycc.verbose = ENERGY_TOL, RESIDUAL_TOL, LIMIT, 0
    mycc.callback = lambda env: stamps.append(time.perf_counter())
    mycc.kernel()
    return dict(energy=float(mycc.e_tot), converged=bool(mycc.converged), iterations=len(stamps),
                solve_seconds=time.perf_counter() - start, iteration_seconds=_intervals(stamps))


def run_ccpy(mf, frozen, method, roots, irrep_name):
    from ccpy.drivers.driver import Driver

    log = io.StringIO()
    out = {}
    with contextlib.redirect_stdout(log):
        start = time.perf_counter()
        driver = Driver.from_pyscf(mf, nfrozen=frozen)
        tight = method == "eom-ccsdt"
        driver.options.update(energy_convergence=1e-10 if tight else ENERGY_TOL,
                              amp_convergence=1e-8 if tight else RESIDUAL_TOL,
                              maximum_iterations=LIMIT)
        transformed = time.perf_counter()
        driver.run_cc(method="ccsdtq" if method == "ccsdtq" else "ccsdt")
        ground = time.perf_counter()
        if method == "eom-ccsdt":
            driver.run_hbar(method="ccsdt")
            driver.run_guess(method="cis", multiplicity=1,
                             roots_per_irrep={name: (roots if name == irrep_name else 0)
                                              for name in mf.mol.irrep_name})
            driver.options["amp_convergence"] = EOM_TOL
            driver.run_eomcc(method="eomccsdt", state_index=list(range(1, roots + 1)))
            out.update(eom_seconds=time.perf_counter() - ground,
                       excitation_energies=[float(driver.vertical_excitation_energy[k])
                                            for k in range(1, roots + 1)])
    # The ground-state iterations only (an EOM run prints its own table after them).
    text = log.getvalue().split("CC calculation started")[1].split("CC calculation successfully converged")[0]
    converged = "CC calculation successfully converged" in log.getvalue()
    # ccpy prints each iteration's wall time to 0.1 s; the run_cc time over the
    # iteration count is the precise per-iteration figure.
    rows = len(re.findall(r"^\s*\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+\([\d.]+m\s*[\d.]+s\)", text, re.M))
    out.update(energy=float(driver.system.reference_energy + driver.correlation_energy),
               converged=converged, solve_seconds=ground - start,
               iterations=rows, iteration_seconds=[(ground - transformed) / rows] * rows if rows else [])
    return out


NWCHEM_INPUT = """start bench
permanent_dir {work}
scratch_dir {work}
memory total {memory} mb
geometry units angstrom
{atoms}
end
basis spherical
 * library {basis}
end
scf
 thresh 1e-10
 tol2e 1e-14
end
tce
 {method}
 freeze {frozen}
 thresh {thresh}
 maxiter {limit}
{eom}end
task tce energy
"""


def run_nwchem(molecule, basis, method, ranks, memory_mb, roots, irrep_name):
    spec = MOLECULES[molecule]
    atoms = "\n".join(" " + a.strip() for a in spec["atom"].split(";"))
    eom = f" nroots {roots}\n symmetry\n targetsym {irrep_name}\n" if method == "eom-ccsdt" else ""
    executable = shutil.which("nwchem")
    with tempfile.TemporaryDirectory(prefix="nwchem-") as work:
        Path(work, "bench.nw").write_text(NWCHEM_INPUT.format(
            work=work, memory=memory_mb, atoms=atoms, basis=basis, frozen=spec["frozen"],
            method="ccsdt" if method == "eom-ccsdt" else method,       # nroots selects EOM
            # One TCE threshold serves the ground state and EOM: use the EOM tolerance.
            thresh=EOM_TOL if method == "eom-ccsdt" else RESIDUAL_TOL, eom=eom, limit=LIMIT))
        command = ["mpirun", "--oversubscribe", "-np", str(ranks), executable, "bench.nw"]
        text = subprocess.run(command, cwd=work, capture_output=True, text=True, errors="replace").stdout
    Path(f"nwchem-{method}-{basis}.out").write_text(text)       # kept next to the results
    converged = "Iterations converged" in text
    if not converged and LIMIT == 200:
        raise RuntimeError("NWChem did not converge:\n" + text[-3000:])
    tag = "CCSDTQ" if method == "ccsdtq" else "CCSDT"
    found = re.search(rf"{tag} total energy / hartree\s*=\s*([-\d.]+)", text)
    energy = float(found.group(1)) if found else None
    # NWChem stops without its task summary when --max-iter ends the iterations.
    found = re.search(r"Task\s+times\s+cpu:\s*[\d.]+s\s+wall:\s*([\d.]+)s", text)
    task = float(found.group(1)) if found else None
    scf = sum(float(t) for t in re.findall(r"Time for solution =\s*([\d.]+)s", text))
    ground_block = text.split("Iterations converged")[0] if converged else text
    rows = re.findall(r"^\s+(\d+)\s+[\d.Ee+-]+\s+-?[\d.]+\s+[\d.]+\s+([\d.]+)\s*$", ground_block, re.M)
    walls = [float(w) for _, w in rows]
    if not walls:
        raise RuntimeError("NWChem printed no iterations:\n" + text[-3000:])
    out = dict(energy=energy, converged=converged, iterations=len(walls), iteration_seconds=walls,
               solve_seconds=None if task is None else task - scf)
    if method == "eom-ccsdt":
        # Every root is in the last block of the EOM iteration table (the summary
        # prints only the target root).
        table = text.split("EOM-CCSDT right-hand side iterations")[1]
        last = table.split("Iteration ")[-1]
        omegas = re.findall(r"^\s+[\d.]+\s+(-?[\d.]+)\s+-?[\d.]+(?:\s+[\d.]+\s+[\d.]+)?\s*$", last, re.M)
        out.update(excitation_energies=sorted(float(e) for e in omegas[:roots]),
                   eom_seconds=task - scf - sum(walls))
        out["solve_seconds"] = sum(walls)
    return out


ORCA_INPUT = """! {reference}AUTOCI-{method} {basis} TightSCF
{pal}%maxcore {memory}
%autoci
  STol {thresh}
  MaxIter {limit}
end
* xyz 0 1
{atoms}
*
"""


def run_orca(molecule, basis, method, ranks, memory_mb):
    """ORCA's AUTOCI. Its residual threshold (STol) is the largest residual
    element, as in IterCC; it has no separate energy threshold."""
    spec = MOLECULES[molecule]
    if spec["frozen"] != 1:
        raise SystemExit("the ORCA input relies on ORCA's default frozen core (O 1s here)")
    atoms = "\n".join(" " + a.strip() for a in spec["atom"].split(";"))
    executable = os.environ.get("ORCA_EXE") or shutil.which("orca")
    work = tempfile.mkdtemp(prefix="orca-", dir=os.environ.get("ORCA_SCRATCH"))
    try:
        Path(work, "bench.inp").write_text(ORCA_INPUT.format(
            reference="UHF " if method == "ccsdtq" else "", method=method.upper(), basis=basis,
            pal=f"%pal nprocs {ranks} end\n" if ranks > 1 else "", memory=memory_mb,
            thresh=RESIDUAL_TOL, limit=LIMIT, atoms=atoms))
        text = subprocess.run([executable, "bench.inp"], cwd=work, capture_output=True, text=True,
                              errors="replace").stdout
    finally:
        shutil.rmtree(work, ignore_errors=True)
    Path(f"orca-{method}-{basis}.out").write_text(text)          # kept next to the results
    if "ORCA TERMINATED NORMALLY" not in text:
        raise RuntimeError("ORCA failed:\n" + text[-3000:])
    table = text.split("CCSDT Iterations" if method == "ccsdt" else "CCSDTQ Iterations")[1]
    rows = re.findall(r"^\s+\d+\s+-?[\d.]+\s+-?[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.]+)\s*$", table, re.M)
    module = re.search(r"AUTOCI module\s+\.\.\.\s+([\d.]+) sec", text)
    return dict(energy=float(re.search(r"FINAL SINGLE POINT ENERGY\s+(-?[\d.]+)", text).group(1)),
                converged="Calculation CONVERGED" in text, iterations=len(rows),
                iteration_seconds=[float(t) for t in rows],
                solve_seconds=float(module.group(1)) if module else None)


def worker(args):
    import numpy as np

    global LIMIT
    LIMIT = args.max_iter or LIMIT
    out = dict(code=args.code, method=args.method, molecule=args.molecule, basis=args.basis,
               partial=args.max_iter is not None)
    if args.code == "orca":
        if args.method == "eom-ccsdt":
            raise SystemExit("ORCA's AUTOCI has no EOM-CCSDT")
        mf, frozen = _rhf(args.molecule, args.basis)
        out.update(run_orca(args.molecule, args.basis, args.method, args.threads, args.orca_memory))
    elif args.code == "nwchem":
        mf, frozen = _rhf(args.molecule, args.basis)
        name = (args.irrep_name or _irrep_name(mf, args.irrep)).lower()
        out.update(run_nwchem(args.molecule, args.basis, args.method, args.threads, args.nwchem_memory,
                              args.roots, name))
    else:
        mf, frozen = _rhf(args.molecule, args.basis)
        if args.code.startswith("itercc"):
            out.update(run_itercc(mf, frozen, args.method, args.code == "itercc", args.roots, args.irrep,
                                  args.basis))
        elif args.code.startswith("pyscf"):
            if args.method == "eom-ccsdt":
                raise SystemExit("PySCF has no EOM-CCSDT")
            out.update(run_pyscf(mf, frozen, args.method, args.code == "pyscf-highm"))
        elif args.code == "ccpy":
            out.update(run_ccpy(mf, frozen, args.method, args.roots,
                                args.irrep_name or _irrep_name(mf, args.irrep)))
        else:
            raise SystemExit(f"unknown code {args.code}")
    nocc = int(np.count_nonzero(mf.mo_occ)) - frozen
    out.update(nocc=nocc, nvir=len(mf.mo_occ) - nocc - frozen, nbasis=len(mf.mo_occ),
               scf_energy=float(mf.e_tot))
    seconds = out.get("iteration_seconds") or []
    out["median_iteration_seconds"] = float(np.median(seconds)) if seconds else None
    print("RESULT " + json.dumps(out), flush=True)


def _tree_rss(process):
    import psutil

    total = 0
    try:
        for p in [process] + process.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += p.memory_info().rss
    except psutil.Error:
        pass
    return total


def run(args):
    import psutil

    threads = str(args.threads)
    env = dict(os.environ, OMP_NUM_THREADS="1" if args.code == "nwchem" else threads,
               OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads, ITERCC_CPU_THREADS=threads)
    command = [sys.executable, "-P", __file__, "worker"] + [
        f"--{key.replace('_', '-')}={value}" for key, value in vars(args).items()
        if key not in ("command", "out", "timeout", "memory_limit_gib") and value is not None]
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    peak, done = [0], threading.Event()
    handle = psutil.Process(proc.pid)

    killed = []

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], _tree_rss(handle))
            if peak[0] > args.memory_limit_gib * 2**30 and not killed:
                killed.append(True)             # stop before the machine runs out of memory
                for p in handle.children(recursive=True) + [handle]:
                    with contextlib.suppress(psutil.Error):
                        p.kill()
            done.wait(0.2)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    status = "ok"
    try:
        output, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        for p in handle.children(recursive=True) + [handle]:
            with contextlib.suppress(psutil.Error):
                p.kill()
        output, _ = proc.communicate()
        status = "timeout"
    done.set()
    sampler.join()
    if killed:
        status = "memory"
    lines = [line for line in output.splitlines() if line.startswith("RESULT ")]
    record = json.loads(lines[-1][7:]) if lines else dict(code=args.code, method=args.method,
                                                         molecule=args.molecule, basis=args.basis)
    if not lines and status == "ok":
        status = "failed"
        record["error"] = output[-2000:]
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    record.update(status=status, wall_seconds=time.perf_counter() - started, peak_gib=peak[0] / 2**30,
                  threads=args.threads, cpu=cpu)
    if args.code == "orca" and os.environ.get("ORCA_SCRATCH", "").startswith("/mnt/"):
        record["peak_gib"] = None       # Windows ORCA started from WSL: its memory is not visible here
    with open(args.out, "a") as handle_out:
        handle_out.write(json.dumps(record) + "\n")
    shown = {k: record.get(k) for k in ("code", "method", "basis", "status", "energy", "iterations",
                                        "median_iteration_seconds", "solve_seconds", "eom_seconds", "peak_gib")}
    print(json.dumps(shown), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--code", required=True)
    parser.add_argument("--method", default="ccsdt", choices=("ccsdt", "ccsdtq", "eom-ccsdt"))
    parser.add_argument("--molecule", default="water", choices=tuple(MOLECULES))
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--roots", type=int, default=3)
    parser.add_argument("--irrep", type=int, default=0, help="PySCF irrep id of the EOM roots")
    parser.add_argument("--irrep-name", help="the same irrep as ccpy or NWChem names it (their axis "
                        "conventions can swap B1 and B2)")
    parser.add_argument("--nwchem-memory", type=int, default=1000, help="MB per MPI rank")
    parser.add_argument("--orca-memory", type=int, default=12000, help="ORCA %%maxcore, MB per process")
    parser.add_argument("--max-iter", type=int, help="time this many iterations instead of converging")
    parser.add_argument("--memory-limit-gib", type=float, default=13.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out", default="compare.jsonl")
    args = parser.parse_args()
    (worker if args.command == "worker" else run)(args)


if __name__ == "__main__":
    main()
