"""Time IterCC against other coupled-cluster programs: same molecule, basis,
frozen core, convergence thresholds, machine and core count.

Each calculation runs in its own process; the driver records the wall time
from a converged RHF to the converged correlated energy (integral
transformation and setup included), the time per iteration, the iteration
count, the energy and the peak memory of the whole process tree.

    python -P benchmarks/compare.py run --code itercc --method ccsdt --basis cc-pvdz --out results.jsonl

Codes: itercc (point-group symmetry), itercc-nosym, pyscf (pyscf.cc ccsd,
rccsdt, rccsdtq), pyscf-highm (the high-memory rccsdt/rccsdtq), ccpy
(coupled-cluster-py), nwchem (TCE, or its CCSD module for ccsd; mpirun; set
NWCHEM_BASIS_LIBRARY, NWCHEM_EXE for a build other than the one on PATH, and
NWCHEM_MPIRUN for another launcher than "mpirun --oversubscribe"),
orca (AUTOCI, or MDCI for ccsd; `orca` on PATH or ORCA_EXE, scratch in
ORCA_SCRATCH; CCSDTQ runs on its UHF code, the only one AUTOCI has for
quadruples), psi4 (fnocc CCSD, ccresponse polarizabilities; PSI4_PYTHON names
the interpreter that imports psi4).
Methods: ccsd, ccsdt, ccsdtq, eom-ccsdt (the lowest --roots singlet roots of the
irrep holding the lowest excitation, after a tight ground state), response
(CCSDT linear-response dipole polarizability, static and 532 nm, after a tight
ground state) and response-ccsd (the same for CCSD).
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
WAVELENGTH_NM = 532.0
FREQUENCY = 45.56335252767 / WAVELENGTH_NM          # hartree (hc = 45.563... Eh nm)
RESPONSE_TOL = 1e-6                                 # relative residual of each linear solve
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
    from itercc import DipoleResponse, EOMSinglet, RestrictedCC, RestrictedCCSDTQ
    from itercc.pyscf_interface import dipole_integrals, from_rhf

    stamps, out = [], {}
    start = time.perf_counter()
    m = from_rhf(mf, frozen=frozen)
    labels = _labels(mf, frozen) if symmetry else None
    if method == "ccsdtq":
        cc = RestrictedCCSDTQ(m.h, m.eri, m.nocc, m.constant, orbital_irreps=labels, backend="cpu")
        result = cc.solve(energy_tol=ENERGY_TOL, residual_tol=RESIDUAL_TOL, max_iter=LIMIT,
                          callback=lambda n, record: stamps.append(time.perf_counter()))
    else:
        rank = 2 if method in ("ccsd", "response-ccsd") else 3
        cc = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=labels, rank=rank,
                          symmetry_tolerance=1e-8 if symmetry else 0.0)
        # EOM and response audit a 1e-10 ground residual.
        tight = method in ("eom-ccsdt", "response", "response-ccsd")
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
    if method in ("response", "response-ccsd"):
        start = time.perf_counter()
        dipoles, dipole_irreps = dipole_integrals(m, mf.mol)
        response = DipoleResponse(cc, result.amplitudes, dipoles, orbital_irreps=labels,
                                  dipole_irreps=dipole_irreps if symmetry else None,
                                  method="ccsd" if method == "response-ccsd" else "ccsdt")
        tensors = [response.polarizability(w, tol=RESPONSE_TOL).tensor_au.real
                   for w in (float("inf"), WAVELENGTH_NM)]
        out.update(response_seconds=time.perf_counter() - start, **_polarizabilities(tensors))
    return out


def _polarizabilities(tensors):
    """Static and 532 nm isotropic values and principal (sorted diagonal) components."""
    import numpy as np

    return dict(alpha_static=float(np.trace(tensors[0]) / 3), alpha_532=float(np.trace(tensors[1]) / 3),
                alpha_static_diagonal=sorted(float(x) for x in np.linalg.eigvalsh(tensors[0])),
                alpha_532_diagonal=sorted(float(x) for x in np.linalg.eigvalsh(tensors[1])))


def run_pyscf(mf, frozen, method, highm):
    if method == "ccsd":
        if highm:
            raise SystemExit("pyscf-highm has no CCSD; use pyscf")
        from pyscf.cc import ccsd as module
        cls = module.CCSD
    elif method == "ccsdtq":
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
        driver.run_cc(method=method if method in ("ccsd", "ccsdtq") else "ccsdt")
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
{body}"""

NWCHEM_TCE = """tce
 {method}
 freeze {frozen}
 thresh {thresh}
 maxiter {limit}
{eom}end
{response}task tce energy
"""

NWCHEM_CCSD = """ccsd
 freeze {frozen}
 thresh {thresh}
 maxiter {limit}
end
task ccsd energy
"""


def run_nwchem(molecule, basis, method, ranks, memory_mb, roots, irrep_name):
    spec = MOLECULES[molecule]
    atoms = "\n".join(" " + a.strip() for a in spec["atom"].split(";"))
    eom = f" nroots {roots}\n symmetry\n targetsym {irrep_name}\n" if method == "eom-ccsdt" else ""
    executable = os.environ.get("NWCHEM_EXE") or shutil.which("nwchem")
    if method == "ccsd":
        body = NWCHEM_CCSD.format(frozen=spec["frozen"], thresh=RESIDUAL_TOL, limit=LIMIT)
    else:
        response = ""
        if method.startswith("response"):
            response = (f"set tce:lineresp T\nset tce:afreq 0.0 {FREQUENCY:.7f}\n"
                        "set tce:respaxis T T T\n")
        body = NWCHEM_TCE.format(
            method={"eom-ccsdt": "ccsdt", "response": "ccsdt", "response-ccsd": "ccsd"}.get(method, method),
            frozen=spec["frozen"], limit=LIMIT, eom=eom, response=response,
            # One TCE threshold serves the ground state, EOM and response.
            thresh=EOM_TOL if method == "eom-ccsdt" else RESIDUAL_TOL)
    with tempfile.TemporaryDirectory(prefix="nwchem-") as work:
        Path(work, "bench.nw").write_text(NWCHEM_INPUT.format(
            work=work, memory=memory_mb, atoms=atoms, basis=basis, body=body))
        # NWCHEM_MPIRUN: the launcher and its options (cluster_campaign.sh sets
        # its own: Open MPI's defaults spin and yield, slowing other jobs).
        launcher = os.environ.get("NWCHEM_MPIRUN", "mpirun --oversubscribe").split()
        command = [*launcher, "-np", str(ranks), executable, "bench.nw"]
        text = subprocess.run(command, cwd=work, capture_output=True, text=True, errors="replace").stdout
    Path(f"nwchem-{method}-{basis}.out").write_text(text)       # kept next to the results
    if method == "ccsd":
        return _nwchem_ccsd(text)
    converged = "Iterations converged" in text
    if not converged and LIMIT == 200:
        raise RuntimeError("NWChem did not converge:\n" + text[-3000:])
    tag = {"ccsdtq": "CCSDTQ", "response-ccsd": "CCSD"}.get(method, "CCSDT")
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
    if method.startswith("response"):
        values = {}
        for block in text.split("Linear Response polarizability / au")[1:]:
            frequency = float(re.search(r"Frequency\s*=\s*(-?[\d.]+)", block).group(1))
            rows = re.findall(r"^\s*([XYZ])\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+\|", block, re.M)
            values[round(frequency, 5)] = [[float(x) for x in row[1:]] for row in rows[:3]]
        import numpy as np

        tensors = [np.array(values[0.0]), np.array(values[round(FREQUENCY, 5)])]
        out.update(response_seconds=task - scf - sum(walls), **_polarizabilities(tensors))
        out["solve_seconds"] = sum(walls)
    return out


def _nwchem_ccsd(text):
    """NWChem's closed-shell CCSD module (task ccsd)."""
    if "converged" not in text:
        raise RuntimeError("NWChem CCSD did not converge:\n" + text[-3000:])
    energy = float(re.search(r"Total CCSD energy:\s*(-?[\d.]+)", text).group(1))
    task = float(re.search(r"Task\s+times\s+cpu:\s*[\d.]+s\s+wall:\s*([\d.]+)s", text).group(1))
    scf = sum(float(t) for t in re.findall(r"Time for solution =\s*([\d.]+)s", text))
    table = text.split("iter     correlation")[1].split("converged")[0]
    rows = re.findall(r"^\s+\d+\s+-?[\d.]+\s+-?[\d.D+-]+\s+[\d.D+-]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$",
                      table, re.M)
    solve = task - scf
    return dict(energy=energy, converged=True, iterations=len(rows), solve_seconds=solve,
                iteration_seconds=[solve / len(rows)] * len(rows))


ORCA_INPUT = """! {reference}{keyword} {basis} TightSCF
{pal}%maxcore {memory}
%{block}
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
    # ORCA freezes the chemical core by default: 1s of every C to Ne atom here.
    first_row = sum(a.split()[0] in ("Li", "Be", "B", "C", "N", "O", "F", "Ne") for a in spec["atom"].split(";"))
    if spec["frozen"] != first_row:
        raise SystemExit("the ORCA input relies on ORCA's default frozen core")
    atoms = "\n".join(" " + a.strip() for a in spec["atom"].split(";"))
    executable = os.environ.get("ORCA_EXE") or shutil.which("orca")
    keyword, block = ("CCSD", "mdci") if method == "ccsd" else (f"AUTOCI-{method.upper()}", "autoci")
    while True:
        work = tempfile.mkdtemp(prefix="orca-", dir=os.environ.get("ORCA_SCRATCH"))
        try:
            Path(work, "bench.inp").write_text(ORCA_INPUT.format(
                reference="UHF " if method == "ccsdtq" else "", keyword=keyword, block=block, basis=basis,
                pal=f"%pal nprocs {ranks} end\n" if ranks > 1 else "", memory=memory_mb,
                thresh=RESIDUAL_TOL, limit=LIMIT, atoms=atoms))
            command = [executable, "bench.inp"]
            if executable.startswith("/mnt/") and executable.endswith(".exe"):
                # A Windows ORCA driven from WSL: parallel runs need its full Windows path.
                windows = subprocess.run(["wslpath", "-w", executable], capture_output=True,
                                         text=True).stdout.strip()
                command = ["cmd.exe", "/c", windows, "bench.inp"]
            text = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                  errors="replace").stdout
        finally:
            shutil.rmtree(work, ignore_errors=True)
        # ORCA 6.1.1's MPI MDCI can abort with many ranks on a small problem: halve them.
        if "ended prematurely" in text and ranks > 1:
            ranks //= 2
            continue
        break
    Path(f"orca-{method}-{basis}.out").write_text(text)          # kept next to the results
    if "ORCA TERMINATED NORMALLY" not in text:
        raise RuntimeError("ORCA failed:\n" + text[-3000:])
    energy = float(re.search(r"FINAL SINGLE POINT ENERGY\s+(-?[\d.]+)", text).group(1))
    if method == "ccsd":
        table = text.split("COUPLED CLUSTER ITERATIONS")[1].split("have converged")[0]
        rows = re.findall(r"^\s+\d+\s+-?[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s+[\d.]+\s+([\d.]+)\s*$", table, re.M)
        module = re.search(r"MDCI module\s+\.\.\.\s+([\d.]+) sec", text)
        converged = "have converged" in text
    else:
        table = text.split("CCSDT Iterations" if method == "ccsdt" else "CCSDTQ Iterations")[1]
        rows = re.findall(r"^\s+\d+\s+-?[\d.]+\s+-?[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.]+)\s*$", table, re.M)
        module = re.search(r"AUTOCI module\s+\.\.\.\s+([\d.]+) sec", text)
        converged = "Calculation CONVERGED" in text
    solve = float(module.group(1)) if module else None
    seconds = [float(t) for t in rows]
    if method == "ccsd" and solve and rows:                # MDCI prints iteration times to 0.01 s
        seconds = [solve / len(rows)] * len(rows)
    return dict(energy=energy, converged=converged, iterations=len(rows), iteration_seconds=seconds,
                solve_seconds=solve, ranks=ranks)


PSI4_SCRIPT = r'''
import json, re, sys, time
import psi4
spec = json.loads(sys.argv[1])
psi4.set_memory(f"{spec['memory']} MB")
psi4.set_num_threads(spec["threads"])
psi4.core.set_output_file("psi4.out", False)
atoms = "\n".join(a.strip() for a in spec["atom"].split(";"))
psi4.geometry(f"0 1\n{atoms}\nunits angstrom\n")
psi4.set_options({"basis": spec["basis"], "freeze_core": True, "scf_type": "pk", "puream": True,
                  "e_convergence": 1e-10, "d_convergence": 1e-10})
energy, wfn = psi4.energy("scf", return_wfn=True)
psi4.set_options({"e_convergence": spec["energy_tol"], "r_convergence": spec["residual_tol"],
                  "maxiter": spec["limit"]})
out = {}
start = time.perf_counter()
if spec["method"] == "ccsd":
    psi4.set_options({"cc_type": "conv", "qc_module": "fnocc"})
    out["energy"] = psi4.energy("ccsd", ref_wfn=wfn)
else:
    # The ground state to 1e-10 (as for IterCC's response), the response equations to RESPONSE_TOL.
    psi4.set_options({"omega": [0.0, spec["frequency"], "au"], "r_convergence": 1e-10,
                      "e_convergence": 1e-12})
    psi4.properties("ccsd", properties=["polarizability"], ref_wfn=wfn)
    out["energy"] = psi4.variable("CCSD TOTAL ENERGY")
    out["alpha_static"] = psi4.variable("CCSD DIPOLE POLARIZABILITY @ INF NM")
    out["alpha_532"] = psi4.variable(f"CCSD DIPOLE POLARIZABILITY @ {round(45.56335252767 / spec['frequency'])}NM")
out["seconds"] = time.perf_counter() - start
text = open("psi4.out").read()
table = text.split("Begin singles and doubles coupled cluster iterations")[-1] if spec["method"] == "ccsd" \
    else text.split("Solving CC Amplitude Equations")[-1]
out["iterations"] = len(re.findall(r"^\s+\d+\s+(?:\d+\s+\d+\s+)?-?[\d.]+\s+", table.split("converged")[0], re.M))
print("PSI4 " + json.dumps(out))
'''


def run_psi4(molecule, basis, method, threads, memory_mb):
    """Psi4 (fnocc conventional CCSD; ccresponse for CCSD polarizabilities),
    run with the interpreter in PSI4_PYTHON."""
    if method not in ("ccsd", "response-ccsd"):
        raise SystemExit("Psi4 has no CCSDT or CCSDTQ of its own")
    spec = dict(MOLECULES[molecule], basis=basis, method=method, threads=threads, memory=memory_mb,
                energy_tol=ENERGY_TOL, residual_tol=RESIDUAL_TOL, limit=LIMIT, frequency=FREQUENCY)
    python = os.environ.get("PSI4_PYTHON", "python")
    with tempfile.TemporaryDirectory(prefix="psi4-") as work:
        script = Path(work, "run.py")
        script.write_text(PSI4_SCRIPT)
        done = subprocess.run([python, str(script), json.dumps(spec)], cwd=work, capture_output=True,
                              text=True, env=dict(os.environ, PSI_SCRATCH=work))
        text = Path(work, "psi4.out").read_text() if Path(work, "psi4.out").exists() else ""
    Path(f"psi4-{method}-{basis}.out").write_text(text)
    lines = [line for line in done.stdout.splitlines() if line.startswith("PSI4 ")]
    if not lines:
        raise RuntimeError("Psi4 failed:\n" + done.stdout[-1500:] + done.stderr[-1500:])
    result = json.loads(lines[-1][5:])
    out = dict(energy=float(result["energy"]), converged=True, iterations=result["iterations"],
               solve_seconds=result["seconds"])
    if method == "ccsd":
        n = max(1, result["iterations"])
        out["iteration_seconds"] = [result["seconds"] / n] * n
    else:
        out.update(response_seconds=result["seconds"], alpha_static=float(result["alpha_static"]),
                   alpha_532=float(result["alpha_532"]))
    return out


def worker(args):
    import numpy as np

    global LIMIT
    LIMIT = args.max_iter or LIMIT
    out = dict(code=args.code, method=args.method, molecule=args.molecule, basis=args.basis,
               partial=args.max_iter is not None)
    if args.code == "psi4":
        mf, frozen = _rhf(args.molecule, args.basis)
        out.update(run_psi4(args.molecule, args.basis, args.method, args.threads, args.psi4_memory))
    elif args.code == "orca":
        if args.method == "eom-ccsdt" or args.method.startswith("response"):
            raise SystemExit("ORCA's AUTOCI has no EOM-CCSDT or CC response")
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
    parser.add_argument("--method", default="ccsdt",
                        choices=("ccsd", "ccsdt", "ccsdtq", "eom-ccsdt", "response", "response-ccsd"))
    parser.add_argument("--molecule", default="water", choices=tuple(MOLECULES))
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--roots", type=int, default=3)
    parser.add_argument("--irrep", type=int, default=0, help="PySCF irrep id of the EOM roots")
    parser.add_argument("--irrep-name", help="the same irrep as ccpy or NWChem names it (their axis "
                        "conventions can swap B1 and B2)")
    parser.add_argument("--nwchem-memory", type=int, default=1000, help="MB per MPI rank")
    parser.add_argument("--orca-memory", type=int, default=12000, help="ORCA %%maxcore, MB per process")
    parser.add_argument("--psi4-memory", type=int, default=12000, help="Psi4 memory, MB")
    parser.add_argument("--max-iter", type=int, help="time this many iterations instead of converging")
    parser.add_argument("--memory-limit-gib", type=float, default=13.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out", default="compare.jsonl")
    args = parser.parse_args()
    (worker if args.command == "worker" else run)(args)


if __name__ == "__main__":
    main()
