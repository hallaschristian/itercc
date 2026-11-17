"""Draw the comparison figures (light and dark SVG) and print the markdown
tables of docs/benchmarks.md from the records of benchmarks/compare.py.

    python benchmarks/plot_compare.py results.jsonl docs/images [--single single-core.jsonl]
        [--threads scaling.jsonl] [--machine TEXT] [--cores N] [--png]

--single adds the one-core comparison (with ORCA), --threads the thread-scaling
figure; --machine and --cores describe where the records were taken (default:
the 12-core desktop); --png also writes PNG copies (for previews).
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PNG = "--png" in sys.argv


def save(fig, out, colors):
    fig.savefig(out, facecolor=colors["surface"])
    if PNG:
        fig.savefig(Path(out).with_suffix(".png"), facecolor=colors["surface"], dpi=110)
    plt.close(fig)


THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c"]),
}
# code -> (legend name, categorical slot, marker, line style)
CODES = OrderedDict([
    ("itercc", ("IterCC", 0, "o", "-")),
    ("itercc-nosym", ("IterCC, no symmetry", 0, "o", (0, (4, 3)))),
    ("pyscf-highm", ("PySCF 2.14", 1, "s", "-")),
    ("ccpy", ("ccpy 0.0.5", 2, "^", "-")),
    ("nwchem", ("NWChem 7.3.1 (TCE)", 3, "D", "-")),
    ("orca", ("ORCA 6.1.1 (AUTOCI)", 4, "v", "-")),
])
# The CCSD comparison: PySCF's CCSD, NWChem's CCSD module, ORCA's MDCI and Psi4.
CCSD_CODES = OrderedDict([
    ("itercc", CODES["itercc"]),
    ("itercc-nosym", CODES["itercc-nosym"]),
    ("pyscf", ("PySCF 2.14", 1, "s", "-")),
    ("ccpy", CODES["ccpy"]),
    ("nwchem", ("NWChem 7.3.1", 3, "D", "-")),
    ("orca", ("ORCA 6.1.1 (MDCI)", 4, "v", "-")),
    ("psi4", ("Psi4 1.11 (fnocc)", 5, "P", "-")),
])
NAMES = {"itercc": "IterCC", "itercc-nosym": "IterCC, no symmetry", "pyscf-highm": "PySCF 2.14, `_highm`",
         "pyscf": "PySCF 2.14", "ccpy": "ccpy 0.0.5", "nwchem": "NWChem 7.3.1", "orca": "ORCA 6.1.1",
         "psi4": "Psi4 1.11"}
BASIS_NAMES = {"6-31g": "6-31G", "6-31g*": "6-31G*", "cc-pvdz": "cc-pVDZ", "aug-cc-pvdz": "aug-cc-pVDZ",
               "cc-pvtz": "cc-pVTZ", "aug-cc-pvtz": "aug-cc-pVTZ", "cc-pvqz": "cc-pVQZ",
               "aug-cc-pvqz": "aug-cc-pVQZ"}


def _option(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# Where the records were taken, for the subtitles.
MACHINE = _option("--machine", "12-core Ryzen 9 3900X")
CORES = int(_option("--cores", "12"))
NWCHEM_RANKS = _option("--nwchem-ranks", str(CORES))     # NWChem's MPI ranks, as the runs used them
SCALING_MACHINE = _option("--scaling-machine", "12-core Ryzen 9 3900X with two memory channels")


def load(path, molecule="water"):
    """The latest record per (code, method, basis) of one molecule. When the
    latest run failed (say, out of memory before converging) but an earlier one
    timed iterations, that earlier time per iteration is kept as `timed`
    (shown with a *)."""
    latest, timed = {}, {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record.get("molecule", "water") != molecule:
            continue
        key = (record["code"], record["method"], record["basis"])
        latest[key] = record
        if record["status"] == "ok" and record.get("median_iteration_seconds"):
            timed[key] = record
    for key, record in latest.items():
        if record["status"] != "ok" and key in timed:
            record["timed"] = dict(timed[key], partial=True)
    return latest


def iteration_record(record):
    """The record whose time per iteration to show, or None."""
    if record and record["status"] != "ok":
        record = record.get("timed")
    return record if record and record.get("median_iteration_seconds") else None


def seconds(value):
    if value >= 100:
        return f"{value:.0f} s"
    if value >= 1:
        return f"{value:.1f} s"
    if value >= 0.1:
        return f"{value:.2f} s"
    return f"{value * 1000:.0f} ms"


SUBTITLE = (f"Water, O 1s frozen; {MACHINE}, {CORES} threads (NWChem: {NWCHEM_RANKS} MPI ranks). "
            "Lower is faster.")


def scaling_figure(records, method, title, out, theme, subtitle=SUBTITLE, legend_at="upper left",
                   codes=CODES):
    colors = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=100)
    fig.patch.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])
    bases = OrderedDict()
    for (code, m, basis), record in records.items():
        if m == method and iteration_record(record):
            bases[basis] = iteration_record(record)["nvir"]
    bases = OrderedDict(sorted(bases.items(), key=lambda item: item[1]))
    handles, ends = [], []
    position = {basis: index for index, basis in enumerate(bases)}      # evenly spaced basis sets
    for code, (name, slot, marker, style) in codes.items():
        points = [(position[b], iteration_record(records.get((code, method, b)))["median_iteration_seconds"])
                  for b in bases if iteration_record(records.get((code, method, b)))]
        if not points:
            continue
        xs, ys = zip(*points)
        color = colors["series"][slot]
        ax.plot(xs, ys, linestyle=style, color=color, linewidth=2, solid_capstyle="round",
                dash_capstyle="round", solid_joinstyle="round", zorder=3 if code == "itercc" else 2)
        ax.scatter(xs, ys, s=52, marker=marker, color=color, edgecolors=colors["surface"],
                   linewidths=2, zorder=4)
        ends.append((xs[-1], ys[-1], seconds(ys[-1])))
        handles.append(Line2D([], [], color=color, linestyle=style, linewidth=2, marker=marker,
                              markersize=7, markeredgecolor=colors["surface"], label=name))
        failed = [b for b in bases if position[b] > xs[-1]
                  and (records.get((code, method, b)) or {}).get("status") == "memory"]
        if failed:                                  # where a program ran out of memory
            ax.annotate(f"{name.split(' (')[0]}:\nout of memory", (position[failed[0]], ys[-1] * 3),
                        ha="center", va="center", fontsize=9, color=colors["secondary"])
    ax.set_yscale("log")
    ax.set_xticks(list(position.values()))
    ax.set_xticklabels([f"{BASIS_NAMES.get(b, b)}\n{n} virtual" for b, n in bases.items()])
    ax.minorticks_off()
    ax.set_xlim(-0.35, len(bases) - 0.45)
    ax.set_ylabel("seconds per iteration (log scale)", color=colors["secondary"])
    _end_labels(fig, ax, ends, colors)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(plain_number))
    ax.grid(axis="y", color=colors["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors["axis"])
    ax.tick_params(colors=colors["muted"], labelcolor=colors["secondary"], length=0, pad=6)
    fig.text(0.015, 0.965, title, ha="left", va="top", fontsize=12, color=colors["ink"], weight="bold")
    fig.text(0.015, 0.905, subtitle, ha="left", va="top", fontsize=9, color=colors["secondary"])
    legend = ax.legend(handles=handles, loc=legend_at, frameon=False, fontsize=9,
                       labelcolor=colors["ink"], handlelength=2.6)
    legend.set_zorder(5)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.17)
    save(fig, out, colors)


def plain_number(value, _=None):
    """Tick labels as plain decimals (0.001, 10, 1000), never 1e-03."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _end_labels(fig, ax, ends, colors, gap=13):
    """Value labels at line ends. Labels closer than `gap` points apart are
    spread out and tied to their points with hairline leaders."""
    fig.canvas.draw()                                   # fix the transforms before measuring
    scale = 72 / fig.dpi
    placed, targets = [], []
    for column in sorted({x for x, _, _ in ends}):      # only labels at one x can collide
        group = sorted(((x, y, text, ax.transData.transform((x, y))[1] * scale)
                        for x, y, text in ends if x == column), key=lambda item: item[3])
        heights = [item[3] for item in group]
        for _ in range(50):                             # push neighbours apart until they clear
            moved = False
            for i in range(1, len(heights)):
                if heights[i] - heights[i - 1] < gap:
                    middle = (heights[i] + heights[i - 1]) / 2
                    heights[i - 1], heights[i] = middle - gap / 2, middle + gap / 2
                    moved = True
            if not moved:
                break
        placed += group
        targets += heights
    for (x, y, text, height), target in zip(placed, targets):
        shift = target - height
        leader = dict(arrowstyle="-", color=colors["muted"], linewidth=0.8, shrinkA=0, shrinkB=4) \
            if abs(shift) > 0.5 else None
        ax.annotate(text, (x, y), xytext=(16 if leader else 9, shift), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=colors["secondary"], arrowprops=leader)


def bar_figure(groups, title, subtitle, xlabel, out, theme):
    """Horizontal bars, one per program, in labelled groups; every bar names its
    program, so identity never rests on color. groups: [(group label,
    [(code, value, text)])]."""
    colors = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    count = sum(len(rows) for _, rows in groups)
    if not count:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * count + 0.5 * len(groups) + 1.4), dpi=100)
    fig.patch.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])
    labels, y, values = [], 0.0, []
    for label, rows in reversed(groups):
        for code, value, text in reversed(rows):
            name, slot, _, _ = CODES[code]
            ax.barh(y, value, height=0.5, color=colors["series"][slot], zorder=3)
            ax.annotate(text, (value, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=colors["secondary"])
            labels.append((y, f"{name}   {label}"))
            values.append(value)
            y += 1
        y += 0.5
    ax.set_yticks([position for position, _ in labels])
    ax.set_yticklabels([text for _, text in labels])
    ax.set_xscale("log")
    lowest = 10 ** int(f"{min(values):e}".split("e")[1])
    ax.set_xlim(lowest, max(values) * 4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(xlabel, color=colors["secondary"])
    ax.grid(axis="x", color=colors["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(colors["axis"])
    ax.tick_params(colors=colors["muted"], labelcolor=colors["secondary"], length=0, pad=6)
    height = fig.get_figheight()
    fig.text(0.015, 1 - 0.12 / height, title, ha="left", va="top", fontsize=12,
             color=colors["ink"], weight="bold")
    fig.text(0.015, 1 - 0.42 / height, subtitle, ha="left", va="top", fontsize=9, color=colors["secondary"])
    fig.subplots_adjust(left=0.4, right=0.95, top=1 - 0.85 / height, bottom=0.65 / height)
    save(fig, out, colors)


def eom_figure(records, out, theme):
    groups = []
    for basis in _bases(records, "eom-ccsdt"):
        rows = []
        for code in ("itercc", "ccpy", "nwchem"):
            record = records.get((code, "eom-ccsdt", basis))
            if record and record["status"] == "ok" and record.get("eom_seconds"):
                rows.append((code, record["eom_seconds"], seconds(record["eom_seconds"])))
        groups.append((BASIS_NAMES[basis], rows))
    bar_figure(groups, "EOM-CCSDT excitation energies",
               f"Water: the three lowest singlet B1 states after the ground state; {MACHINE}. "
               "Lower is faster.",
               "seconds (log scale)", out, theme)


SINGLE_CASES = [("ccsdt", "6-31g"), ("ccsdt", "cc-pvdz"), ("ccsdt", "aug-cc-pvdz"), ("ccsdt", "cc-pvtz"),
                ("ccsdtq", "6-31g")]


def single_core_figure(records, out, theme):
    scaling_figure(records, "ccsdt", "One core: CCSDT time per iteration", out, theme,
                   subtitle="Water, O 1s frozen; one core of a Ryzen 9 3900X for every program "
                            "(ORCA: its Windows build). Lower is faster.", legend_at="lower right")


def single_core_table(records):
    names = dict((code, spec[0]) for code, spec in CODES.items())
    names["pyscf-highm"] = "PySCF 2.14, `_highm`"
    names["pyscf"] = "PySCF 2.14, default"
    print("\nOne core, seconds per iteration (time to the converged energy, iterations):\n")
    print("| | " + " | ".join(f"{m.upper()} {BASIS_NAMES[b]}" for m, b in SINGLE_CASES) + " |")
    print("| --- |" + " --- |" * len(SINGLE_CASES))
    for code in ["itercc", "pyscf-highm", "pyscf", "ccpy", "nwchem", "orca"]:
        cells = []
        for method, basis in SINGLE_CASES:
            record = records.get((code, method, basis))
            cell = _cell(record, "iteration")
            total = _cell(record, "total") if record else ""
            if total and total not in ("-",) and "(" in total:
                cell += f" ({total.split(' (')[0]}, {record['iterations']})"
            cells.append(cell)
        if any(cells):
            print(f"| {names[code]} | " + " | ".join(cells) + " |")


def _bases(records, method):
    found = {b: r["nvir"] for (c, m, b), r in records.items() if m == method and r.get("nvir")}
    return sorted(found, key=found.get)


def _cell(record, field):
    if not record:
        return ""
    if field == "iteration" and iteration_record(record):
        timed = iteration_record(record)
        return seconds(timed["median_iteration_seconds"]) + ("*" if timed.get("partial") else "")
    if record["status"] != "ok":
        return {"memory": "out of memory", "timeout": "timed out"}.get(record["status"], "failed")
    if record.get("partial"):
        return "-"
    if not record.get("converged", True):
        return f"not converged ({record['iterations']} it.)"
    total = record.get("solve_seconds")
    if not total:
        return ""
    memory = "" if record.get("peak_gib") is None else f", {record['peak_gib']:.1f} GiB"
    return f"{seconds(total)} ({record['iterations']} it.{memory})"


def tables(records):
    """Markdown tables for docs/benchmarks.md."""
    names = dict((code, spec[0]) for code, spec in CODES.items())
    names["pyscf-highm"] = "PySCF 2.14, `_highm`"
    names["pyscf"] = "PySCF 2.14, default"
    order = ["itercc", "itercc-nosym", "pyscf-highm", "pyscf", "ccpy", "nwchem"]
    for method in ("ccsd", "ccsdt", "ccsdtq"):
        bases = _bases(records, method)
        if not bases:
            continue
        rows = order
        if method == "ccsd":
            rows = ["itercc", "itercc-nosym", "pyscf", "psi4", "nwchem", "orca", "ccpy"]
            names = dict(names, pyscf="PySCF 2.14", psi4="Psi4 1.11 (fnocc)", nwchem="NWChem 7.3.1 (CCSD)",
                         orca="ORCA 6.1.1 (MDCI)")
        else:
            names = dict(names, pyscf="PySCF 2.14, default", nwchem="NWChem 7.3.1 (TCE)",
                         orca="ORCA 6.1.1 (AUTOCI)")
        header = "| | " + " | ".join(f"{BASIS_NAMES.get(b, b)} ({records_nvir(records, method, b)})"
                                       for b in bases) + " |"
        for field, caption in (("iteration", "seconds per iteration"),
                               ("total", "time from RHF to the converged energy (iterations, peak memory)")):
            print(f"\n{method.upper()}, {caption}:\n")
            print(header)
            print("| --- |" + " --- |" * len(bases))
            for code in rows:
                cells = [_cell(records.get((code, method, b)), field) for b in bases]
                if any(c not in ("", "-") for c in cells):
                    print(f"| {names[code]} | " + " | ".join(cells) + " |")
    print("\nEOM-CCSDT:\n")
    print("| | Basis | Seconds | Roots (eV) |")
    print("| --- | --- | --- | --- |")
    for basis in _bases(records, "eom-ccsdt"):
        for code in ("itercc", "ccpy", "nwchem"):
            r = records.get((code, "eom-ccsdt", basis))
            if r and r["status"] == "ok":
                roots = ", ".join(f"{27.211386 * e:.3f}" for e in r.get("excitation_energies") or [])
                print(f"| {names[code]} | {BASIS_NAMES[basis]} | {seconds(r['eom_seconds'])} | {roots} |")
    print("\nLargest energy difference from IterCC among converged runs (Eh):\n")
    for method in ("ccsd", "ccsdt", "ccsdtq", "eom-ccsdt"):
        for basis in _bases(records, method):
            ref = records.get(("itercc", method, basis))
            if not ref or ref["status"] != "ok" or not ref.get("energy"):
                continue
            spread = {code: abs(r["energy"] - ref["energy"]) for (code, m, b), r in records.items()
                      if m == method and b == basis and r["status"] == "ok" and not r.get("partial")
                      and r.get("converged", True) and r.get("energy")}
            worst = max(spread, key=spread.get)
            print(f"{method:9s} {basis:12s} {spread[worst]:.1e} ({worst})")


def records_nvir(records, method, basis):
    return next(r["nvir"] for (c, m, b), r in records.items() if m == method and b == basis and r.get("nvir"))



def scaling_threads_figure(path, out, theme):
    """IterCC's speedup over one thread, per case, from compare.py records run
    with several --threads values."""
    colors = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    cases = OrderedDict()
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "ok" and r.get("median_iteration_seconds"):
            label = f"{r['method'].upper()} {BASIS_NAMES.get(r['basis'], r['basis'])}"
            cases.setdefault(label, {})[r["threads"]] = r["median_iteration_seconds"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=100)
    fig.patch.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])
    top = max(max(times) for times in cases.values())
    wide = top > 16
    if wide:                                            # 1 .. 56 threads: log-log
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
    ax.plot([1, top], [1, top], color=colors["axis"], linewidth=1, zorder=1)
    ax.annotate("ideal", (top, top), xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=9, color=colors["muted"])
    handles, ends = [], []
    for index, (label, times) in enumerate(cases.items()):
        threads = sorted(times)
        speedup = [times[1] / times[t] for t in threads]
        color = colors["series"][index]
        ax.plot(threads, speedup, color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(threads, speedup, s=52, marker="os^"[index % 3], color=color,
                   edgecolors=colors["surface"], linewidths=2, zorder=4)
        ends.append((threads[-1], speedup[-1], f"{speedup[-1]:.1f}x"))
        handles.append(Line2D([], [], color=color, linewidth=2, marker="os^"[index % 3], markersize=7,
                              markeredgecolor=colors["surface"], label=label))
    ticks = sorted({t for times in cases.values() for t in times})
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if wide:
        ax.set_xlim(0.8, top * 1.6)
        ax.set_ylim(0.8, top * 1.25)
        ax.set_yticks([t for t in (1, 2, 4, 8, 16, 32, 64, 128) if t <= top * 1.25])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_xlim(0.5, top + 1.5)
        ax.set_ylim(0, top + 0.5)
    _end_labels(fig, ax, ends, colors)
    ax.set_xlabel("threads", color=colors["secondary"])
    ax.set_ylabel("speedup over one thread", color=colors["secondary"])
    ax.grid(axis="y", color=colors["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors["axis"])
    ax.tick_params(colors=colors["muted"], labelcolor=colors["secondary"], length=0, pad=6)
    fig.text(0.015, 0.965, "IterCC thread scaling", ha="left", va="top", fontsize=12, color=colors["ink"],
             weight="bold")
    fig.text(0.015, 0.905, f"Water, {SCALING_MACHINE}. Higher is better.",
             ha="left", va="top", fontsize=9, color=colors["secondary"])
    legend = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=colors["ink"])
    legend.set_zorder(5)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.84, bottom=0.13)
    save(fig, out, colors)


def response_table(records):
    """Polarizabilities (au) and the coupled-cluster time after SCF: ground
    state and both response frequencies (Psi4's and NWChem's timings cover
    both; IterCC's are the sum of its ground-state and response solves)."""
    print("\nPolarizabilities, isotropic (au): static, 532 nm; CC time after SCF (ground state + response):\n")
    print("| Method | Program | Basis | static | 532 nm | CC time |")
    print("| --- | --- | --- | --- | --- | --- |")
    for method, label in (("response-ccsd", "CCSD"), ("response", "CCSDT")):
        for basis in _bases(records, method):
            for code in ("itercc", "psi4", "nwchem"):
                r = records.get((code, method, basis))
                if r and r["status"] == "ok" and r.get("alpha_static") is not None:
                    total = r["response_seconds"]
                    if code == "itercc":                      # separate ground-state solve
                        total += r.get("solve_seconds") or 0
                    print(f"| {label} | {NAMES[code]} | {BASIS_NAMES[basis]} | {r['alpha_static']:.4f} | "
                          f"{r['alpha_532']:.4f} | {seconds(total)} |")


def main():
    """plot_compare.py RESULTS OUTDIR [--single ONE_CORE] [--threads SCALING] [--machine TEXT]
    [--cores N] [--nwchem-ranks TEXT] [--scaling-machine TEXT] [--png]"""
    records = load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    option = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None  # noqa: E731
    single = load(option("--single")) if option("--single") else None
    for theme in THEMES:
        scaling_figure(records, "ccsdt", "CCSDT ground state: time per iteration",
                       out / f"ccsdt-scaling-{theme}.svg", theme)
        scaling_figure(records, "ccsdtq", "CCSDTQ ground state: time per iteration",
                       out / f"ccsdtq-scaling-{theme}.svg", theme, legend_at="best")
        if any(m == "ccsd" for (_, m, _) in records):
            scaling_figure(records, "ccsd", "CCSD ground state: time per iteration",
                           out / f"ccsd-scaling-{theme}.svg", theme, codes=CCSD_CODES,
                           subtitle=SUBTITLE if not any(c == "orca" and m == "ccsd" for c, m, _ in records)
                           else f"Water, O 1s frozen; {CORES} cores (NWChem: {NWCHEM_RANKS} MPI ranks; ORCA: up to "
                                f"{CORES}). Lower is faster.")
        eom_figure(records, out / f"eom-ccsdt-{theme}.svg", theme)
        if single:
            single_core_figure(single, out / f"single-core-{theme}.svg", theme)
        if option("--threads"):
            scaling_threads_figure(option("--threads"), out / f"thread-scaling-{theme}.svg", theme)
    tables(records)
    response_table(records)
    if single:
        single_core_table(single)


if __name__ == "__main__":
    main()
