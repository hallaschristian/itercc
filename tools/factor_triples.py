"""Group equivalent triples diagrams from the pinned T1-dressed equations."""

import ast
from collections import defaultdict
from pathlib import Path
import re
import sys


def parity(indices):
    return (-1) ** sum(a > b for i, a in enumerate(indices) for b in indices[i + 1:])


def canonical(spec):
    """Rename external virtual/occupied labels and dummy labels consistently."""
    inputs, output = spec.split("->")
    ordered = tuple(dict.fromkeys(inputs.replace(",", "")))
    groups = ("abc", "ijk")
    mapping = {}
    for group in groups:
        mapping.update(zip((x for x in ordered if x in group), group))
    mapping.update(zip((x for x in ordered if x not in output), "deflmn"))
    sign = parity(tuple(mapping[x] for x in output[:3]))
    sign *= parity(tuple(mapping[x] for x in output[3:]))
    return "".join(mapping.get(x, x) for x in inputs) + "->" + output, sign


def append_factored(source):
    node = next(x for x in ast.parse(source).body
                if isinstance(x, ast.FunctionDef) and x.name == "triples")
    groups = defaultdict(float)
    ladder = 0
    for line in ast.get_source_segment(source, node).splitlines()[2:]:
        if "einsum(" in line or "ladder(" in line:
            target, expression = re.split(r" \+?= ", line.strip())
            weight = 1
            if " * einsum(" in expression:
                coefficient, expression = expression.split(" * ", 1)
                weight = float(coefficient)
            call = ast.parse(expression, mode="eval").body
            if call.func.id == "ladder":
                key, term = None, weight
            else:
                spec, sign = canonical(call.args[0].value)
                operands = tuple(map(ast.unparse, call.args[1:]))
                # Keep the original contraction path, which is invariant to
                # index renaming and does not depend on axis ordering.
                keywords = tuple(map(ast.unparse, call.keywords))
                key, term = (spec, operands, keywords), sign * weight
            if target == "triples_res":
                if key is None:
                    ladder += term
                else:
                    groups[key] += term
        elif "contracted_intermediate" in line:
            if ".transpose" in line:
                axes = ast.parse(line.split(".transpose", 1)[1], mode="eval").body
                p = ast.literal_eval(axes)
                sign = parity(p[:3]) * parity(p[3:])
            else:
                sign = 1
            sign *= -1 if "-=" in line else 1
            if key is None:
                ladder += sign * term
            else:
                groups[key] += sign * term
    lines = ["", "", "def triples_factored(f, g, t2, t3, ladder, einsum=einsum):",
             "    # Accumulate one representative of each orbit of external permutations.",
             "    # Final antisymmetrization accounts for their signed multiplicities.",
             "    r = t3 * 0.0"]
    for (spec, operands, keywords), weight in groups.items():
        if spec == "abde,decijk->abcijk" and operands == ("g['vvvv']", "t3"):
            ladder += 2 * weight
        elif weight:
            args = ", ".join((repr(spec), *operands, *keywords))
            lines.append(f"    r += {weight} * einsum({args})")
    lines.append(f"    r += {float(ladder)} * ladder(g['vvvv'], t3)")
    lines.extend([
        "    r = _antisymmetrize(r, 0)",
        "    return _antisymmetrize(r, 3)",
        "",
        "",
        "def _antisymmetrize(tensor, first):",
        "    # A3 = (1 - P12 - P13) (1 - P23) / 6.",
        "    r = tensor - tensor.swapaxes(first + 1, first + 2)",
        "    out = r.copy()",
        "    out -= r.swapaxes(first, first + 1)",
        "    out -= r.swapaxes(first, first + 2)",
        "    out /= 6.0",
        "    return out",
    ])
    return source.rstrip() + "\n".join(lines) + "\n"


if __name__ == "__main__":
    src, dst = map(Path, sys.argv[1:])
    dst.write_text(append_factored(src.read_text(encoding="utf-8")), encoding="utf-8")

