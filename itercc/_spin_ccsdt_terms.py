"""Spin-integrated EOM-CCSDT Jacobian terms for closed-shell references.

The T1-dressed spin-orbital CCSDT residual (`_spin_quadruples_diagrams`,
pdaggerq; each product followed by the normalized external antisymmetrizer)
is differentiated along a direction (r1, r2, r3): T2 and T3 are replaced by
r2 and r3, and the T1 legs of the dressed f and g give r1 terms
(creator legs -r1 g, annihilator legs +g r1; the dressed Fock adds
sum_ie <pi||qe> r1[e,i]). The terms are then integrated over spin onto
canonical blocks of an M_S = 0 direction of spin parity p (+1 singlet, -1
triplet): alpha-dominant blocks a, aa, ab, aaa, aab (particles then holes,
alpha letters first in each half); f is 'a', g = <pq||rs> is 'aaaa' or
'abab' (p, r alpha; q, s beta). A beta-dominant pattern maps onto its alpha-
dominant block by a global spin flip, which multiplies a quantity linear in
the direction by p (`Factor.flipped`). Each output block is antisymmetrized
over its equal-spin axes afterwards.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from math import factorial

from ._spin_quadruples_diagrams import DIAGRAMS

ALPHA, BETA = 0, 1
DIRECTIONS = ("r1", "r2", "r3")
OUTPUT_BLOCKS = {1: ("a",), 2: ("aa", "ab"), 3: ("aaa", "aab")}
# Equal-spin axis groups of each canonical block (antisymmetric within a group).
SAME_SPIN = {"a": (), "aa": ((0, 1), (2, 3)), "ab": (), "aaa": ((0, 1, 2), (3, 4, 5)),
             "aab": ((0, 1), (3, 4)), "aaaa": ((0, 1), (2, 3)), "abab": ()}
VIRTUAL_LETTERS, OCCUPIED_LETTERS = "abcdefwxuv", "ijklmnyzqp"


def parity(order):
    return (-1) ** sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))


@dataclass(frozen=True)
class Factor:
    tensor: str        # 'f', 'g', 't2', 't3', 'r1', 'r2', 'r3'
    block: str         # canonical spin block
    space: str         # 'o'/'v' of each axis of the canonical block
    letters: str       # letters along the block's axes
    flipped: bool      # reached by a global spin flip


@dataclass(frozen=True)
class Term:
    coefficient: float
    output: str        # '3:aab', ...
    letters: str       # output letters along the output block's axes
    factors: tuple


def _amplitude(tensor, letters, spins, spaces):
    n = len(letters) // 2
    parts, holes = letters[:n], letters[n:]
    sp, sh = [spins[c] for c in parts], [spins[c] for c in holes]
    if sum(sp) != sum(sh):
        return None
    flipped = sum(sp) * 2 > n
    if flipped:
        sp, sh = [1 - s for s in sp], [1 - s for s in sh]
    op = sorted(range(n), key=lambda k: (sp[k], k))
    oh = sorted(range(n), key=lambda k: (sh[k], k))
    new = "".join(parts[k] for k in op) + "".join(holes[k] for k in oh)
    block = "".join("ab"[sp[k]] for k in op)
    return parity(op) * parity(oh), Factor(tensor, block, "".join(spaces[c] for c in new), new, flipped)


def _integral(tensor, letters, spins, spaces):
    p, q, r, s = letters
    sp = [spins[c] for c in letters]
    if sorted(sp[:2]) != sorted(sp[2:]):
        return None
    if sp[0] == sp[1]:
        return 1, Factor(tensor, "aaaa", "".join(spaces[c] for c in letters), letters, sp[0] == BETA)
    sign = 1
    if sp[0] == BETA:
        p, q, sign = q, p, -sign
    if sp[2] == BETA:
        r, s, sign = s, r, -sign
    new = p + q + r + s
    return sign, Factor(tensor, "abab", "".join(spaces[c] for c in new), new, False)


def _fock(tensor, letters, spins, spaces):
    if spins[letters[0]] != spins[letters[1]]:
        return None
    return 1, Factor(tensor, "a", "".join(spaces[c] for c in letters), letters, spins[letters[0]] == BETA)


def _canonical(name, letters, spins, spaces):
    if name[0] == "f":
        return _fock("f", letters, spins, spaces)
    if name[0] == "g":
        return _integral("g", letters, spins, spaces)
    return _amplitude(name, letters, spins, spaces)


def _spaces(spec, names):
    inputs = spec.split("->")[0].split(",")
    spaces = {}
    for letters, name in zip(inputs, names):
        kinds = name[1:] if name[0] in "fg" else "v" * (len(letters) // 2) + "o" * (len(letters) // 2)
        for c, k in zip(letters, kinds):
            if spaces.setdefault(c, k) != k:
                raise ValueError(f"Inconsistent orbital space for {c} in {spec}")
    return spaces


def _canonical_key(term):
    order = {}
    for c in term.letters:
        order.setdefault(c, len(order))
    for f in sorted(term.factors, key=lambda f: (f.tensor, f.block, f.space, f.flipped)):
        for c in f.letters:
            order.setdefault(c, len(order))
    rename = {c: chr(ord("A") + i) for c, i in order.items()}
    factors = tuple(sorted((f.tensor, f.block, f.space, "".join(rename[c] for c in f.letters), f.flipped)
                           for f in term.factors))
    return term.output, "".join(rename[c] for c in term.letters), factors


def combine(terms):
    """Sum the coefficients of terms equal up to renaming their letters."""
    total, sample = defaultdict(float), {}
    for t in terms:
        key = _canonical_key(t)
        total[key] += t.coefficient
        sample.setdefault(key, t)
    return [Term(c, sample[k].output, sample[k].letters, sample[k].factors)
            for k, c in total.items() if abs(c) > 1e-14]


def integrate(terms, rank, block):
    """Spin-integrated terms of one output block for spin-orbital terms
    [(coefficient, spec, names)], each followed by the normalized external
    antisymmetrizer of `rank`; the block's equal-spin antisymmetrizer is
    applied afterwards. Each distinct spin labelling of the term's external
    letters (a coset of the equal-spin permutations) is one term, weighted by
    the group order over (rank!)**2."""
    out_spins = [ALPHA if c == "a" else BETA for c in block]
    groups = (factorial(out_spins.count(ALPHA)) * factorial(out_spins.count(BETA))) ** 2
    weight = groups / factorial(rank) ** 2
    result = []
    for coefficient, spec, names in terms:
        inputs, output = spec.split("->")
        inputs = inputs.split(",")
        parts, holes = output[:rank], output[rank:]
        spaces = _spaces(spec, names)
        internal = sorted(set("".join(inputs)) - set(output))
        labellings = {}
        for sigma in permutations(range(rank)):
            for tau in permutations(range(rank)):
                spins = {parts[k]: out_spins[sigma[k]] for k in range(rank)}
                spins.update({holes[k]: out_spins[tau[k]] for k in range(rank)})
                key = tuple(sorted(spins.items()))
                if key in labellings:
                    continue
                axes = [None] * (2 * rank)
                for k in range(rank):
                    axes[sigma[k]] = parts[k]
                    axes[rank + tau[k]] = holes[k]
                labellings[key] = (parity(sigma) * parity(tau), "".join(axes), spins)
        for sign, out_letters, spins in labellings.values():
            for inner in product((ALPHA, BETA), repeat=len(internal)):
                full = dict(spins, **dict(zip(internal, inner)))
                factors, total = [], coefficient * weight * sign
                for letters, name in zip(inputs, names):
                    mapped = _canonical(name, letters, full, spaces)
                    if mapped is None:
                        break
                    total *= mapped[0]
                    factors.append(mapped[1])
                else:
                    result.append(Term(total, f"{rank}:{block}", out_letters, tuple(factors)))
    return combine(result)


def _fresh(spec, kind):
    used = set(spec)
    for c in (VIRTUAL_LETTERS[6:] if kind == "v" else OCCUPIED_LETTERS[6:]):
        if c not in used:
            return c
    raise ValueError("No fresh letter")


def residual_spinorbital(rank):
    return [t for t in DIAGRAMS[rank] if "t4" not in t[2]]


def jacobian_spinorbital(rank):
    """Spin-orbital terms of the rank-`rank` part of J r."""
    out = []
    for coefficient, spec, names in residual_spinorbital(rank):
        inputs, output = spec.split("->")
        inputs = inputs.split(",")
        for k, name in enumerate(names):
            rest_in, rest_names = inputs[:k] + inputs[k + 1:], names[:k] + names[k + 1:]

            def add(c, new_inputs, new_names):
                out.append((c, ",".join(new_inputs) + "->" + output, tuple(new_names)))

            if name in ("t2", "t3"):
                add(coefficient, inputs, names[:k] + ("r" + name[1],) + names[k + 1:])
                continue
            letters, space, tag = inputs[k], name[1:], name[0]
            for leg, (letter, s) in enumerate(zip(letters, space)):
                creator = leg < len(letters) // 2
                if creator and s == "v":            # -sum_m r1[p,m] x[m...]
                    m = _fresh(spec, "o")
                    add(-coefficient, rest_in + [letters[:leg] + m + letters[leg + 1:], letter + m],
                        rest_names + (tag + space[:leg] + "o" + space[leg + 1:], "r1"))
                elif not creator and s == "o":      # +sum_e r1[e,r] x[...e...]
                    e = _fresh(spec, "v")
                    add(coefficient, rest_in + [letters[:leg] + e + letters[leg + 1:], e + letter],
                        rest_names + (tag + space[:leg] + "v" + space[leg + 1:], "r1"))
            if tag == "f":                          # + sum_ie <p i||q e> r1[e,i]
                p, q = letters
                i, e = _fresh(spec, "o"), _fresh(spec, "v")
                add(coefficient, rest_in + [p + i + q + e, e + i],
                    rest_names + ("g" + space[0] + "o" + space[1] + "v", "r1"))
    return out


@lru_cache(maxsize=None)
def jacobian_terms():
    """The spin-integrated terms of J r, each with exactly one direction factor."""
    terms = []
    for rank in (1, 2, 3):
        spinorbital = jacobian_spinorbital(rank)
        for block in OUTPUT_BLOCKS[rank]:
            terms.extend(integrate(spinorbital, rank, block))
    return tuple(terms)


@lru_cache(maxsize=None)
def residual_terms():
    """The spin-integrated terms of the ground residual (for tests)."""
    terms = []
    for rank in (1, 2, 3):
        for block in OUTPUT_BLOCKS[rank]:
            terms.extend(integrate(residual_spinorbital(rank), rank, block))
    return tuple(terms)
