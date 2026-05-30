"""measure_chunked_simplify.py

Test the "do it in chunks" idea rigorously.

CHUNKED strategy: walk the Sympy expression bottom-up; at every Add node with
more than `group` terms, split the terms into groups of size `group`,
fully-simplify each group (bounded: a group of k fractions costs ~2^k, tiny
for k<=4), and re-assemble WITHOUT combining the groups. This bounds the
exponential blow-up while still combining small clusters of fractions.

We compare, on real converted SLIM trees:
    orig        - input node count
    FULL        - full sympy (5s probe cap per op)
    OPAQUE      - divisions opaque (never combine fractions)
    CHUNKED(3)  - chunked cancel/simplify, group size 3

If CHUNKED beats OPAQUE and stays fast, "por partes" genuinely recovers
reduction. If CHUNKED ~ FULL on the cases where FULL wins (and both stay
small where FULL explodes), it confirms chunking is viable. If CHUNKED ~
OPAQUE, chunking adds nothing over the simpler hybrid.
"""
from __future__ import annotations

import os
import sys
import time
import signal

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, os.path.dirname(_THIS))

import sympy as sp

from slim_gsgp.main_slim import slim
from slim_gsgp.datasets.data_loader import load_bioav
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.utils.simplification import (
    convert_slim_individual_to_normal_tree,
    _count_tree_nodes,
    _clean_node_name,
    _resolve_constant_value,
    _sympy_to_tree,
)


class _TO(Exception):
    pass


def _timed(fn, seconds):
    def _h(s, f):
        raise _TO()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    t0 = time.time()
    try:
        return fn(), time.time() - t0, False
    except _TO:
        return None, time.time() - t0, True
    except Exception:
        return None, time.time() - t0, False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _safe(fn, expr, seconds=1.0):
    res, _, _ = _timed(lambda: fn(expr), seconds)
    return res if res is not None else expr


def tree_to_sympy(structure, cdict, cache):
    if not isinstance(structure, tuple):
        if isinstance(structure, (int, float)):
            return sp.Float(float(structure))
        name = _clean_node_name(structure)
        cv = _resolve_constant_value(name, cdict)
        if cv is not None:
            return sp.Float(cv)
        if name not in cache:
            cache[name] = sp.Symbol(name)
        return cache[name]
    op = _clean_node_name(structure[0])
    if len(structure) == 2 and op.lower() in ("negative", "neg"):
        a = tree_to_sympy(structure[1], cdict, cache)
        return None if a is None else -a
    if len(structure) == 3:
        l = tree_to_sympy(structure[1], cdict, cache)
        r = tree_to_sympy(structure[2], cdict, cache)
        if l is None or r is None:
            return None
        return {"add": lambda: l + r, "subtract": lambda: l - r,
                "multiply": lambda: l * r, "divide": lambda: l / r}[op]()
    return None


def chunked_simplify(expr, group=3, per_op=1.0, _depth=0):
    """Bottom-up; bound every cancel/simplify to <= `group` fraction terms."""
    if expr.is_Atom or _depth > 60:
        return expr
    new_args = tuple(chunked_simplify(a, group, per_op, _depth + 1) for a in expr.args)
    rebuilt = expr.func(*new_args)
    if rebuilt.is_Add and len(rebuilt.args) > group:
        terms = list(rebuilt.args)
        out = sp.Integer(0)
        for i in range(0, len(terms), group):
            chunk = sp.Add(*terms[i:i + group])
            out = out + _safe(sp.cancel, chunk, per_op)
        # cheap, non-exponential like-term collection across the whole thing
        return _safe(sp.expand, out, per_op)
    if rebuilt.is_Add:
        return _safe(sp.cancel, rebuilt, per_op)
    return rebuilt


def full_best(expr, orig):
    best = orig
    total = 0.0
    for b in (sp.simplify, sp.cancel, sp.together, sp.factor, sp.expand):
        res, dt, to = _timed(lambda bb=b: bb(expr), 5.0)
        total += dt
        if res is None:
            continue
        try:
            if sp.count_ops(res) >= best:
                continue
            back, ok = _sympy_to_tree(res, sp)
        except Exception:
            continue
        if ok and back is not None:
            best = min(best, _count_tree_nodes(back))
    return best, total


def opaque_best(structure, cdict, orig):
    # crude opaque: replace divide subtrees' combination by not running cancel;
    # reuse full but only cheap ops (no cancel/together/factor on sums)
    cache = {}
    expr = tree_to_sympy(structure, cdict, cache)
    if expr is None:
        return orig, 0.0
    best = orig
    total = 0.0
    for b in (sp.expand,):  # only the non-fraction-combining cheap op
        res, dt, to = _timed(lambda bb=b: bb(expr), 5.0)
        total += dt
        if res is None:
            continue
        try:
            if sp.count_ops(res) >= best:
                continue
            back, ok = _sympy_to_tree(res, sp)
        except Exception:
            continue
        if ok and back is not None:
            best = min(best, _count_tree_nodes(back))
    return best, total


def chunked_best(structure, cdict, orig, group=3):
    cache = {}
    expr = tree_to_sympy(structure, cdict, cache)
    if expr is None:
        return orig, 0.0
    res, dt, to = _timed(lambda: chunked_simplify(expr, group=group), 5.0)
    if res is None:
        return orig, dt
    try:
        if sp.count_ops(res) >= orig:
            return orig, dt
        back, ok = _sympy_to_tree(res, sp)
        if ok and back is not None:
            return min(orig, _count_tree_nodes(back)), dt
    except Exception:
        pass
    return orig, dt


def main():
    seeds = [57, 50, 60, 45, 70]
    print(f"{'seed':>4} {'role':<16} {'div':>4} {'orig':>5} "
          f"{'FULL':>5} {'FULLt':>8} {'OPAQ':>5} {'CHUNK3':>7} {'CHUNKt':>8}")
    print("-" * 74)
    for seed in seeds:
        X, y = load_bioav(X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, p_test=0.4, seed=seed)
        Xv, Xte, yv, yte = train_test_split(Xte, yte, p_test=0.5, seed=seed)
        res = slim(
            X_train=Xtr, y_train=ytr, X_test=Xv, y_test=yv,
            dataset_name="bioavailability", slim_version="SLIM+N1",
            pop_size=100, n_iter=100, ms_lower=0, ms_upper=1, p_inflate=0.5,
            reconstruct=True, tournament_type="pareto", tournament_size=5,
            multi_obj_attrs=["fitness", "size"], use_simplification=False,
            seed=seed, verbose=0,
        )
        for role, ind in (("best_fitness", res.best_fitness),
                          ("best_normalized", res.best_normalized)):
            tree, dicts = convert_slim_individual_to_normal_tree(ind)
            if tree is None:
                continue
            cd = dicts["CONSTANTS"]
            ndiv = str(tree).count("'divide'")
            orig = _count_tree_nodes(tree)
            cache = {}
            expr = tree_to_sympy(tree, cd, cache)
            full_n, full_t = full_best(expr, orig) if expr is not None else (orig, 0)
            opaq_n, _ = opaque_best(tree, cd, orig)
            chk_n, chk_t = chunked_best(tree, cd, orig, group=3)
            print(f"{seed:>4} {role:<16} {ndiv:>4} {orig:>5} "
                  f"{full_n:>5} {full_t*1000:>6.0f}ms {opaq_n:>5} "
                  f"{chk_n:>7} {chk_t*1000:>6.0f}ms")
    print("""
FULL = full sympy (may explode).  OPAQ = no fraction-combining.
CHUNK3 = chunked cancel, groups of 3 fractions (bounded).
""")


if __name__ == "__main__":
    main()
