"""measure_opaque_division_loss.py

Question: if we make divisions OPAQUE to Sympy (so it never combines a sum of
fractions over a common denominator), how much node-reduction do we actually
lose versus full Sympy?

Hypothesis: almost none — because combining fractions over a common
denominator GROWS the tree (we'd reject it anyway), while the reductions that
matter (cancel within a fraction, fold constants, x/x->1) survive even with
opaque divisions, since Sympy still simplifies the numerator/denominator
arguments of an opaque function.

Method: take real converted trees (division-light and division-heavy) and
compare final node counts after:
  (1) FULL sympy   (simplify/cancel/together/factor/expand, 5s probe cap)
  (2) OPAQUE-div   (map divide -> Function('pdiv'); Sympy simplifies args but
                    never merges denominators)
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

_PDIV = sp.Function("pdiv")  # opaque protected-division placeholder


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


# --- conversions -----------------------------------------------------------

def tree_to_sympy(structure, cdict, cache, opaque_div):
    """Like the production _tree_to_sympy but optionally maps divide to the
    opaque pdiv function."""
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
        a = tree_to_sympy(structure[1], cdict, cache, opaque_div)
        return None if a is None else -a
    if len(structure) == 3:
        l = tree_to_sympy(structure[1], cdict, cache, opaque_div)
        r = tree_to_sympy(structure[2], cdict, cache, opaque_div)
        if l is None or r is None:
            return None
        if op == "add":
            return l + r
        if op == "subtract":
            return l - r
        if op == "multiply":
            return l * r
        if op == "divide":
            return _PDIV(l, r) if opaque_div else l / r
    return None


def sympy_to_tree_opaque(expr):
    """Convert back, turning pdiv(a,b) into ('divide', a, b)."""
    if expr.func == _PDIV:
        a, ok1 = sympy_to_tree_opaque(expr.args[0])
        b, ok2 = sympy_to_tree_opaque(expr.args[1])
        if not (ok1 and ok2):
            return None, False
        return ("divide", a, b), True
    return _sympy_to_tree(expr, sp)


def best_nodes(structure, cdict, opaque_div, cap=5.0):
    n_in = _count_tree_nodes(structure)
    cache = {}
    expr = tree_to_sympy(structure, cdict, cache, opaque_div)
    if expr is None:
        return n_in, 0.0, False
    builders = [sp.simplify, sp.cancel, sp.together, sp.factor, sp.expand]
    best = n_in
    total_t = 0.0
    timed_out = False
    for b in builders:
        res, dt, to = _timed(lambda e=expr, bb=b: bb(e), cap)
        total_t += dt
        if to:
            timed_out = True
            continue
        if res is None:
            continue
        try:
            if sp.count_ops(res) >= best:
                continue
            back, ok = (sympy_to_tree_opaque(res) if opaque_div
                        else _sympy_to_tree(res, sp))
        except Exception:
            continue
        if ok and back is not None:
            c = _count_tree_nodes(back)
            if c < best:
                best = c
    return best, total_t, timed_out


def main():
    seeds = [57, 50, 60, 45, 70]
    print(f"{'seed':>4} {'role':<16} {'div':>4} {'orig':>5} "
          f"{'FULL nd':>8} {'FULL t':>9} {'OPAQUE nd':>10} {'OPAQUE t':>9}")
    print("-" * 76)
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
            full_n, full_t, full_to = best_nodes(tree, cd, opaque_div=False)
            opaq_n, opaq_t, opaq_to = best_nodes(tree, cd, opaque_div=True)
            fts = f"{full_t*1000:6.0f}ms" + ("!" if full_to else " ")
            ots = f"{opaq_t*1000:6.0f}ms" + ("!" if opaq_to else " ")
            print(f"{seed:>4} {role:<16} {ndiv:>4} {orig:>5} "
                  f"{full_n:>8} {fts:>9} {opaq_n:>10} {ots:>9}")
    print("""
'!' marks that at least one Sympy op timed out (5s).
FULL nd  = final node count with full rational simplification.
OPAQUE nd= final node count with divisions opaque.
If OPAQUE nd ≈ FULL nd, we lose ~no real reduction by making divisions opaque,
while OPAQUE t stays small (no blow-up).
""")


if __name__ == "__main__":
    main()
