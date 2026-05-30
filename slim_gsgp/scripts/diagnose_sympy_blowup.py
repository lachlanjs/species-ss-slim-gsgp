"""diagnose_sympy_blowup.py

Find the ROOT CAUSE of the Sympy blow-up, not just confirm the timeout works.

Hypothesis: the converted SLIM trees contain many `divide` nodes. When Sympy
tries to simplify a sum of terms with independent denominators, it places them
over a common denominator (= product of all denominators), whose polynomial
expansion grows combinatorially. That, not tree size per se, is what makes
simplify/factor/cancel explode.

Method:
  1. Run slim() on bioavailability seed=57 (the hanging case) and grab the
     converted-tree view of every Pareto-frontier individual.
  2. For each tree, measure:
       - node count
       - number of `divide` nodes
       - time for sympy.simplify under a 5s probe timeout
  3. Build a "division-free twin" of each tree (replace every `divide` with
     `multiply`) and time sympy.simplify on it.
  4. If the division-free twins are fast while the originals time out, the
     root cause is divisions, and the structural fix is to avoid rational
     recombination (e.g. simplify per-denominator-free subtree, or skip the
     expensive passes when divide-count is high).
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

import slim_gsgp.utils.simplification as simp
from slim_gsgp.utils.simplification import (
    convert_slim_individual_to_normal_tree,
    _tree_to_sympy,
    _count_tree_nodes,
)


class _Timeout(Exception):
    pass


def _timed(fn, seconds):
    def _h(s, f):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    t0 = time.time()
    try:
        fn()
        return time.time() - t0, False
    except _Timeout:
        return time.time() - t0, True
    except Exception:
        return time.time() - t0, False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _count_op(tree, opname):
    stack = [tree]
    n = 0
    while stack:
        node = stack.pop()
        if isinstance(node, tuple):
            if str(node[0]) == opname:
                n += 1
            stack.extend(node[1:])
    return n


def _replace_divide_with_multiply(tree):
    if not isinstance(tree, tuple):
        return tree
    op = "multiply" if str(tree[0]) == "divide" else tree[0]
    return (op,) + tuple(_replace_divide_with_multiply(c) for c in tree[1:])


# Capture converted trees of the Pareto frontier by monkeypatching
# simplify_population so we intercept the exact inputs.
_captured = []


def _capturing_simplify_tree_sympy(tree_structure, constants_dict, **kw):
    _captured.append((tree_structure, constants_dict))
    # Return unchanged so the run finishes fast; we analyze offline.
    return tree_structure, 0, False


def main():
    simp.simplify_tree_sympy = _capturing_simplify_tree_sympy

    from slim_gsgp.main_slim import slim
    from slim_gsgp.datasets.data_loader import load_bioav
    from slim_gsgp.utils.utils import train_test_split

    seed = 57
    X, y = load_bioav(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)

    print(f"Running bioavailability seed={seed} to capture Pareto-frontier trees...\n")
    slim(
        X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
        dataset_name="bioavailability", slim_version="SLIM+N1",
        pop_size=100, n_iter=100, ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True, tournament_type="pareto", tournament_size=5,
        multi_obj_attrs=["fitness", "size"], use_simplification=True,
        seed=seed, verbose=0,
    )

    print(f"Captured {len(_captured)} frontier trees.\n")
    hdr = f"{'#':>3} {'nodes':>6} {'divides':>8} {'simplify(orig)':>16} {'simplify(no-div)':>18}"
    print(hdr)
    print("-" * len(hdr))

    for i, (tree, cdict) in enumerate(_captured):
        nodes = _count_tree_nodes(tree)
        ndiv = _count_op(tree, "divide")

        cache = {}
        expr = _tree_to_sympy(tree, cdict, sp, cache)
        if expr is None:
            print(f"{i:>3} {nodes:>6} {ndiv:>8}   <tree_to_sympy None>")
            continue
        dt_o, to_o = _timed(lambda e=expr: sp.simplify(e), 5.0)

        # division-free twin
        tree_nodiv = _replace_divide_with_multiply(tree)
        cache2 = {}
        expr_nodiv = _tree_to_sympy(tree_nodiv, cdict, sp, cache2)
        dt_n, to_n = _timed(lambda e=expr_nodiv: sp.simplify(e), 5.0)

        f_o = "TIMEOUT" if to_o else f"{dt_o*1000:8.1f} ms"
        f_n = "TIMEOUT" if to_n else f"{dt_n*1000:8.1f} ms"
        print(f"{i:>3} {nodes:>6} {ndiv:>8} {f_o:>16} {f_n:>18}")

    print("""
INTERPRETATION
--------------
If trees with high 'divides' TIMEOUT in the orig column but their
division-free twins are fast, divisions (common-denominator blow-up) are the
root cause — and the timeout is the correct mitigation at the algebra layer,
while the structural fix is to avoid expensive rational recombination.
""")


if __name__ == "__main__":
    main()
