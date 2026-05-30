"""repro_sympy_hang.py

Reproduce the post-evolution hang reported on bioavailability seed=57.

We monkeypatch slim_gsgp.utils.simplification.simplify_tree_sympy with an
instrumented version that times each individual Sympy operation under a
hard per-call SIGALRM timeout, logging which builder is slow / hangs and
on what input size. Then we run slim() with the exact config the user
used so the same Pareto frontier is produced.
"""
from __future__ import annotations

import os
import sys
import time
import signal

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, os.path.dirname(_THIS))

import slim_gsgp.utils.simplification as simp
from slim_gsgp.utils.simplification import (
    _tree_to_sympy,
    _sympy_to_tree,
    _count_tree_nodes,
)


class _Timeout(Exception):
    pass


def _call_with_timeout(fn, seconds):
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    t0 = time.time()
    try:
        return fn(), time.time() - t0, False
    except _Timeout:
        return None, time.time() - t0, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def instrumented_simplify(tree_structure, constants_dict, max_input_nodes=200):
    import sympy as sp
    if tree_structure is None:
        return tree_structure, 0, False
    original = _count_tree_nodes(tree_structure)
    if original > max_input_nodes:
        print(f"    [skip] input has {original} nodes > cap {max_input_nodes}")
        return tree_structure, 0, False

    cache = {}
    expr = _tree_to_sympy(tree_structure, constants_dict, sp, cache)
    if expr is None:
        print(f"    [skip] tree_to_sympy returned None ({original} nodes)")
        return tree_structure, 0, False

    print(f"    individual: converted={original} nodes")
    for name, builder in (
        ("simplify", sp.simplify),
        ("cancel", sp.cancel),
        ("together", sp.together),
        ("factor", sp.factor),
        ("expand", sp.expand),
    ):
        res, dt, timed_out = _call_with_timeout(lambda b=builder: b(expr), seconds=5.0)
        flag = "TIMEOUT" if timed_out else f"{dt*1000:7.1f} ms"
        print(f"      {name:<10} {flag}")
        if timed_out:
            print(f"      >>> '{name}' is the culprit (input {original} nodes)")

    return tree_structure, 0, False


def main():
    simp.simplify_tree_sympy = instrumented_simplify

    from slim_gsgp.main_slim import slim
    from slim_gsgp.datasets.data_loader import load_bioav
    from slim_gsgp.utils.utils import train_test_split

    seed = 57
    X, y = load_bioav(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)

    print(f"Running slim() on bioavailability seed={seed} (SLIM+N1, pop=100, n_iter=100)...")
    print("Instrumented simplify_tree_sympy will time each Sympy op (5s cap).\n")

    results = slim(
        X_train=X_train, y_train=y_train,
        X_test=X_val, y_test=y_val,
        dataset_name="bioavailability",
        slim_version="SLIM+N1",
        pop_size=100, n_iter=100,
        ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True,
        tournament_type="pareto", tournament_size=5,
        multi_obj_attrs=["fitness", "size"],
        use_simplification=True,
        seed=seed, verbose=0,
    )
    print("\nDone — slim() returned without hanging (instrumented bypass).")


if __name__ == "__main__":
    main()
