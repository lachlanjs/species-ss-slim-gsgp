"""compare_abs_vs_n1_divisions.py

Why does the Sympy hang show up with SLIM+N1 but (apparently) not SLIM+ABS?

ABS and N1 share (op, sig, trees) = (sum, False, False); the only difference
is normalized mutation. So any difference in the converted trees must come
from WHICH individuals each variant evolves, not from a different structural
template.

This script runs both variants on the same dataset across several seeds and
reports, for each returned individual:
  - n_blocks
  - converted node count
  - total `divide` nodes in the converted tree
  - max divides inside a single block's random tree
  - sympy.simplify timing (probe, 5s cap)

If N1 consistently shows more divisions / longer sympy times than ABS, the
difference is systematic (normalized mutation favours division-heavy random
trees). If it's noisy, it's a statistical accident of the search.
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
    _tree_to_sympy,
    _count_tree_nodes,
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
        fn(); return time.time() - t0, False
    except _TO:
        return time.time() - t0, True
    except Exception:
        return time.time() - t0, False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _count_divides(tree):
    stack = [tree]; n = 0
    while stack:
        node = stack.pop()
        if isinstance(node, tuple):
            if str(node[0]) == "divide":
                n += 1
            stack.extend(node[1:])
    return n


def _per_block_max_divides(ind):
    """Max divide-count across the individual's blocks' base trees."""
    best = 0
    for tree in getattr(ind, "collection", []):
        struct = tree.structure
        if isinstance(struct, tuple):
            best = max(best, _count_divides(struct))
        elif isinstance(struct, list):
            for el in struct[1:]:
                if hasattr(el, "structure") and isinstance(el.structure, tuple):
                    best = max(best, _count_divides(el.structure))
    return best


def run_variant(slim_version, seed):
    X, y = load_bioav(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)
    # Disable simplification so the run never hangs while we collect raw trees.
    return slim(
        X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
        dataset_name="bioavailability", slim_version=slim_version,
        pop_size=100, n_iter=100, ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True, tournament_type="pareto", tournament_size=5,
        multi_obj_attrs=["fitness", "size"], use_simplification=False,
        seed=seed, verbose=0,
    )


def main():
    seeds = [57, 50, 60]
    print(f"{'version':<10} {'seed':>4} {'role':<16} {'blocks':>6} {'conv':>5} "
          f"{'tot_div':>7} {'max_blk_div':>11} {'simplify':>10}")
    print("-" * 84)
    for seed in seeds:
        for version in ("SLIM+ABS", "SLIM+N1"):
            res = run_variant(version, seed)
            for role, ind in (("best_fitness", res.best_fitness),
                              ("best_normalized", res.best_normalized),
                              ("smallest", res.smallest)):
                tree, dicts = convert_slim_individual_to_normal_tree(ind)
                if tree is None:
                    continue
                nodes = _count_tree_nodes(tree)
                tot_div = _count_divides(tree)
                blk_div = _per_block_max_divides(ind)
                expr = _tree_to_sympy(tree, dicts["CONSTANTS"], sp, {})
                if expr is not None:
                    dt, to = _timed(lambda e=expr: sp.simplify(e), 5.0)
                    tstr = "TIMEOUT" if to else f"{dt*1000:7.1f}ms"
                else:
                    tstr = "N/A"
                print(f"{version:<10} {seed:>4} {role:<16} "
                      f"{len(ind.collection):>6} {nodes:>5} {tot_div:>7} "
                      f"{blk_div:>11} {tstr:>10}")
        print()


if __name__ == "__main__":
    main()
