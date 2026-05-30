"""measure_sympy_impact.py

Diagnose how much work Sympy actually does in the simplification pipeline.

For each individual produced by a real slim() run, we report:

  - original_nodes  : node count of the converted "normal tree"
  - after_sympy     : node count after simplify_tree_sympy
  - after_simple    : node count after simplify_constant_operations
                      (i.e. what simplify_population ends up using)
  - sympy_applied   : True iff Sympy strictly reduced the tree
  - sympy_removed   : nodes removed by the Sympy pass alone
  - simple_removed  : nodes removed by the conservative pass after Sympy

This makes it visible whether Sympy is actually contributing or silently
bailing out (e.g. because the converted tree contains an operator outside
the {add, subtract, multiply, divide} set that the Sympy round-trip
supports).

Run from the repo root::

    cd slim_gsgp
    python scripts/measure_sympy_impact.py
"""
from __future__ import annotations

import os
import sys

# Make `slim_gsgp.*` importable when invoked directly from this folder.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_THIS_DIR))  # repo root
sys.path.insert(0, _PKG_PARENT)
sys.path.insert(0, os.path.dirname(_THIS_DIR))             # slim_gsgp/

from slim_gsgp.main_slim import slim
from slim_gsgp.datasets.data_loader import load_airfoil
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.utils.simplification import (
    convert_slim_individual_to_normal_tree,
    simplify_tree_sympy,
    simplify_constant_operations,
    _count_tree_nodes,
)


def _collect_op_names(structure, out):
    """Walk a tree and record every operator/leaf token encountered."""
    if not isinstance(structure, tuple):
        return
    out.add(str(structure[0]))
    for child in structure[1:]:
        _collect_op_names(child, out)


def run_for_version(slim_version, seed=42, pop_size=50, n_iter=20):
    print(f"\n{'=' * 78}")
    print(f"  {slim_version}  (seed={seed}, pop_size={pop_size}, n_iter={n_iter})")
    print(f"{'=' * 78}")

    X, y = load_airfoil(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)

    results = slim(
        X_train=X_train, y_train=y_train,
        X_test=X_val, y_test=y_val,
        dataset_name="airfoil",
        slim_version=slim_version,
        pop_size=pop_size, n_iter=n_iter,
        ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True, seed=seed, verbose=0,
    )

    individuals = [
        ("best_fitness", results.best_fitness),
        ("best_normalized", results.best_normalized),
        ("smallest", results.smallest),
    ]

    # Also drill into the full final population so we have a richer sample.
    # (When available — the wrapper above only exposes the three picks.)
    extra_pop = getattr(results.best_fitness, "_population", None)

    totals = {
        "n": 0,
        "sympy_applied": 0,
        "sympy_removed": 0,
        "simple_removed": 0,
        "skipped_unknown_op": 0,
    }
    all_ops = set()

    header = f"{'role':<18} {'orig':>5} {'after_sym':>10} {'after_simp':>11} {'sym_applied':>13} {'sym-':>5} {'simp-':>6}"
    print(header)
    print("-" * len(header))

    for role, ind in individuals:
        tree_structure, tree_dicts = convert_slim_individual_to_normal_tree(ind)
        if tree_structure is None:
            print(f"  {role:<18}  <conversion returned None>")
            continue

        seen_ops: set = set()
        _collect_op_names(tree_structure, seen_ops)
        all_ops |= seen_ops

        original = _count_tree_nodes(tree_structure)
        sympy_tree, sympy_removed, sympy_applied = simplify_tree_sympy(
            tree_structure, tree_dicts["CONSTANTS"]
        )
        after_sympy = _count_tree_nodes(sympy_tree)

        simp_tree, simp_removed = simplify_constant_operations(
            sympy_tree, tree_dicts["CONSTANTS"]
        )
        after_simp = _count_tree_nodes(simp_tree)

        totals["n"] += 1
        if sympy_applied:
            totals["sympy_applied"] += 1
        totals["sympy_removed"] += sympy_removed
        totals["simple_removed"] += max(after_sympy - after_simp, 0)

        # Detect whether the input tree contained any token outside the
        # 4-operator set — those are the ones that force the Sympy pass to
        # bail.
        SUPPORTED = {"add", "subtract", "multiply", "divide", "negative", "neg"}
        unknown = {op for op in seen_ops if op not in SUPPORTED and not _looks_like_leaf(op)}
        if unknown and not sympy_applied:
            totals["skipped_unknown_op"] += 1

        print(
            f"  {role:<18} {original:>5d} {after_sympy:>10d} {after_simp:>11d} "
            f"{('yes' if sympy_applied else 'no'):>13} {sympy_removed:>5d} {(after_sympy - after_simp):>6d}"
        )
        if unknown:
            print(f"      unknown ops in tree (forced fallback): {sorted(unknown)}")

    print()
    print(f"  Summary: sympy_applied={totals['sympy_applied']}/{totals['n']} "
          f"individuals  | total sympy_removed={totals['sympy_removed']}  | "
          f"total post-simple_removed={totals['simple_removed']}")
    if all_ops:
        SUPPORTED = {"add", "subtract", "multiply", "divide", "negative", "neg"}
        unknown_all = {op for op in all_ops if op not in SUPPORTED and not _looks_like_leaf(op)}
        if unknown_all:
            print(f"  Operator tokens that bypass Sympy: {sorted(unknown_all)}")
        else:
            print(f"  All operator tokens were Sympy-supported.")


def _looks_like_leaf(token: str) -> bool:
    """Heuristic: anything that isn't a binary/unary operator name we
    recognize might just be a leaf (variable/constant). Returns True if
    the token looks like a terminal."""
    if token.startswith("x") and token[1:].isdigit():
        return True
    if token.startswith("constant_"):
        return True
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def main():
    for v in ["SLIM+ABS", "SLIM*ABS", "SLIM+SIG2", "SLIM+N1", "SLIM*N1"]:
        run_for_version(v, seed=42, pop_size=50, n_iter=20)


if __name__ == "__main__":
    main()
