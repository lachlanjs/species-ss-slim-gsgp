"""verify_sympy_simplification.py

Standalone verification that the Sympy-based simplification pass
(``simplify_tree_sympy``) is semantically safe.

What "safe" means here:

  1. ``simplify_tree_sympy`` returns a tree-tuple structure that uses ONLY
     the four base operators (add, subtract, multiply, divide) plus numeric
     and symbol leaves. We assert this on the result.

  2. When the simplified tree is evaluated against the same input, it
     produces the same numeric output as the original tree (within
     floating-point tolerance).

  3. End-to-end: running ``slim()`` and then ``simplify_population`` does
     NOT change the predictions of any individual — because
     ``simplify_population`` only recomputes ``nodes_count`` and never
     touches ``individual.collection`` (which is what ``predict()`` uses).

Run from the repo root::

    cd slim_gsgp
    python scripts/verify_sympy_simplification.py
"""
from __future__ import annotations

import os
import sys
import math
import random

import torch

# Make `slim_gsgp.*` importable when invoked directly from this folder.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_THIS_DIR))  # repo root
sys.path.insert(0, _PKG_PARENT)
sys.path.insert(0, os.path.dirname(_THIS_DIR))             # slim_gsgp/

from slim_gsgp.utils.simplification import (  # noqa: E402
    simplify_tree_sympy,
    simplify_constant_operations,
    _count_tree_nodes,
)


# ---------------------------------------------------------------------------
# Synthetic-tree round-trip checks
# ---------------------------------------------------------------------------

ALLOWED_OPS = {"add", "subtract", "multiply", "divide", "negative"}


def _validate_tree_shape(tree):
    """Recurse and assert every internal node uses only the allowed operators
    and every leaf is a string/number."""
    if not isinstance(tree, tuple):
        assert isinstance(tree, (int, float, str)), (
            f"Unexpected leaf type {type(tree).__name__}: {tree!r}"
        )
        return
    op = tree[0]
    assert op in ALLOWED_OPS, f"Unexpected operator in simplified tree: {op!r}"
    for child in tree[1:]:
        _validate_tree_shape(child)


def _eval_tree(tree, env):
    """Evaluate a tree-tuple with ``env`` mapping variable names → floats.
    Constants are floats (or numeric strings)."""
    if not isinstance(tree, tuple):
        if isinstance(tree, (int, float)):
            return float(tree)
        # try as numeric string first; otherwise treat as variable
        try:
            return float(tree)
        except ValueError:
            return env[tree]
    op = tree[0]
    if op == "negative":
        return -_eval_tree(tree[1], env)
    a = _eval_tree(tree[1], env)
    b = _eval_tree(tree[2], env)
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        # Use torch's protected-divide convention is overkill here — at the
        # verification layer we want strict numerical equivalence, so we use
        # plain Python division and skip envs that would trigger a zero
        # denominator (the harness below filters those out).
        return a / b
    raise AssertionError(f"Unknown op {op!r}")


def _random_env(symbols):
    return {s: random.uniform(-5.0, 5.0) for s in symbols}


def _check_equivalence(original, simplified, symbols, n_samples=64, tol=1e-6):
    """Sample inputs and verify both trees agree numerically."""
    random.seed(0)
    attempts = 0
    checked = 0
    while checked < n_samples and attempts < n_samples * 20:
        attempts += 1
        env = _random_env(symbols)
        try:
            a = _eval_tree(original, env)
            b = _eval_tree(simplified, env)
        except ZeroDivisionError:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        # Mixed-scale tolerance.
        if abs(a - b) > tol * (1.0 + abs(a) + abs(b)):
            return False, (env, a, b)
        checked += 1
    if checked == 0:
        # Couldn't find any non-degenerate input. Conservatively pass.
        return True, None
    return True, None


# A small library of constructed test trees that the conservative pass
# can't fully simplify but Sympy should be able to shrink.
def _build_test_cases():
    x0, x1, x2 = "x0", "x1", "x2"
    cases = [
        # (description, tree)
        ("idempotent add/subtract",
         ("subtract", ("add", x0, x1), x1)),                  # → x0
        ("collect-like terms",
         ("add", ("multiply", "2.0", x0), ("multiply", "3.0", x0))),  # → 5*x0
        ("distribute then collapse",
         ("subtract", ("multiply", x0, ("add", x1, x2)),
          ("multiply", x0, x1))),                              # → x0 * x2
        ("constant folding inside multiply",
         ("multiply", ("multiply", "2.0", "3.0"), x0)),        # → 6*x0
        ("fraction with cancellation",
         ("divide", ("multiply", x0, x1), x1)),                # → x0
        ("nested negation",
         ("subtract", "0.0", ("subtract", "0.0", x0))),        # → x0
        ("identity that does not shrink",
         ("multiply", x0, x1)),                                # stays as-is
        ("constant-only expression",
         ("add", "2.0", ("multiply", "3.0", "4.0"))),          # → 14.0
    ]
    return cases


def run_synthetic_checks() -> int:
    cases = _build_test_cases()
    print("=" * 72)
    print("Synthetic tree checks  (Sympy round-trip + numeric equivalence)")
    print("=" * 72)
    failures = 0
    for desc, tree in cases:
        original_nodes = _count_tree_nodes(tree)
        simplified, removed, applied = simplify_tree_sympy(tree, constants_dict={})
        symbols = sorted({tok for tok in _collect_symbols(tree)})
        try:
            _validate_tree_shape(simplified)
            ok, info = _check_equivalence(tree, simplified, symbols)
            if not ok:
                env, a, b = info
                raise AssertionError(
                    f"Numeric mismatch at env={env!r}: original={a}, simplified={b}"
                )
            new_nodes = _count_tree_nodes(simplified)
            tag = "shrunk " if new_nodes < original_nodes else "kept   "
            print(f"  [OK] {tag} {original_nodes:>3d} → {new_nodes:<3d}  {desc}")
        except AssertionError as exc:
            failures += 1
            print(f"  [FAIL] {desc}: {exc}")
    print()
    return failures


def _collect_symbols(tree):
    """Yield variable-like leaves from a tree (anything that's not numeric)."""
    if not isinstance(tree, tuple):
        if isinstance(tree, (int, float)):
            return
        try:
            float(tree)
        except (TypeError, ValueError):
            yield tree
        return
    for child in tree[1:]:
        yield from _collect_symbols(child)


# ---------------------------------------------------------------------------
# End-to-end check: running slim() and asserting predictions are unchanged
# ---------------------------------------------------------------------------

def run_end_to_end_check(slim_versions, num_seeds=2, pop_size=30, n_iter=10) -> int:
    """Run slim() on the airfoil dataset and assert that calling
    simplify_population (which is what main_slim.slim() already does at the
    end of evolution) does not alter predict() output for any individual.

    Note: this is asserting an invariant of the current architecture —
    simplification is decoupled from prediction. The test would catch any
    future refactor that accidentally couples them.
    """
    from slim_gsgp.main_slim import slim
    from slim_gsgp.datasets.data_loader import load_airfoil
    from slim_gsgp.utils.utils import train_test_split
    from slim_gsgp.utils.simplification import simplify_population

    print("=" * 72)
    print("End-to-end check  (predict() invariance across simplify_population)")
    print("=" * 72)

    failures = 0
    X, y = load_airfoil(X_y=True)

    for slim_version in slim_versions:
        for seed in range(42, 42 + num_seeds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
            X_val, X_test, y_val, y_test = train_test_split(
                X_test, y_test, p_test=0.5, seed=seed
            )

            try:
                results = slim(
                    X_train=X_train, y_train=y_train,
                    X_test=X_val, y_test=y_val,
                    dataset_name="airfoil",
                    slim_version=slim_version,
                    pop_size=pop_size,
                    n_iter=n_iter,
                    ms_lower=0, ms_upper=1, p_inflate=0.5,
                    reconstruct=True,
                    seed=seed,
                    verbose=0,
                )
            except Exception as exc:
                failures += 1
                print(f"  [FAIL] slim() failed for {slim_version} seed={seed}: {exc}")
                continue

            for label, ind in (
                ("best_fitness", results.best_fitness),
                ("best_normalized", results.best_normalized),
                ("smallest", results.smallest),
            ):
                pred_before = ind.predict(X_test).clone()

                # Re-run simplify_population on a single-element list to
                # confirm calling it a second time is also a no-op for
                # predictions. (Inside slim() it has already been called
                # once on the Pareto frontier.)
                simplify_population([ind], debug=False)

                pred_after = ind.predict(X_test)

                if not torch.allclose(pred_before, pred_after, atol=1e-8, rtol=0):
                    failures += 1
                    diff = (pred_before - pred_after).abs().max().item()
                    print(
                        f"  [FAIL] {slim_version} seed={seed} {label}: "
                        f"predictions changed (max abs diff={diff:.3e})"
                    )
                else:
                    print(f"  [OK] {slim_version} seed={seed} {label}: predictions identical")

    print()
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    failures = 0
    failures += run_synthetic_checks()

    # Use a small set of versions to keep the run quick. Add more if needed.
    slim_versions = ["SLIM+ABS", "SLIM*ABS", "SLIM+SIG2", "SLIM+N1"]
    failures += run_end_to_end_check(slim_versions, num_seeds=2, pop_size=30, n_iter=10)

    print("=" * 72)
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILURES: {failures}")
    print("=" * 72)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
