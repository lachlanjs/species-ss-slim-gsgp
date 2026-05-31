"""Tests for the Sympy-based simplification pass in
``slim_gsgp.utils.simplification``.

Two layers of coverage:

* **Unit-level** — synthetic trees with a known simplified shape. Verifies
  that the result uses only the four base operators and evaluates to the
  same numeric output as the input across many random samples.

* **Integration** — running ``slim()`` end-to-end and asserting that
  predictions are unchanged after ``simplify_population``. This is the
  invariant the user cares about: simplification must not "break" any
  individual's behaviour.
"""
import math
import random

import pytest
import torch

from slim_gsgp.utils.simplification import (
    _count_tree_nodes,
    simplify_tree_sympy,
    simplify_population,
)


ALLOWED_OPS = {"add", "subtract", "multiply", "divide", "negative"}


def _validate_shape(tree):
    if not isinstance(tree, tuple):
        assert isinstance(tree, (int, float, str))
        return
    assert tree[0] in ALLOWED_OPS, f"unexpected op {tree[0]!r}"
    for child in tree[1:]:
        _validate_shape(child)


def _eval_tree(tree, env):
    if not isinstance(tree, tuple):
        if isinstance(tree, (int, float)):
            return float(tree)
        try:
            return float(tree)
        except ValueError:
            return env[tree]
    op = tree[0]
    if op == "negative":
        return -_eval_tree(tree[1], env)
    a = _eval_tree(tree[1], env)
    b = _eval_tree(tree[2], env)
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]


def _symbols_in(tree):
    if not isinstance(tree, tuple):
        if isinstance(tree, (int, float)):
            return set()
        try:
            float(tree)
            return set()
        except (TypeError, ValueError):
            return {tree}
    out = set()
    for child in tree[1:]:
        out |= _symbols_in(child)
    return out


SIMPLIFIABLE_CASES = [
    # (description, tree, expected_strict_shrink)
    ("idempotent add/subtract",
     ("subtract", ("add", "x0", "x1"), "x1"), True),
    ("collect-like terms",
     ("add", ("multiply", "2.0", "x0"), ("multiply", "3.0", "x0")), True),
    ("distribute then collapse",
     ("subtract",
      ("multiply", "x0", ("add", "x1", "x2")),
      ("multiply", "x0", "x1")), True),
    ("constant folding",
     ("multiply", ("multiply", "2.0", "3.0"), "x0"), True),
    ("fraction with cancellation",
     ("divide", ("multiply", "x0", "x1"), "x1"), True),
    ("nested double negation",
     ("subtract", "0.0", ("subtract", "0.0", "x0")), True),
    ("already minimal",
     ("multiply", "x0", "x1"), False),
    ("constant-only",
     ("add", "2.0", ("multiply", "3.0", "4.0")), True),
]


@pytest.mark.parametrize("desc,tree,must_shrink", SIMPLIFIABLE_CASES, ids=[c[0] for c in SIMPLIFIABLE_CASES])
def test_sympy_round_trip_shape_is_valid(desc, tree, must_shrink):
    simplified, _, _ = simplify_tree_sympy(tree, constants_dict={})
    _validate_shape(simplified)


@pytest.mark.parametrize("desc,tree,must_shrink", SIMPLIFIABLE_CASES, ids=[c[0] for c in SIMPLIFIABLE_CASES])
def test_sympy_preserves_numeric_value(desc, tree, must_shrink):
    simplified, _, _ = simplify_tree_sympy(tree, constants_dict={})
    symbols = sorted(_symbols_in(tree))
    random.seed(0)
    samples = 64
    checked = 0
    for _ in range(samples * 20):
        env = {s: random.uniform(-5.0, 5.0) for s in symbols}
        try:
            a = _eval_tree(tree, env)
            b = _eval_tree(simplified, env)
        except ZeroDivisionError:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        assert abs(a - b) <= 1e-6 * (1.0 + abs(a) + abs(b)), (
            f"mismatch on env={env!r}: original={a}, simplified={b}"
        )
        checked += 1
        if checked >= samples:
            break


@pytest.mark.parametrize("desc,tree,must_shrink", SIMPLIFIABLE_CASES, ids=[c[0] for c in SIMPLIFIABLE_CASES])
def test_sympy_shrinks_when_possible(desc, tree, must_shrink):
    simplified, removed, applied = simplify_tree_sympy(tree, constants_dict={})
    if must_shrink:
        assert applied is True, "expected sympy to reduce node count"
        assert _count_tree_nodes(simplified) < _count_tree_nodes(tree)
        assert removed == _count_tree_nodes(tree) - _count_tree_nodes(simplified)
    else:
        # If sympy can't shrink, the input must come back unchanged and
        # ``removed`` must be 0 — we never grow the tree.
        assert _count_tree_nodes(simplified) <= _count_tree_nodes(tree)
        assert removed == 0


def test_sympy_handles_unknown_constants_dict_gracefully():
    # callable-style constants matching the SLIM CONSTANTS convention
    constants_dict = {
        "constant_2": lambda _: torch.tensor(2.0),
        "constant__3": lambda _: torch.tensor(-3.0),
    }
    tree = ("add", "constant_2", ("multiply", "constant__3", "x0"))
    simplified, _, _ = simplify_tree_sympy(tree, constants_dict=constants_dict)
    _validate_shape(simplified)


def test_sympy_skips_oversized_trees():
    # Build a deeply nested expression and confirm we bail out cleanly.
    tree = "x0"
    for _ in range(300):
        tree = ("add", tree, "x0")
    out, removed, applied = simplify_tree_sympy(tree, constants_dict={}, max_input_nodes=200)
    assert applied is False
    assert removed == 0
    assert out is tree  # untouched


def _build_pathological_tree(depth):
    """Nested divisions/products of distinct variables — the shape that made
    sympy.simplify/factor/cancel run for minutes (effectively hanging) before
    the timeout safeguard was added."""
    tree = "x0"
    for i in range(depth):
        var = f"x{i % 8}"
        # Alternate divide / multiply / subtract to defeat trivial folding.
        if i % 3 == 0:
            tree = ("divide", tree, ("add", var, "1.0"))
        elif i % 3 == 1:
            tree = ("multiply", tree, ("subtract", var, "0.5"))
        else:
            tree = ("subtract", tree, ("divide", "1.0", ("add", var, "2.0")))
    return tree


def test_sympy_pass_is_time_bounded_on_pathological_input():
    """Regression test for the bioavailability seed=57 hang.

    Before the fix, sympy.simplify/factor/cancel had no timeout and could run
    for many minutes on medium-sized expressions with nested divisions. This
    asserts the pass returns quickly regardless, and never grows the tree.
    """
    import time

    tree = _build_pathological_tree(depth=40)  # ~ a few hundred nodes worth of structure
    n_in = _count_tree_nodes(tree)

    start = time.time()
    out, removed, applied = simplify_tree_sympy(
        tree, constants_dict={}, cheap_timeout=1.0, full_timeout=2.0,
        max_divisions_for_full=7,
    )
    elapsed = time.time() - start

    # Generous bound: division-heavy trees skip the full pass entirely, and any
    # op that does run is capped, so this returns far below the bound.
    assert elapsed < 20.0, f"simplify_tree_sympy took {elapsed:.1f}s — timeout not effective"
    # Must never inflate the tree.
    assert _count_tree_nodes(out) <= n_in
    assert removed >= 0


@pytest.mark.parametrize("n_nodes", [70, 90, 110])
def test_sympy_bounded_for_each_size(n_nodes):
    """Each medium/large input must return quickly (the sizes that timed out
    in the repro: 71, 87, 99 nodes)."""
    import time

    # Grow a nested-fraction tree until it reaches ~n_nodes.
    tree = "x0"
    i = 0
    while _count_tree_nodes(tree) < n_nodes:
        var = f"x{i % 6}"
        tree = ("divide", ("subtract", tree, var), ("add", var, "1.0"))
        i += 1

    start = time.time()
    out, removed, applied = simplify_tree_sympy(
        tree, constants_dict={}, cheap_timeout=1.0, full_timeout=2.0,
        max_divisions_for_full=7,
    )
    elapsed = time.time() - start
    assert elapsed < 15.0, f"{n_nodes}-node input took {elapsed:.1f}s"
    assert _count_tree_nodes(out) <= _count_tree_nodes(tree)


# -----------------------------------------------------------------------------
# Integration: predict() invariance across simplify_population
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "slim_version",
    ["SLIM+ABS", "SLIM*ABS", "SLIM+SIG2", "SLIM+N1", "SLIM+N2", "SLIM*N2"],
)
def test_simplify_population_does_not_change_predictions(slim_version):
    from slim_gsgp.main_slim import slim
    from slim_gsgp.datasets.data_loader import load_airfoil
    from slim_gsgp.utils.utils import train_test_split

    X, y = load_airfoil(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=42)

    results = slim(
        X_train=X_train, y_train=y_train,
        X_test=X_val, y_test=y_val,
        dataset_name="airfoil",
        slim_version=slim_version,
        pop_size=30, n_iter=8,
        ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True,
        seed=42, verbose=0,
    )

    for ind in (results.best_fitness, results.best_normalized, results.smallest):
        before = ind.predict(X_test).clone()
        simplify_population([ind], debug=False)
        after = ind.predict(X_test)
        assert torch.allclose(before, after, atol=1e-8, rtol=0), (
            f"predictions changed after simplify_population for {slim_version}"
        )


@pytest.mark.parametrize("slim_version", ["SLIM+N1", "SLIM*N1", "SLIM+N2", "SLIM*N2"])
def test_normalized_mutation_train_test_consistency(slim_version):
    """N1/N2 must reuse the TRAINING normalization constants (mean/std or
    min/max) at inference time. We check that predict() on the validation set
    reproduces the test_fitness computed during evolution — which only holds
    if the stored constants, not freshly recomputed ones, are used."""
    from slim_gsgp.main_slim import slim
    from slim_gsgp.datasets.data_loader import load_airfoil
    from slim_gsgp.evaluators.fitness_functions import rmse
    from slim_gsgp.utils.utils import train_test_split

    X, y = load_airfoil(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=42)

    results = slim(
        X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
        dataset_name="airfoil", slim_version=slim_version,
        pop_size=30, n_iter=8, ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True, seed=42, verbose=0, test_elite=True,
    )

    elite = results.best_fitness
    stored = float(elite.test_fitness)
    recomputed = float(rmse(y_true=y_val, y_pred=elite.predict(X_val)))
    assert abs(stored - recomputed) <= 1e-4 * (1 + abs(stored)), (
        f"{slim_version}: predict() ({recomputed}) disagrees with stored "
        f"test_fitness ({stored}) — normalization constants not reused"
    )
