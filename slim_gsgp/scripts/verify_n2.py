"""verify_n2.py

Verify the new N2 (min-max normalization) SLIM version:

  1. SLIM+N2 / SLIM*N2 run end-to-end without error.
  2. Train/test consistency: predict() on the validation set reproduces the
     test_fitness that was computed during evolution. This confirms that the
     N2 blocks reuse the TRAINING min/max constants at inference time (rather
     than recomputing min/max on the new data), exactly as the paper requires
     (m, delta, min, max are numeric constants fixed once TR is generated).
  3. Sanity: N2 differs from N1 (different normalization → different search).
"""
from __future__ import annotations

import os
import sys

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, os.path.dirname(_THIS))

from slim_gsgp.main_slim import slim
from slim_gsgp.datasets.data_loader import load_airfoil
from slim_gsgp.evaluators.fitness_functions import rmse
from slim_gsgp.utils.utils import train_test_split


def run(version, seed=42):
    X, y = load_airfoil(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)
    res = slim(
        X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
        dataset_name="airfoil", slim_version=version,
        pop_size=50, n_iter=15, ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True, seed=seed, verbose=0, test_elite=True,
    )
    return res, (X_val, y_val)


def main():
    failures = 0
    print("=" * 70)
    print("N2 verification")
    print("=" * 70)

    for version in ["SLIM+N2", "SLIM*N2"]:
        try:
            res, (X_val, y_val) = run(version)
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {version} raised: {exc}")
            import traceback
            traceback.print_exc()
            continue

        elite = res.best_fitness
        # test_fitness stored during evolution (uses reused-train-stats test semantics)
        stored = float(elite.test_fitness)
        # Independent recompute through predict() on the same validation set.
        preds = elite.predict(X_val)
        recomputed = float(rmse(y_true=y_val, y_pred=preds))

        ok = abs(stored - recomputed) <= 1e-4 * (1 + abs(stored))
        tag = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{tag}] {version}: stored test_fitness={stored:.6f}  "
              f"predict()-recomputed={recomputed:.6f}  nodes={elite.nodes_count}")

    # N1 vs N2 should produce different individuals on the same seed.
    res1, _ = run("SLIM+N1")
    res2, _ = run("SLIM+N2")
    f1 = float(res1.best_fitness.fitness)
    f2 = float(res2.best_fitness.fitness)
    print(f"\nN1 train fitness={f1:.6f}  vs  N2 train fitness={f2:.6f}  "
          f"({'different' if abs(f1 - f2) > 1e-9 else 'identical (suspicious!)'})")

    print("\n" + "=" * 70)
    print("ALL N2 CHECKS PASSED" if failures == 0 else f"FAILURES: {failures}")
    print("=" * 70)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
