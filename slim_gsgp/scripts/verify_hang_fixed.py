"""verify_hang_fixed.py

Confirm that the previously-hanging run (bioavailability, seed=57, SLIM+N1,
pop=100, n_iter=100) now completes, and time the post-evolution
simplification phase end-to-end.
"""
from __future__ import annotations

import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, os.path.dirname(_THIS))

from slim_gsgp.main_slim import slim
from slim_gsgp.datasets.data_loader import load_bioav
from slim_gsgp.utils.utils import train_test_split


def main():
    seed = 57
    X, y = load_bioav(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)

    print(f"Running the previously-hanging config: bioavailability seed={seed}, SLIM+N1...")
    t0 = time.time()
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
    dt = time.time() - t0
    print(f"\nCOMPLETED in {dt:.1f}s (no hang).")
    print(f"  best_fitness    : fitness={float(results.best_fitness.fitness):.5f}  nodes={results.best_fitness.nodes_count}")
    print(f"  best_normalized : fitness={float(results.best_normalized.fitness):.5f}  nodes={results.best_normalized.nodes_count}")
    print(f"  smallest        : fitness={float(results.smallest.fitness):.5f}  nodes={results.smallest.nodes_count}")


if __name__ == "__main__":
    main()
