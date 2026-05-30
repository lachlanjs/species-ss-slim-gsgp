"""check_simplification_invariance.py

Empirical check: does the use_simplification flag change fitness numbers?

We run slim() twice with identical seed/config, only flipping
use_simplification, and compare the three returned individuals.

Expectation derived from the source (main_slim.py: use_simplification is
only read AFTER optimizer.solve() returns):

  - best_fitness   : same fitness, same predictions  (selected during evolution by pure fitness)
  - best_normalized: MAY differ                       (Pareto-frontier pick depends on simplified size)
  - smallest       : MAY differ                       (min over population uses simplified count for frontier members)
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
from slim_gsgp.utils.utils import train_test_split


def run(use_simp, slim_version, seed):
    X, y = load_airfoil(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)
    return slim(
        X_train=X_train, y_train=y_train,
        X_test=X_val, y_test=y_val,
        dataset_name="airfoil",
        slim_version=slim_version,
        pop_size=50, n_iter=15,
        ms_lower=0, ms_upper=1, p_inflate=0.5,
        reconstruct=True,
        tournament_type="pareto", tournament_size=5,
        multi_obj_attrs=["fitness", "size"],
        use_simplification=use_simp,
        seed=seed, verbose=0,
    ), X_test, y_test


def cmp(label, a, b, X_test):
    fit_a = float(a.fitness)
    fit_b = float(b.fitness)
    pred_a = a.predict(X_test)
    pred_b = b.predict(X_test)
    same_pred = torch.allclose(pred_a, pred_b, atol=1e-10, rtol=0)
    fit_eq = abs(fit_a - fit_b) < 1e-12
    print(f"  {label:<18} fitness  WITH={fit_a:.8f}  WITHOUT={fit_b:.8f}  equal={fit_eq}")
    print(f"  {' ' * 18} nodes_count WITH={a.nodes_count}  WITHOUT={b.nodes_count}")
    print(f"  {' ' * 18} predictions identical? {same_pred}")


def main():
    for slim_version in ["SLIM+ABS", "SLIM+N1", "SLIM+SIG2"]:
        for seed in (42, 43):
            print("=" * 72)
            print(f"  {slim_version}  seed={seed}")
            print("=" * 72)
            (r_with, Xte, _) = run(True, slim_version, seed)
            (r_without, _, _) = run(False, slim_version, seed)
            cmp("best_fitness", r_with.best_fitness, r_without.best_fitness, Xte)
            cmp("best_normalized", r_with.best_normalized, r_without.best_normalized, Xte)
            cmp("smallest", r_with.smallest, r_without.smallest, Xte)
            print()


if __name__ == "__main__":
    main()
