"""diagnose_simplification.py

Cuantifica por qué la simplificación con Sympy reduce tan pocos nodos en
individuos SLIM reales. Mide cuatro hipótesis:

  (A) El cap de 200 nodos en simplify_tree_sympy salta muchos individuos.
  (B) La conversión convert_slim_individual_to_normal_tree es lossy:
      ignora el ms y la fórmula del variator → Sympy ve una caricatura.
  (C) Los bloques de SLIM son funcionalmente independientes; cada inflate
      añade un random_tree fresco con variables nuevas, así que hay poca
      estructura algebraica que cancelar entre bloques.
  (D) Muchos bloques tienen una contribución semántica despreciable
      (ms ≈ 0 o tree.semantics ≈ 0). Esto es lo que de verdad explica el
      bloat, y la simplificación algebraica no puede tocarlo.

Salida: tabla de métricas por versión + recomendación.
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
from slim_gsgp.utils.simplification import (
    convert_slim_individual_to_normal_tree,
    simplify_tree_sympy,
    _count_tree_nodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_pop(results):
    """Try to recover the full final population from a SlimResults object.
    SlimResults exposes best_fitness / best_normalized / smallest plus a
    pass-through __getattr__ to best_fitness, so we sample those three
    plus rely on optimizer.population through one of them if available."""
    # The three explicit picks already span the interesting cases.
    return [
        ("best_fitness", results.best_fitness),
        ("best_normalized", results.best_normalized),
        ("smallest", results.smallest),
    ]


def _block_semantic_norms(individual):
    """Per-block RMS of the stored train_semantics. A block with norm ~ 0
    contributes essentially nothing to the individual's output."""
    norms = []
    if not hasattr(individual, "train_semantics"):
        return norms
    ts = individual.train_semantics  # stacked tensor: (n_blocks, n_samples)
    if ts is None or ts.numel() == 0:
        return norms
    for i in range(ts.shape[0]):
        v = ts[i]
        if v.numel() == 0:
            norms.append(0.0)
        else:
            norms.append(float(v.float().pow(2).mean().sqrt().item()))
    return norms


def _operator_kind(individual):
    """Detect sum vs mul for the SLIM version associated with the individual."""
    v = getattr(individual, "version", "SLIM+ABS")
    return "mul" if "*" in v else "sum"


# ---------------------------------------------------------------------------
# Main diagnosis per individual
# ---------------------------------------------------------------------------

def diagnose_individual(role, ind):
    out = {"role": role, "version": getattr(ind, "version", "?")}

    # --- Block-level counts (the "true" cost of the individual) ---
    out["n_blocks"]      = len(ind.collection) if hasattr(ind, "collection") else None
    out["raw_size"]      = ind.size if hasattr(ind, "size") else None
    out["raw_nodes_cnt"] = ind.nodes_count

    # --- Block semantic contribution diagnostics ---
    norms = _block_semantic_norms(ind)
    out["block_sem_norms"] = norms
    out["blocks_near_zero"] = sum(1 for n in norms if n < 1e-8)
    out["blocks_tiny"]      = sum(1 for n in norms if n < 1e-4)
    out["blocks_small"]     = sum(1 for n in norms if n < 1e-2)

    # --- Converted "normal tree" view (what Sympy sees) ---
    structure, dicts = convert_slim_individual_to_normal_tree(ind)
    if structure is None:
        out["converted_nodes"]  = None
        out["sympy_applied"]    = False
        out["sympy_removed"]    = 0
        out["sympy_no_cap_rem"] = 0
        out["hit_cap"]          = False
        return out

    converted_nodes = _count_tree_nodes(structure)
    out["converted_nodes"] = converted_nodes
    out["hit_cap"] = converted_nodes > 200

    # Sympy WITH the standard cap (what the production pipeline does)
    _, sympy_removed_capped, applied_capped = simplify_tree_sympy(
        structure, dicts["CONSTANTS"], max_input_nodes=200
    )
    out["sympy_applied"] = applied_capped
    out["sympy_removed"] = sympy_removed_capped

    # Sympy WITHOUT the cap (what could be achieved if we removed it)
    _, sympy_removed_nocap, _ = simplify_tree_sympy(
        structure, dicts["CONSTANTS"], max_input_nodes=10**6
    )
    out["sympy_no_cap_rem"] = sympy_removed_nocap

    return out


# ---------------------------------------------------------------------------
# Per-version report
# ---------------------------------------------------------------------------

def report_for_version(slim_version, pop_size=100, n_iter=100, seed=42):
    print(f"\n{'=' * 78}")
    print(f"  {slim_version}    pop={pop_size}  n_iter={n_iter}  seed={seed}")
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
        reconstruct=True,
        tournament_type="pareto", tournament_size=5,
        multi_obj_attrs=["fitness", "size"],
        seed=seed, verbose=0,
    )

    diagnostics = [diagnose_individual(role, ind) for role, ind in _all_pop(results)]

    print(f"  {'role':<18} {'blocks':>7} {'conv_nd':>8} {'cap?':>5} "
          f"{'sym_red':>8} {'sym_nocap':>10} "
          f"{'near0':>6} {'tiny':>5} {'small':>6}")
    print("  " + "-" * 76)
    for d in diagnostics:
        cap_flag = "Y" if d["hit_cap"] else "-"
        conv = d["converted_nodes"] if d["converted_nodes"] is not None else "-"
        print(f"  {d['role']:<18} {d['n_blocks']:>7} {conv:>8} {cap_flag:>5} "
              f"{d['sympy_removed']:>8d} {d['sympy_no_cap_rem']:>10d} "
              f"{d['blocks_near_zero']:>6d} {d['blocks_tiny']:>5d} {d['blocks_small']:>6d}")

    # Aggregate interpretation
    print("\n  Interpretation:")
    for d in diagnostics:
        if d["converted_nodes"] is None:
            continue
        cap_effect = d["sympy_no_cap_rem"] - d["sympy_removed"]
        ratio_blocks_quiet = (d["blocks_small"] / max(d["n_blocks"], 1)) * 100 if d["n_blocks"] else 0
        algebraic_ratio = (d["sympy_no_cap_rem"] / max(d["converted_nodes"], 1)) * 100
        msg = (
            f"    [{d['role']:<18}]  "
            f"algebraic reducibility (no cap): {algebraic_ratio:5.1f}% of converted view; "
        )
        if cap_effect > 0:
            msg += f"cap is hiding {cap_effect} more nodes; "
        msg += f"{ratio_blocks_quiet:.0f}% of blocks have RMS<1e-2 (candidates for SEMANTIC pruning)"
        print(msg)


def main():
    versions = ["SLIM+ABS", "SLIM+SIG2", "SLIM+N1"]
    for v in versions:
        report_for_version(v, pop_size=100, n_iter=100, seed=42)

    print("\n" + "=" * 78)
    print("  SUMMARY OF WHAT THIS MEANS")
    print("=" * 78)
    print("""
  * 'algebraic reducibility' is how much Sympy could cut from the converted
    tree view if we removed the size cap. Low numbers (<10%) confirm that
    SLIM individuals are mostly NOT algebraically redundant — each inflate
    appends an independent random tree with fresh variables.

  * '% of blocks with RMS<1e-2' is the real cost driver: blocks whose
    semantic contribution is tiny. These cannot be removed by an algebraic
    pass — they require SEMANTIC pruning (look at train_semantics, drop
    blocks whose contribution is below tolerance, then renumber).

  * If the cap (Y in 'cap?') fires often AND 'sym_nocap' is much larger
    than 'sym_red', removing the cap is a quick win.

  * If 'algebraic reducibility' is consistently <5-10% and '% small blocks'
    is high, the structural fix is semantic pruning, not better algebra.
""")


if __name__ == "__main__":
    main()
