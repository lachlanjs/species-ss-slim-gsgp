# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
run.py — Main entry point for SLIM-GSGP experiments.

Usage:
    cd slim_gsgp
    python run.py

Only edit the CONFIGURATION block below to control:
  - Which SLIM version is run (selects the inflate-mutation formula, including N1)
  - Which orthogonal variant flags are enabled (OMS, LS, Pareto, Simplification)
  - Which datasets are included or skipped
  - How many times each dataset is run

Note: NM (Normalized Mutation) is NOT a variant flag. It is intrinsic to the
SLIM+N1 / SLIM*N1 versions — choose one of those as SLIM_VERSION to use NM.
"""

# ============================================================================
#
#                         C O N F I G U R A T I O N
#
# ============================================================================

SLIM_VERSION = 'SLIM+N1'
# Available options:
#   'SLIM+ABS'   — Inflate with absolute value and sum operator  (recommended)
#   'SLIM+SIG2'  — Inflate with sigmoid (version 2) and sum operator
#   'SLIM+SIG1'  — Inflate with sigmoid (version 1) and sum operator
#   'SLIM*ABS'   — Inflate with absolute value and product operator
#   'SLIM*SIG2'  — Inflate with sigmoid (version 2) and product operator
#   'SLIM*SIG1'  — Inflate with sigmoid (version 1) and product operator
#   'SLIM+N1'    — Normalized mutation (standardization) with sum operator
#   'SLIM*N1'    — Normalized mutation (standardization) with product operator

USE_OMS = True
USE_LINEAR_SCALING = True
USE_PARETO_TOURNAMENT = True
USE_SIMPLIFICATION = True

NUM_RUNS = 30
BASE_SEED = 42

DATASETS = {
    'airfoil':                  True,   # Dataset  1
    'bike_sharing':             True,   # Dataset  2
    'bioavailability':          True,   # Dataset  3
    'boston':                   True,   # Dataset  4
    'breast_cancer':            True,   # Dataset  5
    'concrete_slump':           True,   # Dataset  6
    'concrete_strength':        True,   # Dataset  7
    'diabetes':                 True,   # Dataset  8
    'efficiency_cooling':       True,   # Dataset  9
    'efficiency_heating':       True,   # Dataset 10
    'forest_fires':             True,   # Dataset 11
    'istanbul':                 False,  # Dataset 12 — disabled by default 
    'parkinson_updrs':          True,   # Dataset 13
    'ppb':                      True,   # Dataset 14
    'resid_build_sale_price':   True,   # Dataset 15
}

# ============================================================================
#
#              END OF CONFIGURATION — Do not edit below this line
#
# ============================================================================

import os
import sys
import traceback
from datetime import datetime
import pandas as pd

# Ensure relative imports work regardless of the current working directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
# Ensure the project root takes priority over the installed package in site-packages,
# so that 'from slim_gsgp.xxx import ...' resolves to the local source code.
sys.path.insert(0, os.path.dirname(_this_dir))

from run_single_dataset_multiple_runs import (
    run_single_dataset_multiple_times,
    DATASET_LOADERS,
)
from utils.naming_utils import build_execution_type, build_variant_name


def _validate_config():
    """Validates configuration before starting and applies compatibility rules."""
    oms = USE_OMS

    compatible_oms_versions = ("SLIM+ABS", "SLIM+SIG2", "SLIM+SIG1", "SLIM+N1")
    if oms and SLIM_VERSION not in compatible_oms_versions:
        print(
            f"  WARNING: OMS only works with '+' versions. "
            f"Disabling OMS for '{SLIM_VERSION}'."
        )
        oms = False

    enabled_datasets = [name for name, active in DATASETS.items() if active]
    unknown = [d for d in enabled_datasets if d not in DATASET_LOADERS]
    if unknown:
        raise ValueError(
            f"Unknown datasets: {unknown}. "
            f"Available: {list(DATASET_LOADERS.keys())}"
        )

    if not enabled_datasets:
        raise ValueError(
            "No active dataset. Set at least one to True in DATASETS."
        )

    if NUM_RUNS < 1:
        raise ValueError("NUM_RUNS must be >= 1.")

    return oms, enabled_datasets


def main():
    print("=" * 80)
    print("  SLIM-GSGP — Experiment")
    print("=" * 80)

    # --- Validation and conflict resolution ---
    try:
        use_oms, enabled_datasets = _validate_config()
    except ValueError as e:
        print(f"\n[Configuration ERROR] {e}")
        sys.exit(1)

    # --- Configuration summary ---
    variant_name = build_variant_name(
        slim_version=SLIM_VERSION,
        use_oms=use_oms,
        use_linear_scaling=USE_LINEAR_SCALING,
        use_pareto_tournament=USE_PARETO_TOURNAMENT,
        use_simplification=USE_SIMPLIFICATION,
    )
    execution_type = build_execution_type(
        use_linear_scaling=USE_LINEAR_SCALING,
        use_oms=use_oms,
        use_pareto_tournament=USE_PARETO_TOURNAMENT,
        use_simplification=USE_SIMPLIFICATION,
    )

    print(f"\n  Variant            : {variant_name}")
    print(f"  SLIM version       : {SLIM_VERSION}")
    print(f"  OMS                : {'✓' if use_oms else '✗'}")
    print(f"  Linear Scaling     : {'✓' if USE_LINEAR_SCALING else '✗'}")
    print(f"  Pareto Tournament  : {'✓' if USE_PARETO_TOURNAMENT else '✗'}")
    print(f"  Simplification     : {'✓' if USE_SIMPLIFICATION else '✗'}")
    print(f"\n  Runs/dataset       : {NUM_RUNS}")
    print(f"  Base seed          : {BASE_SEED}")
    print(f"\n  Datasets ({len(enabled_datasets):2d}): {', '.join(enabled_datasets)}")
    print("=" * 80)

    start_time = datetime.now()
    successful = []
    failed = []
    all_raw_rows = []

    # --- Main loop ---
    for i, dataset_name in enumerate(enabled_datasets, start=1):
        print(f"\n[{i}/{len(enabled_datasets)}] Starting '{dataset_name}'...")
        try:
            raw_rows = run_single_dataset_multiple_times(
                dataset_name=dataset_name,
                num_runs=NUM_RUNS,
                slim_version=SLIM_VERSION,
                use_oms=use_oms,
                use_linear_scaling=USE_LINEAR_SCALING,
                use_pareto_tournament=USE_PARETO_TOURNAMENT,
                use_simplification=USE_SIMPLIFICATION,
                base_seed=BASE_SEED,
            )
            if raw_rows:
                all_raw_rows.extend(raw_rows)
            successful.append(dataset_name)
        except Exception as e:
            print(f"\n  [ERROR] Dataset '{dataset_name}' failed: {e}")
            print(traceback.format_exc())
            failed.append(dataset_name)

    # --- CSV consolidado en formato reproduce_results ---
    if all_raw_rows:
        os.makedirs("log", exist_ok=True)
        df_raw = pd.DataFrame(all_raw_rows)
        df_raw = df_raw.set_index(['dataset', 'individual', 'run']).sort_index()
        safe_name = variant_name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
        raw_csv_path = os.path.join("log", f"{safe_name}.csv")
        df_raw.to_csv(raw_csv_path)
        print(f"\n  CSV consolidado guardado: {raw_csv_path}")

    # --- Consolidated Excel generation (optional) ---
    if successful:
        try:
            import generate_consolidated_excel  # noqa: F401  (optional external module)
            excel_filename = os.path.join(
                "log", f"results_{execution_type.replace(' ', '_')}.xlsx"
            )
            generate_consolidated_excel.OUTPUT_FILE = excel_filename
            generate_consolidated_excel.create_consolidated_excel(
                execution_type, variant_name
            )
            print(f"\n  Consolidated Excel generated: {excel_filename}")
        except ImportError:
            pass  # Module not available; individual CSVs are already saved
        except Exception as e:
            print(f"\n  [WARNING] Could not generate consolidated Excel: {e}")

    # --- Final summary ---
    end_time = datetime.now()
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"  Datasets attempted  : {len(enabled_datasets)}")
    print(f"  Completed           : {len(successful)}")
    if successful:
        print(f"    → {', '.join(successful)}")
    print(f"  Failed              : {len(failed)}")
    if failed:
        print(f"    → {', '.join(failed)}")
    print(f"  Total time          : {end_time - start_time}")
    print("=" * 80)


if __name__ == "__main__":
    main()
