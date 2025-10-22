"""
Script to run all datasets sequentially with multiple runs for statistical analysis.
Executes run_single_dataset_multiple_runs.py for each dataset.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these variables
# ============================================================================
NUM_RUNS = 30
SLIM_VERSION = "SLIM+ABS"          # Options: SLIM+ABS, SLIM+SIG2, SLIM*ABS, SLIM*SIG2
BASE_SEED = 42                     # Base seed (not used anymore, each run is random)

# Feature flags - Set to True to enable
USE_OMS = False                     # Enable OMS (only works with SLIM+ABS or SLIM+SIG2)
USE_LINEAR_SCALING = True         # Enable Linear Scaling
USE_PARETO_TOURNAMENT = True       # Enable Pareto Tournament
USE_SIMPLIFICATION = True         # Enable simplification when selecting best_normalized
# ============================================================================

# List of all datasets
DATASETS = [
    "airfoil",
    "bike_sharing", 
    "bioavailability",
    "boston",
    "breast_cancer",
    "concrete_slump",
    "concrete_strength",
    "diabetes",
    "efficiency_cooling",
    "efficiency_heating",
    "forest_fires",
    "istanbul",
    "parkinson_updrs",
    "ppb",
    "resid_build_sale_price"
]

def run_dataset(dataset_name, num_runs, slim_version, base_seed, use_oms=False, 
                use_linear_scaling=False, use_pareto_tournament=False, use_simplification=True):
    """
    Execute run_single_dataset_multiple_runs.py for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to run
        num_runs: Number of runs to execute
        slim_version: SLIM version to use
        base_seed: Base seed for random number generation
        use_oms: Enable OMS
        use_linear_scaling: Enable Linear Scaling
        use_pareto_tournament: Enable Pareto Tournament
        use_simplification: Enable simplification when selecting best_normalized
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "run_single_dataset_multiple_runs.py",
        "--dataset", dataset_name,
        "--num_runs", str(num_runs),
        "--slim_version", slim_version,
        "--base_seed", str(base_seed)
    ]
    
    # Add feature flags
    if use_oms:
        cmd.append("--oms")
    if use_linear_scaling:
        cmd.append("--linear_scaling")
    if use_pareto_tournament:
        cmd.append("--pareto_tournament")
    if not use_simplification:
        cmd.append("--no_simplification")
    
    try:
        print(f"\nExecuting: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    """Main execution function."""
    total_datasets = len(DATASETS)
    success_count = 0
    fail_count = 0
    failed_datasets = []
    
    start_time = datetime.now()
    
    print("=" * 80)
    print("SEQUENTIAL EXECUTION OF ALL DATASETS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of runs per dataset: {NUM_RUNS}")
    print(f"  - SLIM version: {SLIM_VERSION}")
    print(f"  - Base seed: {BASE_SEED} (not used - random seeds per run)")
    print(f"  - OMS: {'✓ Enabled' if USE_OMS else '✗ Disabled'}")
    print(f"  - Linear Scaling: {'✓ Enabled' if USE_LINEAR_SCALING else '✗ Disabled'}")
    print(f"  - Pareto Tournament: {'✓ Enabled' if USE_PARETO_TOURNAMENT else '✗ Disabled'}")
    print(f"  - Simplification: {'✓ Enabled' if USE_SIMPLIFICATION else '✗ Disabled'}")
    print(f"\nTotal datasets to process: {total_datasets}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print("=" * 80)
    
    for idx, dataset in enumerate(DATASETS, 1):
        dataset_start_time = datetime.now()
        
        print("\n")
        print("=" * 80)
        print(f"[{idx}/{total_datasets}] Processing dataset: {dataset}")
        print("=" * 80)
        
        success = run_dataset(dataset, NUM_RUNS, SLIM_VERSION, BASE_SEED, 
                            USE_OMS, USE_LINEAR_SCALING, USE_PARETO_TOURNAMENT, USE_SIMPLIFICATION)
        
        dataset_end_time = datetime.now()
        duration = dataset_end_time - dataset_start_time
        
        if success:
            success_count += 1
            print(f"\n✓ Dataset '{dataset}' completed successfully")
            print(f"  Duration: {duration}")
        else:
            fail_count += 1
            failed_datasets.append(dataset)
            print(f"\n✗ Dataset '{dataset}' failed")
            print(f"  Duration: {duration}")
        
        print("=" * 80)
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Final summary
    print("\n\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {total_datasets}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if failed_datasets:
        print(f"\nFailed datasets: {', '.join(failed_datasets)}")
    
    success_rate = (success_count / total_datasets) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    print(f"Total execution time: {total_duration}")
    
    avg_seconds = total_duration.total_seconds() / total_datasets
    avg_time_str = str(datetime.utcfromtimestamp(avg_seconds).strftime('%H:%M:%S'))
    print(f"Average time per dataset: {avg_time_str}")
    
    print(f"\nAll results saved in the 'log/' directory")
    print("=" * 80)
    
    # Generate consolidated Excel file
    if success_count > 0:
        print("\n")
        print("=" * 80)
        print("GENERATING CONSOLIDATED EXCEL FILE")
        print("=" * 80)
        
        # Build execution type name
        execution_type = "slim"
        if USE_LINEAR_SCALING:
            execution_type += "_linear_scaling"
        if USE_OMS:
            execution_type += "_oms"
        if USE_PARETO_TOURNAMENT:
            execution_type += "_pareto"
        if not USE_SIMPLIFICATION:
            execution_type += "_no_simplif"
        
        # Build variant name
        variant_parts = [SLIM_VERSION]
        if USE_OMS:
            variant_parts.append("OMS")
        if USE_LINEAR_SCALING:
            variant_parts.append("LS")
        if USE_PARETO_TOURNAMENT:
            variant_parts.append("Pareto")
        variant_name = " + ".join(variant_parts)
        
        excel_filename = f"consolidated_results_{execution_type}.xlsx"
        
        try:
            import generate_consolidated_excel
            generate_consolidated_excel.OUTPUT_FILE = excel_filename
            generate_consolidated_excel.create_consolidated_excel(execution_type, variant_name)
            print(f"✓ Consolidated Excel file created: {excel_filename}")
        except Exception as e:
            print(f"⚠ Error generating consolidated Excel: {e}")
            print("  You can generate it manually with:")
            print(f"  python generate_consolidated_excel.py --execution_type {execution_type} --variant_name \"{variant_name}\" --output {excel_filename}")
    
    # Return exit code based on success
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
