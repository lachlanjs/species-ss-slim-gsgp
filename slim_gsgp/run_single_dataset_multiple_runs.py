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

from main_slim import slim
from datasets.data_loader import (
    load_airfoil, load_bike_sharing, load_bioav, load_boston, load_breast_cancer,
    load_concrete_slump, load_concrete_strength, load_diabetes, load_efficiency_cooling,
    load_efficiency_heating, load_forest_fires, load_istanbul, load_parkinson_updrs,
    load_ppb, load_resid_build_sale_price
)
from evaluators.fitness_functions import rmse
from utils.utils import train_test_split
import csv
import os
import numpy as np
from datetime import datetime
import traceback

# Dictionary mapping dataset names to their loader functions
DATASET_LOADERS = {
    'airfoil': load_airfoil,
    'bike_sharing': load_bike_sharing,
    'bioavailability': load_bioav,
    'boston': load_boston,
    'breast_cancer': load_breast_cancer,
    'concrete_slump': load_concrete_slump,
    'concrete_strength': load_concrete_strength,
    'diabetes': load_diabetes,
    'efficiency_cooling': load_efficiency_cooling,
    'efficiency_heating': load_efficiency_heating,
    'forest_fires': load_forest_fires,
    'istanbul': load_istanbul,
    'parkinson_updrs': load_parkinson_updrs,
    'ppb': load_ppb,
    'resid_build_sale_price': load_resid_build_sale_price
}

def calculate_statistics(values):
    """
    Calculate mean, std, median, and IQR for a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        dict with mean, std, median, q1, q3, iqr
    """
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Sample standard deviation
    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }

def save_individual_runs(runs_data, dataset_name, execution_type, output_dir="log"):
    """
    Save individual run results to CSV.
    
    Args:
        runs_data: List of dictionaries with run results
        dataset_name: Name of the dataset
        execution_type: Type of execution
        output_dir: Output directory
    """
    filename = os.path.join(output_dir, f"individual_runs_{dataset_name}_{execution_type.replace(' ', '_')}.csv")
    
    fieldnames = ['run', 'seed',
                 'bf_train_rmse', 'bf_val_rmse', 'bf_test_rmse', 'bf_nodes', 'bf_depth',
                 'bn_train_rmse', 'bn_val_rmse', 'bn_test_rmse', 'bn_nodes', 'bn_depth',
                 'sm_train_rmse', 'sm_val_rmse', 'sm_test_rmse', 'sm_nodes', 'sm_depth']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs_data)
    
    print(f"   Individual runs saved to: {filename}")

def save_statistics_summary(stats, dataset_name, execution_type, output_dir="log"):
    """
    Save statistical summary (Mean, STD, Median, IQR) to CSV.
    
    Args:
        stats: Dictionary with statistics for each metric
        dataset_name: Name of the dataset
        execution_type: Type of execution
        output_dir: Output directory
    """
    filename = os.path.join(output_dir, f"statistics_{dataset_name}_{execution_type.replace(' ', '_')}.csv")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['individual', 'metric', 'mean', 'std', 'median', 'iqr', 'q1', 'q3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Best Fitness statistics
        for metric in ['train_rmse', 'val_rmse', 'test_rmse', 'nodes', 'depth']:
            row = {'individual': 'best_fitness', 'metric': metric}
            row.update(stats[f'bf_{metric}'])
            writer.writerow(row)
        
        # Best Normalized statistics
        for metric in ['train_rmse', 'val_rmse', 'test_rmse', 'nodes', 'depth']:
            row = {'individual': 'best_normalized', 'metric': metric}
            row.update(stats[f'bn_{metric}'])
            writer.writerow(row)
        
        # Smallest statistics
        for metric in ['train_rmse', 'val_rmse', 'test_rmse', 'nodes', 'depth']:
            row = {'individual': 'smallest', 'metric': metric}
            row.update(stats[f'sm_{metric}'])
            writer.writerow(row)
    
    print(f"   Statistics summary saved to: {filename}")

def save_formatted_table(stats, dataset_name, execution_type, output_dir="log"):
    """
    Save formatted table with Smallest Model, Optimal Compromise, and Best Fitness.
    Creates three tables: Training, Test, and Size.
    
    Args:
        stats: Dictionary with statistics for each metric
        dataset_name: Name of the dataset
        execution_type: Type of execution
        output_dir: Output directory
    """
    filename = os.path.join(output_dir, f"summary_table_{dataset_name}_{execution_type.replace(' ', '_')}.csv")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Helper function to format values
        def format_median_iqr(stat):
            return f"{stat['median']:.5f} ({stat['iqr']:.5f})"
        
        def format_mean_std(stat):
            return f"{stat['mean']:.5f} ({stat['std']:.5f})"
        
        # === TRAINING TABLE ===
        writer.writerow(['TRAINING RESULTS'])
        writer.writerow(['Smallest Model', '', 'Optimal Compromise', '', 'Best Fitness', ''])
        writer.writerow(['Median (IQR)', 'Mean (STD)', 'Median (IQR)', 'Mean (STD)', 'Median (IQR)', 'Mean (STD)'])
        writer.writerow([format_median_iqr(stats['sm_train_rmse']),
                        format_mean_std(stats['sm_train_rmse']),
                        format_median_iqr(stats['bn_train_rmse']),
                        format_mean_std(stats['bn_train_rmse']),
                        format_median_iqr(stats['bf_train_rmse']),
                        format_mean_std(stats['bf_train_rmse'])])
        writer.writerow([])
        
        # === TEST TABLE ===
        writer.writerow(['TEST RESULTS'])
        writer.writerow(['Smallest Model', '', 'Optimal Compromise', '', 'Best Fitness', ''])
        writer.writerow(['Median (IQR)', 'Mean (STD)', 'Median (IQR)', 'Mean (STD)', 'Median (IQR)', 'Mean (STD)'])
        writer.writerow([format_median_iqr(stats['sm_test_rmse']),
                        format_mean_std(stats['sm_test_rmse']),
                        format_median_iqr(stats['bn_test_rmse']),
                        format_mean_std(stats['bn_test_rmse']),
                        format_median_iqr(stats['bf_test_rmse']),
                        format_mean_std(stats['bf_test_rmse'])])
        writer.writerow([])
        
        # === SIZE TABLE ===
        writer.writerow(['SIZE (Number of Nodes)'])
        writer.writerow(['Smallest Model', '', 'Optimal Compromise', '', 'Best Fitness', ''])
        writer.writerow(['Median (IQR)', 'Mean (STD)', 'Median (IQR)', 'Mean (STD)', 'Median (IQR)', 'Mean (STD)'])
        writer.writerow([format_median_iqr(stats['sm_nodes']),
                        format_mean_std(stats['sm_nodes']),
                        format_median_iqr(stats['bn_nodes']),
                        format_mean_std(stats['bn_nodes']),
                        format_median_iqr(stats['bf_nodes']),
                        format_mean_std(stats['bf_nodes'])])
    
    print(f"   Formatted summary table saved to: {filename}")

def run_single_dataset_multiple_times(dataset_name, num_runs=30, slim_version='SLIM+ABS', 
                                     use_oms=False, use_linear_scaling=False, 
                                     use_pareto_tournament=False, base_seed=None):
    """
    Run a single dataset multiple times with different seeds and calculate statistics.
    
    Args:
        dataset_name: Name of the dataset to run
        num_runs: Number of times to run the algorithm (default: 30)
        slim_version: Version of SLIM to use
        use_oms: Whether to use OMS
        use_linear_scaling: Whether to use linear scaling
        use_pareto_tournament: Whether to use Pareto tournament
        base_seed: Base seed for reproducibility (if None, uses random seeds)
    """
    # Validate dataset name
    if dataset_name not in DATASET_LOADERS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(DATASET_LOADERS.keys())}")
        return
    
    # Validate OMS usage
    two_trees_versions = ["SLIM+SIG2", "SLIM*SIG2"]
    if use_oms and slim_version not in two_trees_versions:
        print(f"⚠️  WARNING: OMS requires two_trees=True (SLIM+SIG2 or SLIM*SIG2).")
        print(f"   Current version: {slim_version}. OMS will be disabled.")
        use_oms = False
    
    # Build execution type name
    execution_type = "slim"
    if use_linear_scaling:
        execution_type += "_linear_scaling"
    if use_oms:
        execution_type += "_oms"
    if use_pareto_tournament:
        execution_type += "_pareto"
    
    print("=" * 80)
    print(f"RUNNING DATASET '{dataset_name.upper()}' - {num_runs} RUNS")
    print("=" * 80)
    print(f"SLIM Version: {slim_version}")
    print(f"Configuration:")
    print(f"  Linear Scaling: {'✓ Enabled' if use_linear_scaling else '✗ Disabled'}")
    print(f"  OMS: {'✓ Enabled' if use_oms else '✗ Disabled'}")
    print(f"  Pareto Tournament: {'✓ Enabled' if use_pareto_tournament else '✗ Disabled'}")
    print(f"Execution type: {execution_type}")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset '{dataset_name}'...")
    load_function = DATASET_LOADERS[dataset_name]
    X, y = load_function(X_y=True)
    print(f"Dataset loaded - Shape: {X.shape}, Target shape: {y.shape}")
    
    # Storage for all runs
    runs_data = []
    
    # Collect results for statistics
    bf_train_rmse_list = []
    bf_val_rmse_list = []
    bf_test_rmse_list = []
    bf_nodes_list = []
    bf_depth_list = []
    
    bn_train_rmse_list = []
    bn_val_rmse_list = []
    bn_test_rmse_list = []
    bn_nodes_list = []
    bn_depth_list = []
    
    sm_train_rmse_list = []
    sm_val_rmse_list = []
    sm_test_rmse_list = []
    sm_nodes_list = []
    sm_depth_list = []
    
    # Run the algorithm multiple times
    successful_runs = 0
    failed_runs = 0
    
    for run_idx in range(1, num_runs + 1):
        # Generate seed for this run
        if base_seed is not None:
            seed = base_seed + run_idx
        else:
            seed = np.random.randint(0, 100000)
        
        print(f"\n[Run {run_idx}/{num_runs}] Seed: {seed}")
        
        try:
            # Split data with the current seed
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)
            
            # Prepare algorithm parameters
            algorithm_params = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_val,
                'y_test': y_val,
                'dataset_name': dataset_name,
                'slim_version': slim_version,
                'pop_size': 100,
                'n_iter': 100,
                'ms_lower': 0,
                'ms_upper': 1,
                'p_inflate': 0.5,
                'reconstruct': True,
                'oms': use_oms,
                'linear_scaling': use_linear_scaling,
                'seed': seed
            }
            
            # Add Pareto tournament parameters if enabled
            if use_pareto_tournament:
                algorithm_params.update({
                    'tournament_type': "pareto",
                    'tournament_size': 5,
                    'multi_obj_attrs': ["fitness", "size"]
                })
            
            # Run the algorithm
            results = slim(**algorithm_params)
            
            # Extract all three individuals
            best_fitness_individual = results.best_fitness
            best_normalized_individual = results.best_normalized
            smallest_individual = results.smallest
            
            # === Process Best Fitness Individual ===
            best_fitness_individual.calculate_semantics(X_val, testing=True)
            best_fitness_individual.evaluate(rmse, y_val, testing=True, operator="sum")
            predictions_bf = best_fitness_individual.predict(X_test)
            test_rmse_bf = float(rmse(y_true=y_test, y_pred=predictions_bf))
            
            bf_train = round(float(best_fitness_individual.fitness) if hasattr(best_fitness_individual.fitness, 'item') else float(best_fitness_individual.fitness), 5)
            bf_val = round(float(best_fitness_individual.test_fitness) if hasattr(best_fitness_individual.test_fitness, 'item') else float(best_fitness_individual.test_fitness), 5)
            bf_test = round(test_rmse_bf, 5)
            bf_nodes = best_fitness_individual.nodes_count
            bf_depth = best_fitness_individual.depth
            
            # === Process Best Normalized Individual ===
            best_normalized_individual.calculate_semantics(X_val, testing=True)
            best_normalized_individual.evaluate(rmse, y_val, testing=True, operator="sum")
            predictions_bn = best_normalized_individual.predict(X_test)
            test_rmse_bn = float(rmse(y_true=y_test, y_pred=predictions_bn))
            
            bn_train = round(float(best_normalized_individual.fitness) if hasattr(best_normalized_individual.fitness, 'item') else float(best_normalized_individual.fitness), 5)
            bn_val = round(float(best_normalized_individual.test_fitness) if hasattr(best_normalized_individual.test_fitness, 'item') else float(best_normalized_individual.test_fitness), 5)
            bn_test = round(test_rmse_bn, 5)
            bn_nodes = best_normalized_individual.nodes_count
            bn_depth = best_normalized_individual.depth
            
            # === Process Smallest Individual ===
            smallest_individual.calculate_semantics(X_val, testing=True)
            smallest_individual.evaluate(rmse, y_val, testing=True, operator="sum")
            predictions_sm = smallest_individual.predict(X_test)
            test_rmse_sm = float(rmse(y_true=y_test, y_pred=predictions_sm))
            
            sm_train = round(float(smallest_individual.fitness) if hasattr(smallest_individual.fitness, 'item') else float(smallest_individual.fitness), 5)
            sm_val = round(float(smallest_individual.test_fitness) if hasattr(smallest_individual.test_fitness, 'item') else float(smallest_individual.test_fitness), 5)
            sm_test = round(test_rmse_sm, 5)
            sm_nodes = smallest_individual.nodes_count
            sm_depth = smallest_individual.depth
            
            # Store results (format as strings to ensure proper CSV writing)
            runs_data.append({
                'run': run_idx,
                'seed': seed,
                'bf_train_rmse': f"{bf_train:.5f}",
                'bf_val_rmse': f"{bf_val:.5f}",
                'bf_test_rmse': f"{bf_test:.5f}",
                'bf_nodes': bf_nodes,
                'bf_depth': bf_depth,
                'bn_train_rmse': f"{bn_train:.5f}",
                'bn_val_rmse': f"{bn_val:.5f}",
                'bn_test_rmse': f"{bn_test:.5f}",
                'bn_nodes': bn_nodes,
                'bn_depth': bn_depth,
                'sm_train_rmse': f"{sm_train:.5f}",
                'sm_val_rmse': f"{sm_val:.5f}",
                'sm_test_rmse': f"{sm_test:.5f}",
                'sm_nodes': sm_nodes,
                'sm_depth': sm_depth
            })
            
            # Collect for statistics
            bf_train_rmse_list.append(bf_train)
            bf_val_rmse_list.append(bf_val)
            bf_test_rmse_list.append(bf_test)
            bf_nodes_list.append(bf_nodes)
            bf_depth_list.append(bf_depth)
            
            bn_train_rmse_list.append(bn_train)
            bn_val_rmse_list.append(bn_val)
            bn_test_rmse_list.append(bn_test)
            bn_nodes_list.append(bn_nodes)
            bn_depth_list.append(bn_depth)
            
            sm_train_rmse_list.append(sm_train)
            sm_val_rmse_list.append(sm_val)
            sm_test_rmse_list.append(sm_test)
            sm_nodes_list.append(sm_nodes)
            sm_depth_list.append(sm_depth)
            
            print(f"   ✓ Run completed")
            print(f"      BF: Test={bf_test:.5f}, Nodes={bf_nodes}")
            print(f"      BN: Test={bn_test:.5f}, Nodes={bn_nodes}")
            print(f"      SM: Test={sm_test:.5f}, Nodes={sm_nodes}")
            
            successful_runs += 1
            
        except Exception as e:
            print(f"   ✗ Error in run {run_idx}: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            failed_runs += 1
            continue
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("CALCULATING STATISTICS")
    print("=" * 80)
    
    stats = {
        'bf_train_rmse': calculate_statistics(bf_train_rmse_list),
        'bf_val_rmse': calculate_statistics(bf_val_rmse_list),
        'bf_test_rmse': calculate_statistics(bf_test_rmse_list),
        'bf_nodes': calculate_statistics(bf_nodes_list),
        'bf_depth': calculate_statistics(bf_depth_list),
        'bn_train_rmse': calculate_statistics(bn_train_rmse_list),
        'bn_val_rmse': calculate_statistics(bn_val_rmse_list),
        'bn_test_rmse': calculate_statistics(bn_test_rmse_list),
        'bn_nodes': calculate_statistics(bn_nodes_list),
        'bn_depth': calculate_statistics(bn_depth_list),
        'sm_train_rmse': calculate_statistics(sm_train_rmse_list),
        'sm_val_rmse': calculate_statistics(sm_val_rmse_list),
        'sm_test_rmse': calculate_statistics(sm_test_rmse_list),
        'sm_nodes': calculate_statistics(sm_nodes_list),
        'sm_depth': calculate_statistics(sm_depth_list),
    }
    
    # Save results
    save_individual_runs(runs_data, dataset_name, execution_type)
    save_statistics_summary(stats, dataset_name, execution_type)
    save_formatted_table(stats, dataset_name, execution_type)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    
    def print_stats_row(label, stats_dict):
        mean = stats_dict['mean']
        std = stats_dict['std']
        median = stats_dict['median']
        iqr = stats_dict['iqr']
        print(f"{label:<25} {median:>10.5f} ({iqr:<10.5f}) {mean:>10.5f} ({std:<10.5f})")
    
    print(f"\n{'Metric':<25} {'Median (IQR)':<25} {'Mean (STD)':<25}")
    print("-" * 80)
    
    print("\n--- BEST FITNESS INDIVIDUAL ---")
    print_stats_row("Train RMSE", stats['bf_train_rmse'])
    print_stats_row("Validation RMSE", stats['bf_val_rmse'])
    print_stats_row("Test RMSE", stats['bf_test_rmse'])
    print_stats_row("Nodes", stats['bf_nodes'])
    print_stats_row("Depth", stats['bf_depth'])
    
    print("\n--- BEST NORMALIZED INDIVIDUAL ---")
    print_stats_row("Train RMSE", stats['bn_train_rmse'])
    print_stats_row("Validation RMSE", stats['bn_val_rmse'])
    print_stats_row("Test RMSE", stats['bn_test_rmse'])
    print_stats_row("Nodes", stats['bn_nodes'])
    print_stats_row("Depth", stats['bn_depth'])
    
    print("\n--- SMALLEST INDIVIDUAL ---")
    print_stats_row("Train RMSE", stats['sm_train_rmse'])
    print_stats_row("Validation RMSE", stats['sm_val_rmse'])
    print_stats_row("Test RMSE", stats['sm_test_rmse'])
    print_stats_row("Nodes", stats['sm_nodes'])
    print_stats_row("Depth", stats['sm_depth'])
    
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total runs attempted: {num_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {(successful_runs/num_runs)*100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    # ============================================================================
    # CONFIGURATION - Modify these variables
    # ============================================================================
    dataset_name = 'airfoil'           # Dataset to run
    num_runs = 30                      # Number of runs (default: 30)
    slim_version = 'SLIM+ABS'          # SLIM version
    use_oms = False                    # Enable OMS
    use_linear_scaling = False         # Enable Linear Scaling
    use_pareto_tournament = False      # Enable Pareto Tournament
    base_seed = 42                     # Base seed for reproducibility (None = random)
    # ============================================================================
    
    # Parse command line arguments (optional)
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        num_runs = int(sys.argv[2])
    if len(sys.argv) > 3:
        slim_version = sys.argv[3]
    
    # Parse additional boolean arguments
    for arg in sys.argv[4:]:
        if '=' in arg:
            key, value = arg.split('=')
            value_bool = value.lower() in ['true', '1', 'yes']
            if key.lower() in ['oms', 'use_oms']:
                use_oms = value_bool
            elif key.lower() in ['ls', 'linear_scaling', 'use_linear_scaling']:
                use_linear_scaling = value_bool
            elif key.lower() in ['pareto', 'pareto_tournament', 'use_pareto_tournament']:
                use_pareto_tournament = value_bool
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Number of runs: {num_runs}")
    print(f"  SLIM Version: {slim_version}")
    print(f"  OMS: {use_oms}")
    print(f"  Linear Scaling: {use_linear_scaling}")
    print(f"  Pareto Tournament: {use_pareto_tournament}")
    print(f"  Base seed: {base_seed}")
    print()
    
    start_time = datetime.now()
    print(f"Starting execution at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_single_dataset_multiple_times(
        dataset_name=dataset_name,
        num_runs=num_runs,
        slim_version=slim_version,
        use_oms=use_oms,
        use_linear_scaling=use_linear_scaling,
        use_pareto_tournament=use_pareto_tournament,
        base_seed=base_seed
    )
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nExecution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {execution_time}")
