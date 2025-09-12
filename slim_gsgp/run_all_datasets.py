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

from main_slim import slim  # import the slim_gsgp library
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
from datetime import datetime
import traceback

def slim_linear_scaling(**kwargs):
    """
    Wrapper function for slim with linear_scaling=True
    """
    kwargs['linear_scaling'] = True
    return slim(**kwargs)

def save_results_to_file(dataset_name, training_rmse, validation_rmse, test_rmse, nodes_count, execution_type, filename="results_all_datasets.csv"):
    """
    Save the results to a CSV file.
    
    Args:
        dataset_name: Name of the dataset used
        training_rmse: Training fitness (RMSE)
        validation_rmse: Validation fitness (RMSE)
        test_rmse: Final test fitness (RMSE)
        nodes_count: Number of nodes in the final tree
        execution_type: Type of execution
        filename: Name of the output file
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'dataset_name', 'training_rmse', 'validation_rmse', 'test_rmse', 'nodes_count', 'execution_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the results
        writer.writerow({
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'training_rmse': training_rmse,
            'validation_rmse': validation_rmse,
            'test_rmse': test_rmse,
            'nodes_count': nodes_count,
            'execution_type': execution_type
        })

def run_algorithm(algorithm_func, algorithm_name, dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, oms_enabled, use_pareto_tournament=False, slim_version='SLIM+SIG2', output_filename="results_all_datasets.csv"):
    """
    Run a specific algorithm with given parameters.
    
    Args:
        algorithm_func: The algorithm function to run (slim or slim_linear_scaling)
        algorithm_name: Name of the algorithm for logging
        dataset_name: Name of the dataset
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        oms_enabled: Whether to use OMS
        use_pareto_tournament: Whether to use Pareto tournament selection
        slim_version: Version of SLIM to use (e.g., 'SLIM+SIG2', 'SLIM+ABS', 'SLIM*ABS')
        output_filename: Name of the output CSV file
    
    Returns:
        None
    """
    try:
        oms_suffix = " oms" if oms_enabled else ""
        pareto_suffix = " pareto" if use_pareto_tournament else ""
        execution_type = f"{algorithm_name}{oms_suffix}{pareto_suffix}"
        
        print(f"  Running {execution_type}...")
        
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
            'oms': oms_enabled
        }
        
        # Add Pareto tournament parameters if enabled
        if use_pareto_tournament:
            algorithm_params.update({
                'tournament_type': "pareto",
                'tournament_size': 5,
                'multi_obj_attrs': ["fitness", "size"]
            })
        
        # Run the algorithm
        final_tree = algorithm_func(**algorithm_params)
        
        # Evaluate the final tree on validation data
        final_tree.calculate_semantics(X_val, testing=True)
        final_tree.evaluate(rmse, y_val, testing=True, operator="sum")
        
        # Get the prediction on test set
        predictions = final_tree.predict(X_test)
        test_rmse = float(rmse(y_true=y_test, y_pred=predictions))
        
        # Save results
        save_results_to_file(
            dataset_name=dataset_name,
            training_rmse=final_tree.fitness,
            validation_rmse=final_tree.test_fitness,
            test_rmse=test_rmse,
            nodes_count=final_tree.nodes_count,
            execution_type=execution_type,
            filename=output_filename
        )
        
        print(f"    ✓ {execution_type} completed - Train: {final_tree.fitness:.6f}, Val: {final_tree.test_fitness:.6f}, Test: {test_rmse:.6f}, Nodes: {final_tree.nodes_count}")
        
    except Exception as e:
        print(f"    ✗ Error in {execution_type}: {str(e)}")
        print(f"    Traceback: {traceback.format_exc()}")

def get_valid_execution_configs(slim_version):
    """
    Get valid execution configurations based on SLIM version.
    OMS only works with two_trees=True versions (SLIM+SIG2, SLIM*SIG2)
    
    Args:
        slim_version: Version of SLIM being used
        
    Returns:
        list: Valid execution configurations for this SLIM version
    """
    # Check if this SLIM version supports OMS (requires two_trees=True)
    two_trees_versions = ["SLIM+SIG2", "SLIM*SIG2"]
    supports_oms = slim_version in two_trees_versions
    
    if supports_oms:
        # Full configurations for versions that support OMS
        return [
            (False, False),  # Standard
            (True, False),   # OMS only
            (True, True)     # OMS + Pareto tournament
        ]
    else:
        # Limited configurations for versions that don't support OMS
        return [
            (False, False),  # Standard only
            (False, True)    # Pareto tournament only (without OMS)
        ]

def run_all_datasets(slim_version='SLIM+SIG2', output_filename=None):
    """
    Run all datasets with all algorithm combinations.
    
    Args:
        slim_version: Version of SLIM to use (e.g., 'SLIM+SIG2', 'SLIM+ABS', 'SLIM*ABS')
        output_filename: Custom filename for results CSV. If None, generates based on slim_version
    """
    # Generate filename if not provided
    if output_filename is None:
        # Convert version name to valid filename
        version_suffix = slim_version.replace('+', '_').replace('*', '_').replace(' ', '_').lower()
        output_filename = f"results_all_datasets_{version_suffix}.csv"
    # Define all available datasets
    datasets = [
        ('airfoil', load_airfoil),
        ('bike_sharing', load_bike_sharing),
        ('bioavailability', load_bioav),
        ('boston', load_boston),
        ('breast_cancer', load_breast_cancer),
        ('concrete_slump', load_concrete_slump),
        ('concrete_strength', load_concrete_strength),
        ('diabetes', load_diabetes),
        ('efficiency_cooling', load_efficiency_cooling),
        ('efficiency_heating', load_efficiency_heating),
        ('forest_fires', load_forest_fires),
        ('istanbul', load_istanbul),
        ('parkinson_updrs', load_parkinson_updrs),
        ('ppb', load_ppb),
        ('resid_build_sale_price', load_resid_build_sale_price)
    ]
    
    # Algorithm configurations
    algorithms = [
        (slim, "slim"),
        (slim_linear_scaling, "slim linear scaling")
    ]
    
    # Configuration combinations: (oms_enabled, use_pareto_tournament)
    execution_configs = get_valid_execution_configs(slim_version)
    
    # Get readable names for execution types
    def get_execution_type_name(oms_enabled, use_pareto_tournament, supports_oms):
        if oms_enabled and use_pareto_tournament:
            return "OMS + Pareto Tournament"
        elif oms_enabled:
            return "OMS only"
        elif use_pareto_tournament:
            return "Pareto Tournament only"
        else:
            return "Standard"
    
    supports_oms = slim_version in ["SLIM+SIG2", "SLIM*SIG2"]
    execution_type_names = [get_execution_type_name(oms, pareto, supports_oms) 
                           for oms, pareto in execution_configs]
    
    print("=" * 80)
    print("RUNNING ALL DATASETS WITH ALL ALGORITHM COMBINATIONS")
    print("=" * 80)
    print(f"SLIM Version: {slim_version}")
    print(f"Output file: {output_filename}")
    print(f"OMS Support: {'Yes' if supports_oms else 'No (requires two_trees=True)'}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total algorithms: {len(algorithms)}")
    print(f"Total execution configurations: {len(execution_configs)}")
    print(f"Total executions: {len(datasets) * len(algorithms) * len(execution_configs)}")
    print(f"Execution types: {', '.join(execution_type_names)}")
    print("=" * 80)
    
    successful_runs = 0
    failed_runs = 0
    total_runs = len(datasets) * len(algorithms) * len(execution_configs)
    
    for dataset_idx, (dataset_name, load_function) in enumerate(datasets, 1):
        print(f"\n[{dataset_idx}/{len(datasets)}] Processing dataset: {dataset_name}")
        
        try:
            # Load dataset
            X, y = load_function(X_y=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)
            
            print(f"  Dataset loaded - Shape: {X.shape}, Target shape: {y.shape}")
            
            # Run all algorithm combinations
            for algorithm_func, algorithm_name in algorithms:
                for oms_enabled, use_pareto_tournament in execution_configs:
                    run_algorithm(
                        algorithm_func, algorithm_name, dataset_name,
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        oms_enabled, use_pareto_tournament, slim_version, output_filename
                    )
                    successful_runs += 1
                    
        except Exception as e:
            print(f"  ✗ Error loading dataset {dataset_name}: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            failed_runs += len(algorithms) * len(execution_configs)
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total executions attempted: {total_runs}")
    print(f"Successful executions: {successful_runs}")
    print(f"Failed executions: {failed_runs}")
    print(f"Success rate: {(successful_runs/total_runs)*100:.1f}%")
    print(f"Results saved to: {output_filename}")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    slim_version = 'SLIM+SIG2'  # Default
    output_filename = None
    
    if len(sys.argv) > 1:
        slim_version = sys.argv[1]
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    
    print(f"Configuration:")
    print(f"  SLIM Version: {slim_version}")
    print(f"  Output filename: {output_filename if output_filename else 'Auto-generated'}")
    print()
    
    start_time = datetime.now()
    print(f"Starting execution at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_all_datasets(slim_version=slim_version, output_filename=output_filename)
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nExecution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {execution_time}")
