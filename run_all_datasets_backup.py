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
from main_slim_linear_scaling import slim_linear_scaling
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

def run_algorithm(algorithm_func, algorithm_name, dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, oms_enabled):
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
    
    Returns:
        None
    """
    try:
        oms_suffix = " oms" if oms_enabled else ""
        execution_type = f"{algorithm_name}{oms_suffix}"
        
        print(f"  Running {execution_type}...")
        
        # Run the algorithm
        final_tree = algorithm_func(
            X_train=X_train, y_train=y_train,
            X_test=X_val, y_test=y_val,
            dataset_name=dataset_name, 
            slim_version='SLIM+SIG2', 
            pop_size=100, 
            n_iter=100,
            ms_lower=0, 
            ms_upper=1, 
            p_inflate=0.5, 
            reconstruct=True,
            oms=oms_enabled
        )
        
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
            execution_type=execution_type
        )
        
        print(f"    ✓ {execution_type} completed - Train: {final_tree.fitness:.6f}, Val: {final_tree.test_fitness:.6f}, Test: {test_rmse:.6f}, Nodes: {final_tree.nodes_count}")
        
    except Exception as e:
        print(f"    ✗ Error in {execution_type}: {str(e)}")
        print(f"    Traceback: {traceback.format_exc()}")

def run_all_datasets():
    """
    Run all datasets with all algorithm combinations.
    """
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
    
    oms_configs = [False, True]  # Without OMS first, then with OMS
    
    print("=" * 80)
    print("RUNNING ALL DATASETS WITH ALL ALGORITHM COMBINATIONS")
    print("=" * 80)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total algorithms: {len(algorithms)}")
    print(f"Total OMS configurations: {len(oms_configs)}")
    print(f"Total executions: {len(datasets) * len(algorithms) * len(oms_configs)}")
    print("=" * 80)
    
    successful_runs = 0
    failed_runs = 0
    total_runs = len(datasets) * len(algorithms) * len(oms_configs)
    
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
                for oms_enabled in oms_configs:
                    run_algorithm(
                        algorithm_func, algorithm_name, dataset_name,
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        oms_enabled
                    )
                    successful_runs += 1
                    
        except Exception as e:
            print(f"  ✗ Error loading dataset {dataset_name}: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            failed_runs += len(algorithms) * len(oms_configs)
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total executions attempted: {total_runs}")
    print(f"Successful executions: {successful_runs}")
    print(f"Failed executions: {failed_runs}")
    print(f"Success rate: {(successful_runs/total_runs)*100:.1f}%")
    print(f"Results saved to: results_all_datasets.csv")
    print("=" * 80)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting execution at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_all_datasets()
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nExecution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {execution_time}")
