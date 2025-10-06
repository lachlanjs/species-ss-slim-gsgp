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
    # Ensure filename includes log directory if not already present
    if not filename.startswith("log"):
        filename = os.path.join("log", filename)
    
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

def save_simplification_to_txt(dataset_name, execution_type, individual_type, original_expr, simplified_expr, nodes_removed, original_nodes, simplified_nodes, txt_file="log/simplifications_all.txt"):
    """
    Append a simplification entry to a text file.
    
    Args:
        dataset_name: Name of the dataset
        execution_type: Type of execution
        individual_type: "fitness" or "normalized"
        original_expr: Original expression
        simplified_expr: Simplified expression
        nodes_removed: Number of nodes removed
        original_nodes: Original node count
        simplified_nodes: Simplified node count
        txt_file: Path to the text file
    """
    import os
    from datetime import datetime
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(txt_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_exists = os.path.exists(txt_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(txt_file, 'a', encoding='utf-8') as f:
        # Write header if file is new
        if not file_exists:
            f.write("="*100 + "\n")
            f.write("SIMPLIFIED TREES - ALL DATASETS\n")
            f.write("="*100 + "\n\n")
        
        f.write("-"*100 + "\n")
        f.write(f"[{timestamp}] {dataset_name} | {execution_type} | {individual_type.upper()}\n")
        f.write(f"Nodes: {original_nodes} → {simplified_nodes} (removed {nodes_removed})\n")
        f.write(f"\nOriginal:\n  {original_expr}\n")
        f.write(f"\nSimplified:\n  {simplified_expr}\n")
        f.write("-"*100 + "\n\n")

def save_no_simplification_to_txt(dataset_name, execution_type, individual_type, original_expr, node_count, txt_file="log/simplifications_all.txt"):
    """
    Append an entry for a tree that could not be simplified.
    
    Args:
        dataset_name: Name of the dataset
        execution_type: Type of execution
        individual_type: "fitness" or "normalized"
        original_expr: Original expression
        node_count: Number of nodes
        txt_file: Path to the text file
    """
    import os
    from datetime import datetime
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(txt_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_exists = os.path.exists(txt_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(txt_file, 'a', encoding='utf-8') as f:
        # Write header if file is new
        if not file_exists:
            f.write("="*100 + "\n")
            f.write("SIMPLIFIED TREES - ALL DATASETS\n")
            f.write("="*100 + "\n\n")
        
        f.write("-"*100 + "\n")
        f.write(f"[{timestamp}] {dataset_name} | {execution_type} | {individual_type.upper()}\n")
        f.write(f"Nodes: {node_count}\n")
        f.write(f"Status: No simplifications possible (no adjacent constant operations found)\n")
        f.write(f"\nTree expression:\n  {original_expr}\n")
        f.write("-"*100 + "\n\n")

def run_algorithm(algorithm_name, dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, oms_enabled, linear_scaling_enabled, use_pareto_tournament=False, slim_version='SLIM+SIG2', output_filename="results_all_datasets.csv"):
    """
    Run a specific algorithm with given parameters.
    
    Args:
        algorithm_name: Name of the algorithm for logging
        dataset_name: Name of the dataset
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        oms_enabled: Whether to use OMS
        linear_scaling_enabled: Whether to use linear scaling
        use_pareto_tournament: Whether to use Pareto tournament selection
        slim_version: Version of SLIM to use (e.g., 'SLIM+SIG2', 'SLIM+ABS', 'SLIM*ABS')
        output_filename: Name of the output CSV file
    
    Returns:
        None
    """
    try:
        oms_suffix = " oms" if oms_enabled else ""
        linear_suffix = " linear scaling" if linear_scaling_enabled else ""
        pareto_suffix = " pareto" if use_pareto_tournament else ""
        execution_type = f"slim{linear_suffix}{oms_suffix}{pareto_suffix}"
        
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
            'oms': oms_enabled,
            'linear_scaling': linear_scaling_enabled
        }
        
        # Add Pareto tournament parameters if enabled
        if use_pareto_tournament:
            algorithm_params.update({
                'tournament_type': "pareto",
                'tournament_size': 5,
                'multi_obj_attrs': ["fitness", "size"]
            })
        
        # Run the algorithm
        final_tree = slim(**algorithm_params)
        
        # Evaluate the final tree on validation data
        final_tree.calculate_semantics(X_val, testing=True)
        final_tree.evaluate(rmse, y_val, testing=True, operator="sum")
        
        # Get the prediction on test set
        predictions = final_tree.predict(X_test)
        test_rmse = float(rmse(y_true=y_test, y_pred=predictions))
        
        # Try to simplify and save trees
        try:
            from utils.simplification import convert_slim_individual_to_normal_tree, simplify_constant_operations
            import re
            
            # Helper function to convert tree to math expression
            def tree_to_expression(structure):
                def clean_node(node):
                    if isinstance(node, tuple):
                        return node
                    node_str = str(node)
                    if "np.str_(" in node_str:
                        node_str = re.sub(r"np\.str_\('([^']+)'\)", r"\1", node_str)
                    node_str = node_str.strip("'\"")
                    if node_str.startswith("constant_"):
                        const_val = node_str.replace("constant_", "")
                        if const_val.startswith("_"):
                            const_val = "-" + const_val[1:]
                        return const_val
                    return node_str
                
                def convert_rec(struct):
                    if not isinstance(struct, tuple) or len(struct) < 3:
                        return clean_node(struct)
                    op = clean_node(struct[0])
                    left = convert_rec(struct[1])
                    right = convert_rec(struct[2])
                    op_sym = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}.get(op, op)
                    prec = {'add': 1, 'subtract': 1, 'multiply': 2, 'divide': 2}
                    curr_prec = prec.get(op, 0)
                    
                    if isinstance(struct[1], tuple) and len(struct[1]) >= 3:
                        left_op = clean_node(struct[1][0])
                        if prec.get(left_op, 0) < curr_prec:
                            left = f"({left})"
                    
                    if isinstance(struct[2], tuple) and len(struct[2]) >= 3:
                        right_op = clean_node(struct[2][0])
                        right_prec = prec.get(right_op, 0)
                        if right_prec < curr_prec or (right_prec == curr_prec and op in ['subtract', 'divide']):
                            right = f"({right})"
                    
                    return f"{left} {op_sym} {right}"
                
                return convert_rec(structure)
            
            def count_nodes(struct):
                if not isinstance(struct, tuple):
                    return 1
                return 1 + sum(count_nodes(struct[i]) for i in range(1, len(struct)))
            
            # Process fitness individual
            tree_struct, tree_dicts = convert_slim_individual_to_normal_tree(final_tree.best_fitness)
            if tree_struct and tree_dicts:
                simplified_tree, nodes_removed = simplify_constant_operations(tree_struct, tree_dicts['CONSTANTS'])
                original_expr = tree_to_expression(tree_struct)
                node_count = count_nodes(tree_struct)
                
                if nodes_removed > 0:
                    simplified_expr = tree_to_expression(simplified_tree)
                    save_simplification_to_txt(
                        dataset_name, execution_type, "fitness",
                        original_expr, simplified_expr, nodes_removed,
                        node_count, count_nodes(simplified_tree)
                    )
                else:
                    # No simplification possible - save the tree with a message
                    save_no_simplification_to_txt(
                        dataset_name, execution_type, "fitness",
                        original_expr, node_count
                    )
            
            # Process normalized individual if exists
            if hasattr(final_tree, 'best_normalized') and final_tree.best_normalized:
                tree_struct_norm, tree_dicts_norm = convert_slim_individual_to_normal_tree(final_tree.best_normalized)
                if tree_struct_norm and tree_dicts_norm:
                    simplified_tree_norm, nodes_removed_norm = simplify_constant_operations(tree_struct_norm, tree_dicts_norm['CONSTANTS'])
                    original_expr_norm = tree_to_expression(tree_struct_norm)
                    node_count_norm = count_nodes(tree_struct_norm)
                    
                    if nodes_removed_norm > 0:
                        simplified_expr_norm = tree_to_expression(simplified_tree_norm)
                        save_simplification_to_txt(
                            dataset_name, execution_type, "normalized",
                            original_expr_norm, simplified_expr_norm, nodes_removed_norm,
                            node_count_norm, count_nodes(simplified_tree_norm)
                        )
                    else:
                        # No simplification possible - save the tree with a message
                        save_no_simplification_to_txt(
                            dataset_name, execution_type, "normalized",
                            original_expr_norm, node_count_norm
                        )
        except Exception as e:
            # Simplification is optional, don't fail the whole execution
            print(f"Warning: Error during simplification for {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
        output_filename = os.path.join("log", f"results_all_datasets_{version_suffix}.csv")
    
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
    
    # Algorithm configurations - now we use linear_scaling parameter
    linear_scaling_configs = [
        (False, "slim"),                    # Standard SLIM
        (True, "slim linear scaling")       # SLIM with Linear Scaling
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
    print(f"Simplifications file: log/simplifications_all.txt")
    oms_msg = "Yes" if supports_oms else "No (requires two_trees=True)"
    print(f"OMS Support: {oms_msg}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total linear scaling configs: {len(linear_scaling_configs)}")
    print(f"Total execution configurations: {len(execution_configs)}")
    print(f"Total executions: {len(datasets) * len(linear_scaling_configs) * len(execution_configs)}")
    print(f"Execution types: {', '.join(execution_type_names)}")
    print("=" * 80)
    
    successful_runs = 0
    failed_runs = 0
    total_runs = len(datasets) * len(linear_scaling_configs) * len(execution_configs)
    
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
            for linear_scaling_enabled, algorithm_name in linear_scaling_configs:
                for oms_enabled, use_pareto_tournament in execution_configs:
                    run_algorithm(
                        algorithm_name, dataset_name,
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        oms_enabled, linear_scaling_enabled, use_pareto_tournament,
                        slim_version, output_filename
                    )
                    successful_runs += 1
                    
        except Exception as e:
            print(f"  ✗ Error loading dataset {dataset_name}: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            failed_runs += len(linear_scaling_configs) * len(execution_configs)
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