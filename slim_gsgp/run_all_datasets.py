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

def initialize_results_file(filename="results_all_datasets.csv"):
    """
    Initialize (or overwrite) the results CSV file with headers only.
    
    Args:
        filename: Name of the output file
    """
    # Ensure filename includes log directory if not already present
    if not filename.startswith("log"):
        filename = os.path.join("log", filename)
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'dataset_name', 'execution_type',
                     'bf_train_rmse', 'bf_val_rmse', 'bf_test_rmse', 'bf_nodes', 'bf_depth',
                     'bn_train_rmse', 'bn_val_rmse', 'bn_test_rmse', 'bn_nodes', 'bn_depth',
                     'sm_train_rmse', 'sm_val_rmse', 'sm_test_rmse', 'sm_nodes', 'sm_depth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def save_results_to_file(dataset_name, 
                        best_fitness_train, best_fitness_val, best_fitness_test, best_fitness_nodes, best_fitness_depth,
                        best_norm_train, best_norm_val, best_norm_test, best_norm_nodes, best_norm_depth,
                        smallest_train, smallest_val, smallest_test, smallest_nodes, smallest_depth,
                        execution_type, filename="results_all_datasets.csv"):
    """
    Append results of all three individuals to a CSV file.
    
    Args:
        dataset_name: Name of the dataset used
        best_fitness_*: Metrics for best fitness individual
        best_norm_*: Metrics for best normalized individual
        smallest_*: Metrics for smallest size individual
        execution_type: Type of execution
        filename: Name of the output file
    """
    # Ensure filename includes log directory if not already present
    if not filename.startswith("log"):
        filename = os.path.join("log", filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'dataset_name', 'execution_type',
                     'bf_train_rmse', 'bf_val_rmse', 'bf_test_rmse', 'bf_nodes', 'bf_depth',
                     'bn_train_rmse', 'bn_val_rmse', 'bn_test_rmse', 'bn_nodes', 'bn_depth',
                     'sm_train_rmse', 'sm_val_rmse', 'sm_test_rmse', 'sm_nodes', 'sm_depth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the results
        writer.writerow({
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'execution_type': execution_type,
            'bf_train_rmse': best_fitness_train,
            'bf_val_rmse': best_fitness_val,
            'bf_test_rmse': best_fitness_test,
            'bf_nodes': best_fitness_nodes,
            'bf_depth': best_fitness_depth,
            'bn_train_rmse': best_norm_train,
            'bn_val_rmse': best_norm_val,
            'bn_test_rmse': best_norm_test,
            'bn_nodes': best_norm_nodes,
            'bn_depth': best_norm_depth,
            'sm_train_rmse': smallest_train,
            'sm_val_rmse': smallest_val,
            'sm_test_rmse': smallest_test,
            'sm_nodes': smallest_nodes,
            'sm_depth': smallest_depth
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

def run_algorithm(algorithm_name, dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, oms_enabled, linear_scaling_enabled, use_pareto_tournament=False, use_simplification=True, slim_version='SLIM+SIG2', output_filename="results_all_datasets.csv"):
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
        use_simplification: Whether to use simplification during evolution
        slim_version: Version of SLIM to use (e.g., 'SLIM+SIG2', 'SLIM+ABS', 'SLIM*ABS')
        output_filename: Name of the output CSV file
    
    Returns:
        None
    """
    try:
        oms_suffix = " oms" if oms_enabled else ""
        linear_suffix = " linear scaling" if linear_scaling_enabled else ""
        pareto_suffix = " pareto" if use_pareto_tournament else ""
        simplification_suffix = " no_simplif" if not use_simplification else ""
        execution_type = f"slim{linear_suffix}{oms_suffix}{pareto_suffix}{simplification_suffix}"
        
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
            'linear_scaling': linear_scaling_enabled,
            'use_simplification': use_simplification
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
            tree_struct, tree_dicts = convert_slim_individual_to_normal_tree(best_fitness_individual)
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
            
            # Process normalized individual
            tree_struct_norm, tree_dicts_norm = convert_slim_individual_to_normal_tree(best_normalized_individual)
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
            
            # Process smallest individual
            tree_struct_sm, tree_dicts_sm = convert_slim_individual_to_normal_tree(smallest_individual)
            if tree_struct_sm and tree_dicts_sm:
                simplified_tree_sm, nodes_removed_sm = simplify_constant_operations(tree_struct_sm, tree_dicts_sm['CONSTANTS'])
                original_expr_sm = tree_to_expression(tree_struct_sm)
                node_count_sm = count_nodes(tree_struct_sm)
                
                if nodes_removed_sm > 0:
                    simplified_expr_sm = tree_to_expression(simplified_tree_sm)
                    save_simplification_to_txt(
                        dataset_name, execution_type, "smallest",
                        original_expr_sm, simplified_expr_sm, nodes_removed_sm,
                        node_count_sm, count_nodes(simplified_tree_sm)
                    )
                else:
                    # No simplification possible - save the tree with a message
                    save_no_simplification_to_txt(
                        dataset_name, execution_type, "smallest",
                        original_expr_sm, node_count_sm
                    )
        except Exception as e:
            # Simplification is optional, don't fail the whole execution
            print(f"Warning: Error during simplification for {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Save results of all three individuals
        save_results_to_file(
            dataset_name=dataset_name,
            best_fitness_train=bf_train,
            best_fitness_val=bf_val,
            best_fitness_test=bf_test,
            best_fitness_nodes=bf_nodes,
            best_fitness_depth=bf_depth,
            best_norm_train=bn_train,
            best_norm_val=bn_val,
            best_norm_test=bn_test,
            best_norm_nodes=bn_nodes,
            best_norm_depth=bn_depth,
            smallest_train=sm_train,
            smallest_val=sm_val,
            smallest_test=sm_test,
            smallest_nodes=sm_nodes,
            smallest_depth=sm_depth,
            execution_type=execution_type,
            filename=output_filename
        )
        
        print(f"    ✓ {execution_type} completed")
        print(f"       Best Fitness    - Train: {bf_train:.5f}, Val: {bf_val:.5f}, Test: {bf_test:.5f}, Nodes: {bf_nodes}, Depth: {bf_depth}")
        print(f"       Best Normalized - Train: {bn_train:.5f}, Val: {bn_val:.5f}, Test: {bn_test:.5f}, Nodes: {bn_nodes}, Depth: {bn_depth}")
        print(f"       Smallest        - Train: {sm_train:.5f}, Val: {sm_val:.5f}, Test: {sm_test:.5f}, Nodes: {sm_nodes}, Depth: {sm_depth}")
        
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

def run_all_datasets(slim_version='SLIM+ABS', output_filename=None, 
                     use_oms=True, use_linear_scaling=False, use_pareto_tournament=False, use_simplification=True):
    """
    Run all datasets with a single specified configuration.
    
    Args:
        slim_version: Version of SLIM to use (e.g., 'SLIM+SIG2', 'SLIM+ABS', 'SLIM*ABS')
        output_filename: Custom filename for results CSV. If None, generates based on configuration
        use_oms: Whether to use OMS (Operator Mutation Selection)
        use_linear_scaling: Whether to use linear scaling
        use_pareto_tournament: Whether to use Pareto tournament selection
        use_simplification: Whether to use simplification during evolution
    """
    # Validate OMS usage
    two_trees_versions = ["SLIM+SIG2", "SLIM*SIG2"]
    if use_oms and slim_version not in two_trees_versions:
        print(f"⚠️  WARNING: OMS requires two_trees=True (SLIM+SIG2 or SLIM*SIG2).")
        print(f"   Current version: {slim_version}. OMS will be disabled.")
        use_oms = False
    
    # Generate filename if not provided
    if output_filename is None:
        # Convert version name to valid filename
        version_suffix = slim_version.replace('+', '_').replace('*', '_').replace(' ', '_').lower()
        config_suffix = ""
        if use_linear_scaling:
            config_suffix += "_ls"
        if use_oms:
            config_suffix += "_oms"
        if use_pareto_tournament:
            config_suffix += "_pareto"
        if not use_simplification:
            config_suffix += "_no_simplif"
        output_filename = os.path.join("log", f"results_all_datasets_{version_suffix}{config_suffix}.csv")
    
    # Define all available datasets
    # NOTE: Dataset numbering goes from 1 to 14 (dataset 12 'istanbul' is commented out)
    # To re-enable dataset 12, uncomment the 'istanbul' line below
    datasets = [
        ('airfoil', load_airfoil),              # Dataset 1
        ('bike_sharing', load_bike_sharing),    # Dataset 2
        ('bioavailability', load_bioav),        # Dataset 3
        ('boston', load_boston),                # Dataset 4
        ('breast_cancer', load_breast_cancer),  # Dataset 5
        ('concrete_slump', load_concrete_slump),# Dataset 6
        ('concrete_strength', load_concrete_strength),  # Dataset 7
        ('diabetes', load_diabetes),            # Dataset 8
        ('efficiency_cooling', load_efficiency_cooling),  # Dataset 9
        ('efficiency_heating', load_efficiency_heating),  # Dataset 10
        ('forest_fires', load_forest_fires),    # Dataset 11
        # ('istanbul', load_istanbul),          # Dataset 12 - COMMENTED OUT (uncomment to re-enable)
        ('parkinson_updrs', load_parkinson_updrs),  # Dataset 13
        ('ppb', load_ppb),                      # Dataset 14
        ('resid_build_sale_price', load_resid_build_sale_price)  # Dataset 15
    ]
    
    # Build execution type name
    execution_type = "slim"
    if use_linear_scaling:
        execution_type += " linear scaling"
    if use_oms:
        execution_type += " oms"
    if use_pareto_tournament:
        execution_type += " pareto"
    if not use_simplification:
        execution_type += " no_simplif"
    
    # Initialize results file (this will overwrite any existing file)
    initialize_results_file(output_filename)
    
    print("=" * 80)
    print("RUNNING ALL DATASETS WITH SINGLE CONFIGURATION")
    print("=" * 80)
    print(f"SLIM Version: {slim_version}")
    print(f"Output file: {output_filename}")
    print(f"Simplifications file: log/simplifications_all.txt")
    print(f"\nConfiguration:")
    print(f"  Linear Scaling: {'✓ Enabled' if use_linear_scaling else '✗ Disabled'}")
    print(f"  OMS: {'✓ Enabled' if use_oms else '✗ Disabled'}")
    print(f"  Pareto Tournament: {'✓ Enabled' if use_pareto_tournament else '✗ Disabled'}")
    print(f"  Simplification: {'✓ Enabled' if use_simplification else '✗ Disabled'}")
    print(f"\nExecution type: {execution_type}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total executions: {len(datasets)}")
    print("=" * 80)
    
    successful_runs = 0
    failed_runs = 0
    total_runs = len(datasets)
    
    for dataset_idx, (dataset_name, load_function) in enumerate(datasets, 1):
        print(f"\n[{dataset_idx}/{len(datasets)}] Processing dataset: {dataset_name}")
        
        try:
            # Load dataset
            X, y = load_function(X_y=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)
            
            print(f"  Dataset loaded - Shape: {X.shape}, Target shape: {y.shape}")
            
            # Run the single specified configuration
            run_algorithm(
                "slim", dataset_name,
                X_train, y_train, X_val, y_val, X_test, y_test,
                use_oms, use_linear_scaling, use_pareto_tournament, use_simplification,
                slim_version, output_filename
            )
            successful_runs += 1
                    
        except Exception as e:
            print(f"  ✗ Error loading dataset {dataset_name}: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            failed_runs += 1
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
    
    # ============================================================================
    # CONFIGURATION - Modify these variables to select the execution configuration
    # ============================================================================
    slim_version = 'SLIM+ABS'          # SLIM version: 'SLIM+ABS', 'SLIM+SIG2', 'SLIM*ABS', 'SLIM*SIG2'
    use_oms = False                    # Enable OMS (Operator Mutation Selection) - requires SLIM+SIG2 or SLIM*SIG2
    use_linear_scaling = False         # Enable Linear Scaling
    use_pareto_tournament = False      # Enable Pareto Tournament Selection
    use_simplification = True          # Enable Simplification during evolution
    output_filename = None             # Custom output filename (None = auto-generate)
    # ============================================================================
    
    # Parse command line arguments (optional - overrides above configuration)
    if len(sys.argv) > 1:
        slim_version = sys.argv[1]
    if len(sys.argv) > 2:
        # Parse boolean arguments: python run_all_datasets.py SLIM+ABS oms=True ls=True pareto=False simplif=False
        for arg in sys.argv[2:]:
            if '=' in arg:
                key, value = arg.split('=')
                value_bool = value.lower() in ['true', '1', 'yes']
                if key.lower() in ['oms', 'use_oms']:
                    use_oms = value_bool
                elif key.lower() in ['ls', 'linear_scaling', 'use_linear_scaling']:
                    use_linear_scaling = value_bool
                elif key.lower() in ['pareto', 'pareto_tournament', 'use_pareto_tournament']:
                    use_pareto_tournament = value_bool
                elif key.lower() in ['simplif', 'simplification', 'use_simplification']:
                    use_simplification = value_bool
                elif key.lower() in ['output', 'output_filename']:
                    output_filename = value
    
    print(f"Configuration:")
    print(f"  SLIM Version: {slim_version}")
    print(f"  OMS: {use_oms}")
    print(f"  Linear Scaling: {use_linear_scaling}")
    print(f"  Pareto Tournament: {use_pareto_tournament}")
    print(f"  Simplification: {use_simplification}")
    print(f"  Output filename: {output_filename if output_filename else 'Auto-generated'}")
    print()
    
    start_time = datetime.now()
    print(f"Starting execution at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_all_datasets(
        slim_version=slim_version, 
        output_filename=output_filename,
        use_oms=use_oms,
        use_linear_scaling=use_linear_scaling,
        use_pareto_tournament=use_pareto_tournament,
        use_simplification=use_simplification
    )
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nExecution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {execution_time}")