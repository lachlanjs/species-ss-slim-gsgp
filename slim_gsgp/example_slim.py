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
from slim_gsgp.main_slim import slim  # import the slim_gsgp library
from slim_gsgp.datasets.data_loader import load_airfoil  # import the loader for the dataset airfoil
from slim_gsgp.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim_gsgp.utils.utils import train_test_split  # import the train-test split function
import csv
import os
from datetime import datetime

def save_results_to_file(dataset_name, training_rmse, validation_rmse, test_rmse, execution_type, filename="results_slim.csv"):
    """
    Save the results to a CSV file.
    
    Args:
        dataset_name: Name of the dataset used
        training_rmse: Training fitness (RMSE)
        validation_rmse: Validation fitness (RMSE)
        test_rmse: Final test fitness (RMSE)
        execution_type: Type of execution (e.g., "slim")
        filename: Name of the output file
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'dataset_name', 'training_rmse', 'validation_rmse', 'test_rmse', 'execution_type']
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
            'execution_type': execution_type
        })

# Load the airfoil dataset
X, y = load_airfoil(X_y=True)

# Split into train and test sets (with fixed seed for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=42)

# Split the test set into validation and test sets (with fixed seed for reproducibility)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=42)

# Apply the SLIM GSGP algorithm (with fixed seed for reproducibility and automatic simplification)
results = slim(X_train=X_train, y_train=y_train,
               X_test=X_val, y_test=y_val,
               dataset_name='airfoil', slim_version='SLIM+ABS', pop_size=100, n_iter=100,
               ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True, 
               # tournament_type="pareto", tournament_size=5, multi_obj_attrs=["fitness", "size"], 
               oms=False, linear_scaling=True, enable_plotting=False, auto_simplify=True, seed=42)

# Extract both individuals
best_fitness_individual = results.best_fitness
best_normalized_individual = results.best_normalized

print("\n" + "="*80)
print("RESULTS FOR BEST FITNESS INDIVIDUAL")
print("="*80)

# Show the best fitness individual structure
best_fitness_individual.print_tree_representation()

# Show linear scaling information if enabled
if hasattr(best_fitness_individual, 'print_scaling_info'):
    best_fitness_individual.print_scaling_info()

# Evaluate the best fitness individual on validation data
best_fitness_individual.calculate_semantics(X_val, testing=True)
best_fitness_individual.evaluate(rmse, y_val, testing=True, operator="sum")

# Show fitness information for best fitness individual
print(f"\n=== FITNESS INFORMATION (Best Fitness) ===")
print(f"Training fitness (RMSE): {best_fitness_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {best_fitness_individual.test_fitness:.6f}")

# Get the prediction of the best fitness individual on the test set
predictions_best_fitness = best_fitness_individual.predict(X_test)
test_rmse_best_fitness = float(rmse(y_true=y_test, y_pred=predictions_best_fitness))
print(f"Final test fitness (RMSE): {test_rmse_best_fitness:.6f}")

print(f"\n=== SUMMARY (Best Fitness) ===")
print(f"Number of nodes: {best_fitness_individual.nodes_count}")
print(f"Tree depth: {best_fitness_individual.depth}")
print(f"Train -> Validation -> Test RMSE: {best_fitness_individual.fitness:.6f} -> {best_fitness_individual.test_fitness:.6f} -> {test_rmse_best_fitness:.6f}")

print("\n" + "="*80)
print("RESULTS FOR BEST NORMALIZED INDIVIDUAL ")
print("="*80)

# Show the best normalized individual structure
best_normalized_individual.print_tree_representation()

# Show linear scaling information if enabled
if hasattr(best_normalized_individual, 'print_scaling_info'):
    best_normalized_individual.print_scaling_info()

# Evaluate the best normalized individual on validation data
best_normalized_individual.calculate_semantics(X_val, testing=True)
best_normalized_individual.evaluate(rmse, y_val, testing=True, operator="sum")

# Show fitness information for best normalized individual
print(f"\n=== FITNESS INFORMATION (Best Normalized) ===")
print(f"Training fitness (RMSE): {best_normalized_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {best_normalized_individual.test_fitness:.6f}")

# Get the prediction of the best normalized individual on the test set
predictions_best_normalized = best_normalized_individual.predict(X_test)
test_rmse_best_normalized = float(rmse(y_true=y_test, y_pred=predictions_best_normalized))
print(f"Final test fitness (RMSE): {test_rmse_best_normalized:.6f}")

print(f"\n=== SUMMARY (Best Normalized) ===")
print(f"Number of nodes: {best_normalized_individual.nodes_count}")
print(f"Tree depth: {best_normalized_individual.depth}")
print(f"Train -> Validation -> Test RMSE: {best_normalized_individual.fitness:.6f} -> {best_normalized_individual.test_fitness:.6f} -> {test_rmse_best_normalized:.6f}")

# Save results to file
dataset_name = 'airfoil'
# Determine execution type based on OMS and linear scaling usage
oms_used = False  # Change this to match the oms parameter above
linear_scaling_used = True  # Change this to match the linear_scaling parameter above

if linear_scaling_used and oms_used:
    execution_type = 'slim linear scaling oms'
elif linear_scaling_used:
    execution_type = 'slim linear scaling'
elif oms_used:
    execution_type = 'slim oms'
else:
    execution_type = 'slim'

save_results_to_file(
    dataset_name=dataset_name,
    training_rmse=best_fitness_individual.fitness,
    validation_rmse=best_fitness_individual.test_fitness,
    test_rmse=test_rmse_best_fitness,
    execution_type=execution_type
)

print(f"\n=== RESULTS SAVED ===")
print(f"Results have been saved to 'results_slim.csv' using best fitness individual")

# Show simplification information if auto_simplify was used
if hasattr(results, 'simplification_info') and results.simplification_info is not None:
    print(f"\n" + "="*80)
    print("AUTOMATIC SIMPLIFICATION RESULTS")
    print("="*80)
    
    # Always show the converted tree structure
    if 'converted_tree_structure' in results.simplification_info:
        print(f"🌳 Mathematical expression:")
        
        # Convert tree structure to mathematical notation
        def tree_to_math_expression(structure):
            import re
            
            def clean_node_name(node):
                """Clean node names by removing np.str_ wrappers and quotes."""
                if isinstance(node, tuple):
                    return node
                node_str = str(node)
                # Remove np.str_() wrapper
                if "np.str_(" in node_str:
                    node_str = re.sub(r"np\.str_\('([^']+)'\)", r"\1", node_str)
                # Remove quotes and clean constants
                node_str = node_str.strip("'\"")
                if node_str.startswith("constant_"):
                    node_str = node_str.replace("constant_", "")
                return node_str
            
            def convert_recursive(structure, parent_op=None):
                """Recursively convert tree structure to expression with proper parentheses."""
                if not isinstance(structure, tuple):
                    return clean_node_name(structure)
                
                if len(structure) < 3:
                    return clean_node_name(structure)
                
                operator = clean_node_name(structure[0])
                left = structure[1] 
                right = structure[2]
                
                # Convert operator names to symbols
                op_symbol = {
                    'add': '+',
                    'subtract': '-', 
                    'multiply': '*',
                    'divide': '/'
                }.get(operator, operator)
                
                # Recursively convert operands
                left_expr = convert_recursive(left, operator)
                right_expr = convert_recursive(right, operator)
                
                # Always add parentheses around composite expressions to preserve structure
                if isinstance(left, tuple) and len(left) >= 3:
                    left_expr = f"({left_expr})"
                
                if isinstance(right, tuple) and len(right) >= 3:
                    right_expr = f"({right_expr})"
                
                expression = f"{left_expr} {op_symbol} {right_expr}"
                
                # Add outer parentheses if this is a composite expression within another operation
                # This ensures mathematical equivalence with the original tree structure
                if parent_op is not None and isinstance(structure, tuple):
                    expression = f"({expression})"
                
                return expression
            
            return convert_recursive(structure)
        
        # Use simplified structure if available, otherwise use converted structure
        display_structure = results.simplification_info.get('simplified_structure', 
                                                           results.simplification_info['converted_tree_structure'])
        
        math_expression = tree_to_math_expression(display_structure)
        print(f"   {math_expression}")
        
        # Generate PNG visualization of the tree
        try:
            from slim_gsgp.utils.tree_to_png import save_tree_as_png_simple
            
            png_path = save_tree_as_png_simple(display_structure, "slim_tree_visualization.png")
            print(f"\n🖼️  Tree visualization saved as PNG: slim_tree_visualization.png")
            
        except ImportError as e:
            print(f"\n⚠️  Could not generate PNG (missing dependencies): {e}")
        except Exception as e:
            print(f"\n⚠️  Error generating PNG: {e}")
        
        print()  # Extra line for spacing
    
    if results.simplification_info['applied']:
        mathematical_simplifications = results.simplification_info.get('mathematical_simplifications', 0)
        nodes_removed = results.simplification_info['nodes_removed']
        
        if nodes_removed > 0:
            print(f"🎉 Structural simplification successful!")
            print(f"   📊 Original model: {results.simplification_info['original_nodes']} nodes, depth {results.simplification_info['original_depth']}")
            print(f"   ✅ Simplified to: {results.simplification_info['simplified_nodes']} nodes, depth {results.simplification_info['simplified_depth']}")
            print(f"   📈 Reduction: {nodes_removed} nodes removed ({nodes_removed/results.simplification_info['original_nodes']*100:.1f}%)")
            print(f"   🎯 Result: Model is {nodes_removed} nodes smaller with same performance!")
        elif mathematical_simplifications > 0:
            print(f"🎉 Mathematical simplification successful!")
            print(f"   📊 Model structure: {results.simplification_info['simplified_nodes']} nodes, depth {results.simplification_info['simplified_depth']}")
            print(f"   🔧 Applied {mathematical_simplifications} mathematical simplification(s)")
            print(f"   ✨ Combined duplicate terms (x+x→2x, etc.)")
            print(f"   🎯 Result: Mathematically equivalent but cleaner expression!")
        
        # Show simplified structure if different
        if 'simplified_structure' in results.simplification_info:
            print(f"\n🔧 Simplified mathematical expression:")
            simplified_math = tree_to_math_expression(results.simplification_info['simplified_structure'])
            print(f"   {simplified_math}")
    else:
        print(f"ℹ️  {results.simplification_info['reason']}")
        print(f"   📊 Model remains: {results.simplification_info['original_nodes']} nodes, depth {results.simplification_info['original_depth']}")
        print(f"   💡 The model was already optimally simplified.")
    
    print(f"🏁 Simplification process completed")
