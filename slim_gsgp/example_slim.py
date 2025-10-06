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
from slim_gsgp.datasets.data_loader import load_airfoil  # import the loader for the airfoil dataset
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
    # Ensure filename includes log directory if not already present
    if not filename.startswith("log"):
        filename = os.path.join("log", filename)
    
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

# Apply the SLIM GSGP algorithm (with fixed seed for reproducibility)
results = slim(X_train=X_train, y_train=y_train,
               X_test=X_val, y_test=y_val,
               dataset_name='airfoil', slim_version='SLIM+ABS', pop_size=100, n_iter=100,
               ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True, 
               tournament_type="pareto", tournament_size=5, multi_obj_attrs=["fitness", "size"], 
               oms=True, linear_scaling=True, enable_plotting=False, seed=42)

# Extract all three individuals
best_fitness_individual = results.best_fitness
best_normalized_individual = results.best_normalized
smallest_individual = results.smallest

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
print(f"Training fitness (RMSE): {best_fitness_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {best_fitness_individual.test_fitness:.6f}")

# Get the prediction of the best fitness individual on the test set
predictions_best_fitness = best_fitness_individual.predict(X_test)
test_rmse_best_fitness = float(rmse(y_true=y_test, y_pred=predictions_best_fitness))
print(f"Final test fitness (RMSE): {test_rmse_best_fitness:.6f}")
print()

print(f"Number of nodes: {best_fitness_individual.nodes_count}")
print(f"Tree depth: {best_fitness_individual.depth}")

# Convert SLIM GSGP to readable tree and visualize (Best Fitness)
try:
    from slim_gsgp.utils.simplification import convert_slim_individual_to_normal_tree, simplify_constant_operations
    import re
    
    # Convert the best fitness individual to a readable tree
    tree_structure_fitness, tree_dicts_fitness = convert_slim_individual_to_normal_tree(best_fitness_individual)
    
    if tree_structure_fitness and tree_dicts_fitness:
        
        # Apply constant simplification
        simplified_tree_fitness, nodes_removed_fitness = simplify_constant_operations(
            tree_structure_fitness, tree_dicts_fitness['CONSTANTS']
        )
        
        # Convert tree structure to mathematical notation
        def tree_to_math_expression(structure):
            def clean_node_name(node):
                if isinstance(node, tuple):
                    if len(node) == 2 and str(node[0]).upper() == 'NEGATIVE':
                        return f"-{clean_node_name(node[1])}"
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
                
                if node_str.startswith("_") and node_str[1:].replace(".", "").replace("-", "").isdigit():
                    return node_str[1:]
                
                return node_str
            
            def convert_recursive(structure, parent_op=None, is_right=False):
                if not isinstance(structure, tuple):
                    return clean_node_name(structure)
                
                if len(structure) < 3:
                    return clean_node_name(structure)
                
                operator = clean_node_name(structure[0])
                left = structure[1]
                right = structure[2]
                
                op_symbol = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}.get(operator, operator)
                precedence = {'add': 1, 'subtract': 1, 'multiply': 2, 'divide': 2}
                current_prec = precedence.get(operator, 0)
                
                # Recursively convert children
                left_expr = convert_recursive(left, operator, is_right=False)
                right_expr = convert_recursive(right, operator, is_right=True)
                
                # Add parentheses to left child if it has lower precedence
                if isinstance(left, tuple) and len(left) >= 3:
                    left_op = clean_node_name(left[0])
                    left_prec = precedence.get(left_op, 0)
                    if left_prec < current_prec:
                        left_expr = f"({left_expr})"
                
                # Add parentheses to right child if needed
                if isinstance(right, tuple) and len(right) >= 3:
                    right_op = clean_node_name(right[0])
                    right_prec = precedence.get(right_op, 0)
                    
                    if right_prec < current_prec:
                        right_expr = f"({right_expr})"
                    elif right_prec == current_prec:
                        if operator in ['subtract', 'divide']:
                            right_expr = f"({right_expr})"
                        elif operator == 'multiply' and right_op in ['multiply', 'divide']:
                            right_expr = f"({right_expr})"
                
                expression = f"{left_expr} {op_symbol} {right_expr}"
                
                return expression
            
            return convert_recursive(structure)
        
        # Define node counting function
        def count_nodes(struct):
            if not isinstance(struct, tuple):
                return 1
            count = 1
            for i in range(1, len(struct)):
                count += count_nodes(struct[i])
            return count
        
        # Count nodes and show results
        original_node_count_fitness = count_nodes(tree_structure_fitness)
        simplified_node_count_fitness = count_nodes(simplified_tree_fitness)
        
        if nodes_removed_fitness > 0:
            # Show before and after
            print(f"\n🌳 Original expression ({original_node_count_fitness} nodes):")
            print(f"   {tree_to_math_expression(tree_structure_fitness)}")
            
            print(f"\n✨ Simplified expression ({simplified_node_count_fitness} nodes):")
            print(f"   {tree_to_math_expression(simplified_tree_fitness)}")
            
            print(f"\n📉 Simplified {nodes_removed_fitness} constant operation(s)")
        else:
            # No simplification
            print(f"\n🌳 Mathematical expression ({original_node_count_fitness} nodes):")
            print(f"   {tree_to_math_expression(tree_structure_fitness)}")
            print(f"\n✓ No constant operations to simplify")
        
        # Generate PNG visualization (use simplified tree)
        try:
            from slim_gsgp.utils.tree_to_png import save_tree_as_png_simple
            
            png_path_fitness = save_tree_as_png_simple(simplified_tree_fitness, os.path.join("log", "slim_tree_visualization_best_fitness.png"))
            
        except ImportError as e:
            print(f"\n⚠️  Could not generate PNG (missing dependencies): {e}")
            print(f"    Install required packages: pip install graphviz")
        except Exception as e:
            print(f"\n⚠️  Error generating PNG: {e}")
    else:
        print("❌ Could not convert SLIM GSGP to tree structure")
        
except Exception as e:
    print(f"❌ Error during tree conversion: {e}")
    import traceback
    traceback.print_exc()

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
print(f"Training fitness (RMSE): {best_normalized_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {best_normalized_individual.test_fitness:.6f}")

# Get the prediction of the best normalized individual on the test set
predictions_best_normalized = best_normalized_individual.predict(X_test)
test_rmse_best_normalized = float(rmse(y_true=y_test, y_pred=predictions_best_normalized))
print(f"Final test fitness (RMSE): {test_rmse_best_normalized:.6f}")
print()

print(f"Number of nodes: {best_normalized_individual.nodes_count}")
print(f"Tree depth: {best_normalized_individual.depth}")

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
# Convert SLIM GSGP to readable tree and visualize
try:
    from slim_gsgp.utils.simplification import convert_slim_individual_to_normal_tree
    import re
    
    # Convert the best normalized individual to a readable tree
    tree_structure, tree_dicts = convert_slim_individual_to_normal_tree(best_normalized_individual)
    
    if tree_structure and tree_dicts:
        
        # Apply constant simplification
        simplified_tree, nodes_removed = simplify_constant_operations(
            tree_structure, tree_dicts['CONSTANTS']
        )
        
        # Convert tree structure to mathematical notation
        def tree_to_math_expression(structure):
            def clean_node_name(node):
                if isinstance(node, tuple):
                    if len(node) == 2 and str(node[0]).upper() == 'NEGATIVE':
                        return f"-{clean_node_name(node[1])}"
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
                
                if node_str.startswith("_") and node_str[1:].replace(".", "").replace("-", "").isdigit():
                    return node_str[1:]
                
                return node_str
            
            def convert_recursive(structure, parent_op=None, is_right=False):
                if not isinstance(structure, tuple):
                    return clean_node_name(structure)
                
                if len(structure) < 3:
                    return clean_node_name(structure)
                
                operator = clean_node_name(structure[0])
                left = structure[1]
                right = structure[2]
                
                op_symbol = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}.get(operator, operator)
                precedence = {'add': 1, 'subtract': 1, 'multiply': 2, 'divide': 2}
                current_prec = precedence.get(operator, 0)
                
                # Recursively convert children
                left_expr = convert_recursive(left, operator, is_right=False)
                right_expr = convert_recursive(right, operator, is_right=True)
                
                # Add parentheses to left child if it has lower precedence
                if isinstance(left, tuple) and len(left) >= 3:
                    left_op = clean_node_name(left[0])
                    left_prec = precedence.get(left_op, 0)
                    if left_prec < current_prec:
                        left_expr = f"({left_expr})"
                
                # Add parentheses to right child if needed
                if isinstance(right, tuple) and len(right) >= 3:
                    right_op = clean_node_name(right[0])
                    right_prec = precedence.get(right_op, 0)
                    
                    # Need parens if:
                    # 1. Right has lower precedence than current
                    # 2. Same precedence but operator is non-associative (subtract/divide)
                    # 3. For multiply: if right is also multiply or divide (to preserve order)
                    if right_prec < current_prec:
                        right_expr = f"({right_expr})"
                    elif right_prec == current_prec:
                        # For subtract and divide: always need parens on right
                        # For multiply: need parens if right is multiply or divide
                        if operator in ['subtract', 'divide']:
                            right_expr = f"({right_expr})"
                        elif operator == 'multiply' and right_op in ['multiply', 'divide']:
                            right_expr = f"({right_expr})"
                
                # Build the expression (no outer parens added here)
                expression = f"{left_expr} {op_symbol} {right_expr}"
                
                return expression
            
            return convert_recursive(structure)
        
        # Define node counting function
        def count_nodes(struct):
            if not isinstance(struct, tuple):
                return 1
            count = 1
            for i in range(1, len(struct)):
                count += count_nodes(struct[i])
            return count
        
        # Count nodes and show results
        original_node_count = count_nodes(tree_structure)
        simplified_node_count = count_nodes(simplified_tree)
        
        if nodes_removed > 0:
            # Show before and after
            print(f"\n🌳 Original expression ({original_node_count} nodes):")
            print(f"   {tree_to_math_expression(tree_structure)}")
            
            print(f"\n✨ Simplified expression ({simplified_node_count} nodes):")
            print(f"   {tree_to_math_expression(simplified_tree)}")
            
            print(f"\n📉 Simplified {nodes_removed} constant operation(s)")
        else:
            # No simplification
            print(f"\n🌳 Mathematical expression ({original_node_count} nodes):")
            print(f"   {tree_to_math_expression(tree_structure)}")
            print(f"\n✓ No constant operations to simplify")
        
        # Generate PNG visualization (use simplified tree)
        try:
            from slim_gsgp.utils.tree_to_png import save_tree_as_png_simple
            
            png_path = save_tree_as_png_simple(simplified_tree, os.path.join("log", "slim_tree_visualization.png"))
            
        except ImportError as e:
            print(f"\n⚠️  Could not generate PNG (missing dependencies): {e}")
            print(f"    Install required packages: pip install graphviz")
        except Exception as e:
            print(f"\n⚠️  Error generating PNG: {e}")
    else:
        print("❌ Could not convert SLIM GSGP to tree structure")
        
except Exception as e:
    print(f"❌ Error during tree conversion: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("RESULTS FOR SMALLEST SIZE INDIVIDUAL")
print("="*80)

# Show the smallest individual structure
smallest_individual.print_tree_representation()

# Show linear scaling information if enabled
if hasattr(smallest_individual, 'print_scaling_info'):
    smallest_individual.print_scaling_info()

# Evaluate the smallest individual on validation data
smallest_individual.calculate_semantics(X_val, testing=True)
smallest_individual.evaluate(rmse, y_val, testing=True, operator="sum")

# Show fitness information for smallest individual
print(f"Training fitness (RMSE): {smallest_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {smallest_individual.test_fitness:.6f}")

# Get the prediction of the smallest individual on the test set
predictions_smallest = smallest_individual.predict(X_test)
test_rmse_smallest = float(rmse(y_true=y_test, y_pred=predictions_smallest))
print(f"Final test fitness (RMSE): {test_rmse_smallest:.6f}")
print()

print(f"Number of nodes: {smallest_individual.nodes_count}")
print(f"Tree depth: {smallest_individual.depth}")

# Convert SLIM GSGP to readable tree and visualize (Smallest)
try:
    from slim_gsgp.utils.simplification import convert_slim_individual_to_normal_tree, simplify_constant_operations
    import re
    
    # Convert the smallest individual to a readable tree
    tree_structure_smallest, tree_dicts_smallest = convert_slim_individual_to_normal_tree(smallest_individual)
    
    if tree_structure_smallest and tree_dicts_smallest:
        
        # Apply constant simplification
        simplified_tree_smallest, nodes_removed_smallest = simplify_constant_operations(
            tree_structure_smallest, tree_dicts_smallest['CONSTANTS']
        )
        
        # Convert tree structure to mathematical notation
        def tree_to_math_expression(structure):
            def clean_node_name(node):
                if isinstance(node, tuple):
                    if len(node) == 2 and str(node[0]).upper() == 'NEGATIVE':
                        return f"-{clean_node_name(node[1])}"
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
                
                if node_str.startswith("_") and node_str[1:].replace(".", "").replace("-", "").isdigit():
                    return node_str[1:]
                
                return node_str
            
            def convert_recursive(structure, parent_op=None, is_right=False):
                if not isinstance(structure, tuple):
                    return clean_node_name(structure)
                
                if len(structure) < 3:
                    return clean_node_name(structure)
                
                operator = clean_node_name(structure[0])
                left = structure[1]
                right = structure[2]
                
                op_symbol = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}.get(operator, operator)
                precedence = {'add': 1, 'subtract': 1, 'multiply': 2, 'divide': 2}
                current_prec = precedence.get(operator, 0)
                
                # Recursively convert children
                left_expr = convert_recursive(left, operator, is_right=False)
                right_expr = convert_recursive(right, operator, is_right=True)
                
                # Add parentheses to left child if it has lower precedence
                if isinstance(left, tuple) and len(left) >= 3:
                    left_op = clean_node_name(left[0])
                    left_prec = precedence.get(left_op, 0)
                    if left_prec < current_prec:
                        left_expr = f"({left_expr})"
                
                # Add parentheses to right child if needed
                if isinstance(right, tuple) and len(right) >= 3:
                    right_op = clean_node_name(right[0])
                    right_prec = precedence.get(right_op, 0)
                    
                    if right_prec < current_prec:
                        right_expr = f"({right_expr})"
                    elif right_prec == current_prec:
                        if operator in ['subtract', 'divide']:
                            right_expr = f"({right_expr})"
                        elif operator == 'multiply' and right_op in ['multiply', 'divide']:
                            right_expr = f"({right_expr})"
                
                expression = f"{left_expr} {op_symbol} {right_expr}"
                
                return expression
            
            return convert_recursive(structure)
        
        # Define node counting function
        def count_nodes(struct):
            if not isinstance(struct, tuple):
                return 1
            count = 1
            for i in range(1, len(struct)):
                count += count_nodes(struct[i])
            return count
        
        # Count nodes and show results
        original_node_count_smallest = count_nodes(tree_structure_smallest)
        simplified_node_count_smallest = count_nodes(simplified_tree_smallest)
        
        if nodes_removed_smallest > 0:
            # Show before and after
            print(f"\n🌳 Original expression ({original_node_count_smallest} nodes):")
            print(f"   {tree_to_math_expression(tree_structure_smallest)}")
            
            print(f"\n✨ Simplified expression ({simplified_node_count_smallest} nodes):")
            print(f"   {tree_to_math_expression(simplified_tree_smallest)}")
            
            print(f"\n📉 Simplified {nodes_removed_smallest} constant operation(s)")
        else:
            # No simplification
            print(f"\n🌳 Mathematical expression ({original_node_count_smallest} nodes):")
            print(f"   {tree_to_math_expression(tree_structure_smallest)}")
            print(f"\n✓ No constant operations to simplify")
        
        # Generate PNG visualization (use simplified tree)
        try:
            from slim_gsgp.utils.tree_to_png import save_tree_as_png_simple
            
            png_path_smallest = save_tree_as_png_simple(simplified_tree_smallest, os.path.join("log", "slim_tree_visualization_smallest.png"))
            
        except ImportError as e:
            print(f"\n⚠️  Could not generate PNG (missing dependencies): {e}")
            print(f"    Install required packages: pip install graphviz")
        except Exception as e:
            print(f"\n⚠️  Error generating PNG: {e}")
    else:
        print("❌ Could not convert SLIM GSGP to tree structure")
        
except Exception as e:
    print(f"❌ Error during tree conversion: {e}")
    import traceback
    traceback.print_exc()

# Summary table with results from all three individuals
print("\n" + "="*80)
print("SUMMARY TABLE - ALL INDIVIDUALS")
print("="*80)
print()

# Create header
print(f"{'Metric':<30} {'Best Fitness':<20} {'Best Normalized':<20} {'Smallest Size':<20}")
print("-" * 90)

# Training RMSE
print(f"{'Training RMSE':<30} {best_fitness_individual.fitness:<20.6f} {best_normalized_individual.fitness:<20.6f} {smallest_individual.fitness:<20.6f}")

# Validation RMSE
print(f"{'Validation RMSE':<30} {best_fitness_individual.test_fitness:<20.6f} {best_normalized_individual.test_fitness:<20.6f} {smallest_individual.test_fitness:<20.6f}")

# Test RMSE
print(f"{'Test RMSE':<30} {test_rmse_best_fitness:<20.6f} {test_rmse_best_normalized:<20.6f} {test_rmse_smallest:<20.6f}")

print("-" * 90)

# Number of nodes
print(f"{'Number of nodes':<30} {best_fitness_individual.nodes_count:<20} {best_normalized_individual.nodes_count:<20} {smallest_individual.nodes_count:<20}")

# Tree depth
print(f"{'Tree depth':<30} {best_fitness_individual.depth:<20} {best_normalized_individual.depth:<20} {smallest_individual.depth:<20}")

print("\n" + "="*80)
