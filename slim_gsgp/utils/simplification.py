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
Tree conversion utilities for SLIM_GSGP individuals.
Converts SLIM GSGP representation to readable tree structure for visualization.
"""

import torch


def convert_slim_individual_to_normal_tree(individual, data_sample=None):
    """
    Convert a SLIM GSGP individual to a normal tree representation by 
    combining ALL trees in the collection (both base and composite).
    
    Parameters
    ----------
    individual : Individual
        SLIM GSGP individual to convert
    data_sample : torch.Tensor, optional
        Small sample of data to help determine structure (not currently used)
        
    Returns
    -------
    tuple or None
        Normal tree structure that can be read and visualized, or None if conversion fails
    dict or None
        Combined dictionaries for the tree (FUNCTIONS, TERMINALS, CONSTANTS), or None if conversion fails
    """
    from slim_gsgp.utils.utils import check_slim_version
    
    if not hasattr(individual, "collection") or not individual.collection:
        return None, None
        
    # Get the SLIM version information
    operator, sig, trees = check_slim_version(slim_version=individual.version)
    
    # Collect all trees and their structures
    all_tree_structures = []
    combined_functions = {}
    combined_terminals = {}
    combined_constants = {}
    
    for i, tree in enumerate(individual.collection):
        # Collect dictionaries from each tree
        if hasattr(tree, 'FUNCTIONS') and tree.FUNCTIONS:
            combined_functions.update(tree.FUNCTIONS)
        if hasattr(tree, 'TERMINALS') and tree.TERMINALS:
            combined_terminals.update(tree.TERMINALS)  
        if hasattr(tree, 'CONSTANTS') and tree.CONSTANTS:
            combined_constants.update(tree.CONSTANTS)
        
        if isinstance(tree.structure, tuple):
            # Base tree - use structure directly
            all_tree_structures.append(tree.structure)
        else:
            # Composite tree - extract the base tree from the mutation/crossover structure
            # For composite trees (lists), we need to extract the base trees
            if isinstance(tree.structure, list) and len(tree.structure) >= 2:
                # The structure is typically [operator, base_tree1, base_tree2, ...params]
                # We'll extract the base trees (Tree objects) from the structure
                base_trees_in_composite = []
                for element in tree.structure[1:]:  # Skip the operator (first element)
                    if hasattr(element, 'structure') and isinstance(element.structure, tuple):
                        base_trees_in_composite.append(element.structure)
                
                # If we found base trees, combine them
                if base_trees_in_composite:
                    if len(base_trees_in_composite) == 1:
                        all_tree_structures.append(base_trees_in_composite[0])
                    else:
                        # Create a composite structure representing the mutation
                        composite_structure = base_trees_in_composite[0]
                        for j in range(1, len(base_trees_in_composite)):
                            composite_structure = ('add', composite_structure, base_trees_in_composite[j])
                        all_tree_structures.append(composite_structure)
    
    # If no structures were extracted, return None
    if not all_tree_structures:
        return None, None
    
    # Add the combining operator to functions if needed
    if operator == torch.sum:
        combined_functions['add'] = {'arity': 2, 'function': torch.add}
        operator_name = 'add'
    elif operator == torch.prod:
        combined_functions['multiply'] = {'arity': 2, 'function': torch.mul}
        operator_name = 'multiply'
    else:
        # Default to addition
        combined_functions['add'] = {'arity': 2, 'function': torch.add}
        operator_name = 'add'
    
    # Combine all tree structures using the operator
    if len(all_tree_structures) == 1:
        result_structure = all_tree_structures[0]
    else:
        # Combine all structures with the operator
        result_structure = all_tree_structures[0]
        for i in range(1, len(all_tree_structures)):
            result_structure = (operator_name, result_structure, all_tree_structures[i])
    
    return result_structure, {
        'FUNCTIONS': combined_functions,
        'TERMINALS': combined_terminals,
        'CONSTANTS': combined_constants
    }


def simplify_constant_operations(tree_structure, constants_dict):
    """
    Simplify algebraic operations in the tree structure.
    
    Simplifications applied:
    1. Identity simplifications:
       - x - x → 0
       - x / x → 1
    2. Constant folding (all operations):
       - 5.0 + 3.0 → 8.0
       - 10.0 - 4.0 → 6.0
       - 2.0 * 3.0 → 6.0
       - 4.0 / 2.0 → 2.0
    3. Nested simplifications:
       - ('add', x, ('add', 5, 3)) → ('add', x, 8)
       - ('multiply', 2, ('add', 3, 2)) → ('multiply', 2, 5)
    
    Parameters
    ----------
    tree_structure : tuple
        Tree structure to simplify
    constants_dict : dict
        Dictionary mapping constant names to values
        
    Returns
    -------
    tuple
        Simplified tree structure
    int
        Number of nodes removed during simplification
    """
    if not isinstance(tree_structure, tuple) or len(tree_structure) < 3:
        return tree_structure, 0
    
    nodes_removed = 0
    
    def is_constant(node):
        """Check if a node is a constant value"""
        if isinstance(node, (int, float)):
            return True
        if isinstance(node, str):
            # Check if it's a constant name in the dictionary
            if node in constants_dict:
                return True
            # Check if it's a numeric string
            try:
                float(node)
                return True
            except (ValueError, TypeError):
                return False
        return False
    
    def get_constant_value(node):
        """Get the numeric value of a constant node"""
        if isinstance(node, (int, float)):
            return node
        if isinstance(node, str):
            if node in constants_dict:
                const_val = constants_dict[node]
                # Handle lambda functions that return torch tensors
                if callable(const_val):
                    try:
                        tensor_val = const_val(None)
                        return float(tensor_val.item())
                    except:
                        return None
                # Handle direct numeric values
                return float(const_val) if isinstance(const_val, (int, float)) else None
            try:
                return float(node)
            except (ValueError, TypeError):
                return None
        return None
    
    def simplify_recursive(structure):
        nonlocal nodes_removed
        
        if not isinstance(structure, tuple) or len(structure) < 3:
            return structure
        
        operator = structure[0]
        left = structure[1]
        right = structure[2]
        
        # First, recursively simplify children
        left_simplified = simplify_recursive(left)
        right_simplified = simplify_recursive(right)
        
        # Simplification 1: x - x → 0
        if operator == 'subtract' and left_simplified == right_simplified:
            nodes_removed += 2  # Removed operator node and one operand
            return '0.0'
        
        # Simplification 2: x / x → 1
        if operator == 'divide' and left_simplified == right_simplified:
            nodes_removed += 2  # Removed operator node and one operand
            return '1.0'
        
        # Constant folding for all operations with two constants
        if operator in ['add', 'subtract', 'multiply', 'divide']:
            if is_constant(left_simplified) and is_constant(right_simplified):
                left_val = get_constant_value(left_simplified)
                right_val = get_constant_value(right_simplified)
                
                if left_val is not None and right_val is not None:
                    if operator == 'add':
                        result = left_val + right_val
                    elif operator == 'subtract':
                        result = left_val - right_val
                    elif operator == 'multiply':
                        result = left_val * right_val
                    elif operator == 'divide':
                        # Avoid division by zero
                        if right_val != 0:
                            result = left_val / right_val
                        else:
                            # Cannot simplify division by zero, return as is
                            return (operator, left_simplified, right_simplified)
                    
                    # Convert result to string for consistency
                    nodes_removed += 2  # Removed operator node and one constant
                    return str(result)
        
        # If no simplification occurred, return the structure with simplified children
        return (operator, left_simplified, right_simplified)
    
    simplified_tree = simplify_recursive(tree_structure)
    return simplified_tree, nodes_removed
