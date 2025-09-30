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
Automatic simplification utilities for SLIM_GSGP individuals.
"""

import torch
import copy
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.GP.representations.tree_utils import tree_depth, flatten


def convert_slim_individual_to_normal_tree(individual, data_sample=None):
    """
    Convert a SLIM GSGP individual to a normal tree representation by 
    combining ALL trees in the collection (both base and composite).
    
    Parameters
    ----------
    individual : Individual
        SLIM GSGP individual to convert
    data_sample : torch.Tensor, optional
        Small sample of data to help determine structure
        
    Returns
    -------
    tuple
        Normal tree structure that can be simplified
    dict
        Combined dictionaries for the tree
    """
    from slim_gsgp.algorithms.GSGP.representations.tree_utils import apply_tree
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
    
    print(f"   🔍 Debug: Converting SLIM individual with {len(individual.collection)} trees")
    print(f"   🔍 Debug: SLIM version: {individual.version}, operator: {operator}")
    
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
            print(f"   🔍 Debug: Tree {i} (base): {tree.structure}")
            all_tree_structures.append(tree.structure)
        else:
            # Composite tree - extract the base tree from the mutation/crossover structure
            print(f"   🔍 Debug: Tree {i} (composite): {type(tree.structure)} with length {len(tree.structure) if hasattr(tree.structure, '__len__') else 'N/A'}")
            
            # For composite trees (lists), we need to extract the base trees
            if isinstance(tree.structure, list) and len(tree.structure) >= 2:
                # The structure is typically [operator, base_tree1, base_tree2, ...params]
                # We'll extract the base trees (Tree objects) from the structure
                base_trees_in_composite = []
                for element in tree.structure[1:]:  # Skip the operator (first element)
                    if hasattr(element, 'structure') and isinstance(element.structure, tuple):
                        base_trees_in_composite.append(element.structure)
                        print(f"   🔍 Debug: Found base tree in composite: {element.structure}")
                
                # If we found base trees, combine them
                if base_trees_in_composite:
                    if len(base_trees_in_composite) == 1:
                        all_tree_structures.append(base_trees_in_composite[0])
                    else:
                        # Create a composite structure representing the mutation
                        # This is a simplified representation
                        composite_structure = base_trees_in_composite[0]
                        for j in range(1, len(base_trees_in_composite)):
                            composite_structure = ('add', composite_structure, base_trees_in_composite[j])
                        all_tree_structures.append(composite_structure)
                        print(f"   🔍 Debug: Created composite structure from {len(base_trees_in_composite)} base trees")
                else:
                    print(f"   🔍 Debug: No extractable base trees found in composite tree {i}")
    
    print(f"   🔍 Debug: Extracted {len(all_tree_structures)} tree structures")
    
    # If no structures were extracted, return None
    if not all_tree_structures:
        print(f"   🔍 Debug: No tree structures could be extracted")
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
        print(f"   🔍 Debug: Single tree structure: {result_structure}")
    else:
        # Combine all structures with the operator
        result_structure = all_tree_structures[0]
        for i in range(1, len(all_tree_structures)):
            result_structure = (operator_name, result_structure, all_tree_structures[i])
        print(f"   🔍 Debug: Combined {len(all_tree_structures)} trees with operator '{operator_name}'")
    
    return result_structure, {
        'FUNCTIONS': combined_functions,
        'TERMINALS': combined_terminals,
        'CONSTANTS': combined_constants
    }


def simplify_tree_structure(tree_structure, FUNCTIONS, TERMINALS, CONSTANTS):
    """
    Simplify a tree structure by removing redundant nodes and operations.
    
    Parameters
    ----------
    tree_structure : tuple or str
        The tree structure to simplify.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    TERMINALS : dict
        Dictionary of terminals allowed in the tree.
    CONSTANTS : dict
        Dictionary of constants allowed in the tree.
        
    Returns
    -------
    tuple or str
        The simplified tree structure.
    """
    if not isinstance(tree_structure, tuple):
        # Terminal node, return as is
        return tree_structure
    
    function_name = tree_structure[0]
    
    # Handle numpy string types (convert to regular string)
    if hasattr(function_name, 'item'):
        function_name = str(function_name.item())
    else:
        function_name = str(function_name)
    
    # Check if function exists in FUNCTIONS dictionary
    if function_name not in FUNCTIONS:
        return tree_structure
        
    arity = FUNCTIONS[function_name]["arity"]
    
    if arity == 1:
        # Unary function
        child = simplify_tree_structure(tree_structure[1], FUNCTIONS, TERMINALS, CONSTANTS)
        
        # Simplification rules for unary functions
        if function_name == "sin" and isinstance(child, str) and child in CONSTANTS:
            try:
                if CONSTANTS[child](None) == 0:
                    # sin(0) = 0
                    return child
            except:
                pass
        elif function_name == "cos" and isinstance(child, str) and child in CONSTANTS:
            try:
                if CONSTANTS[child](None) == 0:
                    # cos(0) = 1 - replace with constant 1 if available
                    for const_name, const_func in CONSTANTS.items():
                        if const_func(None) == 1:
                            return const_name
            except:
                pass
        
        return (tree_structure[0], child)
    
    elif arity == 2:
        # Binary function
        left = simplify_tree_structure(tree_structure[1], FUNCTIONS, TERMINALS, CONSTANTS)
        right = simplify_tree_structure(tree_structure[2], FUNCTIONS, TERMINALS, CONSTANTS)
        
        # Simplification rules for binary functions
        if function_name == "add":
            # x + 0 = x
            if isinstance(right, str) and right in CONSTANTS:
                try:
                    if CONSTANTS[right](None) == 0:
                        return left
                except:
                    pass
            elif isinstance(left, str) and left in CONSTANTS:
                try:
                    if CONSTANTS[left](None) == 0:
                        return right
                except:
                    pass
        
        elif function_name == "subtract":
            # x - 0 = x
            if isinstance(right, str) and right in CONSTANTS:
                try:
                    if CONSTANTS[right](None) == 0:
                        return left
                except:
                    pass
            # x - x = 0 (if both are the same terminal)
            elif left == right and isinstance(left, str):
                for const_name, const_func in CONSTANTS.items():
                    try:
                        if const_func(None) == 0:
                            return const_name
                    except:
                        continue
        
        elif function_name == "multiply":
            # x * 0 = 0
            if isinstance(right, str) and right in CONSTANTS:
                try:
                    if CONSTANTS[right](None) == 0:
                        return right
                except:
                    pass
            elif isinstance(left, str) and left in CONSTANTS:
                try:
                    if CONSTANTS[left](None) == 0:
                        return left
                except:
                    pass
            # x * 1 = x
            elif isinstance(right, str) and right in CONSTANTS:
                try:
                    if CONSTANTS[right](None) == 1:
                        return left
                except:
                    pass
            elif isinstance(left, str) and left in CONSTANTS:
                try:
                    if CONSTANTS[left](None) == 1:
                        return right
                except:
                    pass
        
        elif function_name == "divide":
            # x / 1 = x
            if isinstance(right, str) and right in CONSTANTS:
                try:
                    if CONSTANTS[right](None) == 1:
                        return left
                except:
                    pass
            # 0 / x = 0 (assuming x != 0)
            elif isinstance(left, str) and left in CONSTANTS:
                try:
                    if CONSTANTS[left](None) == 0:
                        return left
                except:
                    pass
        
        return (tree_structure[0], left, right)
    
    # Return original structure if no simplification applied
    return tree_structure


def simplify_individual(individual, max_simplification_iterations=3, debug=False):
    """
    Apply automatic simplification to a SLIM_GSGP individual by converting it to 
    a normal tree and then applying simplification.
    
    This function:
    1. Converts the SLIM GSGP individual to a normal tree representation
    2. Applies algebraic simplification rules to the converted tree
    3. Creates a new Individual with the simplified structure
    
    Parameters
    ----------
    individual : Individual
        The SLIM GSGP individual to simplify.
    max_simplification_iterations : int, optional
        Maximum number of simplification iterations to apply. Default is 3.
    debug : bool, optional
        Whether to print debug information. Default is False.
        
    Returns
    -------
    Individual
        A new simplified individual, or the original if no simplification was possible.
    """
    if not hasattr(individual, 'collection') or individual.collection is None:
        if debug:
            print(f"   🔍 Debug: Individual has no collection attribute or collection is None")
        return individual
    
    if debug:
        print(f"   🔍 Debug: Individual has {len(individual.collection)} trees in collection")
        print(f"   🔍 Debug: Individual version: {individual.version}")
    
    # Convert SLIM individual to normal tree
    converted_tree_structure, tree_dicts = convert_slim_individual_to_normal_tree(individual)
    
    if converted_tree_structure is None or tree_dicts is None:
        if debug:
            print(f"   🔍 Debug: Could not convert SLIM individual to normal tree")
        return individual
    
    if debug:
        print(f"   🔍 Debug: Successfully converted to normal tree structure")
        print(f"   🔍 Debug: Available functions: {list(tree_dicts['FUNCTIONS'].keys())}")
        print(f"   🔍 Debug: Converted tree structure: {converted_tree_structure}")
        
        # Pretty print the tree structure for better readability
        def pretty_print_tree(structure, indent=0):
            spaces = "  " * indent
            if isinstance(structure, tuple) and len(structure) >= 2:
                print(f"{spaces}({structure[0]}")
                for i in range(1, len(structure)):
                    pretty_print_tree(structure[i], indent + 1)
                print(f"{spaces})")
            else:
                print(f"{spaces}{structure}")
        
        print(f"   🔍 Debug: Tree structure (formatted):")
        pretty_print_tree(converted_tree_structure)
    
    # Apply simplification to the converted tree
    simplified_structure = converted_tree_structure
    total_simplifications = 0
    
    for iteration in range(max_simplification_iterations):
        if debug:
            print(f"   🔍 Debug: Simplification iteration {iteration+1}")
            
        if debug:
            print(f"   🔍 Debug: Before simplification: {simplified_structure}")
            
        new_structure = simplify_tree_structure(
            simplified_structure, 
            tree_dicts['FUNCTIONS'], 
            tree_dicts['TERMINALS'], 
            tree_dicts['CONSTANTS']
        )
        
        if debug:
            print(f"   🔍 Debug: After simplification attempt: {new_structure}")
        
        if new_structure == simplified_structure:
            # No more simplifications possible
            if debug:
                print(f"   🔍 Debug: No more simplifications possible (structures are identical)")
            break
        else:
            total_simplifications += 1
            if debug:
                print(f"   🔍 Debug: Simplification {total_simplifications} applied!")
                print(f"   🔍 Debug: Changed from: {simplified_structure}")
                print(f"   🔍 Debug: Changed to:   {new_structure}")
            
        simplified_structure = new_structure
    
    if debug:
        print(f"   🔍 Debug: Total simplifications applied: {total_simplifications}")
    
    # Create a copy of the individual to add conversion information
    result_individual = copy.deepcopy(individual)
    
    # Add simplification information (always, even if no simplifications were made)
    if not hasattr(result_individual, '_simplification_info'):
        result_individual._simplification_info = {}
    
    result_individual._simplification_info.update({
        'original_structure': str(converted_tree_structure),
        'simplified_structure': str(simplified_structure),
        'simplifications_applied': total_simplifications,
        'conversion_successful': True,
        'converted_tree_structure': converted_tree_structure,
        'simplified_tree_structure': simplified_structure
    })
    
    # If no simplifications were made, still return the individual with conversion info
    if total_simplifications == 0:
        if debug:
            print(f"   🔍 Debug: No simplifications applied, returning original individual with conversion info")
        return result_individual
    
    if debug:
        print(f"   🔍 Debug: Created individual with {total_simplifications} simplifications")
    
    return result_individual


def add_simplification_method_to_individual():
    """
    Add a simplify method to the Individual class.
    
    This function dynamically adds a simplify method to the Individual class
    that calls simplify_individual on the instance.
    """
    from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual
    
    def simplify(self, max_iterations=3):
        """
        Simplify this individual.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of simplification iterations. Default is 3.
            
        Returns
        -------
        Individual
            A new simplified individual.
        """
        return simplify_individual(self, max_iterations)
    
    # Add the method to the Individual class
    Individual.simplify = simplify