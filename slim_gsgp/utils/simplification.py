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


def find_and_simplify_patterns(tree_structure, FUNCTIONS, TERMINALS, CONSTANTS):
    """
    Find and simplify complex patterns that ACTUALLY reduce complexity.
    
    Focus on simplifications that reduce the number of operations:
    - (a + b) - b = a  (reduces 2 ops to 0)
    - (a - b) + b = a  (reduces 2 ops to 0) 
    - (a * b) / b = a  (reduces 2 ops to 0)
    - (a + 0) = a      (reduces 1 op to 0)
    - (a * 1) = a      (reduces 1 op to 0)
    """
    if not isinstance(tree_structure, tuple) or len(tree_structure) < 3:
        return tree_structure
    
    function_name = str(tree_structure[0])
    left = tree_structure[1]
    right = tree_structure[2]
    
    # Normalize for comparison
    left_norm = normalize_term_for_comparison(left)
    right_norm = normalize_term_for_comparison(right)
    
    # Pattern: (a + b) - b = a (REAL complexity reduction: 2 operations → 0)
    if function_name == "subtract" and isinstance(left, tuple) and len(left) >= 3:
        left_op = str(left[0])
        if left_op == "add":
            left_right_norm = normalize_term_for_comparison(left[2])
            left_left_norm = normalize_term_for_comparison(left[1])
            
            # (a + b) - b = a
            if left_right_norm == right_norm:
                print(f"   🔍 Debug: Pattern match (a+b)-b=a: {left[1]}")
                return left[1]  # return a
            # (a + b) - a = b  
            elif left_left_norm == right_norm:
                print(f"   🔍 Debug: Pattern match (a+b)-a=b: {left[2]}")
                return left[2]  # return b
    
    # Pattern: (a - b) + b = a (REAL complexity reduction: 2 operations → 0)
    elif function_name == "add" and isinstance(left, tuple) and len(left) >= 3:
        left_op = str(left[0])
        if left_op == "subtract":
            left_right_norm = normalize_term_for_comparison(left[2])
            
            # (a - b) + b = a
            if left_right_norm == right_norm:
                print(f"   🔍 Debug: Pattern match (a-b)+b=a: {left[1]}")
                return left[1]  # return a
    
    # Pattern: (a * b) / b = a (REAL complexity reduction: 2 operations → 0)
    elif function_name == "divide" and isinstance(left, tuple) and len(left) >= 3:
        left_op = str(left[0])
        if left_op == "multiply":
            left_right_norm = normalize_term_for_comparison(left[2])
            left_left_norm = normalize_term_for_comparison(left[1])
            
            # (a * b) / b = a
            if left_right_norm == right_norm:
                print(f"   🔍 Debug: Pattern match (a*b)/b=a: {left[1]}")
                return left[1]  # return a
            # (a * b) / a = b
            elif left_left_norm == right_norm:
                print(f"   🔍 Debug: Pattern match (a*b)/a=b: {left[2]}")
                return left[2]  # return b
    
    # Pattern: (a / b) * b = a (REAL complexity reduction: 2 operations → 0)
    elif function_name == "multiply" and isinstance(left, tuple) and len(left) >= 3:
        left_op = str(left[0])
        if left_op == "divide":
            left_right_norm = normalize_term_for_comparison(left[2])
            
            # (a / b) * b = a
            if left_right_norm == right_norm:
                print(f"   🔍 Debug: Pattern match (a/b)*b=a: {left[1]}")
                return left[1]  # return a
    
    return tree_structure


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
            
            # x + x = 2*x (same terms)
            if left == right and isinstance(left, str):
                # Create multiplication by 2
                for const_name, const_func in CONSTANTS.items():
                    try:
                        if const_func(None) == 2:
                            return ("multiply", const_name, left)
                    except:
                        continue
            
            # Constant folding: c1 + c2 = c3
            if (isinstance(left, str) and left in CONSTANTS and 
                isinstance(right, str) and right in CONSTANTS):
                try:
                    left_val = CONSTANTS[left](None)
                    right_val = CONSTANTS[right](None)
                    result_val = left_val + right_val
                    
                    # Look for existing constant with this value
                    for const_name, const_func in CONSTANTS.items():
                        try:
                            if const_func(None) == result_val:
                                return const_name
                        except:
                            continue
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
            
            # Constant folding: c1 - c2 = c3
            if (isinstance(left, str) and left in CONSTANTS and 
                isinstance(right, str) and right in CONSTANTS):
                try:
                    left_val = CONSTANTS[left](None)
                    right_val = CONSTANTS[right](None)
                    result_val = left_val - right_val
                    
                    # Look for existing constant with this value
                    for const_name, const_func in CONSTANTS.items():
                        try:
                            if const_func(None) == result_val:
                                return const_name
                        except:
                            continue
                except:
                    pass
        
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
            
            # x * x = x^2 (same terms)
            if left == right and isinstance(left, str):
                # Create power of 2 if power function exists
                if "power" in FUNCTIONS:
                    for const_name, const_func in CONSTANTS.items():
                        try:
                            if const_func(None) == 2:
                                return ("power", left, const_name)
                        except:
                            continue
            
            # Constant folding: c1 * c2 = c3
            if (isinstance(left, str) and left in CONSTANTS and 
                isinstance(right, str) and right in CONSTANTS):
                try:
                    left_val = CONSTANTS[left](None)
                    right_val = CONSTANTS[right](None)
                    result_val = left_val * right_val
                    
                    # Look for existing constant with this value
                    for const_name, const_func in CONSTANTS.items():
                        try:
                            if const_func(None) == result_val:
                                return const_name
                        except:
                            continue
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
            
            # x / x = 1 (same terms, assuming x != 0)
            elif left == right and isinstance(left, str):
                for const_name, const_func in CONSTANTS.items():
                    try:
                        if const_func(None) == 1:
                            return const_name
                    except:
                        continue
            
            # Constant folding: c1 / c2 = c3
            if (isinstance(left, str) and left in CONSTANTS and 
                isinstance(right, str) and right in CONSTANTS):
                try:
                    left_val = CONSTANTS[left](None)
                    right_val = CONSTANTS[right](None)
                    if right_val != 0:  # Avoid division by zero
                        result_val = left_val / right_val
                        
                        # Look for existing constant with this value
                        for const_name, const_func in CONSTANTS.items():
                            try:
                                if abs(const_func(None) - result_val) < 1e-10:  # Floating point comparison
                                    return const_name
                            except:
                                continue
                except:
                    pass
        
        # After basic simplifications, check for advanced patterns
        basic_result = (tree_structure[0], left, right)
        advanced_result = find_and_simplify_patterns(basic_result, FUNCTIONS, TERMINALS, CONSTANTS)
        
        if advanced_result != basic_result:
            return advanced_result
        
        return basic_result
    
    # Return original structure if no simplification applied
    return tree_structure


def collect_terms_in_sum(tree_structure):
    """
    Collect all terms in a sum expression and separate positive/negative terms.
    
    For example: ((a + b) - c) becomes positive: [a, b], negative: [c]
    This allows us to find and cancel terms more easily.
    """
    if not isinstance(tree_structure, tuple):
        return [tree_structure]
    
    if len(tree_structure) >= 3:
        op = str(tree_structure[0])
        if op == "add":
            left_terms = collect_terms_in_sum(tree_structure[1])
            right_terms = collect_terms_in_sum(tree_structure[2])
            return left_terms + right_terms
        elif op == "subtract":
            # For subtraction, collect left terms as positive, right as negative
            left_terms = collect_terms_in_sum(tree_structure[1])
            # Mark right terms as negative by wrapping in a special marker
            right_terms = collect_terms_in_sum(tree_structure[2])
            # Convert right terms to negative terms
            negative_terms = []
            for term in right_terms:
                negative_terms.append(('NEGATIVE', term))
            return left_terms + negative_terms
    
    return [tree_structure]


def normalize_term_for_comparison(term):
    """
    Normalize a term for comparison by removing np.str_ wrappers and converting to standard format.
    """
    if isinstance(term, tuple):
        # Recursively normalize tuple elements
        return tuple(normalize_term_for_comparison(elem) for elem in term)
    elif hasattr(term, 'item'):
        # Handle numpy string types
        return str(term.item())
    else:
        return str(term)

def detect_and_cancel_terms(terms):
    """
    Detect and cancel terms that appear both positive and negative.
    
    For example: [a, b, NEGATIVE(b), c] → [a, c] (b cancels out)
    This provides REAL complexity reduction.
    """
    positive_terms = []
    negative_terms = []
    
    # Separate positive and negative terms
    for term in terms:
        if isinstance(term, tuple) and len(term) == 2 and term[0] == 'NEGATIVE':
            negative_terms.append(term[1])  # Remove the NEGATIVE marker
        else:
            positive_terms.append(term)
    
    # Find terms that appear in both lists (for cancellation)
    canceled_terms = []
    remaining_positive = []
    remaining_negative = []
    
    # Normalize terms for comparison
    positive_normalized = [(normalize_term_for_comparison(t), t) for t in positive_terms]
    negative_normalized = [(normalize_term_for_comparison(t), t) for t in negative_terms]
    
    # Check for cancellations
    for pos_norm, pos_term in positive_normalized:
        canceled = False
        for i, (neg_norm, neg_term) in enumerate(negative_normalized):
            if pos_norm == neg_norm:
                # Found a cancellation!
                print(f"   🔍 Debug: Cancellation found: +{pos_term} and -{neg_term}")
                canceled_terms.append(pos_term)
                negative_normalized.pop(i)  # Remove the canceled negative term
                canceled = True
                break
        
        if not canceled:
            remaining_positive.append(pos_term)
    
    # Add remaining negative terms back (as subtract operations)
    remaining_negative = [term for _, term in negative_normalized]
    
    return remaining_positive, remaining_negative, len(canceled_terms)

def simplify_collected_terms(terms, FUNCTIONS, TERMINALS, CONSTANTS):
    """
    Simplify a list of terms by finding real complexity reductions:
    1. Cancel terms that appear both positive and negative
    2. Combine constants
    3. Look for other patterns that reduce operations
    """
    if not terms:
        return terms
    
    # First, detect and cancel terms
    positive_terms, negative_terms, cancellations_made = detect_and_cancel_terms(terms)
    
    if cancellations_made > 0:
        print(f"   🔍 Debug: Made {cancellations_made} term cancellation(s)")
    
    # Separate constants from other terms
    constants = []
    other_terms = []
    
    for term in positive_terms:
        # Check if it's a constant
        term_str = str(term)
        normalized_term = normalize_term_for_comparison(term)
        
        if isinstance(term, str) and term in CONSTANTS:
            constants.append(term)
        elif (isinstance(normalized_term, str) and 
              (normalized_term in CONSTANTS or normalized_term.startswith('constant_'))):
            constants.append(term)
        else:
            other_terms.append(term)
    
    # Add remaining negative terms as subtraction operations
    for neg_term in negative_terms:
        if "subtract" in FUNCTIONS:
            # We'll handle this when rebuilding the expression
            other_terms.append(('NEGATIVE', neg_term))
    
    # Combine constants - improved logic
    if constants:
        try:
            total = 0
            for const in constants:
                const_str = str(const)
                if const_str.startswith('constant_'):
                    # Extract numeric value from constant_X.X format
                    value_str = const_str.replace('constant_', '')
                    total += float(value_str)
                elif const in CONSTANTS:
                    total += CONSTANTS[const](None)
                else:
                    # Try to parse as number directly
                    total += float(const_str)
            
            # Only add the combined constant if we actually combined multiple constants
            if len(constants) > 1:
                print(f"   🔍 Debug: Combined {len(constants)} constants into {total}")
                # Find existing constant with this value or create representation
                found_constant = None
                for const_name, const_func in CONSTANTS.items():
                    try:
                        if abs(const_func(None) - total) < 1e-10:
                            found_constant = const_name
                            break
                    except:
                        continue
                
                if found_constant:
                    other_terms.append(found_constant)
                else:
                    # Create a constant representation
                    if abs(total - round(total)) < 1e-10:
                        other_terms.append(f"constant_{int(round(total))}.0")
                    else:
                        other_terms.append(f"constant_{total}")
            else:
                # Only one constant, keep it as-is
                other_terms.extend(constants)
                        
        except Exception as e:
            print(f"   🔍 Debug: Error combining constants: {e}")
            other_terms.extend(constants)
    
    return other_terms


def rebuild_sum_from_terms(terms, FUNCTIONS):
    """
    Rebuild a sum expression from a list of terms, handling both positive and negative terms.
    """
    if not terms:
        return None
    
    # Separate positive and negative terms
    positive_terms = []
    negative_terms = []
    
    for term in terms:
        if isinstance(term, tuple) and len(term) == 2 and term[0] == 'NEGATIVE':
            negative_terms.append(term[1])
        else:
            positive_terms.append(term)
    
    if not positive_terms and not negative_terms:
        return None
    
    # Start with the first positive term, or zero if no positive terms
    if positive_terms:
        result = positive_terms[0]
        # Add remaining positive terms
        for i in range(1, len(positive_terms)):
            if "add" in FUNCTIONS:
                result = ("add", result, positive_terms[i])
            else:
                break
    else:
        # If only negative terms, start with zero (would need a zero constant)
        result = "constant_0.0"
    
    # Subtract negative terms
    for neg_term in negative_terms:
        if "subtract" in FUNCTIONS:
            result = ("subtract", result, neg_term)
        else:
            break
    
    return result


def simplify_individual(individual, max_simplification_iterations=5, debug=False):
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
            
        # Apply basic simplification
        new_structure = simplify_tree_structure(
            simplified_structure, 
            tree_dicts['FUNCTIONS'], 
            tree_dicts['TERMINALS'], 
            tree_dicts['CONSTANTS']
        )
        
        # Apply advanced term collection and simplification for addition expressions
        if isinstance(new_structure, tuple) and len(new_structure) >= 3 and str(new_structure[0]) == "add":
            if debug:
                print(f"   🔍 Debug: Attempting advanced term simplification...")
            
            terms = collect_terms_in_sum(new_structure)
            if debug:
                print(f"   🔍 Debug: Collected {len(terms)} terms: {terms}")
            
            simplified_terms = simplify_collected_terms(terms, tree_dicts['FUNCTIONS'], 
                                                       tree_dicts['TERMINALS'], tree_dicts['CONSTANTS'])
            if debug:
                print(f"   🔍 Debug: Simplified to {len(simplified_terms)} terms: {simplified_terms}")
            
            if len(simplified_terms) != len(terms) or simplified_terms != terms:
                rebuilt_structure = rebuild_sum_from_terms(simplified_terms, tree_dicts['FUNCTIONS'])
                if rebuilt_structure is not None and rebuilt_structure != new_structure:
                    new_structure = rebuilt_structure
                    if debug:
                        print(f"   🔍 Debug: Advanced simplification applied: {new_structure}")
        
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