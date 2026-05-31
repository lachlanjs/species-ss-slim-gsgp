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

Two simplification passes are available:

  1. ``simplify_constant_operations`` — conservative, hand-written constant
     folding and identity rules.
  2. ``simplify_tree_sympy`` — algebraic simplification via Sympy. Only kept
     when the result round-trips through the 4-operator (+, -, *, /) tree
     format and yields fewer nodes than the input.

Both passes operate on the *converted view* returned by
``convert_slim_individual_to_normal_tree`` and never touch
``individual.collection``, which is what ``predict()`` uses. As a result the
prediction path is bit-identical before and after simplification — the
simplified tree is only used to recompute ``individual.nodes_count`` for
Pareto-frontier selection. See ``scripts/verify_sympy_simplification.py``
for an end-to-end check.
"""

import re
import signal

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
    # Use default version if not set (during evolution)
    # Default to 'SLIM+ABS' as it's the most common
    slim_version = getattr(individual, 'version', 'SLIM+ABS')
    operator, sig, trees = check_slim_version(slim_version=slim_version)
    
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
       - x + 0 → x
       - x - 0 → x
       - x * 0 → 0
       - x * 1 → x
       - x / 1 → x
       - 0 + x → x
       - 0 * x → 0
       - 1 * x → x
    2. Constant folding (all operations):
       - 5.0 + 3.0 → 8.0
       - 10.0 - 4.0 → 6.0
       - 2.0 * 3.0 → 6.0
       - 4.0 / 2.0 → 2.0
    3. Nested constant operations:
       - c1 * x * c2 → (c1*c2) * x
       - x * c1 * c2 → (c1*c2) * x
       - x / c1 / c2 → x / (c1*c2)
       - x / c1 * c2 → x * (c2/c1)
    
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
        
        # === IDENTITY SIMPLIFICATIONS ===
        
        # x - x → 0
        if operator == 'subtract' and left_simplified == right_simplified:
            nodes_removed += 2
            return '0.0'
        
        # x / x → 1
        if operator == 'divide' and left_simplified == right_simplified:
            nodes_removed += 2
            return '1.0'
        
        # Get constant values if they exist
        left_val = get_constant_value(left_simplified) if is_constant(left_simplified) else None
        right_val = get_constant_value(right_simplified) if is_constant(right_simplified) else None
        
        # === ZERO IDENTITIES ===
        
        # x + 0 → x  or  0 + x → x
        if operator == 'add':
            if left_val is not None and left_val == 0:
                nodes_removed += 1
                return right_simplified
            if right_val is not None and right_val == 0:
                nodes_removed += 1
                return left_simplified
        
        # x - 0 → x
        if operator == 'subtract' and right_val is not None and right_val == 0:
            nodes_removed += 1
            return left_simplified
        
        # x * 0 → 0  or  0 * x → 0
        if operator == 'multiply':
            if (left_val is not None and left_val == 0) or (right_val is not None and right_val == 0):
                nodes_removed += 2
                return '0.0'
        
        # 0 / x → 0 (but not x / 0)
        if operator == 'divide' and left_val is not None and left_val == 0:
            nodes_removed += 1
            return '0.0'
        
        # === ONE IDENTITIES ===
        
        # x * 1 → x  or  1 * x → x
        if operator == 'multiply':
            if left_val is not None and left_val == 1:
                nodes_removed += 1
                return right_simplified
            if right_val is not None and right_val == 1:
                nodes_removed += 1
                return left_simplified
        
        # x / 1 → x
        if operator == 'divide' and right_val is not None and right_val == 1:
            nodes_removed += 1
            return left_simplified
        
        # === CONSTANT FOLDING ===
        
        if operator in ['add', 'subtract', 'multiply', 'divide']:
            if left_val is not None and right_val is not None:
                if operator == 'add':
                    result = left_val + right_val
                elif operator == 'subtract':
                    result = left_val - right_val
                elif operator == 'multiply':
                    result = left_val * right_val
                elif operator == 'divide':
                    if right_val != 0:
                        result = left_val / right_val
                    else:
                        return (operator, left_simplified, right_simplified)
                
                nodes_removed += 2
                return str(result)
        
        # === NESTED CONSTANT OPERATIONS ===
        
        # c1 * (c2 * x) → (c1*c2) * x
        if operator == 'multiply' and left_val is not None:
            if isinstance(right_simplified, tuple) and len(right_simplified) == 3:
                if right_simplified[0] == 'multiply':
                    right_left = right_simplified[1]
                    right_right = right_simplified[2]
                    right_left_val = get_constant_value(right_left) if is_constant(right_left) else None
                    
                    if right_left_val is not None:
                        # c1 * (c2 * x) → (c1*c2) * x
                        new_const = left_val * right_left_val
                        nodes_removed += 1
                        return ('multiply', str(new_const), right_right)
        
        # (x * c1) * c2 → x * (c1*c2)
        if operator == 'multiply' and right_val is not None:
            if isinstance(left_simplified, tuple) and len(left_simplified) == 3:
                if left_simplified[0] == 'multiply':
                    left_left = left_simplified[1]
                    left_right = left_simplified[2]
                    left_right_val = get_constant_value(left_right) if is_constant(left_right) else None
                    
                    if left_right_val is not None:
                        # (x * c1) * c2 → x * (c1*c2)
                        new_const = left_right_val * right_val
                        nodes_removed += 1
                        return ('multiply', left_left, str(new_const))
                    
                    # (c1 * x) * c2 → (c1*c2) * x
                    left_left_val = get_constant_value(left_left) if is_constant(left_left) else None
                    if left_left_val is not None:
                        new_const = left_left_val * right_val
                        nodes_removed += 1
                        return ('multiply', str(new_const), left_right)
        
        # x / c1 / c2 → x / (c1*c2)
        if operator == 'divide' and right_val is not None and right_val != 0:
            if isinstance(left_simplified, tuple) and len(left_simplified) == 3:
                if left_simplified[0] == 'divide':
                    left_left = left_simplified[1]
                    left_right = left_simplified[2]
                    left_right_val = get_constant_value(left_right) if is_constant(left_right) else None
                    
                    if left_right_val is not None and left_right_val != 0:
                        # x / c1 / c2 → x / (c1*c2)
                        new_const = left_right_val * right_val
                        nodes_removed += 1
                        return ('divide', left_left, str(new_const))
        
        # x / c1 * c2 → x * (c2/c1)  or  x * (c1/c2)
        if operator == 'multiply' and right_val is not None:
            if isinstance(left_simplified, tuple) and len(left_simplified) == 3:
                if left_simplified[0] == 'divide':
                    left_left = left_simplified[1]
                    left_right = left_simplified[2]
                    left_right_val = get_constant_value(left_right) if is_constant(left_right) else None
                    
                    if left_right_val is not None and left_right_val != 0:
                        # x / c1 * c2 → x * (c2/c1)
                        new_const = right_val / left_right_val
                        nodes_removed += 1
                        return ('multiply', left_left, str(new_const))
        
        # If no simplification occurred, return the structure with simplified children
        return (operator, left_simplified, right_simplified)
    
    simplified_tree = simplify_recursive(tree_structure)
    return simplified_tree, nodes_removed



def _count_tree_nodes(structure):
    """Count nodes in a tree-tuple structure (leaves count as 1).

    Iterative (explicit stack) so it cannot overflow Python's recursion limit
    on pathologically deep trees — e.g. a giant polynomial chain produced by
    Sympy's ``expand``.
    """
    stack = [structure]
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        if isinstance(node, tuple):
            stack.extend(node[1:])
    return count


def _clean_node_name(node):
    """Strip np.str_(...) wrappers and surrounding quotes from a leaf name."""
    s = str(node)
    if "np.str_(" in s:
        s = re.sub(r"np\.str_\('([^']+)'\)", r"\1", s)
    return s.strip("'\"")


def _format_number_leaf(value):
    """Stringify a numeric leaf in the canonical '<float>' shape used by the
    rest of the simplification code (so node counters and downstream
    constant-folding stay consistent)."""
    if isinstance(value, bool):
        value = float(value)
    f = float(value)
    if f.is_integer():
        return f"{f:.1f}"
    return repr(f)


def _resolve_constant_value(name, constants_dict):
    """Return the numeric value of a constant leaf, or None if not resolvable."""
    if name in constants_dict:
        v = constants_dict[name]
        if callable(v):
            try:
                return float(v(None).item())
            except Exception:
                return None
        try:
            return float(v)
        except Exception:
            return None
    if name.startswith("constant_"):
        raw = name.replace("constant_", "")
        if raw.startswith("_"):
            raw = "-" + raw[1:]
        try:
            return float(raw)
        except ValueError:
            return None
    try:
        return float(name)
    except (TypeError, ValueError):
        return None


def _tree_to_sympy(structure, constants_dict, sp, symbol_cache):
    """Convert a tree-tuple to a Sympy expression.

    Returns the expression or ``None`` if the tree contains a node we don't
    know how to translate (anything other than the 4 base operators, a
    'negative' unary, a terminal, or a numeric constant).
    """
    if not isinstance(structure, tuple):
        name = _clean_node_name(structure) if not isinstance(structure, (int, float)) else None
        if isinstance(structure, (int, float)):
            return sp.Float(float(structure))
        if name is None:
            return None
        const_val = _resolve_constant_value(name, constants_dict)
        if const_val is not None:
            return sp.Float(const_val)
        # Treat as a terminal variable (e.g. 'x0', 'x1', ...).
        if name not in symbol_cache:
            symbol_cache[name] = sp.Symbol(name)
        return symbol_cache[name]

    if len(structure) == 0:
        return None

    op = _clean_node_name(structure[0])

    if len(structure) == 2:
        arg = _tree_to_sympy(structure[1], constants_dict, sp, symbol_cache)
        if arg is None:
            return None
        if op.lower() in ("negative", "neg"):
            return -arg
        return None

    if len(structure) == 3:
        left = _tree_to_sympy(structure[1], constants_dict, sp, symbol_cache)
        right = _tree_to_sympy(structure[2], constants_dict, sp, symbol_cache)
        if left is None or right is None:
            return None
        if op == "add":
            return left + right
        if op == "subtract":
            return left - right
        if op == "multiply":
            return left * right
        if op == "divide":
            return left / right
        return None

    return None


def _sympy_to_tree(expr, sp):
    """Convert a Sympy expression back to a tree-tuple structure.

    Returns ``(structure, True)`` on success, ``(None, False)`` if the
    expression contains a construct the SLIM tree format cannot represent
    (non-trivial powers, transcendental functions, etc.).

    The output uses only ``add``, ``subtract``, ``multiply``, ``divide`` and
    numeric/symbol leaves — exactly the set produced elsewhere in the
    codebase, so downstream passes (PNG rendering, node counting,
    ``simplify_constant_operations``) keep working unchanged.
    """
    if expr.is_Symbol:
        return str(expr), True

    if expr.is_Number:
        try:
            return _format_number_leaf(float(expr)), True
        except (TypeError, ValueError):
            return None, False

    if expr.is_Add:
        # Split into "positive" and "negative" terms so we emit subtract where
        # natural, instead of "x + (-1)*y" which inflates the node count.
        positives, negatives = [], []
        for term in expr.args:
            coeff, rest = term.as_coeff_Mul()
            if coeff.is_Number and float(coeff) < 0:
                negated = (-coeff) * rest if rest != 1 else (-coeff)
                negatives.append(negated)
            else:
                positives.append(term)

        sub_pos = []
        for t in positives:
            r, ok = _sympy_to_tree(t, sp)
            if not ok:
                return None, False
            sub_pos.append(r)
        sub_neg = []
        for t in negatives:
            r, ok = _sympy_to_tree(t, sp)
            if not ok:
                return None, False
            sub_neg.append(r)

        if not sub_pos and not sub_neg:
            return _format_number_leaf(0.0), True
        if not sub_pos:
            # All terms negative: emit 0 - sum(...).
            result = sub_neg[0]
            for s in sub_neg[1:]:
                result = ("add", result, s)
            return ("subtract", _format_number_leaf(0.0), result), True

        result = sub_pos[0]
        for s in sub_pos[1:]:
            result = ("add", result, s)
        for s in sub_neg:
            result = ("subtract", result, s)
        return result, True

    if expr.is_Mul:
        numerator, denominator = [], []
        for factor in expr.args:
            if factor.is_Pow and factor.args[1].is_Integer:
                power = int(factor.args[1])
                if power == -1:
                    denominator.append(factor.args[0])
                    continue
                if power < -1:
                    # x ** -n with n > 1: would require nested multiplications.
                    # Bail out to keep semantics 100% obvious.
                    return None, False
                if power > 1:
                    # Expand x**n into n-1 explicit multiplications when small.
                    if power > 5:
                        return None, False
                    expanded = factor.args[0]
                    for _ in range(power - 1):
                        expanded = expanded * factor.args[0]
                    numerator.append(expanded)
                    continue
            numerator.append(factor)

        if not numerator:
            num_tree = _format_number_leaf(1.0)
        else:
            r0, ok = _sympy_to_tree(numerator[0], sp)
            if not ok:
                return None, False
            num_tree = r0
            for n in numerator[1:]:
                r, ok = _sympy_to_tree(n, sp)
                if not ok:
                    return None, False
                num_tree = ("multiply", num_tree, r)

        if not denominator:
            return num_tree, True

        r0, ok = _sympy_to_tree(denominator[0], sp)
        if not ok:
            return None, False
        den_tree = r0
        for d in denominator[1:]:
            r, ok = _sympy_to_tree(d, sp)
            if not ok:
                return None, False
            den_tree = ("multiply", den_tree, r)
        return ("divide", num_tree, den_tree), True

    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.is_Integer:
            n = int(exponent)
            if n == -1:
                base_tree, ok = _sympy_to_tree(base, sp)
                if not ok:
                    return None, False
                return ("divide", _format_number_leaf(1.0), base_tree), True
            if 2 <= n <= 5:
                base_tree, ok = _sympy_to_tree(base, sp)
                if not ok:
                    return None, False
                result = base_tree
                for _ in range(n - 1):
                    result = ("multiply", result, base_tree)
                return result, True
        return None, False

    return None, False


class _SympyTimeout(Exception):
    """Raised internally when a Sympy operation exceeds its time budget."""


# ---------------------------------------------------------------------------
# Telemetry for the Sympy division (rational) simplification pass.
#
# Tracks, across every simplify_population call, how many individuals could
# NOT have their divisions simplified — i.e. the full (rational) pass was
# either preemptively skipped because the tree has too many divisions ("gated",
# it would explode) or it was attempted but timed out ("timeout"). Both mean
# the divisions of that individual were left un-combined.
#
# run.py resets these at the start and reports per-execution deltas and a
# global summary at the end.
# ---------------------------------------------------------------------------
_simp_telemetry = {
    "executions": 0,     # number of simplify_population calls (≈ slim runs that simplified)
    "individuals": 0,    # individuals attempted (valid converted tree)
    "div_failed": 0,     # individuals whose division pass did not complete (gated + timeout + skipped)
    "div_gated": 0,      # subset: skipped preemptively (divisions > threshold)
    "div_timeout": 0,    # subset: full op actually timed out
}

# Outcome of the most recent simplify_tree_sympy full pass, read by
# simplify_population to update the telemetry above. One of:
#   "completed" | "gated" | "timeout" | "skipped"
_last_full_pass = "skipped"


def reset_simplification_telemetry():
    """Zero all Sympy simplification telemetry counters."""
    for k in _simp_telemetry:
        _simp_telemetry[k] = 0


def get_simplification_telemetry():
    """Return a snapshot (copy) of the Sympy simplification telemetry counters."""
    return dict(_simp_telemetry)


def _can_use_alarm():
    """SIGALRM-based timeouts only work on Unix and only from the main thread
    of a process. Returns True when both conditions hold."""
    import threading
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


def _run_with_alarm(fn, seconds):
    """Run ``fn()`` under a wall-clock timeout using SIGALRM.

    Returns the result, or ``None`` if it timed out or raised. Restores any
    previously installed SIGALRM handler and cancels the timer on the way out,
    so we never leave a dangling alarm.
    """
    def _handler(signum, frame):
        raise _SympyTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    except _SympyTimeout:
        return None
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _count_divide_nodes(structure):
    """Iteratively count 'divide' operator nodes in a tree-tuple structure.

    This is the cheap predictor of the Sympy rational blow-up: the cost of
    combining a sum of fractions over a common denominator is ~2**(number of
    distinct denominators), so a high divide count flags an expression where
    the full pass would explode (and, empirically, yield no node reduction).
    """
    stack = [structure]
    n = 0
    while stack:
        node = stack.pop()
        if isinstance(node, tuple):
            if str(node[0]) == "divide":
                n += 1
            stack.extend(node[1:])
    return n


def simplify_tree_sympy(
    tree_structure,
    constants_dict,
    max_input_nodes=400,
    cheap_timeout=3.0,
    full_timeout=5.0,
    max_divisions_for_full=7,
):
    """Algebraically simplify a tree using Sympy — HYBRID strategy.

    The hard problem (see scripts/prove_sympy_blowup_is_intrinsic.py) is that
    the powerful rational ops (``cancel``/``simplify``/``factor``) place a sum
    of fractions over a common denominator, which is exponential in the number
    of distinct denominators. Empirically, when those ops *help* they finish in
    well under a second; when they *explode* they never finish AND yield no node
    reduction anyway. So:

    1. **Cheap pass (always):** ``expand`` + ``together``. These never form a
       common denominator over a sum, so they cannot explode (a short timeout
       still guards the SLIM\\* product-of-sums edge case). Empirically this
       already captures essentially all the achievable node reduction.

    2. **Full pass (generous timeout, division-gated):** ``cancel`` →
       ``simplify`` → ``factor``, each given ``full_timeout`` seconds —
       deliberately generous so the full simplification has plenty of room to
       complete on normal individuals (which finish in <1s). It is only
       attempted when the tree has at most ``max_divisions_for_full`` division
       nodes: beyond that the rational machinery would explode AND
       (empirically) yield zero node reduction, so spending the generous
       budget there is pure waste. If the first op still times out, the
       remaining ones share the same explosive structure, so we stop early.

    The smallest valid round-trip across every candidate (and the original) is
    returned, so the result is never larger than the input.

    When SIGALRM is unavailable (non-Unix, or not the main thread — e.g. a
    multiprocessing worker), the full pass is skipped entirely and only the
    intrinsically-bounded cheap pass runs, guaranteeing termination everywhere.

    Parameters
    ----------
    tree_structure : tuple or scalar
        The tree to simplify, in the format produced by
        ``convert_slim_individual_to_normal_tree``.
    constants_dict : dict
        Maps constant names to either a numeric value or a callable returning
        a torch tensor (the convention used by the SLIM ``CONSTANTS`` dict).
    max_input_nodes : int
        Skip Sympy entirely above this many nodes (sanity bound on building the
        symbolic expression at all).
    cheap_timeout : float
        Wall-clock seconds allowed per cheap op (``expand``/``together``).
    full_timeout : float
        Wall-clock seconds allowed per full op (``cancel``/``simplify``/
        ``factor``). Generous on purpose — normal individuals finish in well
        under a second; only borderline ones ever hit this.
    max_divisions_for_full : int
        Skip the full pass when the tree has more division nodes than this.
        Such trees make the rational ops explode for no node-count benefit, so
        only the cheap pass runs on them.

    Returns
    -------
    tuple
        (simplified_tree, nodes_removed, applied)

        ``applied`` is ``True`` when a strictly smaller tree was produced.
        ``nodes_removed`` is always >= 0.
    """
    global _last_full_pass
    _last_full_pass = "skipped"  # default until/unless the full pass runs

    try:
        import sympy as sp
    except ImportError:
        return tree_structure, 0, False

    if tree_structure is None:
        return tree_structure, 0, False

    original_nodes = _count_tree_nodes(tree_structure)
    if original_nodes > max_input_nodes:
        return tree_structure, 0, False

    symbol_cache: dict = {}
    try:
        expr = _tree_to_sympy(tree_structure, constants_dict, sp, symbol_cache)
    except Exception:
        return tree_structure, 0, False
    if expr is None:
        return tree_structure, 0, False

    use_alarm = _can_use_alarm()
    candidates = []

    # --- Cheap pass: never combines fractions, so it cannot blow up. -------
    for builder in (sp.expand, sp.together):
        if use_alarm:
            res = _run_with_alarm(lambda b=builder: b(expr), cheap_timeout)
        else:
            try:
                res = builder(expr)
            except Exception:
                res = None
        if res is not None:
            candidates.append(res)

    # --- Full pass: powerful but potentially explosive. Only when we can ----
    #     bound it with an alarm AND the division count is low enough that it
    #     can actually finish usefully. Generous per-op budget; bail on first
    #     timeout since the remaining ops share the same explosive structure.
    if use_alarm and _count_divide_nodes(tree_structure) <= max_divisions_for_full:
        timed_out = False
        for builder in (sp.cancel, sp.simplify, sp.factor):
            res = _run_with_alarm(lambda b=builder: b(expr), full_timeout)
            if res is None:
                timed_out = True
                break
            candidates.append(res)
        _last_full_pass = "timeout" if timed_out else "completed"
    elif use_alarm:
        # Skipped preemptively: too many divisions → would explode for no gain.
        _last_full_pass = "gated"
    else:
        # No timeout mechanism available → full pass not attempted.
        _last_full_pass = "skipped"

    candidates.append(expr)

    best_tree = tree_structure
    best_count = original_nodes
    for cand in candidates:
        # Cheap, non-recursive size estimate. A candidate whose Sympy op count
        # already meets/exceeds the best can never win, so skip the (recursive,
        # potentially very deep) round-trip — also guards against expand()
        # producing a huge polynomial.
        try:
            if sp.count_ops(cand) >= best_count:
                continue
        except Exception:
            pass
        try:
            back, ok = _sympy_to_tree(cand, sp)
        except Exception:
            continue
        if not ok or back is None:
            continue
        cnt = _count_tree_nodes(back)
        if cnt < best_count:
            best_count = cnt
            best_tree = back

    nodes_removed = max(original_nodes - best_count, 0)
    return best_tree, nodes_removed, nodes_removed > 0


def simplify_population(population, debug=False):
    """
    Apply simplification to all individuals in a population.
    
    This function converts each SLIM individual to a tree structure, applies constant
    folding and identity simplifications, then updates the individual's nodes_count
    to reflect the simplified tree size.
    
    This is typically applied to the Pareto frontier (non-dominated individuals)
    at the end of evolution, before selecting the final solution based on 
    normalized fitness and size.
    
    Parameters
    ----------
    population : list[Individual]
        List of individuals to simplify (typically the Pareto frontier)
    debug : bool
        If True, print debug information for first failed individual
        
    Returns
    -------
    dict
        Dictionary with statistics:
        - 'total': Total number of individuals
        - 'simplified': Number of individuals that were simplified
        - 'failed': Number that failed to simplify
        - 'nodes_removed_total': Total nodes removed across all individuals
        - 'div_individuals': Individuals with a valid tree (division pass attempted)
        - 'div_failed': Of those, individuals whose division pass did not complete
        - 'div_gated': Subset skipped preemptively (too many divisions)
        - 'div_timeout': Subset where the full op actually timed out
    """
    total = len(population)
    simplified = 0
    failed = 0
    nodes_removed_total = 0
    debug_printed = False

    # --- Per-call telemetry for the division simplification ---
    _simp_telemetry["executions"] += 1
    call_individuals = 0     # individuals with a valid converted tree (attempted)
    call_div_failed = 0      # of those, divisions not simplified (gated/timeout/skipped)
    call_div_gated = 0
    call_div_timeout = 0

    for i, individual in enumerate(population):
        try:
            # Convert SLIM individual to tree structure
            tree_structure, tree_dicts = convert_slim_individual_to_normal_tree(individual)
            
            if tree_structure and tree_dicts:
                # Count nodes in tree BEFORE simplification
                def count_nodes(tree):
                    if not isinstance(tree, tuple):
                        return 1
                    if len(tree) == 2:  # Unary operator
                        return 1 + count_nodes(tree[1])
                    elif len(tree) == 3:  # Binary operator
                        return 1 + count_nodes(tree[1]) + count_nodes(tree[2])
                    return 1

                original_node_count = count_nodes(tree_structure)

                # First pass: Sympy-based algebraic simplification. It only
                # accepts results that round-trip through the 4-operator tree
                # format, so the output is always something the rest of the
                # pipeline can render and count.
                sympy_tree, sympy_removed, _ = simplify_tree_sympy(
                    tree_structure, tree_dicts['CONSTANTS']
                )

                # Telemetry: did the division (rational) pass complete for this
                # individual? simplify_tree_sympy recorded the outcome.
                call_individuals += 1
                if _last_full_pass != "completed":
                    call_div_failed += 1
                    if _last_full_pass == "gated":
                        call_div_gated += 1
                    elif _last_full_pass == "timeout":
                        call_div_timeout += 1

                # Second pass: conservative constant folding cleans up any
                # remaining identity/constant patterns the Sympy result may
                # still expose (e.g. when Sympy was skipped due to size).
                simplified_tree, nodes_removed = simplify_constant_operations(
                    sympy_tree, tree_dicts['CONSTANTS']
                )
                nodes_removed += sympy_removed

                # Count nodes in simplified tree
                simplified_node_count = count_nodes(simplified_tree)
                
                # Update individual's node count with simplified count
                individual.nodes_count = simplified_node_count
                
                # Track if simplification occurred (compare converted tree sizes)
                if simplified_node_count < original_node_count:
                    simplified += 1
                    nodes_removed_total += (original_node_count - simplified_node_count)
                    
                    # Show first 3 simplifications if debug mode is enabled
                    if debug and simplified <= 3:
                        print(f"\n[DEBUG] Individual {i} simplified: {original_node_count} → {simplified_node_count} nodes")
                        print(f"  nodes_removed from simplify_constant_operations: {nodes_removed}")
                        print(f"  Actual node difference: {original_node_count - simplified_node_count}")
            else:
                failed += 1
                if debug and not debug_printed:
                    print(f"\n[DEBUG] Individual {i}: convert_slim_individual_to_normal_tree returned None")
                    print(f"  Has collection: {hasattr(individual, 'collection')}")
                    if hasattr(individual, 'collection'):
                        print(f"  Collection length: {len(individual.collection) if individual.collection else 0}")
                        print(f"  Collection type: {type(individual.collection)}")
                    print(f"  Has version: {hasattr(individual, 'version')}")
                    if hasattr(individual, 'version'):
                        print(f"  Version: {individual.version}")
                    debug_printed = True
                
        except Exception as e:
            # If simplification fails for any reason, keep original node count
            failed += 1
            if debug and not debug_printed:
                print(f"\n[DEBUG] Individual {i}: Exception during simplification: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                debug_printed = True
    
    # Roll this call's division telemetry into the global counters.
    _simp_telemetry["individuals"] += call_individuals
    _simp_telemetry["div_failed"] += call_div_failed
    _simp_telemetry["div_gated"] += call_div_gated
    _simp_telemetry["div_timeout"] += call_div_timeout

    return {
        'total': total,
        'simplified': simplified,
        'failed': failed,
        'nodes_removed_total': nodes_removed_total,
        # Division-simplification telemetry for this call.
        'div_individuals': call_individuals,
        'div_failed': call_div_failed,
        'div_gated': call_div_gated,
        'div_timeout': call_div_timeout,
    }
