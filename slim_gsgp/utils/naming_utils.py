"""
Utility functions for building consistent execution type and variant names.
These functions ensure naming consistency across different execution scripts.
"""

def build_execution_type(use_linear_scaling=False, use_oms=False, 
                        use_pareto_tournament=False, use_simplification=True):
    """
    Build execution type name based on enabled features.
    
    Args:
        use_linear_scaling: Whether linear scaling is enabled
        use_oms: Whether OMS is enabled
        use_pareto_tournament: Whether Pareto tournament is enabled
        use_simplification: Whether simplification is enabled
        
    Returns:
        str: Execution type name (e.g., "slim_linear_scaling_pareto")
        
    Examples:
        >>> build_execution_type()
        'slim'
        >>> build_execution_type(use_linear_scaling=True, use_pareto_tournament=True)
        'slim_linear_scaling_pareto'
        >>> build_execution_type(use_oms=True, use_simplification=False)
        'slim_oms_no_simplif'
    """
    execution_type = "slim"
    
    if use_linear_scaling:
        execution_type += "_linear_scaling"
    if use_oms:
        execution_type += "_oms"
    if use_pareto_tournament:
        execution_type += "_pareto"
    if not use_simplification:
        execution_type += "_no_simplif"
    
    return execution_type


def build_variant_name(slim_version, use_oms=False, use_linear_scaling=False, 
                      use_pareto_tournament=False):
    """
    Build variant name based on SLIM version and enabled features.
    
    Args:
        slim_version: SLIM version (e.g., "SLIM+ABS", "SLIM+SIG2")
        use_oms: Whether OMS is enabled
        use_linear_scaling: Whether linear scaling is enabled
        use_pareto_tournament: Whether Pareto tournament is enabled
        
    Returns:
        str: Variant name with features (e.g., "SLIM+ABS + LS + Pareto")
        
    Examples:
        >>> build_variant_name("SLIM+ABS")
        'SLIM+ABS'
        >>> build_variant_name("SLIM+ABS", use_linear_scaling=True)
        'SLIM+ABS + LS'
        >>> build_variant_name("SLIM+ABS", use_oms=True, use_linear_scaling=True, use_pareto_tournament=True)
        'SLIM+ABS + OMS + LS + Pareto'
    """
    variant_parts = [slim_version]
    
    if use_oms:
        variant_parts.append("OMS")
    if use_linear_scaling:
        variant_parts.append("LS")
    if use_pareto_tournament:
        variant_parts.append("Pareto")
    
    return " + ".join(variant_parts)
