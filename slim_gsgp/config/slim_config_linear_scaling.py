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
Configuration file for SLIM GSGP with Linear Scaling
"""

import torch
from slim_gsgp.config.slim_config import *

# Inherit all original configurations
slim_gsgp_parameters_linear_scaling = slim_gsgp_parameters.copy()
slim_gsgp_solve_parameters_linear_scaling = slim_gsgp_solve_parameters.copy()
slim_gsgp_pi_init_linear_scaling = slim_gsgp_pi_init.copy()

# Add linear scaling specific parameters
slim_gsgp_parameters_linear_scaling["use_linear_scaling"] = True
slim_gsgp_solve_parameters_linear_scaling["linear_scaling"] = True

def calculate_linear_scaling_params(y_raw, y_true):
    """
    Calculate optimal linear scaling parameters a and b for: y_scaled = a + y_raw * b
    
    Parameters
    ----------
    y_raw : torch.Tensor
        Raw output from the individual
    y_true : torch.Tensor
        True target values
        
    Returns
    -------
    tuple
        (a, b) scaling parameters where a is intercept and b is slope
    """
    try:
        # Create design matrix [ones, y_raw] for y_scaled = a + y_raw * b
        A = torch.stack([torch.ones(len(y_raw)), y_raw], dim=1)
        
        # Solve least squares: A * [a, b]^T = y_true
        coeffs = torch.linalg.lstsq(A, y_true).solution
        
        a, b = coeffs[0], coeffs[1]
        
        # Ensure finite values
        if not torch.isfinite(a) or not torch.isfinite(b):
            return 0.0, 1.0
            
        return float(a), float(b)
    except:
        # Fallback to no scaling if calculation fails
        return 0.0, 1.0
