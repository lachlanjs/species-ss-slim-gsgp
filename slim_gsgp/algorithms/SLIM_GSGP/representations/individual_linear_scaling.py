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
Individual Class with Linear Scaling for SLIM GSGP.
"""

import torch
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual as BaseIndividual
from slim_gsgp.config.slim_config_linear_scaling import calculate_linear_scaling_params

class IndividualLinearScaling(BaseIndividual):
    """
    Individual class with Linear Scaling support for SLIM_GSGP algorithm.
    
    Extends the base Individual class to include linear scaling parameters.
    """
    
    def __init__(self, collection, train_semantics=None, test_semantics=None, reconstruct=False):
        """
        Initialize Individual with Linear Scaling support.
        
        Parameters
        ----------
        collection : list
            The list of trees representing the individual.
        train_semantics : torch.Tensor, optional
            Training semantics associated with the Individual.
        test_semantics : torch.Tensor, optional
            Testing semantics associated with the Individual.
        reconstruct : bool, optional
            Whether to reconstruct the tree structure.
        """
        super().__init__(collection, train_semantics, test_semantics, reconstruct)
        
        # Linear scaling parameters: y_scaled = a + y_raw * b
        self.scaling_a = None  # intercept term
        self.scaling_b = None  # slope term
        self.use_linear_scaling = False
        
        # Set default version (needed for tree representation)
        if not hasattr(self, 'version'):
            self.version = "SLIM+SIG2"
    
    def calculate_linear_scaling(self, y_true):
        """
        Calculate and store linear scaling parameters for this individual.
        Formula: y_scaled = a + y_raw * b
        
        Parameters
        ----------
        y_true : torch.Tensor
            True target values to fit scaling parameters to.
        """
        if self.train_semantics is not None:
            # Get raw semantics (before scaling)
            y_raw = torch.sum(self.train_semantics, dim=0) if len(self.train_semantics.shape) > 1 else self.train_semantics
            
            # Calculate optimal scaling parameters
            self.scaling_a, self.scaling_b = calculate_linear_scaling_params(y_raw, y_true)
            self.use_linear_scaling = True
    
    def predict(self, data, apply_scaling=True):
        """
        Make predictions with optional linear scaling.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data for prediction.
        apply_scaling : bool, optional
            Whether to apply linear scaling to the predictions.
            
        Returns
        -------
        torch.Tensor
            Predicted values, optionally scaled.
        """
        # Get raw prediction from parent class
        raw_prediction = super().predict(data)
        
        # Apply linear scaling if enabled and parameters are available
        if apply_scaling and self.use_linear_scaling and self.scaling_a is not None:
            return self.scaling_a + raw_prediction * self.scaling_b
        
        return raw_prediction
    
    def get_scaling_info(self):
        """
        Get linear scaling parameters information.
        
        Returns
        -------
        dict
            Dictionary containing scaling parameters and status.
        """
        return {
            'use_linear_scaling': self.use_linear_scaling,
            'scaling_a': self.scaling_a,
            'scaling_b': self.scaling_b,
            'scaling_formula': f"y_scaled = {self.scaling_a:.6f} + y_raw * {self.scaling_b:.6f}" if self.use_linear_scaling else "No scaling applied"
        }
    
    def print_scaling_info(self):
        """
        Print linear scaling information for this individual.
        """
        info = self.get_scaling_info()
        print(f"Linear Scaling Status: {'Enabled' if info['use_linear_scaling'] else 'Disabled'}")
        if info['use_linear_scaling']:
            print(f"Scaling Parameter a: {info['scaling_a']:.6f}")
            print(f"Scaling Parameter b: {info['scaling_b']:.6f}")
            print(f"Scaling Formula: {info['scaling_formula']}")
        else:
            print("No linear scaling applied")
