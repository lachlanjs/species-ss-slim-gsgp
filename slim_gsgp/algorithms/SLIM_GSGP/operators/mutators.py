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
Mutation Functions for SLIM GSGP.
"""

import random

import torch
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp.utils.utils import get_random_tree

# Global counter for OMS transformations (ms < 0.1 -> 0)
_oms_zero_transformations_count = 0

def reset_oms_counter():
    """Reset the OMS zero transformations counter."""
    global _oms_zero_transformations_count
    _oms_zero_transformations_count = 0

def get_oms_counter():
    """Get the current OMS zero transformations count."""
    global _oms_zero_transformations_count
    return _oms_zero_transformations_count

# Global counter for NM degenerate directions (||s_r|| ≈ 0, normalization skipped)
_nm_degenerate_count = 0

def reset_nm_counter():
    """Reset the NM degenerate directions counter."""
    global _nm_degenerate_count
    _nm_degenerate_count = 0

def get_nm_counter():
    """Get the current NM degenerate directions count."""
    global _nm_degenerate_count
    return _nm_degenerate_count


def normalize_semantics(s_R, mode='zscore', epsilon=1e-8):
    """
    Apply statistical normalization to the semantic vector of a random program.

    Implements the Normalized Mutation Operator (Bakurov et al., 2024),
    inspired by batch normalization from deep learning.  Normalization is
    computed over the full training semantic vector so that the perturbation
    has a well-defined, bounded magnitude regardless of tree depth or dataset
    scale.

    Parameters
    ----------
    s_R : torch.Tensor, shape (n,)
        Semantic vector of the random program TR evaluated on the n training
        (or test) points.
    mode : str, optional
        'zscore' -> standardisation: (s_R - mean) / std  [default]
        'minmax' -> normalisation:   2*(s_R - min)/(max - min) - 1
    epsilon : float, optional
        Numerical threshold to avoid division by zero. Default: 1e-8.

    Returns
    -------
    torch.Tensor, shape (n,)
        Normalised semantic vector ready to be used as perturbation in GSM.
        If normalisation would divide by zero (constant / degenerate tree),
        returns a zero vector (null perturbation) and increments the global
        degenerate counter.
    """
    global _nm_degenerate_count
    if mode == 'zscore':
        mu = s_R.mean()
        sigma = s_R.std()
        if sigma < epsilon:
            _nm_degenerate_count += 1
            return torch.zeros_like(s_R)
        return (s_R - mu) / sigma
    elif mode == 'minmax':
        min_val = s_R.min()
        max_val = s_R.max()
        rango = max_val - min_val
        if rango < epsilon:
            _nm_degenerate_count += 1
            return torch.zeros_like(s_R)
        return 2.0 * (s_R - min_val) / rango - 1.0
    else:
        raise ValueError(f"norm_mode must be 'zscore' or 'minmax', got: {mode}")


# two tree function
def two_trees_delta(operator="sum"):
    """
    Generate a function for the two-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).

    Returns
    -------
    Callable
        A mutation function (`tt_delta`) for two Individuals that returns the mutated semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree individual.
        tr2 : Individual
            The second tree individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.

    Notes
    -----
    The returned function ('tt_delta_{operator}') takes as input two individuals, the mutation step, a boolean
    representing whether to use the train or test semantics, and returns the calculated semantics of the new individual.
    """

    def tt_delta(tr1, tr2, ms, testing):
        """
        Performs delta mutation between two trees based on their semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree Individual.
        tr2 : Individual
            The second tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
        """        
        if testing:
            return (
                torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
                if operator == "sum"
                else torch.add(
                    1, torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
                )
            )
        else:
            return (
                torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics))
                if operator == "sum"
                else torch.add(
                    1,
                    torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics)),
                )
            )

    tt_delta.__name__ += "_" + operator

    return tt_delta


def one_tree_delta(operator="sum", sig=False):
    """
    Generate a function for the one-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        A mutation function (`ot_delta`) for one-tree mutation.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
    Notes
    -----
    The returned function ('ot_delta_{operator}_{sig}') takes as input one individual, the mutation step,
    a boolean representing whether to use the train or test semantics, and returns the mutated semantics.
    """
    def ot_delta(tr1, ms, testing):
        """
        Performs delta mutation on one tree based on its semantics.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
        """
        if sig:
            if testing:
                return (
                    torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1))
                    if operator == "sum"
                    else torch.add(
                        1, torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1))
                    )
                )
            else:
                return (
                    torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1))
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)),
                    )
                )
        else: # abs
            if testing:
                return (
                    torch.mul(
                        ms,
                        torch.sub(
                            1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics)))
                        ),
                    )
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(
                            ms,
                            torch.sub(
                                1,
                                torch.div(
                                    2, torch.add(1, torch.abs(tr1.test_semantics))
                                ),
                            ),
                        ),
                    )
                )
            else: 
                return (
                    torch.mul(
                        ms,
                        torch.sub(
                            1,
                            torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))),
                        ),
                    )
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(
                            ms,
                            torch.sub(
                                1,
                                torch.div(
                                    2, torch.add(1, torch.abs(tr1.train_semantics))
                                ),
                            ),
                        ),
                    )
                )

    ot_delta.__name__ += "_" + operator + "_" + str(sig)
    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS,CONSTANTS,two_trees=True,operator="sum",single_tree_sigmoid=False,sig=False, oms: bool=False, nm: bool=False, norm_mode: str = 'zscore'):
    """
    Generate an inflate mutation function.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    two_trees : bool
        Boolean indicating if two trees should be used.
    operator : str
        The operator to be used in the mutation.
    single_tree_sigmoid : bool
        Boolean indicating if sigmoid should be applied to a single tree.
    sig : bool
        Boolean indicating if sigmoid should be applied.
    oms : bool
        Boolean indicating whether the optimal step mutation should be used
    nm : bool
        Boolean indicating whether normalized mutation should be used.
        Applies statistical normalization (z-score or min-max) over the full
        semantic vector of TR before scaling by ms, as in Bakurov et al.
        (2024). Ignored if oms=True.
    norm_mode : str, optional
        Normalization mode for NM. 'zscore' (default) standardises the
        semantic vector to zero mean and unit variance; 'minmax' maps it to
        the interval [-1, 1]. Only used when nm=True.

    Returns
    -------
    Callable
        An inflate mutation function (`inflate`).

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.

    Notes
    -----
    The returned function performs inflate mutation on Individuals, using either one or two randomly generated trees
    and applying either delta mutation or sigmoid mutation based on the parameters.
    """
    def inflate(
        individual,
        ms,
        X,
        max_depth=8,
        p_c=0.1,
        X_test=None,
        grow_probability=1,
        reconstruct=True,
        y_train=None,
        y_test=None,
    ):
        """
        Perform inflate mutation on the given Individual.

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).
        y_train : torch.Tensor, optional
            Target training data for guiding the optimal mutation step
        y_test : torch.Tensor, optional
            Target testing data for guiding the optimal mutation step

        Returns
        -------
        Individual
            The mutated tree Individual.
        """
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=True,
            )
            random_tree2 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=True,
            )
            # adding the random trees to a list, to be used in the creation of a new block
            random_trees = [random_tree1, random_tree2]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(X_test, testing=True, logistic=True)
                    for rt in random_trees
                ]                        
        else:
            # getting one random tree
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=single_tree_sigmoid or sig,
            )
            # adding the random tree to a list, to be used in the creation of a new block
            random_trees = [random_tree1]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(
                        X_test, testing=True, logistic=single_tree_sigmoid or sig
                    )
                    for rt in random_trees
                ]
                    
        # getting the correct mutation operator, based on the number of random trees used
        variator = (
            two_trees_delta(operator=operator)
            if two_trees
            else one_tree_delta(operator=operator, sig=sig)
        )
        
        if operator == "sum":
            operator_f = torch.sum
        else:
            operator_f = torch.prod

        # compute semantic direction s_r (shared by nm and oms blocks)
        if nm or oms:
            if two_trees:
                tr1, tr2 = random_trees
                s_r = torch.sub(tr1.train_semantics, tr2.train_semantics)
            else:  # one tree
                tr1, = random_trees
                if sig:
                    s_r = torch.sub(torch.mul(2, tr1.train_semantics), 1)
                else:
                    s_r = torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))))

        # calculate the optimal mutation step value here
        if oms:
            s_r_inv = s_r / (1e-7 + torch.mul(y_train.shape[0]), s_r * s_r) if s_r.shape == torch.Size([1]) else s_r / (1e-5 + torch.sum(s_r * s_r))                        
            ms = torch.vdot(s_r_inv.broadcast_to(y_train.shape), y_train - operator_f(individual.train_semantics, dim=0)) # .flatten()
            ms = torch.clamp(ms, -100.0, 100.0)  
            
            # Convert near-zero mutation steps to 0 to prevent overfitting and bloat
            if torch.abs(ms) < 0.1:
                global _oms_zero_transformations_count
                _oms_zero_transformations_count += 1
                ms = torch.tensor(0.0)
            

        # creating the new block for the individual, based on the random trees and operators
        # For NM, apply statistical normalization (Bakurov et al., 2024) to the full semantic
        # vector of TR before scaling by ms (Normalized Mutation Operator).
        if nm and not oms:
            s_r_train_norm = normalize_semantics(s_r, mode=norm_mode)
            if operator == "sum":
                block_train_sem = torch.mul(ms, s_r_train_norm)
            else:
                block_train_sem = torch.add(1, torch.mul(ms, s_r_train_norm))
            if X_test is not None:
                if two_trees:
                    s_r_test = torch.sub(random_trees[0].test_semantics, random_trees[1].test_semantics)
                else:
                    if sig:
                        s_r_test = torch.sub(torch.mul(2, random_trees[0].test_semantics), 1)
                    else:
                        s_r_test = torch.sub(1, torch.div(2, torch.add(1, torch.abs(random_trees[0].test_semantics))))
                s_r_test_norm = normalize_semantics(s_r_test, mode=norm_mode)
                block_test_sem = torch.mul(ms, s_r_test_norm) if operator == "sum" else torch.add(1, torch.mul(ms, s_r_test_norm))
            else:
                block_test_sem = None
        else:
            block_train_sem = variator(*random_trees, ms, testing=False)
            block_test_sem = variator(*random_trees, ms, testing=True) if X_test is not None else None

        ms_struct = ms

        new_block = Tree(
            structure=[variator, *random_trees, ms_struct],
            train_semantics=block_train_sem,
            test_semantics=block_test_sem,
            reconstruct=True,
        )
        # creating the offspring individual, by adding the new block to it
        offs = Individual(
            collection=[*individual.collection, new_block] if reconstruct else None,
            train_semantics=torch.stack(
                [
                    *individual.train_semantics,
                    (
                        new_block.train_semantics
                        if new_block.train_semantics.shape != torch.Size([])
                        else new_block.train_semantics.repeat(len(X))
                    ),
                ]
            ),
            test_semantics=(
                (
                    torch.stack(
                        [
                            *individual.test_semantics,
                            (
                                new_block.test_semantics
                                if new_block.test_semantics.shape != torch.Size([])
                                else new_block.test_semantics.repeat(len(X_test))
                            ),
                        ]
                    )
                )
                if individual.test_semantics is not None
                else None
            ),
            reconstruct=reconstruct,
            use_linear_scaling=getattr(individual, 'use_linear_scaling', False),
        )
        # computing offspring attributes
        offs.size = individual.size + 1
        offs.nodes_collection = [*individual.nodes_collection, new_block.nodes]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [*individual.depth_collection, new_block.depth]
        offs.depth = max(
            [
                depth - (i - 1) if i != 0 else depth
                for i, depth in enumerate(offs.depth_collection)
            ]
        ) + (offs.size - 1)

        return offs

    return inflate


def deflate_mutation(individual, reconstruct):
    """
    Perform deflate mutation on a given Individual by removing a random 'block'.

    Parameters
    ----------
    individual : Individual
        The Individual to be mutated.
    reconstruct : bool
        Whether to store the Individual's structure after mutation.

    Returns
    -------
    Individual
        The mutated individual
    """
    # choosing the block that will be removed
    mut_point = random.randint(1, individual.size - 1)

    # removing the block from the individual and creating a new Individual
    offs = Individual(
        collection=(
            [
                *individual.collection[:mut_point],
                *individual.collection[mut_point + 1 :],
            ]
            if reconstruct
            else None
        ),
        train_semantics=torch.stack(
            [
                *individual.train_semantics[:mut_point],
                *individual.train_semantics[mut_point + 1 :],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual.test_semantics[:mut_point],
                    *individual.test_semantics[mut_point + 1 :],
                ]
            )
            if individual.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
        use_linear_scaling=getattr(individual, 'use_linear_scaling', False),
    )

    # computing offspring attributes
    offs.size = individual.size - 1
    offs.nodes_collection = [
        *individual.nodes_collection[:mut_point],
        *individual.nodes_collection[mut_point + 1 :],
    ]
    offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

    offs.depth_collection = [
        *individual.depth_collection[:mut_point],
        *individual.depth_collection[mut_point + 1 :],
    ]
    offs.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs.depth_collection)
        ]
    ) + (offs.size - 1)

    return offs
