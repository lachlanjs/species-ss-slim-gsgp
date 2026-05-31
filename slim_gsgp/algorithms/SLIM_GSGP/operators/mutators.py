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


def make_nm_variator(operator, train_stats, two_trees, mode="n1"):
    """
    Create a variator closure for Normalized Mutation that captures training statistics.

    This ensures that predict() and any future reconstruction reuse the same
    numeric constants that were computed from training data during mutation
    (mean/std for N1, min/max for N2), rather than falling back to the standard
    ABS/SIG variator formula.

    Parameters
    ----------
    operator : str
        The aggregation operator ("sum" or "mul").
    train_stats : tuple
        Numeric constants computed from the training semantics during mutation.
        N1 (standardization): (mean_val, std_val).
        N2 (min-max normalization): (min_val, max_val).
    two_trees : bool
        Whether the block was built from two random trees (difference) or one.
    mode : str
        Normalization mode: "n1" (standardization, Eq. 1 of Bakurov et al. 2024)
        or "n2" (min-max normalization to [-1, 1], Eq. 3).

    Returns
    -------
    Callable
        A variator function compatible with the Tree.structure convention:
        ``nm_variator(*random_trees, ms, testing)`` → torch.Tensor.
    """
    epsilon = 1e-8

    def nm_variator(*args, testing):
        # Unpack: (tr1, tr2, ms, testing) for two_trees, (tr1, ms, testing) for one tree.
        if two_trees:
            tr1, tr2, ms = args
            raw = (
                torch.sub(tr1.test_semantics, tr2.test_semantics)
                if testing
                else torch.sub(tr1.train_semantics, tr2.train_semantics)
            )
        else:
            tr1, ms = args
            raw = tr1.test_semantics if testing else tr1.train_semantics

        if mode == "n2":
            # N2: MR = 2 * (TR - min) / (max - min) - 1   (Eq. 3)
            min_val, max_val = train_stats
            rng = max_val - min_val
            if not torch.isfinite(rng) or rng < epsilon:
                norm = torch.zeros_like(raw)
            else:
                norm = 2.0 * (raw - min_val) / rng - 1.0
                norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # N1: SR = (TR - mean) / std   (Eq. 1)
            mean_val, std_val = train_stats
            if not torch.isfinite(std_val) or std_val < epsilon:
                norm = torch.zeros_like(raw)
            else:
                norm = (raw - mean_val) / std_val
                norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

        if operator == "sum":
            return torch.mul(ms, norm)
        else:
            return torch.add(1, torch.mul(ms, norm))

    nm_variator.__name__ = f"nm_variator_{operator}_{'tt' if two_trees else 'ot'}"
    return nm_variator


def normalize_semantics(s_R, mode="n1", epsilon=1e-8, stats=None):
    """
    Normalize the semantic vector of a random program for Normalized Mutation.

    Implements the two operators from Bakurov et al. (2024):

      * N1 — standardization (Eq. 1): ``SR = (TR - mean) / std``. Centers TR to
        mean 0, std 1, so the perturbation is zero-centered.
      * N2 — min-max normalization to [-1, 1] (Eq. 3):
        ``MR = 2 * (TR - min) / (max - min) - 1``.

    The resulting random expression replaces the difference of two random
    programs in GSM:  ``T_M(x) = T_P(x) + ms * <normalized TR>``.

    Parameters
    ----------
    s_R : torch.Tensor, shape (n,)
        Raw semantic vector of the random program TR evaluated on n points.
    mode : str
        "n1" (standardization) or "n2" (min-max normalization to [-1, 1]).
    epsilon : float, optional
        Numerical threshold to avoid division by zero. Default: 1e-8.
    stats : tuple or None, optional
        Pre-computed numeric constants from the training call, to be reused
        when normalizing test semantics (so the same scale factors apply):
        N1 → (mean_val, std_val); N2 → (min_val, max_val). If None they are
        computed from ``s_R`` (the training call).

    Returns
    -------
    tuple(torch.Tensor, tuple)
        - Normalized semantic vector. If the tree is degenerate (std ≈ 0 for
          N1, or max ≈ min for N2) a zero vector is returned and the global
          degenerate counter is incremented.
        - The numeric constants used, always returned so the caller can reuse
          them on test data. N1 → (mean, std); N2 → (min, max).
    """
    global _nm_degenerate_count

    if mode == "n2":
        # N2: min-max normalization to [-1, 1]  (Eqs. 2-3)
        if stats is not None:
            min_val, max_val = stats
        else:
            min_val = s_R.min()
            max_val = s_R.max()
        rng = max_val - min_val
        if not torch.isfinite(rng) or rng < epsilon:
            _nm_degenerate_count += 1
            return torch.zeros_like(s_R), (min_val, max_val)
        normalized = 2.0 * (s_R - min_val) / rng - 1.0
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized, (min_val, max_val)

    # N1: standardization  (Eq. 1)
    if stats is not None:
        mean_val, std_val = stats
    else:
        mean_val = s_R.mean()
        # correction=0 (population std) avoids nan for n=1 and is more stable
        std_val = s_R.std(correction=0)
    if not torch.isfinite(std_val) or std_val < epsilon:
        _nm_degenerate_count += 1
        return torch.zeros_like(s_R), (mean_val, std_val)
    normalized = (s_R - mean_val) / std_val
    # Replace any nan/inf from extreme tree outputs (e.g. inf-inf) with 0
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized, (mean_val, std_val)


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


def inflate_mutation(FUNCTIONS, TERMINALS,CONSTANTS,two_trees=True,operator="sum",single_tree_sigmoid=False,sig=False, oms: bool=False, nm: bool=False, nm_mode: str="n1"):
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
        Normalizes the raw semantics of a single random program before scaling
        by ms, replacing the difference of two random programs.
    nm_mode : str
        Which normalization to apply when ``nm`` is True (Bakurov et al. 2024):
        "n1" → standardization ``(TR - mean) / std`` (Eq. 1);
        "n2" → min-max normalization to [-1, 1] ``2*(TR - min)/(max - min) - 1``
        (Eq. 3).

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
            # For NM the trees must produce raw outputs so that normalization
            # acts on the true semantic range (T_R^raw in the formula).
            logistic_flag = not nm
            # getting two random trees
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=logistic_flag,
            )
            random_tree2 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=logistic_flag,
            )
            # adding the random trees to a list, to be used in the creation of a new block
            random_trees = [random_tree1, random_tree2]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(X_test, testing=True, logistic=logistic_flag)
                    for rt in random_trees
                ]                        
        else:
            # For NM the tree must produce raw outputs (no squashing function);
            # the bounding to [-1, 1] is provided entirely by the normalization.
            logistic_flag = (single_tree_sigmoid or sig) if not nm else False
            # getting one random tree
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=logistic_flag,
            )
            # adding the random tree to a list, to be used in the creation of a new block
            random_trees = [random_tree1]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(
                        X_test, testing=True, logistic=logistic_flag
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

        # compute raw semantic direction s_r
        if nm or oms:
            if two_trees:
                tr1, tr2 = random_trees
                s_r = torch.sub(tr1.train_semantics, tr2.train_semantics)
            else:  # one tree
                tr1, = random_trees
                if nm:
                    # NM uses T_R^raw directly; normalization provides the bounding.
                    s_r = tr1.train_semantics
                elif sig:
                    s_r = torch.sub(torch.mul(2, tr1.train_semantics), 1)
                else:
                    s_r = torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))))

        # If NM is active, normalize s_r now — before OMS so that OMS computes
        # the optimal step over the normalized direction.
        if nm:
            s_r_norm_train, train_stats = normalize_semantics(s_r, mode=nm_mode)
            s_r_for_oms = s_r_norm_train
        elif oms:
            s_r_for_oms = s_r

        # calculate the optimal mutation step value here
        if oms:
            s_r_inv = s_r_for_oms / (1e-7 + torch.mul(y_train.shape[0]), s_r_for_oms * s_r_for_oms) if s_r_for_oms.shape == torch.Size([1]) else s_r_for_oms / (1e-5 + torch.sum(s_r_for_oms * s_r_for_oms))
            ms = torch.vdot(s_r_inv.broadcast_to(y_train.shape), y_train - operator_f(individual.train_semantics, dim=0))
            ms = torch.clamp(ms, -100.0, 100.0)

            # Convert near-zero mutation steps to 0 to prevent overfitting and bloat
            if torch.abs(ms) < 0.1:
                global _oms_zero_transformations_count
                _oms_zero_transformations_count += 1
                ms = torch.tensor(0.0)

        # Build block semantics.
        # When NM is active (alone or combined with OMS), use the normalized direction.
        # When only OMS is active, or neither, use the standard variator.
        if nm:
            # Normalize test semantics reusing training statistics — never recompute on test.
            if operator == "sum":
                block_train_sem = torch.mul(ms, s_r_norm_train)
            else:
                block_train_sem = torch.add(1, torch.mul(ms, s_r_norm_train))
            if X_test is not None:
                if two_trees:
                    s_r_test = torch.sub(random_trees[0].test_semantics, random_trees[1].test_semantics)
                else:
                    # Raw test semantics, mirroring how s_r was built for train.
                    s_r_test = random_trees[0].test_semantics
                s_r_test_norm, _ = normalize_semantics(s_r_test, mode=nm_mode, stats=train_stats)
                block_test_sem = torch.mul(ms, s_r_test_norm) if operator == "sum" else torch.add(1, torch.mul(ms, s_r_test_norm))
            else:
                block_test_sem = None
        else:
            block_train_sem = variator(*random_trees, ms, testing=False)
            block_test_sem = variator(*random_trees, ms, testing=True) if X_test is not None else None

        ms_struct = ms

        # When NM is active, store a closure that captures training statistics so
        # that predict() and any future reconstruction apply the correct NM formula
        # instead of falling back to the ABS/SIG variator.
        structure_variator = (
            make_nm_variator(operator, train_stats, two_trees, mode=nm_mode)
            if nm
            else variator
        )

        new_block = Tree(
            structure=[structure_variator, *random_trees, ms_struct],
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
