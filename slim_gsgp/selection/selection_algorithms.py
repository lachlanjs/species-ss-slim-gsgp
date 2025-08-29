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
Selection operator implementation.
"""

import random

import numpy as np

import pandas as pd

def tournament_selection_min(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop):
        """
        Selects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts


def tournament_selection_max(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts

# --- keep

def tournament_selection(pool_size, minimization: bool):

    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.
    minimization : bool
        whether or not the objective is to minimize

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop):
        """
        Selects the individual with the highest (or lowest, depending on {minimization}) fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness * (-1 if minimization else 1) for ind in pool])]

    return ts    

def calculate_non_dominated(pop, attrs: list[str], minimization: bool):

    """
    Helper function. Calculates the non-dominated set of candidates in the population and returns this as a list

    A candidate is non-dominated if there is no other candidate which is better than it in terms of all attributes

    Parameters
        ----------
        pop : list[Individual]
            The population from which individuals are drawn.

        attrs : list[str]
            The attributes by which the non-dominated set will be calculated (i.e. fitness, size, depth, loss, etc.)            

        Returns
        -------
        list[int]
            The indices of the individuals in the non-dominated set # NOTE: could sort by the first attr
    """

    # construct a dataframe where each row contains:
    # - the idx of an individual
    # - its place in the ranking of each attribute (lower is better)

    non_dom_df = pd.DataFrame({
        "idx": range(len(pop)), # NOTE: maybe not necessary
        **{attr: [ind.__dict__[attr] * (1 if minimization else -1) for ind in pop]
           for attr in attrs}
    })    

    # print()
    # print("----")
    # print(non_dom_df)

    # carve away anything that is dominated
    for idx in range(len(pop)):
        if idx not in non_dom_df.index:
            continue

        # remove_idxs = non_dom_df.index
        remove_idxs = pd.Series([True for i in range(len(non_dom_df))])
        
        for attr in attrs:
            # print(f"{idx}'s {attr}: {non_dom_df[attr][idx]}")
            # print(f"remove_idxs: {remove_idxs}")
            # print(f"{non_dom_df[attr] < non_dom_df[attr][idx]}")
            # print()
            # remove_idxs = remove_idxs.intersection(non_dom_df[attr] < non_dom_df[attr][idx])
            remove_idxs = remove_idxs & (non_dom_df[attr] < non_dom_df[attr][idx])

        # print(f"remove_idxs: {remove_idxs}")

        # NOTE: we should never have idx in remove_idxs... could check this

        # non_dom_df = non_dom_df.iloc[non_dom_df.index.difference(remove_idxs)]
        
        non_dom_df = non_dom_df[~remove_idxs]

    # the remaining df contains the non-dominated set        
    non_dom_idxs = list(non_dom_df.index)    

    # print(f"non_dom_idxs: {non_dom_idxs}")

    return non_dom_idxs

def tournament_selection_pareto(pool_size, attrs: list[str], minimization: bool=True):

    """
    Returns a function that performs pareto-tournament selection to select individuals for crossover

    Parameters
    ----------
    pool_size : int
        Number of individuals forming a random subset from which the non-dominating set is calculated

    attrs : list[str]
        The attributes by which the non-dominated set will be calculated (i.e. fitness, size, depth, loss, etc.)            

    minimization : bool
        whether the objective is to minimize or maximize

    Returns
    -------
    Callable
        A function ('pts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            Random individual from the non-dominated set sample 
    Notes
    -----
    The returned function performs pareto tournament selection, which means a random sample of 
    pool_size individuals is taken and then the non-dominated candidates of this set are the 
    winners of the tournament

    Note that if all but one of the attributes constrain to a single value, then there will only be one
    candidate on the frontier, thus this reduces to tournament selection
    """

    def pts(pop):

        # get a random sample without replacement from the population
        rand_pop_sample = random.sample(pop.population, pool_size)

        # calculate the non-dominated set
        non_dom_idxs = calculate_non_dominated(rand_pop_sample, attrs, minimization)
        
        # take one individual from this set
        selected_idx = random.choice(non_dom_idxs)

        return pop.population[selected_idx]    

    return pts