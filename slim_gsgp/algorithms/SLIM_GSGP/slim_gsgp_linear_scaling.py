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
SLIM_GSGP with Linear Scaling Class for Evolutionary Computation using PyTorch.
"""

import random
import sys
import time

import numpy as np
import torch
from slim_gsgp.algorithms.GP.representations.tree import Tree as GP_Tree
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual_linear_scaling import IndividualLinearScaling
from slim_gsgp.algorithms.SLIM_GSGP.representations.population import Population
from slim_gsgp.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim_gsgp.utils.diversity import gsgp_pop_div_from_vectors
from slim_gsgp.utils.logger import logger
from slim_gsgp.utils.utils import verbose_reporter


class SLIM_GSGP_LinearScaling(SLIM_GSGP):
    """
    SLIM_GSGP with Linear Scaling support.
    Extends the base SLIM_GSGP class to use IndividualLinearScaling instead of Individual.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize SLIM_GSGP_LinearScaling.
        Filters out linear scaling specific parameters before calling parent constructor.
        """
        # Remove linear scaling specific parameters that aren't accepted by parent class
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_linear_scaling', 'linear_scaling']}
        super().__init__(*args, **filtered_kwargs)

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        run_info,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=None,
        max_depth=17,
        n_elites=1,
        reconstruct=True,
        n_jobs=1,
        **kwargs):
        """
        Solve the optimization problem using SLIM_GSGP with Linear Scaling.

        Parameters
        ----------
        X_train : array-like
            Training input data.
        X_test : array-like
            Testing input data.
        y_train : array-like
            Training output data.
        y_test : array-like
            Testing output data.
        curr_dataset : str or int
            Identifier for the current dataset.
        run_info : dict
            Information about the current run.
        n_iter : int
            Number of iterations. Default is 20.
        elitism : bool
            Whether elitism is used during evolution. Default is True.
        log : int or str
            Logging level (e.g., 0 for no logging, 1 for basic, etc.). Default is 0.
        verbose : int
            Verbosity level for logging outputs. Default is 0.
        test_elite : bool
            Whether elite individuals should be tested. Default is False.
        log_path : str
            File path for saving log outputs. Default is None.
        ffunction : function
            Fitness function used to evaluate individuals. Default is None.
        max_depth : int
            Maximum depth for the trees. Default is 17.
        n_elites : int
            Number of elite individuals to retain during selection. Default is True.
        reconstruct : bool
            Indicates if reconstruction of the solution is needed. Default is True.
        n_jobs : int
            Maximum number of concurrently running jobs for joblib parallelization. Default is 1.
        **kwargs : dict
            Additional keyword arguments (linear_scaling parameter will be ignored)

        """
        
        # Filter out linear scaling specific parameters that aren't used in this method
        # This allows the method to accept 'linear_scaling' parameter without errors
        kwargs.pop('linear_scaling', None)

        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting time count
        start = time.time()

        # creating the initial population with IndividualLinearScaling
        population = Population(
            [
                IndividualLinearScaling(
                    collection=[
                        Tree(
                            tree,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True,
                        )
                    ],
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )

        # calculating initial population semantics
        population.calculate_semantics(X_train)

        # evaluating the initial population (with raw outputs)
        population.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs)

        # DEBUG: Mostrar fitness ANTES de aplicar linear scaling
        print("\n" + "="*80, flush=True)
        print("=== FITNESS ANTES DE APLICAR LINEAR SCALING ===", flush=True)
        print("="*80, flush=True)
        fitness_values_before = []
        for i, individual in enumerate(population.population):
            print(f"Individuo {i+1:3d}: Fitness = {individual.fitness:.6f}, Nodos = {individual.nodes_count:3d}", flush=True)
            fitness_values_before.append(individual.fitness)
        
        print(f"\nEstadísticas ANTES del scaling:", flush=True)
        print(f"  Fitness mínimo:     {min(fitness_values_before):.6f}", flush=True)
        print(f"  Fitness máximo:     {max(fitness_values_before):.6f}", flush=True)
        print(f"  Fitness promedio:   {np.mean(fitness_values_before):.6f}", flush=True)
        print(f"  Desviación estándar: {np.std(fitness_values_before):.6f}", flush=True)
        print(f"  Mediana:            {np.median(fitness_values_before):.6f}", flush=True)
        print("="*80 + "\n", flush=True)

        # Calculate linear scaling for the initial population
        for individual in population.population:
            individual.calculate_linear_scaling(y_train)

        # Re-evaluate population with linear scaling applied
        for individual in population.population:
            # Apply linear scaling to predictions
            raw_prediction = torch.sum(individual.train_semantics, dim=0) if len(individual.train_semantics.shape) > 1 else individual.train_semantics
            scaled_prediction = individual.scaling_a + raw_prediction * individual.scaling_b
            # Recalculate fitness with scaled predictions
            individual.fitness = float(ffunction(y_train, scaled_prediction))

        # Update population fitness array after re-evaluation
        population.fit = [individual.fitness for individual in population.population]

        # DEBUG: Mostrar fitness DESPUÉS de aplicar linear scaling
        print("\n" + "="*80, flush=True)
        print("=== FITNESS DESPUÉS DE APLICAR LINEAR SCALING ===", flush=True)
        print("="*80, flush=True)
        fitness_values_after = []
        for i, individual in enumerate(population.population):
            print(f"Individuo {i+1:3d}: Fitness = {individual.fitness:.6f}, Nodos = {individual.nodes_count:3d}, "
                  f"Scaling: a={individual.scaling_a:.4f}, b={individual.scaling_b:.4f}", flush=True)
            fitness_values_after.append(individual.fitness)
        
        print(f"\nEstadísticas DESPUÉS del scaling:", flush=True)
        print(f"  Fitness mínimo:     {min(fitness_values_after):.6f}", flush=True)
        print(f"  Fitness máximo:     {max(fitness_values_after):.6f}", flush=True)
        print(f"  Fitness promedio:   {np.mean(fitness_values_after):.6f}", flush=True)
        print(f"  Desviación estándar: {np.std(fitness_values_after):.6f}", flush=True)
        print(f"  Mediana:            {np.median(fitness_values_after):.6f}", flush=True)
        
        # Mostrar comparación entre antes y después
        print(f"\n--- COMPARACIÓN ANTES vs DESPUÉS ---", flush=True)
        print(f"  Mejora promedio en fitness: {np.mean(fitness_values_before) - np.mean(fitness_values_after):.6f}", flush=True)
        print(f"  Mejor fitness antes:        {min(fitness_values_before):.6f}", flush=True)
        print(f"  Mejor fitness después:      {min(fitness_values_after):.6f}", flush=True)
        print(f"  Mejora del mejor:           {min(fitness_values_before) - min(fitness_values_after):.6f}", flush=True)
        
        print("="*80, flush=True)
        print("=== FIN DEBUG LINEAR SCALING ===", flush=True)
        print("="*80 + "\n", flush=True)

        end = time.time()

        # setting up the elite(s)
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # calculating the testing semantics and the elite's testing fitness if test_elite is true
        if test_elite:
            population.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(
                ffunction, y=y_test, testing=True, operator=self.operator
            )

        # logging the results based on the log level
        if log != 0:
            if log == 2:
                gen_diversity = (
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.sum(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        ),
                    )
                    if self.operator == "sum"
                    else gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.prod(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        )
                    )
                )
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    float(gen_diversity),
                    np.std(population.fit),
                    log,
                ]

            elif log == 3:
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            elif log == 4:
                gen_diversity = (
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.sum(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        ),
                    )
                    if self.operator == "sum"
                    else gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.prod(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        )
                    )
                )
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    float(gen_diversity),
                    np.std(population.fit),
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            else:

                add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

            logger(
                log_path,
                0,
                self.elite.fitness,
                end - start,
                float(population.nodes_count),
                additional_infos=add_info,
                run_info=run_info,
                seed=self.seed,
            )

        # displaying the results on console if verbose level is more than 0
        if verbose != 0:
            verbose_reporter(
                curr_dataset,
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.nodes_count,
            )

        # beginning the evolution process
        for it in range(1, n_iter + 1, 1):
            # starting an empty offspring population
            offs_pop, start = [], time.time()

            # adding the elite to the offspring population, if applicable
            if elitism:
                # Ensure elites are IndividualLinearScaling instances
                elite_copies = []
                for elite in self.elites:
                    if isinstance(elite, IndividualLinearScaling):
                        elite_copies.append(elite)
                    else:
                        # Convert elite to IndividualLinearScaling (shouldn't happen but just in case)
                        elite_ls = IndividualLinearScaling(
                            collection=elite.collection,
                            train_semantics=elite.train_semantics,
                            test_semantics=elite.test_semantics,
                            reconstruct=reconstruct
                        )
                        # Copy all attributes
                        for attr in ['nodes_collection', 'nodes_count', 'depth_collection', 'depth', 'size', 'fitness']:
                            if hasattr(elite, attr):
                                setattr(elite_ls, attr, getattr(elite, attr))
                        elite_copies.append(elite_ls)
                offs_pop.extend(elite_copies)

            # filling the offspring population
            while len(offs_pop) < self.pop_size:

                # choosing between crossover and mutation
                if random.random() > self.p_m:

                    # performing crossover
                    p1, p2 = self.selector(population, 2)
                    
                    # Note: For linear scaling, we typically don't use crossover
                    # but if we do, we need to handle it properly with IndividualLinearScaling
                    offspring = self.crossover(p1, p2, ms_lower=0, ms_upper=1, operator=self.operator,
                                               reconstruct=reconstruct)
                    
                    if isinstance(offspring, IndividualLinearScaling):
                        # Inherit linear scaling parameters from p1 (first parent)
                        offspring.scaling_a = p1.scaling_a
                        offspring.scaling_b = p1.scaling_b
                        offspring.use_linear_scaling = p1.use_linear_scaling
                        offs_pop.append(offspring)
                    else:
                        # Convert to IndividualLinearScaling and inherit scaling parameters
                        offspring_ls = IndividualLinearScaling(
                            collection=offspring.collection,
                            train_semantics=offspring.train_semantics,
                            test_semantics=offspring.test_semantics,
                            reconstruct=reconstruct
                        )
                        # Inherit linear scaling parameters from p1
                        offspring_ls.scaling_a = p1.scaling_a
                        offspring_ls.scaling_b = p1.scaling_b
                        offspring_ls.use_linear_scaling = p1.use_linear_scaling
                        offs_pop.append(offspring_ls)

                else:
                    # performing mutation
                    p = self.selector(population)

                    if random.random() < self.p_inflate:
                        ms_ = self.ms()
                        offspring = self.inflate_mutator(
                            p, ms_, X_train, 
                            max_depth=max_depth,
                            p_c=self.pi_init["p_c"],
                            X_test=X_test,
                            reconstruct=reconstruct
                        )
                    else:
                        # Check if parent has only one block before deflation
                        if p.size == 1:
                            # if copy parent is set to true, the parent who cannot be deflated will be copied as the offspring
                            if self.copy_parent:
                                offspring = IndividualLinearScaling(
                                    collection=p.collection if reconstruct else None,
                                    train_semantics=p.train_semantics,
                                    test_semantics=p.test_semantics,
                                    reconstruct=reconstruct
                                )
                                (
                                    offspring.nodes_collection,
                                    offspring.nodes_count,
                                    offspring.depth_collection,
                                    offspring.depth,
                                    offspring.size,
                                ) = (
                                    p.nodes_collection,
                                    p.nodes_count,
                                    p.depth_collection,
                                    p.depth,
                                    p.size,
                                )
                                # Inherit linear scaling parameters from parent
                                offspring.scaling_a = p.scaling_a
                                offspring.scaling_b = p.scaling_b
                                offspring.use_linear_scaling = p.use_linear_scaling
                            else:
                                # if we choose to not copy the parent, we inflate it instead
                                ms_ = self.ms()
                                offspring = self.inflate_mutator(
                                    p, ms_, X_train,
                                    max_depth=max_depth,
                                    p_c=self.pi_init["p_c"],
                                    X_test=X_test,
                                    reconstruct=reconstruct
                                )
                        else:
                            # if the size of the parent is more than 1, normal deflation can occur
                            offspring = self.deflate_mutator(p, reconstruct=reconstruct)

                    # Ensure offspring is IndividualLinearScaling and inherit scaling parameters
                    if not isinstance(offspring, IndividualLinearScaling):
                        offspring_ls = IndividualLinearScaling(
                            collection=offspring.collection,
                            train_semantics=offspring.train_semantics,
                            test_semantics=offspring.test_semantics,
                            reconstruct=reconstruct
                        )
                        # Inherit linear scaling parameters from parent
                        offspring_ls.scaling_a = p.scaling_a
                        offspring_ls.scaling_b = p.scaling_b
                        offspring_ls.use_linear_scaling = p.use_linear_scaling
                        offspring = offspring_ls
                    else:
                        # Already IndividualLinearScaling, inherit scaling parameters
                        offspring.scaling_a = p.scaling_a
                        offspring.scaling_b = p.scaling_b
                        offspring.use_linear_scaling = p.use_linear_scaling

                    offs_pop.append(offspring)

            # Calculate semantics and evaluate offspring
            offs_pop_obj = Population(offs_pop)
            offs_pop_obj.calculate_semantics(X_train)
            offs_pop_obj.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs)

            # Note: Linear scaling parameters are inherited from parents, no need to recalculate
            # But we need to re-evaluate fitness with linear scaling applied
            for individual in offs_pop_obj.population:
                if individual.use_linear_scaling:
                    # Apply linear scaling to predictions
                    raw_prediction = torch.sum(individual.train_semantics, dim=0) if len(individual.train_semantics.shape) > 1 else individual.train_semantics
                    scaled_prediction = individual.scaling_a + raw_prediction * individual.scaling_b
                    # Recalculate fitness with scaled predictions
                    individual.fitness = float(ffunction(y_train, scaled_prediction))

            # Update population fitness array after re-evaluation
            offs_pop_obj.fit = [individual.fitness for individual in offs_pop_obj.population]

            end = time.time()

            # setting up the elite(s)
            self.elites, self.elite = self.find_elit_func(offs_pop_obj, n_elites)

            # evaluating the elite on the test set if test_elite is true
            if test_elite:
                offs_pop_obj.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(
                    ffunction, y=y_test, testing=True, operator=self.operator
                )

            # updating the population
            population = offs_pop_obj

            # logging the results based on the log level
            if log != 0:
                if log == 2:
                    gen_diversity = (
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.sum(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            ),
                        )
                        if self.operator == "sum"
                        else gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.prod(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            )
                        )
                    )
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        float(gen_diversity),
                        np.std(population.fit),
                        log,
                    ]

                elif log == 3:
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        " ".join([str(ind.nodes_count) for ind in population.population]),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                elif log == 4:
                    gen_diversity = (
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.sum(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            ),
                        )
                        if self.operator == "sum"
                        else gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.prod(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            )
                        )
                    )
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        float(gen_diversity),
                        np.std(population.fit),
                        " ".join([str(ind.nodes_count) for ind in population.population]),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                else:

                    add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

                logger(
                    log_path,
                    it,
                    self.elite.fitness,
                    end - start,
                    float(population.nodes_count),
                    additional_infos=add_info,
                    run_info=run_info,
                    seed=self.seed,
                )

            # displaying the results on console if verbose level is more than 0
            if verbose != 0:
                verbose_reporter(
                    curr_dataset,
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.nodes_count,
                )
