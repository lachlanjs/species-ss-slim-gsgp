# this file is intended to reproduce the results

from multiprocessing import Pool, Manager
import time
import random
import sys
import signal

import pandas as pd

from datasets.data_loader import (
    load_airfoil, load_bike_sharing, load_bioav, load_boston, load_breast_cancer,
    load_concrete_slump, load_concrete_strength, load_diabetes, load_efficiency_cooling,
    load_efficiency_heating, load_forest_fires, load_istanbul, load_parkinson_updrs,
    load_ppb, load_resid_build_sale_price
)

from main_slim import slim
from utils.utils import train_test_split
from evaluators.fitness_functions import rmse

import os

DATASET_LOADERS = {
    'airfoil': load_airfoil,                    # Dataset 1
    'bike_sharing': load_bike_sharing,          # Dataset 2
    'bioavailability': load_bioav,              # Dataset 3
    'boston': load_boston,                      # Dataset 4
    'breast_cancer': load_breast_cancer,        # Dataset 5
    'concrete_slump': load_concrete_slump,      # Dataset 6
    'concrete_strength': load_concrete_strength,# Dataset 7
    'diabetes': load_diabetes,                  # Dataset 8
    'efficiency_cooling': load_efficiency_cooling,  # Dataset 9
    'efficiency_heating': load_efficiency_heating,  # Dataset 10
    'forest_fires': load_forest_fires,          # Dataset 11
    # 'istanbul': load_istanbul,                # Dataset 12 - COMMENTED OUT (uncomment to re-enable)
    'parkinson_updrs': load_parkinson_updrs,    # Dataset 13
    'ppb': load_ppb,                            # Dataset 14
    'resid_build_sale_price': load_resid_build_sale_price  # Dataset 15
}

DATASETS = list(DATASET_LOADERS.keys())

VARIANTS_DICT = {
    tuple():                "BASE",
    ("PT"):                 "BASE + PT",
    ("OMS"):                "BASE + OMS",
    ("LS"):                 "BASE + LS",
    ("AS"):                 "BASE + AS",
    ("OMS", "LS", "AS"):    "ALL - PT",
    ("PT", "LS", "AS"):     "ALL - OMS",
    ("PT", "OMS", "AS"):    "ALL - LS",
    ("PT", "OMS", "LS"):    "ALL - AS",
    ("PT", "OMS", "LS", "AS"):    "ALL"
}

BASE_ALGO_PARAMS = {
    'verbose': 0,
    'slim_version': "SLIM+ABS",
    'pop_size': 100,
    'n_iter': 100,
    'ms_lower': 0,
    'ms_upper': 1,
    'p_inflate': 0.5,
    'reconstruct': True    
}

DATASET_ITERATIONS = 15

UPDATE_TIME_INTERVAL = 2.0

def process_individual(individual, X_val, y_val, X_test, y_test):

    individual.calculate_semantics(X_val, testing=True)
    individual.evaluate(rmse, y_val, testing=True, operator="sum")
    predictions_bf = individual.predict(X_test)

    test_rmse_bf = float(rmse(y_true=y_test, y_pred=predictions_bf))    
    size = int(individual.nodes_count)

    return test_rmse_bf, size

def get_result_dataset(variant_tuple: tuple[str], dataset_name: str, seed: int):

    # load data:
    X, y = DATASET_LOADERS[dataset_name](X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=seed)

    # get the variant specifics:
    use_pt =    "PT" in variant_tuple
    use_oms =   "OMS" in variant_tuple
    use_ls =    "LS" in variant_tuple
    use_as =    "AS" in variant_tuple

    algo_params = {
        **BASE_ALGO_PARAMS,
        **{
            "seed": seed,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_val,
            "y_test": y_val,
            "dataset_name": dataset_name,
            "oms": use_oms,
            "linear_scaling": use_ls,
            "use_simplification": use_as            
        },
        **({} if not use_pt else {
            "tournament_type": "pareto",
            "tournament_size": 5,
            "multi_obj_attrs": ["fitness", "size"]
        })
    }

    results = slim(**algo_params)

    metrics = ["fitness", "size"]

    def process_helper(individual): 
        return process_individual(individual, X_val, y_val, X_test, y_test)
    
    result_dict = {
        "best_fitness":         dict(zip(metrics, process_helper(results.best_fitness))),
        "best_size":            dict(zip(metrics, process_helper(results.smallest))),
        "optimal_compromise":   dict(zip(metrics, process_helper(results.best_normalized)))
    }

    return result_dict

def get_results(args):    

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # extract arguments
    (variant_tuple, variant_name), progress_dict, process_id = args    

    # print start message
    print(f"Variant {variant_name} started with proc id: {process_id}")

    rows = []
    
    for dataset_id, dataset_name in enumerate(DATASETS):

        for seed in range(DATASET_ITERATIONS):
            
            result = get_result_dataset(variant_tuple, dataset_name, seed)

            for individual_name in ["best_fitness", "best_size", "optimal_compromise"]:
                row = {
                    'dataset': dataset_name,
                    'individual': individual_name,
                    'run': seed,
                    "fitness": result[individual_name]["fitness"],
                    "size": result[individual_name]["size"]
                }

                rows.append(row)
            
            # update progress
            progress_dict[process_id] = f"{dataset_id}/{len(DATASETS)} : {seed}/{DATASET_ITERATIONS}"

    results_df = pd.DataFrame(rows)

    # Set hierarchical index
    results_df = results_df.set_index(['dataset', 'individual', 'run'])
    
    # Sort index for better organization
    results_df = results_df.sort_index()

    # save the datasets:
    base_output_file_path = f"{REPRODUCED_RESULTS_FILEPATH}/{variant_name}"

    # csv:
    try:
        results_df.to_csv(base_output_file_path + ".csv")
    except Exception as e:
        print(f"Error with {variant_name} with id {process_id} writing to {base_output_file_path}.csv")
        print(e)

    # xlsx:
    # results_df.to_excel(base_output_file_path + ".xlsx")
    
    print(f"Variant {variant_name} with proc id: {process_id} completed!")

    return process_id

def print_progress(progress_dict, total_processes):
    """Helper to print current progress"""
    status = []
    for i in range(total_processes):
        completed = progress_dict.get(i, 0)
        status.append(f"Config {i}: {completed:15}")
    print(" | ".join(status))


if __name__ == '__main__':

    # note: must be abs path or child processes might write to the wrong path
    REPRODUCED_RESULTS_FILEPATH = os.path.abspath("./slim_gsgp/reproduced_results_2")
    
    print(f"Starting {len(VARIANTS_DICT)} experiments")
    
    # create shared dictionary for progress tracking
    with Manager() as manager:

        progress_dict = manager.dict()
        
        # initialize progress for each process
        for i in range(len(VARIANTS_DICT)):
            progress_dict[i] = 0
        
        # Prepare arguments: (config, progress_dict, process_id)
        args = [(config, progress_dict, i) for i, config in enumerate(VARIANTS_DICT.items())]        
        
        with Pool(processes=2) as pool:
            try: 
            
                # start async so we can monitor progress
                result = pool.map_async(get_results, args)
                
                # Monitor progress while running
                print(f"Progress updates (every {UPDATE_TIME_INTERVAL} seconds):")
                print("-" * 80)
                while not result.ready():
                    print_progress(progress_dict, len(VARIANTS_DICT))
                    result.wait(timeout=UPDATE_TIME_INTERVAL)
                
                # Final progress report
                print_progress(progress_dict, len(VARIANTS_DICT))
                print("-" * 80)
            except KeyboardInterrupt:
                print("\n\n Ctrl+C detected! Terminating all processes...")
                pool.terminate()  # immediately stop all workers
                pool.join()       # wait for them to finish terminating
                print(" X All processes terminated X.")
                sys.exit(1)            
    
    print("\n✓ All experiments complete!")

