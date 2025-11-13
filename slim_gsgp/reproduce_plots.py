import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import os

from critdd import Diagram

from reproduce_results import DATASET_LOADERS
DATASET_NAMES = list(DATASET_LOADERS.keys())

VARIANTS_DICT = {
    tuple():                    "BASE",
    ("PT"):                     "BASE + PT",
    ("OMS"):                    "BASE + OMS",
    ("LS"):                     "BASE + LS",
    ("AS"):                     "BASE + AS",
    ("OMS", "LS", "AS"):        "ALL - PT",
    ("PT", "LS", "AS"):         "ALL - OMS",
    ("PT", "OMS", "AS"):        "ALL - LS",
    ("PT", "OMS", "LS"):        "ALL - AS",
    ("PT", "OMS", "LS", "AS"):  "ALL"
}

VARIANT_COLORS = {
    # BASE                       
    "BASE":                     "#000000",
    # ADD ONE                    
    "BASE + PT":                "#FF3333",
    "BASE + OMS":               "#33FF99",
    "BASE + LS":                "#9933FF",
    "BASE + AS":                "#3377FF",
    # REMOVE ONE                 
    "ALL - PT":                 "#FF3333",
    "ALL - OMS":                "#33FF99",
    "ALL - LS":                 "#9933FF",
    "ALL - AS":                 "#3377FF",
    # ALL                        
    "ALL":                      "#000000"
}

VARIANT_LINESTYLES = {
    # BASE                       
    "BASE":                     "--",
    # ADD ONE                    
    "BASE + PT":                "-",
    "BASE + OMS":               "-",
    "BASE + LS":                "-",
    "BASE + AS":                "-",
    # REMOVE ONE                 
    "ALL - PT":                 "--",
    "ALL - OMS":                "--",
    "ALL - LS":                 "--",
    "ALL - AS":                 "--",
    # ALL                        
    "ALL":                      "-"
}

VARIANT_MARKERS = {
    # BASE                       
    "BASE":                     "",
    # ADD ONE                    
    "BASE + PT":                "o",
    "BASE + OMS":               "^",
    "BASE + LS":                "d",
    "BASE + AS":                "s",
    # REMOVE ONE                 
    "ALL - PT":                 "o",
    "ALL - OMS":                "^",
    "ALL - LS":                 "d",
    "ALL - AS":                 "s",
    # ALL                        
    "ALL":                      "*"
}

# PLOT SETTINGS
MARKER_SIZE = 2.5
LINE_ALPHA = 0.85

# MATPLOTLIB LATEX SETTINGS
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
    'font.size' : 2,    
    'pgf.rcfonts': False,
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ])
})

FIGWIDTH_INCHES = 3 # 4.8
FIGSIZE = (FIGWIDTH_INCHES, (8/14) * FIGWIDTH_INCHES)

BASE_NAME="BASE"


def collate_variants(base_path: str):

    variant_dfs = {
        variant_name: pd.read_csv(f"{base_path}/{variant_name}.csv", index_col=[0, 1, 2])
        for variant_name in VARIANTS_DICT.values()
    }

    for variant_name in VARIANTS_DICT.values():
        variant_dfs[variant_name] = variant_dfs[variant_name].reset_index()
        variant_dfs[variant_name]["variant"] = variant_name
        
    full_df = pd.concat(variant_dfs.values(), ignore_index=True)

    full_df = full_df.set_index(['variant', 'dataset', 'individual', 'run'])

    full_df = full_df.sort_index()

    # get the means and medians for each dataset    

    return full_df

def create_line_plot(data, output_filepath, y_label: str, baseline_data):

    # Gorka's line plot

    plt.figure(figsize=FIGSIZE)        
    # fig, ax = plt.subplots(figsize=(14, 8))
    
    plt.axhline(0, color=VARIANT_COLORS[BASE_NAME], linestyle='--', linewidth=1.5, alpha=0.8, label=BASE_NAME)

    x_positions = np.arange(1, len(DATASET_NAMES) + 1)

    for variant_idx, variant_name in enumerate(VARIANTS_DICT.values()):
        if variant_name == BASE_NAME: continue

        variant_data = data.loc[variant_name]

        # i.e. halving the size / RMSE is a 50% improvement, halving again would be a 75% improvement 
        diff_pcs = ((baseline_data - variant_data) / baseline_data) * 100.0

        print(variant_name)
        print(VARIANT_LINESTYLES[variant_name])
        plt.plot(
            x_positions, diff_pcs,
            linewidth=1, markersize=MARKER_SIZE,
            marker=VARIANT_MARKERS[variant_name],
            linestyle=VARIANT_LINESTYLES[variant_name],
            color=VARIANT_COLORS[variant_name],
            label=variant_name, alpha=LINE_ALPHA
        )

    # plt.xlabel("Dataset", fontsize=12, fontweight="bold")
    plt.ylabel(y_label, fontsize=10, fontweight="bold")

    dataset_names_formatted = [" ".join(sub.capitalize() for sub in name.split("_")) for name in DATASET_NAMES]
    plt.xticks(x_positions, dataset_names_formatted, weight = "bold", rotation=20, ha="right")
    # plt.xticks(x_positions, x_positions, weight = "bold")
    plt.xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

    plt.grid(True, alpha=0.3)
    plt.legend

    plt.legend(loc='best', fontsize=9, prop={'weight': 'bold'})
    plt.tight_layout()

    # plt.savefig(f"{output_filepath}.png", dpi=300, bbox_inches="tight")

    plt.savefig(f"{output_filepath}.pgf", format="pgf", bbox_inches="tight")

    # export LaTeX:
    # ...
    
    return plt.gcf() # Return current figure instead of plt module



def produce_cdd(df_cdd, metric: str, filepath: str):

    arr_cdd = df_cdd[metric].to_numpy()

    # figure out if any columns are the same and remove as necessary,
    # noting the difference with new names
    variants_same = {}
    variants_removed = []
    print(list(enumerate(df_cdd[metric].columns)))
    for variant_1_idx, variant_1 in enumerate(df_cdd[metric].columns):
        for variant_2_idx, variant_2 in enumerate(df_cdd[metric].columns):
            if variant_2_idx <= variant_1_idx: continue
            if (variant_1_idx, variant_1) in variants_removed: continue
            if (variant_2_idx, variant_2) in variants_removed: continue

            if np.linalg.norm(arr_cdd[:, variant_1_idx] - arr_cdd[:, variant_2_idx]) < 1e-5:
                if not variant_1 in variants_same.keys():
                    variants_same[variant_1] = [variant_2]
                else:
                    variants_same[variant_1] += [variant_2]

                print(f"removing: {(variant_2_idx, variant_2)}")
                variants_removed += [(variant_2_idx, variant_2)]

    # change variant names based on those that are the same
    new_variant_names = [
        " \& ".join([variant_name] + variants_same[variant_name]) 
        if variant_name in variants_same.keys() else variant_name
        for variant_idx, variant_name in enumerate(df_cdd[metric].columns) if not (variant_idx, variant_name) in variants_removed
    ]            

    # new_variant_names = [
    #     " & ".join([first_variant_name] + same_list)
    #     for first_variant_name, same_list in variants_same.items()
    # ]

    print("variants removed:")
    print(variants_removed)
    print("because they were the same as:")
    print(variants_same)

    # modify data array to remove those columns
    keep_idxs = list(range(len(df_cdd[metric].columns)))
    for variant_idx, _ in variants_removed:
        print(variant_idx)
        keep_idxs.remove(variant_idx)

    print(f"shape: {arr_cdd.shape}")
    print(f"keep_idxs: {keep_idxs}")
    print(f"new_variant_names: {new_variant_names}")


    diagram = Diagram(
        arr_cdd[:, np.array(keep_idxs)],
        # treatment_names = df_cdd[metric].columns,
        treatment_names = new_variant_names,
        maximize_outcome = False        
    )

    diagram.to_file(
        # f"slim_gsgp/cdd_plots/{individual}/{metric}_cdd.tex",
        filepath,
        alpha = .05,
        adjustment = "holm",
        reverse_x = True,
        # axis_options = {"title": "critdd"},
        axis_options = {"width": "\\linewidth"}
    )

    print(f"<<<")

    return


if __name__ == "__main__":

    REPRODUCED_RESULTS_FILEPATH = os.path.abspath("./slim_gsgp/reproduced_results_2")
    base_plots_path = os.path.abspath("slim_gsgp/reproduced_plots")

    full_df = collate_variants(REPRODUCED_RESULTS_FILEPATH)

    full_means = full_df.groupby(level=["variant", "dataset", "individual"]).mean()
    full_medians = full_df.groupby(level=["variant", "dataset", "individual"]).median()    

    # create line plots:
    for metric in ["fitness", "size"]:
        y_label="RMSE Improvement %" if metric == "fitness" else "Size Improvement %"
        for individual in ["best_fitness", "best_size", "optimal_compromise"]:
            create_line_plot(
                full_means.xs(individual, level=2)[metric], 
                f"{base_plots_path}/mean/{individual}_{metric}",
                y_label=y_label,
                baseline_data=full_means.xs("best_fitness", level=2)[metric].loc[BASE_NAME]
            )
            create_line_plot(
                full_medians.xs(individual, level=2)[metric], 
                f"{base_plots_path}/median/{individual}_{metric}",
                y_label=y_label,
                baseline_data=full_medians.xs("best_fitness", level=2)[metric].loc[BASE_NAME]
            )

    # create cdds:    
    
    for metric in ["fitness", "size"]:
        df_cdd = full_medians.reset_index()
        
        df_cdd["variant_individual"] = df_cdd["variant"].astype(str) + " " + df_cdd["individual"].astype(str)
        df_cdd = df_cdd.pivot(index="dataset", columns="variant_individual")
        
        produce_cdd(df_cdd, metric, f"slim_gsgp/cdd_plots/full_{metric}_cdd.tex")
    for metric in ["fitness", "size"]:
        for individual in ["best_fitness", "best_size", "optimal_compromise"]:
            df_cdd = full_medians.xs(individual, level=2).reset_index()
            
            df_cdd = df_cdd.pivot(index="dataset", columns="variant")
            
            produce_cdd(df_cdd, metric, f"slim_gsgp/cdd_plots/{individual}/{metric}_cdd.tex")

        #for individual in ["best_fitness", "best_size", "optimal_compromise"]:
        # print(f"CDD Plot: {individual} : {metric}")
        # print(f">>>")

        """
        df_cdd = full_medians.reset_index()
        # df_cdd = full_medians.xs(individual, level=2).reset_index()
        df_cdd["variant_individual"] = df_cdd["variant"].astype(str) + " " + df_cdd["individual"].astype(str)
        df_cdd = df_cdd.pivot(index="dataset", columns="variant_individual")
        # df_cdd = df_cdd.pivot(index="dataset", columns="variant")


        arr_cdd = df_cdd[metric].to_numpy()

        # figure out if any columns are the same and remove as necessary,
        # noting the difference with new names
        variants_same = {}
        variants_removed = []
        print(list(enumerate(df_cdd[metric].columns)))
        for variant_1_idx, variant_1 in enumerate(df_cdd[metric].columns):
            for variant_2_idx, variant_2 in enumerate(df_cdd[metric].columns):
                if variant_2_idx <= variant_1_idx: continue
                if (variant_1_idx, variant_1) in variants_removed: continue
                if (variant_2_idx, variant_2) in variants_removed: continue

                if np.linalg.norm(arr_cdd[:, variant_1_idx] - arr_cdd[:, variant_2_idx]) < 1e-5:
                    if not variant_1 in variants_same.keys():
                        variants_same[variant_1] = [variant_2]
                    else:
                        variants_same[variant_1] += [variant_2]

                    print(f"removing: {(variant_2_idx, variant_2)}")
                    variants_removed += [(variant_2_idx, variant_2)]

        # change variant names based on those that are the same
        new_variant_names = [
            " \& ".join([variant_name] + variants_same[variant_name]) 
            if variant_name in variants_same.keys() else variant_name
            for variant_idx, variant_name in enumerate(df_cdd[metric].columns) if not (variant_idx, variant_name) in variants_removed
        ]            

        # new_variant_names = [
        #     " & ".join([first_variant_name] + same_list)
        #     for first_variant_name, same_list in variants_same.items()
        # ]

        print("variants removed:")
        print(variants_removed)
        print("because they were the same as:")
        print(variants_same)

        # modify data array to remove those columns
        keep_idxs = list(range(len(df_cdd[metric].columns)))
        for variant_idx, _ in variants_removed:
            print(variant_idx)
            keep_idxs.remove(variant_idx)

        print(f"shape: {arr_cdd.shape}")
        print(f"keep_idxs: {keep_idxs}")
        print(f"new_variant_names: {new_variant_names}")


        diagram = Diagram(
            arr_cdd[:, np.array(keep_idxs)],
            # treatment_names = df_cdd[metric].columns,
            treatment_names = new_variant_names,
            maximize_outcome = False
        )

        diagram.to_file(
            # f"slim_gsgp/cdd_plots/{individual}/{metric}_cdd.tex",
            f"slim_gsgp/cdd_plots/{metric}_cdd.tex",
            alpha = .05,
            adjustment = "holm",
            reverse_x = True,
            # axis_options = {"title": "critdd"},
        )

        print(f"<<<")

        """

else:
    REPRODUCED_RESULTS_FILEPATH = os.path.abspath("./reproduced_results_2")