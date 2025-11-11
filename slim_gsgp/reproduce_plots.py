import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# PLOT SETTINGS
MARKER_SIZE = 10
LINE_ALPHA = 0.85

# MATPLOTLIB LATEX SETTINGS



def collate_variants(base_path: str):

    variant_dfs = {
        variant_name: pd.read_csv(f"{base_path}/reproduced_results/{variant_name}.csv", index_col=[0, 1, 2])
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

def create_line_plot(data, output_filepath, y_label: str, BASE="BASE"):

    # Gorka's line plot

    plt.figure(figsize=(14, 8))        
    # fig, ax = plt.subplots(figsize=(14, 8))

    baseline_data = data.loc[BASE]
    plt.axhline(0, color=VARIANT_COLORS[BASE], linestyle='--', linewidth=2, alpha=0.7, label=BASE)

    x_positions = np.arange(1, len(DATASET_NAMES) + 1)

    for variant_idx, variant_name in enumerate(VARIANTS_DICT.values()):
        if variant_name == BASE: continue

        variant_data = data.loc[variant_name]

        diff_pcs = ((variant_data - baseline_data) / baseline_data) * 100.0

        print(variant_name)
        print(VARIANT_LINESTYLES[variant_name])
        plt.plot(
            x_positions, diff_pcs, 
            linewidth=2, markersize=MARKER_SIZE,
            linestyle=VARIANT_LINESTYLES[variant_name],
            color=VARIANT_COLORS[variant_name],
            label=variant_name, alpha=LINE_ALPHA
        )

    plt.xlabel("Dataset", fontsize=12, fontweight="bold")
    plt.ylabel(y_label, fontsize=12, fontweight="bold")

    plt.xticks(x_positions, DATASET_NAMES, weight = "bold", rotation=20, ha="right")
    # plt.xticks(x_positions, x_positions, weight = "bold")
    plt.xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

    plt.grid(True, alpha=0.3)
    plt.legend

    plt.legend(loc='best', fontsize=9, prop={'weight': 'bold'})
    plt.tight_layout()

    plt.savefig(f"{output_filepath}.png", dpi=300, bbox_inches="tight")

    # export LaTeX:
    # ...
    
    return plt.gcf() # Return current figure instead of plt module


def create_cdd(data, file_name):

    # CDD



    return



if __name__ == "__main__":

    full_df = collate_variants("slim_gsgp")

    full_means = full_df.groupby(level=["variant", "dataset", "individual"]).mean()
    full_medians = full_df.groupby(level=["variant", "dataset", "individual"]).median()
    

    base_plots_path = "slim_gsgp/reproduced_plots"

    # create line plots:
    for metric in ["fitness", "size"]:
        for individual in ["best_fitness", "best_size", "optimal_compromise"]:
            create_line_plot(
                full_means.xs(individual, level=2)[metric], 
                f"{base_plots_path}/mean/{individual}_{metric}.pgf",
                y_label="RMSE Improvement %"
            )
            create_line_plot(
                full_medians.xs(individual, level=2)[metric], 
                f"{base_plots_path}/median/{individual}_{metric}.pgf",
                y_label="Size Improvement %"
            )

    # create cdds:
    