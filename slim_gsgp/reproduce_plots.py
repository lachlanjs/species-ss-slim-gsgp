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

# PLOT SETTINGS
MARKER_SIZE = 10
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

FIGSIZE = (4.804, (8/14) * 4.804)


def collate_variants(base_path: str):

    variant_dfs = {
        variant_name: pd.read_csv(f"{REPRODUCED_RESULTS_FILEPATH}/{variant_name}.csv", index_col=[0, 1, 2])
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

    plt.figure(figsize=FIGSIZE)        
    # fig, ax = plt.subplots(figsize=(14, 8))

    baseline_data = data.loc[BASE]
    plt.axhline(0, color=VARIANT_COLORS[BASE], linestyle='--', linewidth=2, alpha=0.7, label=BASE)

    x_positions = np.arange(1, len(DATASET_NAMES) + 1)

    for variant_idx, variant_name in enumerate(VARIANTS_DICT.values()):
        if variant_name == BASE: continue

        variant_data = data.loc[variant_name]

        # i.e. halving the size / RMSE is a 50% improvement, halving again would be a 75% improvement 
        diff_pcs = ((baseline_data - variant_data) / baseline_data) * 100.0

        print(variant_name)
        print(VARIANT_LINESTYLES[variant_name])
        plt.plot(
            x_positions, diff_pcs, 
            linewidth=1, markersize=MARKER_SIZE,
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


def create_cdd(data, output_filepath, y_label: str, BASE="BASE"):

    # CDD    




    return



if __name__ == "__main__":

    REPRODUCED_RESULTS_FILEPATH = os.path.abspath("./slim_gsgp/reproduced_results_2")
    base_plots_path = os.path.abspath("slim_gsgp/reproduced_plots")

    full_df = collate_variants("slim_gsgp")

    full_means = full_df.groupby(level=["variant", "dataset", "individual"]).mean()
    full_medians = full_df.groupby(level=["variant", "dataset", "individual"]).median()    

    # create line plots:
    for metric in ["fitness", "size"]:
        y_label="RMSE Improvement %" if metric == "fitness" else "Size Improvement %"
        for individual in ["best_fitness", "best_size", "optimal_compromise"]:
            create_line_plot(
                full_means.xs(individual, level=2)[metric], 
                f"{base_plots_path}/mean/{individual}_{metric}",
                y_label=y_label
            )
            create_line_plot(
                full_medians.xs(individual, level=2)[metric], 
                f"{base_plots_path}/median/{individual}_{metric}",
                y_label=y_label
            )

    # create cdds:
    cdd_df = full_means.pivot()

    for metric in ["fitness", "size"]:
        diagram = Diagram(
            cdd_df.to_numpy(),
            treatment_names = cdd_df.columns,
            maximize_outcome = True
        )