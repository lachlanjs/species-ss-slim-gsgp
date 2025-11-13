import numpy as np
import pandas as pd

import os

from reproduce_results import DATASET_LOADERS
DATASET_NAMES = list(DATASET_LOADERS.keys())
from reproduce_plots import collate_variants

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

BASE_NAME="BASE"

color_cell_test_2 = """

"""

def custom_latex_converter(df, metric: str, caption="Median test RMSE", label="tab:rmse"):
    """
    Custom converter that generates LaTeX with your exact specifications.
    
    This approach gives you:
    - Full control over LaTeX commands
    - Custom \colorcelltest{} wrapper
    - Flexible bold logic
    - Exact table structure matching your collaborator's format
    
    Parameters:
    -----------
    df : pandas.DataFrame or Series with MultiIndex
        Structure: (Variant, Problem, Individual) -> value
    """
    
    # Convert Series to DataFrame if needed
    if isinstance(df, pd.Series):
        df = df.to_frame(name='value')
    
    # Get the structure    
    variants = VARIANTS_DICT.values()
    problems = DATASET_NAMES

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append(r"\tiny")
    
    # Column specification
    col_spec = "l" + "c" * len(variants)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append(r"\toprule")
    latex_lines.append("")
    
    # Header
    header = "\\textbf{Problem} & " + " & ".join([f"\\textbf{{{v}}}" for v in variants]) + r"\\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")
    
    # Data rows
    for i, problem in enumerate(problems):
        latex_lines.append(f"\\multirow{{3}}{{*}}{{{i + 1}}}")

        best_in_problem = df.xs(problem, level=1)[metric].idxmin()
        print(f"best in: {problem} is {best_in_problem}")
        
        for individual in ["best_fitness", "best_size", "optimal_compromise"]:

            best_individual = df.xs(problem, level=1).xs(individual, level=1)[metric].idxmin()
            print(f"best {individual}: {best_individual}")

            row_parts = []
            for variant in variants:
                try:
                    # Get value from DataFrame                    
                    value = df.loc[(variant, problem, individual), metric]                    

                    # get baseline value
                    base_value = df.loc[("BASE", problem, "best_fitness"), metric]
                                        
                    color_val = (base_value - value) / base_value # % improvement from baseline
                    color_val *= 0.5

                    # Format value
                    value_str = f"{value:.2f}"
                    
                    # Wrap in colorcelltest
                    cell = f"\\colorcelltesttwo{{{color_val}}}{{{value_str}}}"
                    
                    # Apply bold formatting for best in class values
                    if (variant, individual) == best_in_problem:                        
                        cell = f"\\textbf{{{cell}}}"
                    if variant == best_individual:
                        cell = f"\\textit{{{cell}}}"
                    if variant == "BASE" and individual == "best_fitness":
                        cell = f"\\underline{{{cell}}}"
                    
                    row_parts.append(cell)
                except (KeyError, IndexError):
                    row_parts.append("")
            
            # Format the row
            if individual == "best_fitness":
                latex_lines.append("   ")
                latex_lines.append(" & " + " & ".join(row_parts) + r"  \\")
            else:
                latex_lines.append(" & " + " & ".join(row_parts) + r"  \\")
        
        # Add midrule between problems
        if i < len(problems) - 1:
            latex_lines.append(r" \midrule")
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines)

if __name__ == "__main__":

    REPRODUCED_RESULTS_FILEPATH = os.path.abspath("./slim_gsgp/reproduced_results_2")
    base_plots_path = os.path.abspath("slim_gsgp/reproduced_plots")

    full_df = collate_variants(REPRODUCED_RESULTS_FILEPATH)

    full_means = full_df.groupby(level=["variant", "dataset", "individual"]).mean()
    full_medians = full_df.groupby(level=["variant", "dataset", "individual"]).median()    

    table_tex_fitness = custom_latex_converter(full_medians, "fitness", caption = "Median test RMSE", label="tab:rmse")
    table_tex_size = custom_latex_converter(full_medians, "size", caption = "Median test RMSE", label="tab:rmse")

    with open("slim_gsgp/table_latex/fitness_table.tex", "w") as fp:
        fp.write(table_tex_fitness)

    with open("slim_gsgp/table_latex/size_table.tex", "w") as fp:
        fp.write(table_tex_size)
        
