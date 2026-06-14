import numpy as np
import pandas as pd

import os
import subprocess

from reproduce_results import DATASET_LOADERS
DATASET_NAMES = list(DATASET_LOADERS.keys())
from reproduce_plots import collate_variants


def _compile_tables_pdf(out_dir, version_tag):
    """Bundle the fitness + size LaTeX tables of one version into a single PDF.

    Wraps the two table fragments (which use a custom \\colorcelltesttwo macro,
    booktabs, multirow and \\cellcolor) in a standalone document and compiles
    it with tectonic. Best-effort: if tectonic is missing or compilation
    fails, the .tex files are still there.
    """
    wrapper = "\n".join([
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{tikz}",  # provides pgfmath for the macro below
        # The paper defines \colorcelltesttwo in its own preamble; reproduce a
        # close version here: arg1 = scaled improvement (positive = better),
        # mapped to a green cell background (clamped); arg2 = the value text.
        # \cv must be defined GLOBALLY (\xdef): inside a tabular cell colortbl
        # processes \cellcolor's argument in a context where a local \cv would
        # be lost ("Undefined control sequence" on green!\cv).
        r"\newcommand{\colorcelltesttwo}[2]{%",
        r"  \pgfmathtruncatemacro{\cvtmp}{max(0, min(85, #1*100))}%",
        r"  \xdef\cv{\cvtmp}%",
        r"  \cellcolor{green!\cv}#2%",
        r"}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        f"\\input{{fitness_table_{version_tag}.tex}}",
        f"\\input{{size_table_{version_tag}.tex}}",
        r"\end{document}",
    ])
    wrapper_path = os.path.join(out_dir, f"all_tables_{version_tag}.tex")
    with open(wrapper_path, "w") as fp:
        fp.write(wrapper)

    try:
        proc = subprocess.run(["tectonic", wrapper_path],
                              capture_output=True, text=True)
    except FileNotFoundError:
        print("  [WARNING] 'tectonic' not found; tables PDF not built "
              f"(the .tex files are in {out_dir}).")
        return
    pdf_path = os.path.join(out_dir, f"all_tables_{version_tag}.pdf")
    if proc.returncode != 0 or not os.path.exists(pdf_path):
        print("  [WARNING] tectonic failed to build the tables PDF:")
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        return
    print(f"Saved PDF: {pdf_path}")

# Orthogonal variant flags. NM is NOT included — it is a property of the
# SLIM version (SLIM+N1 / SLIM*N1), not a variant flag.
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
    ("PT", "OMS", "LS", "AS"):  "ALL",
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
                    color_val *= 0.25

                    # Format value
                    value_str = f"{value:.2f}".strip()
                    
                    # Wrap in colorcelltest
                    cell = f"\\colorcelltesttwo{{{color_val}}}{{{value_str}}}"
                    
                    # Apply bold formatting for best in class values
                    if (variant, individual) == best_in_problem:                        
                        cell = f"\\underline{{{cell}}}"
                    if variant == best_individual:
                        cell = f"\\textbf{{{cell}}}"
                    if variant == "BASE" and individual == "best_fitness":
                        cell = f"({cell})"
                    
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

def build_tables(slim_version):
    """Generate the LaTeX fitness/size tables for one SLIM version.

    Reads the per-version results from reproduced_results_2/<version>/ and
    writes table_latex/fitness_table_<TAG>.tex and size_table_<TAG>.tex.
    Callable both from this script's __main__ and from reproduce_plots.py.
    """
    version_dir = slim_version.replace("*", "x")              # folder name (matches reproduce_results.py)
    version_tag = slim_version.replace("SLIM+", "").replace("SLIM*", "x")  # short tag: N1, N2, ABS, ...

    results_path = os.path.abspath(
        f"./slim_gsgp/reproduced_results_2/{version_dir}"
    )

    print(f"Tabulating version : {slim_version}  (tag '{version_tag}')")
    print(f"Reading results    : {results_path}")

    full_df = collate_variants(results_path)

    full_medians = full_df.groupby(level=["variant", "dataset", "individual"]).median()

    fitness_caption = "Median fitness results. Rows for each dataset are ordered: Best Fitness, Best Size, Optimal Compromise. Values in bold, underline, and green represent the best in row, dataset and improvement over SLIM respectively."
    size_caption = "Median size results. Rows for each dataset are ordered: Best Fitness, Best Size, Optimal Compromise. Values in bold, underline, and green represent the best in row, dataset and improvement over SLIM respectively. "

    table_tex_fitness = custom_latex_converter(full_medians, "fitness", caption=fitness_caption, label="tab:rmse")
    table_tex_size = custom_latex_converter(full_medians, "size", caption=size_caption, label="tab:size")

    out_dir = "slim_gsgp/reproduced_plots/table_latex"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/fitness_table_{version_tag}.tex", "w") as fp:
        fp.write(table_tex_fitness)
    print(f"Saved: {out_dir}/fitness_table_{version_tag}.tex")

    with open(f"{out_dir}/size_table_{version_tag}.tex", "w") as fp:
        fp.write(table_tex_size)
    print(f"Saved: {out_dir}/size_table_{version_tag}.tex")

    # Bundle both tables into a single PDF (requires tectonic).
    _compile_tables_pdf(out_dir, version_tag)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Which SLIM version's results to tabulate. Must match the SLIM_VERSION
    # used in reproduce_results.py (results live in a per-version subfolder).
    # ------------------------------------------------------------------
    SLIM_VERSION = "SLIM+N1"
    build_tables(SLIM_VERSION)
        
