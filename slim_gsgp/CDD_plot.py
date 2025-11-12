# this file generates the CDD plots
# using: https://github.com/mirkobunse/critdd?tab=readme-ov-file

from critdd import Diagram
import pandas as pd

# download example data
df_ = pd.read_excel("slim_gsgp/manual_set_results_test_fitness_size.xlsx")


# wrangle into relevant dataframes
variants = variants_to_show = [
    'VARIANT 1b', 
    'VARIANT 2', 
    'VARIANT 3b', 
    'VARIANT 4b', 
    'VARIANT 5', 
    'VARIANT 6b', 
    'VARIANT 7',
    'VARIANT 8'
]

cols = list(df_.columns)
variant_col_idxs = [cols.index(variant) for variant in variants]

variant_smallest_means_cols = [cols[idx + 1] for idx in variant_col_idxs]
variant_optimal_means_cols = [cols[idx + 3] for idx in variant_col_idxs]
variant_best_means_cols = [cols[idx + 5] for idx in variant_col_idxs]

smallest_means_df = df_[variant_smallest_means_cols]
smallest_means_df = smallest_means_df.rename(columns={df_.columns[col_idx+ 1]: variants[idx] for idx, col_idx in enumerate(variant_col_idxs)})
optimal_means_df = df_[variant_optimal_means_cols]
optimal_means_df = optimal_means_df.rename(columns={df_.columns[col_idx + 3]: variants[idx] for idx, col_idx in enumerate(variant_col_idxs)})
best_means_df = df_[variant_best_means_cols]
best_means_df = best_means_df.rename(columns={df_.columns[col_idx + 5]: variants[idx] for idx, col_idx in enumerate(variant_col_idxs)})


def wrangle_individual(individual_df: pd.DataFrame):

    df_fitness = individual_df[2:16].map(lambda v: float(v.split()[0]) if type(v) != float else 0.0)
    df_size = smallest_means_df[23:].map(lambda v: float(v.split()[0]) if type(v) != float else 0.0)    

    return df_fitness, df_size


def export_cdd_diagram(output_file: str, df: pd.DataFrame):

    # create a CD diagram from the Pandas DataFrame
    diagram = Diagram(
        df.to_numpy(),
        treatment_names = df.columns,
        maximize_outcome = False
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    diagram.average_ranks # the average rank of each treatment
    diagram.get_groups(alpha=.05, adjustment="holm")

    # export the diagram to a file
    diagram.to_file(
        output_file,
        as_document=True,
        alpha = .05,
        adjustment = "holm",
        reverse_x = False,
        axis_options = {"title": "critdd"},
    )

output_mapping_dict = {
    "smallest": smallest_means_df,
    "optimal": optimal_means_df,
    "best": best_means_df
}

for output_file_base_name, individual_df in output_mapping_dict.items():

    fitness_df, size_df = wrangle_individual(individual_df)
    
    fitness_filename = output_file_base_name + "_fitness.tex"
    size_filename = output_file_base_name + "_fitness.tex"

    # fitness
    export_cdd_diagram(f"cdd_plots/{fitness_filename}", fitness_df)
    # size
    export_cdd_diagram(f"cdd_plots/{size_filename}", size_df)