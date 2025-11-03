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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Try to import tikzplotlib for LaTeX export
try:
    import tikzplotlib
    TIKZ_AVAILABLE = True
except ImportError:
    TIKZ_AVAILABLE = False
    print("Warning: tikzplotlib not installed. Install with: pip install tikzplotlib")
    print("TikZ/LaTeX export will be skipped.\n")

def extract_mean_value(mean_str):
    """
    Extract the main value (median or mean) from a cell (not the one in parentheses).
    
    Args:
        mean_str: String containing value, possibly with value in parentheses
        
    Returns:
        float: The extracted value
    """
    if pd.isna(mean_str):
        return None
    
    mean_str = str(mean_str)
    match = re.match(r'^\s*([-+]?[0-9]*\.?[0-9]+)', mean_str)
    
    if match:
        return float(match.group(1))
    
    return None

def load_excel_data(excel_file="manual_set_results_test_fitness_size.xlsx", sheet_name=0, table_type='fitness'):
    """
    Load the results from Excel file.
    
    Args:
        excel_file: Path to the Excel file with results
        sheet_name: Sheet name or index to read (0 for first sheet)
        table_type: 'fitness' or 'size' - which table to load
        
    Returns:
        pandas.DataFrame: DataFrame with the results
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Results file '{excel_file}' not found.")
    
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=[0, 1, 2])
        print(f"Loaded Excel file: {excel_file}")
        print(f"Table type: {table_type.upper()}")
        
        first_col = df.iloc[:, 0]
        
        # Find the row index where the second table starts
        model_size_idx = None
        for idx, val in enumerate(first_col):
            if pd.notna(val) and 'MODEL SIZE' in str(val).upper():
                model_size_idx = idx
                break
        
        # Select the appropriate table based on table_type
        if table_type.lower() == 'size':
            if model_size_idx is not None:
                df = df.iloc[model_size_idx:].copy()
                print(f"Selected MODEL SIZE table")
            else:
                raise ValueError("MODEL SIZE table not found in the Excel file")
        else:
            if model_size_idx is not None:
                df = df.iloc[:model_size_idx].copy()
                print(f"Selected TEST FITNESS table")
        
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

def extract_all_variants_data(df, model_type):
    """
    Extract median values for all variants for a specific model type.
    
    Args:
        df: DataFrame with 3-level multi-index columns
        model_type: 'Smallest Model', 'Optimal Compromise', or 'Best Fitness'
        
    Returns:
        dict: { 'VARIANT 1': {dataset_num: median, ...}, 'VARIANT 2': {...}, ... }
    """
    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in the Excel sheet")
    
    # Find all unique variants in the first level of columns
    variants = set()
    for c in cols:
        variant_name = str(c[0]).strip()
        if variant_name.startswith('VARIANT'):
            variants.add(variant_name)
    
    variants = sorted(variants, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    print(f"Found variants: {variants}")
    
    results = {}
    
    for variant_name in variants:
        # Look for the median column for this variant and model type
        median_col = None
        for c in cols:
            if (str(c[0]).strip() == variant_name and 
                str(c[1]).strip() == model_type and 
                'Median' in str(c[2])):
                median_col = c
                break
        
        if median_col is None:
            print(f"  Warning: No Median column found for {variant_name} - {model_type}")
            continue
        
        variant_data = {}
        
        # Extract dataset numbers and median values
        for idx, row in df.iterrows():
            first_cell = row.iloc[0]
            if pd.isna(first_cell):
                continue
            ds_name = str(first_cell)
            m = re.search(r'Dataset\s*(\d+)', ds_name, re.IGNORECASE)
            if not m:
                continue
            ds_num = int(m.group(1))
            
            raw = row[median_col]
            median_val = extract_mean_value(raw)
            if median_val is not None:
                variant_data[ds_num] = median_val
        
        results[variant_name] = variant_data
        print(f"  {variant_name}: {len(variant_data)} datasets")
    
    return results

def create_individual_comparison_plot(all_variants_data, model_type, table_type, baseline_variant='VARIANT 20'):
    """
    Create a plot comparing all variants for a specific model type.
    The baseline (VARIANT 20) is shown as the zero line.
    Other variants are shown as differences from baseline.
    
    Args:
        all_variants_data: dict with variant data
        model_type: 'Smallest Model', 'Optimal Compromise', or 'Best Fitness'
        table_type: 'fitness' or 'size'
        baseline_variant: The baseline variant (default: 'VARIANT 20')
    """
    # Mapping of variant names for legend
    variant_labels = {
        'VARIANT 20': 'SLIM-GSGP (Baseline)',
        'VARIANT 1': 'OMS',
        'VARIANT 1b': 'OMS (OMS=0)',
        'VARIANT 2': 'LS',
        'VARIANT 3': 'OMS + LS',
        'VARIANT 3b': 'OMS + LS (OMS=0)',
        'VARIANT 4': 'OMS + PT',
        'VARIANT 4b': 'OMS + PT (OMS=0)',
        'VARIANT 5': 'LS + PT',
        'VARIANT 6': 'OMS + PT + AS',
        'VARIANT 6b': 'OMS + PT + AS (OMS=0)',
        'VARIANT 7': 'LS + PT + AS'
    }
    
    plt.figure(figsize=(14, 8))
    
    # Define colors for different variants (cycle through if more variants)
    color_palette = plt.cm.tab20.colors
    
    baseline_data = all_variants_data.get(baseline_variant, {})
    
    if not baseline_data:
        print(f"Warning: No baseline data found for {baseline_variant}")
        return None
    
    # Draw zero reference line (baseline)
    baseline_label = variant_labels.get(baseline_variant, baseline_variant)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label=baseline_label)
    
    # Filter to show only original 7 variants (1-7, excluding 'b' variants)
    variants_to_show = ['VARIANT 1', 'VARIANT 2', 'VARIANT 3', 'VARIANT 4', 'VARIANT 5', 'VARIANT 6', 'VARIANT 7']
    
    # Plot each variant (except baseline) as difference from baseline
    variant_idx = 0
    for variant_name in sorted(all_variants_data.keys(), 
                               key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
        if variant_name == baseline_variant:
            continue
        
        # Skip variants not in the filter list
        if variant_name not in variants_to_show:
            continue
        
        variant_data = all_variants_data[variant_name]
        
        datasets = []
        differences = []
        
        # Calculate differences for datasets 1-15
        for ds_num in range(1, 16):
            baseline_val = baseline_data.get(ds_num, None)
            variant_val = variant_data.get(ds_num, None)
            
            if baseline_val is not None and variant_val is not None:
                datasets.append(ds_num)
                # For size, calculate percentage difference; for fitness, absolute difference
                if table_type.lower() == 'size':
                    diff = ((variant_val - baseline_val) / baseline_val) * 100
                else:
                    diff = variant_val - baseline_val
                differences.append(diff)
        
        if datasets:
            # Use default color palette
            color = color_palette[variant_idx % len(color_palette)]
            
            variant_label = variant_labels.get(variant_name, variant_name)
            
            plt.plot(datasets, differences,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=variant_label,
                    alpha=0.85)
            
            variant_idx += 1
    
    plt.xlabel('Dataset Number', fontsize=12, fontweight='bold')
    
    # Use "RMSE" for fitness labels, and "Reduction in size (%)" for size
    if table_type.lower() == 'fitness':
        ylabel = 'RMSE'
        title_metric = "RMSE"
    else:
        ylabel = 'Reduction in size (%)'
        title_metric = "Size %"
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    
    model_type_short = model_type.replace(' ', '_').lower()
    # plt.title(f'{model_type} - All Variants Comparison ({title_metric})', 
    #          fontsize=14, fontweight='bold')
    
    plt.xticks(range(1, 16), weight='bold')
    plt.xlim(0.5, 15.5)
    plt.grid(True, alpha=0.3)
    
    # Place legend inside the plot area with bold text
    plt.legend(loc='best', fontsize=9, prop={'weight': 'bold'})
    plt.tight_layout()
    
    return plt.gcf()  # Return current figure instead of plt module

def print_summary_table(all_variants_data, model_type, table_type='fitness', baseline_variant='VARIANT 20'):
    """
    Print a summary table showing values for all variants and datasets.
    """
    print("\n" + "="*100)
    print(f"SUMMARY TABLE - {model_type} ({table_type.upper()})")
    print("="*100)
    
    # Get all datasets
    all_datasets = set()
    for variant_data in all_variants_data.values():
        all_datasets.update(variant_data.keys())
    all_datasets = sorted(all_datasets)
    
    # Get all variants sorted
    variants = sorted(all_variants_data.keys(),
                     key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    
    # Print header
    header = f"{'Dataset':<12}"
    for variant in variants:
        header += f"{variant:<15}"
    print(header)
    print("-" * len(header))
    
    # Print values for each dataset
    baseline_data = all_variants_data.get(baseline_variant, {})
    
    for ds_num in all_datasets:
        row = f"Dataset {ds_num:<4}"
        
        for variant in variants:
            val = all_variants_data[variant].get(ds_num, None)
            if val is not None:
                if variant == baseline_variant:
                    row += f"{val:<15.4f}"
                else:
                    # Show difference from baseline
                    baseline_val = baseline_data.get(ds_num, None)
                    if baseline_val is not None:
                        if table_type.lower() == 'size':
                            # Show percentage difference for size
                            diff_pct = ((val - baseline_val) / baseline_val) * 100
                            row += f"{val:.1f}({diff_pct:+.1f}%)"[:15].ljust(15)
                        else:
                            # Show absolute difference for fitness
                            diff = val - baseline_val
                            row += f"{val:.4f}({diff:+.2f})"[:15].ljust(15)
                    else:
                        row += f"{val:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        
        print(row)

def main(excel_file="manual_set_results_test_fitness_size.xlsx", output_dir="plots_by_individual"):
    """
    Main function to generate all 6 comparison plots.
    
    Args:
        excel_file: Path to the Excel file with results
        output_dir: Directory to save the output plots
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        model_types = ['Smallest Model', 'Optimal Compromise', 'Best Fitness']
        table_types = ['fitness', 'size']
        
        plot_count = 0
        
        for table_type in table_types:
            print(f"\n{'='*100}")
            print(f"Processing {table_type.upper()} data")
            print(f"{'='*100}")
            
            # Load data for this table type
            df = load_excel_data(excel_file, table_type=table_type)
            
            for model_type in model_types:
                print(f"\n{'-'*100}")
                print(f"Processing: {model_type}")
                print(f"{'-'*100}")
                
                # Extract data for all variants for this model type
                all_variants_data = extract_all_variants_data(df, model_type)
                
                if not all_variants_data:
                    print(f"No data found for {model_type}")
                    continue
                
                # Print summary table
                print_summary_table(all_variants_data, model_type, table_type)
                
                # Create and save plot
                fig = create_individual_comparison_plot(all_variants_data, model_type, table_type)
                
                if fig is not None:
                    # Generate filename
                    model_type_short = model_type.replace(' ', '_').lower()
                    output_file_png = os.path.join(output_dir, f"{model_type_short}_{table_type}_comparison.png")
                    output_file_tex = os.path.join(output_dir, f"{model_type_short}_{table_type}_comparison.tex")
                    
                    # Save as PNG
                    fig.savefig(output_file_png, dpi=300, bbox_inches='tight')
                    print(f"\nPlot saved: {output_file_png}")
                    
                    # Save as TikZ/LaTeX if available
                    if TIKZ_AVAILABLE:
                        try:
                            tikzplotlib.save(output_file_tex,
                                           figureheight='8cm',
                                           figurewidth='14cm',
                                           strict=False)
                            print(f"TikZ saved: {output_file_tex}")
                        except Exception as e:
                            print(f"Warning: Could not save TikZ file: {e}")
                    
                    plt.close(fig)
                    
                    plot_count += 1
        
        print(f"\n{'='*100}")
        print(f"SUCCESS! Generated {plot_count} plots in '{output_dir}' directory")
        if TIKZ_AVAILABLE:
            print(f"  - {plot_count} PNG files (.png)")
            print(f"  - {plot_count} TikZ/LaTeX files (.tex)")
        else:
            print(f"  - {plot_count} PNG files (.png)")
            print(f"  - Install tikzplotlib for TikZ/LaTeX export: pip install tikzplotlib")
        print(f"{'='*100}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    excel_file = "manual_set_results_test_fitness_size.xlsx"
    output_dir = "plots_by_individual"
    
    # Show help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage: python plot_by_individual_type.py [OUTPUT_DIR]")
        print()
        print("Generates 6 comparison plots:")
        print("  - 3 plots for FITNESS (one per model type)")
        print("  - 3 plots for SIZE (one per model type)")
        print()
        print("Each plot shows:")
        print("  - Zero line: VARIANT 20 baseline")
        print("  - Multiple lines: Other variants as differences from baseline")
        print("  - X-axis: Dataset numbers (1-15)")
        print("  - Y-axis: Difference from baseline")
        print()
        print("Parameters:")
        print("  OUTPUT_DIR  - Directory to save plots (default: 'plots_by_individual')")
        print()
        print("Examples:")
        print("  python plot_by_individual_type.py")
        print("    → Saves plots to 'plots_by_individual/' directory")
        print()
        print("  python plot_by_individual_type.py my_plots")
        print("    → Saves plots to 'my_plots/' directory")
        sys.exit(0)
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"Configuration:")
    print(f"  Excel file: {excel_file}")
    print(f"  Output directory: {output_dir}")
    print()
    
    main(excel_file=excel_file, output_dir=output_dir)
