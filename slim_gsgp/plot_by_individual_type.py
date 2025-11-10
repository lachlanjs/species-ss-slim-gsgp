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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Configure matplotlib PGF backend for LaTeX export
# This configuration allows PGF export without requiring LaTeX to be installed
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",  # Use pdflatex (most common)
    "pgf.rcfonts": False,          # Don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ])
})

LATEX_EXPORT_AVAILABLE = True

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
        'VARIANT 1b': 'OMS',
        'VARIANT 1c': 'OMS 0.5',
        'VARIANT 1d': 'OMS 1',
        'VARIANT 2': 'LS',
        'VARIANT 3': 'OMS + LS',
        'VARIANT 3b': 'OMS + LS',
        'VARIANT 4': 'OMS + PT',
        'VARIANT 4b': 'OMS + PT',
        'VARIANT 5': 'LS + PT',
        'VARIANT 6': 'OMS + PT + AS',
        'VARIANT 6b': 'OMS + PT + AS',
        'VARIANT 7': 'LS + PT + AS'
    }
    
    plt.figure(figsize=(14, 8))
    
    # Define fixed colors for each variant
    # Blue tones for OMS variants, Orange tones for LS variants, Green for OMS+LS
    # Darker colors for variants with more elements
    variant_colors = {
        'VARIANT 1': '#3399FF',    # OMS (1 element) - Light Blue
        'VARIANT 1b': '#3399FF',   # OMS 0.1 (1 element) - Light Blue
        'VARIANT 1c': '#3399FF',   # OMS 0.5 (1 element) - Light Blue
        'VARIANT 1d': '#3399FF',   # OMS 1 (1 element) - Light Blue
        'VARIANT 2': '#FF9944',    # LS (1 element) - Light Orange
        'VARIANT 3': '#66CC66',    # OMS + LS (2 elements) - Light Green
        'VARIANT 3b': '#66CC66',   # OMS + LS (OMS=0) (2 elements) - Light Green
        'VARIANT 4': '#0066CC',    # OMS + PT (2 elements) - Medium Blue
        'VARIANT 4b': '#0066CC',   # OMS + PT (OMS=0) (2 elements) - Medium Blue
        'VARIANT 5': '#FF6600',    # LS + PT (2 elements) - Medium Orange
        'VARIANT 6': '#003366',    # OMS + PT + AS (3 elements) - Dark Blue
        'VARIANT 6b': '#003366',   # OMS + PT + AS (OMS=0) (3 elements) - Dark Blue
        'VARIANT 7': '#CC4400',    # LS + PT + AS (3 elements) - Dark Orange
    }
    
    # Define different markers for each variant (for better B&W printing)
    # Rounded shapes for OMS variants, Triangular shapes for LS variants, Star for OMS+LS
    variant_markers = {
        'VARIANT 1': 'o',      # OMS - Circle (rounded)
        'VARIANT 1b': 'o',     # OMS - Circle (rounded)
        'VARIANT 1c': 'o',     # OMS - Circle (rounded)
        'VARIANT 1d': 'o',     # OMS - Circle (rounded)
        'VARIANT 2': '^',      # LS - Triangle up (triangular)
        'VARIANT 3': '*',      # OMS + LS - Star (combination symbol)
        'VARIANT 3b': '*',     # OMS + LS - Star (combination symbol)
        'VARIANT 4': 's',      # OMS + PT - Square (rounded corners)
        'VARIANT 4b': 's',     # OMS + PT - Square (rounded corners)
        'VARIANT 5': 'v',      # LS + PT - Triangle down (triangular)
        'VARIANT 6': 'p',      # OMS + PT + AS - Pentagon (rounded)
        'VARIANT 6b': 'p',     # OMS + PT + AS - Pentagon (rounded)
        'VARIANT 7': 'd',      # LS + PT + AS - Thin diamond (triangular)
    }
    
    baseline_data = all_variants_data.get(baseline_variant, {})
    
    if not baseline_data:
        print(f"Warning: No baseline data found for {baseline_variant}")
        return None
    
    # Draw zero reference line (baseline)
    baseline_label = variant_labels.get(baseline_variant, baseline_variant)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label=baseline_label)
    
    # Filter to show only original 7 variants (1-7, excluding 'b' variants)
    variants_to_show = ['VARIANT 1b', 'VARIANT 2', 'VARIANT 3b', 'VARIANT 4b', 'VARIANT 5', 'VARIANT 6b', 'VARIANT 7']
    
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
        
        # Calculate differences for datasets 1-15, excluding dataset 12 (istanbul)
        # NOTE: Dataset 12 'istanbul' is excluded. To re-enable, remove the 'if ds_num == 12: continue' line
        for ds_num in range(1, 16):
            if ds_num == 12:  # Skip dataset 12 (istanbul)
                continue
            
            baseline_val = baseline_data.get(ds_num, None)
            variant_val = variant_data.get(ds_num, None)
            
            if baseline_val is not None and variant_val is not None:
                datasets.append(ds_num)
                # Calculate percentage difference for both size and fitness
                diff = ((variant_val - baseline_val) / baseline_val) * 100
                differences.append(diff)
        
        if datasets:
            # Use fixed color and marker for each variant
            color = variant_colors.get(variant_name, '#000000')  # Default to black if not defined
            marker = variant_markers.get(variant_name, 'o')  # Default to circle if not defined
            
            variant_label = variant_labels.get(variant_name, variant_name)
            
            plt.plot(datasets, differences,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=8,
                    label=variant_label,
                    alpha=0.85)
            
            variant_idx += 1
    
    plt.xlabel('Dataset Number', fontsize=12, fontweight='bold')
    
    # Use percentage for both fitness and size
    ylabel = 'Reduction (%)'
    if table_type.lower() == 'fitness':
        title_metric = "RMSE %"
    else:
        title_metric = "Size %"
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    
    model_type_short = model_type.replace(' ', '_').lower()
    # plt.title(f'{model_type} - All Variants Comparison ({title_metric})', 
    #          fontsize=14, fontweight='bold')
    
    # Adjust x-axis to show datasets 1-15 excluding dataset 12 (istanbul)
    # NOTE: To re-enable dataset 12, change back to range(1, 16) and remove the list comprehension filter
    dataset_labels = [i for i in range(1, 16) if i != 12]  # [1,2,3,4,5,6,7,8,9,10,11,13,14,15]
    plt.xticks(dataset_labels, weight='bold')
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
                    # Show percentage difference for both fitness and size
                    baseline_val = baseline_data.get(ds_num, None)
                    if baseline_val is not None:
                        diff_pct = ((val - baseline_val) / baseline_val) * 100
                        row += f"{val:.4f}({diff_pct:+.1f}%)"[:15].ljust(15)
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
                    output_file_pgf = os.path.join(output_dir, f"{model_type_short}_{table_type}_comparison.pgf")
                    
                    # Save as PNG
                    fig.savefig(output_file_png, dpi=300, bbox_inches='tight')
                    print(f"\nPlot saved: {output_file_png}")
                    
                    # Save as PDF (vector format, works without LaTeX and can be used in LaTeX documents)
                    output_file_pdf = os.path.join(output_dir, f"{model_type_short}_{table_type}_comparison.pdf")
                    fig.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
                    print(f"PDF (vector) saved: {output_file_pdf}")
                    
                    # Try to save as PGF for LaTeX (requires LaTeX installation)
                    if LATEX_EXPORT_AVAILABLE:
                        try:
                            fig.savefig(output_file_pgf, format='pgf', bbox_inches='tight')
                            print(f"LaTeX/PGF saved: {output_file_pgf}")
                        except Exception as e:
                            print(f"Note: PGF format not available (requires LaTeX in PATH)")
                            print(f"  → Use the PDF file instead: {output_file_pdf}")
                    
                    plt.close(fig)
                    
                    plot_count += 1
        
        print(f"\n{'='*100}")
        print(f"SUCCESS! Generated {plot_count} plots in '{output_dir}' directory")
        print(f"  - {plot_count} PNG files (.png) - for presentations/documents")
        print(f"  - {plot_count} PDF files (.pdf) - vector format for LaTeX")
        print(f"\nTo use in LaTeX:")
        print(f"  \\usepackage{{graphicx}}")
        print(f"  \\includegraphics[width=\\textwidth]{{filename.pdf}}")
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
