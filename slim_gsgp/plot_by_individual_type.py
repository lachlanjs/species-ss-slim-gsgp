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
    plt.figure(figsize=(14, 8))
    
    # Define colors for different variants (cycle through if more variants)
    color_palette = plt.cm.tab20.colors
    
    baseline_data = all_variants_data.get(baseline_variant, {})
    
    if not baseline_data:
        print(f"Warning: No baseline data found for {baseline_variant}")
        return None
    
    # Draw zero reference line (baseline)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'{baseline_variant} (baseline)')
    
    # Plot each variant (except baseline) as difference from baseline
    variant_idx = 0
    for variant_name in sorted(all_variants_data.keys(), 
                               key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
        if variant_name == baseline_variant:
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
                diff = variant_val - baseline_val
                differences.append(diff)
        
        if datasets:
            color = color_palette[variant_idx % len(color_palette)]
            
            plt.plot(datasets, differences,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=variant_name,
                    alpha=0.85)
            
            # Add value annotations with smart positioning
            for ds, diff in zip(datasets, differences):
                # Alternate offset direction to reduce overlap
                y_offset = 15 if variant_idx % 2 == 0 else -15
                if diff < 0:
                    y_offset = -y_offset
                
                plt.annotate(f'{diff:.2f}', 
                           xy=(ds, diff), 
                           xytext=(0, y_offset),
                           textcoords='offset points',
                           ha='center',
                           fontsize=6,
                           color=color,
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   edgecolor=color,
                                   alpha=0.7,
                                   linewidth=1))
            
            variant_idx += 1
    
    plt.xlabel('Dataset Number', fontsize=12, fontweight='bold')
    
    # Use "RMSE" instead of "fitness" in labels
    metric_name = "RMSE" if table_type.lower() == 'fitness' else table_type.upper()
    plt.ylabel(f'Difference from {baseline_variant} ({metric_name})', fontsize=12, fontweight='bold')
    
    model_type_short = model_type.replace(' ', '_').lower()
    plt.title(f'{model_type} - All Variants Comparison ({metric_name})', 
             fontsize=14, fontweight='bold')
    
    plt.xticks(range(1, 16))
    plt.xlim(0.5, 15.5)
    plt.grid(True, alpha=0.3)
    
    # Place legend outside the plot area
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    
    return plt.gcf()  # Return current figure instead of plt module

def print_summary_table(all_variants_data, model_type, baseline_variant='VARIANT 20'):
    """
    Print a summary table showing values for all variants and datasets.
    """
    print("\n" + "="*100)
    print(f"SUMMARY TABLE - {model_type}")
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
                print_summary_table(all_variants_data, model_type)
                
                # Create and save plot
                fig = create_individual_comparison_plot(all_variants_data, model_type, table_type)
                
                if fig is not None:
                    # Generate filename
                    model_type_short = model_type.replace(' ', '_').lower()
                    output_file = os.path.join(output_dir, f"{model_type_short}_{table_type}_comparison.png")
                    
                    fig.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"\nPlot saved: {output_file}")
                    plt.close(fig)
                    
                    plot_count += 1
        
        print(f"\n{'='*100}")
        print(f"SUCCESS! Generated {plot_count} plots in '{output_dir}' directory")
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
