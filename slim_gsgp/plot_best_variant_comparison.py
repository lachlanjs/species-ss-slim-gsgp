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

def extract_variant_data_all_models(df, variant_name):
    """
    Extract median values for all 3 model types for a specific variant.
    
    Args:
        df: DataFrame with 3-level multi-index columns
        variant_name: Name of the variant (e.g., 'VARIANT 1', 'VARIANT 20')
        
    Returns:
        dict: {
            'Smallest Model': {dataset_num: median, ...},
            'Optimal Compromise': {dataset_num: median, ...},
            'Best Fitness': {dataset_num: median, ...}
        }
    """
    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in the Excel sheet")
    
    model_types = ['Smallest Model', 'Optimal Compromise', 'Best Fitness']
    results = {}
    
    for model_type in model_types:
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
            results[model_type] = {}
            continue
        
        model_data = {}
        
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
                model_data[ds_num] = median_val
        
        results[model_type] = model_data
    
    return results

def find_best_variant(df_fitness, df_size, baseline_variant='VARIANT 20'):
    """
    Find the best variant by considering both RMSE reduction and SIZE reduction.
    Uses normalization to combine both metrics into a single score.
    
    Args:
        df_fitness: DataFrame with fitness data
        df_size: DataFrame with size data
        baseline_variant: The baseline variant to exclude from comparison
        
    Returns:
        str: Name of the best variant
    """
    cols = df_fitness.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in the Excel sheet")
    
    # Find all unique variants
    variants = set()
    for c in cols:
        variant_name = str(c[0]).strip()
        if variant_name.startswith('VARIANT') and variant_name != baseline_variant:
            variants.add(variant_name)
    
    print("\n" + "="*80)
    print("FINDING BEST VARIANT (considering both RMSE and SIZE reduction)")
    print("="*80)
    
    # Get baseline data
    baseline_fitness_data = extract_variant_data_all_models(df_fitness, baseline_variant)
    baseline_size_data = extract_variant_data_all_models(df_size, baseline_variant)
    
    # Calculate baseline averages
    baseline_fitness_values = []
    for model_data in baseline_fitness_data.values():
        baseline_fitness_values.extend(model_data.values())
    baseline_avg_fitness = np.mean(baseline_fitness_values)
    
    baseline_size_values = []
    for model_data in baseline_size_data.values():
        baseline_size_values.extend(model_data.values())
    baseline_avg_size = np.mean(baseline_size_values)
    
    print(f"\nBaseline ({baseline_variant}):")
    print(f"  Average RMSE: {baseline_avg_fitness:.6f}")
    print(f"  Average Size: {baseline_avg_size:.2f}")
    print()
    
    variant_scores = {}
    variant_fitness = {}
    variant_size = {}
    
    for variant_name in variants:
        # Get fitness data
        fitness_data = extract_variant_data_all_models(df_fitness, variant_name)
        fitness_values = []
        for model_data in fitness_data.values():
            fitness_values.extend(model_data.values())
        
        # Get size data
        size_data = extract_variant_data_all_models(df_size, variant_name)
        size_values = []
        for model_data in size_data.values():
            size_values.extend(model_data.values())
        
        if fitness_values and size_values:
            avg_fitness = np.mean(fitness_values)
            avg_size = np.mean(size_values)
            
            # Calculate percentage improvement over baseline
            # Negative is better (reduction)
            fitness_improvement = ((avg_fitness - baseline_avg_fitness) / baseline_avg_fitness) * 100
            size_improvement = ((avg_size - baseline_avg_size) / baseline_avg_size) * 100
            
            # Combined score: average of both improvements (lower is better)
            # Both metrics contribute equally
            combined_score = (fitness_improvement + size_improvement) / 2
            
            variant_scores[variant_name] = combined_score
            variant_fitness[variant_name] = fitness_improvement
            variant_size[variant_name] = size_improvement
            
            print(f"{variant_name}:")
            print(f"  RMSE: {avg_fitness:.6f} ({fitness_improvement:+.2f}%)")
            print(f"  Size: {avg_size:.2f} ({size_improvement:+.2f}%)")
            print(f"  Combined Score: {combined_score:+.2f}%")
            print()
    
    if not variant_scores:
        raise ValueError("No variant data found")
    
    # Find the variant with the lowest combined score (most improvement)
    best_variant = min(variant_scores, key=variant_scores.get)
    print(f"{'='*80}")
    print(f"✓ BEST VARIANT: {best_variant}")
    print(f"  RMSE improvement: {variant_fitness[best_variant]:+.2f}%")
    print(f"  Size improvement: {variant_size[best_variant]:+.2f}%")
    print(f"  Combined Score: {variant_scores[best_variant]:+.2f}%")
    print("="*80 + "\n")
    
    return best_variant

def create_comparison_plot(baseline_data, best_variant_data, best_variant_name, table_type, output_dir="plots_by_individual"):
    """
    Create a plot comparing the 3 model types between baseline and best variant.
    Shows 6 lines total: 3 for baseline and 3 for best variant.
    
    Args:
        baseline_data: dict with data for baseline variant
        best_variant_data: dict with data for best variant
        best_variant_name: Name of the best variant
        table_type: 'fitness' or 'size'
        output_dir: Directory to save the plot
    """
    model_types = ['Smallest Model', 'Optimal Compromise', 'Best Fitness']
    
    # Define colors and markers for each model type
    # Baseline: dashed lines
    # Best variant: solid lines
    model_colors = {
        'Smallest Model': '#3399FF',      # Blue
        'Optimal Compromise': '#66CC66',  # Green
        'Best Fitness': '#FF6600',        # Orange
    }
    
    model_markers = {
        'Smallest Model': 'o',   # Circle
        'Optimal Compromise': 's',  # Square
        'Best Fitness': '^',     # Triangle
    }
    
    plt.figure(figsize=(14, 8))
    
    # Datasets to plot (excluding dataset 12)
    # We'll use consecutive x-positions (1-14) but label them with real dataset numbers
    dataset_numbers = [i for i in range(1, 16) if i != 12]  # [1,2,3,...,11,13,14,15]
    x_positions = list(range(1, len(dataset_numbers) + 1))  # [1,2,3,...,14]
    
    # Plot baseline (VARIANT 20) with dashed lines
    for model_type in model_types:
        x_vals = []
        y_vals = []
        
        model_data = baseline_data.get(model_type, {})
        for x_pos, ds_num in zip(x_positions, dataset_numbers):
            val = model_data.get(ds_num, None)
            if val is not None:
                x_vals.append(x_pos)
                y_vals.append(val)
        
        if x_vals:
            plt.plot(x_vals, y_vals,
                    color=model_colors[model_type],
                    marker=model_markers[model_type],
                    linewidth=2,
                    markersize=8,
                    linestyle='--',
                    label=f'Baseline - {model_type}',
                    alpha=0.85)
    
    # Plot best variant with solid lines
    for model_type in model_types:
        x_vals = []
        y_vals = []
        
        model_data = best_variant_data.get(model_type, {})
        for x_pos, ds_num in zip(x_positions, dataset_numbers):
            val = model_data.get(ds_num, None)
            if val is not None:
                x_vals.append(x_pos)
                y_vals.append(val)
        
        if x_vals:
            base_color = model_colors[model_type]
            plt.plot(x_vals, y_vals,
                    color=base_color,
                    marker=model_markers[model_type],
                    linewidth=2.5,
                    markersize=8,
                    linestyle='-',
                    label=f'{best_variant_name} - {model_type}',
                    alpha=1.0)
    
    plt.xlabel('Dataset Number', fontsize=12, fontweight='bold')
    
    if table_type.lower() == 'fitness':
        plt.ylabel('RMSE', fontsize=12, fontweight='bold')
        metric_name = "Test Fitness"
    else:
        plt.ylabel('Model Size (nodes)', fontsize=12, fontweight='bold')
        metric_name = "Model Size"
    
    # Set x-ticks with real dataset numbers (skipping 12)
    plt.xticks(x_positions, dataset_numbers, weight='bold')
    plt.xlim(0.5, len(dataset_numbers) + 0.5)
    plt.grid(True, alpha=0.3)
    
    # Place legend outside the plot on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, prop={'weight': 'bold'})
    plt.tight_layout()
    
    return plt.gcf()

def main(excel_file="manual_set_results_test_fitness_size.xlsx", output_dir="plots_by_individual"):
    """
    Main function to generate comparison plots between baseline and best variant.
    
    Args:
        excel_file: Path to the Excel file with results
        output_dir: Directory to save the output plots
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        baseline_variant = 'VARIANT 20'
        
        # Load both fitness and size data
        print(f"\n{'='*100}")
        print(f"Loading FITNESS data")
        print(f"{'='*100}")
        df_fitness = load_excel_data(excel_file, table_type='fitness')
        
        print(f"\n{'='*100}")
        print(f"Loading SIZE data")
        print(f"{'='*100}")
        df_size = load_excel_data(excel_file, table_type='size')
        
        # Find the best variant considering both fitness and size
        best_variant = find_best_variant(df_fitness, df_size, baseline_variant)
        
        # Generate plots for both fitness and size
        table_types = ['fitness', 'size']
        
        for table_type in table_types:
            print(f"\n{'='*100}")
            print(f"Creating {table_type.upper()} comparison plot")
            print(f"{'='*100}")
            
            # Select appropriate DataFrame
            df = df_fitness if table_type == 'fitness' else df_size
            
            # Extract data for baseline
            print(f"\nExtracting data for {baseline_variant} (Baseline)...")
            baseline_data = extract_variant_data_all_models(df, baseline_variant)
            
            # Extract data for best variant
            print(f"\nExtracting data for {best_variant} (Best Variant)...")
            best_variant_data = extract_variant_data_all_models(df, best_variant)
            
            # Create comparison plot
            print(f"\nCreating comparison plot for {table_type}...")
            fig = create_comparison_plot(baseline_data, best_variant_data, 
                                        best_variant, table_type, output_dir)
            
            if fig is not None:
                # Generate filename
                output_file_png = os.path.join(output_dir, f"best_variant_{table_type}_comparison.png")
                output_file_pdf = os.path.join(output_dir, f"best_variant_{table_type}_comparison.pdf")
                
                # Save as PNG
                fig.savefig(output_file_png, dpi=300, bbox_inches='tight')
                print(f"✓ Plot saved: {output_file_png}")
                
                # Save as PDF
                fig.savefig(output_file_pdf, format='pdf', bbox_inches='tight')
                print(f"✓ PDF saved: {output_file_pdf}")
                
                plt.close(fig)
        
        print(f"\n{'='*100}")
        print(f"SUCCESS! Generated comparison plots in '{output_dir}' directory")
        print(f"  - Best variant selected: {best_variant}")
        print(f"  - Fitness comparison: {baseline_variant} vs {best_variant}")
        print(f"  - Size comparison: {baseline_variant} vs {best_variant}")
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
        print("Usage: python plot_best_variant_comparison.py [OUTPUT_DIR]")
        print()
        print("Generates 2 comparison plots:")
        print("  - Fitness: Comparing 3 model types between baseline and best variant")
        print("  - Size: Comparing 3 model types between baseline and best variant")
        print()
        print("Each plot shows 6 lines:")
        print("  - 3 dashed lines: Baseline (VARIANT 20) for each model type")
        print("  - 3 solid lines: Best variant for each model type")
        print()
        print("The best variant is automatically selected based on combined score:")
        print("  - Considers both RMSE reduction and SIZE reduction")
        print("  - Both metrics contribute equally (50% each)")
        print("  - Lower combined score = better variant")
        print()
        print("Parameters:")
        print("  OUTPUT_DIR  - Directory to save plots (default: 'plots_by_individual')")
        print()
        print("Examples:")
        print("  python plot_best_variant_comparison.py")
        print("    → Saves plots to 'plots_by_individual/' directory")
        print()
        print("  python plot_best_variant_comparison.py my_plots")
        print("    → Saves plots to 'my_plots/' directory")
        sys.exit(0)
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"Configuration:")
    print(f"  Excel file: {excel_file}")
    print(f"  Output directory: {output_dir}")
    print()
    
    main(excel_file=excel_file, output_dir=output_dir)
