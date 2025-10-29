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
    
    # Convert to string if not already
    mean_str = str(mean_str)
    
    # Extract the first number (before any parentheses)
    # Pattern matches numbers with optional decimal point and digits
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
        # Read with 3 levels of headers: title row, variant+model row, median/mean row
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=[0, 1, 2])
        print(f"Loaded Excel file: {excel_file}")
        print(f"Sheet: {sheet_name if isinstance(sheet_name, str) else 'First sheet'}")
        print(f"Shape: {df.shape}")
        print(f"Table type: {table_type.upper()}")
        
        # The Excel has multiple tables. We need to find where the first table ends
        # Look for rows where the first column contains "PERFORMANCE - MODEL SIZE"
        first_col = df.iloc[:, 0]
        
        # Find the row index where the second table starts
        model_size_idx = None
        for idx, val in enumerate(first_col):
            if pd.notna(val) and 'MODEL SIZE' in str(val).upper():
                model_size_idx = idx
                print(f"Found 'PERFORMANCE - MODEL SIZE' table starting at row {idx}")
                break
        
        # Select the appropriate table based on table_type
        if table_type.lower() == 'size':
            # Load SIZE table
            if model_size_idx is not None:
                df = df.iloc[model_size_idx:].copy()
                print(f"Selected MODEL SIZE table, shape: {df.shape}")
            else:
                raise ValueError("MODEL SIZE table not found in the Excel file")
        else:
            # Load FITNESS table (default)
            if model_size_idx is not None:
                df = df.iloc[:model_size_idx].copy()
                print(f"Selected TEST FITNESS table, shape: {df.shape}")
        
        print(f"Final shape after processing: {df.shape}")
        print(f"Sample columns (first 10): {df.columns[:10].tolist()}")
        
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

def filter_variant_20(df):
    """
    Filter the data to only include VARIANT 20.
    Since the Excel has VARIANT 20 in the header, we don't need to filter.
    This function is kept for consistency but just returns the dataframe.
    
    Args:
        df: DataFrame with all data
        
    Returns:
        pandas.DataFrame: The same DataFrame (already VARIANT 20)
    """
    print(f"Data is already from VARIANT 20 (from Excel structure)")
    print(f"Number of rows: {len(df)}")
    
    return df

def extract_variant_means(df, variant_name):
    """
    Extract median values for each dataset and model type for a given variant.

    Args:
        df: DataFrame with 3-level multi-index columns (title, variant+model, median/mean)
        variant_name: string like 'VARIANT 20'

    Returns:
        dict: { 'Smallest Model': {dataset_num: median, ...}, ... }
    """
    model_types = ['Smallest Model', 'Optimal Compromise', 'Best Fitness']
    results = {m: {} for m in model_types}

    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in the Excel sheet")

    # Map model types to median column under given variant
    median_col_map = {}
    for model in model_types:
        # Look for columns where:
        # - level 0 (variant) == variant_name
        # - level 1 (model) == model name
        # - level 2 (statistic) contains "Median"
        candidates = [c for c in cols 
                     if str(c[0]).strip() == variant_name 
                     and str(c[1]).strip() == model
                     and 'Median' in str(c[2])]
        
        if candidates:
            median_col_map[model] = candidates[0]
            print(f"  Found Median column for {model}: {candidates[0]}")
        else:
            print(f"  Warning: No Median column found for {model} in {variant_name}")
            print(f"  Looking for: variant='{variant_name}', model='{model}', statistic contains 'Median'")
            median_col_map[model] = None

    # Iterate rows and extract dataset number and medians
    for idx, row in df.iterrows():
        first_cell = row.iloc[0]
        if pd.isna(first_cell):
            continue
        ds_name = str(first_cell)
        m = re.search(r'Dataset\s*(\d+)', ds_name, re.IGNORECASE)
        if not m:
            continue
        ds_num = int(m.group(1))

        for model in model_types:
            col = median_col_map.get(model)
            if col is None:
                continue
            raw = row[col]
            median_val = extract_mean_value(raw)
            if median_val is None:
                continue
            results[model][ds_num] = median_val

    return results


def prepare_plot_data(df):
    """
    Prepare data for plotting from the filtered DataFrame (VARIANT 20).
    Returns the same format as earlier: { model: {'datasets':[], 'medians':[]} }
    """
    variant_name = 'VARIANT 20'
    extracted = extract_variant_means(df, variant_name)

    plot_data = {}
    for model, mapping in extracted.items():
        ds_sorted = sorted(mapping.items())
        if ds_sorted:
            datasets, medians = zip(*ds_sorted)
            plot_data[model] = {'datasets': list(datasets), 'means': list(medians)}
        else:
            plot_data[model] = {'datasets': [], 'means': []}

    # Print summary of extracted data
    print("\nExtracted data summary:")
    for model_type in plot_data:
        print(f"{model_type}: {len(plot_data[model_type]['datasets'])} datasets")

    return plot_data


def create_variant_20_plot(plot_data):
    """
    Create a plot showing median values for VARIANT 20 by model type.
    """
    plt.figure(figsize=(14, 8))

    colors = {
        'Smallest Model': '#1f77b4',
        'Optimal Compromise': '#ff7f0e',
        'Best Fitness': '#2ca02c'
    }

    markers = {
        'Smallest Model': 'o',
        'Optimal Compromise': 's',
        'Best Fitness': 'D'
    }

    # Plot each model type and add individual annotations with different offsets
    # to avoid overlapping when values are close
    offsets = {
        'Smallest Model': 25,      # Offset más alto
        'Optimal Compromise': 5,   # Offset medio
        'Best Fitness': -20        # Offset más bajo
    }
    
    for model_type in ['Smallest Model', 'Optimal Compromise', 'Best Fitness']:
        if plot_data[model_type]['datasets']:
            datasets = plot_data[model_type]['datasets']
            medians = plot_data[model_type]['means']
            sorted_pairs = sorted(zip(datasets, medians))
            datasets_sorted, medians_sorted = zip(*sorted_pairs)

            plt.plot(datasets_sorted, medians_sorted,
                     color=colors[model_type],
                     marker=markers[model_type],
                     linewidth=2,
                     markersize=8,
                     label=model_type,
                     alpha=0.8)
            
            # Add individual value annotations for each point on this line
            # Use different vertical offsets for each model type to avoid overlap
            y_offset = offsets[model_type]
            for ds, median in zip(datasets_sorted, medians_sorted):
                plt.annotate(f'{median:.2f}', 
                           xy=(ds, median), 
                           xytext=(0, y_offset),
                           textcoords='offset points',
                           ha='center',
                           fontsize=7,
                           fontweight='bold',
                           color=colors[model_type],
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   edgecolor=colors[model_type],
                                   alpha=0.8,
                                   linewidth=1.5))

    plt.xlabel('Dataset Number', fontsize=12, fontweight='bold')
    plt.ylabel('Median Value', fontsize=12, fontweight='bold')
    plt.title('Original SLIM GSGP (baseline)', fontsize=14, fontweight='bold')
    plt.xticks(range(1, 16))
    plt.xlim(0.5, 15.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    return plt


def create_compare_plot(df, compare_variant='VARIANT 1', baseline_variant='VARIANT 20'):
    """
    Create a plot comparing `compare_variant` against `baseline_variant`.
    The baseline is shown as reference (zero), and the plotted values are
    (compare - baseline) for each dataset and model.
    """
    baseline = extract_variant_means(df, baseline_variant)
    compare = extract_variant_means(df, compare_variant)

    model_types = ['Smallest Model', 'Optimal Compromise', 'Best Fitness']

    colors = {
        'Smallest Model': '#1f77b4',
        'Optimal Compromise': '#ff7f0e',
        'Best Fitness': '#2ca02c'
    }
    markers = {
        'Smallest Model': 'o',
        'Optimal Compromise': 's',
        'Best Fitness': 'D'
    }

    plt.figure(figsize=(14, 8))

    # Draw zero reference line (baseline)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)

    any_data = False
    
    # Different offsets for each model type to avoid overlapping annotations
    offsets = {
        'Smallest Model': 25,      # Offset más alto
        'Optimal Compromise': 5,   # Offset medio
        'Best Fitness': -20        # Offset más bajo (pero se ajusta según el signo)
    }
    
    for model in model_types:
        ds = []
        deltas = []
        # consider datasets 1..15 to keep consistent x-axis
        for i in range(1, 16):
            b = baseline.get(model, {}).get(i, None)
            c = compare.get(model, {}).get(i, None)
            if b is None or c is None:
                continue
            ds.append(i)
            delta = c - b
            deltas.append(delta)

        if ds:
            any_data = True
            line = plt.plot(ds, deltas,
                     color=colors[model],
                     marker=markers[model],
                     linewidth=2,
                     markersize=8,
                     label=f"{model} ({compare_variant} - {baseline_variant})",
                     alpha=0.85)
            
            # Add individual value annotations for each point on this line
            # Use different vertical offsets for each model type to avoid overlap
            base_offset = offsets[model]
            for ds_num, delta in zip(ds, deltas):
                # Adjust offset based on sign of delta
                if delta >= 0:
                    y_offset = base_offset
                else:
                    y_offset = -base_offset
                    
                plt.annotate(f'{delta:.2f}', 
                           xy=(ds_num, delta), 
                           xytext=(0, y_offset),
                           textcoords='offset points',
                           ha='center',
                           fontsize=7,
                           fontweight='bold',
                           color=colors[model],
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   edgecolor=colors[model],
                                   alpha=0.8,
                                   linewidth=1.5))

    if not any_data:
        print("Warning: No overlapping data found between variants for comparison.")

    plt.xlabel('Dataset Number', fontsize=12, fontweight='bold')
    plt.ylabel(f'Median difference ({compare_variant} - {baseline_variant})', fontsize=12, fontweight='bold')
    plt.title(f'{compare_variant} improvement over original SLIM GSGP', fontsize=14, fontweight='bold')
    plt.xticks(range(1, 16))
    plt.xlim(0.5, 15.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    return plt

def print_detailed_values(plot_data):
    """
    Print detailed values for each dataset and model type.
    
    Args:
        plot_data: Dictionary with plot data organized by model type
    """
    print("\n" + "="*80)
    print("DETAILED VALUES BY DATASET")
    print("="*80)
    
    # Get all datasets (sorted)
    all_datasets = set()
    for model_type in plot_data:
        all_datasets.update(plot_data[model_type]['datasets'])
    all_datasets = sorted(all_datasets)
    
    # Print header
    print(f"\n{'Dataset':<15} {'Smallest Model':<20} {'Optimal Compromise':<20} {'Best Fitness':<20}")
    print("-" * 80)
    
    # Print values for each dataset
    for ds_num in all_datasets:
        row = [f"Dataset {ds_num}"]
        
        for model_type in ['Smallest Model', 'Optimal Compromise', 'Best Fitness']:
            datasets = plot_data[model_type]['datasets']
            means = plot_data[model_type]['means']
            
            # Find the index of this dataset
            try:
                idx = datasets.index(ds_num)
                value = means[idx]
                row.append(f"{value:<20.5f}")
            except (ValueError, IndexError):
                row.append(f"{'N/A':<20}")
        
        print(f"{row[0]:<15} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

def print_summary_statistics(plot_data):
    """
    Print summary statistics for the plot data.
    
    Args:
        plot_data: Dictionary with plot data organized by model type
    """
    print("\n" + "="*80)
    print("VARIANT 20 SUMMARY STATISTICS")
    print("="*80)
    
    for model_type in ['Smallest Model', 'Optimal Compromise', 'Best Fitness']:
        medians = plot_data[model_type]['means']
        
        if medians:
            print(f"\n{model_type}:")
            print(f"  Number of datasets: {len(medians)}")
            print(f"  Mean of medians: {np.mean(medians):.4f}")
            print(f"  Std deviation: {np.std(medians):.4f}")
            print(f"  Min value: {np.min(medians):.4f}")
            print(f"  Max value: {np.max(medians):.4f}")
        else:
            print(f"\n{model_type}: No data found")

def main(excel_file="manual_set_results_test_fitness_size.xlsx", output_plot=None, table_type='fitness'):
    """
    Main function to generate the VARIANT 20 plot.
    
    Args:
        excel_file: Path to the Excel file with results
        output_plot: Custom output filename for the plot. If None, auto-generates
        table_type: 'fitness' or 'size' - which table to plot
    """
    try:
        # Load data
        print(f"Loading results data from: {excel_file}")
        df = load_excel_data(excel_file, table_type=table_type)

        # Filter for VARIANT 20
        print("\nFiltering for VARIANT 20...")
        df_variant_20 = filter_variant_20(df)

        # Prepare plot data
        print(f"\nPreparing plot data for VARIANT 20 (baseline) - {table_type.upper()}...")
        plot_data = prepare_plot_data(df_variant_20)

        # Print detailed values for verification
        print_detailed_values(plot_data)

        # Print summary statistics
        print_summary_statistics(plot_data)

        # By default create baseline plot
        print(f"\nCreating VARIANT 20 {table_type.upper()} plot...")
        fig = create_variant_20_plot(plot_data)

        # Generate output filename if not provided
        if output_plot is None:
            output_plot = f"variant_20_{table_type}_plot.png"

        # Save the plot
        fig.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {output_plot}")
        # Show the plot so the user can close it manually
        try:
            plt.show()
        except Exception:
            # If showing fails (headless env), continue silently
            pass
        print("Plot generation complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Fixed Excel file - always the same
    excel_file = "manual_set_results_test_fitness_size.xlsx"
    
    # Parse command line arguments: [COMPARE_VARIANT] [TABLE_TYPE]
    compare_variant = None
    table_type = 'fitness'  # Default: fitness
    
    # Show help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage: python plot_fitness_size_results.py [COMPARE_VARIANT] [TABLE_TYPE]")
        print()
        print("Parameters:")
        print("  COMPARE_VARIANT  - Variant to compare against VARIANT 20, e.g., 'VARIANT 1' (optional)")
        print("  TABLE_TYPE       - 'fitness' or 'size' (default: fitness)")
        print()
        print("Examples:")
        print("  python plot_fitness_size_results.py")
        print("    → Baseline plot for VARIANT 20 (fitness)")
        print("    → Output: variant_20_fitness_plot.png")
        print()
        print("  python plot_fitness_size_results.py 'VARIANT 1'")
        print("    → Compare VARIANT 1 vs VARIANT 20 (fitness)")
        print("    → Output: compare_VARIANT_1_vs_VARIANT_20_fitness.png")
        print()
        print("  python plot_fitness_size_results.py 'VARIANT 1' size")
        print("    → Compare VARIANT 1 vs VARIANT 20 (size)")
        print("    → Output: compare_VARIANT_1_vs_VARIANT_20_size.png")
        print()
        print("  python plot_fitness_size_results.py '' size")
        print("    → Baseline plot for VARIANT 20 (size)")
        print("    → Output: variant_20_size_plot.png")
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1]:
        compare_variant = sys.argv[1]  # e.g., 'VARIANT 1'
    if len(sys.argv) > 2 and sys.argv[2]:
        table_type = sys.argv[2]  # 'fitness' or 'size'
    
    # Handle comparison mode vs baseline mode
    if compare_variant:
        print(f"Configuration:")
        print(f"  Excel file: {excel_file}")
        print(f"  Compare variant: {compare_variant}")
        print(f"  Table type: {table_type.upper()}")
        print()
        print(f"Will compare {compare_variant} against baseline VARIANT 20 using {table_type.upper()} data")
        # Load df and call compare plot directly to avoid re-reading inside main
        df = load_excel_data(excel_file, table_type=table_type)
        plt_obj = create_compare_plot(df, compare_variant=compare_variant, baseline_variant='VARIANT 20')
        output_plot = f"compare_{compare_variant.replace(' ', '_')}_vs_VARIANT_20_{table_type}.png"
        plt_obj.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as: {output_plot}")
        try:
            plt_obj.show()
        except Exception:
            pass
    else:
        print(f"Configuration:")
        print(f"  Excel file: {excel_file}")
        print(f"  Table type: {table_type.upper()}")
        print()
        main(excel_file=excel_file, output_plot=None, table_type=table_type)
