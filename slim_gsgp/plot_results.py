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

def load_results_data(csv_file="results_all_datasets.csv"):
    """
    Load the results from CSV file.
    
    Args:
        csv_file: Path to the CSV file with results
        
    Returns:
        pandas.DataFrame: DataFrame with the results
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results file '{csv_file}' not found. Please run the experiments first.")
    
    try:
        # Try to load the CSV normally first
        df = pd.read_csv(csv_file)
        return df
    except pd.errors.ParserError as e:
        print(f"Warning: CSV parsing error: {e}")
        print("Attempting to fix CSV format issues...")
        
        # Try to load with error handling for inconsistent columns
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            print(f"Loaded {len(df)} valid rows, skipped problematic lines.")
            return df
        except Exception as e2:
            print(f"Failed to load CSV even with error handling: {e2}")
            raise e2

def calculate_improvement_percentage(df, baseline_execution="slim"):
    """
    Calculate improvement percentage for each execution type relative to baseline.
    
    Args:
        df: DataFrame with results
        baseline_execution: Execution type to use as baseline for improvement calculation
        
    Returns:
        dict: Dictionary with improvement percentages for each dataset and execution type
    """
    # Get unique datasets and execution types
    datasets = df['dataset_name'].unique()
    execution_types = df['execution_type'].unique()
    
    improvement_data = {}
    
    for dataset in datasets:
        dataset_data = df[df['dataset_name'] == dataset]
        
        # Get baseline RMSE (lower is better)
        baseline_row = dataset_data[dataset_data['execution_type'] == baseline_execution]
        
        if baseline_row.empty:
            print(f"Warning: No baseline '{baseline_execution}' found for dataset '{dataset}'")
            continue
            
        baseline_rmse = baseline_row['test_rmse'].iloc[0]
        
        improvement_data[dataset] = {}
        
        for exec_type in execution_types:
            exec_row = dataset_data[dataset_data['execution_type'] == exec_type]
            
            if exec_row.empty:
                improvement_data[dataset][exec_type] = None
                continue
                
            current_rmse = exec_row['test_rmse'].iloc[0]
            
            # Calculate improvement percentage
            # For RMSE (lower is better): improvement = (baseline - current) / baseline * 100
            # Positive percentage means improvement (reduction in RMSE)
            # Negative percentage means degradation (increase in RMSE)
            improvement_pct = ((baseline_rmse - current_rmse) / baseline_rmse) * 100
            improvement_data[dataset][exec_type] = improvement_pct
    
    return improvement_data

def create_improvement_plot(improvement_data, baseline_execution="slim"):
    """
    Create a plot showing improvement percentages for each execution type.
    
    Args:
        improvement_data: Dictionary with improvement data
        baseline_execution: Baseline execution type
    """
    # Filter out bike_sharing dataset
    filtered_improvement_data = {k: v for k, v in improvement_data.items() if k != 'bike_sharing'}
    
    # Prepare data for plotting
    datasets = list(filtered_improvement_data.keys())
    execution_types = [
        'slim', 'slim oms', 'slim oms pareto',
        'slim linear scaling', 'slim linear scaling oms', 'slim linear scaling oms pareto'
    ]
    
    # Create figure and axis
    plt.figure(figsize=(15, 10))
    
    # Colors and markers for each execution type
    colors = {
        'slim': '#1f77b4',
        'slim oms': '#ff7f0e', 
        'slim oms pareto': '#2ca02c',
        'slim linear scaling': '#d62728',
        'slim linear scaling oms': '#9467bd',
        'slim linear scaling oms pareto': '#8c564b'
    }
    
    markers = {
        'slim': 'o',
        'slim oms': 's',
        'slim oms pareto': '^',
        'slim linear scaling': 'D', 
        'slim linear scaling oms': 'v',
        'slim linear scaling oms pareto': 'p'
    }
    
    # Plot lines for each execution type
    for exec_type in execution_types:
        y_values = []
        x_positions = []
        
        for i, dataset in enumerate(datasets):
            if dataset in filtered_improvement_data and exec_type in filtered_improvement_data[dataset]:
                value = filtered_improvement_data[dataset][exec_type]
                if value is not None:
                    y_values.append(value)
                    x_positions.append(i)
        
        if y_values:  # Only plot if we have data
            plt.plot(x_positions, y_values, 
                    color=colors[exec_type], 
                    marker=markers[exec_type],
                    linewidth=2,
                    markersize=8,
                    label=exec_type.title(),
                    alpha=0.8)
    
    # Customize the plot
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Datasets', fontsize=12, fontweight='bold')
    plt.ylabel('Test RMSE Improvement (%)', fontsize=12, fontweight='bold')
    plt.title(f'Test RMSE Improvement Percentage by Algorithm\n(Baseline: {baseline_execution.title()})', 
              fontsize=14, fontweight='bold')
    
    # Set x-axis
    plt.xticks(range(len(datasets)), datasets, rotation=45, ha='right')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add text explanation
    plt.figtext(0.02, 0.02, 
                "Positive values indicate improvement (lower RMSE), negative values indicate degradation (higher RMSE)",
                fontsize=10, style='italic', alpha=0.7)
    
    return plt

def print_dataset_improvements(improvement_data, baseline_execution="slim"):
    """
    Print improvement percentages for each dataset and execution type.
    
    Args:
        improvement_data: Dictionary with improvement data
        baseline_execution: Baseline execution type
    """
    execution_types = [
        'slim', 'slim oms', 'slim oms pareto',
        'slim linear scaling', 'slim linear scaling oms', 'slim linear scaling oms pareto'
    ]
    
    print("\n" + "="*80)
    print(f"IMPROVEMENT PERCENTAGES BY DATASET (Baseline: {baseline_execution.upper()})")
    print("="*80)
    
    for dataset in improvement_data:
        print(f"\nDataset: {dataset.upper()}")
        print("-" * (len(dataset) + 10))
        
        for exec_type in execution_types:
            if exec_type in improvement_data[dataset] and improvement_data[dataset][exec_type] is not None:
                improvement = improvement_data[dataset][exec_type]
                status = "✓ IMPROVEMENT" if improvement > 0 else "✗ DEGRADATION" if improvement < 0 else "= NO CHANGE"
                print(f"  {exec_type:<25}: {improvement:>8.2f}% {status}")
            else:
                print(f"  {exec_type:<25}: {'N/A':>8} (No data)")

def create_summary_statistics(improvement_data):
    """
    Create summary statistics for the improvement data.
    
    Args:
        improvement_data: Dictionary with improvement data
    """
    execution_types = [
        'slim', 'slim oms', 'slim oms pareto',
        'slim linear scaling', 'slim linear scaling oms', 'slim linear scaling oms pareto'
    ]
    
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY STATISTICS")
    print("="*80)
    
    for exec_type in execution_types:
        improvements = []
        for dataset in improvement_data:
            if exec_type in improvement_data[dataset] and improvement_data[dataset][exec_type] is not None:
                improvements.append(improvement_data[dataset][exec_type])
        
        if improvements:
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            min_improvement = np.min(improvements)
            max_improvement = np.max(improvements)
            
            print(f"\n{exec_type.upper()}:")
            print(f"  Mean improvement: {mean_improvement:.2f}%")
            print(f"  Std deviation: {std_improvement:.2f}%")
            print(f"  Min improvement: {min_improvement:.2f}%")
            print(f"  Max improvement: {max_improvement:.2f}%")
            print(f"  Datasets with improvement (>0%): {sum(1 for x in improvements if x > 0)}/{len(improvements)}")

def main():
    """
    Main function to generate the improvement plot.
    """
    try:
        # Load data
        print("Loading results data...")
        df = load_results_data()
        
        print(f"Loaded {len(df)} results from {len(df['dataset_name'].unique())} datasets")
        print(f"Execution types found: {list(df['execution_type'].unique())}")
        
        # Calculate improvements
        print("\nCalculating improvement percentages...")
        baseline = "slim"  # You can change this if you want a different baseline
        improvement_data = calculate_improvement_percentage(df, baseline_execution=baseline)
        
        # Print improvement percentages for each dataset
        print_dataset_improvements(improvement_data, baseline_execution=baseline)
        
        # Create summary statistics
        create_summary_statistics(improvement_data)
        
        # Create plot
        print(f"\nCreating improvement plot (baseline: {baseline})...")
        plt = create_improvement_plot(improvement_data, baseline_execution=baseline)
        
        # Save the plot
        output_file = "test_rmse_improvement_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_file}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'run_all_datasets.py' first to generate the results.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
