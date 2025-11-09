"""
Generate consolidated Excel file with all datasets in a single table per metric.
Reads individual summary_table CSV files and consolidates them into one Excel file.
"""

import pandas as pd
import os
from pathlib import Path
import re

# Configuration
LOG_DIR = "log"
OUTPUT_FILE = "consolidated_results.xlsx"

# NOTE: Dataset numbering goes from 1 to 14 (dataset 12 'istanbul' is commented out)
# To re-enable dataset 12, uncomment the 'istanbul' line below
# List of all datasets in order
DATASETS = [
    "airfoil",              # Dataset 1
    "bike_sharing",         # Dataset 2
    "bioavailability",      # Dataset 3
    "boston",               # Dataset 4
    "breast_cancer",        # Dataset 5
    "concrete_slump",       # Dataset 6
    "concrete_strength",    # Dataset 7
    "diabetes",             # Dataset 8
    "efficiency_cooling",   # Dataset 9
    "efficiency_heating",   # Dataset 10
    "forest_fires",         # Dataset 11
    # "istanbul",           # Dataset 12 - COMMENTED OUT (uncomment to re-enable)
    "parkinson_updrs",      # Dataset 13
    "ppb",                  # Dataset 14
    "resid_build_sale_price"# Dataset 15
]

def read_summary_table(dataset_name, execution_type="slim"):
    """
    Read the summary table CSV for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        execution_type: Type of execution (slim, slim_oms, etc.)
        
    Returns:
        Dictionary with training, test, and size data
    """
    filename = os.path.join(LOG_DIR, f"summary_table_{dataset_name}_{execution_type}.csv")
    
    if not os.path.exists(filename):
        print(f"Warning: File not found: {filename}")
        return None
    
    # Read the CSV
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse the three tables
    result = {
        'training': None,
        'test': None,
        'size': None
    }
    
    # Find TRAINING RESULTS (line 3 has the data)
    if len(lines) > 3:
        training_data = lines[3].strip().split(',')
        result['training'] = training_data
    
    # Find TEST RESULTS (line 8 has the data)
    if len(lines) > 8:
        test_data = lines[8].strip().split(',')
        result['test'] = test_data
    
    # Find SIZE (line 13 has the data)
    if len(lines) > 13:
        size_data = lines[13].strip().split(',')
        result['size'] = size_data
    
    return result

def create_consolidated_excel(execution_type="slim", variant_name="VARIANT 1"):
    """
    Create consolidated Excel file with all datasets.
    
    Args:
        execution_type: Type of execution (slim, slim_oms, etc.)
        variant_name: Name to show in the table header
    """
    # Prepare data structures for each metric
    training_data = []
    test_data = []
    size_data = []
    
    print("=" * 80)
    print("GENERATING CONSOLIDATED EXCEL FILE")
    print("=" * 80)
    print(f"Execution type: {execution_type}")
    print(f"Variant name: {variant_name}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)
    
    # Collect data from all datasets
    for idx, dataset in enumerate(DATASETS, 1):
        print(f"Processing [{idx}/{len(DATASETS)}]: {dataset}...", end=" ")
        
        data = read_summary_table(dataset, execution_type)
        
        if data is None:
            print("SKIPPED (file not found)")
            # Add empty row
            empty_row = [f"Dataset {idx} {dataset}"] + [''] * 6
            training_data.append(empty_row)
            test_data.append(empty_row)
            size_data.append(empty_row)
        else:
            # Add dataset name as first column
            training_row = [f"Dataset {idx} {dataset}"] + data['training']
            test_row = [f"Dataset {idx} {dataset}"] + data['test']
            size_row = [f"Dataset {idx} {dataset}"] + data['size']
            
            training_data.append(training_row)
            test_data.append(test_row)
            size_data.append(size_row)
            print("✓")
    
    # Create DataFrames
    columns = ['Dataset', 
               'SM_Median(IQR)', 'SM_Mean(STD)', 
               'OC_Median(IQR)', 'OC_Mean(STD)', 
               'BF_Median(IQR)', 'BF_Mean(STD)']
    
    df_training = pd.DataFrame(training_data, columns=columns)
    df_test = pd.DataFrame(test_data, columns=columns)
    df_size = pd.DataFrame(size_data, columns=columns)
    
    # Create Excel file with multiple sheets
    print("\nWriting Excel file...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Write Training sheet
        df_training.to_excel(writer, sheet_name='Training', index=False, startrow=2)
        ws_training = writer.sheets['Training']
        ws_training['A1'] = f'PERFORMANCE - TRAIN FITNESS (30 runs) {variant_name}'
        ws_training['A2'] = ''
        ws_training['B2'] = 'Smallest Model'
        ws_training['D2'] = 'Optimal Compromise'
        ws_training['F2'] = 'Best Fitness'
        
        # Write Test sheet
        df_test.to_excel(writer, sheet_name='Test', index=False, startrow=2)
        ws_test = writer.sheets['Test']
        ws_test['A1'] = f'PERFORMANCE - TEST FITNESS (30 runs) {variant_name}'
        ws_test['A2'] = ''
        ws_test['B2'] = 'Smallest Model'
        ws_test['D2'] = 'Optimal Compromise'
        ws_test['F2'] = 'Best Fitness'
        
        # Write Size sheet
        df_size.to_excel(writer, sheet_name='Size', index=False, startrow=2)
        ws_size = writer.sheets['Size']
        ws_size['A1'] = f'PERFORMANCE - MODEL SIZE (30 runs) {variant_name}'
        ws_size['A2'] = ''
        ws_size['B2'] = 'Smallest Model'
        ws_size['D2'] = 'Optimal Compromise'
        ws_size['F2'] = 'Best Fitness'
        
        # Adjust column widths
        for ws in [ws_training, ws_test, ws_size]:
            ws.column_dimensions['A'].width = 30
            for col in ['B', 'C', 'D', 'E', 'F', 'G']:
                ws.column_dimensions[col].width = 20
    
    print(f"✓ Excel file created: {OUTPUT_FILE}")
    print("=" * 80)
    print("SHEETS CREATED:")
    print("  - Training: Training RMSE results")
    print("  - Test: Test RMSE results")
    print("  - Size: Model size (number of nodes)")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate consolidated Excel file from CSV results')
    parser.add_argument('--execution_type', type=str, default='slim',
                        help='Execution type matching the CSV filenames (e.g., slim, slim_oms, slim_linear_scaling)')
    parser.add_argument('--variant_name', type=str, default='VARIANT 1',
                        help='Variant name to display in table headers')
    parser.add_argument('--output', type=str, default='consolidated_results.xlsx',
                        help='Output Excel filename')
    
    args = parser.parse_args()
    OUTPUT_FILE = args.output
    
    create_consolidated_excel(args.execution_type, args.variant_name)
