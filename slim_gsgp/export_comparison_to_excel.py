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
import numpy as np
import os
import re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, numbers
from openpyxl.utils import get_column_letter
import locale

def extract_mean_value(mean_str):
    """
    Extract the main mean value (not the one in parentheses).
    
    Args:
        mean_str: String containing mean value, possibly with value in parentheses
        
    Returns:
        float: The extracted mean value
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

def load_excel_data(excel_file="results_test_fitness_size.xlsx", sheet_name=0):
    """
    Load the results from Excel file.
    
    Args:
        excel_file: Path to the Excel file with results
        sheet_name: Sheet name or index to read (0 for first sheet)
        
    Returns:
        pandas.DataFrame: DataFrame with the results
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Results file '{excel_file}' not found.")
    
    try:
        # Read the first sheet with multi-level headers
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=[0, 1])
        print(f"Loaded Excel file: {excel_file}")
        print(f"Sheet: {sheet_name if isinstance(sheet_name, str) else 'First sheet'}")
        print(f"Shape: {df.shape}")
        
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
        
        # Keep only rows before the second table (TEST FITNESS table)
        if model_size_idx is not None:
            df = df.iloc[:model_size_idx].copy()
            print(f"Filtered to TEST FITNESS table only, new shape: {df.shape}")
        
        # The first row contains the actual statistic names (Median (IQR), Mean (STD))
        # We need to skip this row and use it to rename columns
        if len(df) > 0:
            first_row = df.iloc[0]
            
            # Drop the first row (which was the statistic names)
            df = df.iloc[1:].reset_index(drop=True)
            
            print(f"Updated shape after removing header row: {df.shape}")
        
        print(f"Sample columns: {df.columns[:10].tolist()}")
        
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

def extract_variant_means(df, variant_name):
    """
    Extract mean values for each dataset and model type for a given variant.

    Args:
        df: DataFrame with multi-index columns (variant, statistic)
        variant_name: string like 'VARIANT 20'

    Returns:
        dict: { 'Smallest Model': {dataset_num: mean, ...}, ... }
    """
    model_types = ['Smallest Model', 'Optimal Compromise', 'Best Fitness']
    results = {m: {} for m in model_types}

    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in the Excel sheet")

    # Map model types to mean column under given variant
    mean_col_map = {}
    for model in model_types:
        # Look for columns where:
        # - level0 == variant_name 
        # - level1 contains the model name AND has .1 suffix (indicating it's the Mean column)
        candidates = [c for c in cols 
                     if str(c[0]).strip() == variant_name 
                     and model in str(c[1]) 
                     and ('.1' in str(c[1]) or 'Mean' in str(c[1]))]
        
        if candidates:
            mean_col_map[model] = candidates[0]
            print(f"  Found Mean column for {model}: {candidates[0]}")
        else:
            print(f"  Warning: No Mean column found for {model} in {variant_name}")
            mean_col_map[model] = None

    # Iterate rows and extract dataset number and means
    for idx, row in df.iterrows():
        first_cell = row.iloc[0]
        if pd.isna(first_cell):
            continue
        ds_name = str(first_cell)
        
        # Skip rows that contain "PERFORMANCE - TEST FITNESS" or similar headers
        if 'PERFORMANCE' in ds_name.upper() or 'TEST FITNESS' in ds_name.upper():
            continue
        
        m = re.search(r'Dataset\s*(\d+)', ds_name, re.IGNORECASE)
        if not m:
            continue
        ds_num = int(m.group(1))

        for model in model_types:
            col = mean_col_map.get(model)
            if col is None:
                continue
            raw = row[col]
            mean_val = extract_mean_value(raw)
            if mean_val is None:
                continue
            results[model][ds_num] = mean_val

    return results

def get_dataset_name(df, dataset_num):
    """
    Get the full dataset name from the dataframe.
    
    Args:
        df: DataFrame with the data
        dataset_num: Dataset number (1-15)
        
    Returns:
        str: Dataset name (e.g., 'airfoil', 'bike_sharing')
    """
    for idx, row in df.iterrows():
        first_cell = row.iloc[0]
        if pd.isna(first_cell):
            continue
        ds_name = str(first_cell)
        
        # Skip rows that contain "PERFORMANCE - TEST FITNESS" or similar headers
        if 'PERFORMANCE' in ds_name.upper() or 'TEST FITNESS' in ds_name.upper():
            continue
        
        m = re.search(r'Dataset\s*(\d+)\s*(.+)', ds_name, re.IGNORECASE)
        if m and int(m.group(1)) == dataset_num:
            return m.group(2).strip()
    return ""

def get_all_variants(df):
    """
    Extract all variant names from the DataFrame columns.
    
    Args:
        df: DataFrame with multi-index columns
        
    Returns:
        list: Sorted list of variant names with VARIANT 20 first
    """
    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns in the Excel sheet")
    
    # Get unique variant names from level 0 of MultiIndex
    variants = set()
    for col in cols:
        variant_name = str(col[0]).strip()
        # Filter out unwanted column names
        if (variant_name and 
            'Unnamed' not in variant_name and
            'PERFORMANCE' not in variant_name.upper() and
            'TEST FITNESS' not in variant_name.upper()):
            variants.add(variant_name)
    
    # Sort variants: VARIANT 20 first, then others numerically
    def variant_sort_key(v):
        match = re.search(r'VARIANT\s*(\d+)', v, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            # Put VARIANT 20 first (use -1), then sort others numerically
            if num == 20:
                return -1
            return num
        return 999
    
    sorted_variants = sorted(variants, key=variant_sort_key)
    return sorted_variants

def create_comparison_excel(df, output_file='comparison_results.xlsx'):
    """
    Create an Excel file with all variants.
    
    Args:
        df: DataFrame with the data
        output_file: Output Excel filename
    """
    # Get all variants
    print("\nDetecting all variants in the data...")
    all_variants = get_all_variants(df)
    print(f"Found {len(all_variants)} variants: {', '.join(all_variants)}")
    
    # Extract data for all variants
    variants_data = {}
    for variant in all_variants:
        print(f"Extracting data for {variant}...")
        variants_data[variant] = extract_variant_means(df, variant)
    
    # Create a new workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Comparison"
    
    # Define styles
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=12)
    dataset_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    dataset_font = Font(bold=True, size=11)
    model_font = Font(size=10)
    center_alignment = Alignment(horizontal="center", vertical="center")
    left_alignment = Alignment(horizontal="left", vertical="center", indent=1)
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Write headers - column A for dataset names, then one column per variant
    ws['A1'] = ""
    for idx, variant in enumerate(all_variants):
        col_letter = get_column_letter(idx + 2)  # Start from column B
        ws[f'{col_letter}1'] = variant
    
    # Apply header styling to all columns
    for idx in range(len(all_variants) + 1):
        col_letter = get_column_letter(idx + 1)
        cell = ws[f'{col_letter}1']
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_alignment
        cell.border = thin_border
    
    # Dataset configuration - excluding dataset 4
    dataset_configs = [
        (1, 'airfoil'),
        (2, 'bike_sharing'),
        (3, 'bioavailability'),
        (5, 'breast_cancer'),
        (6, 'concrete_slump'),
        (7, 'concrete_strength'),
        (8, 'diabetes'),
        (9, 'efficiency_cooling'),
        (10, 'efficiency_heating'),
        (11, 'forest_fires'),
        (12, 'istanbul'),
        (13, 'parkinson_updrs'),
        (14, 'ppb'),
        (15, 'resid_build_sale_price')
    ]
    
    model_types_display = {
        'Smallest Model': 'Smallest',
        'Optimal Compromise': 'Best normalized',
        'Best Fitness': 'Best fitness'
    }
    
    current_row = 2
    
    # Iterate through datasets
    for dataset_num, dataset_name in dataset_configs:
        # Get actual dataset name from dataframe if available
        actual_name = get_dataset_name(df, dataset_num)
        if not actual_name:
            actual_name = dataset_name
        
        # Write dataset header
        dataset_label = f"Dataset {dataset_num} {actual_name}"
        ws[f'A{current_row}'] = dataset_label
        ws[f'A{current_row}'].fill = dataset_fill
        ws[f'A{current_row}'].font = dataset_font
        ws[f'A{current_row}'].alignment = left_alignment
        ws[f'A{current_row}'].border = thin_border
        
        # Apply border to all variant columns for this row
        for idx in range(len(all_variants)):
            col_letter = get_column_letter(idx + 2)
            ws[f'{col_letter}{current_row}'].border = thin_border
        
        current_row += 1
        
        # Write model types and their values
        for model_type_key, model_type_display in model_types_display.items():
            ws[f'A{current_row}'] = model_type_display
            ws[f'A{current_row}'].font = model_font
            ws[f'A{current_row}'].alignment = left_alignment
            ws[f'A{current_row}'].border = thin_border
            
            # Write values for each variant
            for idx, variant in enumerate(all_variants):
                col_letter = get_column_letter(idx + 2)  # Start from column B
                variant_val = variants_data[variant].get(model_type_key, {}).get(dataset_num)
                
                if variant_val is not None:
                    ws[f'{col_letter}{current_row}'] = round(variant_val, 6)
                    # Force number format with dot as decimal separator
                    ws[f'{col_letter}{current_row}'].number_format = '0.000000'
                else:
                    ws[f'{col_letter}{current_row}'] = "N/A"
                
                ws[f'{col_letter}{current_row}'].alignment = center_alignment
                ws[f'{col_letter}{current_row}'].border = thin_border
            
            current_row += 1
    
    # Adjust column widths
    ws.column_dimensions['A'].width = 35
    for idx in range(len(all_variants)):
        col_letter = get_column_letter(idx + 2)
        ws.column_dimensions[col_letter].width = 15
    
    # Save the workbook
    wb.save(output_file)
    print(f"\n✓ Excel file created successfully: {output_file}")
    print(f"  Total variants: {len(all_variants)}")
    print(f"  Total datasets: {len(dataset_configs)}")

def main(input_excel="results_test_fitness_size.xlsx", 
         output_excel="comparison_results.xlsx"):
    """
    Main function to generate comparison Excel with all variants.
    
    Args:
        input_excel: Path to the input Excel file with results
        output_excel: Path for the output comparison Excel file
    """
    try:
        print("="*80)
        print("EXCEL COMPARISON GENERATOR - ALL VARIANTS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Input file: {input_excel}")
        print(f"  Output file: {output_excel}")
        
        # Load data
        df = load_excel_data(input_excel)
        
        # Create comparison Excel with all variants
        create_comparison_excel(df, output_file=output_excel)
        
        print("\n" + "="*80)
        print("PROCESS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    input_excel = "results_test_fitness_size.xlsx"
    output_excel = "comparison_results.xlsx"
    
    if len(sys.argv) > 1:
        input_excel = sys.argv[1]
    if len(sys.argv) > 2:
        output_excel = sys.argv[2]
    
    print("\nUsage: python export_comparison_to_excel.py [input_excel] [output_excel]")
    print("Example: python export_comparison_to_excel.py results_test_fitness_size.xlsx comparison.xlsx\n")
    print("Note: All variants found in the input file will be included in the output.\n")
    
    main(input_excel=input_excel, 
         output_excel=output_excel)
