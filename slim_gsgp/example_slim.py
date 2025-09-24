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
from slim_gsgp.main_slim import slim  # import the slim_gsgp library
from slim_gsgp.datasets.data_loader import load_airfoil  # import the loader for the dataset airfoil
from slim_gsgp.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim_gsgp.utils.utils import train_test_split  # import the train-test split function
import csv
import os
from datetime import datetime

def save_results_to_file(dataset_name, training_rmse, validation_rmse, test_rmse, execution_type, filename="results_slim.csv"):
    """
    Save the results to a CSV file.
    
    Args:
        dataset_name: Name of the dataset used
        training_rmse: Training fitness (RMSE)
        validation_rmse: Validation fitness (RMSE)
        test_rmse: Final test fitness (RMSE)
        execution_type: Type of execution (e.g., "slim")
        filename: Name of the output file
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'dataset_name', 'training_rmse', 'validation_rmse', 'test_rmse', 'execution_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the results
        writer.writerow({
            'timestamp': timestamp,
            'dataset_name': dataset_name,
            'training_rmse': training_rmse,
            'validation_rmse': validation_rmse,
            'test_rmse': test_rmse,
            'execution_type': execution_type
        })

# Load the airfoil dataset
X, y = load_airfoil(X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)

# Split the test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# Apply the SLIM GSGP algorithm
results = slim(X_train=X_train, y_train=y_train,
               X_test=X_val, y_test=y_val,
               dataset_name='airfoil', slim_version='SLIM+ABS', pop_size=100, n_iter=100,
               ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True, 
               # tournament_type="pareto", tournament_size=5, multi_obj_attrs=["fitness", "size"], 
               oms=False, linear_scaling=True, enable_plotting=True)

# Extract both individuals
best_fitness_individual = results.best_fitness
best_normalized_individual = results.best_normalized

print("\n" + "="*80)
print("RESULTS FOR BEST FITNESS INDIVIDUAL")
print("="*80)

# Show the best fitness individual structure
best_fitness_individual.print_tree_representation()

# Show linear scaling information if enabled
if hasattr(best_fitness_individual, 'print_scaling_info'):
    best_fitness_individual.print_scaling_info()

# Evaluate the best fitness individual on validation data
best_fitness_individual.calculate_semantics(X_val, testing=True)
best_fitness_individual.evaluate(rmse, y_val, testing=True, operator="sum")

# Show fitness information for best fitness individual
print(f"\n=== FITNESS INFORMATION (Best Fitness) ===")
print(f"Training fitness (RMSE): {best_fitness_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {best_fitness_individual.test_fitness:.6f}")

# Get the prediction of the best fitness individual on the test set
predictions_best_fitness = best_fitness_individual.predict(X_test)
test_rmse_best_fitness = float(rmse(y_true=y_test, y_pred=predictions_best_fitness))
print(f"Final test fitness (RMSE): {test_rmse_best_fitness:.6f}")

print(f"\n=== SUMMARY (Best Fitness) ===")
print(f"Number of nodes: {best_fitness_individual.nodes_count}")
print(f"Tree depth: {best_fitness_individual.depth}")
print(f"Train -> Validation -> Test RMSE: {best_fitness_individual.fitness:.6f} -> {best_fitness_individual.test_fitness:.6f} -> {test_rmse_best_fitness:.6f}")

print("\n" + "="*80)
print("RESULTS FOR BEST NORMALIZED INDIVIDUAL (Pareto-based selection)")
print("="*80)

# Show the best normalized individual structure
best_normalized_individual.print_tree_representation()

# Show linear scaling information if enabled
if hasattr(best_normalized_individual, 'print_scaling_info'):
    best_normalized_individual.print_scaling_info()

# Evaluate the best normalized individual on validation data
best_normalized_individual.calculate_semantics(X_val, testing=True)
best_normalized_individual.evaluate(rmse, y_val, testing=True, operator="sum")

# Show fitness information for best normalized individual
print(f"\n=== FITNESS INFORMATION (Best Normalized) ===")
print(f"Training fitness (RMSE): {best_normalized_individual.fitness:.6f}")
print(f"Validation fitness (RMSE): {best_normalized_individual.test_fitness:.6f}")

# Get the prediction of the best normalized individual on the test set
predictions_best_normalized = best_normalized_individual.predict(X_test)
test_rmse_best_normalized = float(rmse(y_true=y_test, y_pred=predictions_best_normalized))
print(f"Final test fitness (RMSE): {test_rmse_best_normalized:.6f}")

print(f"\n=== SUMMARY (Best Normalized) ===")
print(f"Number of nodes: {best_normalized_individual.nodes_count}")
print(f"Tree depth: {best_normalized_individual.depth}")
print(f"Train -> Validation -> Test RMSE: {best_normalized_individual.fitness:.6f} -> {best_normalized_individual.test_fitness:.6f} -> {test_rmse_best_normalized:.6f}")

# Save results to file
dataset_name = 'airfoil'
# Determine execution type based on OMS and linear scaling usage
oms_used = False  # Change this to match the oms parameter above
linear_scaling_used = True  # Change this to match the linear_scaling parameter above

if linear_scaling_used and oms_used:
    execution_type = 'slim linear scaling oms'
elif linear_scaling_used:
    execution_type = 'slim linear scaling'
elif oms_used:
    execution_type = 'slim oms'
else:
    execution_type = 'slim'

save_results_to_file(
    dataset_name=dataset_name,
    training_rmse=best_fitness_individual.fitness,
    validation_rmse=best_fitness_individual.test_fitness,
    test_rmse=test_rmse_best_fitness,
    execution_type=execution_type
)

print(f"\n=== RESULTS SAVED ===")
print(f"Results have been saved to 'results_slim.csv' using best fitness individual")
