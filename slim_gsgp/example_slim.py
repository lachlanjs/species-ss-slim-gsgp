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
from slim_gsgp.datasets.data_loader import load_bike_sharing  # import the loader for the dataset bike_sharing
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

# Load the bike_sharing dataset
X, y = load_bike_sharing(X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)

# Split the test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# Apply the SLIM GSGP algorithm
# final_tree = slim(X_train=X_train, y_train=y_train,
#                   X_test=X_val, y_test=y_val,
#                   dataset_name='parkinson_total_UPDRS', slim_version='SLIM+SIG2', pop_size=100, n_iter=100,
#                   ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True, tournament_type="pareto", tournament_size=5, multi_obj_attrs=["fitness", "size"], oms=True)

final_tree = slim(X_train=X_train, y_train=y_train,
                  X_test=X_val, y_test=y_val,
                  dataset_name='bike_sharing', slim_version='SLIM+SIG2', pop_size=100, n_iter=100,
                  ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True, 
                  # tournament_type="pareto", tournament_size=5, multi_obj_attrs=["fitness", "size"], 
                  oms=False, linear_scaling=True)  

# Show the best individual structure at the last generation
final_tree.print_tree_representation()

# Show linear scaling information if enabled
if hasattr(final_tree, 'print_scaling_info'):
    final_tree.print_scaling_info()

# Evaluate the final tree on validation data to get the fitness
final_tree.calculate_semantics(X_val, testing=True)
final_tree.evaluate(rmse, y_val, testing=True, operator="sum")

# Show fitness information
print(f"\n=== FITNESS INFORMATION ===")
print(f"Training fitness (RMSE): {final_tree.fitness:.6f}")
print(f"Validation fitness (RMSE): {final_tree.test_fitness:.6f}")

# Get the prediction of the best individual on the test set
predictions = final_tree.predict(X_test)

# Compute and print the RMSE on the test set
test_rmse = float(rmse(y_true=y_test, y_pred=predictions))
print(f"Final test fitness (RMSE): {test_rmse:.6f}")

print(f"\n=== SUMMARY ===")
print(f"Number of nodes: {final_tree.nodes_count}")
print(f"Tree depth: {final_tree.depth}")
print(f"Train -> Validation -> Test RMSE: {final_tree.fitness:.6f} -> {final_tree.test_fitness:.6f} -> {test_rmse:.6f}")

# Save results to file
dataset_name = 'bike_sharing'
# Determine execution type based on OMS and linear scaling usage
oms_used = False  # Change this to match the oms parameter above
linear_scaling_used = False  # Change this to match the linear_scaling parameter above

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
    training_rmse=final_tree.fitness,
    validation_rmse=final_tree.test_fitness,
    test_rmse=test_rmse,
    execution_type=execution_type
)

print(f"\n=== RESULTS SAVED ===")
print(f"Results have been saved to 'results_slim.csv'")
