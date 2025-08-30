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
from main_slim_linear_scaling import slim_linear_scaling  # import the slim_gsgp library with Linear Scaling
from datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from evaluators.fitness_functions import rmse  # import the rmse fitness metric
from utils.utils import train_test_split  # import the train-test split function

# Load the PPB dataset
X, y = load_ppb(X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)

# Split the test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# Apply the SLIM GSGP algorithm with Linear Scaling using Pareto Tournament
final_tree = slim_linear_scaling(X_train=X_train, y_train=y_train,
                                X_test=X_val, y_test=y_val,
                                dataset_name='ppb', slim_version='SLIM+SIG2', pop_size=100, n_iter=100,
                                ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True,
                                tournament_type="pareto", 
                                multi_obj_attrs=["fitness", "nodes_count"])

# Show the best individual structure at the last generation
final_tree.print_tree_representation()

# Show the linear scaling parameters
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
