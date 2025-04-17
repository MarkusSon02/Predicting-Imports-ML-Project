import pandas as pd
import numpy as np
from utils import LinearRegression, RidgeRegression, tune_alpha, tune_decision_tree, set_up, MSE_best_decision_tree
import matplotlib.pyplot as plt

df = pd.read_excel("data/final_data.xlsx")
df = df.reset_index()
df = df.sort_values(by="Time")

"""
Set-up stage
"""
unique_time = df['Time'].sort_values().unique()

# 70% train set, 15% validation set, 15% test set
train_cutoff = unique_time[int(0.7 * len(unique_time))]
val_cutoff = unique_time[int(0.85 * len(unique_time))]

# Split into train, validation, and test sets 
train_df = df[df['Time'] < train_cutoff]
val_df = df[(df['Time'] >= train_cutoff) & (df['Time'] < val_cutoff)]
test_df = df[df['Time'] >= val_cutoff]

train_df.to_excel("train_df.xlsx")
val_df.to_excel("val_df.xlsx")
test_df.to_excel("test_df.xlsx")

# First set of features
X_filter = ["Official Exchange Rate percent change", 'CPI Price, % y-o-y, not seas. adj.,, [CPTOTSAXNZGY]', 'ln_Labor', "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]", "GDP per capita growth (annual %) [NY.GDP.PCAP.KD.ZG]", "Imports of goods and services (% of GDP) [NE.IMP.GNFS.ZS]", "Unemployment, total (% of total labor force) (modeled ILO estimate) [SL.UEM.TOTL.ZS]"]

# Second set of features
X_filter_2 = ["Official Exchange Rate percent change", 'ln_Labor', "GDP (current US$) [NY.GDP.MKTP.CD]", "Imports of goods and services (% of GDP) [NE.IMP.GNFS.ZS]"]

# Datasets with first set of features (more features)
X_train, y_train, X_val, y_val, X_test, y_test = set_up(train_df, val_df, test_df, X_filter)

# Datasets with second set of features (less features)
X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = set_up(train_df, val_df, test_df, X_filter_2)


"""
Testing with first feature set
"""
# Testing performance with linear regression
model = LinearRegression()
model.fit(X_train, y_train)

train_mse = model.MSE(X_train, y_train)
val_mse = model.MSE(X_val, y_val)
test_mse = model.MSE(X_test, y_test)

print()
print("Linear Regression of First Feature Set:")
print(f"Training MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")


# Tuning alpha of ridge regression
alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]

ridge_mse_list, ridge_r_squared_list, best_ridge_regression, best_alpha, best_ridge_mse, best_ridge_r_squared = tune_alpha(
    X_train, y_train, X_val, y_val, alphas
)

print()
print("Tune Alpha of Ridge Regression using First Feature Set")
print(f"Best alpha found: {best_alpha}")
print(f"Best Val MSE: {best_ridge_mse}")
print(f"Best Val R-squared: {best_ridge_r_squared}")

# Testing performance with ridge regression with best alpha
model = RidgeRegression(best_alpha)
model.fit(X_train, y_train)

train_mse = model.MSE(X_train, y_train)
val_mse = model.MSE(X_val, y_val)
test_mse = model.MSE(X_test, y_test)

print()
print("Best Ridge Regression of First Feature Set:")
print(f"Train MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")


# Testing and tuning depth of decision tree
depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, None]

tree_mse_list, tree_r_squared_list, best_tree_model, best_depth, best_tree_mse, best_tree_r_squared = tune_decision_tree(
    X_train, y_train, X_val, y_val, depths
)

print()
print("Tune Depth of Decision Tree Regressor using First Feature Set")
print(f"Best max_depth found: {best_depth}")
print(f"Best Val MSE: {best_tree_mse}")
print(f"Best Val R-squared: {best_tree_r_squared}")

train_mse, val_mse, test_mse = MSE_best_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, best_depth)

print()
print("Best Decision Tree Regression of First Feature Set:")
print(f"Train MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")


"""
Testing with second feature set
"""
# Testing performance with linear regression
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train_2)

train_mse_2 = model_2.MSE(X_train_2, y_train_2)
val_mse_2 = model_2.MSE(X_val_2, y_val_2)
test_mse_2 = model_2.MSE(X_test_2, y_test_2)

print()
print("Linear Regression of Second Feature Set:")
print(f"Training MSE: {train_mse_2}")
print(f"Validation MSE: {val_mse_2}")
print(f"Test MSE: {test_mse_2}")

# Tuning alpha of ridge regression
alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]

ridge_mse_list_2, ridge_r_squared_list_2, best_ridge_regression_2, best_alpha_2, best_ridge_mse_2, best_ridge_r_squared_2 = tune_alpha(
    X_train_2, y_train_2, X_val_2, y_val_2, alphas
)

print()
print("Tune Alpha of Ridge Regression using Second Feature Set")
print(f"Best alpha found: {best_alpha_2}")
print(f"Best Val MSE: {best_ridge_mse_2}")
print(f"Best Val R-squared: {best_ridge_r_squared_2}")

# Testing performance with ridge regression
model_2 = RidgeRegression(best_alpha_2)
model_2.fit(X_train_2, y_train_2)

train_mse_2 = model_2.MSE(X_train_2, y_train_2)
val_mse_2 = model_2.MSE(X_val_2, y_val_2)
test_mse_2 = model_2.MSE(X_test_2, y_test_2)

print()
print("Best Ridge Regression of Second Feature Set:")
print(f"Train MSE: {train_mse_2}")
print(f"Validation MSE: {val_mse_2}")
print(f"Test MSE: {test_mse_2}")


# Testing and tuning depth of decision tree
depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, None]

tree_mse_list_2, tree_r_squared_list_2, best_tree_model_2, best_depth_2, best_tree_mse_2, best_tree_r_squared_2 = tune_decision_tree(
    X_train_2, y_train_2, X_val_2, y_val_2, depths
)

print()
print("Tune Depth of Decision Tree Regressor using Second Feature Set")
print(f"Best max_depth found: {best_depth_2}")
print(f"Best Val MSE: {best_tree_mse_2}")
print(f"Best Val R-squared: {best_tree_r_squared_2}")

train_mse_2, val_mse_2, test_mse_2 = MSE_best_decision_tree(X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2, best_depth_2)

print()
print("Best Decision Tree Regression of Second Feature Set:")
print(f"Train MSE: {train_mse_2}")
print(f"Validation MSE: {val_mse_2}")
print(f"Test MSE: {test_mse_2}")


"""
Plotting results
"""
# Plot 1: Ridge Validation MSE vs λ for first feature set
plt.figure()
plt.semilogx(alphas, ridge_mse_list, marker='o', linestyle='-')
plt.xlabel('Regularization strength λ (log scale)')
plt.ylabel('Validation MSE')
plt.title('Figure 1: Ridge Regression: Validation MSE vs λ (First Feature Set)')
plt.grid(True)
plt.savefig("results/ridge_validation_accuracy")

# Plot 2: Decision tree Validation MSE vs max depth for first feature set
plt.figure()
plt.plot(depths, tree_mse_list, marker='o', linestyle='-')
plt.xlabel('Max Depth')
plt.ylabel('Validation MSE')
plt.title('Figure 3: Decision Tree: Validation MSE vs Max Depth (First Feature Set)')
plt.grid(True)
plt.savefig("results/tree_validation_accuracy")

# Plot 3: Ridge Validation MSE vs λ for second feature set
plt.figure()
plt.semilogx(alphas, ridge_mse_list_2, marker='o', linestyle='-')
plt.xlabel('Regularization strength λ (log scale)')
plt.ylabel('Validation MSE')
plt.title('Figure 2: Ridge Regression: Validation MSE vs λ (Second Feature Set)')
plt.grid(True)
plt.savefig("results/ridge_validation_accuracy_2")

# Plot 4: Decision tree Validation MSE vs max depth for second feature set
plt.figure()
plt.plot(depths, tree_mse_list_2, marker='o', linestyle='-')
plt.xlabel('Max Depth')
plt.ylabel('Validation MSE')
plt.title('Figure 4: Decision Tree: Validation MSE vs Max Depth (Second Feature Set)')
plt.grid(True)
plt.savefig("results/tree_validation_accuracy_2")
