import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        XtX = X_bias.T @ X_bias
        Xty = X_bias.T @ y
        self.weights = np.linalg.inv(XtX) @ Xty

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.weights
    
    def score(self, X, y):
        y_predict = self.predict(X)
        SS_res = np.sum((y - y_predict)**2)
        SS_total = np.sum((y - np.mean(y))**2)
        return 1 - (SS_res / SS_total)

    def MSE(self, X, y):
        y_predict = self.predict(X)
        return np.mean((y - y_predict)**2)

class RidgeRegression:
    def __init__(self, alpha=5.0):
        # Regularization strength (alpha=lambda)
        self.alpha = alpha
        self.weights = None
    
    def fit(self, X, y):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Identity matrix for regularization
        I = np.eye(X_bias.shape[1])
        I[0, 0]
        
        XtX = X_bias.T @ X_bias
        Xty = X_bias.T @ y

        self.weights = np.linalg.inv(XtX + self.alpha * I) @ Xty

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.weights
    
    def score(self, X, y):
        y_predict = self.predict(X)
        SS_res = np.sum((y - y_predict)**2)
        SS_total = np.sum((y - np.mean(y))**2)
        return 1 - (SS_res / SS_total)

    def MSE(self, X, y):
        y_predict = self.predict(X)
        return np.mean((y - y_predict)**2)


# Setting up the train, validation, and test sets
def set_up(train_df, val_df, test_df, X_filter):
    X_train = train_df[X_filter]
    X_val = val_df[X_filter]
    X_test = test_df[X_filter]

    y_train = train_df['ln_imports']
    y_val = val_df['ln_imports']
    y_test = test_df['ln_imports']

    # Convert to type float
    X_train = X_train.astype(float)
    y_train = y_train.astype(float)

    X_val = X_val.astype(float)
    y_val = y_val.astype(float)

    X_test = X_test.astype(float)
    y_test = y_test.astype(float)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# Function for tuning alpha of RidgeRegression    
def tune_alpha(X_train, y_train, X_val, y_val, alphas):
    best_alpha = None
    best_model = None
    best_val_mse = float('inf')
    best_r_squared = None
    mse_list = []
    r_squared_list = []

    for alpha in alphas:
        model = RidgeRegression(alpha)
        model.fit(X_train, y_train)

        val_mse = model.MSE(X_val, y_val)
        r_squared = model.score(X_val, y_val)
        mse_list.append(val_mse)
        r_squared_list.append(r_squared)


        if val_mse < best_val_mse:
            best_alpha = alpha
            best_model = model
            best_val_mse = val_mse
            best_r_squared = r_squared

    return mse_list, r_squared_list, best_model, best_alpha, best_val_mse, best_r_squared


# Function for tuning depth of DecisionTreeRegressor
def tune_decision_tree(X_train, y_train, X_val, y_val, depths):
    best_model = None
    best_depth = None
    best_val_mse = float('inf')
    best_r_squared = None
    mse_list = []
    r_squared_list = []

    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_val)

        val_mse = mean_squared_error(y_val, y_predict)
        r_squared = r2_score(y_val, y_predict)
        mse_list.append(val_mse)
        r_squared_list.append(r_squared)


        if val_mse < best_val_mse:
            best_model = model
            best_depth = depth
            best_val_mse = val_mse
            best_r_squared = r_squared

    return mse_list, r_squared_list ,best_model, best_depth, best_val_mse, best_r_squared

def MSE_best_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, depth):
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    y_train_predict = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_predict)

    y_val_predict = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_predict)

    y_test_predict = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_predict)

    return train_mse, val_mse, test_mse