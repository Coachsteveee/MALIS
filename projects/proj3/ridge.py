# using alpha to represent lambda to avoid naming conflicts
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.x_mean_ = None
        self.x_std_ = None

    def fit(self, X, y):
        # Standardizing the feature matrix X
        self.x_mean_ = np.mean(X, axis=0)
        self.x_std_ = np.std(X, axis=0)
        X_std = (X - self.x_mean_) / self.x_std_

        # Adding an intercept term to the feature matrix X
        n_samples = X.shape[0]
        X_std = np.hstack([np.ones((n_samples, 1)), X_std])  # Add intercept column

        # Creating the penalty matrix for Ridge Regression
        n_features = X_std.shape[1]
        A = np.eye(n_features)
        A[0, 0] = 0  # No regularization for the intercept term
        penalty = self.alpha * A

        # Solving the linear equation (X^T * X + alpha * I) * w = X^T * y for weights
        self.coef_ = np.linalg.solve(X_std.T @ X_std + penalty, X_std.T @ y)

        # Extract the intercept from the fitted coefficients
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        X_std = (X - self.x_mean_) / self.x_std_
        X_std = np.hstack([np.ones((X_std.shape[0], 1)), X_std])  # Add intercept column
        return X_std @ np.hstack([self.intercept_, self.coef_])
