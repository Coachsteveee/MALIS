# REPORT

Project 3: Ridge Regression

&#x1F465; Team: Ekemini Ekong, Brice Robert 

- [ ] Task 1. Development of the [`ridge.py`](ridge.py) file to implement Ridge Regression without using external libraries like scikit-learn.

```python
# using alpha to represent lambda to avoid naming conflicts
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

```

- [ ] Task 2. Documentation and analysis of the code implementation.

Click here &#x1F449; [Experiment](ridge.ipynb) 

We initiated the project by constructing a `Ridge Regression` **class** in &#x1F40D; Python. The primary focus was on integrating the L2 regularization directly into the linear regression framework and using a python class.

The key aspects of our implementation include a straightforward approach to the Ridge Regression algorithm with data standardization (When we standardize the features, the resulting coefficients give an indication of the importance of each feature. A larger absolute value of the coefficient means that feature is more important for predicting the target variable.), as we worked with a single feature &#x1F4BE; dataset &#x1F3C3;(Olympics 100m dataset)&#x1F3C5; . The focus was on correctly applying the &#x26D4; `L2 penalty` to the coefficients during the model fitting process.

The fit method integrates the &#x1F4CF; regularization term into the &#x1F4C9; linear regression, while the predict method generates the model predictions. 
&#x1F4D1; Special attention was given to the numerical stability of the algorithm, opting for `np.linalg.solve` over `matrix inversion` methods for solving the linear equations.

Subsequent to the model development, we embarked on comparing our custom model's performance with `scikit-learn`'s Ridge Regression implementation. Using the `Mean Squared Error (MSE)` metric, we observed remarkably similar results:

| | |
|-|-|
| Our Model's MSE:        | 0.19447330491999762 |
| scikit-learn Model MSE: | 0.18584114069831603 |

This close similarity in performance strongly validates our model's accuracy and reliability.

Regarding the choice of the regularization parameter $\alpha$ (alpha), we conducted a series of experiments by varying $\alpha$'s value. The optimal $\alpha$ was determined based on the balance between `bias` and `variance`, observing the changes in `MSE`. We found that a moderate value of $\alpha$ provided the best trade-off, effectively minimizing overfitting while maintaining prediction accuracy.

In conclusion, our `Ridge Regression implementation` demonstrated robust performance, closely mirroring that of established machine learning libraries. The insights gained from the analysis of $\alpha$'s impact on the model's performance were invaluable, highlighting the importance of `regularization` in machine learning algorithms.

## &#x1F4DD; Displaimer
- [ ] ChatGPT was used all along the project
- [ ] Any references that helped making the project have been added at the bottom of each notebooks in the `References` section
