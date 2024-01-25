# REPORT

Project 3: Ridge Regression

&#x1F465; Team: Ekemini Ekong, Brice Robert 

- [ ] Task 1. Development of the [`ridge.py`](ridge.py) file to implement Ridge Regression without using external libraries like scikit-learn.

```python
import numpy as np

# using alpha to represent lambda to avoid naming conflicts
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        # Add an intercept term to the feature matrix X
        n_samples = X.shape[0]
        X_intercept = np.ones((n_samples, 1))
        X = np.hstack((X_intercept, X))

        # Create the penalty matrix for Ridge Regression
        n_features = X.shape[1]
        A = self.alpha * np.eye(n_features)
        A[0, 0] = 0  # No regularization for the intercept term

        # Solve the linear equation (X^T * X + alpha * I) * w = X^T * y for weights
        self.weights = np.linalg.solve(X.T @ X + A, X.T @ y)

    def predict(self, X):
        n_samples = X.shape[0]
        X_intercept = np.ones((n_samples, 1))
        X = np.hstack((X_intercept, X))
        return X.dot(self.weights)
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
