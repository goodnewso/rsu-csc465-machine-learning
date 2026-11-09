"""Unit 5: Ridge (L2) and Lasso (L1) regularisation.

A degree-9 polynomial overfits 20 points. Ridge shrinks all the
weights; Lasso sets many of them exactly to zero.
"""
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(465)


def truth(x):
    return np.sin(1.5 * x) + 0.5 * x


x_tr = np.sort(rng.uniform(0, 4, 20)).reshape(-1, 1)
y_tr = truth(x_tr).ravel() + rng.normal(0, 0.3, 20)
x_te = rng.uniform(x_tr.min(), x_tr.max(), 200).reshape(-1, 1)
y_te = truth(x_te).ravel() + rng.normal(0, 0.3, 200)

models = {"no penalty": LinearRegression(),
          "Ridge a=0.01": Ridge(alpha=0.01),
          "Ridge a=1": Ridge(alpha=1),
          "Lasso a=0.001": Lasso(alpha=0.001, max_iter=200_000),
          "Lasso a=0.01": Lasso(alpha=0.01, max_iter=200_000)}
print(f"{'model':14s} {'test MSE':>8s} {'max |w|':>9s} {'zero w':>6s}")
for name, reg in models.items():
    pipe = make_pipeline(PolynomialFeatures(9, include_bias=False),
                         StandardScaler(), reg).fit(x_tr, y_tr)
    w = pipe[-1].coef_
    te = mean_squared_error(y_te, pipe.predict(x_te))
    print(f"{name:14s} {te:8.3f} {np.abs(w).max():9.2f} "
          f"{int(np.sum(np.abs(w) < 1e-6)):6d}")
