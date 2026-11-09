"""Unit 5: polynomial regression, underfitting and overfitting.

Twenty noisy points from a smooth curve are fitted with polynomials of
degree 1, 3 and 9; a separate test set shows which degree generalises.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(465)


def truth(x):
    return np.sin(1.5 * x) + 0.5 * x


x_tr = np.sort(rng.uniform(0, 4, 20)).reshape(-1, 1)
y_tr = truth(x_tr).ravel() + rng.normal(0, 0.3, 20)
x_te = rng.uniform(x_tr.min(), x_tr.max(), 200).reshape(-1, 1)
y_te = truth(x_te).ravel() + rng.normal(0, 0.3, 200)
grid = np.linspace(0, 4, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, d in zip(axes, (1, 3, 9)):
    model = make_pipeline(PolynomialFeatures(d, include_bias=False),
                          StandardScaler(),
                          LinearRegression()).fit(x_tr, y_tr)
    tr = mean_squared_error(y_tr, model.predict(x_tr))
    te = mean_squared_error(y_te, model.predict(x_te))
    print(f"degree {d:2d}: training MSE {tr:.3f}   test MSE {te:.3f}")
    ax.scatter(x_tr, y_tr, s=15, color="k")
    ax.plot(grid, model.predict(grid), color="#2a7f62")
    ax.set(title=f"degree {d}", ylim=(-1.5, 3.5), xlabel="x")
fig.tight_layout()
fig.savefig(FIG / "u05_polynomial.png", dpi=150)
