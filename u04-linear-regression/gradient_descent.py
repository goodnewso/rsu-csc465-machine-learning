"""Unit 4: fitting linear regression by batch gradient descent.

Cost J(w, b) = (1/2m) * sum((w*x + b - y)^2). Income is standardised
first so that gradient descent converges quickly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
house = pd.read_csv(DATA / "housing.csv")
x = house["median_income"].to_numpy()
y = house["median_house_value"].to_numpy() / 1000
x = (x - x.mean()) / x.std()                  # standardise the input
m = len(x)


def cost(w, b):
    return np.sum((w * x + b - y) ** 2) / (2 * m)


w, b, alpha = 0.0, 0.0, 0.1                   # start at zero
for it in range(201):
    err = w * x + b - y                       # prediction errors
    dw = np.dot(err, x) / m                   # partial derivative dJ/dw
    db = err.mean()                           # partial derivative dJ/db
    w, b = w - alpha * dw, b - alpha * db
    if it in (0, 1, 5, 20, 50, 200):
        print(f"iteration {it:3d}: w = {w:7.2f}  b = {b:7.2f}  "
              f"J = {cost(w, b):8.1f}")

wn, bn = np.polyfit(x, y, 1)                  # exact least squares
print(f"exact solution:  w = {wn:7.2f}  b = {bn:7.2f}  "
      f"J = {cost(wn, bn):8.1f}")
