"""Unit 1: vectors and matrices with NumPy.

A tiny data set of four houses: size (in 100 m^2), number of rooms,
and price (in millions of naira). We find the weights w that best fit
price = w0 + w1*size + w2*rooms by solving the normal equation.
"""
import numpy as np

X = np.array([[1.0, 1.2, 3],     # first column of 1s gives the intercept
              [1.0, 1.5, 3],
              [1.0, 2.0, 4],
              [1.0, 2.6, 5]])
y = np.array([18.0, 22.0, 30.0, 39.0])

print("shape of X:", X.shape)
print("dot product of rows 0 and 1:", X[0] @ X[1])
print("X transpose X:\n", X.T @ X)

# Normal equation: (X^T X) w = X^T y. Solve it rather than invert.
w = np.linalg.solve(X.T @ X, X.T @ y)
print("weights w0, w1, w2:", np.round(w, 3))
print("predicted prices:", np.round(X @ w, 2))
print("norm of the error vector:", round(np.linalg.norm(X @ w - y), 3))
