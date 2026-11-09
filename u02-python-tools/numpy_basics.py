"""Unit 2: NumPy arrays, vectorised arithmetic, broadcasting, indexing."""
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

a = np.array([2.0, 4.0, 6.0, 8.0])
print("a * 10 =", a * 10)                  # no loop needed
print("mean, std:", a.mean(), round(a.std(), 3))

M = np.arange(12).reshape(3, 4)            # 3 rows, 4 columns
print("M =\n", M)
print("row 1:", M[1], " column 2:", M[:, 2])
print("column means:", M.mean(axis=0))
print("M - column means (broadcasting):\n", M - M.mean(axis=0))
print("elements greater than 6:", M[M > 6])

# Load the penguin measurements as a NumPy array (skip text columns).
cols = (2, 3, 4, 5)                        # bill length, bill depth,
X = np.genfromtxt(DATA / "penguins.csv",   # flipper length, body mass
                  delimiter=",", skip_header=1, usecols=cols)
print("penguin array shape:", X.shape)
print("rows with a missing value:", int(np.isnan(X).any(axis=1).sum()))
print("column means (ignoring NaN):", np.round(np.nanmean(X, axis=0), 1))
