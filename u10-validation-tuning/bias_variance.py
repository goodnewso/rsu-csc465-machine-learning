"""Unit 10: underfitting and overfitting seen in a validation curve.

Tree depth controls model complexity. Shallow trees have high bias
(both scores low); deep trees have high variance (training score far
above the cross-validated score).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "diabetes_balanced.csv")
X, y = df.drop(columns="Diabetes"), df["Diabetes"]

depths = [1, 2, 3, 4, 5, 7, 10, 15]
train, valid = validation_curve(
    DecisionTreeClassifier(random_state=465), X, y,
    param_name="max_depth", param_range=depths, cv=5)
print("depth  train  validation  gap")
for d, tr, va in zip(depths, train.mean(axis=1), valid.mean(axis=1)):
    print(f"{d:5d}  {tr:.3f}  {va:.3f}       {tr - va:.3f}")
best = depths[int(np.argmax(valid.mean(axis=1)))]
print("best depth by validation score:", best)
