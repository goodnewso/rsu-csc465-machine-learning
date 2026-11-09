"""Unit 8: a decision tree for penguin species, and impurity by hand."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

DATA = Path(__file__).resolve().parents[1] / "data"
peng = pd.read_csv(DATA / "penguins.csv").dropna()
cols = ["culmen_length_mm", "culmen_depth_mm",
        "flipper_length_mm", "body_mass_g"]
X_train, X_test, y_train, y_test = train_test_split(
    peng[cols], peng["species"], test_size=0.3,
    stratify=peng["species"], random_state=465)

p = y_train.value_counts(normalize=True).to_numpy()
print(f"root node: Gini = {1 - np.sum(p ** 2):.3f}, "
      f"entropy = {-np.sum(p * np.log2(p)):.3f} bits")

tree = DecisionTreeClassifier(max_depth=2, random_state=465)
tree.fit(X_train, y_train)
print(export_text(tree, feature_names=cols, decimals=1))

print("max_depth  leaves  train  test")
for depth in (1, 2, 4, None):
    t = DecisionTreeClassifier(max_depth=depth, random_state=465)
    t.fit(X_train, y_train)
    print(f"{str(depth):>9s}  {t.get_n_leaves():6d}  "
          f"{t.score(X_train, y_train):.3f}  {t.score(X_test, y_test):.3f}")
