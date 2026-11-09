"""Unit 6: k-nearest neighbours: classifying penguins by species.

k-NN measures distances, so features on large scales (body mass in
grams) swamp small ones (bill depth in mm) unless the data is scaled.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
peng = pd.read_csv(DATA / "penguins.csv").dropna()
cols = ["culmen_length_mm", "culmen_depth_mm",
        "flipper_length_mm", "body_mass_g"]
X_train, X_test, y_train, y_test = train_test_split(
    peng[cols], peng["species"], test_size=0.3,
    stratify=peng["species"], random_state=465)

# k-NN by hand for one test penguin, k = 5, on scaled data
sc = StandardScaler().fit(X_train)
A, b = sc.transform(X_train), sc.transform(X_test.iloc[[0]])[0]
dist = np.sqrt(((A - b) ** 2).sum(axis=1))      # Euclidean distance
near = np.argsort(dist)[:5]
print("five nearest neighbours:", list(y_train.iloc[near]))
print("vote ->", y_train.iloc[near].mode()[0],
      "| true species:", y_test.iloc[0])

print("\n k   unscaled  scaled")
for k in (1, 3, 5, 15, 51):
    raw = KNeighborsClassifier(k).fit(X_train, y_train)
    scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(k))
    scaled.fit(X_train, y_train)
    print(f"{k:2d}    {raw.score(X_test, y_test):.3f}    "
          f"{scaled.score(X_test, y_test):.3f}")
