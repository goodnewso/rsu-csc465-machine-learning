"""Unit 10: why one train-test split is not enough; k-fold CV."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "diabetes_balanced.csv")
X, y = df.drop(columns="Diabetes"), df["Diabetes"]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

scores = []
for seed in range(20):                         # 20 different random splits
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=seed)
    scores.append(model.fit(X_tr, y_tr).score(X_te, y_te))
print(f"20 single splits: accuracy from {min(scores):.3f} "
      f"to {max(scores):.3f}")

kf = KFold(n_splits=5, shuffle=True, random_state=465)
s = cross_val_score(model, X, y, cv=kf)
print("5-fold scores:", s.round(3), f"mean {s.mean():.3f} sd {s.std():.3f}")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=465)
s = cross_val_score(model, X, y, cv=skf)
print(f"stratified 10-fold: mean {s.mean():.3f} sd {s.std():.3f}")
