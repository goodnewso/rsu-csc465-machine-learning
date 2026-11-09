"""Unit 10: grid search and random search for hyperparameters (SVM)."""
from pathlib import Path
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "diabetes_balanced.csv")
X, y = df.drop(columns="Diabetes"), df["Diabetes"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=465)
pipe = Pipeline([("scale", StandardScaler()), ("svm", SVC())])

grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": [0.001, 0.01, 0.1, 1]}
gs = GridSearchCV(pipe, grid, cv=5).fit(X_tr, y_tr)
print(f"grid search: {len(gs.cv_results_['params'])} combinations, "
      f"best {gs.best_params_}, CV accuracy {gs.best_score_:.3f}")

dist = {"svm__C": loguniform(0.01, 1000),
        "svm__gamma": loguniform(1e-4, 10)}
rs = RandomizedSearchCV(pipe, dist, n_iter=16, cv=5,
                        random_state=465).fit(X_tr, y_tr)
best = {k: round(float(v), 4) for k, v in rs.best_params_.items()}
print(f"random search: 16 samples, best {best}, "
      f"CV accuracy {rs.best_score_:.3f}")
print(f"test accuracy of the grid-search model: {gs.score(X_te, y_te):.3f}")
