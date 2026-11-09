"""Unit 8: boosting: AdaBoost, gradient boosting, XGBoost (diabetes)."""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "diabetes_balanced.csv")
X, y = df.drop(columns="Diabetes"), df["Diabetes"]

cv = StratifiedKFold(5, shuffle=True, random_state=465)
models = [
    ("random forest", RandomForestClassifier(300, random_state=465)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                    random_state=465)),
    ("gradient boosting", GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        random_state=465)),
    ("XGBoost", XGBClassifier(n_estimators=150, learning_rate=0.05,
                              max_depth=3, subsample=0.8,
                              random_state=465)),
]
print(f"{'model':18s} accuracy   ROC-AUC")
for name, m in models:
    s = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
    print(f"{name:18s} {s['test_accuracy'].mean():.3f}      "
          f"{s['test_roc_auc'].mean():.3f}")
