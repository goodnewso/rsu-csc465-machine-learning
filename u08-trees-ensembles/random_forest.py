"""Unit 8: one tree versus bagging versus a random forest (diabetes data)."""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "diabetes_balanced.csv")
X, y = df.drop(columns="Diabetes"), df["Diabetes"]

models = {
    "single tree": DecisionTreeClassifier(random_state=465),
    "bagging (100 trees)": BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=100, random_state=465),
    "random forest (100)": RandomForestClassifier(
        n_estimators=100, random_state=465),
}
for name, m in models.items():
    acc = cross_val_score(m, X, y, cv=5)
    print(f"{name:20s} accuracy {acc.mean():.3f} (+/- {acc.std():.3f})")

rf = RandomForestClassifier(n_estimators=300, oob_score=True,
                            random_state=465).fit(X, y)
print(f"\nout-of-bag accuracy of a 300-tree forest: {rf.oob_score_:.3f}")
imp = pd.Series(rf.feature_importances_, index=X.columns)
print("feature importances:")
print(imp.sort_values(ascending=False).round(3).to_string())
