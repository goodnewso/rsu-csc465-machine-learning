"""Unit 6: logistic regression: who is likely to have diabetes?

Data: 5,000 adults from the US CDC Behavioral Risk Factor Surveillance
System 2015, sampled from the balanced (50% diabetic) version on Kaggle:
alexteboul/diabetes-health-indicators-dataset (CC0).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "diabetes_balanced.csv")
features = ["HighBP", "HighChol", "BMI", "GenHlth", "Age",
            "PhysActivity", "HvyAlcoholConsump", "Income"]
X, y = df[features], df["Diabetes"]
print(f"adults: {len(y)} | with diabetes: {y.sum()} ({y.mean():.1%})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=465)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
print(f"test accuracy: {model.score(X_test, y_test):.3f}")
print(f"accuracy of always answering 'no diabetes': "
      f"{1 - y_test.mean():.3f}")

coef = pd.Series(model[-1].coef_[0], index=features)
print("\nodds ratio for a one-standard-deviation increase:")
print(np.exp(coef).sort_values(ascending=False).round(2).to_string())

adult = pd.DataFrame([{"HighBP": 0, "HighChol": 1, "BMI": 30,
                       "GenHlth": 3, "Age": 8, "PhysActivity": 1,
                       "HvyAlcoholConsump": 0, "Income": 5}])
p = model.predict_proba(adult)[0, 1]
print(f"\nP(diabetes) for the example adult: {p:.3f}")
for t in (0.3, 0.5, 0.7):
    label = "diabetes" if p >= t else "no diabetes"
    print(f"  threshold {t}: predict {label}")
