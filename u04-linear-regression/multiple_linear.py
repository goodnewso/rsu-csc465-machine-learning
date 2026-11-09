"""Unit 4: multiple linear regression with a train-test split."""
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
house = pd.read_csv(DATA / "housing.csv").dropna()
house["rooms_per_household"] = house["total_rooms"] / house["households"]
house["people_per_household"] = house["population"] / house["households"]
features = ["median_income", "housing_median_age",
            "rooms_per_household", "people_per_household",
            "latitude", "longitude"]
X = house[features]
y = house["median_house_value"] / 1000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=465)
model = LinearRegression().fit(X_train, y_train)

print(f"intercept: {model.intercept_:.1f}")
for name, c in zip(features, model.coef_):
    print(f"  {name:22s} {c:8.2f}")
print(f"R^2 on training data: {model.score(X_train, y_train):.3f}")
print(f"R^2 on test data:     {model.score(X_test, y_test):.3f}")
