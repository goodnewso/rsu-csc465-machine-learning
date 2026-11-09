"""Unit 4: simple linear regression by formula and with scikit-learn.

Predict median house value from median income (California housing).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).resolve().parents[1] / "data"
house = pd.read_csv(DATA / "housing.csv")
x = house["median_income"].to_numpy()          # tens of thousands of US$
y = house["median_house_value"].to_numpy() / 1000   # thousands of US$

# Least-squares formulas: slope = cov(x, y) / var(x)
b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b0 = y.mean() - b1 * x.mean()
print(f"by formula:  value = {b0:.2f} + {b1:.2f} * income")

model = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"scikit-learn: value = {model.intercept_:.2f} + "
      f"{model.coef_[0]:.2f} * income")

for inc in (2.0, 5.0, 8.0):
    print(f"income {inc:.0f}0k -> predicted value "
          f"{model.predict([[inc]])[0]:.0f}k")
mse = np.mean((y - model.predict(x.reshape(-1, 1))) ** 2)
print(f"mean squared error: {mse:.0f}  (in thousands of US$ squared)")
