"""Unit 9: MAE, MSE, RMSE and R^2, by hand and with scikit-learn."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# five predictions checked by hand (house values in thousands of US$)
y = np.array([200, 150, 320, 410, 260])
p = np.array([210, 140, 300, 450, 250])
e = y - p
print("errors:", e)
print(f"MAE  = {np.mean(np.abs(e)):.1f}")
print(f"MSE  = {np.mean(e ** 2):.1f}")
print(f"RMSE = {np.sqrt(np.mean(e ** 2)):.2f}")
print(f"R^2  = {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):.4f}")

DATA = Path(__file__).resolve().parents[1] / "data"
house = pd.read_csv(DATA / "housing.csv").dropna()
X = house[["median_income", "housing_median_age", "latitude", "longitude"]]
y = house["median_house_value"] / 1000
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=465)
pred = LinearRegression().fit(X_tr, y_tr).predict(X_te)
mse = mean_squared_error(y_te, pred)
print(f"\nhousing test set: MAE {mean_absolute_error(y_te, pred):.1f}, "
      f"RMSE {np.sqrt(mse):.1f}, R^2 {r2_score(y_te, pred):.3f}")
print(f"baseline (predict the mean): RMSE "
      f"{np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)):.1f}")
