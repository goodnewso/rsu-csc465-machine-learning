"""Unit 3: detecting outliers with the IQR rule and z-scores."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
house = pd.read_csv(DATA / "housing.csv")
rooms = house["total_rooms"] / house["households"]   # rooms per household

q1, q3 = rooms.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = (rooms < low) | (rooms > high)
print(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")
print(f"fences: [{low:.2f}, {high:.2f}]  outliers: {int(out.sum())}")
print("largest values:", rooms.nlargest(3).round(1).tolist())

z = (rooms - rooms.mean()) / rooms.std()
print("points with |z| > 3:", int((z.abs() > 3).sum()))

capped = rooms.clip(low, high)                       # one remedy: capping
print(f"mean before capping {rooms.mean():.2f}, after {capped.mean():.2f}")

top = (house["median_house_value"] >= 500000).sum()
print("\nhouse values at the 500,001 ceiling:", int(top),
      "- these are censored, not true values")
