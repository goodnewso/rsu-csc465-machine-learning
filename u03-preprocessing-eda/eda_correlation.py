"""Unit 3: exploratory data analysis and correlation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
house = pd.read_csv(HERE.parent / "data" / "housing.csv")
house["rooms_per_household"] = house["total_rooms"] / house["households"]
house["bedroom_ratio"] = house["total_bedrooms"] / house["total_rooms"]

print(house["ocean_proximity"].value_counts())
corr = house.select_dtypes("number").corr()
target = corr["median_house_value"].drop("median_house_value")
print("\ncorrelation with median_house_value:")
print(target.sort_values(ascending=False).round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".1f",
            annot_kws={"size": 7}, ax=ax)
fig.tight_layout()
fig.savefig(FIG / "u03_heatmap.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(house["median_income"], house["median_house_value"],
           s=2, alpha=0.2)
ax.set(xlabel="median income (tens of thousands of US$)",
       ylabel="median house value (US$)")
fig.tight_layout()
fig.savefig(FIG / "u03_income_value.png", dpi=150)
