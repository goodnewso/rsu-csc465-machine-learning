"""Unit 3: finding and handling missing values."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

DATA = Path(__file__).resolve().parents[1] / "data"
peng = pd.read_csv(DATA / "penguins.csv")

print("sex values:", peng["sex"].value_counts(dropna=False).to_dict())
peng["sex"] = peng["sex"].replace(".", np.nan)    # "." is not a sex
num = ["culmen_length_mm", "culmen_depth_mm",
       "flipper_length_mm", "body_mass_g"]
empty = peng[num].isna().all(axis=1)
print("rows with no measurements at all:", int(empty.sum()))
peng = peng[~empty].copy()                        # drop them: nothing to use

# Fill each numeric column with its median, the sex with the mode.
peng[num] = SimpleImputer(strategy="median").fit_transform(peng[num])
peng["sex"] = peng["sex"].fillna(peng["sex"].mode()[0])
print("missing after cleaning:", int(peng.isna().sum().sum()),
      "| rows kept:", len(peng))

house = pd.read_csv(DATA / "housing.csv")
miss = house["total_bedrooms"].isna()
print("\nhousing rows missing total_bedrooms:", int(miss.sum()),
      f"({miss.mean():.1%})")
med = house["total_bedrooms"].median()
house["total_bedrooms"] = house["total_bedrooms"].fillna(med)
print("filled with the median:", med)
