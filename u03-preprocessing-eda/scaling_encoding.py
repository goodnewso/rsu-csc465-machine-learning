"""Unit 3: feature scaling and encoding categorical data."""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler,
                                   OneHotEncoder, StandardScaler)

pd.set_option("display.width", 78)
DATA = Path(__file__).resolve().parents[1] / "data"
peng = pd.read_csv(DATA / "penguins.csv").dropna()
peng = peng[peng["sex"] != "."]
num = ["flipper_length_mm", "body_mass_g"]

mm = MinMaxScaler().fit_transform(peng[num])       # to the range [0, 1]
st = StandardScaler().fit_transform(peng[num])     # mean 0, std 1
print("raw ranges:", peng[num].min().tolist(), peng[num].max().tolist())
print("min-max first row:", mm[0].round(3))
print("standardised first row:", st[0].round(3),
      "| column means:", st.mean(axis=0).round(6) + 0.0)

le = LabelEncoder()                                 # for the target only
y = le.fit_transform(peng["species"])
print("\nlabel encoding:", dict(zip(le.classes_, range(3))))

ohe = OneHotEncoder(sparse_output=False)            # for input features
islands = ohe.fit_transform(peng[["island"]])
print("one-hot columns:", list(ohe.get_feature_names_out()))
print("first three rows:\n", islands[:3])
print("\nthe same with pandas:\n",
      pd.get_dummies(peng["island"], dtype=int).head(3))
