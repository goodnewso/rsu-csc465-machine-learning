"""Unit 3: train-validation-test split, and scaling without leakage."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
peng = pd.read_csv(DATA / "penguins.csv").dropna()
X = peng[["culmen_length_mm", "culmen_depth_mm",
          "flipper_length_mm", "body_mass_g"]]
y = peng["species"]

# 60% train, 20% validation, 20% test; stratify keeps species shares.
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=465)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=465)
for name, part in [("train", y_train), ("validation", y_val),
                   ("test", y_test)]:
    shares = part.value_counts(normalize=True).round(2).to_dict()
    print(f"{name:10s} {len(part):3d} rows  {shares}")

scaler = StandardScaler().fit(X_train)       # learn mean/std on train ONLY
X_val_s = scaler.transform(X_val)            # then apply to the others
print("\ntrain means used for scaling:", scaler.mean_.round(1))
print("validation means after scaling (close to, not exactly, 0):",
      X_val_s.mean(axis=0).round(2))
