"""Unit 2: Pandas DataFrames: load, inspect, select, filter, group."""
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 78)       # fit printed output to a page
pd.set_option("display.max_columns", 10)

DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "penguins.csv")

print(df.head(3))
print("\nshape:", df.shape)
print("\ncolumn types:\n", df.dtypes)
print("\nmissing values per column:\n", df.isna().sum())

print("\npenguins per species:\n", df["species"].value_counts())
heavy = df[df["body_mass_g"] > 5500]       # boolean filtering
print("\npenguins heavier than 5.5 kg:", len(heavy),
      "- species:", heavy["species"].unique())

summary = (df.groupby("species")[["flipper_length_mm", "body_mass_g"]]
             .mean().round(1))
print("\nmean flipper length and body mass by species:\n", summary)
