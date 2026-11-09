"""Unit 2: plots with Matplotlib and Seaborn (saved to figures/)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # save files; no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
df = pd.read_csv(HERE.parent / "data" / "penguins.csv").dropna()

# Matplotlib: a histogram and a scatter plot side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(df["body_mass_g"], bins=20, color="#2a7f62")
ax[0].set(title="Body mass", xlabel="grams", ylabel="count")
for sp, g in df.groupby("species"):
    ax[1].scatter(g["flipper_length_mm"], g["body_mass_g"], s=12, label=sp)
ax[1].set(title="Flipper length vs body mass",
          xlabel="flipper length (mm)", ylabel="body mass (g)")
ax[1].legend()
fig.tight_layout()
fig.savefig(FIG / "u02_hist_scatter.png", dpi=150)

# Seaborn: every pair of measurements, coloured by species
g = sns.pairplot(df, hue="species", height=2.0,
                 vars=["culmen_length_mm", "culmen_depth_mm",
                       "flipper_length_mm", "body_mass_g"])
g.savefig(FIG / "u02_pairplot.png", dpi=120)
print("saved", sorted(p.name for p in FIG.glob("u02_*.png")))
