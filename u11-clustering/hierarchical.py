"""Unit 11: agglomerative (hierarchical) clustering and a dendrogram."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
peng = pd.read_csv(HERE.parent / "data" / "penguins.csv").dropna()
cols = ["culmen_length_mm", "culmen_depth_mm",
        "flipper_length_mm", "body_mass_g"]
sample = peng.groupby("species").sample(10, random_state=465)
X = StandardScaler().fit_transform(sample[cols])

for method in ("single", "complete", "average", "ward"):
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=3, criterion="maxclust")
    table = pd.crosstab(sample["species"].values, labels)
    pure = table.max(axis=0).sum() / len(sample)
    print(f"{method:8s} linkage: purity of 3 clusters = {pure:.2f}")

Z = linkage(X, method="ward")
fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Z, labels=sample["species"].str[:3].tolist(), ax=ax,
           leaf_font_size=8)
ax.set(title="Ward linkage on 30 penguins", ylabel="merge distance")
fig.tight_layout()
fig.savefig(FIG / "u11_dendrogram.png", dpi=150)
