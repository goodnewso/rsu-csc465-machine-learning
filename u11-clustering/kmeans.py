"""Unit 11: k-means clustering of penguins, choosing k.

The species labels are NOT used to cluster; afterwards we compare the
clusters with the species to see what the algorithm discovered.
"""
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
peng = pd.read_csv(DATA / "penguins.csv").dropna()
cols = ["culmen_length_mm", "culmen_depth_mm",
        "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(peng[cols])

print(" k   inertia  silhouette")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=465).fit(X)
    print(f"{k:2d} {km.inertia_:9.1f}  {silhouette_score(X, km.labels_):.3f}")

km = KMeans(n_clusters=3, n_init=10, random_state=465).fit(X)
peng["cluster"] = km.labels_
print("\nclusters (columns) against species (rows):")
print(pd.crosstab(peng["species"], peng["cluster"]))
print(f"adjusted Rand index: "
      f"{adjusted_rand_score(peng['species'], km.labels_):.3f}")
