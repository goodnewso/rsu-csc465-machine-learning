"""Unit 11: DBSCAN finds clusters of any shape, and marks noise.

Two interleaving half-moons defeat k-means but not DBSCAN.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
X, y = make_moons(n_samples=300, noise=0.07, random_state=465)
X = np.vstack([X, [[-1.2, 1.3], [2.3, -0.8], [0.5, 1.6]]])   # 3 strays
y = np.append(y, [-1, -1, -1])

km = KMeans(n_clusters=2, n_init=10, random_state=465).fit(X)
db = DBSCAN(eps=0.2, min_samples=5).fit(X)
print(f"k-means ARI: {adjusted_rand_score(y, km.labels_):.3f}")
print(f"DBSCAN  ARI: {adjusted_rand_score(y, db.labels_):.3f}")
print("DBSCAN clusters found:", len(set(db.labels_) - {-1}),
      "| points marked as noise:", int((db.labels_ == -1).sum()))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, lab, t in [(ax[0], km.labels_, "k-means (k = 2)"),
                  (ax[1], db.labels_, "DBSCAN (eps = 0.2)")]:
    a.scatter(X[:, 0], X[:, 1], c=lab, cmap="viridis", s=12)
    a.set_title(t)
fig.tight_layout()
fig.savefig(FIG / "u11_dbscan.png", dpi=150)
