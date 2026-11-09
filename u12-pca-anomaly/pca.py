"""Unit 12: principal component analysis on the breast cancer data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
X, y = load_breast_cancer(return_X_y=True)
Xs = StandardScaler().fit_transform(X)          # PCA needs scaled data

pca = PCA().fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
print("variance explained by PC1, PC2:",
      pca.explained_variance_ratio_[:2].round(3))
for target in (0.80, 0.90, 0.95):
    print(f"components for {target:.0%} of the variance:",
          int(np.argmax(cum >= target)) + 1)

for k in (2, 5, 10, 30):
    clf = make_pipeline(StandardScaler(), PCA(n_components=k),
                        LogisticRegression(max_iter=5000))
    acc = cross_val_score(clf, X, y, cv=5).mean()
    print(f"{k:2d} components -> 5-fold accuracy {acc:.3f}")

Z = pca.transform(Xs)[:, :2]
fig, ax = plt.subplots(figsize=(6, 4.5))
for lab, name in [(0, "malignant"), (1, "benign")]:
    ax.scatter(Z[y == lab, 0], Z[y == lab, 1], s=8, label=name)
ax.set(xlabel="PC1", ylabel="PC2", title="30 features projected to 2")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "u12_pca.png", dpi=150)
