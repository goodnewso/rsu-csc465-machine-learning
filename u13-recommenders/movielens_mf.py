"""Unit 13: matrix factorisation on MovieLens (100,000 ratings).

The data (F. M. Harper and J. A. Konstan, GroupLens, 2015) may be used
for teaching but not redistributed, so this program downloads it from
GroupLens the first time it runs. Kaggle page: search "MovieLens".
Each user u and film i get k hidden factors p_u and q_i; the predicted
rating is mu + b_u + b_i + p_u . q_i, fitted by stochastic gradient
descent on the training ratings.
"""
import io
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DIR = Path(__file__).resolve().parents[1] / "data" / "movielens"
if not (DIR / "ratings.csv").exists():
    print("downloading MovieLens from GroupLens ...")
    raw = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in ("ratings.csv", "movies.csv"):
            (DIR / name).parent.mkdir(parents=True, exist_ok=True)
            (DIR / name).write_bytes(z.read(f"ml-latest-small/{name}"))
r = pd.read_csv(DIR / "ratings.csv")
movies = pd.read_csv(DIR / "movies.csv").set_index("movieId")["title"]
print(f"{len(r):,} ratings, {r.userId.nunique()} users, "
      f"{r.movieId.nunique():,} films")

u = r.userId.astype("category").cat.codes.to_numpy()
i = r.movieId.astype("category").cat.codes.to_numpy()
films = r.movieId.astype("category").cat.categories
y = r.rating.to_numpy()
rng = np.random.default_rng(465)
test = rng.random(len(y)) < 0.2
mu = y[~test].mean()


def rmse(pred):
    return np.sqrt(np.mean((y[test] - pred[test]) ** 2))


base = rmse(np.full_like(y, mu))
print(f"RMSE when always predicting the mean {mu:.2f}: {base:.3f}")

k, lr, reg = 20, 0.01, 0.05
P = rng.normal(0, 0.1, (u.max() + 1, k))
Q = rng.normal(0, 0.1, (i.max() + 1, k))
bu, bi = np.zeros(u.max() + 1), np.zeros(i.max() + 1)
train = np.flatnonzero(~test)
for epoch in range(1, 21):
    for t in rng.permutation(train):
        a, b = u[t], i[t]
        e = y[t] - (mu + bu[a] + bi[b] + P[a] @ Q[b])
        bu[a] += lr * (e - reg * bu[a])
        bi[b] += lr * (e - reg * bi[b])
        P[a], Q[b] = (P[a] + lr * (e * Q[b] - reg * P[a]),
                      Q[b] + lr * (e * P[a] - reg * Q[b]))
    if epoch in (1, 5, 10, 20):
        pred = mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1)
        print(f"epoch {epoch:2d}: test RMSE {rmse(pred):.3f}")

seen = set(i[u == 0])
scores = mu + bu[0] + bi + Q @ P[0]
best = [j for j in np.argsort(-scores) if j not in seen][:5]
print("top 5 unseen films for user 1:")
for j in best:
    print(f"  {scores[j]:.2f}  {movies[films[j]]}")
