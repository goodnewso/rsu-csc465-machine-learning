"""Unit 13: user-based collaborative filtering on a small ratings table.

Six students rate six courses from 1 to 5 (0 = not rated). Predict
Ada's rating for CSC 465 from the students whose tastes match hers.
"""
import numpy as np
import pandas as pd

R = pd.DataFrame(
    [[5, 4, 0, 1, 2, 0],
     [4, 5, 4, 1, 1, 5],
     [5, 5, 4, 2, 1, 4],
     [1, 2, 1, 5, 5, 2],
     [2, 1, 2, 4, 5, 1],
     [4, 4, 5, 2, 0, 5]],
    index=["Ada", "Bayo", "Chidi", "Dami", "Ebi", "Femi"],
    columns=["CSC 309", "AIT 304", "DTS 201",
             "CYB 201", "ICT 305", "CSC 465"])
print(R, "\n")
M = R.replace(0, np.nan)
centred = M.sub(M.mean(axis=1), axis=0).fillna(0)   # remove each bias


def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


target, item = "Ada", "CSC 465"
sims = {s: cosine(centred.loc[target], centred.loc[s])
        for s in R.index if s != target and pd.notna(M.at[s, item])}
print("similarity of Ada to the others (mean-centred cosine):")
for s, v in sorted(sims.items(), key=lambda kv: -kv[1]):
    print(f"  {s:5s} {v:6.3f}")

k = 2
nbrs = sorted(sims, key=sims.get, reverse=True)[:k]
num = sum(sims[s] * centred.at[s, item] for s in nbrs)
den = sum(abs(sims[s]) for s in nbrs)
pred = M.loc[target].mean() + num / den
print(f"\nnearest {k} neighbours: {nbrs}")
print(f"predicted rating of {item} for {target}: {pred:.2f}")
