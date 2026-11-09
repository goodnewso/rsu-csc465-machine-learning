"""Unit 13: content-based recommendation of courses.

Each course is described by words; TF-IDF turns the words into a
vector and cosine similarity finds the courses most like a given one.
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parents[1] / "data"
courses = pd.read_csv(DATA / "courses.csv")
text = courses["title"] + " " + courses["description"]
tfidf = TfidfVectorizer(stop_words="english")
V = tfidf.fit_transform(text)                  # one row per course
print("courses:", V.shape[0], "| distinct terms:", V.shape[1])
S = cosine_similarity(V)


def similar(code, n=3):
    i = courses.index[courses["code"] == code][0]
    best = S[i].argsort()[::-1][1:n + 1]       # skip the course itself
    return [(courses.at[j, "code"], courses.at[j, "title"],
             round(S[i, j], 3)) for j in best]


for code in ("CSC 465", "CSC 309", "CYB 201"):
    print(f"\nstudents who liked {code} may like:")
    for c, t, s in similar(code):
        print(f"  {c:8s} {t:34s} similarity {s}")
