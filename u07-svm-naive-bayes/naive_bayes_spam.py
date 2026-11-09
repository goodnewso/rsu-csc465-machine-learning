"""Unit 7: a Naive Bayes spam filter for SMS messages.

Data: SMS Spam Collection (Almeida and Hidalgo, UCI, CC BY 4.0; also on
Kaggle as uciml/sms-spam-collection-dataset).
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA = Path(__file__).resolve().parents[1] / "data"
sms = pd.read_csv(DATA / "sms_spam.csv")
print(sms["label"].value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    sms["text"], sms["label"], test_size=0.25, stratify=sms["label"],
    random_state=465)
vec = CountVectorizer(lowercase=True)          # bag of words
Xtr = vec.fit_transform(X_train)               # learn the vocabulary
Xte = vec.transform(X_test)
print("vocabulary size:", len(vec.vocabulary_))

nb = MultinomialNB(alpha=1.0).fit(Xtr, y_train)   # Laplace smoothing
print(f"test accuracy: {nb.score(Xte, y_test):.3f}")

words = vec.get_feature_names_out()
log_ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
print("most 'spammy' words:", list(words[log_ratio.argsort()[-8:]]))

new = ["Congratulations! You have won N500,000. Call now to claim",
       "Please bring the lab manual to class tomorrow"]
for text, p in zip(new, nb.predict_proba(vec.transform(new))[:, 1]):
    print(f"P(spam) = {p:.3f}  <- {text}")
