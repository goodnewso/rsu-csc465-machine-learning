"""Unit 12: anomaly detection on mobile-money transfers.

The transfers are SIMULATED (seed fixed): 1,000 ordinary transfers and
12 planted frauds: amounts that are high but not extreme, sent between
midnight and 4 a.m. to new recipients.
For real card-fraud data see Kaggle: mlg-ulb/creditcardfraud.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(465)
n = 1000
amount = rng.lognormal(mean=9.0, sigma=0.8, size=n)     # naira
hour = rng.normal(14, 3.5, n).clip(0, 23.9)
new_recipient = rng.random(n) < 0.15
fraud_amt = rng.uniform(30_000, 90_000, 12)
amount = np.append(amount, fraud_amt)
hour = np.append(hour, rng.uniform(0.5, 4.0, 12))
new_recipient = np.append(new_recipient, np.ones(12, bool))
is_fraud = np.r_[np.zeros(n, bool), np.ones(12, bool)]
X = np.column_stack([np.log(amount), hour, new_recipient])

z = (np.log(amount) - np.log(amount).mean()) / np.log(amount).std()
flag_z = np.abs(z) > 3
print(f"z-score rule:      flagged {flag_z.sum():3d}, "
      f"frauds caught {np.sum(flag_z & is_fraud)} of 12")

iso = IsolationForest(contamination=0.015, random_state=465).fit(X)
flag_iso = iso.predict(X) == -1
print(f"isolation forest:  flagged {flag_iso.sum():3d}, "
      f"frauds caught {np.sum(flag_iso & is_fraud)} of 12")
top = np.argsort(iso.score_samples(X))[:5]
for i in top:
    print(f"  N{amount[i]:>9,.0f} at {hour[i]:4.1f}h  "
          f"new={bool(new_recipient[i])}  fraud={bool(is_fraud[i])}")
