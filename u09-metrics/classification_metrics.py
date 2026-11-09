"""Unit 9: the confusion matrix, precision, recall, F1 and ROC-AUC."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)
df = pd.read_csv(HERE.parent / "data" / "diabetes.csv")
X, y = df.drop(columns="Diabetes"), df["Diabetes"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=465)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
prob = model.predict_proba(X_te)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_te, prob >= 0.5).ravel()
print(f"TN {tn}  FP {fp}\nFN {fn}  TP {tp}")
print(f"accuracy  {(tp + tn) / len(y_te):.3f}")
print(f"precision {tp / (tp + fp):.3f}   recall {tp / (tp + fn):.3f}")
print(classification_report(y_te, prob >= 0.5, digits=3,
                            target_names=["no diabetes", "diabetes"]))
print(f"ROC-AUC {roc_auc_score(y_te, prob):.3f}")

print("\nthreshold  precision  recall")
for t in (0.2, 0.3, 0.5, 0.7):
    pred = prob >= t
    prec = (pred & (y_te == 1)).sum() / pred.sum()
    rec = (pred & (y_te == 1)).sum() / (y_te == 1).sum()
    print(f"   {t:.1f}       {prec:.3f}    {rec:.3f}")

fig, ax = plt.subplots(figsize=(4.5, 4.5))
RocCurveDisplay.from_predictions(y_te, prob, ax=ax)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
fig.tight_layout()
fig.savefig(HERE / "figures" / "u09_roc.png", dpi=150)
