"""Unit 7: support vector machines on the breast cancer data.

The Wisconsin Diagnostic Breast Cancer data ships with scikit-learn
(also on Kaggle: uciml/breast-cancer-wisconsin-data). We compare a
linear kernel with an RBF kernel; features must be scaled for SVMs.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)     # 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=465)
print("samples:", X.shape[0], "features:", X.shape[1])

for kernel, C in [("linear", 1), ("rbf", 1), ("rbf", 100)]:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C))
    model.fit(X_train, y_train)
    n_sv = model[-1].n_support_.sum()
    print(f"kernel={kernel:6s} C={C:<3d} test accuracy "
          f"{model.score(X_test, y_test):.3f}  support vectors {n_sv}")

unscaled = SVC(kernel="rbf").fit(X_train, y_train)
print(f"RBF without scaling: test accuracy "
      f"{unscaled.score(X_test, y_test):.3f}")
