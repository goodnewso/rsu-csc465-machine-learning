# RSU-CSC 465: Machine Learning — course code

Companion code for the **RSU-CSC 465 Machine Learning** lecture note,
Department of Computer Science, Faculty of Computing, Rivers State
University, Port Harcourt.

**Authors:** Dr. Goodnews Ogboada Jaja and Professor Nuka D. Nwiabu

The lecture note prints only short extracts. Every complete, tested
program is here, in one folder per unit. Each program prints its results
and saves any figures to a `figures/` folder beside it.

| Unit | Folder | Topic |
|---|---|---|
| 1 | [u01-foundations](u01-foundations) | Linear algebra, gradients, probability, Bayes, hypothesis tests |
| 2 | [u02-python-tools](u02-python-tools) | NumPy, Pandas, Matplotlib, Seaborn |
| 3 | [u03-preprocessing-eda](u03-preprocessing-eda) | Missing values, outliers, scaling, encoding, splitting, correlation |
| 4 | [u04-linear-regression](u04-linear-regression) | Simple and multiple linear regression, gradient descent |
| 5 | [u05-regularised-regression](u05-regularised-regression) | Polynomial regression, Ridge, Lasso |
| 6 | [u06-logistic-knn](u06-logistic-knn) | Logistic regression, k-nearest neighbours |
| 7 | [u07-svm-naive-bayes](u07-svm-naive-bayes) | Support vector machines, Naive Bayes spam filter |
| 8 | [u08-trees-ensembles](u08-trees-ensembles) | Decision trees, random forests, bagging, boosting, XGBoost |
| 9 | [u09-metrics](u09-metrics) | MSE, RMSE, R², confusion matrix, precision, recall, F1, ROC-AUC |
| 10 | [u10-validation-tuning](u10-validation-tuning) | Bias and variance, k-fold cross-validation, grid and random search |
| 11 | [u11-clustering](u11-clustering) | k-means, hierarchical clustering, DBSCAN |
| 12 | [u12-pca-anomaly](u12-pca-anomaly) | Principal component analysis, anomaly detection |
| 13 | [u13-recommenders](u13-recommenders) | Content-based and collaborative filtering, matrix factorisation |
| 14 | [u14-neural-networks](u14-neural-networks) | Backpropagation, PyTorch tensors and autograd, an MLP |
| 15 | [u15-reinforcement](u15-reinforcement) | Markov decision processes, value iteration, Q-learning |

## 1. Get your own copy: fork, then clone

1. Sign in to GitHub (create a free account at github.com if you have none).
2. Open https://github.com/goodnewso/rsu-csc465-machine-learning and click
   **Fork** (top right), then **Create fork**. You now have your own copy at
   `https://github.com/YOUR-USERNAME/rsu-csc465-machine-learning`.
3. Clone **your fork** to your computer:

       git clone https://github.com/YOUR-USERNAME/rsu-csc465-machine-learning.git
       cd rsu-csc465-machine-learning

No GitHub account? Click **Code → Download ZIP** on the repository page,
or run `git clone https://github.com/goodnewso/rsu-csc465-machine-learning.git`.

## 2. Install Python and the libraries

You need Python 3.10 or later. In the repository folder:

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

PyTorch (Unit 14) is a large download (about 130 MB); everything else is
small. No GPU is needed for any program in this course.

## 3. Run a program

    cd u04-linear-regression
    python gradient_descent.py

To check everything at once: `python tests/run_all.py` (from the top folder).

## 4. No computer? Use Kaggle or Google Colab

Both run Python in a web browser and already have these libraries.

* **Kaggle:** sign in at kaggle.com → **Create → New Notebook**. In a cell run

      !git clone https://github.com/goodnewso/rsu-csc465-machine-learning.git
      %cd rsu-csc465-machine-learning/u06-logistic-knn
      !python logistic.py

  (Turn on **Internet** under *Session options* if the clone fails.)
* **Google Colab:** open colab.research.google.com → **New notebook** and run
  the same three lines.

## 5. Keep your fork up to date

New programs and corrections are added during the semester. On the page of
your fork, click **Sync fork → Update branch**. Or, on your computer:

    git remote add upstream https://github.com/goodnewso/rsu-csc465-machine-learning.git
    git fetch upstream
    git merge upstream/main

## 6. Save and share your own work

    git checkout -b lab04                   # a branch for your laboratory work
    # ... edit or add files ...
    git add u04-linear-regression/my_model.py
    git commit -m "Lab 4: multiple regression with my own features"
    git push -u origin lab04

Found a mistake in the course code? Open an **Issue** on the original
repository, or send a **Pull Request** from your fork
(**Contribute → Open pull request**).

## Data

The data sets are listed, with their sources, Kaggle pages and licences, in
[data/README.md](data/README.md). Small open-licence data sets are included
in `data/`; MovieLens is downloaded automatically by the program that uses it.

## Licence

The code is released under the MIT licence (see [LICENSE](LICENSE)).
Each data set keeps its own licence. The lecture note text is not covered.
