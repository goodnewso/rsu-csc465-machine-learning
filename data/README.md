# Data sets

| File | Used in | Source | Kaggle page | Licence |
|---|---|---|---|---|
| `penguins.csv` | Units 2, 3, 6, 8, 11 | Palmer Station LTER, K. B. Gorman, T. D. Williams and W. R. Fraser (2014); `palmerpenguins` (A. M. Horst, A. P. Hill and K. B. Gorman, 2020) | [parulpandey/palmer-archipelago-antarctica-penguin-data](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) (`penguins_size.csv`) | CC0 |
| `housing.csv` | Units 3, 4, 9 | California Housing, 1990 US census; R. K. Pace and R. Barry (1997) | [camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | CC0 |
| `iris.csv` | exercises | R. A. Fisher (1936) | [uciml/iris](https://www.kaggle.com/datasets/uciml/iris) | CC0 |
| `sms_spam.csv` | Unit 7 | SMS Spam Collection, T. A. Almeida and J. M. Gómez Hidalgo (2011), UCI Machine Learning Repository, doi:10.24432/C5CC84 | [uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | CC BY 4.0 (UCI) |
| `diabetes.csv` | Unit 9 | CDC Behavioral Risk Factor Surveillance System 2015, cleaned by A. Teboul: 5,000 rows drawn at random (seed 465) from `diabetes_binary_health_indicators_BRFSS2015.csv` (13.6% diabetic, as in the population) | [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | CC0 |
| `diabetes_balanced.csv` | Units 6, 8, 10 | the same source: 5,000 rows drawn at random (seed 465) from `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (about 50% diabetic) | as above | CC0 |
| `courses.csv` | Unit 13 | Short course descriptions written for this course | — | MIT, as the code |
| `movielens/` (downloaded) | Unit 13 | MovieLens latest-small, F. M. Harper and J. A. Konstan (2015), GroupLens | search Kaggle for "MovieLens" | GroupLens terms: use permitted, redistribution not permitted, so it is downloaded, not stored here |
| built into scikit-learn | Units 7, 12, 14 | Breast Cancer Wisconsin (Diagnostic); handwritten digits (8 × 8) | [uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data); MNIST: [hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) | loaded through scikit-learn |
| simulated in code | Units 1, 5, 11, 12, 15 | random numbers with a fixed seed (e.g. mobile-money transfers in Unit 12) | — | — |

For real credit-card fraud data to try the Unit 12 methods on, see
[mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Open Database Licence; 150 MB, not included).

## Downloading from Kaggle yourself

1. Create a Kaggle account, open **Settings → API → Create New Token**.
2. `pip install kaggle`, then for example:

       kaggle datasets download -d camnugent/california-housing-prices --unzip -p data/downloads

Or simply click **Download** on the data set's Kaggle page.
