"""Unit 1: probability, Bayes' theorem and a hypothesis test.

(a) A malaria rapid test: what is P(malaria | positive)?
(b) A normal distribution of examination scores.
(c) A two-sample t-test: did a new teaching method change the scores?
"""
import numpy as np
from scipy import stats

# (a) Bayes' theorem
p_m = 0.10                      # prior: 10% of patients tested have malaria
sens, spec = 0.95, 0.90         # P(+ | m) and P(- | no m)
p_pos = sens * p_m + (1 - spec) * (1 - p_m)
print(f"P(positive) = {p_pos:.3f}")
print(f"P(malaria | positive) = {sens * p_m / p_pos:.3f}")

# (b) a normal distribution: mean 55, standard deviation 12
scores = stats.norm(55, 12)
print(f"P(score >= 70) = {1 - scores.cdf(70):.3f}")
print(f"score at the 90th percentile = {scores.ppf(0.90):.1f}")

# (c) a t-test on two groups of 40 students (simulated, seed fixed)
rng = np.random.default_rng(465)
old = rng.normal(55, 12, 40)
new = rng.normal(61, 12, 40)
t, p = stats.ttest_ind(new, old)
print(f"means: old {old.mean():.1f}, new {new.mean():.1f}")
print(f"t = {t:.2f}, p-value = {p:.4f}",
      "-> reject H0 at 5%" if p < 0.05 else "-> do not reject H0")
