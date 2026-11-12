# Baseline predictors. Each returns one curve of 11 percentages.

import numpy as np

from soilgsd.config import N_SIEVES


# Organisers' constant baseline: equal mass in each of the 11 size classes,
# i.e. [9.09, 18.18, ..., 100.0]. Scores 102.0 on the training labels.
# The organisers do not publish the curve itself. We picked this one because it
# is the simple constant that reproduces their reported 102.0 (measured 102.02);
# a straight line from 0 gives 95.7 and does not match.
def constant_curve():
    return 100.0 * np.arange(1, N_SIEVES + 1) / N_SIEVES


# Mean of the training curves, sieve by sieve. Pass training-fold soils only.
# In: (n_soils, 11) percent. Out: (11,) percent.
def mean_curve(P_train_pct):
    return np.asarray(P_train_pct, dtype=float).mean(axis=0)


# GRAI-style direct regression to two Weibull parameters.
def weibull_regression(*args, **kwargs):
    raise NotImplementedError("TODO: Weibull regression baseline")


# Curve from SAM grain measurements alone, no learned part.
def sam_only(*args, **kwargs):
    raise NotImplementedError("TODO: SAM-only baseline")


# SAM coarse grains with a Rosin-Rammler or Swebrec tail into the fines.
def parametric_tail(*args, **kwargs):
    raise NotImplementedError("TODO: parametric tail baseline")


# Regression on texture statistics only.
def texture_only(*args, **kwargs):
    raise NotImplementedError("TODO: texture-only baseline")
