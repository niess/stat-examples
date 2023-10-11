#! /usr/bin/env python3
from typing import NamedTuple

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plot
import numpy
import scipy.linalg as linalg
import scipy.stats as stats



# =============================================================================
#
# 0. Preamble
#
# =============================================================================

# Let us first write down explictly the results of HFlav table using NamedTuple
# containers. That is

class Measurement(NamedTuple):
    """An experimental measurement."""
    observation: float
    stat_uncertainty: float
    syst_uncertainty: float

class Correlation(NamedTuple):
    """Correlations between measurements within a same experiment."""
    stat: float
    syst: float

class Experiment(NamedTuple):
    """Data relative to a given experiment."""
    name: str
    S: Measurement
    C: Measurement
    correlation: Correlation


experiments = [
    Experiment(
        name = "BaBar",
        S = Measurement(-0.68, 0.10, 0.03),
        C = Measurement(-0.25, 0.08, 0.02),
        correlation = Correlation(-0.06, syst = 0.00)
    ),
    Experiment(
        name = "Belle",
        S = Measurement(-0.64, 0.08, 0.03),
        C = Measurement(-0.33, 0.06, 0.03),
        correlation = Correlation(-0.10, syst = 0.00)
    ),
    Experiment(
        name = "LHCb",
        S = Measurement(-0.672, 0.034, 0.000),
        C = Measurement(-0.320, 0.038, 0.000),
        correlation = Correlation(0.405, syst = 0.000)
    ),
]


# Then, let us interpret data using a 2-dimensional statistical model, as
# following. The outcome of each experiment is assumed to behave as
#
# Obs = mu + Eps_stat + Eps_syst,
#
# where, mu stands for the true unknown value of (S, C), and where Eps_stat
# (Eps_syst) is a 2d r.v. representing the statistical (systematic) error.
# Furthermore, errors are assumed to be Gaussian distributed, as
#
# Eps = (G_C(sigma_S), G_S(sigma_C))
#
# whith standard deviations given by experimental uncertainties, and with an a
# priori non nul correlation factor, rho_{S,C}, within a same error category.
# However, stat. and syst. errors, as well as errors between different
# experiments, are considered as statistically independent.


# =============================================================================
#
# 1. Total covariance and graphical representation
#
# =============================================================================

# For a given experiment, the covariance of stat or syst. errors is a 2x2 matrix
# that can be obtained from the following function.

def covariance2d(sigma_S: float, sigma_C: float, rho_SC: float):
    """Builds the covariance matrix for a 2-dim observation."""

    # Create an empty 2x2 matrix.
    C = numpy.empty((2, 2))

    # Fill diagonal terms (i.e. variances).
    C[0, 0] = sigma_S**2
    C[1, 1] = sigma_C**2

    # Fill off-diagonal term (i.e. correlation).
    C[0, 1] = C[1, 0] = rho_SC * sigma_S * sigma_C

    return C

# Since stat and syst errors are not correlated, the total covariance matrix of
# `Eps = Eps_stat + Eps_syst` is simply the sum of individual covariances. Let
# us define a new function in order to perform this task.

def covariance(experiment: Experiment):
    """returns the total covariance matrix for a given experiment."""

    C_stat = covariance2d(
        experiment.S.stat_uncertainty,
        experiment.C.stat_uncertainty,
        experiment.correlation.stat
    )

    C_syst = covariance2d(
        experiment.S.syst_uncertainty,
        experiment.C.syst_uncertainty,
        experiment.correlation.syst
    )

    return C_stat + C_syst

# Under the assumption of Gaussian distributed errors, `Chi^2 = Eps^T C^{-1}
# Eps` follows a chi-square distribution with two degrees of freedom. The
# contour defined by `Chi^2 = 1` is an ellipse. The parameters of this ellipse
# can be solved analyticaly (see e.g. slides 56 & 57 of the Introduction to
# stat. methods). However, in the following we rely on alternative numeric
# method, using Cholesky decomposition. That is, any covariance matrix can be
# expressed as
#
# C = L^T L
#
# where L is a triangular matrix. Following, given a vector N of Normal
# distributed and independent variables, the vector `X = L N`, is Gaussian
# distributed, and it has the same covariance than Eps. Thus, X is actually
# equal to Eps. That is, we can express any set of correlated Gaussian r.v. as a
# linear combination of independent normal variables. Furthermore,
# substituting, we find `Chi^2 = N^T N`. That is, as previously stated, Chi^2
# indeed follows a chi-squared distribution.
#
# In practice, this implies that the contour ellipse for Eps can be derived from
# the contour on N (which is simply a circle in 2d), by the linear
# coordinates transformation L. We make use of this in the function below.

def plot_ellipse(experiment, chi2=1.0, **kwargs):
    """Plots the error ellipse for a given chi^2 value."""

    # Independent normal (2D) variables, sampled over the contour chi^2 = cst
    # i.e. a circle in this specific case.
    theta = numpy.linspace(0, 2 * numpy.pi, 10000)
    N = numpy.sqrt(chi2) * numpy.array((numpy.cos(theta), numpy.sin(theta)))

    # Compute the total covariance.
    C = covariance(experiment)

    # Cholesky decomposition of covariance matrix as C = L^T * L, where L is
    # a triangular matrix.
    L = numpy.linalg.cholesky(C)

    # Error contour corresponding to the total covariance, for the given chi^2
    # limit.
    Eps = L @ N

    # Plot the result, first the ellipse contour, then its centre.
    plot.plot(
        experiment.S.observation + Eps[0,:],
        experiment.C.observation + Eps[1,:],
        "-",
        **kwargs
    )
    plot.plot(
        experiment.S.observation,
        experiment.C.observation,
        "o",
        label = experiment.name,
        **kwargs
    )

# Let us use the previous function in order to produce an overlay plot of
# experimental results.

colors = ["green", "blue", "purple"]

plot.figure()
for i, experiment in enumerate(experiments):
    plot_ellipse(experiment, color=colors[i])
plot.axis("equal")
plot.xlabel("$S_{CP}$")
plot.ylabel("$C_{CP}$")
plot.title("$\chi^2 = 1$ confidence intervals")
plot.legend()
plot.show(block=False)


# =============================================================================
#
# 2. Confidence level.
#
# =============================================================================

# The previous ellipse contours define a 2d confidence interval of CL given by
# the CDF of the chi-square distribution, as

chi2, ndof = 1.0, 2
CL = stats.chi2.cdf(chi2, ndof)
print(f"CL[chi^2 = 1] = {100 * CL:.1f}%")

# In order to get a CL of 95%, we need to find the Chi^2 value corresponding to
# a CDF of 0.95. This can be obtained as

chi2_95 = stats.chi2.ppf(0.95, ndof)

print(f"Chi^2[CL = 95%] = {chi2_95:.1f}")

# Let us repeat the previous figure with the corresponding CL level. However,
# this time, let us define a function for this purpose.

def plot_overlay(experiments, CL=0.95):
    """Performs an overlay plot of experimental results."""

    ndof = 2
    chi2 = stats.chi2.ppf(CL, ndof)

    plot.figure()
    for i, experiment in enumerate(experiments):
        plot_ellipse(experiment, chi2=chi2, color=colors[i])
    plot.axis("equal")
    plot.xlabel("$S_{CP}$")
    plot.ylabel("$C_{CP}$")
    plot.title(r"$CL = {:g}$\% confidence intervals".format(100 * CL))
    plot.legend()
    plot.show(block=False)

plot_overlay(experiments)


# =============================================================================
#
# 3. Combination.
#
# =============================================================================

# See e.g. slide 35 of lesson 4.

def combine_lsq(experiments):
    """Combines experimental results according to Least-Squares method."""

    # Build matrices.
    n = len(experiments)
    A = numpy.vstack([numpy.eye(2) for _ in range(n)])
    V = linalg.block_diag(*[covariance(experiment)
                            for experiment in experiments])
    y = numpy.hstack(
        [(experiment.S.observation, experiment.C.observation)
         for experiment in experiments]
    )

    # Compute least square estimate. Note that the covariance of the LSQ
    # estimate, given by `V_LSQ = B^T V B`, can be shown to be (A^T V^-1 A)^-1.
    V_inv = numpy.linalg.inv(V)
    V_lsq = numpy.linalg.inv(A.T @ V_inv @ A)
    B = V_lsq @ A.T @ V_inv
    theta_lsq = B @ y

    # Pack results as an experimental combination.
    S = Measurement(
        observation = theta_lsq[0],
        stat_uncertainty = numpy.sqrt(V_lsq[0, 0]),
        syst_uncertainty = 0
    )
    C = Measurement(
        observation = theta_lsq[1],
        stat_uncertainty = numpy.sqrt(V_lsq[1, 1]),
        syst_uncertainty = 0
    )
    correlation = Correlation(
        stat = V_lsq[0, 1] / numpy.sqrt(V_lsq[0, 0] * V_lsq[1, 1]),
        syst = 0
    )

    return Experiment(
        name = "Combination",
        S = S,
        C = C,
        correlation = correlation
    )


# =============================================================================
#
# 4. Application.
#
# =============================================================================

# Combine experimetal results according to LSQ.
combination = combine_lsq(experiments)

# Pretty print the result.
print("Combination:")
print(f"S = {combination.S.observation:.3f} +- {combination.S.stat_uncertainty:.3f}")
print(f"C = {combination.C.observation:.3f} +- {combination.C.stat_uncertainty:.3f}")
print(f"correlation = {combination.correlation.stat:.3f}")

# Append the combination as an `experimental` result, and redo the overlay plot.
experiments.append(combination)
colors.append("red")
plot_overlay(experiments)

# Let us now check the consistency of experimental results. For this purpose,
# let us assume that the true parameter values (S, C) are given by the
# combination, and let us perform a 1-sided hypothesis test that all three
# experimental results are distributed accordingly. That is, the corresponding
# chi^2 value is T = sum_i{Y_i - Theta) V_i^-1 (Y_i - Theta)}. This can be
# computed as:

T = 0.0
for experiment in experiments[:-1]:
    delta = numpy.array((
        experiment.S.observation - combination.S.observation,
        experiment.C.observation - combination.C.observation,
    ))
    V_inv = numpy.linalg.inv(covariance(experiment))
    T += delta.T @ V_inv @ delta

# Under the null hpothesis, this test statistic follows a Chi^2 distribution of
# 6 (observations) - 2 (parameters) = 4 dof. Thus

ndof = 4

# The CL of this test is given as the frequency of events that exhibit a test
# value greater than the observed one. This is

CL = 1.0 - stats.chi2.cdf(T, ndof)

# It can also be expressed as an equivalent number of standard deviations for a
# 1d Gaussian distribution, as

sig = numpy.sqrt(stats.chi2.ppf(1.0 - CL, 1))

print(f"T = {T:.1f} / {ndof} (CL={CL:0.2f} -> {sig:.1f} sigma)")


# =============================================================================
#
# 5. Hypothesis testing.
#
# =============================================================================

# The last point is about testing the hypothesis S = 0, or C = 0 (or both). This
# could be done as in question 4., but substituting (0, 0) for the truth. That
# is

delta = numpy.array((
    combination.S.observation,
    combination.C.observation,
))
V_inv = numpy.linalg.inv(covariance(combination))
T = delta.T @ V_inv @ delta

# Instead, it is proposed to use a graphical method. That for, let us pack the
# hypothesis as a fake experimental result, for conveniency.

hypothesis = Experiment(
    name = "Hypothesis",
    S = Measurement(0.0, combination.S.stat_uncertainty, 0.0),
    C = Measurement(0.0, combination.C.stat_uncertainty, 0.0),
    correlation = Correlation(combination.correlation.stat, syst = 0.0)
)

# Then, the test value could be adjusted by trials and errors until the
# observation lays on the ellipse contour. This should yield the follozing plot.

plot.figure()
plot.plot(combination.S.observation, combination.C.observation, "ro",
          label="Observation")
plot_ellipse(hypothesis, chi2=T, color="black")
plot.axis("equal")
plot.xlabel("$S_{CP}$")
plot.ylabel("$C_{CP}$")
plot.legend()
plot.title("Testing the $(S=0, C=0)$ hypothesis")
plot.show(block=False)

# The corresponding CL and significance are numerically out of reach using
# scipy chi2 methods.

ndof = 2
CL = 1.0 - stats.chi2.cdf(T, ndof)
sig = numpy.sqrt(stats.chi2.ppf(1.0 - CL, 1))

# Dedicated algorithms would be needed in order to compute those.

print(f"T = {T:.1f} / {ndof} (CL={CL:0.2f} -> {sig:.1f} sigma)")


# The line below blocks execution until all figures are closed.
plot.show()
