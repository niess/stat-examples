#! /usr/bin/env python3
from typing import NamedTuple

import matplotlib.pyplot as plot
import numpy
import scipy.stats as stats
import scipy.special as special


# =============================================================================
#
# 1. Generation
#
# =============================================================================

# First, let us wrap a pseudo random stream as a Uniform r.v., by defining a
# class as.

class Uniform:
    """Representation of a Uniformly distributed r.v."""

    def __init__(self):
        self._rng = numpy.random.default_rng()

    def __call__(self):
        """Returns a realisation of the r.v."""
        return self._rng.random()

# Then, let us instanciate two of these r.v. Note that these (pseudo) r.v. are
# (almost) independents.

U0, U1 = Uniform(), Uniform()

# A realisation of X_2 can be obtained as

def x2():
    return (U0() + U1()) / 2

# Let us generate 10^6 realisations of X_2.

samples = [x2() for _ in range(1000000)]

# Let us draw an histogram of the corresponding distribution. Note that we could
# directly use plot.hist(..., density=True). However, doing so would not show
# uncertainties on the PDF estimate. Thus, we define our own Histogram class,
# wrapping numpy.histogram, as

class Histogram(NamedTuple):
    """Histogramed samples, with errorbars."""

    x: numpy.ndarray
    y: numpy.ndarray
    xerr: numpy.ndarray
    yerr: numpy.ndarray

    @classmethod
    def new(cls, samples):
        """Generates a new histogram from samples."""

        N = len(samples)
        counts, edges = numpy.histogram(samples, bins=41)
        width = edges[1:] - edges[:-1]
        y = counts / (width * N)
        yerr = numpy.sqrt(counts) / (width * N)
        x = 0.5 * (edges[1:] + edges[:-1])
        xerr = [x - edges[:-1], edges[1:] - x]

        return cls(x, y, xerr, yerr)

    def errorbar(self, **kwargs):
        """Generates an errorbar plot of this histogram."""

        plot.errorbar(self.x, self.y, xerr=self.xerr, yerr=self.yerr, **kwargs)

# Let us use the previous class, now.

histo = Histogram.new(samples)

plot.figure()
histo.errorbar(fmt="k.", label="Histogram")
plot.xlabel("$X_2$")
plot.ylabel("pdf")
plot.show(block=False)


# =============================================================================
#
# 2. Bates distribution.
#
# =============================================================================

# Let us implement Bates pdf as function, as for now.

def bates_pdf(x, n):
    """Returns the pdf of the Bates distribution."""

    pdf = 0.0
    for k in range(n + 1):
        delta = n * x - k
        pdf += (-1)**k * special.binom(n, k) * delta**(n - 1) * numpy.sign(delta)
    return n * pdf / (2 * numpy.math.factorial(n - 1))

# Then, let us superimpose the PDF values on the previous histogram.

x = histo.x
pdf = [bates_pdf(xi, 2) for xi in x]
plot.show(block=False)

plot.plot(x, pdf, "r-", label="Bates")

# =============================================================================
#
# 3. Fitting with a Normal law.
#
# =============================================================================

# Let us first use scipy package in order to perform this fit, and let us
# superimpose the result on the previous plot.

mu, sigma = stats.norm.fit(samples)
plot.plot(x, stats.norm.pdf(x, mu, sigma), "k--", label="Normal")
plot.show(block=False)

# As a comparison, let us compute the sample mean and sample varaince:

sample_mean = numpy.mean(samples)
sample_std = numpy.std(samples)

print(f"mu    = {mu:.7E} ({sample_mean:.7E})")
print(f"sigma = {sigma:.7E} ({sample_std:.7E})")

# Does are exactly the same, as shown below. That is, in the case of a Normal
# distribution, the MLE is equivalent to setting mu and sigma parameters to
# the sample mean and variance estimates.

assert(mu == sample_mean)
assert(sigma == sample_std)

# The estimated value of n can be obtained from the expression of Bates
# variancem, as

n_hat = numpy.round(1.0 / (12.0 * sample_std**2))
print(f"n_hat = {n_hat:.0f}")

plot.legend()
plot.show()
