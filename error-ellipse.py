#! /usr/bin/env python3
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy


def plot_ellipse(center, covariance, chi2=1.0, method="analytical", **kwargs):
    """Draws an error ellipse for the given covariance matrix and chi^2 value.
    """

    # Scaling of error ellipse.
    scale = numpy.sqrt(chi2)

    def plot_patch(lambda_plus, lambda_minus, theta):
        """Draws error ellipse using a patch object."""
        ellipse = Ellipse(
            xy = center,
            width = 2 * scale * numpy.sqrt(lambda_plus),
            height = 2 * scale * numpy.sqrt(lambda_minus),
            angle = numpy.degrees(theta),
            **kwargs
        )
        plt.gca().add_patch(ellipse)

    if method == "analytical":
        # Extract parameters from covariance matrix.
        sigma1_sq = covariance[0, 0]
        sigma2_sq = covariance[1, 1]
        cross_term = covariance[0, 1] + covariance[1, 0]

        # Apply analytical formulae.
        b = sigma1_sq + sigma2_sq
        delta = numpy.sqrt((sigma1_sq - sigma2_sq)**2 + cross_term**2)

        lambda_plus = (b + delta) / 2
        lambda_minus = (b - delta) / 2
        theta = numpy.arctan2(
            cross_term,
            sigma1_sq - sigma2_sq
        ) / 2

        # Draw the patch object.
        plot_patch(
            lambda_plus,
            lambda_minus,
            theta
        )

    elif method == "numerical":
        # Diagonalise the covariance matrix.
        lambdas, A = numpy.linalg.eig(covariance)

        # Extract ellipse parameters.
        lambda_plus, lambda_minus = lambdas
        theta = numpy.arctan2(
            A[1, 0],
            A[0, 0]
        )

        # Draw the patch object.
        plot_patch(
            lambda_plus,
            lambda_minus,
            theta
        )

    elif method == "parametric":
        # Generate a sampling of the unit circle.
        phi = numpy.linspace(0, 2 * numpy.pi, 3601)
        n = numpy.array((
            numpy.cos(phi),
            numpy.sin(phi),
        ))

        # Decompose the covariance matrix to a unit representation. That is,
        #
        # V = L^T L,
        #
        # where L is a triangular matrix. Note that informally L is the
        # square-root of the covariance.
        L = numpy.linalg.cholesky(covariance)

        # Apply the latter transformation to the unit circle.
        r = scale * (L @ n)

        # Draw the result.
        plt.plot(
            center[0] + r[0, :],
            center[1] + r[1, :],
            **kwargs
        )

    else:
        raise ValueError(f"bad method ({method})")


if __name__ == "__main__":
    # HFLAV average.
    center = numpy.array((-0.666, -0.311))
    covariance = numpy.array((
        (0.029**2, 0.288 * 0.029 * 0.030),
        (0.288 * 0.029 * 0.030, 0.030**2),
    ))

    # Overlay plot using different methods.
    plt.figure()
    plot_ellipse(center, covariance,
        color="blue", alpha=0.5, method="analytical", label="Analytical")
    plot_ellipse(center, covariance,
        fill=False, hatch="\\", method="numerical", label="Numerical")
    plot_ellipse(center, covariance,
        color="red", method="parametric", label="Parametric")
    plt.xlabel("S")
    plt.ylabel("C")
    plt.axis("equal")
    plt.legend()
    plt.show()
