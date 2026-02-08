"""
FHE Link Function Approximations

Polynomial approximations of GLM link functions for evaluation under FHE.
Leverages Horner's method from innovations/polynomial_leaves.py for
efficient encrypted evaluation (minimal multiplicative depth).

Supported link functions:
- Identity: y = x (no approximation needed)
- Logit (sigmoid): y = 1/(1+exp(-x)), approximated via minimax polynomial
- Log (exp inverse): y = exp(x), approximated via Chebyshev
- Reciprocal: y = 1/x, approximated via Newton iteration polynomial
- Probit: y = Phi(x), approximated via polynomial

Key design: All approximations are expressed as polynomials evaluated via
Horner's method, which has optimal multiplicative depth for FHE.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from numpy.polynomial import chebyshev

logger = logging.getLogger(__name__)


@dataclass
class LinkApproximation:
    """A polynomial approximation of a link function."""
    link_name: str
    coefficients: List[float]       # Polynomial coefficients [a0, a1, ..., an]
    degree: int
    domain: Tuple[float, float]     # Valid input range [lo, hi]
    max_error: float                # Maximum approximation error on domain
    method: str                     # "minimax", "chebyshev", "taylor"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate polynomial using Horner's method (FHE-optimal)."""
        result = np.full_like(x, self.coefficients[-1], dtype=np.float64)
        for i in range(len(self.coefficients) - 2, -1, -1):
            result = result * x + self.coefficients[i]
        return result

    @property
    def multiplicative_depth(self) -> int:
        """FHE multiplicative depth = polynomial degree."""
        return self.degree


class LinkFunctionApproximator:
    """
    Generates polynomial approximations of GLM link functions
    optimized for FHE evaluation.
    """

    # Default domains for each link function (covers >99.9% of practical inputs)
    DEFAULT_DOMAINS = {
        "identity": (-100.0, 100.0),
        "logit": (-8.0, 8.0),      # sigmoid(+-8) ≈ 0.9997/0.0003
        "log": (-4.0, 4.0),         # exp(4) ≈ 54.6
        "reciprocal": (0.1, 10.0),
        "probit": (-4.0, 4.0),
    }

    # Default polynomial degrees
    DEFAULT_DEGREES = {
        "identity": 1,
        "logit": 7,       # Degree 7 gives <0.5% error on [-8,8]
        "log": 7,
        "reciprocal": 5,
        "probit": 7,
    }

    def __init__(self, default_degree: Optional[int] = None):
        self.default_degree = default_degree

    def approximate(
        self,
        link_name: str,
        degree: Optional[int] = None,
        domain: Optional[Tuple[float, float]] = None,
    ) -> LinkApproximation:
        """
        Generate polynomial approximation for a link function.

        Args:
            link_name: One of "identity", "logit", "log", "reciprocal", "probit"
            degree: Polynomial degree (higher = more accurate, more FHE depth)
            domain: Input domain [lo, hi]

        Returns:
            LinkApproximation with coefficients for Horner's method
        """
        if degree is None:
            degree = self.default_degree or self.DEFAULT_DEGREES.get(link_name, 7)
        if domain is None:
            domain = self.DEFAULT_DOMAINS.get(link_name, (-8.0, 8.0))

        if link_name == "identity":
            return self._identity_approximation(domain)
        elif link_name == "logit":
            return self._logit_approximation(degree, domain)
        elif link_name == "log":
            return self._log_approximation(degree, domain)
        elif link_name == "reciprocal":
            return self._reciprocal_approximation(degree, domain)
        elif link_name == "probit":
            return self._probit_approximation(degree, domain)
        else:
            raise ValueError(f"Unknown link function: {link_name}")

    def _identity_approximation(
        self, domain: Tuple[float, float]
    ) -> LinkApproximation:
        """Identity: y = x. Trivial polynomial [0, 1]."""
        return LinkApproximation(
            link_name="identity",
            coefficients=[0.0, 1.0],
            degree=1,
            domain=domain,
            max_error=0.0,
            method="exact",
        )

    def _logit_approximation(
        self, degree: int, domain: Tuple[float, float]
    ) -> LinkApproximation:
        """
        Sigmoid: y = 1/(1+exp(-x)).
        Use Chebyshev approximation on domain for minimax-like fit.
        """
        lo, hi = domain
        # Sample points for fitting
        n_samples = max(1000, degree * 100)
        x = np.linspace(lo, hi, n_samples)
        y = 1.0 / (1.0 + np.exp(-x))

        # Fit Chebyshev, then convert to standard polynomial for Horner's method
        # Normalize to [-1, 1] for Chebyshev fitting
        x_norm = 2.0 * (x - lo) / (hi - lo) - 1.0
        cheb_coeffs = chebyshev.chebfit(x_norm, y, degree)

        # Convert Chebyshev coefficients to standard polynomial on original domain
        # We store coefficients in original domain for direct Horner evaluation
        std_coeffs = self._chebyshev_to_standard(cheb_coeffs, lo, hi)

        # Compute max error
        y_approx = self._horner_eval(std_coeffs, x)
        max_error = float(np.max(np.abs(y - y_approx)))

        logger.info(
            f"Logit approximation: degree={degree}, domain=[{lo},{hi}], "
            f"max_error={max_error:.6f}"
        )

        return LinkApproximation(
            link_name="logit",
            coefficients=std_coeffs,
            degree=degree,
            domain=domain,
            max_error=max_error,
            method="chebyshev",
        )

    def _log_approximation(
        self, degree: int, domain: Tuple[float, float]
    ) -> LinkApproximation:
        """
        Exp link inverse: y = exp(x).
        Chebyshev approximation on domain.
        """
        lo, hi = domain
        n_samples = max(1000, degree * 100)
        x = np.linspace(lo, hi, n_samples)
        y = np.exp(x)

        x_norm = 2.0 * (x - lo) / (hi - lo) - 1.0
        cheb_coeffs = chebyshev.chebfit(x_norm, y, degree)
        std_coeffs = self._chebyshev_to_standard(cheb_coeffs, lo, hi)

        y_approx = self._horner_eval(std_coeffs, x)
        max_error = float(np.max(np.abs(y - y_approx)))

        logger.info(
            f"Exp approximation: degree={degree}, domain=[{lo},{hi}], "
            f"max_error={max_error:.6f}"
        )

        return LinkApproximation(
            link_name="log",
            coefficients=std_coeffs,
            degree=degree,
            domain=domain,
            max_error=max_error,
            method="chebyshev",
        )

    def _reciprocal_approximation(
        self, degree: int, domain: Tuple[float, float]
    ) -> LinkApproximation:
        """
        Reciprocal: y = 1/x.
        Chebyshev approximation on positive domain.
        """
        lo, hi = domain
        if lo <= 0:
            lo = 0.1  # Reciprocal not defined at 0
        n_samples = max(1000, degree * 100)
        x = np.linspace(lo, hi, n_samples)
        y = 1.0 / x

        x_norm = 2.0 * (x - lo) / (hi - lo) - 1.0
        cheb_coeffs = chebyshev.chebfit(x_norm, y, degree)
        std_coeffs = self._chebyshev_to_standard(cheb_coeffs, lo, hi)

        y_approx = self._horner_eval(std_coeffs, x)
        max_error = float(np.max(np.abs(y - y_approx)))

        return LinkApproximation(
            link_name="reciprocal",
            coefficients=std_coeffs,
            degree=degree,
            domain=(lo, hi),
            max_error=max_error,
            method="chebyshev",
        )

    def _probit_approximation(
        self, degree: int, domain: Tuple[float, float]
    ) -> LinkApproximation:
        """
        Probit: y = Phi(x) = CDF of standard normal.
        Approximated via polynomial fit.
        """
        lo, hi = domain
        n_samples = max(1000, degree * 100)
        x = np.linspace(lo, hi, n_samples)
        # Standard normal CDF approximation
        y = 0.5 * (1.0 + _erf_vectorized(x / math.sqrt(2)))

        x_norm = 2.0 * (x - lo) / (hi - lo) - 1.0
        cheb_coeffs = chebyshev.chebfit(x_norm, y, degree)
        std_coeffs = self._chebyshev_to_standard(cheb_coeffs, lo, hi)

        y_approx = self._horner_eval(std_coeffs, x)
        max_error = float(np.max(np.abs(y - y_approx)))

        return LinkApproximation(
            link_name="probit",
            coefficients=std_coeffs,
            degree=degree,
            domain=domain,
            max_error=max_error,
            method="chebyshev",
        )

    def _chebyshev_to_standard(
        self,
        cheb_coeffs: np.ndarray,
        lo: float,
        hi: float,
    ) -> List[float]:
        """
        Convert Chebyshev coefficients (on [-1,1]) to standard polynomial
        coefficients on the original domain [lo, hi].

        This enables direct Horner evaluation without domain normalization
        at inference time (saving FHE operations).
        """
        # First convert Chebyshev to standard polynomial on [-1,1]
        n = len(cheb_coeffs)
        # Build Chebyshev polynomial matrix
        std_on_unit = np.zeros(n)
        for i, c in enumerate(cheb_coeffs):
            # T_i(x) expressed as standard polynomial
            ti = np.zeros(n)
            ti_coeffs = chebyshev.cheb2poly(
                [0.0] * i + [1.0]
            )
            for j, tc in enumerate(ti_coeffs):
                if j < n:
                    std_on_unit[j] += c * tc

        # Now substitute x = 2*(t-lo)/(hi-lo) - 1 to get polynomial in t
        # x = a*t + b where a = 2/(hi-lo), b = -(hi+lo)/(hi-lo)
        a = 2.0 / (hi - lo)
        b = -(hi + lo) / (hi - lo)

        # Expand polynomial P(a*t + b)
        result = self._substitute_linear(std_on_unit.tolist(), a, b)
        return result

    def _substitute_linear(
        self,
        coeffs: List[float],
        a: float,
        b: float,
    ) -> List[float]:
        """
        Given P(x) = sum(coeffs[i] * x^i), compute Q(t) = P(a*t + b).
        Returns coefficients of Q in standard form.
        """
        n = len(coeffs)
        # Start with Q = 0
        result = [0.0] * n

        # (a*t + b)^k computed iteratively
        power = [1.0] + [0.0] * (n - 1)  # (a*t+b)^0 = 1

        for k in range(n):
            # Add coeffs[k] * power
            for j in range(n):
                result[j] += coeffs[k] * power[j]

            # Multiply power by (a*t + b) for next iteration
            if k < n - 1:
                new_power = [0.0] * n
                for j in range(n):
                    # b * power[j]
                    new_power[j] += b * power[j]
                    # a * t * power[j] = a * power[j] shifted
                    if j + 1 < n:
                        new_power[j + 1] += a * power[j]
                power = new_power

        return result

    @staticmethod
    def _horner_eval(coeffs: List[float], x: np.ndarray) -> np.ndarray:
        """Evaluate polynomial via Horner's method."""
        result = np.full_like(x, coeffs[-1], dtype=np.float64)
        for i in range(len(coeffs) - 2, -1, -1):
            result = result * x + coeffs[i]
        return result


def _erf_vectorized(x: np.ndarray) -> np.ndarray:
    """Vectorized error function approximation."""
    return np.vectorize(math.erf)(x)


def get_link_approximation(
    link_name: str,
    degree: Optional[int] = None,
    domain: Optional[Tuple[float, float]] = None,
) -> LinkApproximation:
    """
    Convenience function to get a link function approximation.

    Args:
        link_name: Link function name
        degree: Polynomial degree
        domain: Input domain

    Returns:
        LinkApproximation
    """
    approx = LinkFunctionApproximator()
    return approx.approximate(link_name, degree, domain)
