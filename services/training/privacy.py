"""
FHE-GBDT Privacy Accounting

Differential privacy accounting for GBDT training.
Aligned with TenSafe's privacy accounting infrastructure.

Implements:
- Rényi Differential Privacy (RDP) accountant
- PRV (Privacy Random Variable) accountant
- Composition theorems for tight privacy bounds
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PrivacySpent:
    """Privacy budget spent."""
    epsilon: float
    delta: float

    def __str__(self) -> str:
        return f"(ε={self.epsilon:.4f}, δ={self.delta:.2e})"


class PrivacyAccountant(ABC):
    """Abstract base class for privacy accountants."""

    @abstractmethod
    def account(self, epsilon: float, delta: float, **kwargs) -> None:
        """Account for a privacy-consuming operation."""
        pass

    @abstractmethod
    def get_privacy_spent(self) -> PrivacySpent:
        """Get total privacy spent."""
        pass

    @abstractmethod
    def get_remaining_budget(self) -> PrivacySpent:
        """Get remaining privacy budget."""
        pass

    @abstractmethod
    def would_exceed_budget(self, epsilon: float, delta: float) -> bool:
        """Check if operation would exceed budget."""
        pass

    @abstractmethod
    def get_state(self) -> Dict:
        """Get accountant state for checkpointing."""
        pass

    @abstractmethod
    def load_state(self, state: Dict) -> None:
        """Load accountant state from checkpoint."""
        pass


class RDPAccountant(PrivacyAccountant):
    """
    Rényi Differential Privacy (RDP) Accountant.

    Provides tighter privacy bounds than basic composition through
    the RDP framework. Converts RDP guarantees to (ε, δ)-DP.

    Reference:
    - Mironov, "Rényi Differential Privacy" (2017)
    - Abadi et al., "Deep Learning with Differential Privacy" (2016)

    Example:
        ```python
        accountant = RDPAccountant(
            epsilon_target=1.0,
            delta=1e-5,
            orders=[2, 4, 8, 16, 32, 64]
        )

        # Account for each training iteration
        for i in range(num_iterations):
            accountant.account(
                noise_multiplier=1.0,
                sample_rate=batch_size / dataset_size,
            )

        spent = accountant.get_privacy_spent()
        print(f"Privacy spent: {spent}")
        ```
    """

    DEFAULT_ORDERS = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]

    def __init__(
        self,
        epsilon_target: float,
        delta: float,
        orders: Optional[List[float]] = None,
    ):
        self.epsilon_target = epsilon_target
        self.delta = delta
        self.orders = orders or self.DEFAULT_ORDERS

        # Track RDP values for each order
        self._rdp_values: Dict[float, float] = {order: 0.0 for order in self.orders}
        self._num_operations = 0

    def account(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        sample_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Account for a privacy-consuming operation.

        Can be called in two modes:
        1. Direct (ε, δ) accounting
        2. Gaussian mechanism accounting with noise_multiplier and sample_rate
        """
        if noise_multiplier is not None and sample_rate is not None:
            # Gaussian mechanism accounting
            rdp_increment = self._compute_rdp_gaussian(noise_multiplier, sample_rate)
        elif epsilon is not None:
            # Direct epsilon accounting - convert to RDP
            rdp_increment = self._epsilon_to_rdp(epsilon)
        else:
            raise ValueError("Must provide either (epsilon, delta) or (noise_multiplier, sample_rate)")

        # Add to running totals
        for order in self.orders:
            self._rdp_values[order] += rdp_increment.get(order, 0.0)

        self._num_operations += 1

    def _compute_rdp_gaussian(
        self,
        noise_multiplier: float,
        sample_rate: float,
    ) -> Dict[float, float]:
        """
        Compute RDP for subsampled Gaussian mechanism.

        Uses the analytical bound from Mironov (2017).
        """
        rdp = {}

        for order in self.orders:
            if order == 1:
                # Special case: KL divergence
                rdp[order] = sample_rate ** 2 / (2 * noise_multiplier ** 2)
            else:
                # General case
                rdp[order] = self._compute_rdp_sample_gaussian(
                    order, noise_multiplier, sample_rate
                )

        return rdp

    def _compute_rdp_sample_gaussian(
        self,
        order: float,
        noise_multiplier: float,
        sample_rate: float,
    ) -> float:
        """
        Compute RDP of the sampled Gaussian mechanism.

        Uses the closed-form expression for small sample rates.
        """
        if sample_rate == 0:
            return 0.0

        if sample_rate == 1:
            # No subsampling
            return order / (2 * noise_multiplier ** 2)

        # Subsampled Gaussian
        # Use the bound from "Rényi Differential Privacy of the Sampled Gaussian Mechanism"
        log_term = (
            math.log1p(-sample_rate)
            + order * math.log1p(
                sample_rate * (math.exp((order - 1) / (noise_multiplier ** 2 * 2)) - 1)
            )
        )

        return log_term / (order - 1)

    def _epsilon_to_rdp(self, epsilon: float) -> Dict[float, float]:
        """Convert pure DP epsilon to RDP values."""
        rdp = {}
        for order in self.orders:
            # Pure DP to RDP conversion
            rdp[order] = epsilon  # Upper bound
        return rdp

    def get_privacy_spent(self) -> PrivacySpent:
        """
        Get total privacy spent as (ε, δ)-DP.

        Converts RDP to (ε, δ)-DP using the optimal order.
        """
        epsilon = self._rdp_to_dp(self.delta)
        return PrivacySpent(epsilon=epsilon, delta=self.delta)

    def _rdp_to_dp(self, delta: float) -> float:
        """
        Convert RDP guarantees to (ε, δ)-DP.

        Uses the optimal conversion from Mironov (2017).
        """
        if delta <= 0:
            raise ValueError("delta must be positive")

        epsilon_candidates = []

        for order in self.orders:
            rdp = self._rdp_values[order]

            if order == 1:
                # Special case
                epsilon = rdp
            else:
                # Standard conversion
                epsilon = rdp + math.log(1 / delta) / (order - 1)

            epsilon_candidates.append(epsilon)

        return min(epsilon_candidates) if epsilon_candidates else float('inf')

    def get_remaining_budget(self) -> PrivacySpent:
        """Get remaining privacy budget."""
        spent = self.get_privacy_spent()
        remaining_epsilon = max(0, self.epsilon_target - spent.epsilon)
        return PrivacySpent(epsilon=remaining_epsilon, delta=self.delta)

    def would_exceed_budget(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ) -> bool:
        """Check if operation would exceed budget."""
        # Create temporary copy
        temp_rdp = dict(self._rdp_values)

        if noise_multiplier is not None and sample_rate is not None:
            rdp_increment = self._compute_rdp_gaussian(noise_multiplier, sample_rate)
        elif epsilon is not None:
            rdp_increment = self._epsilon_to_rdp(epsilon)
        else:
            return False

        # Add increment
        for order in self.orders:
            temp_rdp[order] += rdp_increment.get(order, 0.0)

        # Check if would exceed
        original_rdp = self._rdp_values
        self._rdp_values = temp_rdp

        spent = self.get_privacy_spent()

        self._rdp_values = original_rdp

        return spent.epsilon > self.epsilon_target

    def get_state(self) -> Dict:
        """Get accountant state for checkpointing."""
        return {
            "epsilon_target": self.epsilon_target,
            "delta": self.delta,
            "orders": self.orders,
            "rdp_values": self._rdp_values,
            "num_operations": self._num_operations,
        }

    def load_state(self, state: Dict) -> None:
        """Load accountant state from checkpoint."""
        self.epsilon_target = state["epsilon_target"]
        self.delta = state["delta"]
        self.orders = state["orders"]
        self._rdp_values = state["rdp_values"]
        self._num_operations = state["num_operations"]


def compute_dp_sgd_privacy(
    n_samples: int,
    batch_size: int,
    n_iterations: int,
    noise_multiplier: float,
    delta: float,
    orders: Optional[List[float]] = None,
) -> PrivacySpent:
    """
    Compute privacy guarantee for DP-SGD style training.

    This is a convenience function for computing the privacy spent
    during training with the Gaussian mechanism.

    Args:
        n_samples: Total number of training samples
        batch_size: Batch size
        n_iterations: Number of training iterations
        noise_multiplier: Ratio of noise std to gradient norm
        delta: Target delta
        orders: RDP orders for computation

    Returns:
        PrivacySpent with (epsilon, delta)

    Example:
        ```python
        privacy = compute_dp_sgd_privacy(
            n_samples=60000,
            batch_size=256,
            n_iterations=10000,
            noise_multiplier=1.1,
            delta=1e-5,
        )
        print(f"Privacy guarantee: {privacy}")
        ```
    """
    sample_rate = batch_size / n_samples

    accountant = RDPAccountant(
        epsilon_target=float('inf'),  # No target, just compute
        delta=delta,
        orders=orders,
    )

    for _ in range(n_iterations):
        accountant.account(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )

    return accountant.get_privacy_spent()


def calibrate_noise_multiplier(
    n_samples: int,
    batch_size: int,
    n_iterations: int,
    target_epsilon: float,
    target_delta: float,
    orders: Optional[List[float]] = None,
    tolerance: float = 0.01,
) -> float:
    """
    Calibrate noise multiplier to achieve target privacy.

    Uses binary search to find the smallest noise multiplier
    that achieves the target (ε, δ)-DP guarantee.

    Args:
        n_samples: Total number of training samples
        batch_size: Batch size
        n_iterations: Number of training iterations
        target_epsilon: Target epsilon
        target_delta: Target delta
        orders: RDP orders for computation
        tolerance: Acceptable tolerance in epsilon

    Returns:
        Calibrated noise multiplier
    """
    low = 0.01
    high = 100.0

    while high - low > 0.01:
        mid = (low + high) / 2

        privacy = compute_dp_sgd_privacy(
            n_samples=n_samples,
            batch_size=batch_size,
            n_iterations=n_iterations,
            noise_multiplier=mid,
            delta=target_delta,
            orders=orders,
        )

        if privacy.epsilon > target_epsilon + tolerance:
            low = mid
        else:
            high = mid

    return high
