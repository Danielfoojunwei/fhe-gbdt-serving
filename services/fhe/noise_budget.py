"""
Production-Grade Noise Budget Tracking

Provides accurate noise budget estimation and management for FHE operations.
Critical for ensuring correctness of encrypted computation results.

Key Features:
- Pre-operation noise budget validation
- Per-operation noise consumption tracking
- Warning system for low noise budget
- Automatic bootstrapping triggers (for supported schemes)
- Operation chain analysis

References:
- SEAL Noise Budget: https://github.com/microsoft/SEAL/blob/main/native/examples/examples.h
- Noise Growth in FHE: https://eprint.iacr.org/2021/204
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class NoiseLevel(Enum):
    """Noise budget health levels."""
    HEALTHY = "healthy"      # > 50% budget remaining
    WARNING = "warning"      # 20-50% budget remaining
    CRITICAL = "critical"    # < 20% budget remaining
    EXHAUSTED = "exhausted"  # Budget depleted


@dataclass
class OperationCost:
    """Noise cost for different operations."""
    # Basic operations
    ADD_CIPHER_CIPHER: float = 0.5    # Negligible
    ADD_CIPHER_PLAIN: float = 0.1     # Very cheap
    NEGATE: float = 0.1               # Very cheap

    # Multiplication (expensive)
    MUL_CIPHER_CIPHER: float = 15.0   # Most expensive
    MUL_CIPHER_PLAIN: float = 3.0     # Moderate

    # Special operations
    ROTATION: float = 2.0             # Moderate
    RELINEARIZATION: float = 1.0      # After multiplication
    RESCALE: float = 40.0             # For CKKS (loses a level)

    # Compound operations
    DOT_PRODUCT: float = 18.0         # mul + rotations
    POLYVAL_PER_DEGREE: float = 12.0  # Per polynomial degree


@dataclass
class NoiseBudgetState:
    """Current state of noise budget."""
    initial_budget: float = 100.0
    current_budget: float = 100.0
    operations_history: List[Tuple[str, float, float]] = field(default_factory=list)
    warnings_issued: int = 0
    bootstraps_triggered: int = 0

    @property
    def remaining_percentage(self) -> float:
        """Percentage of budget remaining."""
        return (self.current_budget / self.initial_budget) * 100

    @property
    def level(self) -> NoiseLevel:
        """Current noise level."""
        pct = self.remaining_percentage
        if pct <= 0:
            return NoiseLevel.EXHAUSTED
        elif pct < 20:
            return NoiseLevel.CRITICAL
        elif pct < 50:
            return NoiseLevel.WARNING
        else:
            return NoiseLevel.HEALTHY


class NoiseBudgetTracker:
    """
    Production-grade noise budget tracking system.

    Monitors noise budget consumption across all FHE operations
    and provides warnings before budget exhaustion.

    Example:
        ```python
        tracker = NoiseBudgetTracker(initial_budget=100.0)

        # Before multiplication
        if tracker.can_perform("multiply"):
            tracker.record_operation("multiply")
        else:
            print("WARNING: Insufficient noise budget for multiplication")

        # Check current state
        state = tracker.get_state()
        print(f"Budget remaining: {state.remaining_percentage:.1f}%")
        ```
    """

    def __init__(
        self,
        initial_budget: float = 100.0,
        warning_threshold: float = 50.0,
        critical_threshold: float = 20.0,
        auto_warnings: bool = True
    ):
        """
        Initialize noise budget tracker.

        Args:
            initial_budget: Starting noise budget (arbitrary units)
            warning_threshold: Percentage at which to issue warnings
            critical_threshold: Percentage at which to issue critical warnings
            auto_warnings: Automatically log warnings
        """
        self._state = NoiseBudgetState(
            initial_budget=initial_budget,
            current_budget=initial_budget
        )
        self._costs = OperationCost()
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._auto_warnings = auto_warnings
        self._callbacks: List[Callable[[str, NoiseBudgetState], None]] = []

        logger.info(f"NoiseBudgetTracker initialized: budget={initial_budget}")

    def get_operation_cost(self, operation: str) -> float:
        """
        Get noise cost for an operation.

        Args:
            operation: Operation name

        Returns:
            Estimated noise cost
        """
        cost_map = {
            "add": self._costs.ADD_CIPHER_CIPHER,
            "add_plain": self._costs.ADD_CIPHER_PLAIN,
            "subtract": self._costs.ADD_CIPHER_CIPHER,
            "negate": self._costs.NEGATE,
            "multiply": self._costs.MUL_CIPHER_CIPHER,
            "multiply_plain": self._costs.MUL_CIPHER_PLAIN,
            "rotation": self._costs.ROTATION,
            "dot": self._costs.DOT_PRODUCT,
            "sum": self._costs.ROTATION * 4,  # log2(n) rotations
            "polyval": self._costs.POLYVAL_PER_DEGREE * 4,  # Assume degree 4
            "rescale": self._costs.RESCALE,
            "relinearize": self._costs.RELINEARIZATION,
        }

        return cost_map.get(operation.lower(), 5.0)  # Default cost

    def can_perform(self, operation: str, count: int = 1) -> bool:
        """
        Check if an operation can be performed without exhausting budget.

        Args:
            operation: Operation name
            count: Number of operations

        Returns:
            True if operation can be performed
        """
        cost = self.get_operation_cost(operation) * count
        return self._state.current_budget >= cost

    def record_operation(
        self,
        operation: str,
        custom_cost: Optional[float] = None
    ) -> NoiseBudgetState:
        """
        Record an operation and update budget.

        Args:
            operation: Operation name
            custom_cost: Override default cost

        Returns:
            Updated state
        """
        cost = custom_cost if custom_cost is not None else self.get_operation_cost(operation)
        previous_budget = self._state.current_budget

        self._state.current_budget = max(0, self._state.current_budget - cost)
        self._state.operations_history.append((
            operation,
            cost,
            self._state.current_budget
        ))

        # Check for warnings
        self._check_and_warn()

        # Trigger callbacks
        for callback in self._callbacks:
            callback(operation, self._state)

        return self._state

    def record_operations(
        self,
        operations: List[Tuple[str, int]]
    ) -> NoiseBudgetState:
        """
        Record multiple operations at once.

        Args:
            operations: List of (operation_name, count) tuples

        Returns:
            Updated state
        """
        for op_name, count in operations:
            for _ in range(count):
                self.record_operation(op_name)

        return self._state

    def _check_and_warn(self):
        """Check budget levels and issue warnings."""
        if not self._auto_warnings:
            return

        pct = self._state.remaining_percentage
        level = self._state.level

        if level == NoiseLevel.EXHAUSTED:
            logger.error(
                "NOISE BUDGET EXHAUSTED - Results will be incorrect! "
                "Consider reducing computation depth or using bootstrapping."
            )
            self._state.warnings_issued += 1

        elif level == NoiseLevel.CRITICAL and pct >= 0:
            if self._state.warnings_issued == 0 or len(self._state.operations_history) % 10 == 0:
                logger.warning(
                    f"CRITICAL: Noise budget at {pct:.1f}%. "
                    "Limited operations remaining."
                )
                self._state.warnings_issued += 1

        elif level == NoiseLevel.WARNING:
            if self._state.warnings_issued == 0:
                logger.warning(
                    f"WARNING: Noise budget at {pct:.1f}%. "
                    "Consider optimizing operation chain."
                )
                self._state.warnings_issued += 1

    def estimate_chain_cost(
        self,
        operations: List[str]
    ) -> Tuple[float, bool]:
        """
        Estimate cost of an operation chain.

        Args:
            operations: List of operation names

        Returns:
            Tuple of (total_cost, can_complete)
        """
        total_cost = sum(self.get_operation_cost(op) for op in operations)
        can_complete = self._state.current_budget >= total_cost

        return total_cost, can_complete

    def estimate_tree_cost(
        self,
        depth: int,
        num_features: int,
        sign_poly_degree: int = 7
    ) -> Tuple[float, bool]:
        """
        Estimate cost of one oblivious tree evaluation.

        Args:
            depth: Tree depth
            num_features: Number of features (for column packing)
            sign_poly_degree: Degree of sign polynomial

        Returns:
            Tuple of (total_cost, can_complete)
        """
        # Per-level: comparison (1 polyval) + depth multiplications for indicators
        comparison_cost = self._costs.POLYVAL_PER_DEGREE * sign_poly_degree / 2
        level_cost = comparison_cost + self._costs.ADD_CIPHER_PLAIN * 2

        # Leaf indicators: depth multiplications each
        num_leaves = 2 ** depth
        indicator_cost = depth * self._costs.MUL_CIPHER_CIPHER

        # Aggregation: num_leaves plaintext multiplications + additions
        agg_cost = (
            num_leaves * self._costs.MUL_CIPHER_PLAIN +
            num_leaves * self._costs.ADD_CIPHER_CIPHER
        )

        total_cost = depth * level_cost + num_leaves * indicator_cost + agg_cost
        can_complete = self._state.current_budget >= total_cost

        return total_cost, can_complete

    def estimate_ensemble_cost(
        self,
        num_trees: int,
        depth: int,
        num_features: int
    ) -> Tuple[float, int]:
        """
        Estimate cost of ensemble evaluation.

        Returns number of trees that can be evaluated.

        Args:
            num_trees: Total trees in ensemble
            depth: Tree depth
            num_features: Number of features

        Returns:
            Tuple of (total_cost, max_trees_possible)
        """
        tree_cost, _ = self.estimate_tree_cost(depth, num_features)
        max_trees = int(self._state.current_budget / tree_cost)
        total_cost = num_trees * tree_cost

        return total_cost, min(max_trees, num_trees)

    def reset(self, new_budget: Optional[float] = None):
        """
        Reset budget tracker.

        Args:
            new_budget: New initial budget (uses original if None)
        """
        budget = new_budget if new_budget is not None else self._state.initial_budget
        self._state = NoiseBudgetState(
            initial_budget=budget,
            current_budget=budget
        )
        logger.info(f"NoiseBudgetTracker reset: budget={budget}")

    def add_callback(
        self,
        callback: Callable[[str, NoiseBudgetState], None]
    ):
        """
        Add callback for operation events.

        Args:
            callback: Function called after each operation
        """
        self._callbacks.append(callback)

    def trigger_bootstrap(self) -> bool:
        """
        Trigger bootstrapping to refresh noise budget.

        Note: Actual bootstrapping requires scheme support (TFHE/CKKS).

        Returns:
            True if bootstrap was triggered
        """
        if self._state.level in [NoiseLevel.CRITICAL, NoiseLevel.EXHAUSTED]:
            logger.info("Triggering bootstrap to refresh noise budget")
            self._state.bootstraps_triggered += 1

            # Simulate bootstrap: restore budget minus bootstrap cost
            bootstrap_cost = 20.0  # Bootstrap has its own cost
            self._state.current_budget = self._state.initial_budget - bootstrap_cost

            return True

        return False

    def get_state(self) -> NoiseBudgetState:
        """Get current state."""
        return self._state

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        ops_by_type: Dict[str, int] = {}
        for op, _, _ in self._state.operations_history:
            ops_by_type[op] = ops_by_type.get(op, 0) + 1

        return {
            "initial_budget": self._state.initial_budget,
            "current_budget": self._state.current_budget,
            "remaining_percentage": self._state.remaining_percentage,
            "level": self._state.level.value,
            "total_operations": len(self._state.operations_history),
            "operations_by_type": ops_by_type,
            "warnings_issued": self._state.warnings_issued,
            "bootstraps_triggered": self._state.bootstraps_triggered,
        }


class AdaptiveNoiseManager:
    """
    Adaptive noise budget management.

    Automatically adjusts computation strategy based on remaining budget.
    """

    def __init__(self, tracker: NoiseBudgetTracker):
        """
        Initialize adaptive manager.

        Args:
            tracker: NoiseBudgetTracker instance
        """
        self.tracker = tracker
        self._strategies: Dict[NoiseLevel, str] = {
            NoiseLevel.HEALTHY: "full_precision",
            NoiseLevel.WARNING: "reduced_precision",
            NoiseLevel.CRITICAL: "minimal_computation",
            NoiseLevel.EXHAUSTED: "bootstrap_required",
        }

    def get_recommended_strategy(self) -> str:
        """Get recommended computation strategy."""
        level = self.tracker.get_state().level
        return self._strategies[level]

    def get_recommended_poly_degree(self) -> int:
        """Get recommended polynomial degree for sign approximation."""
        level = self.tracker.get_state().level

        if level == NoiseLevel.HEALTHY:
            return 7  # High accuracy
        elif level == NoiseLevel.WARNING:
            return 5  # Balanced
        else:
            return 3  # Fast, lower accuracy

    def should_prune_ensemble(self, num_trees: int, depth: int) -> Tuple[bool, int]:
        """
        Check if ensemble should be pruned to fit budget.

        Args:
            num_trees: Number of trees
            depth: Tree depth

        Returns:
            Tuple of (should_prune, recommended_trees)
        """
        _, max_trees = self.tracker.estimate_ensemble_cost(num_trees, depth, 10)

        if max_trees < num_trees:
            return True, max_trees
        return False, num_trees

    def plan_batched_inference(
        self,
        total_samples: int,
        num_trees: int,
        depth: int
    ) -> List[int]:
        """
        Plan batched inference to stay within budget.

        Args:
            total_samples: Total samples to process
            num_trees: Trees in ensemble
            depth: Tree depth

        Returns:
            List of batch sizes
        """
        tree_cost, _ = self.tracker.estimate_tree_cost(depth, 10)
        budget_per_sample = tree_cost * num_trees

        # Calculate max samples per batch
        max_samples = int(self.tracker.get_state().initial_budget / budget_per_sample)
        max_samples = max(1, max_samples)

        batches = []
        remaining = total_samples
        while remaining > 0:
            batch_size = min(max_samples, remaining)
            batches.append(batch_size)
            remaining -= batch_size

        return batches


# Convenience functions

def create_noise_tracker(
    initial_budget: float = 100.0
) -> NoiseBudgetTracker:
    """Create noise budget tracker."""
    return NoiseBudgetTracker(initial_budget=initial_budget)


def estimate_gbdt_budget(
    num_trees: int,
    depth: int,
    num_features: int
) -> float:
    """
    Estimate required noise budget for GBDT inference.

    Args:
        num_trees: Number of trees
        depth: Tree depth
        num_features: Number of features

    Returns:
        Estimated required budget
    """
    tracker = NoiseBudgetTracker(initial_budget=1000.0)
    total_cost, _ = tracker.estimate_ensemble_cost(num_trees, depth, num_features)
    return total_cost
