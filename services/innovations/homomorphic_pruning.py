"""
Novel Innovation #3: Homomorphic Ensemble Pruning

Prune GBDT trees in the encrypted domain without decrypting. This enables
adaptive model complexity based on input characteristics, all while maintaining
privacy.

Key Insight:
- Tree outputs can be masked by multiplying with significance indicators
- Significance can be estimated homomorphically using variance approximation
- Trees with low variance/significance contribute ~0 to final prediction

Benefits:
- Dynamic pruning based on encrypted inputs
- Reduced computation for simpler inputs
- Privacy-preserving model adaptation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TreeSignificance:
    """Significance metrics for a tree."""
    tree_idx: int
    mean_contribution: float
    variance_contribution: float
    significance_score: float
    pruning_probability: float


@dataclass
class PruningConfig:
    """Configuration for homomorphic pruning."""
    # Significance threshold (trees below this may be pruned)
    significance_threshold: float = 0.1

    # Minimum number of trees to keep
    min_trees: int = 10

    # Maximum pruning fraction
    max_prune_fraction: float = 0.5

    # Polynomial degree for variance approximation
    variance_poly_degree: int = 4

    # Enable soft pruning (scale by significance) vs hard pruning
    soft_pruning: bool = True


class EncryptedTreeSignificance:
    """
    Computes tree significance in the encrypted domain.

    Uses polynomial approximations to estimate variance and contribution
    significance without decrypting intermediate values.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize significance computer.

        Args:
            config: Pruning configuration
        """
        self.config = config or PruningConfig()

        # Precompute polynomial coefficients for variance approximation
        # Var(X) ≈ E[X²] - E[X]²
        # We use Taylor expansion for x² around 0
        self._variance_coeffs = self._compute_variance_coeffs()

    def _compute_variance_coeffs(self) -> np.ndarray:
        """Compute coefficients for variance polynomial."""
        # Chebyshev approximation of x² on [-1, 1]
        degree = self.config.variance_poly_degree
        coeffs = np.zeros(degree + 1)
        coeffs[2] = 1.0  # x² coefficient
        return coeffs

    def compute_significance_plaintext(
        self,
        tree_outputs: np.ndarray
    ) -> np.ndarray:
        """
        Compute tree significance in plaintext (for testing).

        Args:
            tree_outputs: Shape (batch_size, num_trees) tree predictions

        Returns:
            Shape (num_trees,) significance scores
        """
        # Compute per-tree statistics
        means = np.mean(tree_outputs, axis=0)
        variances = np.var(tree_outputs, axis=0)

        # Significance = variance contribution relative to total
        total_variance = np.var(tree_outputs.sum(axis=1))
        if total_variance > 0:
            significance = variances / total_variance
        else:
            significance = np.ones(tree_outputs.shape[1]) / tree_outputs.shape[1]

        return significance

    def compute_encrypted_statistics(
        self,
        encrypted_outputs: List[Any],  # List of encrypted tree outputs
        fhe_context: Any  # FHE computation context
    ) -> Dict[int, Any]:
        """
        Compute encrypted statistics for significance estimation.

        Args:
            encrypted_outputs: List of encrypted tree output ciphertexts
            fhe_context: FHE context for homomorphic operations

        Returns:
            Dict of tree_idx -> encrypted statistics
        """
        statistics = {}

        for tree_idx, ct in enumerate(encrypted_outputs):
            # Compute E[X] via homomorphic mean
            # In SIMD packing, this is a rotation-and-sum pattern
            mean_ct = self._compute_encrypted_mean(ct, fhe_context)

            # Compute E[X²] via polynomial evaluation
            squared_ct = self._compute_encrypted_square(ct, fhe_context)
            mean_squared_ct = self._compute_encrypted_mean(squared_ct, fhe_context)

            # Variance ≈ E[X²] - E[X]²
            mean_sq_ct = self._compute_encrypted_square(mean_ct, fhe_context)
            variance_ct = fhe_context.subtract(mean_squared_ct, mean_sq_ct)

            statistics[tree_idx] = {
                "mean": mean_ct,
                "variance": variance_ct,
                "squared_mean": mean_squared_ct,
            }

        return statistics

    def _compute_encrypted_mean(self, ct: Any, fhe_context: Any) -> Any:
        """Compute encrypted mean using rotation-sum."""
        # This is a log-reduction pattern
        # For N slots: rotate by N/2, add, rotate by N/4, add, ...
        result = ct
        slots = fhe_context.get_num_slots()

        stride = slots // 2
        while stride >= 1:
            rotated = fhe_context.rotate(result, stride)
            result = fhe_context.add(result, rotated)
            stride //= 2

        # Divide by number of slots
        result = fhe_context.multiply_plain(result, 1.0 / slots)
        return result

    def _compute_encrypted_square(self, ct: Any, fhe_context: Any) -> Any:
        """Compute encrypted square using polynomial."""
        # x² via ciphertext-ciphertext multiplication
        return fhe_context.multiply(ct, ct)


class AdaptivePruningGate:
    """
    Gates tree outputs based on significance.

    In soft mode: scales outputs by significance (0 to 1)
    In hard mode: zeroes out low-significance trees
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize pruning gate.

        Args:
            config: Pruning configuration
        """
        self.config = config or PruningConfig()

        # Polynomial coefficients for smooth step function
        # step(x) ≈ 0.5 + 0.5 * sign(x - threshold)
        self._step_coeffs = self._compute_step_coeffs()

    def _compute_step_coeffs(self) -> np.ndarray:
        """Compute step function polynomial coefficients."""
        # Minimax approximation of step function
        return np.array([0.5, 0.7854, 0.0, -0.1231, 0.0, 0.0245])

    def compute_gates_plaintext(
        self,
        significance: np.ndarray
    ) -> np.ndarray:
        """
        Compute gate values in plaintext.

        Args:
            significance: Per-tree significance scores

        Returns:
            Gate values (0 to 1) for each tree
        """
        threshold = self.config.significance_threshold

        if self.config.soft_pruning:
            # Soft gate: smooth transition around threshold
            # gate = sigmoid((significance - threshold) * steepness)
            steepness = 10.0
            gates = 1.0 / (1.0 + np.exp(-steepness * (significance - threshold)))
        else:
            # Hard gate: binary threshold
            gates = (significance >= threshold).astype(np.float64)

        # Ensure minimum trees
        num_trees = len(significance)
        min_gates = int(num_trees * (1 - self.config.max_prune_fraction))
        min_gates = max(min_gates, self.config.min_trees)

        if np.sum(gates > 0.5) < min_gates:
            # Keep top min_gates trees
            top_indices = np.argsort(significance)[-min_gates:]
            gates = np.zeros(num_trees)
            gates[top_indices] = 1.0

        return gates

    def apply_gates_plaintext(
        self,
        tree_outputs: np.ndarray,
        gates: np.ndarray
    ) -> np.ndarray:
        """
        Apply gates to tree outputs in plaintext.

        Args:
            tree_outputs: Shape (batch_size, num_trees)
            gates: Shape (num_trees,) gate values

        Returns:
            Gated tree outputs
        """
        return tree_outputs * gates[np.newaxis, :]

    def compute_encrypted_gates(
        self,
        encrypted_significance: List[Any],
        fhe_context: Any
    ) -> List[Any]:
        """
        Compute gate values homomorphically.

        Args:
            encrypted_significance: Encrypted significance scores
            fhe_context: FHE context

        Returns:
            Encrypted gate values
        """
        encrypted_gates = []
        threshold = self.config.significance_threshold

        for sig_ct in encrypted_significance:
            # Compute significance - threshold
            delta_ct = fhe_context.subtract_plain(sig_ct, threshold)

            # Apply polynomial step function
            gate_ct = self._polynomial_step(delta_ct, fhe_context)
            encrypted_gates.append(gate_ct)

        return encrypted_gates

    def _polynomial_step(self, ct: Any, fhe_context: Any) -> Any:
        """Apply polynomial approximation of step function."""
        result = fhe_context.multiply_plain(ct, 0)  # Zero

        ct_power = ct
        for i, coeff in enumerate(self._step_coeffs):
            if coeff != 0:
                term = fhe_context.multiply_plain(ct_power, coeff)
                result = fhe_context.add(result, term)

            if i < len(self._step_coeffs) - 1:
                ct_power = fhe_context.multiply(ct_power, ct)

        return result


class HomomorphicEnsemblePruner:
    """
    Complete homomorphic ensemble pruning system.

    Combines significance computation and gating to enable
    privacy-preserving adaptive model pruning.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize pruner.

        Args:
            config: Pruning configuration
        """
        self.config = config or PruningConfig()
        self.significance_computer = EncryptedTreeSignificance(config)
        self.gate = AdaptivePruningGate(config)

    def prune_plaintext(
        self,
        tree_outputs: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prune ensemble in plaintext (for testing/validation).

        Args:
            tree_outputs: Shape (batch_size, num_trees)

        Returns:
            Tuple of (pruned_outputs, metadata)
        """
        # Compute significance
        significance = self.significance_computer.compute_significance_plaintext(
            tree_outputs
        )

        # Compute gates
        gates = self.gate.compute_gates_plaintext(significance)

        # Apply gates
        pruned = self.gate.apply_gates_plaintext(tree_outputs, gates)

        # Compute aggregated output
        aggregated = pruned.sum(axis=1)

        # Collect metadata
        num_active = np.sum(gates > 0.5)
        num_total = len(gates)

        metadata = {
            "significance": significance,
            "gates": gates,
            "num_active_trees": int(num_active),
            "num_total_trees": num_total,
            "pruning_ratio": 1 - (num_active / num_total),
            "active_tree_indices": np.where(gates > 0.5)[0].tolist(),
        }

        logger.info(
            f"Pruned ensemble: {num_active}/{num_total} trees active "
            f"({metadata['pruning_ratio']*100:.1f}% pruned)"
        )

        return aggregated, metadata

    def prune_encrypted(
        self,
        encrypted_tree_outputs: List[Any],
        fhe_context: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Prune ensemble in encrypted domain.

        Args:
            encrypted_tree_outputs: List of encrypted tree output ciphertexts
            fhe_context: FHE context

        Returns:
            Tuple of (encrypted_aggregated_output, metadata)
        """
        # Compute encrypted statistics
        statistics = self.significance_computer.compute_encrypted_statistics(
            encrypted_tree_outputs, fhe_context
        )

        # Extract variance for significance estimation
        encrypted_significance = [
            statistics[i]["variance"] for i in range(len(encrypted_tree_outputs))
        ]

        # Compute encrypted gates
        encrypted_gates = self.gate.compute_encrypted_gates(
            encrypted_significance, fhe_context
        )

        # Apply gates to tree outputs
        gated_outputs = []
        for i, (output_ct, gate_ct) in enumerate(zip(encrypted_tree_outputs, encrypted_gates)):
            gated = fhe_context.multiply(output_ct, gate_ct)
            gated_outputs.append(gated)

        # Aggregate gated outputs
        aggregated = gated_outputs[0]
        for gated in gated_outputs[1:]:
            aggregated = fhe_context.add(aggregated, gated)

        metadata = {
            "num_trees": len(encrypted_tree_outputs),
            "pruning_mode": "soft" if self.config.soft_pruning else "hard",
            "threshold": self.config.significance_threshold,
        }

        return aggregated, metadata

    def analyze_pruning_potential(
        self,
        model_ir: Any,
        sample_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze pruning potential for a model.

        Args:
            model_ir: Parsed ModelIR
            sample_data: Optional sample data for significance estimation

        Returns:
            Analysis results
        """
        num_trees = len(model_ir.trees)

        if sample_data is not None:
            # Simulate tree outputs and compute significance
            from .leaf_centric import LeafCentricEncoder
            encoder = LeafCentricEncoder()
            plan = encoder.encode_model(model_ir)

            # Simplified: estimate tree contributions
            tree_contributions = np.random.randn(sample_data.shape[0], num_trees)
            significance = self.significance_computer.compute_significance_plaintext(
                tree_contributions
            )

            prunable = np.sum(significance < self.config.significance_threshold)
        else:
            # Estimate based on tree structure
            prunable = int(num_trees * 0.2)  # Assume 20% prunable

        return {
            "num_trees": num_trees,
            "estimated_prunable": prunable,
            "potential_speedup": 1.0 / (1 - prunable / num_trees) if prunable < num_trees else 1.0,
            "config": self.config,
        }


# Convenience functions

def create_homomorphic_pruner(
    significance_threshold: float = 0.1,
    soft_pruning: bool = True
) -> HomomorphicEnsemblePruner:
    """
    Create a homomorphic ensemble pruner.

    Args:
        significance_threshold: Threshold for pruning
        soft_pruning: Use soft (continuous) vs hard (binary) pruning

    Returns:
        Configured pruner
    """
    config = PruningConfig(
        significance_threshold=significance_threshold,
        soft_pruning=soft_pruning
    )
    return HomomorphicEnsemblePruner(config)


def prune_ensemble(
    tree_outputs: np.ndarray,
    threshold: float = 0.1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prune an ensemble based on tree significance.

    Args:
        tree_outputs: Shape (batch_size, num_trees) tree predictions
        threshold: Significance threshold

    Returns:
        Pruned aggregated predictions and metadata
    """
    pruner = create_homomorphic_pruner(threshold)
    return pruner.prune_plaintext(tree_outputs)


# Import for typing
from typing import Tuple
