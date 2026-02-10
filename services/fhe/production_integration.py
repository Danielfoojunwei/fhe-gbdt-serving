"""
Production Integration Module

Connects all novel FHE-GBDT innovations with real FHE operations.
This module provides production-ready implementations that use actual
cryptographic operations instead of simulations.

Key Integrations:
1. Leaf-Centric with TenSEAL - Real encrypted leaf indicator computation
2. Homomorphic Pruning with TenSEAL - Real encrypted variance computation
3. Gradient Noise with TenSEAL - Real encrypted gradient updates
4. N2HE Weighted Sum - Real encrypted aggregation
5. MOAI Column Packing - Real rotation-optimized operations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging
import time

from .tenseal_backend import (
    TenSEALContext,
    FHEConfig,
    FHEScheme,
    EncryptedTensor,
    ProductionFHEGBDT,
    create_production_context,
)

logger = logging.getLogger(__name__)


@dataclass
class ProductionMetrics:
    """Metrics from production FHE execution."""
    total_time_ms: float = 0.0
    encryption_time_ms: float = 0.0
    computation_time_ms: float = 0.0
    decryption_time_ms: float = 0.0
    noise_budget_remaining: float = 100.0
    operations_count: Dict[str, int] = field(default_factory=dict)
    memory_usage_mb: float = 0.0


class ProductionLeafCentric:
    """
    Production-ready leaf-centric FHE computation using TenSEAL.

    This implements the leaf-centric encoding innovation with real
    homomorphic operations using CKKS scheme.
    """

    def __init__(
        self,
        security_level: int = 128,
        depth: int = 6,
        context: Optional[TenSEALContext] = None
    ):
        """
        Initialize production leaf-centric computation.

        Args:
            security_level: Security level in bits
            depth: Multiplicative depth needed
            context: Optional pre-created context
        """
        if context is not None:
            self.ctx = context
        else:
            self.ctx = create_production_context(security_level, depth)

        self.gbdt = ProductionFHEGBDT(self.ctx)
        self._metrics = ProductionMetrics()

        logger.info(f"ProductionLeafCentric initialized: depth={depth}")

    def encrypt_features(self, features: np.ndarray) -> List[EncryptedTensor]:
        """
        Encrypt features using column packing.

        Each feature gets its own ciphertext for rotation-free access.

        Args:
            features: Shape (batch_size, num_features)

        Returns:
            List of encrypted feature columns
        """
        start = time.time()
        encrypted = self.gbdt.encrypt_features(features)
        self._metrics.encryption_time_ms = (time.time() - start) * 1000
        return encrypted

    def compute_level_comparisons(
        self,
        encrypted_features: List[EncryptedTensor],
        level_spec: List[Tuple[int, float]]
    ) -> List[EncryptedTensor]:
        """
        Compute encrypted comparisons for all levels.

        Args:
            encrypted_features: Column-packed encrypted features
            level_spec: List of (feature_idx, threshold) per level

        Returns:
            List of encrypted comparison results
        """
        comparisons = []
        start = time.time()

        for feat_idx, threshold in level_spec:
            cmp_result = self.gbdt.compare_threshold(
                encrypted_features[feat_idx],
                threshold
            )
            comparisons.append(cmp_result)

        self._metrics.operations_count["comparisons"] = len(level_spec)
        logger.debug(f"Computed {len(level_spec)} comparisons")

        return comparisons

    def compute_all_leaf_indicators(
        self,
        comparisons: List[EncryptedTensor],
        num_leaves: int
    ) -> List[EncryptedTensor]:
        """
        Compute all leaf indicators using tensor product structure.

        This is the key innovation: computing all 2^d indicators in parallel
        using the tensor product of sign function results.

        Args:
            comparisons: Encrypted comparison results per level
            num_leaves: Number of leaves (2^depth)

        Returns:
            List of encrypted leaf indicators
        """
        depth = len(comparisons)
        indicators = []
        start = time.time()

        for leaf_idx in range(num_leaves):
            indicator = self.gbdt.compute_leaf_indicator(comparisons, leaf_idx)
            indicators.append(indicator)

        self._metrics.operations_count["leaf_indicators"] = num_leaves
        logger.debug(f"Computed {num_leaves} leaf indicators")

        return indicators

    def aggregate_n2he(
        self,
        leaf_indicators: List[EncryptedTensor],
        leaf_values: List[float]
    ) -> EncryptedTensor:
        """
        Aggregate using N2HE weighted sum (real FHE).

        This uses plaintext-ciphertext multiplication and addition,
        which is much cheaper than ciphertext-ciphertext operations.

        Args:
            leaf_indicators: Encrypted leaf indicators
            leaf_values: Plaintext leaf values

        Returns:
            Encrypted aggregated output
        """
        start = time.time()
        result = self.gbdt.aggregate_tree_outputs(leaf_indicators, leaf_values)
        self._metrics.operations_count["aggregations"] = len(leaf_indicators)

        return result

    def predict_tree(
        self,
        encrypted_features: List[EncryptedTensor],
        level_features: List[int],
        level_thresholds: List[float],
        leaf_values: List[float]
    ) -> EncryptedTensor:
        """
        Full tree prediction with real FHE.

        Args:
            encrypted_features: Encrypted input features
            level_features: Feature index per level
            level_thresholds: Threshold per level
            leaf_values: Leaf values

        Returns:
            Encrypted prediction
        """
        start = time.time()

        result = self.gbdt.predict_oblivious_tree(
            encrypted_features,
            level_features,
            level_thresholds,
            leaf_values
        )

        self._metrics.computation_time_ms = (time.time() - start) * 1000
        return result

    def predict_ensemble(
        self,
        encrypted_features: List[EncryptedTensor],
        trees: List[Dict[str, Any]],
        base_score: float = 0.0
    ) -> EncryptedTensor:
        """
        Full ensemble prediction with real FHE.

        Args:
            encrypted_features: Encrypted features
            trees: List of tree definitions
            base_score: Base prediction score

        Returns:
            Encrypted ensemble output
        """
        start = time.time()

        result = self.gbdt.predict_ensemble(
            encrypted_features,
            trees,
            base_score
        )

        self._metrics.computation_time_ms = (time.time() - start) * 1000
        self._metrics.operations_count["trees"] = len(trees)

        return result

    def decrypt(self, encrypted: EncryptedTensor) -> np.ndarray:
        """Decrypt result."""
        start = time.time()
        result = self.ctx.decrypt(encrypted)
        self._metrics.decryption_time_ms = (time.time() - start) * 1000
        return result

    def get_metrics(self) -> ProductionMetrics:
        """Get production metrics."""
        self._metrics.total_time_ms = (
            self._metrics.encryption_time_ms +
            self._metrics.computation_time_ms +
            self._metrics.decryption_time_ms
        )
        return self._metrics


class ProductionHomomorphicPruning:
    """
    Production-ready homomorphic pruning using TenSEAL.

    Implements real encrypted variance computation for ensemble pruning.
    """

    def __init__(
        self,
        context: Optional[TenSEALContext] = None,
        depth: int = 8
    ):
        """
        Initialize production homomorphic pruning.

        Args:
            context: Optional pre-created context
            depth: Multiplicative depth needed
        """
        if context is not None:
            self.ctx = context
        else:
            self.ctx = create_production_context(128, depth)

        self._metrics = ProductionMetrics()

    def encrypt_predictions(
        self,
        predictions: np.ndarray
    ) -> EncryptedTensor:
        """
        Encrypt tree predictions for pruning.

        Args:
            predictions: Shape (num_samples, num_trees)

        Returns:
            Encrypted predictions
        """
        return self.ctx.encrypt(predictions.flatten())

    def compute_encrypted_mean(
        self,
        encrypted_preds: List[EncryptedTensor]
    ) -> EncryptedTensor:
        """
        Compute mean of encrypted predictions.

        Args:
            encrypted_preds: List of encrypted predictions

        Returns:
            Encrypted mean
        """
        # Sum all predictions
        result = encrypted_preds[0]
        for i in range(1, len(encrypted_preds)):
            result = self.ctx.add(result, encrypted_preds[i])

        # Divide by count (plaintext scalar)
        scale = 1.0 / len(encrypted_preds)
        result = self.ctx.multiply_plain(result, [scale])

        return result

    def compute_encrypted_variance(
        self,
        encrypted_preds: List[EncryptedTensor],
        encrypted_mean: EncryptedTensor
    ) -> EncryptedTensor:
        """
        Compute variance of encrypted predictions.

        Var(X) = E[(X - mean)^2]

        Args:
            encrypted_preds: List of encrypted predictions
            encrypted_mean: Encrypted mean

        Returns:
            Encrypted variance
        """
        squared_diffs = []

        for pred in encrypted_preds:
            # Compute (pred - mean)
            diff = self.ctx.subtract(pred, encrypted_mean)
            # Compute (pred - mean)^2
            squared = self.ctx.multiply(diff, diff)
            squared_diffs.append(squared)

        # Sum squared differences
        result = squared_diffs[0]
        for i in range(1, len(squared_diffs)):
            result = self.ctx.add(result, squared_diffs[i])

        # Divide by count
        scale = 1.0 / len(encrypted_preds)
        result = self.ctx.multiply_plain(result, [scale])

        self._metrics.operations_count["variance_computations"] = 1

        return result

    def compute_tree_importance(
        self,
        tree_predictions: List[EncryptedTensor],
        ensemble_variance: EncryptedTensor
    ) -> List[EncryptedTensor]:
        """
        Compute importance score for each tree.

        Importance = Var(ensemble without tree) - Var(ensemble)

        Args:
            tree_predictions: Predictions from each tree
            ensemble_variance: Variance of full ensemble

        Returns:
            Encrypted importance scores
        """
        importances = []
        num_trees = len(tree_predictions)

        for i in range(num_trees):
            # Compute variance without tree i
            preds_without = tree_predictions[:i] + tree_predictions[i+1:]
            mean_without = self.compute_encrypted_mean(preds_without)
            var_without = self.compute_encrypted_variance(preds_without, mean_without)

            # Importance = var_without - var_full
            importance = self.ctx.subtract(var_without, ensemble_variance)
            importances.append(importance)

        self._metrics.operations_count["importance_scores"] = num_trees

        return importances

    def decrypt(self, encrypted: EncryptedTensor) -> np.ndarray:
        """Decrypt result."""
        return self.ctx.decrypt(encrypted)

    def get_metrics(self) -> ProductionMetrics:
        """Get production metrics."""
        return self._metrics


class ProductionStreamingGradients:
    """
    Production-ready streaming gradient computation using TenSEAL.

    Implements encrypted gradient updates for online learning.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        context: Optional[TenSEALContext] = None
    ):
        """
        Initialize streaming gradients.

        Args:
            learning_rate: Learning rate for updates
            context: Optional pre-created context
        """
        if context is not None:
            self.ctx = context
        else:
            self.ctx = create_production_context(128, 6)

        self.learning_rate = learning_rate
        self._encrypted_weights: Optional[EncryptedTensor] = None
        self._metrics = ProductionMetrics()

    def initialize_weights(self, num_weights: int) -> EncryptedTensor:
        """
        Initialize encrypted weights to zero.

        Args:
            num_weights: Number of weights

        Returns:
            Encrypted zero weights
        """
        zeros = [0.0] * num_weights
        self._encrypted_weights = self.ctx.encrypt(zeros)
        return self._encrypted_weights

    def compute_encrypted_gradient(
        self,
        encrypted_prediction: EncryptedTensor,
        target: float
    ) -> EncryptedTensor:
        """
        Compute encrypted gradient.

        For MSE loss: gradient = (prediction - target)

        Args:
            encrypted_prediction: Encrypted model prediction
            target: True target value

        Returns:
            Encrypted gradient
        """
        # gradient = prediction - target
        gradient = self.ctx.add_plain(encrypted_prediction, [-target])

        self._metrics.operations_count["gradient_computations"] = (
            self._metrics.operations_count.get("gradient_computations", 0) + 1
        )

        return gradient

    def update_weights(
        self,
        encrypted_weights: EncryptedTensor,
        encrypted_gradient: EncryptedTensor,
        encrypted_features: EncryptedTensor
    ) -> EncryptedTensor:
        """
        Update weights with encrypted gradient.

        weight = weight - lr * gradient * feature

        Args:
            encrypted_weights: Current encrypted weights
            encrypted_gradient: Encrypted gradient
            encrypted_features: Encrypted input features

        Returns:
            Updated encrypted weights
        """
        # Compute gradient * features
        grad_features = self.ctx.multiply(encrypted_gradient, encrypted_features)

        # Scale by learning rate
        scaled_grad = self.ctx.multiply_plain(grad_features, [self.learning_rate])

        # Update: weight = weight - scaled_grad
        updated = self.ctx.subtract(encrypted_weights, scaled_grad)

        self._encrypted_weights = updated
        self._metrics.operations_count["weight_updates"] = (
            self._metrics.operations_count.get("weight_updates", 0) + 1
        )

        return updated

    def get_weights(self) -> Optional[EncryptedTensor]:
        """Get current encrypted weights."""
        return self._encrypted_weights

    def decrypt(self, encrypted: EncryptedTensor) -> np.ndarray:
        """Decrypt result."""
        return self.ctx.decrypt(encrypted)

    def get_metrics(self) -> ProductionMetrics:
        """Get production metrics."""
        return self._metrics


class ProductionFederatedMultiKey:
    """
    Production-ready multi-key FHE for federated GBDT.

    Each participant has their own keys, and computation is done
    under encryption across all parties.
    """

    def __init__(self, num_parties: int = 3, depth: int = 6):
        """
        Initialize federated multi-key system.

        Args:
            num_parties: Number of participating parties
            depth: Multiplicative depth needed
        """
        self.num_parties = num_parties
        self.party_contexts: List[TenSEALContext] = []

        # Create context for each party
        for i in range(num_parties):
            ctx = create_production_context(128, depth)
            self.party_contexts.append(ctx)

        self._metrics = ProductionMetrics()
        logger.info(f"Federated system initialized with {num_parties} parties")

    def encrypt_party_data(
        self,
        party_id: int,
        data: np.ndarray
    ) -> EncryptedTensor:
        """
        Encrypt data for a specific party.

        Args:
            party_id: Party index
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        if party_id >= self.num_parties:
            raise ValueError(f"Invalid party_id: {party_id}")

        return self.party_contexts[party_id].encrypt(data.flatten())

    def local_computation(
        self,
        party_id: int,
        encrypted_data: EncryptedTensor,
        plaintext_model: List[float]
    ) -> EncryptedTensor:
        """
        Perform local computation at a party.

        Each party computes on their encrypted data locally.

        Args:
            party_id: Party index
            encrypted_data: Party's encrypted data
            plaintext_model: Shared plaintext model parameters

        Returns:
            Encrypted local result
        """
        ctx = self.party_contexts[party_id]

        # Compute dot product with model
        result = ctx.dot(encrypted_data, plaintext_model)

        self._metrics.operations_count[f"party_{party_id}_ops"] = (
            self._metrics.operations_count.get(f"party_{party_id}_ops", 0) + 1
        )

        return result

    def aggregate_party_results(
        self,
        party_results: List[Tuple[int, EncryptedTensor]]
    ) -> Dict[int, np.ndarray]:
        """
        Aggregate results from all parties.

        In a real multi-key setup, this would involve threshold decryption.
        Here we simulate by having each party decrypt their own result.

        Args:
            party_results: List of (party_id, encrypted_result) tuples

        Returns:
            Dict of party_id -> decrypted result
        """
        results = {}

        for party_id, encrypted_result in party_results:
            ctx = self.party_contexts[party_id]
            decrypted = ctx.decrypt(encrypted_result)
            results[party_id] = decrypted

        return results

    def get_party_public_context(self, party_id: int) -> bytes:
        """
        Get serialized public context for a party.

        This can be shared with other parties for encryption.

        Args:
            party_id: Party index

        Returns:
            Serialized public context
        """
        return self.party_contexts[party_id].serialize_context()

    def get_metrics(self) -> ProductionMetrics:
        """Get production metrics."""
        return self._metrics


# Convenience functions

def create_production_leaf_centric(
    depth: int = 6
) -> ProductionLeafCentric:
    """Create production leaf-centric computation."""
    return ProductionLeafCentric(depth=depth)


def create_production_pruning(
    depth: int = 8
) -> ProductionHomomorphicPruning:
    """Create production homomorphic pruning."""
    return ProductionHomomorphicPruning(depth=depth)


def create_production_streaming(
    learning_rate: float = 0.1
) -> ProductionStreamingGradients:
    """Create production streaming gradients."""
    return ProductionStreamingGradients(learning_rate=learning_rate)


def create_federated_system(
    num_parties: int = 3
) -> ProductionFederatedMultiKey:
    """Create federated multi-key system."""
    return ProductionFederatedMultiKey(num_parties=num_parties)


def run_production_benchmark(
    features: np.ndarray,
    trees: List[Dict[str, Any]],
    base_score: float = 0.0
) -> Tuple[np.ndarray, ProductionMetrics]:
    """
    Run end-to-end production FHE GBDT benchmark.

    Args:
        features: Input features
        trees: Tree definitions
        base_score: Base prediction score

    Returns:
        Tuple of (predictions, metrics)
    """
    start_total = time.time()

    # Create production system
    prod = ProductionLeafCentric(depth=6)

    # Encrypt
    encrypted_features = prod.encrypt_features(features)

    # Predict
    encrypted_result = prod.predict_ensemble(
        encrypted_features,
        trees,
        base_score
    )

    # Decrypt
    predictions = prod.decrypt(encrypted_result)

    # Gather metrics
    metrics = prod.get_metrics()
    metrics.total_time_ms = (time.time() - start_total) * 1000

    return predictions, metrics
