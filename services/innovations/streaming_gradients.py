"""
Novel Innovation #8: Streaming Encrypted Gradients (Online Learning)

Enable online/incremental learning on encrypted data streams. Update GBDT
leaf values incrementally using homomorphically computed gradients.

Key Insight:
- Gradient computation is weighted sum (N2HE's sweet spot)
- Leaf updates can be done homomorphically: leaf_new = leaf_old + lr * gradient
- Enables continuous learning without decrypting training data

Benefits:
- Continuous model improvement on encrypted streams
- No need to decrypt for training updates
- Maintains privacy during incremental learning
- Compatible with federated learning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingSample:
    """A single sample in the encrypted stream."""
    sample_id: str
    features: Any  # Encrypted features
    target: Any  # Encrypted target (optional)
    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0


@dataclass
class StreamingConfig:
    """Configuration for streaming encrypted training."""
    # Learning rate
    learning_rate: float = 0.01

    # Decay factor for learning rate
    lr_decay: float = 0.999

    # Minimum learning rate
    min_lr: float = 0.0001

    # Batch size for gradient accumulation
    batch_size: int = 32

    # Maximum samples to buffer
    buffer_size: int = 1000

    # Update frequency (in samples)
    update_frequency: int = 32

    # Momentum for gradient updates
    momentum: float = 0.9

    # Enable gradient clipping
    gradient_clip: Optional[float] = 1.0


@dataclass
class GradientStats:
    """Statistics about gradient computation."""
    num_updates: int = 0
    total_samples: int = 0
    avg_gradient_norm: float = 0.0
    current_lr: float = 0.0
    last_update_time: float = 0.0


class HomomorphicGradientComputer:
    """
    Computes gradients homomorphically for GBDT updates.

    For regression with MSE loss:
        gradient = y - f(x) = y - Σ leaf_i × indicator_i

    For binary classification with log loss:
        gradient = y - sigmoid(f(x))

    All operations stay in encrypted domain.
    """

    def __init__(self, loss_type: str = "mse"):
        """
        Initialize gradient computer.

        Args:
            loss_type: "mse" for regression, "logloss" for classification
        """
        self.loss_type = loss_type

        # Polynomial coefficients for sigmoid approximation
        # sigmoid(x) ≈ 0.5 + 0.197x - 0.004x³
        self._sigmoid_coeffs = [0.5, 0.197, 0.0, -0.004]

    def compute_gradient_plaintext(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute gradient in plaintext (for validation)."""
        if self.loss_type == "mse":
            return y_true - y_pred
        elif self.loss_type == "logloss":
            sigmoid_pred = 1 / (1 + np.exp(-y_pred))
            return y_true - sigmoid_pred
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def compute_encrypted_gradient(
        self,
        y_true_ct: Any,
        y_pred_ct: Any,
        fhe_context: Any
    ) -> Any:
        """
        Compute gradient on encrypted values.

        Args:
            y_true_ct: Encrypted true target
            y_pred_ct: Encrypted prediction
            fhe_context: FHE context

        Returns:
            Encrypted gradient
        """
        if self.loss_type == "mse":
            # gradient = y - f(x)
            return fhe_context.subtract(y_true_ct, y_pred_ct)
        elif self.loss_type == "logloss":
            # gradient = y - sigmoid(f(x))
            sigmoid_ct = self._compute_encrypted_sigmoid(y_pred_ct, fhe_context)
            return fhe_context.subtract(y_true_ct, sigmoid_ct)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _compute_encrypted_sigmoid(self, x_ct: Any, fhe_context: Any) -> Any:
        """Compute sigmoid using polynomial approximation."""
        # sigmoid(x) ≈ 0.5 + 0.197x - 0.004x³
        result = fhe_context.create_constant(self._sigmoid_coeffs[0])

        # Linear term
        linear = fhe_context.multiply_plain(x_ct, self._sigmoid_coeffs[1])
        result = fhe_context.add(result, linear)

        # Cubic term
        x_squared = fhe_context.multiply(x_ct, x_ct)
        x_cubed = fhe_context.multiply(x_squared, x_ct)
        cubic = fhe_context.multiply_plain(x_cubed, self._sigmoid_coeffs[3])
        result = fhe_context.add(result, cubic)

        return result

    def compute_leaf_gradients_plaintext(
        self,
        gradients: np.ndarray,
        leaf_indicators: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-leaf gradients in plaintext.

        Args:
            gradients: Shape (batch_size,) sample gradients
            leaf_indicators: Shape (batch_size, num_leaves) indicators

        Returns:
            Shape (num_leaves,) per-leaf gradients
        """
        # Sum of gradients weighted by leaf membership
        leaf_gradient_sums = leaf_indicators.T @ gradients
        leaf_counts = leaf_indicators.sum(axis=0)

        # Average gradient per leaf
        leaf_gradients = np.divide(
            leaf_gradient_sums, leaf_counts,
            out=np.zeros_like(leaf_gradient_sums),
            where=leaf_counts > 0
        )

        return leaf_gradients

    def compute_encrypted_leaf_gradients(
        self,
        gradient_ct: Any,
        leaf_indicator_cts: List[Any],
        fhe_context: Any
    ) -> List[Any]:
        """
        Compute per-leaf gradients homomorphically.

        Args:
            gradient_ct: Encrypted gradient
            leaf_indicator_cts: Encrypted leaf indicators
            fhe_context: FHE context

        Returns:
            List of encrypted leaf gradients
        """
        leaf_gradients = []

        for indicator_ct in leaf_indicator_cts:
            # gradient × indicator
            leaf_grad = fhe_context.multiply(gradient_ct, indicator_ct)
            leaf_gradients.append(leaf_grad)

        return leaf_gradients


class OnlineLeafUpdater:
    """
    Updates leaf values incrementally using encrypted gradients.

    Update rule: leaf_new = leaf_old + lr × gradient
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize updater.

        Args:
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()

        # Momentum buffers
        self._momentum_buffers: Dict[Tuple[int, int], np.ndarray] = {}

        # Statistics
        self.stats = GradientStats(current_lr=self.config.learning_rate)

    def update_leaves_plaintext(
        self,
        leaf_values: np.ndarray,
        leaf_gradients: np.ndarray,
        tree_idx: int
    ) -> np.ndarray:
        """
        Update leaf values in plaintext.

        Args:
            leaf_values: Current leaf values
            leaf_gradients: Computed gradients
            tree_idx: Tree index (for momentum tracking)

        Returns:
            Updated leaf values
        """
        # Get or initialize momentum buffer
        momentum_key = (tree_idx, len(leaf_values))
        if momentum_key not in self._momentum_buffers:
            self._momentum_buffers[momentum_key] = np.zeros_like(leaf_values)

        momentum_buffer = self._momentum_buffers[momentum_key]

        # Apply gradient clipping
        if self.config.gradient_clip is not None:
            grad_norm = np.linalg.norm(leaf_gradients)
            if grad_norm > self.config.gradient_clip:
                leaf_gradients = leaf_gradients * self.config.gradient_clip / grad_norm

        # Update momentum
        momentum_buffer = (
            self.config.momentum * momentum_buffer +
            (1 - self.config.momentum) * leaf_gradients
        )
        self._momentum_buffers[momentum_key] = momentum_buffer

        # Apply update
        lr = self.stats.current_lr
        updated = leaf_values + lr * momentum_buffer

        # Decay learning rate
        self.stats.current_lr = max(
            self.config.min_lr,
            self.stats.current_lr * self.config.lr_decay
        )

        # Update statistics
        self.stats.num_updates += 1
        self.stats.avg_gradient_norm = (
            0.9 * self.stats.avg_gradient_norm + 0.1 * np.linalg.norm(leaf_gradients)
        )
        self.stats.last_update_time = time.time()

        return updated

    def update_encrypted_leaves(
        self,
        leaf_value_cts: List[Any],
        leaf_gradient_cts: List[Any],
        fhe_context: Any
    ) -> List[Any]:
        """
        Update leaf values homomorphically.

        Args:
            leaf_value_cts: Current encrypted leaf values
            leaf_gradient_cts: Encrypted gradients
            fhe_context: FHE context

        Returns:
            Updated encrypted leaf values
        """
        lr = self.stats.current_lr
        updated = []

        for value_ct, grad_ct in zip(leaf_value_cts, leaf_gradient_cts):
            # scaled_grad = lr × gradient
            scaled_grad = fhe_context.multiply_plain(grad_ct, lr)

            # updated = value + scaled_grad
            new_value = fhe_context.add(value_ct, scaled_grad)
            updated.append(new_value)

        # Decay learning rate
        self.stats.current_lr = max(
            self.config.min_lr,
            self.stats.current_lr * self.config.lr_decay
        )

        self.stats.num_updates += 1
        self.stats.last_update_time = time.time()

        return updated


class EncryptedStreamingGBDT:
    """
    Complete streaming GBDT system with encrypted gradient updates.

    Workflow:
    1. Receive encrypted sample stream
    2. Compute encrypted predictions
    3. Compute encrypted gradients
    4. Update leaf values homomorphically
    5. Repeat continuously
    """

    def __init__(
        self,
        model_ir: Any,
        config: Optional[StreamingConfig] = None
    ):
        """
        Initialize streaming GBDT.

        Args:
            model_ir: Initial GBDT model
            config: Streaming configuration
        """
        self.model_ir = model_ir
        self.config = config or StreamingConfig()

        self.gradient_computer = HomomorphicGradientComputer()
        self.leaf_updater = OnlineLeafUpdater(config)

        # Sample buffer
        self._sample_buffer: deque = deque(maxlen=self.config.buffer_size)

        # Current leaf values (mutable copy)
        self._leaf_values = self._extract_leaf_values()

        # Callbacks
        self._update_callbacks: List[Callable] = []

    def _extract_leaf_values(self) -> Dict[int, np.ndarray]:
        """Extract leaf values from model."""
        leaf_values = {}

        for tree_idx, tree in enumerate(self.model_ir.trees):
            leaves = [
                node for node in tree.nodes.values()
                if node.leaf_value is not None
            ]
            values = np.array([leaf.leaf_value for leaf in leaves])
            leaf_values[tree_idx] = values

        return leaf_values

    def process_sample(
        self,
        features: np.ndarray,
        target: float
    ):
        """
        Process a single sample in the stream (plaintext version).

        Args:
            features: Feature values
            target: Target value
        """
        sample = StreamingSample(
            sample_id=f"sample_{len(self._sample_buffer)}",
            features=features,
            target=target
        )
        self._sample_buffer.append(sample)

        # Check if we should update
        if len(self._sample_buffer) >= self.config.update_frequency:
            self._perform_batch_update()

    def _perform_batch_update(self):
        """Perform batch update on buffered samples."""
        if not self._sample_buffer:
            return

        # Convert buffer to arrays
        batch_size = min(self.config.batch_size, len(self._sample_buffer))
        samples = [self._sample_buffer.popleft() for _ in range(batch_size)]

        features = np.array([s.features for s in samples])
        targets = np.array([s.target for s in samples])

        # Compute predictions
        predictions = self._predict(features)

        # Compute gradients
        gradients = self.gradient_computer.compute_gradient_plaintext(
            targets, predictions
        )

        # Update each tree's leaves
        for tree_idx, tree in enumerate(self.model_ir.trees):
            # Get leaf indicators for this tree
            leaf_indicators = self._get_leaf_indicators(tree, features)

            # Compute leaf gradients
            leaf_gradients = self.gradient_computer.compute_leaf_gradients_plaintext(
                gradients, leaf_indicators
            )

            # Update leaves
            self._leaf_values[tree_idx] = self.leaf_updater.update_leaves_plaintext(
                self._leaf_values[tree_idx],
                leaf_gradients,
                tree_idx
            )

        # Notify callbacks
        for callback in self._update_callbacks:
            callback(self.leaf_updater.stats)

    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using current leaf values."""
        predictions = np.full(features.shape[0], self.model_ir.base_score)

        for tree_idx, tree in enumerate(self.model_ir.trees):
            tree_output = self._predict_tree(tree, tree_idx, features)
            predictions += tree_output

        return predictions

    def _predict_tree(
        self,
        tree_ir: Any,
        tree_idx: int,
        features: np.ndarray
    ) -> np.ndarray:
        """Predict with single tree using current leaf values."""
        outputs = np.zeros(features.shape[0])
        leaf_values = self._leaf_values[tree_idx]

        leaves = [
            node for node in tree_ir.nodes.values()
            if node.leaf_value is not None
        ]

        for i in range(features.shape[0]):
            leaf_idx = self._get_leaf_index(tree_ir, features[i])
            if leaf_idx < len(leaf_values):
                outputs[i] = leaf_values[leaf_idx]

        return outputs

    def _get_leaf_index(self, tree_ir: Any, sample: np.ndarray) -> int:
        """Get leaf index for sample."""
        node = tree_ir.nodes.get(tree_ir.root_id)
        leaf_idx = 0

        leaves = [
            node for node in tree_ir.nodes.values()
            if node.leaf_value is not None
        ]
        leaf_ids = [leaf.node_id for leaf in leaves]

        while node is not None:
            if node.leaf_value is not None:
                return leaf_ids.index(node.node_id) if node.node_id in leaf_ids else 0

            if sample[node.feature_index] < node.threshold:
                node = tree_ir.nodes.get(node.left_child_id)
            else:
                node = tree_ir.nodes.get(node.right_child_id)

        return 0

    def _get_leaf_indicators(
        self,
        tree_ir: Any,
        features: np.ndarray
    ) -> np.ndarray:
        """Get leaf indicator matrix."""
        leaves = [
            node for node in tree_ir.nodes.values()
            if node.leaf_value is not None
        ]
        num_leaves = len(leaves)
        indicators = np.zeros((features.shape[0], num_leaves))

        for i in range(features.shape[0]):
            leaf_idx = self._get_leaf_index(tree_ir, features[i])
            if leaf_idx < num_leaves:
                indicators[i, leaf_idx] = 1.0

        return indicators

    async def process_encrypted_stream(
        self,
        stream: Any,  # Async iterator of encrypted samples
        fhe_context: Any
    ):
        """
        Process encrypted sample stream.

        Args:
            stream: Async iterator yielding StreamingSample
            fhe_context: FHE context
        """
        batch = []

        async for sample in stream:
            batch.append(sample)

            if len(batch) >= self.config.batch_size:
                await self._process_encrypted_batch(batch, fhe_context)
                batch = []

        # Process remaining
        if batch:
            await self._process_encrypted_batch(batch, fhe_context)

    async def _process_encrypted_batch(
        self,
        batch: List[StreamingSample],
        fhe_context: Any
    ):
        """Process batch of encrypted samples."""
        # This would implement the full encrypted gradient update
        # For now, we increment stats
        self.leaf_updater.stats.total_samples += len(batch)

    def register_update_callback(self, callback: Callable):
        """Register callback for updates."""
        self._update_callbacks.append(callback)

    def get_current_model(self) -> Dict[str, Any]:
        """Get current model state."""
        return {
            "base_score": self.model_ir.base_score,
            "leaf_values": {k: v.tolist() for k, v in self._leaf_values.items()},
            "stats": {
                "num_updates": self.leaf_updater.stats.num_updates,
                "total_samples": self.leaf_updater.stats.total_samples,
                "current_lr": self.leaf_updater.stats.current_lr,
                "avg_gradient_norm": self.leaf_updater.stats.avg_gradient_norm,
            }
        }


# Convenience functions

def create_streaming_gbdt(
    model_ir: Any,
    learning_rate: float = 0.01
) -> EncryptedStreamingGBDT:
    """
    Create a streaming GBDT from an existing model.

    Args:
        model_ir: Base GBDT model
        learning_rate: Initial learning rate

    Returns:
        EncryptedStreamingGBDT
    """
    config = StreamingConfig(learning_rate=learning_rate)
    return EncryptedStreamingGBDT(model_ir, config)


def update_model_on_stream(
    streaming_gbdt: EncryptedStreamingGBDT,
    X_stream: np.ndarray,
    y_stream: np.ndarray
):
    """
    Update model on a stream of samples.

    Args:
        streaming_gbdt: Streaming GBDT instance
        X_stream: Feature stream
        y_stream: Target stream
    """
    for i in range(len(X_stream)):
        streaming_gbdt.process_sample(X_stream[i], y_stream[i])
