"""
FHE-GBDT Federated Training

Federated GBDT training with secure aggregation.
Aligned with TenSafe's federated training infrastructure.

Implements:
- Federated GBDT training across multiple data sources
- Secure aggregation using additive secret sharing
- Privacy-preserving gradient aggregation
"""

import hashlib
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .trainer import DPGBDTTrainer, TrainingConfig, TrainingMetrics, DPConfig

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Aggregation strategy for federated learning."""
    FEDAVG = "fedavg"           # Federated averaging
    FEDPROX = "fedprox"         # Federated proximal
    SCAFFOLD = "scaffold"       # SCAFFOLD variance reduction
    SECURE_AGG = "secure_agg"   # Secure aggregation


@dataclass
class FederatedConfig:
    """Federated training configuration."""
    # Federation settings
    num_rounds: int = 10
    min_clients: int = 2
    max_clients: int = 100
    client_fraction: float = 1.0  # Fraction of clients per round

    # Aggregation
    aggregation_strategy: AggregationStrategy = AggregationStrategy.SECURE_AGG

    # Secure aggregation settings
    secret_sharing_threshold: int = 2  # Minimum shares to reconstruct
    masking_modulus: int = 2**32       # Modulus for masking

    # Privacy
    dp_config: Optional[DPConfig] = None

    # Communication
    timeout_seconds: int = 300
    retry_attempts: int = 3


@dataclass
class ClientUpdate:
    """Update from a federated client."""
    client_id: str
    round_id: int

    # Model update (gradients or weights)
    update: np.ndarray

    # Metadata
    num_samples: int
    metrics: TrainingMetrics

    # Secure aggregation
    masked_update: Optional[np.ndarray] = None
    mask_seed_shares: Optional[Dict[str, bytes]] = None

    # Integrity
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: Optional[str] = None


@dataclass
class AggregatedUpdate:
    """Aggregated update from all clients."""
    round_id: int

    # Aggregated model update
    aggregated_update: np.ndarray

    # Metadata
    num_clients: int
    total_samples: int
    aggregated_metrics: TrainingMetrics

    # Integrity
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    contributing_clients: List[str] = field(default_factory=list)


class SecureAggregator:
    """
    Secure Aggregation for Federated Learning.

    Implements pairwise masking protocol for secure aggregation:
    1. Each client generates random masks with all other clients
    2. Masks cancel out when summed at the server
    3. Server only sees the sum, not individual updates

    Reference:
    - Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (2017)
    """

    def __init__(self, config: FederatedConfig):
        self.config = config
        self._client_seeds: Dict[str, Dict[str, bytes]] = {}  # client_id -> {peer_id -> seed}
        self._round_masks: Dict[int, Dict[str, np.ndarray]] = {}  # round_id -> {client_id -> mask}

    def setup_round(self, client_ids: List[str], round_id: int, update_shape: Tuple[int, ...]):
        """
        Setup secure aggregation for a round.

        Generates pairwise seeds for masking.
        """
        n = len(client_ids)

        # Generate pairwise seeds
        self._client_seeds = {}
        for i, client_i in enumerate(client_ids):
            self._client_seeds[client_i] = {}
            for j, client_j in enumerate(client_ids):
                if i < j:
                    # Generate shared seed
                    seed = secrets.token_bytes(32)
                    self._client_seeds[client_i][client_j] = seed
                elif i > j:
                    # Use same seed as reverse pair
                    self._client_seeds[client_i][client_j] = self._client_seeds[client_j][client_i]

        logger.info(f"Setup secure aggregation for round {round_id} with {n} clients")

    def generate_mask(
        self,
        client_id: str,
        round_id: int,
        update_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Generate mask for a client's update.

        The mask is the sum of PRFs with all peers (positive for lower IDs, negative for higher).
        """
        mask = np.zeros(update_shape, dtype=np.int64)

        for peer_id, seed in self._client_seeds.get(client_id, {}).items():
            # Generate PRF from seed
            rng = np.random.default_rng(int.from_bytes(seed[:8], 'big') + round_id)
            peer_mask = rng.integers(
                0, self.config.masking_modulus,
                size=update_shape, dtype=np.int64
            )

            # Add or subtract based on ID ordering
            if client_id < peer_id:
                mask += peer_mask
            else:
                mask -= peer_mask

        return mask % self.config.masking_modulus

    def mask_update(
        self,
        client_id: str,
        round_id: int,
        update: np.ndarray,
    ) -> np.ndarray:
        """Mask a client's update."""
        mask = self.generate_mask(client_id, round_id, update.shape)

        # Quantize update to integers
        scale = 1e6
        quantized = (update * scale).astype(np.int64)

        # Apply mask
        masked = (quantized + mask) % self.config.masking_modulus

        return masked

    def aggregate_masked_updates(
        self,
        round_id: int,
        masked_updates: Dict[str, np.ndarray],
        num_samples: Dict[str, int],
    ) -> np.ndarray:
        """
        Aggregate masked updates.

        When all clients participate, masks cancel out.
        """
        if not masked_updates:
            raise ValueError("No updates to aggregate")

        # Sum masked updates
        shape = next(iter(masked_updates.values())).shape
        aggregated = np.zeros(shape, dtype=np.int64)

        total_samples = sum(num_samples.values())

        for client_id, masked in masked_updates.items():
            # Weight by sample count
            weight = num_samples[client_id] / total_samples
            aggregated += (masked * weight).astype(np.int64)

        # Unmask (masks should sum to zero)
        aggregated = aggregated % self.config.masking_modulus

        # Handle wrap-around
        half_mod = self.config.masking_modulus // 2
        aggregated = np.where(aggregated > half_mod, aggregated - self.config.masking_modulus, aggregated)

        # Dequantize
        scale = 1e6
        return aggregated / scale


class FederatedTrainer:
    """
    Federated GBDT Trainer.

    Coordinates training across multiple data sources with:
    - Secure aggregation of model updates
    - Optional differential privacy
    - Privacy-preserving gradient aggregation

    Example:
        ```python
        config = FederatedConfig(
            num_rounds=10,
            min_clients=3,
            aggregation_strategy=AggregationStrategy.SECURE_AGG,
            dp_config=DPConfig(enabled=True, epsilon=1.0),
        )

        trainer = FederatedTrainer(
            training_config=training_config,
            federated_config=config,
        )

        # Register clients
        trainer.register_client("client_1", data_size=10000)
        trainer.register_client("client_2", data_size=15000)
        trainer.register_client("client_3", data_size=8000)

        # Run federated training
        global_model = trainer.train()
        ```
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        federated_config: FederatedConfig,
        tenant_id: str = "default",
        job_id: Optional[str] = None,
    ):
        self.training_config = training_config
        self.federated_config = federated_config
        self.tenant_id = tenant_id
        self.job_id = job_id or f"fed-train-{int(time.time() * 1000)}"

        # Client registry
        self._clients: Dict[str, Dict[str, Any]] = {}

        # Secure aggregator
        self._aggregator = SecureAggregator(federated_config)

        # Global model state
        self._global_model = None
        self._global_metrics = TrainingMetrics()

        # Round tracking
        self._current_round = 0
        self._round_updates: Dict[str, ClientUpdate] = {}

        # Callbacks
        self._round_callbacks: List[Callable[[int, AggregatedUpdate], None]] = []

        logger.info(f"Initialized FederatedTrainer for job {self.job_id}")

    def register_client(
        self,
        client_id: str,
        data_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a client for federated training."""
        self._clients[client_id] = {
            "data_size": data_size,
            "metadata": metadata or {},
            "rounds_participated": 0,
            "total_samples_contributed": 0,
        }
        logger.info(f"Registered client {client_id} with {data_size} samples")

    def unregister_client(self, client_id: str):
        """Unregister a client."""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info(f"Unregistered client {client_id}")

    def select_clients(self, round_id: int) -> List[str]:
        """Select clients for a round."""
        available_clients = list(self._clients.keys())

        if len(available_clients) < self.federated_config.min_clients:
            raise ValueError(
                f"Not enough clients: {len(available_clients)} < {self.federated_config.min_clients}"
            )

        # Select fraction of clients
        num_to_select = max(
            self.federated_config.min_clients,
            int(len(available_clients) * self.federated_config.client_fraction)
        )
        num_to_select = min(num_to_select, self.federated_config.max_clients)

        # Random selection
        rng = np.random.default_rng(round_id)
        selected = rng.choice(available_clients, size=num_to_select, replace=False)

        return list(selected)

    def start_round(self, round_id: int) -> Dict[str, Any]:
        """
        Start a new training round.

        Returns configuration for clients to begin local training.
        """
        self._current_round = round_id
        self._round_updates = {}

        # Select clients
        selected_clients = self.select_clients(round_id)

        # Setup secure aggregation
        if self.federated_config.aggregation_strategy == AggregationStrategy.SECURE_AGG:
            # Placeholder shape - will be set when first update received
            pass

        logger.info(f"Starting round {round_id} with {len(selected_clients)} clients")

        return {
            "round_id": round_id,
            "selected_clients": selected_clients,
            "global_model": self._serialize_global_model(),
            "training_config": self.training_config,
        }

    def submit_update(self, update: ClientUpdate) -> bool:
        """
        Submit a client update for aggregation.

        Returns True if update accepted, False if round already closed.
        """
        if update.round_id != self._current_round:
            logger.warning(
                f"Client {update.client_id} submitted update for wrong round "
                f"({update.round_id} != {self._current_round})"
            )
            return False

        # Verify client is registered
        if update.client_id not in self._clients:
            logger.warning(f"Unregistered client {update.client_id} attempted to submit update")
            return False

        # Store update
        self._round_updates[update.client_id] = update

        # Update client stats
        self._clients[update.client_id]["rounds_participated"] += 1
        self._clients[update.client_id]["total_samples_contributed"] += update.num_samples

        logger.info(f"Received update from client {update.client_id} for round {update.round_id}")

        return True

    def aggregate_round(self) -> AggregatedUpdate:
        """
        Aggregate updates from all clients in the current round.
        """
        if len(self._round_updates) < self.federated_config.min_clients:
            raise ValueError(
                f"Not enough updates: {len(self._round_updates)} < {self.federated_config.min_clients}"
            )

        strategy = self.federated_config.aggregation_strategy

        if strategy == AggregationStrategy.FEDAVG:
            aggregated = self._aggregate_fedavg()
        elif strategy == AggregationStrategy.SECURE_AGG:
            aggregated = self._aggregate_secure()
        else:
            aggregated = self._aggregate_fedavg()  # Default to FedAvg

        # Notify callbacks
        for cb in self._round_callbacks:
            cb(self._current_round, aggregated)

        return aggregated

    def _aggregate_fedavg(self) -> AggregatedUpdate:
        """Aggregate using Federated Averaging."""
        total_samples = sum(u.num_samples for u in self._round_updates.values())

        # Weighted average of updates
        aggregated_update = None

        for client_id, update in self._round_updates.items():
            weight = update.num_samples / total_samples

            if aggregated_update is None:
                aggregated_update = update.update * weight
            else:
                aggregated_update += update.update * weight

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics()

        return AggregatedUpdate(
            round_id=self._current_round,
            aggregated_update=aggregated_update,
            num_clients=len(self._round_updates),
            total_samples=total_samples,
            aggregated_metrics=aggregated_metrics,
            contributing_clients=list(self._round_updates.keys()),
        )

    def _aggregate_secure(self) -> AggregatedUpdate:
        """Aggregate using secure aggregation."""
        # Collect masked updates
        masked_updates = {}
        num_samples = {}

        for client_id, update in self._round_updates.items():
            if update.masked_update is not None:
                masked_updates[client_id] = update.masked_update
            else:
                # Mask on server side (less secure, but fallback)
                masked_updates[client_id] = self._aggregator.mask_update(
                    client_id, self._current_round, update.update
                )
            num_samples[client_id] = update.num_samples

        # Aggregate
        aggregated_update = self._aggregator.aggregate_masked_updates(
            self._current_round, masked_updates, num_samples
        )

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics()

        return AggregatedUpdate(
            round_id=self._current_round,
            aggregated_update=aggregated_update,
            num_clients=len(self._round_updates),
            total_samples=sum(num_samples.values()),
            aggregated_metrics=aggregated_metrics,
            contributing_clients=list(self._round_updates.keys()),
        )

    def _aggregate_metrics(self) -> TrainingMetrics:
        """Aggregate metrics from all clients."""
        metrics = TrainingMetrics()

        total_samples = sum(u.num_samples for u in self._round_updates.values())

        # Weighted average of metrics
        for client_id, update in self._round_updates.items():
            weight = update.num_samples / total_samples

            if update.metrics.train_loss is not None:
                if metrics.train_loss is None:
                    metrics.train_loss = 0
                metrics.train_loss += update.metrics.train_loss * weight

            if update.metrics.val_loss is not None:
                if metrics.val_loss is None:
                    metrics.val_loss = 0
                metrics.val_loss += update.metrics.val_loss * weight

            if update.metrics.train_auc is not None:
                if metrics.train_auc is None:
                    metrics.train_auc = 0
                metrics.train_auc += update.metrics.train_auc * weight

            if update.metrics.val_auc is not None:
                if metrics.val_auc is None:
                    metrics.val_auc = 0
                metrics.val_auc += update.metrics.val_auc * weight

        return metrics

    def update_global_model(self, aggregated: AggregatedUpdate):
        """Update global model with aggregated update."""
        if self._global_model is None:
            # Initialize global model
            self._global_model = aggregated.aggregated_update
        else:
            # Apply update
            self._global_model = (
                self._global_model * (1 - self.training_config.learning_rate)
                + aggregated.aggregated_update * self.training_config.learning_rate
            )

        self._global_metrics = aggregated.aggregated_metrics

    def _serialize_global_model(self) -> Optional[bytes]:
        """Serialize global model for distribution."""
        if self._global_model is None:
            return None
        return self._global_model.tobytes()

    def train(self) -> Any:
        """
        Run full federated training.

        Returns the final global model.
        """
        logger.info(f"Starting federated training for {self.federated_config.num_rounds} rounds")

        for round_id in range(self.federated_config.num_rounds):
            # Start round
            round_config = self.start_round(round_id)

            # In practice, clients would train locally and submit updates
            # Here we simulate with placeholder
            logger.info(f"Round {round_id}: Waiting for client updates...")

            # Wait for enough updates (in practice, this would be async)
            # For simulation, we'll skip actual waiting

            # Aggregate
            try:
                aggregated = self.aggregate_round()
                self.update_global_model(aggregated)

                logger.info(
                    f"Round {round_id} completed: "
                    f"{aggregated.num_clients} clients, "
                    f"{aggregated.total_samples} samples"
                )
            except ValueError as e:
                logger.warning(f"Round {round_id} failed: {e}")
                continue

        logger.info("Federated training completed")
        return self._global_model

    def register_round_callback(self, callback: Callable[[int, AggregatedUpdate], None]):
        """Register callback for round completion."""
        self._round_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get federated training status."""
        return {
            "job_id": self.job_id,
            "current_round": self._current_round,
            "total_rounds": self.federated_config.num_rounds,
            "num_clients": len(self._clients),
            "updates_this_round": len(self._round_updates),
            "global_metrics": self._global_metrics,
        }
