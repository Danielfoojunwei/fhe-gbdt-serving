"""
FHE-GBDT Training Service

Privacy-preserving GBDT training with differential privacy support.
Aligned with TenSafe's training infrastructure.

This module provides:
- DP-GBDT training with privacy accounting
- Federated training support
- Secure aggregation
- Training checkpoints with encryption
"""

from .trainer import (
    DPGBDTTrainer,
    TrainingConfig,
    DPConfig,
    TrainingMetrics,
    TrainingCheckpoint,
)

from .privacy import (
    PrivacyAccountant,
    RDPAccountant,
    PrivacySpent,
    compute_dp_sgd_privacy,
)

from .federated import (
    FederatedTrainer,
    SecureAggregator,
    FederatedConfig,
)

__all__ = [
    # Trainer
    "DPGBDTTrainer",
    "TrainingConfig",
    "DPConfig",
    "TrainingMetrics",
    "TrainingCheckpoint",
    # Privacy
    "PrivacyAccountant",
    "RDPAccountant",
    "PrivacySpent",
    "compute_dp_sgd_privacy",
    # Federated
    "FederatedTrainer",
    "SecureAggregator",
    "FederatedConfig",
]
