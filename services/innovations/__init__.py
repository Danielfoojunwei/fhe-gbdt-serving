"""
FHE-GBDT Novel Innovations Module

This module implements 10 novel architectural innovations that leverage GBDT first
principles with FHE MOAI and N2HE to unlock new capabilities in privacy-preserving
machine learning.

Novel Directions:
1. Leaf-Centric Encoding - Direct leaf indicator computation vs path traversal
2. Gradient-Informed Noise Allocation - GBDT gradients guide FHE precision
3. Homomorphic Ensemble Pruning - Prune trees in encrypted domain
4. N2HE Multi-Key Federated GBDT - Split features across parties
5. Bootstrapping-Aligned Trees - Design trees for optimal noise budget
6. Polynomial Leaf Functions - Expressive leaf polynomials within FHE
7. MOAI-Native Tree Structure - Rotation-optimal tree conversion
8. Streaming Encrypted Gradients - Online learning in encrypted domain
9. Unified Architecture - Integrated novel FHE-GBDT system
10. C++ Kernel Extensions - High-performance novel primitives

Based on:
- MOAI: Module-Optimizing Architecture for Non-Interactive Secure Inference (NDSS 2025)
- N2HE: Optimized FHE for Neural Networks (IEEE TDSC)
"""

from .leaf_centric import (
    LeafCentricEncoder,
    LeafIndicatorComputer,
    DirectLeafPlan,
)

from .gradient_noise import (
    GradientAwareNoiseAllocator,
    FeatureImportanceAnalyzer,
    AdaptivePrecisionEncoder,
)

from .homomorphic_pruning import (
    HomomorphicEnsemblePruner,
    EncryptedTreeSignificance,
    AdaptivePruningGate,
)

from .federated_multikey import (
    FederatedGBDTProtocol,
    MultiKeyParty,
    PartialTraversalResult,
    N2HEMultiKeyCombiner,
)

from .bootstrap_aligned import (
    BootstrapAwareTreeBuilder,
    BootstrapInterleavedEnsemble,
    NoiseAlignedForest,
)

from .polynomial_leaves import (
    PolynomialLeafGBDT,
    PolynomialLeafTrainer,
    FHEPolynomialEvaluator,
)

from .moai_native import (
    MOAINativeTreeBuilder,
    RotationOptimalConverter,
    ObliviousTreeSynthesizer,
)

from .streaming_gradients import (
    EncryptedStreamingGBDT,
    HomomorphicGradientComputer,
    OnlineLeafUpdater,
)

from .unified_architecture import (
    NovelFHEGBDTEngine,
    InnovationConfig,
    UnifiedExecutionPlan,
)

__all__ = [
    # Leaf-Centric
    "LeafCentricEncoder",
    "LeafIndicatorComputer",
    "DirectLeafPlan",
    # Gradient-Noise
    "GradientAwareNoiseAllocator",
    "FeatureImportanceAnalyzer",
    "AdaptivePrecisionEncoder",
    # Homomorphic Pruning
    "HomomorphicEnsemblePruner",
    "EncryptedTreeSignificance",
    "AdaptivePruningGate",
    # Federated Multi-Key
    "FederatedGBDTProtocol",
    "MultiKeyParty",
    "PartialTraversalResult",
    "N2HEMultiKeyCombiner",
    # Bootstrap-Aligned
    "BootstrapAwareTreeBuilder",
    "BootstrapInterleavedEnsemble",
    "NoiseAlignedForest",
    # Polynomial Leaves
    "PolynomialLeafGBDT",
    "PolynomialLeafTrainer",
    "FHEPolynomialEvaluator",
    # MOAI-Native
    "MOAINativeTreeBuilder",
    "RotationOptimalConverter",
    "ObliviousTreeSynthesizer",
    # Streaming Gradients
    "EncryptedStreamingGBDT",
    "HomomorphicGradientComputer",
    "OnlineLeafUpdater",
    # Unified Architecture
    "NovelFHEGBDTEngine",
    "InnovationConfig",
    "UnifiedExecutionPlan",
]

__version__ = "1.0.0"
