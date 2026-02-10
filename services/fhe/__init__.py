"""
FHE Backend Module

Production FHE implementations using real cryptographic libraries.

Supported Backends:
- TenSEAL (CKKS/BFV via Microsoft SEAL)
- Concrete-ML (TFHE via Zama)

Usage:
    # TenSEAL for custom FHE operations
    from services.fhe import TenSEALContext, create_production_context
    ctx = create_production_context()
    encrypted = ctx.encrypt([1.0, 2.0, 3.0])

    # Concrete-ML for native GBDT compilation
    from services.fhe import create_concrete_context, compile_xgboost_to_fhe
    ctx, stats = compile_xgboost_to_fhe(X_train, y_train)
"""

from .tenseal_backend import (
    TenSEALContext,
    FHEConfig,
    FHEScheme,
    EncryptedTensor,
    ProductionFHEGBDT,
    create_production_context,
    encrypt_and_predict,
)

from .concrete_backend import (
    ConcreteMLContext,
    ConcreteMLConfig,
    ConcreteMLServer,
    ConcreteMLClient,
    ConcreteMLSimulator,
    FHECircuitStats,
    QuantizationConfig,
    create_concrete_context,
    compile_xgboost_to_fhe,
    CONCRETE_ML_AVAILABLE,
)

from .production_integration import (
    ProductionMetrics,
    ProductionLeafCentric,
    ProductionHomomorphicPruning,
    ProductionStreamingGradients,
    ProductionFederatedMultiKey,
    create_production_leaf_centric,
    create_production_pruning,
    create_production_streaming,
    create_federated_system,
    run_production_benchmark,
)

from .noise_budget import (
    NoiseBudgetTracker,
    NoiseBudgetState,
    NoiseLevel,
    OperationCost,
    AdaptiveNoiseManager,
    create_noise_tracker,
    estimate_gbdt_budget,
)

__all__ = [
    # TenSEAL
    "TenSEALContext",
    "FHEConfig",
    "FHEScheme",
    "EncryptedTensor",
    "ProductionFHEGBDT",
    "create_production_context",
    "encrypt_and_predict",
    # Concrete-ML
    "ConcreteMLContext",
    "ConcreteMLConfig",
    "ConcreteMLServer",
    "ConcreteMLClient",
    "ConcreteMLSimulator",
    "FHECircuitStats",
    "QuantizationConfig",
    "create_concrete_context",
    "compile_xgboost_to_fhe",
    "CONCRETE_ML_AVAILABLE",
    # Production Integration
    "ProductionMetrics",
    "ProductionLeafCentric",
    "ProductionHomomorphicPruning",
    "ProductionStreamingGradients",
    "ProductionFederatedMultiKey",
    "create_production_leaf_centric",
    "create_production_pruning",
    "create_production_streaming",
    "create_federated_system",
    "run_production_benchmark",
    # Noise Budget
    "NoiseBudgetTracker",
    "NoiseBudgetState",
    "NoiseLevel",
    "OperationCost",
    "AdaptiveNoiseManager",
    "create_noise_tracker",
    "estimate_gbdt_budget",
]
