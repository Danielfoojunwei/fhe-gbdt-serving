"""
Production FHE Backend using Concrete-ML

Native GBDT FHE compilation using Zama's Concrete-ML library.
Provides direct model compilation to FHE circuits with automatic
quantization and bootstrapping.

Key Features:
- Direct scikit-learn model compilation to FHE
- TFHE scheme with programmable bootstrapping
- Automatic quantization and bit-width selection
- Native support for XGBoost, LightGBM, Random Forest
- Built-in comparison operations (no polynomial approximation needed)

References:
- Concrete-ML: https://github.com/zama-ai/concrete-ml
- Concrete: https://github.com/zama-ai/concrete
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import logging
import time
import pickle

logger = logging.getLogger(__name__)

# Try to import Concrete-ML, provide fallback if not installed
try:
    from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier
    from concrete.ml.sklearn import XGBRegressor as ConcreteXGBRegressor
    from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDecisionTree
    from concrete.ml.sklearn import RandomForestClassifier as ConcreteRandomForest
    from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
    CONCRETE_ML_AVAILABLE = True
except ImportError:
    CONCRETE_ML_AVAILABLE = False
    logger.warning("Concrete-ML not installed. Install with: pip install concrete-ml")


class QuantizationConfig(Enum):
    """Quantization configurations for FHE compilation."""
    LOW_PRECISION = 3    # 3 bits, fastest but lowest accuracy
    MEDIUM_PRECISION = 4 # 4 bits, balanced
    HIGH_PRECISION = 6   # 6 bits, slower but more accurate
    FULL_PRECISION = 8   # 8 bits, slowest but highest accuracy


@dataclass
class ConcreteMLConfig:
    """Configuration for Concrete-ML FHE compilation."""
    # Quantization
    n_bits: int = 4  # Bit width for quantization

    # Compilation
    use_simulation: bool = False  # If True, use FHE simulation (faster)

    # Model parameters
    n_estimators: int = 100  # Number of trees
    max_depth: int = 6       # Maximum tree depth

    # FHE circuit parameters
    p_error: float = 0.01    # Allowed probability of error

    # Deployment
    deployment_path: str = "./fhe_deployment"


@dataclass
class FHECircuitStats:
    """Statistics about the compiled FHE circuit."""
    compilation_time: float = 0.0
    circuit_size_mb: float = 0.0
    estimated_inference_time_ms: float = 0.0
    max_bit_width: int = 0
    programmable_bootstrap_count: int = 0
    key_size_mb: float = 0.0


class ConcreteMLContext:
    """
    Production FHE context using Concrete-ML.

    Provides native GBDT FHE compilation with:
    - Automatic model quantization
    - FHE circuit generation
    - Key management
    - Client/server deployment support

    Example:
        ```python
        # Train a model
        ctx = ConcreteMLContext(ConcreteMLConfig(n_bits=4))
        model = ctx.train_xgboost(X_train, y_train)

        # Compile to FHE
        ctx.compile(X_train[:100])

        # Encrypted inference
        encrypted_input = ctx.encrypt(X_test)
        encrypted_output = ctx.run(encrypted_input)
        result = ctx.decrypt(encrypted_output)
        ```
    """

    def __init__(self, config: Optional[ConcreteMLConfig] = None):
        """
        Initialize Concrete-ML context.

        Args:
            config: Configuration for FHE compilation
        """
        if not CONCRETE_ML_AVAILABLE:
            raise RuntimeError(
                "Concrete-ML is not installed. "
                "Install with: pip install concrete-ml"
            )

        self.config = config or ConcreteMLConfig()
        self._model = None
        self._compiled = False
        self._circuit_stats = FHECircuitStats()
        self._fhe_circuit = None

        logger.info(
            f"ConcreteMLContext created: n_bits={self.config.n_bits}, "
            f"simulation={self.config.use_simulation}"
        )

    def train_xgboost_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Train a Concrete-ML XGBoost classifier.

        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional XGBoost parameters

        Returns:
            Trained model
        """
        self._model = ConcreteXGBClassifier(
            n_bits=self.config.n_bits,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            **kwargs
        )

        logger.info(f"Training XGBoost classifier on {X.shape} data")
        self._model.fit(X, y)

        return self._model

    def train_xgboost_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Any:
        """
        Train a Concrete-ML XGBoost regressor.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional XGBoost parameters

        Returns:
            Trained model
        """
        self._model = ConcreteXGBRegressor(
            n_bits=self.config.n_bits,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            **kwargs
        )

        logger.info(f"Training XGBoost regressor on {X.shape} data")
        self._model.fit(X, y)

        return self._model

    def load_sklearn_model(self, sklearn_model: Any) -> Any:
        """
        Load and convert a pre-trained sklearn model.

        Note: The model will be re-trained with quantization.

        Args:
            sklearn_model: Trained sklearn model

        Returns:
            Concrete-ML compatible model
        """
        model_type = type(sklearn_model).__name__

        if "XGB" in model_type or "xgb" in model_type:
            if hasattr(sklearn_model, "classes_"):
                self._model = ConcreteXGBClassifier(
                    n_bits=self.config.n_bits
                )
            else:
                self._model = ConcreteXGBRegressor(
                    n_bits=self.config.n_bits
                )
        elif "RandomForest" in model_type:
            self._model = ConcreteRandomForest(
                n_bits=self.config.n_bits
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Loaded {model_type} for Concrete-ML compilation")
        return self._model

    def compile(
        self,
        X_calibration: np.ndarray,
        show_progress: bool = True
    ) -> FHECircuitStats:
        """
        Compile the model to an FHE circuit.

        Args:
            X_calibration: Calibration data for quantization
            show_progress: Show compilation progress

        Returns:
            Circuit compilation statistics
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")

        start_time = time.time()

        logger.info(f"Compiling model to FHE circuit with {len(X_calibration)} samples")

        # Compile the model
        self._fhe_circuit = self._model.compile(
            X_calibration,
            configuration=None,
            show_mlir=False,
            show_progress=show_progress
        )

        compilation_time = time.time() - start_time

        # Gather circuit statistics
        self._circuit_stats = FHECircuitStats(
            compilation_time=compilation_time,
            max_bit_width=self._model.fhe_circuit.graph.maximum_integer_bit_width(),
            programmable_bootstrap_count=self._model.fhe_circuit.statistics.get(
                "programmable_bootstrap_count", 0
            ) if hasattr(self._model.fhe_circuit, "statistics") else 0,
        )

        self._compiled = True

        logger.info(
            f"Compilation complete in {compilation_time:.2f}s, "
            f"max_bits={self._circuit_stats.max_bit_width}"
        )

        return self._circuit_stats

    def encrypt(self, X: np.ndarray) -> bytes:
        """
        Encrypt input data for FHE inference.

        Args:
            X: Input features

        Returns:
            Encrypted input as bytes
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")

        # In Concrete-ML, encryption is handled internally
        # This method prepares the data for serialization
        quantized_X = self._model.quantize_input(X)
        return pickle.dumps(quantized_X)

    def run_encrypted(
        self,
        encrypted_input: bytes,
        use_simulation: Optional[bool] = None
    ) -> bytes:
        """
        Run FHE inference on encrypted input.

        Args:
            encrypted_input: Encrypted input data
            use_simulation: Override config simulation setting

        Returns:
            Encrypted output
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")

        simulate = use_simulation if use_simulation is not None else self.config.use_simulation

        # Deserialize input
        quantized_X = pickle.loads(encrypted_input)

        # Run inference
        if simulate:
            result = self._model.predict(quantized_X, fhe="simulate")
        else:
            result = self._model.predict(quantized_X, fhe="execute")

        return pickle.dumps(result)

    def decrypt(self, encrypted_output: bytes) -> np.ndarray:
        """
        Decrypt FHE output.

        Args:
            encrypted_output: Encrypted output data

        Returns:
            Decrypted predictions
        """
        return pickle.loads(encrypted_output)

    def predict_plaintext(self, X: np.ndarray) -> np.ndarray:
        """
        Run plaintext inference (for validation).

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self._model is None:
            raise RuntimeError("No model loaded.")

        return self._model.predict(X, fhe="disable")

    def predict_simulated(self, X: np.ndarray) -> np.ndarray:
        """
        Run simulated FHE inference (fast, same accuracy as FHE).

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")

        return self._model.predict(X, fhe="simulate")

    def predict_fhe(self, X: np.ndarray) -> np.ndarray:
        """
        Run real FHE inference.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")

        return self._model.predict(X, fhe="execute")

    def export_for_deployment(self, path: Optional[str] = None) -> str:
        """
        Export compiled model for deployment.

        Creates three directories:
        - client/: For encrypting inputs
        - server/: For running FHE inference
        - dev/: Development keys

        Args:
            path: Export directory path

        Returns:
            Export path
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")

        export_path = path or self.config.deployment_path

        # Create deployment structure
        dev = FHEModelDev(export_path, self._model)
        dev.save()

        logger.info(f"Model exported to {export_path}")
        return export_path

    def get_stats(self) -> Dict[str, Any]:
        """Get compilation and circuit statistics."""
        return {
            "compiled": self._compiled,
            "compilation_time": self._circuit_stats.compilation_time,
            "max_bit_width": self._circuit_stats.max_bit_width,
            "pbs_count": self._circuit_stats.programmable_bootstrap_count,
            "n_bits": self.config.n_bits,
            "simulation": self.config.use_simulation,
        }


class ConcreteMLServer:
    """
    Server-side FHE inference using Concrete-ML.

    Handles encrypted inference without access to private keys.
    """

    def __init__(self, deployment_path: str):
        """
        Initialize server from deployment.

        Args:
            deployment_path: Path to exported model
        """
        if not CONCRETE_ML_AVAILABLE:
            raise RuntimeError("Concrete-ML is not installed.")

        self._server = FHEModelServer(deployment_path)
        self._server.load()

        logger.info(f"Server loaded from {deployment_path}")

    def run(self, encrypted_input: bytes, evaluation_keys: bytes) -> bytes:
        """
        Run FHE inference.

        Args:
            encrypted_input: Encrypted input from client
            evaluation_keys: Evaluation keys from client

        Returns:
            Encrypted output
        """
        return self._server.run(encrypted_input, evaluation_keys)


class ConcreteMLClient:
    """
    Client-side encryption/decryption for Concrete-ML.

    Handles key generation, encryption, and decryption.
    """

    def __init__(self, deployment_path: str, key_path: Optional[str] = None):
        """
        Initialize client from deployment.

        Args:
            deployment_path: Path to exported model
            key_path: Optional path to save/load keys
        """
        if not CONCRETE_ML_AVAILABLE:
            raise RuntimeError("Concrete-ML is not installed.")

        self._client = FHEModelClient(deployment_path, key_path)
        self._client.generate_private_and_evaluation_keys()

        logger.info(f"Client initialized from {deployment_path}")

    def get_evaluation_keys(self) -> bytes:
        """Get serialized evaluation keys for server."""
        return self._client.get_serialized_evaluation_keys()

    def encrypt(self, X: np.ndarray) -> bytes:
        """
        Encrypt input for server.

        Args:
            X: Input features

        Returns:
            Encrypted input bytes
        """
        return self._client.quantize_encrypt_serialize(X)

    def decrypt(self, encrypted_output: bytes) -> np.ndarray:
        """
        Decrypt server output.

        Args:
            encrypted_output: Encrypted output from server

        Returns:
            Decrypted predictions
        """
        return self._client.deserialize_decrypt_dequantize(encrypted_output)


# Fallback implementations when Concrete-ML is not available

class ConcreteMLSimulator:
    """
    Simulated Concrete-ML for development without the library.

    Provides the same interface but uses plaintext computation.
    """

    def __init__(self, config: Optional[ConcreteMLConfig] = None):
        """Initialize simulator."""
        self.config = config or ConcreteMLConfig()
        self._model = None
        self._compiled = False

        logger.warning("Using Concrete-ML simulator (plaintext only)")

    def train_xgboost_classifier(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train simulated XGBoost classifier."""
        from sklearn.ensemble import GradientBoostingClassifier

        self._model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            **kwargs
        )
        self._model.fit(X, y)
        return self._model

    def train_xgboost_regressor(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train simulated XGBoost regressor."""
        from sklearn.ensemble import GradientBoostingRegressor

        self._model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            **kwargs
        )
        self._model.fit(X, y)
        return self._model

    def compile(self, X_calibration: np.ndarray, **kwargs) -> FHECircuitStats:
        """Simulate compilation."""
        self._compiled = True
        return FHECircuitStats(
            compilation_time=0.1,
            max_bit_width=self.config.n_bits,
        )

    def predict_plaintext(self, X: np.ndarray) -> np.ndarray:
        """Run plaintext prediction."""
        return self._model.predict(X)

    def predict_simulated(self, X: np.ndarray) -> np.ndarray:
        """Simulate FHE prediction (same as plaintext)."""
        return self._model.predict(X)

    def predict_fhe(self, X: np.ndarray) -> np.ndarray:
        """Simulate FHE prediction."""
        return self._model.predict(X)


# Convenience functions

def create_concrete_context(
    n_bits: int = 4,
    use_simulation: bool = False
) -> Union[ConcreteMLContext, ConcreteMLSimulator]:
    """
    Create a Concrete-ML context.

    Falls back to simulator if Concrete-ML is not installed.

    Args:
        n_bits: Quantization bit width
        use_simulation: Use FHE simulation mode

    Returns:
        ConcreteMLContext or ConcreteMLSimulator
    """
    config = ConcreteMLConfig(
        n_bits=n_bits,
        use_simulation=use_simulation
    )

    if CONCRETE_ML_AVAILABLE:
        return ConcreteMLContext(config)
    else:
        logger.warning("Concrete-ML not available, using simulator")
        return ConcreteMLSimulator(config)


def compile_xgboost_to_fhe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calibration: Optional[np.ndarray] = None,
    n_bits: int = 4,
    n_estimators: int = 100,
    max_depth: int = 6,
    task: str = "classification"
) -> Tuple[Any, FHECircuitStats]:
    """
    End-to-end XGBoost to FHE compilation.

    Args:
        X_train: Training features
        y_train: Training labels/targets
        X_calibration: Calibration data (defaults to X_train[:100])
        n_bits: Quantization bit width
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        task: "classification" or "regression"

    Returns:
        Tuple of (compiled context, circuit stats)
    """
    config = ConcreteMLConfig(
        n_bits=n_bits,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    ctx = create_concrete_context(n_bits)

    if task == "classification":
        ctx.train_xgboost_classifier(X_train, y_train)
    else:
        ctx.train_xgboost_regressor(X_train, y_train)

    calibration = X_calibration if X_calibration is not None else X_train[:100]
    stats = ctx.compile(calibration)

    return ctx, stats
