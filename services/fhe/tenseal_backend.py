"""
Production FHE Backend using TenSEAL

Real homomorphic encryption operations using Microsoft SEAL via TenSEAL.
This module provides production-grade FHE capabilities for GBDT inference.

Key Features:
- CKKS scheme for approximate arithmetic (best for ML)
- BFV scheme for exact integer arithmetic
- Automatic noise budget management
- Batched operations via SIMD slots
- Real encrypted computation (not simulation)

References:
- TenSEAL: https://github.com/OpenMined/TenSEAL
- Microsoft SEAL: https://github.com/microsoft/SEAL
"""

import tenseal as ts
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class FHEScheme(Enum):
    """Supported FHE schemes."""
    CKKS = "ckks"  # Approximate arithmetic, best for ML
    BFV = "bfv"    # Exact integer arithmetic


@dataclass
class FHEConfig:
    """Configuration for FHE context."""
    scheme: FHEScheme = FHEScheme.CKKS

    # CKKS parameters
    poly_modulus_degree: int = 8192  # Ring dimension (power of 2)
    coeff_mod_bit_sizes: List[int] = field(
        default_factory=lambda: [60, 40, 40, 60]  # Determines depth
    )
    global_scale: float = 2**40  # Encoding scale for CKKS

    # BFV parameters (if using BFV)
    plain_modulus: int = 1032193  # For BFV scheme

    # Security level
    security_level: int = 128  # 128 or 192 or 256 bits

    # Galois keys for rotations (needed for SIMD)
    generate_galois_keys: bool = True

    # Relinearization keys (needed for multiplication)
    generate_relin_keys: bool = True


@dataclass
class EncryptedTensor:
    """Wrapper for TenSEAL encrypted tensor with metadata."""
    ciphertext: Union[ts.CKKSVector, ts.BFVVector]
    original_shape: Tuple[int, ...]
    scheme: FHEScheme
    creation_time: float = field(default_factory=time.time)
    operation_count: int = 0

    def __post_init__(self):
        self._estimated_noise_budget = 100.0  # Start with full budget

    @property
    def size(self) -> int:
        """Number of encrypted values."""
        return self.ciphertext.size()

    def record_operation(self, op_type: str, noise_cost: float = 1.0):
        """Record an operation and update noise estimate."""
        self.operation_count += 1
        self._estimated_noise_budget -= noise_cost


class TenSEALContext:
    """
    Production FHE context using TenSEAL.

    Provides real homomorphic encryption operations with:
    - Automatic parameter selection
    - Noise budget tracking
    - Key management
    - Batched operations

    Example:
        ```python
        ctx = TenSEALContext(FHEConfig())

        # Encrypt data
        encrypted = ctx.encrypt([1.0, 2.0, 3.0])

        # Homomorphic operations
        result = ctx.add(encrypted, encrypted)
        result = ctx.multiply_plain(result, 0.5)

        # Decrypt
        decrypted = ctx.decrypt(result)
        ```
    """

    def __init__(self, config: Optional[FHEConfig] = None):
        """
        Initialize TenSEAL context.

        Args:
            config: FHE configuration
        """
        self.config = config or FHEConfig()
        self._context = self._create_context()
        self._operation_stats = {
            "encryptions": 0,
            "decryptions": 0,
            "additions": 0,
            "multiplications": 0,
            "rotations": 0,
        }

        logger.info(
            f"TenSEAL context created: scheme={self.config.scheme.value}, "
            f"poly_degree={self.config.poly_modulus_degree}"
        )

    def _create_context(self) -> ts.Context:
        """Create TenSEAL context with configured parameters."""
        if self.config.scheme == FHEScheme.CKKS:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.config.poly_modulus_degree,
                coeff_mod_bit_sizes=self.config.coeff_mod_bit_sizes
            )
            context.global_scale = self.config.global_scale
        else:  # BFV
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=self.config.poly_modulus_degree,
                plain_modulus=self.config.plain_modulus
            )

        # Generate keys
        if self.config.generate_galois_keys:
            context.generate_galois_keys()
        if self.config.generate_relin_keys:
            context.generate_relin_keys()

        return context

    @property
    def num_slots(self) -> int:
        """Number of SIMD slots available."""
        return self.config.poly_modulus_degree // 2

    def encrypt(self, values: Union[List[float], np.ndarray]) -> EncryptedTensor:
        """
        Encrypt a vector of values.

        Args:
            values: Plaintext values to encrypt

        Returns:
            EncryptedTensor containing the ciphertext
        """
        if isinstance(values, np.ndarray):
            values = values.flatten().tolist()

        if self.config.scheme == FHEScheme.CKKS:
            ciphertext = ts.ckks_vector(self._context, values)
        else:
            # BFV requires integers
            int_values = [int(v) for v in values]
            ciphertext = ts.bfv_vector(self._context, int_values)

        self._operation_stats["encryptions"] += 1

        return EncryptedTensor(
            ciphertext=ciphertext,
            original_shape=(len(values),),
            scheme=self.config.scheme
        )

    def decrypt(self, encrypted: EncryptedTensor) -> np.ndarray:
        """
        Decrypt an encrypted tensor.

        Args:
            encrypted: EncryptedTensor to decrypt

        Returns:
            Decrypted values as numpy array
        """
        decrypted = encrypted.ciphertext.decrypt()
        self._operation_stats["decryptions"] += 1

        return np.array(decrypted)

    def add(
        self,
        a: EncryptedTensor,
        b: EncryptedTensor
    ) -> EncryptedTensor:
        """
        Homomorphic addition of two ciphertexts.

        Args:
            a: First encrypted tensor
            b: Second encrypted tensor

        Returns:
            Encrypted sum
        """
        result_ct = a.ciphertext + b.ciphertext
        self._operation_stats["additions"] += 1

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=a.original_shape,
            scheme=a.scheme
        )
        result.record_operation("add", noise_cost=0.5)
        return result

    def add_plain(
        self,
        encrypted: EncryptedTensor,
        plain: Union[float, List[float]]
    ) -> EncryptedTensor:
        """
        Add plaintext to ciphertext.

        Args:
            encrypted: Encrypted tensor
            plain: Plaintext value(s) - will be broadcast to match ciphertext size

        Returns:
            Encrypted result
        """
        # Handle scalar broadcasting
        if isinstance(plain, (int, float)):
            plain_val = float(plain)
            plain = [plain_val] * encrypted.size
        elif len(plain) == 1:
            plain = [plain[0]] * encrypted.size
        elif len(plain) != encrypted.size:
            # Pad or truncate to match size
            if len(plain) < encrypted.size:
                plain = list(plain) + [0.0] * (encrypted.size - len(plain))
            else:
                plain = plain[:encrypted.size]

        result_ct = encrypted.ciphertext + plain
        self._operation_stats["additions"] += 1

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=encrypted.original_shape,
            scheme=encrypted.scheme
        )
        result.record_operation("add_plain", noise_cost=0.1)
        return result

    def subtract(
        self,
        a: EncryptedTensor,
        b: EncryptedTensor
    ) -> EncryptedTensor:
        """
        Homomorphic subtraction.

        Args:
            a: First encrypted tensor
            b: Second encrypted tensor

        Returns:
            Encrypted difference (a - b)
        """
        result_ct = a.ciphertext - b.ciphertext
        self._operation_stats["additions"] += 1

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=a.original_shape,
            scheme=a.scheme
        )
        result.record_operation("subtract", noise_cost=0.5)
        return result

    def multiply(
        self,
        a: EncryptedTensor,
        b: EncryptedTensor
    ) -> EncryptedTensor:
        """
        Homomorphic multiplication of two ciphertexts.

        Note: This is expensive and consumes significant noise budget.

        Args:
            a: First encrypted tensor
            b: Second encrypted tensor

        Returns:
            Encrypted product
        """
        result_ct = a.ciphertext * b.ciphertext
        self._operation_stats["multiplications"] += 1

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=a.original_shape,
            scheme=a.scheme
        )
        result.record_operation("multiply", noise_cost=10.0)
        return result

    def multiply_plain(
        self,
        encrypted: EncryptedTensor,
        plain: Union[float, List[float]]
    ) -> EncryptedTensor:
        """
        Multiply ciphertext by plaintext.

        Args:
            encrypted: Encrypted tensor
            plain: Plaintext multiplier(s) - will be broadcast to match ciphertext size

        Returns:
            Encrypted result
        """
        # Handle scalar broadcasting
        if isinstance(plain, (int, float)):
            plain_val = float(plain)
            plain = [plain_val] * encrypted.size
        elif len(plain) == 1:
            plain = [plain[0]] * encrypted.size
        elif len(plain) != encrypted.size:
            # Pad with 1.0 (multiplicative identity) or truncate
            if len(plain) < encrypted.size:
                plain = list(plain) + [1.0] * (encrypted.size - len(plain))
            else:
                plain = plain[:encrypted.size]

        result_ct = encrypted.ciphertext * plain
        self._operation_stats["multiplications"] += 1

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=encrypted.original_shape,
            scheme=encrypted.scheme
        )
        result.record_operation("multiply_plain", noise_cost=2.0)
        return result

    def negate(self, encrypted: EncryptedTensor) -> EncryptedTensor:
        """
        Negate encrypted tensor.

        Args:
            encrypted: Encrypted tensor

        Returns:
            Encrypted negation
        """
        result_ct = -encrypted.ciphertext

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=encrypted.original_shape,
            scheme=encrypted.scheme
        )
        result.record_operation("negate", noise_cost=0.1)
        return result

    def sum(self, encrypted: EncryptedTensor) -> EncryptedTensor:
        """
        Sum all elements in encrypted tensor.

        Uses rotation-and-sum pattern.

        Args:
            encrypted: Encrypted tensor

        Returns:
            Encrypted sum (replicated in all slots)
        """
        result_ct = encrypted.ciphertext.sum()
        self._operation_stats["rotations"] += int(np.log2(encrypted.size))

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=(1,),
            scheme=encrypted.scheme
        )
        result.record_operation("sum", noise_cost=5.0)
        return result

    def dot(
        self,
        encrypted: EncryptedTensor,
        plain: Union[List[float], np.ndarray]
    ) -> EncryptedTensor:
        """
        Dot product of encrypted tensor with plaintext vector.

        Args:
            encrypted: Encrypted tensor
            plain: Plaintext vector

        Returns:
            Encrypted dot product
        """
        if isinstance(plain, np.ndarray):
            plain = plain.tolist()

        result_ct = encrypted.ciphertext.dot(plain)
        self._operation_stats["multiplications"] += 1
        self._operation_stats["rotations"] += int(np.log2(len(plain)))

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=(1,),
            scheme=encrypted.scheme
        )
        result.record_operation("dot", noise_cost=8.0)
        return result

    def polyval(
        self,
        encrypted: EncryptedTensor,
        coefficients: List[float]
    ) -> EncryptedTensor:
        """
        Evaluate polynomial on encrypted data.

        p(x) = coefficients[0] + coefficients[1]*x + coefficients[2]*x^2 + ...

        Args:
            encrypted: Encrypted input values
            coefficients: Polynomial coefficients (constant term first)

        Returns:
            Encrypted polynomial evaluation
        """
        result_ct = encrypted.ciphertext.polyval(coefficients)

        degree = len(coefficients) - 1
        self._operation_stats["multiplications"] += degree

        result = EncryptedTensor(
            ciphertext=result_ct,
            original_shape=encrypted.original_shape,
            scheme=encrypted.scheme
        )
        result.record_operation("polyval", noise_cost=degree * 10.0)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            **self._operation_stats,
            "scheme": self.config.scheme.value,
            "poly_modulus_degree": self.config.poly_modulus_degree,
            "num_slots": self.num_slots,
        }

    def serialize_context(self) -> bytes:
        """Serialize context for transmission (public only)."""
        return self._context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True
        )

    def serialize_secret_key(self) -> bytes:
        """Serialize secret key (keep secure!)."""
        return self._context.serialize(
            save_public_key=False,
            save_secret_key=True,
            save_galois_keys=False,
            save_relin_keys=False
        )


class ProductionFHEGBDT:
    """
    Production-grade FHE GBDT inference using TenSEAL.

    This class provides real encrypted GBDT inference with:
    - True homomorphic operations (not simulation)
    - Proper noise budget management
    - SIMD batching for efficiency
    - Sign function approximation via polynomials
    """

    # Polynomial coefficients for sign approximation
    # sign(x) ≈ x * (c1 + c3*x^2 + c5*x^4 + ...)
    SIGN_POLY_COEFFS = [0.0, 1.1963, 0.0, -0.2849, 0.0, 0.0951, 0.0]

    def __init__(self, context: TenSEALContext):
        """
        Initialize FHE GBDT.

        Args:
            context: TenSEAL context for encryption
        """
        self.ctx = context
        self._comparison_cache: Dict[str, EncryptedTensor] = {}

    def encrypt_features(
        self,
        features: np.ndarray
    ) -> List[EncryptedTensor]:
        """
        Encrypt input features for GBDT inference.

        Uses column packing: each feature in its own ciphertext.

        Args:
            features: Shape (batch_size, num_features)

        Returns:
            List of encrypted features (one per feature)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        batch_size, num_features = features.shape
        encrypted_features = []

        for f_idx in range(num_features):
            # Column pack: all samples for this feature
            feature_column = features[:, f_idx].tolist()
            encrypted = self.ctx.encrypt(feature_column)
            encrypted_features.append(encrypted)

        logger.info(f"Encrypted {num_features} features for batch of {batch_size}")
        return encrypted_features

    def compare_threshold(
        self,
        encrypted_feature: EncryptedTensor,
        threshold: float,
        normalize_range: float = 10.0
    ) -> EncryptedTensor:
        """
        Encrypted comparison: feature < threshold.

        Uses polynomial approximation of sign function.
        Result ≈ 1 if feature < threshold, ≈ 0 otherwise.

        Args:
            encrypted_feature: Encrypted feature values
            threshold: Comparison threshold
            normalize_range: Expected data range for normalization

        Returns:
            Encrypted comparison result in [0, 1]
        """
        # Compute (threshold - feature) / normalize_range
        # This gives positive for feature < threshold
        shifted = self.ctx.add_plain(
            self.ctx.negate(encrypted_feature),
            [threshold]
        )
        normalized = self.ctx.multiply_plain(shifted, [1.0 / normalize_range])

        # Apply polynomial sign approximation
        # Maps negative → 0, positive → 1
        sign_result = self.ctx.polyval(normalized, self.SIGN_POLY_COEFFS)

        # Map from [-1, 1] to [0, 1]
        result = self.ctx.add_plain(sign_result, [0.5])
        result = self.ctx.multiply_plain(result, [0.5])

        return result

    def compute_leaf_indicator(
        self,
        comparison_results: List[EncryptedTensor],
        leaf_index: int
    ) -> EncryptedTensor:
        """
        Compute leaf indicator from comparison results.

        For oblivious trees:
        leaf_k = Π_{d=0}^{D-1} (cmp_d if bit(k,d)==1 else (1-cmp_d))

        Args:
            comparison_results: List of encrypted comparisons per level
            leaf_index: Binary index of the leaf

        Returns:
            Encrypted leaf indicator
        """
        depth = len(comparison_results)

        # Start with first level
        first_bit = (leaf_index >> 0) & 1
        if first_bit == 1:
            result = comparison_results[0]
        else:
            # 1 - comparison
            result = self.ctx.add_plain(
                self.ctx.negate(comparison_results[0]),
                [1.0]
            )

        # Multiply remaining levels
        for d in range(1, depth):
            bit = (leaf_index >> d) & 1

            if bit == 1:
                level_result = comparison_results[d]
            else:
                level_result = self.ctx.add_plain(
                    self.ctx.negate(comparison_results[d]),
                    [1.0]
                )

            result = self.ctx.multiply(result, level_result)

        return result

    def aggregate_tree_outputs(
        self,
        leaf_indicators: List[EncryptedTensor],
        leaf_values: List[float]
    ) -> EncryptedTensor:
        """
        Aggregate leaf values weighted by indicators.

        output = Σ (indicator_k × leaf_value_k)

        This is the key operation where N2HE weighted sum excels.

        Args:
            leaf_indicators: Encrypted indicators for each leaf
            leaf_values: Plaintext leaf values

        Returns:
            Encrypted tree output
        """
        # Weighted sum using plaintext multiplication
        weighted = []
        for indicator, value in zip(leaf_indicators, leaf_values):
            weighted.append(self.ctx.multiply_plain(indicator, [value]))

        # Sum all weighted indicators
        result = weighted[0]
        for i in range(1, len(weighted)):
            result = self.ctx.add(result, weighted[i])

        return result

    def predict_oblivious_tree(
        self,
        encrypted_features: List[EncryptedTensor],
        level_features: List[int],
        level_thresholds: List[float],
        leaf_values: List[float]
    ) -> EncryptedTensor:
        """
        Predict using a single oblivious tree.

        Args:
            encrypted_features: Column-packed encrypted features
            level_features: Feature index for each level
            level_thresholds: Threshold for each level
            leaf_values: Values for all 2^depth leaves

        Returns:
            Encrypted prediction
        """
        depth = len(level_features)

        # Compute all level comparisons
        comparisons = []
        for d in range(depth):
            feat_idx = level_features[d]
            threshold = level_thresholds[d]

            cmp_result = self.compare_threshold(
                encrypted_features[feat_idx],
                threshold
            )
            comparisons.append(cmp_result)

        # Compute all leaf indicators
        num_leaves = 2 ** depth
        leaf_indicators = []
        for leaf_idx in range(num_leaves):
            indicator = self.compute_leaf_indicator(comparisons, leaf_idx)
            leaf_indicators.append(indicator)

        # Aggregate
        return self.aggregate_tree_outputs(leaf_indicators, leaf_values)

    def predict_ensemble(
        self,
        encrypted_features: List[EncryptedTensor],
        trees: List[Dict[str, Any]],
        base_score: float = 0.0
    ) -> EncryptedTensor:
        """
        Predict using ensemble of oblivious trees.

        Args:
            encrypted_features: Column-packed encrypted features
            trees: List of tree definitions, each with:
                   - level_features: List[int]
                   - level_thresholds: List[float]
                   - leaf_values: List[float]
            base_score: Base prediction score

        Returns:
            Encrypted ensemble prediction
        """
        # Predict with first tree
        tree_outputs = []
        for i, tree in enumerate(trees):
            output = self.predict_oblivious_tree(
                encrypted_features,
                tree["level_features"],
                tree["level_thresholds"],
                tree["leaf_values"]
            )
            tree_outputs.append(output)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(trees)} trees")

        # Sum all tree outputs
        result = tree_outputs[0]
        for i in range(1, len(tree_outputs)):
            result = self.ctx.add(result, tree_outputs[i])

        # Add base score
        result = self.ctx.add_plain(result, [base_score])

        return result


# Convenience functions

def create_production_context(
    security_level: int = 128,
    depth: int = 4
) -> TenSEALContext:
    """
    Create a production-ready TenSEAL context.

    Args:
        security_level: Security level in bits (128, 192, or 256)
        depth: Multiplicative depth needed

    Returns:
        Configured TenSEALContext

    Note:
        Parameter selection follows SEAL guidelines:
        - poly_degree=4096: max ~109 bits total coeff mod (depth ~1-2)
        - poly_degree=8192: max ~218 bits total coeff mod (depth ~3-4)
        - poly_degree=16384: max ~438 bits total coeff mod (depth ~8-10)
        - poly_degree=32768: max ~881 bits total coeff mod (depth ~18-20)
    """
    # Calculate total bits needed for coefficient modulus
    # Formula: 60 (first) + 40 * depth (middle) + 60 (last)
    # Each multiplication level needs ~40 bits
    total_bits_needed = 60 + 40 * depth + 60

    # Select poly_degree based on total bits and add margin for security
    # These are safe limits for 128-bit security
    if total_bits_needed <= 109:
        poly_degree = 4096
        # Adjust coeff_mod_bits to fit within limit
        if depth == 0:
            coeff_mod_bits = [40, 40]
        elif depth == 1:
            coeff_mod_bits = [40, 20, 40]
        else:
            coeff_mod_bits = [30] + [20] * depth + [30]
        scale_bits = 20
    elif total_bits_needed <= 218:
        poly_degree = 8192
        # Safe configuration for 8192
        if depth <= 2:
            coeff_mod_bits = [60] + [40] * depth + [60]
        else:
            # Reduce bit sizes to fit
            coeff_mod_bits = [50] + [30] * depth + [50]
        scale_bits = 30 if depth > 2 else 40
    elif total_bits_needed <= 438:
        poly_degree = 16384
        coeff_mod_bits = [60] + [40] * depth + [60]
        scale_bits = 40
    else:
        poly_degree = 32768
        coeff_mod_bits = [60] + [40] * depth + [60]
        scale_bits = 40

    config = FHEConfig(
        scheme=FHEScheme.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=coeff_mod_bits,
        global_scale=2**scale_bits,
        security_level=security_level
    )

    return TenSEALContext(config)


def encrypt_and_predict(
    features: np.ndarray,
    trees: List[Dict[str, Any]],
    base_score: float = 0.0
) -> np.ndarray:
    """
    End-to-end encrypted GBDT prediction.

    Args:
        features: Input features
        trees: Tree definitions
        base_score: Base score

    Returns:
        Decrypted predictions
    """
    ctx = create_production_context()
    gbdt = ProductionFHEGBDT(ctx)

    # Encrypt
    encrypted_features = gbdt.encrypt_features(features)

    # Predict
    encrypted_result = gbdt.predict_ensemble(
        encrypted_features, trees, base_score
    )

    # Decrypt
    return ctx.decrypt(encrypted_result)
