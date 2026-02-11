"""
Linear Model Optimizer for FHE

Generates FHE execution plans for linear models (Logistic Regression, GLMs).
Leverages innovations:
- Gradient noise allocation for adaptive precision encoding
- Polynomial leaves evaluator for Horner's method link function evaluation

FHE Execution Plan for linear model:
1. DOT_PRODUCT: Σ w_i * x_i (native FHE multiply-accumulate)
2. ADD_BIAS: + intercept (plaintext addition)
3. LINK_FUNCTION: apply polynomial approximation of link (Horner's method)
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any

from .ir import (
    ModelIR, ModelFamily, LinkFunction, LinearPlanIR, LinearPlanOp,
    PackingLayout,
)
from .link_functions import LinkFunctionApproximator, LinkApproximation

logger = logging.getLogger(__name__)

# Try to import gradient noise innovation for adaptive precision
try:
    from services.innovations.gradient_noise import (
        GradientAwareNoiseAllocator,
        AdaptivePrecisionConfig,
        FeatureImportance,
    )
    GRADIENT_NOISE_AVAILABLE = True
except ImportError:
    GRADIENT_NOISE_AVAILABLE = False


class LinearModelOptimizer:
    """
    Optimizer for linear/GLM models under FHE.

    Generates LinearPlanIR execution plans with:
    - Column-packed feature layout (reuse MOAI packing)
    - Adaptive precision based on coefficient magnitude
    - Polynomial link function approximation
    """

    MAX_FEATURES = 65536
    MAX_POLY_DEGREE = 15  # Maximum link function polynomial degree

    def __init__(
        self,
        profile: str = "latency",
        target: str = "cpu",
        link_poly_degree: Optional[int] = None,
    ):
        self.profile = profile.lower()
        self.target = target.lower()
        self.link_poly_degree = link_poly_degree
        self.link_approximator = LinkFunctionApproximator(default_degree=link_poly_degree)

        logger.info(
            f"LinearModelOptimizer: profile={profile}, target={target}"
        )

    def optimize(self, model: ModelIR) -> LinearPlanIR:
        """
        Generate FHE execution plan for a linear model.

        Args:
            model: Parsed ModelIR with model_family=LINEAR

        Returns:
            LinearPlanIR execution plan

        Raises:
            ValueError: If model is not a linear model or is invalid
        """
        self._validate(model)

        coefficients = model.coefficients
        num_features = coefficients.num_features
        link_name = model.link_function.value

        # 1. Create packing layout (one ciphertext per feature, column-packed)
        feature_map = {i: i for i in range(num_features)}
        layout = PackingLayout(
            layout_type="moai_column",
            feature_to_ciphertext=feature_map,
            slots=1,  # Linear model: one slot per feature
        )

        # 2. Generate link function approximation
        link_approx = self.link_approximator.approximate(link_name)

        # 3. Build noise allocation (leverage gradient noise innovation)
        noise_metadata = {}
        if GRADIENT_NOISE_AVAILABLE:
            noise_metadata = self._allocate_noise(model)

        # 4. Build operation sequence
        ops = self._build_ops(model, link_approx, noise_metadata)

        # 5. Generate plan ID
        compiled_id = hashlib.sha256(
            f"{model.model_type}:{num_features}:{link_name}".encode()
        ).hexdigest()[:16]

        plan = LinearPlanIR(
            compiled_model_id=compiled_id,
            crypto_params_id="n2he_default",
            packing_layout=layout,
            coefficients=coefficients.weights,
            intercept=coefficients.intercept,
            link_function=link_name,
            poly_coeffs=link_approx.coefficients,
            poly_degree=link_approx.degree,
            num_features=num_features,
            ops=ops,
            metadata={
                "optimizer": "LinearModelOptimizer",
                "profile": self.profile,
                "target": self.target,
                "link_function": link_name,
                "link_max_error": link_approx.max_error,
                "link_domain": list(link_approx.domain),
                "multiplicative_depth": link_approx.multiplicative_depth,
                "noise_allocation": noise_metadata,
                "model_family": model.model_family.value,
                "glm_family": model.glm_family,
            },
        )

        logger.info(
            f"Generated linear plan: {num_features} features, "
            f"link={link_name} (degree {link_approx.degree}, "
            f"err={link_approx.max_error:.6f}), "
            f"depth={link_approx.multiplicative_depth}"
        )

        return plan

    def _validate(self, model: ModelIR) -> None:
        """Validate that model is a valid linear model."""
        if model.model_family != ModelFamily.LINEAR:
            raise ValueError(
                f"Expected LINEAR model family, got {model.model_family}"
            )
        if model.coefficients is None:
            raise ValueError("Linear model must have coefficients")
        if model.coefficients.num_features <= 0:
            raise ValueError("Model has no features")
        if model.coefficients.num_features > self.MAX_FEATURES:
            raise ValueError(
                f"Model has {model.coefficients.num_features} features, "
                f"exceeds max {self.MAX_FEATURES}"
            )

    def _build_ops(
        self,
        model: ModelIR,
        link_approx: LinkApproximation,
        noise_metadata: Dict,
    ) -> List[LinearPlanOp]:
        """Build the operation sequence for FHE evaluation."""
        ops = []
        coefficients = model.coefficients

        # Op 1: DOT_PRODUCT - compute w^T * x
        # In FHE: multiply each encrypted feature by its plaintext weight, then sum
        ops.append(LinearPlanOp(
            op_type="DOT_PRODUCT",
            params={
                "weights": coefficients.weights,
                "num_features": coefficients.num_features,
                "precision_bits": noise_metadata.get("avg_precision", 12),
            },
        ))

        # Op 2: ADD_BIAS - add intercept
        ops.append(LinearPlanOp(
            op_type="ADD_BIAS",
            params={
                "intercept": coefficients.intercept,
            },
        ))

        # Op 3: LINK_FUNCTION - apply polynomial approximation
        if model.link_function != LinkFunction.IDENTITY:
            ops.append(LinearPlanOp(
                op_type="LINK_FUNCTION",
                params={
                    "link_name": model.link_function.value,
                    "poly_coeffs": link_approx.coefficients,
                    "poly_degree": link_approx.degree,
                    "domain": list(link_approx.domain),
                    "method": "horner",  # FHE-optimal evaluation
                },
            ))

        return ops

    def _allocate_noise(self, model: ModelIR) -> Dict:
        """
        Use gradient noise innovation to allocate precision
        based on coefficient magnitude (larger coeff = more important).
        """
        coefficients = model.coefficients
        weights = coefficients.weights

        # Map coefficient magnitudes to feature importance
        max_weight = max((abs(w) for w in weights), default=1.0) or 1.0

        importance_map = {}
        for i, w in enumerate(weights):
            abs_w = abs(w)
            norm_w = abs_w / max_weight
            importance_map[i] = FeatureImportance(
                feature_idx=i,
                gradient_importance=norm_w,
                frequency=1,
                average_split_gain=abs_w,
                depth_weighted_importance=norm_w,
            )

        allocator = GradientAwareNoiseAllocator()
        allocations = allocator.allocate_from_importance(
            importance_map, coefficients.num_features
        )

        if allocations:
            precision_bits = [a.precision_bits for a in allocations.values()]
            avg_p = sum(precision_bits) / len(precision_bits)
            min_p = min(precision_bits)
            max_p = max(precision_bits)
        else:
            avg_p, min_p, max_p = 12, 8, 16

        return {
            "avg_precision": avg_p,
            "min_precision": min_p,
            "max_precision": max_p,
            "innovation": "gradient_noise_allocation",
        }
