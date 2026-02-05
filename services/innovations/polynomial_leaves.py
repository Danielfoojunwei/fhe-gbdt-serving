"""
Novel Innovation #6: Polynomial Leaf Functions

Instead of scalar leaf values, use polynomial functions of features at leaves.
This enables more expressive models within FHE constraints, as polynomial
evaluation is native to FHE schemes.

Key Insight:
- Standard GBDT: leaf value = constant
- Polynomial GBDT: leaf value = a₀ + a₁x + a₂x² + ... (polynomial of features)
- Polynomial evaluation is cheap in FHE (additions and multiplications)
- Captures residual patterns within leaf regions

Benefits:
- More expressive without deeper trees
- Better accuracy with same noise budget
- Natural fit for FHE polynomial arithmetic
- Combines tree structure with local regression
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
import logging
import math

import numpy as np
from numpy.polynomial import chebyshev, polynomial

logger = logging.getLogger(__name__)


@dataclass
class PolynomialLeaf:
    """A leaf with polynomial function instead of scalar value."""
    leaf_id: int
    tree_id: int

    # Polynomial coefficients: prediction = Σ coeffs[i] * x^i
    # where x is the primary feature or a combination
    coefficients: np.ndarray

    # Which feature(s) the polynomial is over
    feature_indices: List[int]

    # Polynomial type
    poly_type: str = "standard"  # "standard", "chebyshev", "legendre"

    # Original scalar leaf value (for comparison)
    scalar_value: Optional[float] = None

    # Fitting statistics
    fit_r2: Optional[float] = None
    num_samples_fit: int = 0

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return len(self.coefficients) - 1

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at given feature values.

        Args:
            features: Shape (batch_size, num_features)

        Returns:
            Shape (batch_size,) polynomial values
        """
        if len(self.feature_indices) == 1:
            # Univariate polynomial
            x = features[:, self.feature_indices[0]]
            return self._evaluate_univariate(x)
        else:
            # Multivariate: sum of univariate polynomials
            result = np.zeros(features.shape[0])
            for i, feat_idx in enumerate(self.feature_indices):
                x = features[:, feat_idx]
                # Use subset of coefficients for this feature
                coeffs = self.coefficients[i::len(self.feature_indices)]
                result += np.polyval(coeffs[::-1], x)
            return result

    def _evaluate_univariate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate univariate polynomial."""
        if self.poly_type == "chebyshev":
            return chebyshev.chebval(x, self.coefficients)
        elif self.poly_type == "legendre":
            from numpy.polynomial import legendre
            return legendre.legval(x, self.coefficients)
        else:
            # Standard polynomial
            return np.polyval(self.coefficients[::-1], x)


@dataclass
class PolynomialLeafConfig:
    """Configuration for polynomial leaf training."""
    # Maximum polynomial degree
    max_degree: int = 3

    # Minimum samples to fit polynomial (otherwise use scalar)
    min_samples_for_poly: int = 10

    # Regularization strength
    regularization: float = 0.01

    # Polynomial type
    poly_type: str = "chebyshev"  # More stable in [-1, 1]

    # Feature selection for polynomial
    use_split_features: bool = True  # Use features from path to leaf
    max_features_in_poly: int = 2

    # R² threshold to keep polynomial (vs fall back to scalar)
    r2_threshold: float = 0.1


class PolynomialLeafTrainer:
    """
    Trains polynomial functions at GBDT leaves.

    Training process:
    1. Train standard GBDT
    2. For each leaf, collect samples that fall into it
    3. Fit polynomial to residuals within leaf
    4. Replace scalar leaf value with polynomial
    """

    def __init__(self, config: Optional[PolynomialLeafConfig] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or PolynomialLeafConfig()

    def fit_leaf_polynomials(
        self,
        model_ir: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[Tuple[int, int], PolynomialLeaf]:
        """
        Fit polynomial functions at all leaves.

        Args:
            model_ir: Trained GBDT model
            X: Training features
            y: Training targets
            sample_weight: Optional sample weights

        Returns:
            Dict of (tree_id, leaf_id) -> PolynomialLeaf
        """
        polynomial_leaves = {}

        # Compute current predictions and residuals
        current_preds = self._compute_predictions(model_ir, X)
        residuals = y - current_preds

        for tree_idx, tree in enumerate(model_ir.trees):
            # Get leaf assignments for all samples
            leaf_assignments = self._get_leaf_assignments(tree, X)

            # Get unique leaves
            leaf_ids = set(leaf_assignments)

            for leaf_id in leaf_ids:
                # Get samples in this leaf
                mask = leaf_assignments == leaf_id
                X_leaf = X[mask]
                residuals_leaf = residuals[mask]

                if len(X_leaf) < self.config.min_samples_for_poly:
                    # Not enough samples, keep scalar
                    continue

                # Get features to use in polynomial
                feature_indices = self._select_features(tree, leaf_id, X_leaf)

                # Fit polynomial
                poly_leaf = self._fit_polynomial(
                    tree_idx, leaf_id, X_leaf, residuals_leaf, feature_indices
                )

                if poly_leaf is not None:
                    polynomial_leaves[(tree_idx, leaf_id)] = poly_leaf

        logger.info(f"Fitted {len(polynomial_leaves)} polynomial leaves")
        return polynomial_leaves

    def _compute_predictions(
        self,
        model_ir: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """Compute current model predictions."""
        predictions = np.full(X.shape[0], model_ir.base_score)

        for tree in model_ir.trees:
            for i in range(X.shape[0]):
                predictions[i] += self._traverse_tree(tree, X[i])

        return predictions

    def _traverse_tree(self, tree_ir: Any, sample: np.ndarray) -> float:
        """Traverse tree for single sample."""
        node = tree_ir.nodes.get(tree_ir.root_id)

        while node is not None:
            if node.leaf_value is not None:
                return node.leaf_value

            if sample[node.feature_index] < node.threshold:
                node = tree_ir.nodes.get(node.left_child_id)
            else:
                node = tree_ir.nodes.get(node.right_child_id)

        return 0.0

    def _get_leaf_assignments(
        self,
        tree_ir: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """Get leaf ID for each sample."""
        assignments = np.zeros(X.shape[0], dtype=np.int32)

        for i in range(X.shape[0]):
            assignments[i] = self._get_leaf_id(tree_ir, X[i])

        return assignments

    def _get_leaf_id(self, tree_ir: Any, sample: np.ndarray) -> int:
        """Get leaf ID for single sample."""
        node = tree_ir.nodes.get(tree_ir.root_id)

        while node is not None:
            if node.leaf_value is not None:
                return node.node_id

            if sample[node.feature_index] < node.threshold:
                node = tree_ir.nodes.get(node.left_child_id)
            else:
                node = tree_ir.nodes.get(node.right_child_id)

        return -1

    def _select_features(
        self,
        tree_ir: Any,
        leaf_id: int,
        X_leaf: np.ndarray
    ) -> List[int]:
        """Select features for polynomial at this leaf."""
        if self.config.use_split_features:
            # Use features from path to this leaf
            path_features = self._get_path_features(tree_ir, leaf_id)
            return path_features[:self.config.max_features_in_poly]
        else:
            # Use features with highest variance in leaf
            variances = np.var(X_leaf, axis=0)
            top_indices = np.argsort(variances)[-self.config.max_features_in_poly:]
            return top_indices.tolist()

    def _get_path_features(self, tree_ir: Any, leaf_id: int) -> List[int]:
        """Get features used on path to leaf."""
        features = []

        # Find path from root to leaf
        path = self._find_path(tree_ir, tree_ir.root_id, leaf_id)

        if path:
            for node_id in path[:-1]:  # Exclude leaf itself
                node = tree_ir.nodes.get(node_id)
                if node and node.feature_index is not None:
                    features.append(node.feature_index)

        return features

    def _find_path(
        self,
        tree_ir: Any,
        current_id: int,
        target_id: int
    ) -> Optional[List[int]]:
        """Find path from current to target node."""
        if current_id == target_id:
            return [current_id]

        node = tree_ir.nodes.get(current_id)
        if node is None or node.leaf_value is not None:
            return None

        for child_id in [node.left_child_id, node.right_child_id]:
            if child_id is not None:
                path = self._find_path(tree_ir, child_id, target_id)
                if path is not None:
                    return [current_id] + path

        return None

    def _fit_polynomial(
        self,
        tree_id: int,
        leaf_id: int,
        X_leaf: np.ndarray,
        residuals: np.ndarray,
        feature_indices: List[int]
    ) -> Optional[PolynomialLeaf]:
        """Fit polynomial to residuals."""
        if not feature_indices:
            return None

        # Use primary feature for univariate polynomial
        primary_feature = feature_indices[0]
        x = X_leaf[:, primary_feature]

        # Normalize to [-1, 1] for numerical stability
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min if x_max > x_min else 1.0
        x_normalized = 2 * (x - x_min) / x_range - 1

        # Fit polynomial
        try:
            if self.config.poly_type == "chebyshev":
                coeffs = chebyshev.chebfit(
                    x_normalized, residuals, self.config.max_degree
                )
            else:
                coeffs = np.polyfit(
                    x_normalized, residuals, self.config.max_degree
                )

            # Evaluate fit quality
            if self.config.poly_type == "chebyshev":
                fitted = chebyshev.chebval(x_normalized, coeffs)
            else:
                fitted = np.polyval(coeffs[::-1], x_normalized)

            ss_res = np.sum((residuals - fitted) ** 2)
            ss_tot = np.sum((residuals - residuals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 < self.config.r2_threshold:
                return None

            return PolynomialLeaf(
                leaf_id=leaf_id,
                tree_id=tree_id,
                coefficients=coeffs,
                feature_indices=feature_indices,
                poly_type=self.config.poly_type,
                scalar_value=residuals.mean(),
                fit_r2=r2,
                num_samples_fit=len(residuals)
            )

        except Exception as e:
            logger.warning(f"Failed to fit polynomial at leaf {leaf_id}: {e}")
            return None


class FHEPolynomialEvaluator:
    """
    Evaluates polynomial leaves in FHE domain.

    Key insight: Polynomial evaluation is native to FHE!
    - Addition: homomorphic addition
    - Scalar multiplication: multiply by plaintext
    - Powers: repeated ciphertext multiplication
    """

    def __init__(self, max_degree: int = 3):
        """
        Initialize evaluator.

        Args:
            max_degree: Maximum polynomial degree to support
        """
        self.max_degree = max_degree

        # Precompute evaluation patterns
        self._patterns = self._compute_patterns()

    def _compute_patterns(self) -> Dict[int, List[Tuple[int, float]]]:
        """Precompute evaluation patterns for different degrees."""
        patterns = {}
        for degree in range(self.max_degree + 1):
            # Pattern: list of (power, coefficient_index)
            patterns[degree] = [(i, i) for i in range(degree + 1)]
        return patterns

    def evaluate_plaintext(
        self,
        poly_leaf: PolynomialLeaf,
        features: np.ndarray
    ) -> np.ndarray:
        """Evaluate polynomial leaf in plaintext."""
        return poly_leaf.evaluate(features)

    def evaluate_encrypted(
        self,
        poly_leaf: PolynomialLeaf,
        encrypted_features: Dict[int, Any],
        fhe_context: Any
    ) -> Any:
        """
        Evaluate polynomial leaf on encrypted features.

        Args:
            poly_leaf: Polynomial leaf definition
            encrypted_features: Dict of feature_idx -> encrypted_value
            fhe_context: FHE computation context

        Returns:
            Encrypted polynomial result
        """
        coeffs = poly_leaf.coefficients
        feat_idx = poly_leaf.feature_indices[0]

        if feat_idx not in encrypted_features:
            raise ValueError(f"Feature {feat_idx} not in encrypted features")

        x_ct = encrypted_features[feat_idx]

        # Evaluate polynomial: c_0 + c_1*x + c_2*x^2 + ...
        # Using Horner's method for efficiency: c_0 + x*(c_1 + x*(c_2 + ...))
        result = fhe_context.multiply_plain(
            fhe_context.create_zero(), coeffs[-1]
        )

        for i in range(len(coeffs) - 2, -1, -1):
            # result = result * x + c_i
            result = fhe_context.multiply(result, x_ct)
            coeff_ct = fhe_context.multiply_plain(
                fhe_context.create_one(), coeffs[i]
            )
            result = fhe_context.add(result, coeff_ct)

        return result

    def estimate_noise_cost(self, degree: int) -> float:
        """Estimate noise cost for polynomial of given degree."""
        # Each multiplication doubles noise approximately
        # Horner's method: degree multiplications
        base_noise = 3.2  # Initial encryption
        mul_noise = 10.0  # Per multiplication
        return base_noise + degree * mul_noise


class PolynomialLeafGBDT:
    """
    Complete GBDT model with polynomial leaf functions.

    Combines standard GBDT tree structure with polynomial leaf refinements
    for improved expressiveness within FHE constraints.
    """

    def __init__(
        self,
        model_ir: Any,
        polynomial_leaves: Optional[Dict[Tuple[int, int], PolynomialLeaf]] = None,
        config: Optional[PolynomialLeafConfig] = None
    ):
        """
        Initialize polynomial leaf GBDT.

        Args:
            model_ir: Base GBDT model
            polynomial_leaves: Pre-fitted polynomial leaves
            config: Configuration
        """
        self.model_ir = model_ir
        self.polynomial_leaves = polynomial_leaves or {}
        self.config = config or PolynomialLeafConfig()
        self.evaluator = FHEPolynomialEvaluator(config.max_degree if config else 3)

    def fit_polynomials(self, X: np.ndarray, y: np.ndarray):
        """Fit polynomial leaves from training data."""
        trainer = PolynomialLeafTrainer(self.config)
        self.polynomial_leaves = trainer.fit_leaf_polynomials(
            self.model_ir, X, y
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using polynomial leaves.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        predictions = np.full(X.shape[0], self.model_ir.base_score)

        for tree_idx, tree in enumerate(self.model_ir.trees):
            predictions += self._predict_tree(tree, tree_idx, X)

        return predictions

    def _predict_tree(
        self,
        tree_ir: Any,
        tree_idx: int,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict using single tree with polynomial leaves."""
        outputs = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            leaf_id = self._get_leaf_id(tree_ir, X[i])
            key = (tree_idx, leaf_id)

            if key in self.polynomial_leaves:
                # Use polynomial leaf
                poly_leaf = self.polynomial_leaves[key]
                outputs[i] = poly_leaf.evaluate(X[i:i+1])[0]
            else:
                # Use scalar leaf
                leaf_node = tree_ir.nodes.get(leaf_id)
                if leaf_node and leaf_node.leaf_value is not None:
                    outputs[i] = leaf_node.leaf_value

        return outputs

    def _get_leaf_id(self, tree_ir: Any, sample: np.ndarray) -> int:
        """Get leaf ID for single sample."""
        node = tree_ir.nodes.get(tree_ir.root_id)

        while node is not None:
            if node.leaf_value is not None:
                return node.node_id

            if sample[node.feature_index] < node.threshold:
                node = tree_ir.nodes.get(node.left_child_id)
            else:
                node = tree_ir.nodes.get(node.right_child_id)

        return -1

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about polynomial leaves."""
        if not self.polynomial_leaves:
            return {"num_polynomial_leaves": 0}

        degrees = [pl.degree for pl in self.polynomial_leaves.values()]
        r2_scores = [pl.fit_r2 for pl in self.polynomial_leaves.values() if pl.fit_r2]

        return {
            "num_polynomial_leaves": len(self.polynomial_leaves),
            "total_leaves": sum(len(t.nodes) for t in self.model_ir.trees) // 2,
            "coverage": len(self.polynomial_leaves) / max(1, sum(len(t.nodes) for t in self.model_ir.trees) // 2),
            "avg_degree": np.mean(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "avg_r2": np.mean(r2_scores) if r2_scores else 0,
        }


# Convenience functions

def create_polynomial_leaf_model(
    model_ir: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_degree: int = 3
) -> PolynomialLeafGBDT:
    """
    Create a polynomial leaf GBDT from a trained model.

    Args:
        model_ir: Trained GBDT model
        X_train: Training features
        y_train: Training targets
        max_degree: Maximum polynomial degree

    Returns:
        PolynomialLeafGBDT with fitted polynomials
    """
    config = PolynomialLeafConfig(max_degree=max_degree)
    poly_model = PolynomialLeafGBDT(model_ir, config=config)
    poly_model.fit_polynomials(X_train, y_train)

    return poly_model
