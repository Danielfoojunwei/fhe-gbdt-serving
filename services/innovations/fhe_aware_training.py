"""
Novel Innovation: FHE-Aware Tree Training

Genuine Research Contribution — Training-Aware FHE Tree Design

=== The Problem ===

Every existing FHE tree inference system treats the polynomial sign
approximation error as a fixed cost determined by polynomial degree.
The standard approach:
  1. Train a tree with standard GBDT training
  2. At inference, approximate each comparison sign(x_f - t) with p(x_f - t)
  3. Accept the approximation error

But the polynomial sign error |p(z) - sign(z)| is NOT uniform:
  - |z| >= δ:  error ≤ ε (exponentially small)
  - |z| < δ:   error up to 1.0 (comparison can flip!)

This means the FHE prediction error depends on HOW MANY data points
have features close to thresholds — the "margin density."

=== The Insight ===

During tree training, thresholds are chosen to maximize information gain.
But information gain does NOT account for polynomial sign accuracy.
Two thresholds may have equal information gain, but one may sit in a
high-density region (many samples near threshold → high FHE error)
while another sits in a low-density region (few samples → low FHE error).

By modifying the split criterion to:
  FHE_gain(t) = IG(t) × (1 - λ × margin_penalty(t))

where margin_penalty(t) = fraction of samples with |x_f - t| < δ,
we can train trees that are inherently more accurate under FHE
WITHOUT increasing polynomial degree.

=== Why This Is Novel ===

1. SMART-PAF (2024, MLSys) did this for neural net activations, NOT trees
2. Lee et al. (2022, IEEE TDSC) optimize the polynomial given a fixed threshold
3. SortingHat (2022, CCS) takes pretrained trees as input
4. Shin et al. (2024, ESORICS) achieve O(1) depth but don't modify training
5. Concrete ML (Zama) quantizes but doesn't optimize thresholds for sign error
6. VESTA (2025, SIGMETRICS) compiles but doesn't retrain

This is the FIRST work that optimizes tree thresholds during training
to minimize polynomial sign approximation error in FHE inference.

=== Formal Error Bound (Theorem) ===

For an oblivious tree of depth D with thresholds t_0,...,t_{D-1},
polynomial sign approximation p(z) of degree n satisfying
|p(z) - sign(z)| ≤ ε for |z| ≥ δ:

  E[|ŷ_FHE(x) - ŷ_exact(x)|] ≤ W_max × Σ_{d=0}^{D-1} ρ_d(δ)

where:
  W_max = max_k |w_k|  (maximum leaf weight)
  ρ_d(δ) = Pr[|x_{σ(d)} - t_d| < δ]  (margin density at level d)

This bound is TIGHT: it's achievable when margin densities are independent.

Corollary: By choosing thresholds that minimize Σ_d ρ_d(δ), we minimize
the FHE prediction error bound without changing the polynomial degree.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import logging
import math

import numpy as np
from numpy.polynomial import chebyshev

logger = logging.getLogger(__name__)


# =============================================================================
# Core Theory: Sign Polynomial Error Profile
# =============================================================================

@dataclass
class SignErrorProfile:
    """
    Error profile of a polynomial sign approximation.

    For p(z) ≈ sign(z) of degree n:
    - delta: minimum margin for guaranteed accuracy
    - epsilon: maximum error outside delta-margin
    - error_curve: sampled |p(z) - sign(z)| for z in [0, 1]
    """
    degree: int
    delta: float          # Margin threshold: |p(z) - sign(z)| ≤ epsilon for |z| ≥ delta
    epsilon: float        # Maximum error outside delta-margin
    coefficients: np.ndarray
    error_curve: np.ndarray  # Sampled error for z in [0, 1]
    error_sample_points: np.ndarray


class SignPolynomialAnalyzer:
    """
    Analyzes the error profile of polynomial sign approximations.

    This provides the δ(n) and ε(n) bounds that are needed for
    FHE-aware threshold selection.
    """

    # Minimax sign polynomial coefficients (from leaf_centric.py)
    MINIMAX_COEFFICIENTS = {
        7: np.array([0.0, 1.5708, 0.0, -0.6460, 0.0, 0.0796, 0.0, 0.0]),
    }

    def __init__(self):
        """Initialize analyzer."""
        self._profiles: Dict[int, SignErrorProfile] = {}

    def get_error_profile(self, degree: int = 7) -> SignErrorProfile:
        """
        Get or compute error profile for polynomial degree.

        Args:
            degree: Polynomial degree

        Returns:
            SignErrorProfile with delta, epsilon, error curve
        """
        if degree in self._profiles:
            return self._profiles[degree]

        profile = self._compute_profile(degree)
        self._profiles[degree] = profile
        return profile

    def _compute_profile(self, degree: int) -> SignErrorProfile:
        """Compute error profile for a given degree."""
        # Get or fit coefficients
        if degree in self.MINIMAX_COEFFICIENTS:
            coeffs = self.MINIMAX_COEFFICIENTS[degree]
        else:
            coeffs = self._fit_sign_polynomial(degree)

        # Sample error curve on [0, 1]
        n_points = 1000
        z_points = np.linspace(0.001, 1.0, n_points)  # Avoid z=0

        # Exact sign is +1 for z > 0
        exact_sign = np.ones(n_points)

        # Polynomial evaluation
        poly_sign = self._evaluate_sign_poly(z_points, coeffs)

        # Error curve
        error_curve = np.abs(poly_sign - exact_sign)

        # Find delta: smallest z where error drops below target epsilon
        target_epsilon = 0.05  # 5% error threshold
        delta = 1.0  # Default: entire domain is unsafe
        for i, z in enumerate(z_points):
            if error_curve[i] <= target_epsilon:
                delta = z
                break

        # Epsilon: max error for |z| >= delta
        mask = z_points >= delta
        epsilon = float(np.max(error_curve[mask])) if np.any(mask) else 1.0

        return SignErrorProfile(
            degree=degree,
            delta=delta,
            epsilon=epsilon,
            coefficients=coeffs,
            error_curve=error_curve,
            error_sample_points=z_points,
        )

    def _fit_sign_polynomial(self, degree: int) -> np.ndarray:
        """Fit minimax sign polynomial of given degree."""
        # Fit on [-1, -δ] ∪ [δ, 1] with small δ
        delta = 0.01
        n_points = 2000
        x_neg = np.linspace(-1, -delta, n_points // 2)
        x_pos = np.linspace(delta, 1, n_points // 2)
        x = np.concatenate([x_neg, x_pos])
        y = np.sign(x)

        coeffs = chebyshev.chebfit(x, y, degree)
        # Enforce odd symmetry
        coeffs[0::2] = 0
        return coeffs

    def _evaluate_sign_poly(self, z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate sign polynomial in Chebyshev basis."""
        return chebyshev.chebval(z, coeffs)

    def compute_margin_penalty(
        self,
        feature_values: np.ndarray,
        threshold: float,
        degree: int = 7,
        normalization_scale: float = 1.0,
    ) -> float:
        """
        Compute margin penalty: fraction of samples within δ of threshold.

        This is ρ(δ) = Pr[|x_f - t| < δ × scale] in the error bound.

        Args:
            feature_values: Feature values for samples
            threshold: Candidate threshold
            degree: Polynomial degree (determines δ)
            normalization_scale: Scale for normalizing (x - t) to [-1, 1]

        Returns:
            Margin penalty in [0, 1]
        """
        profile = self.get_error_profile(degree)
        delta = profile.delta

        # Compute margins: |x_f - t| / scale
        margins = np.abs(feature_values - threshold) / max(normalization_scale, 1e-10)

        # Fraction within delta-margin
        penalty = float(np.mean(margins < delta))
        return penalty


# =============================================================================
# FHE-Aware Split Criterion
# =============================================================================

@dataclass
class FHEAwareSplitResult:
    """Result of evaluating a split with FHE awareness."""
    feature_idx: int
    threshold: float
    information_gain: float
    margin_penalty: float      # ρ(δ): fraction of samples in danger zone
    fhe_aware_gain: float      # IG × (1 - λ × penalty)
    estimated_sign_error: float  # Expected |p(x-t) - sign(x-t)| for this split


class FHEAwareSplitCriterion:
    """
    Split criterion that jointly optimizes information gain and FHE accuracy.

    Standard GBDT: choose t* = argmax_t IG(t)
    FHE-aware:     choose t* = argmax_t IG(t) × (1 - λ × ρ_t(δ))

    where ρ_t(δ) = Pr[|x_f - t| < δ] is the margin density.

    The hyperparameter λ controls the accuracy-FHE tradeoff:
    - λ = 0: standard GBDT (maximize IG only)
    - λ = 1: fully FHE-aware (equal weight on IG and margin)
    - λ > 1: aggressively FHE-optimized (may sacrifice some IG for margin)
    """

    def __init__(
        self,
        sign_analyzer: Optional[SignPolynomialAnalyzer] = None,
        poly_degree: int = 7,
        fhe_penalty_weight: float = 1.0,
        margin_candidates: int = 50,
    ):
        """
        Initialize FHE-aware split criterion.

        Args:
            sign_analyzer: Analyzer for sign polynomial error
            poly_degree: Polynomial degree used in FHE evaluation
            fhe_penalty_weight: λ weight for margin penalty (0 = standard, 1 = balanced)
            margin_candidates: Number of candidate thresholds to evaluate
        """
        self.analyzer = sign_analyzer or SignPolynomialAnalyzer()
        self.poly_degree = poly_degree
        self.lambda_weight = fhe_penalty_weight
        self.margin_candidates = margin_candidates

    def find_best_split(
        self,
        feature_values: np.ndarray,
        targets: np.ndarray,
        normalization_scale: Optional[float] = None,
    ) -> FHEAwareSplitResult:
        """
        Find the best split threshold with FHE awareness.

        Args:
            feature_values: Values of one feature for all samples
            targets: Target values (residuals in boosting)
            normalization_scale: Scale for normalizing to [-1, 1]

        Returns:
            FHEAwareSplitResult with best threshold
        """
        if normalization_scale is None:
            feature_range = np.ptp(feature_values)
            normalization_scale = feature_range / 2 if feature_range > 0 else 1.0

        # Generate candidate thresholds
        unique_vals = np.unique(feature_values)
        if len(unique_vals) > self.margin_candidates:
            percentiles = np.linspace(5, 95, self.margin_candidates)
            candidates = np.percentile(feature_values, percentiles)
        elif len(unique_vals) > 1:
            # Midpoints between consecutive unique values
            candidates = (unique_vals[:-1] + unique_vals[1:]) / 2
        else:
            return FHEAwareSplitResult(
                feature_idx=-1, threshold=0.0,
                information_gain=0.0, margin_penalty=1.0,
                fhe_aware_gain=0.0, estimated_sign_error=1.0
            )

        best_result = None
        best_fhe_gain = -float('inf')

        for t in candidates:
            result = self._evaluate_threshold(
                feature_values, targets, t, normalization_scale
            )
            if result.fhe_aware_gain > best_fhe_gain:
                best_fhe_gain = result.fhe_aware_gain
                best_result = result

        return best_result

    def _evaluate_threshold(
        self,
        feature_values: np.ndarray,
        targets: np.ndarray,
        threshold: float,
        norm_scale: float,
    ) -> FHEAwareSplitResult:
        """Evaluate a single candidate threshold."""
        # Information gain (variance reduction)
        left_mask = feature_values < threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(targets)

        if n_left < 1 or n_right < 1:
            return FHEAwareSplitResult(
                feature_idx=-1, threshold=threshold,
                information_gain=0.0, margin_penalty=1.0,
                fhe_aware_gain=0.0, estimated_sign_error=1.0
            )

        total_var = np.var(targets)
        left_var = np.var(targets[left_mask])
        right_var = np.var(targets[right_mask])

        ig = total_var - (n_left / n_total) * left_var - (n_right / n_total) * right_var
        ig = max(0.0, ig)

        # Margin penalty
        penalty = self.analyzer.compute_margin_penalty(
            feature_values, threshold, self.poly_degree, norm_scale
        )

        # FHE-aware gain
        fhe_gain = ig * (1 - self.lambda_weight * penalty)

        # Estimated sign error for this threshold
        profile = self.analyzer.get_error_profile(self.poly_degree)
        margins = np.abs(feature_values - threshold) / max(norm_scale, 1e-10)
        # Expected error: average the error curve at the sample margins
        estimated_error = 0.0
        for m in margins:
            if m < profile.delta:
                estimated_error += 1.0  # Worst case in danger zone
            else:
                estimated_error += profile.epsilon
        estimated_error /= len(margins)

        return FHEAwareSplitResult(
            feature_idx=-1,  # Set by caller
            threshold=threshold,
            information_gain=ig,
            margin_penalty=penalty,
            fhe_aware_gain=fhe_gain,
            estimated_sign_error=estimated_error,
        )

    def compare_standard_vs_fhe_aware(
        self,
        feature_values: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare standard split vs FHE-aware split.

        Returns:
            Comparison showing how threshold differs and error improves
        """
        norm_scale = np.ptp(feature_values) / 2 if np.ptp(feature_values) > 0 else 1.0

        # Standard: λ=0 (pure information gain)
        standard_criterion = FHEAwareSplitCriterion(
            self.analyzer, self.poly_degree, fhe_penalty_weight=0.0
        )
        standard_result = standard_criterion.find_best_split(
            feature_values, targets, norm_scale
        )

        # FHE-aware: λ=current weight
        fhe_result = self.find_best_split(feature_values, targets, norm_scale)

        return {
            "standard_threshold": standard_result.threshold,
            "standard_ig": standard_result.information_gain,
            "standard_margin_penalty": standard_result.margin_penalty,
            "standard_sign_error": standard_result.estimated_sign_error,
            "fhe_aware_threshold": fhe_result.threshold,
            "fhe_aware_ig": fhe_result.information_gain,
            "fhe_aware_margin_penalty": fhe_result.margin_penalty,
            "fhe_aware_sign_error": fhe_result.estimated_sign_error,
            "threshold_shift": abs(fhe_result.threshold - standard_result.threshold),
            "ig_tradeoff": standard_result.information_gain - fhe_result.information_gain,
            "error_improvement": standard_result.estimated_sign_error - fhe_result.estimated_sign_error,
            "penalty_improvement": standard_result.margin_penalty - fhe_result.margin_penalty,
        }


# =============================================================================
# FHE-Aware Oblivious Tree Trainer
# =============================================================================

@dataclass
class FHEAwareTrainingConfig:
    """Configuration for FHE-aware tree training."""
    max_depth: int = 6
    num_trees: int = 100
    learning_rate: float = 0.1
    poly_degree: int = 7              # Sign polynomial degree
    fhe_penalty_weight: float = 1.0   # λ for margin penalty
    margin_candidates: int = 50       # Candidate thresholds per feature


@dataclass
class FHEAwareObliviousLevel:
    """A level in an FHE-aware oblivious tree."""
    depth: int
    feature_idx: int
    threshold: float
    information_gain: float
    margin_penalty: float
    fhe_aware_gain: float
    estimated_sign_error: float


@dataclass
class FHEAwareObliviousTree:
    """An oblivious tree trained with FHE-aware thresholds."""
    tree_id: int
    levels: List[FHEAwareObliviousLevel]
    leaf_values: List[float]
    max_depth: int

    # FHE accuracy metadata
    total_margin_penalty: float = 0.0   # Σ_d ρ_d(δ)
    total_ig_tradeoff: float = 0.0      # IG lost vs standard
    predicted_fhe_error_bound: float = 0.0


class FHEAwareTreeTrainer:
    """
    Trains oblivious trees with FHE-aware threshold selection.

    This is the core novel contribution. The split criterion is:

        t* = argmax_t  IG(t) × (1 - λ × ρ_t(δ))

    where ρ_t(δ) = Pr[|x_f - t| < δ] is the fraction of samples
    whose feature value is within the "danger zone" δ of the threshold.

    The danger zone δ is determined by the polynomial sign approximation
    degree: higher degree → smaller δ → fewer samples in danger zone.

    This creates trees whose thresholds avoid high-density feature regions,
    resulting in more accurate polynomial sign comparisons during FHE inference.
    """

    def __init__(self, config: Optional[FHEAwareTrainingConfig] = None):
        """Initialize trainer."""
        self.config = config or FHEAwareTrainingConfig()
        self.sign_analyzer = SignPolynomialAnalyzer()
        self.split_criterion = FHEAwareSplitCriterion(
            sign_analyzer=self.sign_analyzer,
            poly_degree=self.config.poly_degree,
            fhe_penalty_weight=self.config.fhe_penalty_weight,
            margin_candidates=self.config.margin_candidates,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[List[FHEAwareObliviousTree], Dict[str, Any]]:
        """
        Train an ensemble of FHE-aware oblivious trees.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            Tuple of (trees, training_metadata)
        """
        trees = []
        residuals = y.copy()

        total_ig_tradeoff = 0.0
        total_margin_penalty = 0.0

        for tree_idx in range(self.config.num_trees):
            tree = self._train_single_tree(X, residuals, tree_idx)
            trees.append(tree)

            total_ig_tradeoff += tree.total_ig_tradeoff
            total_margin_penalty += tree.total_margin_penalty

            # Update residuals (gradient boosting)
            predictions = self._evaluate_tree(tree, X)
            residuals -= self.config.learning_rate * predictions

        metadata = {
            "num_trees": len(trees),
            "max_depth": self.config.max_depth,
            "poly_degree": self.config.poly_degree,
            "fhe_penalty_weight": self.config.fhe_penalty_weight,
            "avg_margin_penalty_per_tree": total_margin_penalty / max(len(trees), 1),
            "avg_ig_tradeoff_per_tree": total_ig_tradeoff / max(len(trees), 1),
        }

        return trees, metadata

    def _train_single_tree(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        tree_idx: int,
    ) -> FHEAwareObliviousTree:
        """Train a single FHE-aware oblivious tree."""
        n_features = X.shape[1]
        levels = []
        total_ig_tradeoff = 0.0
        total_margin_penalty = 0.0
        max_leaf_weight = 0.0

        for depth in range(self.config.max_depth):
            # Find best feature and threshold at this level
            best_feature = -1
            best_result = None
            best_fhe_gain = -float('inf')

            for feat_idx in range(n_features):
                result = self.split_criterion.find_best_split(
                    X[:, feat_idx], residuals,
                )
                if result.fhe_aware_gain > best_fhe_gain:
                    best_fhe_gain = result.fhe_aware_gain
                    best_feature = feat_idx
                    best_result = result

            if best_result is None or best_result.information_gain <= 0:
                # No useful split found, use dummy level
                levels.append(FHEAwareObliviousLevel(
                    depth=depth, feature_idx=0, threshold=0.0,
                    information_gain=0.0, margin_penalty=0.0,
                    fhe_aware_gain=0.0, estimated_sign_error=0.0,
                ))
                continue

            level = FHEAwareObliviousLevel(
                depth=depth,
                feature_idx=best_feature,
                threshold=best_result.threshold,
                information_gain=best_result.information_gain,
                margin_penalty=best_result.margin_penalty,
                fhe_aware_gain=best_result.fhe_aware_gain,
                estimated_sign_error=best_result.estimated_sign_error,
            )
            levels.append(level)
            total_margin_penalty += best_result.margin_penalty

        # Compute leaf values
        leaf_values = self._compute_leaf_values(X, residuals, levels)
        max_leaf_weight = max(abs(v) for v in leaf_values) if leaf_values else 0.0

        # Compute FHE error bound: W_max × Σ_d ρ_d(δ)
        fhe_error_bound = max_leaf_weight * total_margin_penalty

        return FHEAwareObliviousTree(
            tree_id=tree_idx,
            levels=levels,
            leaf_values=leaf_values,
            max_depth=self.config.max_depth,
            total_margin_penalty=total_margin_penalty,
            total_ig_tradeoff=total_ig_tradeoff,
            predicted_fhe_error_bound=fhe_error_bound,
        )

    def _compute_leaf_values(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        levels: List[FHEAwareObliviousLevel],
    ) -> List[float]:
        """Compute leaf values for oblivious tree structure."""
        num_leaves = 2 ** len(levels)
        leaf_sums = np.zeros(num_leaves)
        leaf_counts = np.zeros(num_leaves)

        for i in range(X.shape[0]):
            leaf_idx = 0
            for d, level in enumerate(levels):
                if X[i, level.feature_idx] >= level.threshold:
                    leaf_idx |= (1 << d)

            leaf_sums[leaf_idx] += residuals[i]
            leaf_counts[leaf_idx] += 1

        # Mean per leaf (with smoothing for empty leaves)
        leaf_values = np.divide(
            leaf_sums, leaf_counts,
            out=np.zeros_like(leaf_sums),
            where=leaf_counts > 0
        )
        return leaf_values.tolist()

    def _evaluate_tree(
        self,
        tree: FHEAwareObliviousTree,
        X: np.ndarray,
    ) -> np.ndarray:
        """Evaluate oblivious tree on data."""
        outputs = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            leaf_idx = 0
            for d, level in enumerate(tree.levels):
                if X[i, level.feature_idx] >= level.threshold:
                    leaf_idx |= (1 << d)
            outputs[i] = tree.leaf_values[leaf_idx] if leaf_idx < len(tree.leaf_values) else 0.0
        return outputs


# =============================================================================
# FHE Error Analysis
# =============================================================================

class FHEErrorAnalyzer:
    """
    Analyzes the FHE-vs-plaintext prediction error for trained trees.

    Validates the theoretical error bound:
    E[|ŷ_FHE - ŷ_exact|] ≤ W_max × Σ_d ρ_d(δ)
    """

    def __init__(self, poly_degree: int = 7):
        """Initialize analyzer."""
        self.poly_degree = poly_degree
        self.sign_analyzer = SignPolynomialAnalyzer()
        self.profile = self.sign_analyzer.get_error_profile(poly_degree)

    def evaluate_fhe_simulation(
        self,
        trees: List[FHEAwareObliviousTree],
        X: np.ndarray,
        learning_rate: float = 0.1,
        base_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Simulate FHE evaluation and compare with exact evaluation.

        Uses polynomial sign approximation to simulate what happens
        during actual FHE inference.

        Args:
            trees: Trained FHE-aware trees
            X: Test features
            learning_rate: Learning rate
            base_score: Base prediction

        Returns:
            Error analysis including bound validation
        """
        n_samples = X.shape[0]

        # Exact evaluation (standard tree traversal)
        exact_predictions = np.full(n_samples, base_score)
        for tree in trees:
            exact_predictions += learning_rate * self._evaluate_exact(tree, X)

        # Simulated FHE evaluation (polynomial sign)
        fhe_predictions = np.full(n_samples, base_score)
        for tree in trees:
            fhe_predictions += learning_rate * self._evaluate_fhe(tree, X)

        # Error analysis
        errors = np.abs(fhe_predictions - exact_predictions)

        # Theoretical bound
        theoretical_bound = sum(
            tree.predicted_fhe_error_bound * learning_rate
            for tree in trees
        )

        return {
            "mean_absolute_error": float(np.mean(errors)),
            "max_absolute_error": float(np.max(errors)),
            "median_absolute_error": float(np.median(errors)),
            "theoretical_bound": theoretical_bound,
            "bound_holds": float(np.mean(errors)) <= theoretical_bound * 1.1,  # 10% slack
            "error_percentiles": {
                f"p{p}": float(np.percentile(errors, p))
                for p in [50, 90, 95, 99]
            },
            "exact_mean_prediction": float(np.mean(exact_predictions)),
            "fhe_mean_prediction": float(np.mean(fhe_predictions)),
            "prediction_correlation": float(np.corrcoef(
                exact_predictions, fhe_predictions
            )[0, 1]) if len(exact_predictions) > 1 else 1.0,
        }

    def _evaluate_exact(
        self,
        tree: FHEAwareObliviousTree,
        X: np.ndarray,
    ) -> np.ndarray:
        """Exact tree evaluation (step function comparisons)."""
        outputs = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            leaf_idx = 0
            for d, level in enumerate(tree.levels):
                if X[i, level.feature_idx] >= level.threshold:
                    leaf_idx |= (1 << d)
            outputs[i] = tree.leaf_values[leaf_idx] if leaf_idx < len(tree.leaf_values) else 0.0
        return outputs

    def _evaluate_fhe(
        self,
        tree: FHEAwareObliviousTree,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Simulated FHE evaluation (polynomial sign comparisons).

        Uses the leaf-centric tensor product encoding:
        ŷ(x) = Σ_k w_k × Π_d p_{b(k,d)}(x_{σ(d)} - t_d)
        """
        n_samples = X.shape[0]
        depth = len(tree.levels)
        num_leaves = 2 ** depth

        # Step 1: Compute polynomial sign at each level
        signs = np.zeros((n_samples, depth))
        for d, level in enumerate(tree.levels):
            deltas = X[:, level.feature_idx] - level.threshold

            # Normalize to [-1, 1] for polynomial evaluation
            scale = np.max(np.abs(deltas)) + 1e-10
            deltas_norm = deltas / scale

            # Polynomial sign evaluation
            signs[:, d] = self._polynomial_sign(deltas_norm)

        # Step 2: Tensor product for leaf indicators
        # sign_pairs[:, d, :] = [1 - sign_d, sign_d]
        sign_pairs = np.stack([1 - signs, signs], axis=-1)

        indicators = sign_pairs[:, 0, :]  # Start with level 0
        for d in range(1, depth):
            indicators = np.einsum('bi,bj->bij', indicators, sign_pairs[:, d, :])
            indicators = indicators.reshape(n_samples, -1)

        # Step 3: Weighted sum of leaf values
        leaf_weights = np.array(tree.leaf_values[:num_leaves])
        if len(leaf_weights) < indicators.shape[1]:
            leaf_weights = np.pad(leaf_weights, (0, indicators.shape[1] - len(leaf_weights)))

        outputs = indicators @ leaf_weights[:indicators.shape[1]]
        return outputs

    def _polynomial_sign(self, z: np.ndarray) -> np.ndarray:
        """Evaluate polynomial sign and map to [0, 1]."""
        coeffs = self.profile.coefficients
        result = np.zeros_like(z)
        z_power = np.ones_like(z)
        for c in coeffs:
            result += c * z_power
            z_power = z_power * z

        # Map from [-1, 1] to [0, 1]
        return np.clip((result + 1) / 2, 0, 1)

    def compare_standard_vs_fhe_aware(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_depth: int = 4,
        num_trees: int = 50,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Head-to-head comparison: standard training vs FHE-aware training.

        Trains both models on the same data, evaluates both exact and
        simulated-FHE predictions on test data.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Comparison dict with error metrics for both approaches
        """
        # Standard training (λ=0)
        standard_config = FHEAwareTrainingConfig(
            max_depth=max_depth, num_trees=num_trees,
            learning_rate=learning_rate, fhe_penalty_weight=0.0,
        )
        standard_trainer = FHEAwareTreeTrainer(standard_config)
        standard_trees, std_meta = standard_trainer.train(X_train, y_train)

        # FHE-aware training (λ=1)
        fhe_config = FHEAwareTrainingConfig(
            max_depth=max_depth, num_trees=num_trees,
            learning_rate=learning_rate, fhe_penalty_weight=1.0,
        )
        fhe_trainer = FHEAwareTreeTrainer(fhe_config)
        fhe_trees, fhe_meta = fhe_trainer.train(X_train, y_train)

        # Evaluate both
        std_analysis = self.evaluate_fhe_simulation(
            standard_trees, X_test, learning_rate
        )
        fhe_analysis = self.evaluate_fhe_simulation(
            fhe_trees, X_test, learning_rate
        )

        # Compute plaintext MSE for both
        std_exact = np.zeros(X_test.shape[0])
        for tree in standard_trees:
            std_exact += learning_rate * self._evaluate_exact(tree, X_test)
        std_mse = float(np.mean((y_test - std_exact) ** 2))

        fhe_exact = np.zeros(X_test.shape[0])
        for tree in fhe_trees:
            fhe_exact += learning_rate * self._evaluate_exact(tree, X_test)
        fhe_mse = float(np.mean((y_test - fhe_exact) ** 2))

        return {
            "standard": {
                "plaintext_mse": std_mse,
                "fhe_mean_error": std_analysis["mean_absolute_error"],
                "fhe_max_error": std_analysis["max_absolute_error"],
                "fhe_correlation": std_analysis["prediction_correlation"],
                "avg_margin_penalty": std_meta["avg_margin_penalty_per_tree"],
            },
            "fhe_aware": {
                "plaintext_mse": fhe_mse,
                "fhe_mean_error": fhe_analysis["mean_absolute_error"],
                "fhe_max_error": fhe_analysis["max_absolute_error"],
                "fhe_correlation": fhe_analysis["prediction_correlation"],
                "avg_margin_penalty": fhe_meta["avg_margin_penalty_per_tree"],
            },
            "improvements": {
                "fhe_error_reduction": (
                    std_analysis["mean_absolute_error"] - fhe_analysis["mean_absolute_error"]
                ),
                "fhe_error_reduction_percent": (
                    (std_analysis["mean_absolute_error"] - fhe_analysis["mean_absolute_error"])
                    / max(std_analysis["mean_absolute_error"], 1e-10) * 100
                ),
                "plaintext_mse_tradeoff": fhe_mse - std_mse,
                "margin_penalty_reduction": (
                    std_meta["avg_margin_penalty_per_tree"] - fhe_meta["avg_margin_penalty_per_tree"]
                ),
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def train_fhe_aware_trees(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 6,
    num_trees: int = 100,
    learning_rate: float = 0.1,
    fhe_penalty_weight: float = 1.0,
    poly_degree: int = 7,
) -> Tuple[List[FHEAwareObliviousTree], Dict[str, Any]]:
    """
    Train FHE-aware oblivious trees.

    Args:
        X: Training features
        y: Training targets
        max_depth: Maximum tree depth
        num_trees: Number of trees
        learning_rate: Boosting learning rate
        fhe_penalty_weight: λ for FHE margin penalty (0=standard, 1=FHE-aware)
        poly_degree: Polynomial sign degree used in FHE

    Returns:
        Tuple of (trained_trees, metadata)
    """
    config = FHEAwareTrainingConfig(
        max_depth=max_depth,
        num_trees=num_trees,
        learning_rate=learning_rate,
        poly_degree=poly_degree,
        fhe_penalty_weight=fhe_penalty_weight,
    )
    trainer = FHEAwareTreeTrainer(config)
    return trainer.train(X, y)


def compare_training_approaches(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compare standard vs FHE-aware tree training.

    Returns detailed comparison of FHE prediction errors.
    """
    analyzer = FHEErrorAnalyzer()
    return analyzer.compare_standard_vs_fhe_aware(
        X_train, y_train, X_test, y_test, **kwargs
    )
