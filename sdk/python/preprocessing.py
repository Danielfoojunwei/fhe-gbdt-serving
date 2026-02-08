"""
FHE Preprocessing Pipeline

Client-side preprocessing that runs BEFORE encryption.
These transforms must happen in plaintext on the client side,
then the transformed features are encrypted and sent for inference.

Supports:
- WoE (Weight of Evidence) binning for credit scorecards
- Standardization (z-score normalization)
- Clipping (domain enforcement for polynomial approximation bounds)
- Scorecard points conversion

Usage:
    preprocessor = FHEPreprocessor.from_plan(plan_metadata)
    transformed = preprocessor.transform(raw_features)
    encrypted = key_manager.encrypt(transformed)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WoETransformer:
    """
    Weight of Evidence transformer for credit scorecard features.

    WoE = ln(Distribution of Good / Distribution of Bad)

    In regulatory credit scoring, WoE binning is the standard method
    to convert continuous features into monotonic, interpretable values
    before fitting a logistic regression.
    """

    def __init__(self, binning_table: Dict[str, List[Dict]]):
        """
        Args:
            binning_table: Dict mapping feature_name -> list of bin definitions.
                Each bin: {"lo": float, "hi": float, "woe": float, "iv": float}
        """
        self.binning_table = binning_table

    def transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Transform features using WoE binning."""
        result = features.copy()
        for col_idx, name in enumerate(feature_names):
            if name in self.binning_table:
                bins = self.binning_table[name]
                for i in range(features.shape[0]):
                    result[i, col_idx] = self._lookup_woe(features[i, col_idx], bins)
        return result

    def _lookup_woe(self, value: float, bins: List[Dict]) -> float:
        """Find the WoE value for a given feature value."""
        for bin_def in bins:
            lo = bin_def.get("lo", float("-inf"))
            hi = bin_def.get("hi", float("inf"))
            if lo <= value < hi:
                return bin_def["woe"]
        # Fallback: return 0 (neutral WoE)
        return 0.0

    @property
    def information_values(self) -> Dict[str, float]:
        """Get Information Value (IV) per feature for variable selection."""
        ivs = {}
        for name, bins in self.binning_table.items():
            ivs[name] = sum(b.get("iv", 0.0) for b in bins)
        return ivs


class ScorecardPointsConverter:
    """
    Convert logistic regression output to scorecard points.

    Standard conversion:
        Score = Offset - Factor * ln(odds)
        where odds = p / (1-p) from logistic regression

    Parameters follow industry convention:
        - base_score: Score at base odds (e.g., 600)
        - base_odds: The odds at base_score (e.g., 50:1 = 50.0)
        - pdo: Points to Double Odds (e.g., 20)
    """

    def __init__(
        self,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        pdo: float = 20.0,
    ):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo

        # Derived constants
        self.factor = pdo / math.log(2)
        self.offset = base_score - self.factor * math.log(base_odds)

    def probability_to_score(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert predicted probabilities to scorecard points."""
        # Clip to avoid log(0) or division by zero
        p = np.clip(probabilities, 1e-10, 1 - 1e-10)
        odds = p / (1 - p)
        scores = self.offset - self.factor * np.log(odds)
        return np.round(scores).astype(int)

    def score_to_probability(self, scores: np.ndarray) -> np.ndarray:
        """Convert scorecard points back to probabilities."""
        log_odds = (self.offset - scores) / self.factor
        odds = np.exp(log_odds)
        return odds / (1 + odds)


class FHEPreprocessor:
    """
    Unified preprocessing pipeline applied before FHE encryption.

    Chains together multiple preprocessing steps in order:
    1. WoE binning (if configured)
    2. Standardization (if configured)
    3. Clipping (domain enforcement for link function polynomial bounds)
    """

    def __init__(self, steps: Optional[List[Dict]] = None):
        """
        Args:
            steps: List of preprocessing step configs
        """
        self.steps = steps or []
        self._woe_transformer = None
        self._standardize_params = None
        self._clip_bounds = None

        for step in self.steps:
            step_type = step.get("type") or step.get("step_type")
            params = step.get("params", {})

            if step_type == "woe_binning":
                self._woe_transformer = WoETransformer(params.get("binning_table", {}))
            elif step_type == "standardize":
                self._standardize_params = {
                    "mean": np.array(params.get("mean", [])),
                    "std": np.array(params.get("std", [])),
                }
            elif step_type == "clip":
                self._clip_bounds = (
                    params.get("min", -8.0),
                    params.get("max", 8.0),
                )

    def transform(
        self, features: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply all preprocessing steps in order.

        Args:
            features: Raw features, shape (batch_size, num_features)
            feature_names: Optional feature names (needed for WoE)

        Returns:
            Transformed features ready for encryption
        """
        result = np.array(features, dtype=np.float64, copy=True)

        # 1. WoE binning
        if self._woe_transformer and feature_names:
            result = self._woe_transformer.transform(result, feature_names)

        # 2. Standardization
        if self._standardize_params:
            mean = self._standardize_params["mean"]
            std = self._standardize_params["std"]
            if len(mean) > 0 and len(std) > 0:
                std_safe = np.where(std == 0, 1.0, std)
                result = (result - mean) / std_safe

        # 3. Clipping (enforce polynomial approximation domain)
        if self._clip_bounds:
            lo, hi = self._clip_bounds
            result = np.clip(result, lo, hi)

        return result

    @classmethod
    def from_plan(cls, plan_metadata: Dict) -> "FHEPreprocessor":
        """
        Create preprocessor from a compiled plan's metadata.

        The compiler embeds preprocessing configuration in the plan metadata
        so the client SDK can reconstruct the preprocessing pipeline.
        """
        steps = plan_metadata.get("preprocessing", [])

        # Auto-add clipping based on link function domain
        link_domain = plan_metadata.get("link_domain")
        if link_domain and not any(
            (s.get("type") or s.get("step_type")) == "clip" for s in steps
        ):
            steps.append({
                "type": "clip",
                "params": {"min": link_domain[0], "max": link_domain[1]},
            })

        return cls(steps=steps)

    @classmethod
    def for_credit_scorecard(
        cls,
        binning_table: Dict[str, List[Dict]],
        clip_domain: Tuple[float, float] = (-8.0, 8.0),
    ) -> "FHEPreprocessor":
        """
        Create preprocessor configured for credit scorecard models.

        Args:
            binning_table: WoE binning table per feature
            clip_domain: Clipping domain for sigmoid polynomial approximation
        """
        steps = [
            {"type": "woe_binning", "params": {"binning_table": binning_table}},
            {"type": "clip", "params": {"min": clip_domain[0], "max": clip_domain[1]}},
        ]
        return cls(steps=steps)
