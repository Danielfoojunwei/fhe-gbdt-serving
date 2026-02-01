"""
Input Validation Module for FHE-GBDT Compiler

This module provides comprehensive input validation for model files
before they are processed for FHE compilation. It implements security
checks based on OWASP recommendations and FHE-specific constraints.

Security features:
- Input size limits
- JSON structure validation
- FHE compatibility checks
- Resource consumption limits

References:
- OWASP Input Validation Cheat Sheet
- Cloud Security Alliance FHE Guidelines
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    severity: ValidationSeverity
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of model validation."""
    is_valid: bool
    issues: List[ValidationIssue]

    @property
    def has_errors(self) -> bool:
        return any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                   for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            errors = [i for i in self.issues if i.severity in
                      (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]
            messages = [f"{i.code}: {i.message}" for i in errors]
            raise ValueError("Model validation failed:\n" + "\n".join(messages))


class ModelValidator:
    """
    Validates GBDT models for FHE compatibility and security.

    This validator checks:
    1. Input size limits (DoS prevention)
    2. JSON structure integrity
    3. FHE compatibility constraints
    4. Resource usage estimates
    """

    # Security limits
    MAX_CONTENT_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
    MAX_JSON_DEPTH = 50
    MAX_TREES = 500
    MAX_TREE_DEPTH = 20
    MAX_NODES_PER_TREE = 10000
    MAX_FEATURES = 10000
    MAX_LEAVES_PER_TREE = 1 << 20  # 1M leaves

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def _add_issue(self, code: str, message: str,
                   severity: ValidationSeverity,
                   location: Optional[str] = None) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(code, message, severity, location))

    def validate(self, content: bytes, model_type: str) -> ValidationResult:
        """
        Validate model content.

        Args:
            content: Raw model file content
            model_type: Type of model (xgboost, lightgbm, catboost)

        Returns:
            ValidationResult with all issues found
        """
        self.issues = []

        # Phase 1: Size validation
        self._validate_size(content)
        if self._has_critical_issues():
            return self._build_result()

        # Phase 2: JSON parsing
        data = self._validate_json(content)
        if data is None or self._has_critical_issues():
            return self._build_result()

        # Phase 3: Structure validation
        self._validate_structure(data, model_type)
        if self._has_critical_issues():
            return self._build_result()

        # Phase 4: FHE compatibility
        self._validate_fhe_compatibility(data, model_type)

        # Phase 5: Resource estimation
        self._estimate_resources(data, model_type)

        return self._build_result()

    def _has_critical_issues(self) -> bool:
        """Check if any critical issues exist."""
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    def _build_result(self) -> ValidationResult:
        """Build validation result."""
        is_valid = not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                          for i in self.issues)
        return ValidationResult(is_valid=is_valid, issues=self.issues)

    def _validate_size(self, content: bytes) -> None:
        """Validate content size."""
        if not content:
            self._add_issue(
                "SIZE_001", "Empty content",
                ValidationSeverity.CRITICAL
            )
            return

        if len(content) > self.MAX_CONTENT_SIZE_BYTES:
            self._add_issue(
                "SIZE_002",
                f"Content size ({len(content)} bytes) exceeds limit "
                f"({self.MAX_CONTENT_SIZE_BYTES} bytes)",
                ValidationSeverity.CRITICAL
            )

    def _validate_json(self, content: bytes) -> Optional[Dict]:
        """Validate JSON structure."""
        try:
            # Decode with strict error handling
            text = content.decode('utf-8', errors='strict')

            # Check for null bytes (potential injection)
            if '\x00' in text:
                self._add_issue(
                    "JSON_001", "Content contains null bytes",
                    ValidationSeverity.CRITICAL
                )
                return None

            # Parse JSON with depth checking
            data = json.loads(text)

            # Verify it's a dictionary
            if not isinstance(data, dict):
                self._add_issue(
                    "JSON_002", "Root element must be an object",
                    ValidationSeverity.CRITICAL
                )
                return None

            # Check JSON depth
            depth = self._get_json_depth(data)
            if depth > self.MAX_JSON_DEPTH:
                self._add_issue(
                    "JSON_003",
                    f"JSON depth ({depth}) exceeds limit ({self.MAX_JSON_DEPTH})",
                    ValidationSeverity.ERROR
                )

            return data

        except UnicodeDecodeError as e:
            self._add_issue(
                "JSON_004", f"Invalid UTF-8 encoding: {e}",
                ValidationSeverity.CRITICAL
            )
            return None
        except json.JSONDecodeError as e:
            self._add_issue(
                "JSON_005", f"Invalid JSON: {e.msg} at line {e.lineno}",
                ValidationSeverity.CRITICAL
            )
            return None

    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum depth of JSON structure."""
        if current_depth > self.MAX_JSON_DEPTH + 1:
            return current_depth  # Stop early for very deep structures

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj)
        else:
            return current_depth

    def _validate_structure(self, data: Dict, model_type: str) -> None:
        """Validate model structure based on type."""
        if model_type == "xgboost":
            self._validate_xgboost_structure(data)
        elif model_type == "lightgbm":
            self._validate_lightgbm_structure(data)
        elif model_type == "catboost":
            self._validate_catboost_structure(data)
        else:
            self._add_issue(
                "STRUCT_001", f"Unknown model type: {model_type}",
                ValidationSeverity.ERROR
            )

    def _validate_xgboost_structure(self, data: Dict) -> None:
        """Validate XGBoost model structure."""
        if 'learner' not in data:
            self._add_issue(
                "XGB_001", "Missing 'learner' key in XGBoost model",
                ValidationSeverity.CRITICAL
            )
            return

        learner = data['learner']
        if 'gradient_booster' not in learner:
            self._add_issue(
                "XGB_002", "Missing 'gradient_booster' in learner",
                ValidationSeverity.CRITICAL
            )
            return

        booster = learner['gradient_booster']
        if 'model' not in booster:
            self._add_issue(
                "XGB_003", "Missing 'model' in gradient_booster",
                ValidationSeverity.CRITICAL
            )
            return

        model = booster['model']
        if 'trees' not in model:
            self._add_issue(
                "XGB_004", "Missing 'trees' in model",
                ValidationSeverity.CRITICAL
            )
            return

        trees = model['trees']
        if not isinstance(trees, list):
            self._add_issue(
                "XGB_005", "'trees' must be a list",
                ValidationSeverity.CRITICAL
            )
            return

        # Validate tree count
        if len(trees) > self.MAX_TREES:
            self._add_issue(
                "XGB_006",
                f"Too many trees ({len(trees)} > {self.MAX_TREES})",
                ValidationSeverity.ERROR
            )

        # Validate each tree
        for i, tree in enumerate(trees):
            self._validate_xgboost_tree(tree, f"trees[{i}]")

    def _validate_xgboost_tree(self, tree: Dict, location: str) -> None:
        """Validate a single XGBoost tree."""
        required_keys = ['left_children', 'right_children', 'split_indices', 'split_conditions']
        for key in required_keys:
            if key not in tree:
                self._add_issue(
                    "XGB_TREE_001", f"Missing '{key}' in {location}",
                    ValidationSeverity.ERROR, location
                )
                return

        # Validate array lengths match
        lengths = {k: len(tree[k]) for k in required_keys}
        if len(set(lengths.values())) != 1:
            self._add_issue(
                "XGB_TREE_002",
                f"Inconsistent array lengths in {location}: {lengths}",
                ValidationSeverity.ERROR, location
            )

        # Validate node count
        num_nodes = len(tree['left_children'])
        if num_nodes > self.MAX_NODES_PER_TREE:
            self._add_issue(
                "XGB_TREE_003",
                f"Too many nodes in {location}: {num_nodes} > {self.MAX_NODES_PER_TREE}",
                ValidationSeverity.ERROR, location
            )

    def _validate_lightgbm_structure(self, data: Dict) -> None:
        """Validate LightGBM model structure."""
        if 'tree_info' not in data:
            self._add_issue(
                "LGB_001", "Missing 'tree_info' key in LightGBM model",
                ValidationSeverity.CRITICAL
            )
            return

        trees = data['tree_info']
        if not isinstance(trees, list):
            self._add_issue(
                "LGB_002", "'tree_info' must be a list",
                ValidationSeverity.CRITICAL
            )
            return

        if len(trees) > self.MAX_TREES:
            self._add_issue(
                "LGB_003",
                f"Too many trees ({len(trees)} > {self.MAX_TREES})",
                ValidationSeverity.ERROR
            )

    def _validate_catboost_structure(self, data: Dict) -> None:
        """Validate CatBoost model structure."""
        if 'oblivious_trees' not in data:
            self._add_issue(
                "CAT_001", "Missing 'oblivious_trees' key in CatBoost model",
                ValidationSeverity.CRITICAL
            )
            return

        trees = data['oblivious_trees']
        if not isinstance(trees, list):
            self._add_issue(
                "CAT_002", "'oblivious_trees' must be a list",
                ValidationSeverity.CRITICAL
            )
            return

        if len(trees) > self.MAX_TREES:
            self._add_issue(
                "CAT_003",
                f"Too many trees ({len(trees)} > {self.MAX_TREES})",
                ValidationSeverity.ERROR
            )

    def _validate_fhe_compatibility(self, data: Dict, model_type: str) -> None:
        """Validate FHE compatibility constraints."""
        if model_type == "xgboost":
            self._validate_xgboost_fhe_compat(data)
        elif model_type == "lightgbm":
            self._validate_lightgbm_fhe_compat(data)
        elif model_type == "catboost":
            self._validate_catboost_fhe_compat(data)

    def _validate_xgboost_fhe_compat(self, data: Dict) -> None:
        """Validate XGBoost FHE compatibility."""
        try:
            trees = data['learner']['gradient_booster']['model']['trees']

            max_depth = 0
            max_features = 0

            for tree in trees:
                # Estimate tree depth
                left_children = tree['left_children']
                depth = self._estimate_tree_depth(left_children, 0)
                max_depth = max(max_depth, depth)

                # Track max feature index
                for idx in tree['split_indices']:
                    if idx >= 0:
                        max_features = max(max_features, idx)

            if max_depth > self.MAX_TREE_DEPTH:
                self._add_issue(
                    "FHE_001",
                    f"Tree depth ({max_depth}) exceeds FHE limit ({self.MAX_TREE_DEPTH}). "
                    "Consider reducing max_depth during training.",
                    ValidationSeverity.WARNING
                )

            if max_features > self.MAX_FEATURES:
                self._add_issue(
                    "FHE_002",
                    f"Feature count ({max_features}) exceeds limit ({self.MAX_FEATURES})",
                    ValidationSeverity.ERROR
                )

        except (KeyError, TypeError) as e:
            logger.warning(f"Could not validate FHE compatibility: {e}")

    def _validate_lightgbm_fhe_compat(self, data: Dict) -> None:
        """Validate LightGBM FHE compatibility."""
        try:
            num_leaves = data.get('max_leaf_output', 0)
            if num_leaves > self.MAX_LEAVES_PER_TREE:
                self._add_issue(
                    "FHE_003",
                    f"Number of leaves ({num_leaves}) may cause slow FHE inference",
                    ValidationSeverity.WARNING
                )
        except (KeyError, TypeError):
            pass

    def _validate_catboost_fhe_compat(self, data: Dict) -> None:
        """Validate CatBoost FHE compatibility."""
        try:
            trees = data['oblivious_trees']
            for i, tree in enumerate(trees):
                depth = len(tree.get('splits', []))
                if depth > self.MAX_TREE_DEPTH:
                    self._add_issue(
                        "FHE_004",
                        f"Tree {i} depth ({depth}) exceeds FHE limit ({self.MAX_TREE_DEPTH})",
                        ValidationSeverity.WARNING
                    )
        except (KeyError, TypeError):
            pass

    def _estimate_tree_depth(self, left_children: List[int], node: int = 0, depth: int = 0) -> int:
        """Estimate tree depth from left_children array."""
        if depth > self.MAX_TREE_DEPTH + 5:
            return depth  # Prevent infinite recursion

        if node < 0 or node >= len(left_children):
            return depth

        left_idx = left_children[node]
        if left_idx < 0:  # Leaf node
            return depth

        return self._estimate_tree_depth(left_children, left_idx, depth + 1)

    def _estimate_resources(self, data: Dict, model_type: str) -> None:
        """Estimate resource requirements."""
        try:
            if model_type == "xgboost":
                num_trees = len(data['learner']['gradient_booster']['model']['trees'])
            elif model_type == "lightgbm":
                num_trees = len(data['tree_info'])
            elif model_type == "catboost":
                num_trees = len(data['oblivious_trees'])
            else:
                return

            # Rough estimate: 50ms per tree in FHE
            estimated_latency_ms = num_trees * 50

            if estimated_latency_ms > 10000:  # 10 seconds
                self._add_issue(
                    "PERF_001",
                    f"Estimated inference latency ({estimated_latency_ms}ms) is very high. "
                    "Consider reducing the number of trees.",
                    ValidationSeverity.WARNING
                )

        except (KeyError, TypeError) as e:
            logger.debug(f"Could not estimate resources: {e}")


def validate_model(content: bytes, model_type: str) -> ValidationResult:
    """
    Validate a model for FHE compilation.

    Args:
        content: Raw model file content
        model_type: Type of model (xgboost, lightgbm, catboost)

    Returns:
        ValidationResult with all issues found
    """
    validator = ModelValidator()
    return validator.validate(content, model_type)
