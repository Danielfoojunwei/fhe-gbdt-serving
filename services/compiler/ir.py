from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from enum import Enum
import json


class ModelFamily(str, Enum):
    """Model family classification for optimizer routing."""
    TREE_ENSEMBLE = "tree_ensemble"      # XGBoost, LightGBM, CatBoost
    LINEAR = "linear"                     # Logistic Regression, Linear/GLM
    RANDOM_FOREST = "random_forest"       # Random Forest (non-boosted)
    SINGLE_TREE = "single_tree"           # Single Decision Tree


class LinkFunction(str, Enum):
    """GLM link functions for FHE polynomial approximation."""
    IDENTITY = "identity"       # y = x (linear regression)
    LOGIT = "logit"             # y = 1/(1+exp(-x)) (logistic)
    LOG = "log"                 # y = exp(x) (Poisson/Gamma)
    RECIPROCAL = "reciprocal"   # y = 1/x (Gamma alt)
    PROBIT = "probit"           # y = Phi(x) (probit regression)


class Aggregation(str, Enum):
    """How to combine tree outputs in ensembles."""
    SUM = "sum"                 # GBDT: sum all tree outputs
    MEAN = "mean"               # Random Forest: average outputs
    NONE = "none"               # Single tree: no aggregation


@dataclass
class TreeNode:
    node_id: int
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left_child_id: Optional[int] = None
    right_child_id: Optional[int] = None
    leaf_value: Optional[float] = None
    default_left: bool = True
    depth: int = 0


@dataclass
class TreeIR:
    tree_id: int
    nodes: Dict[int, TreeNode]
    root_id: int
    max_depth: int


@dataclass
class LinearCoefficients:
    """Coefficients for linear/GLM models."""
    weights: List[float]          # Per-feature coefficients
    intercept: float = 0.0        # Bias/intercept term

    @property
    def num_features(self) -> int:
        return len(self.weights)


@dataclass
class PreprocessingStep:
    """A preprocessing step applied before inference."""
    step_type: str              # "woe_binning", "standardize", "clip", "one_hot"
    params: Dict[str, Union[float, str, List, Dict]] = field(default_factory=dict)
    feature_indices: Optional[List[int]] = None


@dataclass
class ModelIR:
    model_type: str  # xgboost, lightgbm, catboost, logistic_regression, linear_glm, random_forest, decision_tree
    trees: List[TreeIR]
    num_features: int
    base_score: float = 0.5

    # --- Linear model fields (multi-model extension) ---
    model_family: ModelFamily = ModelFamily.TREE_ENSEMBLE
    coefficients: Optional[LinearCoefficients] = None
    link_function: LinkFunction = LinkFunction.IDENTITY
    glm_family: Optional[str] = None  # "gaussian", "binomial", "poisson", "gamma", "tweedie"
    aggregation: Aggregation = Aggregation.SUM

    # --- Metadata ---
    feature_names: Optional[List[str]] = None
    preprocessing: List[PreprocessingStep] = field(default_factory=list)
    class_labels: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)

# ObliviousPlanIR structures

@dataclass
class OpSequence:
    op_type: str # DELTA, STEP, ROUTE
    params: Dict[str, Union[int, float, str, List]]

@dataclass
class ScheduleBlock:
    depth_level: int
    node_group_id: int
    ops: List[OpSequence]

@dataclass
class PackingLayout:
    layout_type: str # feature_major
    feature_to_ciphertext: Dict[int, int]
    slots: int

@dataclass
class LinearPlanOp:
    """Operation in a linear model execution plan."""
    op_type: str  # "DOT_PRODUCT", "ADD_BIAS", "LINK_FUNCTION"
    params: Dict[str, Union[int, float, str, List]] = field(default_factory=dict)


@dataclass
class LinearPlanIR:
    """Execution plan for linear/GLM models under FHE."""
    compiled_model_id: str
    crypto_params_id: str
    packing_layout: PackingLayout
    coefficients: List[float]
    intercept: float
    link_function: str            # "identity", "logit", "log", "reciprocal"
    poly_coeffs: List[float]      # Polynomial approximation of link function
    poly_degree: int
    num_features: int
    ops: List[LinearPlanOp] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__ if not isinstance(o, Enum) else o.value, indent=2)


@dataclass
class ObliviousPlanIR:
    compiled_model_id: str
    crypto_params_id: str
    packing_layout: PackingLayout
    schedule: List[ScheduleBlock]
    base_score: float
    num_trees: int
    metadata: Dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__ if not isinstance(o, Enum) else o.value, indent=2)


# Union type for all plan types
ExecutionPlanIR = Union[ObliviousPlanIR, LinearPlanIR]
