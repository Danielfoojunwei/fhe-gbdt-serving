# Multi-Model FHE Serving Platform: Phased Implementation Plan

## Architecture Overview

### Current Pipeline
```
Content (bytes) → Parser → ModelIR → MOAIOptimizer → ObliviousPlanIR → Runtime
```

### Target Pipeline (Unified)
```
Content (bytes)
    ↓
ModelParser (per-library)
    ↓
UnifiedModelIR
    ├── TreeModelIR      (GBDT, RF, Single Tree)
    ├── LinearModelIR    (Logistic Regression, GLM)
    └── metadata (model_type, link_function, aggregation, etc.)
    ↓
UnifiedOptimizer
    ├── TreeOptimizer     (existing MOAI, extended)
    └── LinearOptimizer   (new: dot product + link function)
    ↓
ExecutionPlanIR
    ├── TreeSchedule      (comparisons, leaf selection)
    ├── LinearSchedule    (coefficient multiply-accumulate)
    ├── LinkFunctionSchedule (polynomial approximation)
    └── AggregationSchedule (sum, mean, vote)
    ↓
Runtime (N2HE-HEXL)
```

### Pre-Encryption vs Post-Encryption Boundary

```
CLIENT (plaintext)                    SERVER (encrypted via FHE)
──────────────────────────────────    ──────────────────────────────
1. Load raw data
2. Missing value imputation
3. WoE binning (scorecards)
4. One-hot / label encoding
5. Standardization / normalization
6. Exposure offset (insurance GLM)
7. Quantize to target precision
8. Encrypt features ──────────────>   9. Receive ciphertext
                                      10. Execute plan:
                                          - Dot product (linear)
                                          - Tree comparisons
                                          - Link function (polynomial)
                                          - Aggregation (sum/mean)
                                      11. Return encrypted result
12. Decrypt <─────────────────────
13. Inverse transform (if needed)
14. Extract reason codes (trees)
```

---

## Phase 1: Extend IR and Parser Infrastructure
**Goal:** ModelIR supports both tree and linear models. Parsers load scikit-learn models.

### 1.1 Extend `ModelIR` (backward-compatible)

**File:** `services/compiler/ir.py`

Add optional fields to `ModelIR`:
```python
@dataclass
class ModelIR:
    model_type: str  # "xgboost", "lightgbm", "catboost",
                     # "logistic_regression", "linear_regression",
                     # "glm", "random_forest", "decision_tree"
    trees: List[TreeIR] = field(default_factory=list)
    num_features: int = 0
    base_score: float = 0.5

    # Linear model fields (Phase 1)
    coefficients: Optional[List[float]] = None  # Feature coefficients
    intercept: Optional[float] = None           # Bias term
    link_function: Optional[str] = None         # "identity", "logit", "log", "reciprocal"
    glm_family: Optional[str] = None            # "gaussian", "binomial", "poisson", "gamma", "tweedie"
    tweedie_power: Optional[float] = None       # For Tweedie: 1 < p < 2

    # Aggregation (Phase 1)
    aggregation: str = "sum"  # "sum" (GBDT), "mean" (RF), "none" (single tree)

    # Preprocessing metadata (Phase 1)
    feature_names: Optional[List[str]] = None
    preprocessing: Optional[Dict] = None  # Serialized preprocessing params
```

Add new OpSequence types:
```python
# Existing: "DELTA", "STEP", "ROUTE", "COMPARE_BATCH", "ROTATE"
# New:
#   "LINEAR_EVAL"      - dot product of coefficients with features
#   "LINK_FUNCTION"    - polynomial approximation of link function
#   "MEAN_AGGREGATE"   - average tree outputs (RF)
#   "REASON_CODES"     - per-feature contribution extraction (single tree)
```

### 1.2 Scikit-Learn Model Parsers

**New files:**
- `services/compiler/sklearn_parser.py`

Supports loading models from scikit-learn's JSON export format (or joblib/pickle
with a JSON metadata sidecar). Four model types, one parser file:

```python
class ScikitLearnParser(BaseParser):
    """Parses scikit-learn models exported via JSON."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)
        model_class = data["model_class"]

        if model_class == "LogisticRegression":
            return self._parse_logistic_regression(data)
        elif model_class == "LinearRegression":
            return self._parse_linear_regression(data)
        elif model_class == "RandomForestClassifier":
            return self._parse_random_forest(data, task="classification")
        elif model_class == "RandomForestRegressor":
            return self._parse_random_forest(data, task="regression")
        elif model_class == "DecisionTreeClassifier":
            return self._parse_decision_tree(data, task="classification")
        elif model_class == "DecisionTreeRegressor":
            return self._parse_decision_tree(data, task="regression")
        else:
            raise ValueError(f"Unsupported model class: {model_class}")
```

**Logistic regression parse:** Extract `coef_`, `intercept_`, `classes_`. Set
`link_function="logit"`, `glm_family="binomial"`.

**Linear regression parse:** Extract `coef_`, `intercept_`. Set
`link_function="identity"`, `glm_family="gaussian"`.

**Random forest parse:** Extract `estimators_` (list of decision trees). Build
`TreeIR` for each tree. Set `aggregation="mean"`.

**Decision tree parse:** Extract single tree structure (node arrays:
`children_left`, `children_right`, `feature`, `threshold`, `value`). Build
single `TreeIR`. Set `aggregation="none"`.

### 1.3 Statsmodels GLM Parser

**New file:** `services/compiler/glm_parser.py`

Supports statsmodels GLM export format:
```python
class StatsmodelsGLMParser(BaseParser):
    """Parses statsmodels GLM results exported as JSON."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)
        family = data["family"]  # "Gaussian", "Binomial", "Poisson", "Gamma", "Tweedie"
        link = data["link"]      # "identity", "logit", "log", "inverse"
        params = data["params"]  # Coefficient array
        # ...
```

### 1.4 Register New Parsers

**File:** `services/compiler/parser.py` -- update `get_parser()`:
```python
PARSERS = {
    "xgboost": XGBoostParser,
    "lightgbm": LightGBMParser,
    "catboost": CatBoostParser,
    "scikit-learn": ScikitLearnParser,     # NEW
    "sklearn": ScikitLearnParser,          # alias
    "statsmodels": StatsmodelsGLMParser,   # NEW
}
```

### 1.5 Model Export Utilities (SDK-side)

**New file:** `sdk/python/model_export.py`

Helpers for customers to export their trained models to platform-compatible JSON:
```python
def export_logistic_regression(model, feature_names=None) -> bytes:
    """Export a scikit-learn LogisticRegression to platform JSON."""

def export_glm(model, feature_names=None) -> bytes:
    """Export a statsmodels GLM result to platform JSON."""

def export_random_forest(model, feature_names=None) -> bytes:
    """Export a scikit-learn RandomForest to platform JSON."""

def export_decision_tree(model, feature_names=None) -> bytes:
    """Export a scikit-learn DecisionTree to platform JSON."""
```

### Phase 1 Tests
- Parse scikit-learn LogisticRegression JSON → verify coefficients match
- Parse scikit-learn RandomForest JSON → verify tree count and structure
- Parse scikit-learn DecisionTree JSON → verify single tree with correct depth
- Parse statsmodels GLM JSON → verify family, link, coefficients
- Backward compatibility: existing XGBoost/LightGBM/CatBoost parsing still works
- ModelIR serialization roundtrip with new fields

---

## Phase 2: Linear Model Optimizer + Link Function Approximations
**Goal:** Compile logistic regression and GLMs into FHE execution plans.

### 2.1 Link Function Polynomial Approximations

**New file:** `services/compiler/link_functions.py`

Provides pre-computed polynomial approximations for standard GLM link functions
using minimax (Remez) or Chebyshev polynomials.

```python
@dataclass
class LinkFunctionApprox:
    name: str              # "sigmoid", "exp", "reciprocal", "identity"
    coefficients: List[float]  # Polynomial coefficients [c0, c1, ..., cn]
    degree: int
    domain: Tuple[float, float]  # Valid input range (e.g., [-8, 8])
    max_error: float       # Worst-case approximation error in domain
    method: str            # "chebyshev", "minimax", "taylor"

LINK_FUNCTIONS = {
    "identity": LinkFunctionApprox(
        name="identity",
        coefficients=[0.0, 1.0],  # f(x) = x
        degree=1, domain=(-inf, inf), max_error=0.0, method="exact"
    ),
    "logit": LinkFunctionApprox(
        name="sigmoid",
        # Degree-7 minimax approximation of 1/(1+exp(-x)) on [-8, 8]
        coefficients=[0.5, 0.197, 0.0, -0.004, ...],  # Pre-computed
        degree=7, domain=(-8.0, 8.0), max_error=0.002, method="minimax"
    ),
    "log": LinkFunctionApprox(
        name="exp",
        # Inverse of log link: exp(x) approximated on [-4, 4]
        coefficients=[1.0, 1.0, 0.5, 0.1667, 0.0417, ...],
        degree=7, domain=(-4.0, 4.0), max_error=0.003, method="chebyshev"
    ),
    "reciprocal": LinkFunctionApprox(
        name="reciprocal_inv",
        # Inverse of reciprocal link: 1/x approximated
        coefficients=[...],
        degree=5, domain=(0.1, 10.0), max_error=0.01, method="minimax"
    ),
}
```

**Noise budget analysis per link function:**

| Link Function | FHE Depth | Multiplicative Operations | Noise Cost (bits) |
|---------------|-----------|---------------------------|-------------------|
| Identity      | 0         | 0                         | 0                 |
| Logit (deg 7) | 3         | 7 multiplications         | ~70               |
| Log (deg 7)   | 3         | 7 multiplications         | ~70               |
| Reciprocal (5)| 3         | 5 multiplications         | ~50               |

### 2.2 Linear Model Optimizer

**New file:** `services/compiler/linear_optimizer.py`

```python
class LinearModelOptimizer:
    """
    Optimizer for linear models (logistic regression, GLM).

    Execution plan structure:
    1. LINEAR_EVAL: dot product of encrypted features with plaintext coefficients
       - coefficients are PLAINTEXT (model weights are not secret)
       - features are CIPHERTEXT
       - Result: encrypted linear predictor eta = X @ beta + intercept
    2. LINK_FUNCTION: polynomial approximation applied to eta
       - Evaluated using Horner's method (depth = ceil(log2(degree)))
       - Result: encrypted prediction mu = g_inv(eta)
    """

    def optimize(self, model: ModelIR, profile: str) -> ObliviousPlanIR:
        # 1. Validate linear model
        assert model.coefficients is not None

        # 2. Create packing layout (features packed into slots)
        layout = self._create_linear_packing(model)

        # 3. Build schedule
        schedule = []

        # Phase 1: Dot product
        schedule.append(ScheduleBlock(
            depth_level=0,
            node_group_id=0,
            ops=[OpSequence(
                op_type="LINEAR_EVAL",
                params={
                    "coefficients": model.coefficients,
                    "intercept": model.intercept,
                    "num_features": model.num_features,
                }
            )]
        ))

        # Phase 2: Link function (if not identity)
        if model.link_function and model.link_function != "identity":
            approx = get_link_function_approx(model.link_function)
            schedule.append(ScheduleBlock(
                depth_level=1,
                node_group_id=0,
                ops=[OpSequence(
                    op_type="LINK_FUNCTION",
                    params={
                        "poly_coefficients": approx.coefficients,
                        "degree": approx.degree,
                        "domain": list(approx.domain),
                        "method": "horner",
                    }
                )]
            ))

        return ObliviousPlanIR(...)
```

### 2.3 Update Compiler Dispatch

**File:** `services/compiler/compiler.py`

```python
class Compiler:
    def compile(self, content, library_type, profile) -> ObliviousPlanIR:
        parser = get_parser(library_type)
        model_ir = parser.parse(content)

        # Route to correct optimizer based on model type
        if model_ir.model_type in ("logistic_regression", "linear_regression", "glm"):
            return self.linear_optimizer.optimize(model_ir, profile)
        else:
            # Tree-based models (GBDT, RF, single tree)
            if profile == "throughput":
                return self.optimizer_throughput.optimize(model_ir)
            else:
                return self.optimizer_latency.optimize(model_ir)
```

### Phase 2 Tests
- Compile logistic regression → verify LINEAR_EVAL + LINK_FUNCTION ops in plan
- Compile GLM (Poisson, log link) → verify exp() polynomial in plan
- Compile GLM (Gaussian, identity) → verify no link function op
- Compile GLM (Gamma, reciprocal) → verify reciprocal polynomial
- Polynomial approximation accuracy: sigmoid over [-8,8] within 0.5% of scipy
- Polynomial approximation accuracy: exp over [-4,4] within 0.5% of numpy
- Noise budget estimation for each link function degree
- Compile time benchmark: < 100ms for typical models

---

## Phase 3: Random Forest + Single Tree Optimization
**Goal:** RF and single decision tree compile to optimized FHE plans.

### 3.1 Random Forest Optimizer

**New file:** `services/compiler/random_forest_optimizer.py`

Extends existing `MOAIOptimizer` with RF-specific behavior:

```python
class RandomForestOptimizer(MOAIOptimizer):
    """
    Optimizer for Random Forest models.

    Key differences from GBDT:
    1. Aggregation: mean (not sum) -- divide by num_trees
    2. Trees are independent (not residual-fitting) -- parallelizable
    3. Soft voting preferred for classification (average probabilities)
    4. Can optionally convert to oblivious form (accuracy tradeoff)
    """

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        # Use parent MOAI optimizer for tree evaluation
        plan = super().optimize(model)

        # Replace final aggregation with MEAN_AGGREGATE
        plan.schedule.append(ScheduleBlock(
            depth_level=plan.schedule[-1].depth_level + 1,
            node_group_id=0,
            ops=[OpSequence(
                op_type="MEAN_AGGREGATE",
                params={"num_trees": model.num_trees, "task": "classification"}
            )]
        ))

        plan.metadata["aggregation"] = "mean"
        plan.metadata["model_class"] = "random_forest"
        return plan
```

**Oblivious conversion for RF:** Reuse existing MOAI-native tree conversion
(`services/innovations/moai_native.py`) which converts arbitrary trees to
oblivious form. This gives the same rotation-free comparison benefit as CatBoost
trees. Accuracy tradeoff is acceptable for RF since it's typically a challenger
model, not the champion.

### 3.2 Single Decision Tree Optimizer

**New file:** `services/compiler/decision_tree_optimizer.py`

```python
class DecisionTreeOptimizer(MOAIOptimizer):
    """
    Optimizer for single decision trees.

    Optimized for:
    1. Minimal FHE depth (tree depth 3-5 = very cheap)
    2. Reason code extraction (return per-feature contributions)
    3. Adverse action notice support
    """

    def optimize(self, model: ModelIR) -> ObliviousPlanIR:
        assert len(model.trees) == 1, "Single decision tree expected"
        tree = model.trees[0]

        plan = super().optimize(model)

        # Add reason code extraction op
        # This encodes each split decision as a binary indicator:
        # reason_vector[feature_i] = 1 if feature_i was the deciding split
        plan.schedule.append(ScheduleBlock(
            depth_level=tree.max_depth + 1,
            node_group_id=0,
            ops=[OpSequence(
                op_type="REASON_CODES",
                params={
                    "tree_depth": tree.max_depth,
                    "split_features": self._extract_split_features(tree),
                    "split_thresholds": self._extract_thresholds(tree),
                }
            )]
        ))

        plan.metadata["model_class"] = "decision_tree"
        plan.metadata["supports_reason_codes"] = True
        return plan
```

**Reason code extraction under FHE:** For a single tree of depth d:
- At each split, the comparison result (0 or 1) indicates which branch was taken
- The product of all branch indicators along a path gives the leaf indicator
- The feature that "most changed" the outcome is the primary reason code
- Under FHE: encode split decisions as encrypted binary vector, return alongside
  the prediction. Client decrypts both prediction and reason vector.

### 3.3 Routing in Compiler

Update the compiler to route RF and single tree to their optimizers:

```python
if model_ir.model_type in ("logistic_regression", "linear_regression", "glm"):
    return self.linear_optimizer.optimize(model_ir, profile)
elif model_ir.model_type == "random_forest":
    return self.rf_optimizer.optimize(model_ir)
elif model_ir.model_type == "decision_tree":
    return self.dt_optimizer.optimize(model_ir)
else:  # GBDT (xgboost, lightgbm, catboost)
    return self.tree_optimizer.optimize(model_ir)
```

### Phase 3 Tests
- Compile RF (100 trees, depth 6) → verify MEAN_AGGREGATE in plan
- Compile RF → verify rotation savings comparable to GBDT
- Compile single tree (depth 4) → verify REASON_CODES op in plan
- Single tree plan has no aggregation overhead
- RF plan metadata includes aggregation="mean"
- Decision tree plan metadata includes supports_reason_codes=True
- Verify single tree depth 3 has exactly 3 comparison levels
- Parse scikit-learn RF with 50 trees → 50 TreeIRs in model

---

## Phase 4: Client SDK Preprocessing Pipeline
**Goal:** Standardized, serializable preprocessing that runs before encryption.

### 4.1 Preprocessing Pipeline Module

**New file:** `sdk/python/preprocessing.py`

```python
class FHEPreprocessor:
    """
    Client-side preprocessing pipeline that prepares raw features
    for FHE encryption. All transformations happen in plaintext
    on the client before encryption.

    The pipeline is serialized alongside the model in the GBSP package
    to ensure training-serving consistency.
    """

    def __init__(self):
        self.steps: List[PreprocessingStep] = []

    def add_imputer(self, strategy="median", fill_values=None):
        """Handle missing values (NaN → fill value)."""

    def add_woe_transformer(self, woe_map: Dict[str, Dict]):
        """Apply Weight of Evidence binning for scorecards."""

    def add_scaler(self, method="standard", means=None, stds=None):
        """Standardize or normalize features."""

    def add_encoder(self, method="onehot", categories=None):
        """Encode categorical features."""

    def add_quantizer(self, bits=16, ranges=None):
        """Quantize to FHE-compatible precision."""

    def transform(self, features: Dict[str, Any]) -> List[float]:
        """Apply all steps and return FHE-ready float vector."""

    def to_json(self) -> str:
        """Serialize pipeline for inclusion in GBSP package."""

    @classmethod
    def from_json(cls, json_str: str) -> "FHEPreprocessor":
        """Load pipeline from GBSP package."""
```

### 4.2 WoE Transformer (Credit Scorecards)

```python
class WoETransformer(PreprocessingStep):
    """
    Weight of Evidence transformation for credit scorecard features.

    Takes raw feature values, bins them according to pre-computed
    bin edges, and returns the WoE value for the matching bin.

    Training-side:
    1. Fine classing (20-50 bins per feature)
    2. Coarse classing (Chi-merge to 8-10 bins, monotonic WoE)
    3. Compute WoE = ln(Distribution of Events / Distribution of Non-Events)
    4. Compute IV = sum((Dist_Events - Dist_NonEvents) * WoE)
    5. Select features with 0.02 < IV < 0.5

    Serving-side (this class):
    - Apply pre-computed bin edges to raw values
    - Return WoE float for each feature
    """

    def __init__(self, woe_map: Dict[str, Dict]):
        # woe_map: {"feature_name": {"bin_edges": [...], "woe_values": [...]}}
        self.woe_map = woe_map

    def transform(self, feature_name: str, value: float) -> float:
        bins = self.woe_map[feature_name]
        # Find bin index via bisect
        # Return corresponding WoE value
```

### 4.3 Scorecard Points Converter

```python
class ScorecardPointsConverter:
    """
    Convert logistic regression output to scorecard points.

    Standard scaling: score = offset - factor * ln(odds)
    Where: offset = base_points - factor * ln(base_odds)
           factor = pdo / ln(2)

    Typical parameters:
    - base_points = 600 (score at target odds)
    - base_odds = 1:19 (target odds ratio)
    - pdo = 50 (points to double the odds)
    """

    def __init__(self, base_points=600, base_odds=19.0, pdo=50):
        self.factor = pdo / math.log(2)
        self.offset = base_points - self.factor * math.log(base_odds)

    def log_odds_to_score(self, log_odds: float) -> float:
        return self.offset - self.factor * log_odds
```

### 4.4 Updated Client SDK

**File:** `sdk/python/client.py` -- extend with preprocessing:

```python
class FHEClient:
    """Unified client for all model types."""

    def __init__(self, gateway_addr, tenant_id):
        self.key_manager = N2HEKeyManager(tenant_id)
        self.preprocessor = None  # Loaded from GBSP package

    def load_preprocessor(self, preprocessor_json: str):
        """Load preprocessing pipeline from GBSP package."""
        self.preprocessor = FHEPreprocessor.from_json(preprocessor_json)

    def predict(self, compiled_model_id, raw_features):
        """Full pipeline: preprocess → encrypt → predict → decrypt."""
        # 1. Preprocess (plaintext, client-side)
        if self.preprocessor:
            features = self.preprocessor.transform(raw_features)
        else:
            features = raw_features

        # 2. Encrypt
        payload = self.key_manager.encrypt(features)

        # 3. Send to server (encrypted inference)
        response = self._call_predict(compiled_model_id, payload)

        # 4. Decrypt
        return self.key_manager.decrypt(response.outputs.payload)
```

### Phase 4 Tests
- WoE transformer: raw value → correct bin → correct WoE value
- Scaler: standardize features → mean=0, std=1
- Imputer: NaN values → replaced with median/mean
- Quantizer: float → quantized int → float roundtrip within tolerance
- Scorecard converter: log-odds → points with standard parameters
- Full pipeline: raw dict → preprocessing → encryption → decryption → match
- Pipeline serialization/deserialization roundtrip
- Pipeline applied to real credit data produces valid WoE ranges

---

## Phase 5: End-to-End Integration Tests + Benchmarks
**Goal:** Full pipeline works for all model types with correctness guarantees.

### 5.1 Correctness Tests (Encrypted vs Plaintext)

For each model type, verify that the FHE inference result matches the
plaintext prediction within an acceptable tolerance:

```python
class TestLogisticRegressionE2E:
    def test_encrypted_matches_plaintext(self):
        """Train LR, export, compile, encrypt-predict, compare to sklearn."""
        # 1. Train sklearn LogisticRegression on synthetic credit data
        # 2. Export via model_export.export_logistic_regression()
        # 3. Compile via compiler
        # 4. Encrypt features, run prediction, decrypt
        # 5. Compare to sklearn.predict_proba() -- within 1% tolerance

class TestGLMPoissonE2E:
    def test_poisson_glm_matches_statsmodels(self):
        """Train Poisson GLM, export, compile, encrypt-predict, compare."""
        # 1. Train statsmodels Poisson GLM on insurance claim data
        # 2. Export via model_export.export_glm()
        # 3. Compile (produces LINEAR_EVAL + exp() polynomial)
        # 4. Encrypt, predict, decrypt
        # 5. Compare to statsmodels.predict() -- within 2% tolerance

class TestRandomForestE2E:
    def test_rf_encrypted_matches_plaintext(self):
        """Train RF, export, compile, encrypt-predict, compare."""

class TestDecisionTreeE2E:
    def test_tree_with_reason_codes(self):
        """Train tree, export, compile, verify reason codes match path."""
```

### 5.2 Regulatory Compliance Tests

```python
class TestAdverseActionCompliance:
    def test_decision_tree_produces_reason_codes(self):
        """Verify encrypted single tree returns interpretable denial reasons."""

class TestScorecardCompliance:
    def test_woe_monotonicity(self):
        """Verify WoE values are monotonic across bins."""

    def test_scorecard_points_sum_correctly(self):
        """Verify individual feature points sum to total score."""
```

### 5.3 Performance Benchmarks

| Model Type | Target Latency (p50) | Target Throughput |
|------------|---------------------|-------------------|
| Single Decision Tree (depth 4) | < 10ms | > 5,000 eps |
| Logistic Regression (50 features) | < 15ms | > 3,000 eps |
| GLM (Poisson, 30 features) | < 20ms | > 2,500 eps |
| Random Forest (100 trees, depth 6) | < 65ms | > 500 eps |
| GBDT (100 trees, depth 6) | < 65ms | > 500 eps |

Logistic regression and single trees should be significantly faster than
GBDT because:
- LR: No tree comparisons, just dot product + polynomial
- Single tree: 1 tree vs 100 trees

### 5.4 Cross-Model Validation Tests

```python
class TestUnifiedPipeline:
    def test_same_data_different_models(self):
        """Same dataset produces valid predictions across all model types."""
        # Train LR, RF, DT, GBDT on same dataset
        # Export all four
        # Compile all four
        # Encrypt same features
        # Run all four predictions
        # Verify all produce reasonable outputs

    def test_preprocessing_consistency(self):
        """Same preprocessor works across model types."""
```

### Phase 5 Deliverables
- Correctness: encrypted matches plaintext within tolerance for all 4 new model types
- Compliance: adverse action reasons, scorecard points
- Performance: benchmarks for all model types
- Cross-model: unified pipeline verification

---

## Phase 6: Production Hardening
**Goal:** Protocol updates, GBSP spec extension, documentation.

### 6.1 Protocol Updates

**File:** `proto/control.proto`

```protobuf
message RegisterModelRequest {
    string library_type = 4;
    // Values: "xgboost", "lightgbm", "catboost",
    //         "scikit-learn", "sklearn", "statsmodels"
    string model_class = 5;
    // Values: "gbdt", "logistic_regression", "linear_regression",
    //         "glm", "random_forest", "decision_tree"
}
```

### 6.2 GBSP Package Extension

Extend `manifest.json` with:
```json
{
    "model_class": "logistic_regression",
    "library": "scikit-learn",
    "link_function": "logit",
    "num_features": 50,
    "preprocessing_pipeline": "preprocessing.json",
    "regulatory_classification": {
        "risk_tier": "high",
        "role": "champion",
        "regulations": ["ECOA", "Reg B", "HMDA"],
        "validation_status": "validated",
        "last_validation_date": "2026-01-15"
    }
}
```

### 6.3 Model Validation Metadata

The GBSP package includes evidence for SR 11-7 / OSFI E-23 compliance:
```json
{
    "evidence": {
        "train_auc": 0.82,
        "val_auc": 0.80,
        "gini": 0.60,
        "ks_statistic": 0.45,
        "psi": 0.03,
        "iv_scores": {"feature_1": 0.35, "feature_2": 0.22},
        "woe_monotonicity": true,
        "challenger_model_comparison": {
            "champion_auc": 0.82,
            "challenger_auc": 0.79,
            "model_type": "random_forest"
        }
    }
}
```

---

## Implementation Order & Dependencies

```
Phase 1 (IR + Parsers)          ← Foundation, no dependencies
    │
    ├── Phase 2 (Linear Optimizer)   ← Depends on Phase 1 IR extensions
    │       │
    │       └── Phase 4 (SDK Preprocessing) ← Uses Phase 2 models for testing
    │
    └── Phase 3 (RF + DT Optimizer)  ← Depends on Phase 1 parsers
            │
            └── Phase 5 (E2E Tests)  ← Depends on Phases 1-4
                    │
                    └── Phase 6 (Production)  ← Depends on Phase 5 passing
```

**Phases 2 and 3 can run in parallel** since they're independent optimizers.
Phase 4 can start as soon as Phase 1 is complete.

---

## FHE Operation Cost Summary

| Operation | FHE Cost | Used By |
|-----------|----------|---------|
| Addition (ct + ct) | Free (noise grows linearly) | All models |
| Scalar multiply (ct * plaintext) | Cheap (one NTT) | Linear models, aggregation |
| Ciphertext multiply (ct * ct) | Expensive (noise grows quadratically) | Polynomial eval |
| Comparison (ct < threshold) | Very expensive (sign function) | Tree models only |
| Rotation (slot permutation) | Expensive (key switching) | MOAI eliminates most |
| Bootstrapping (noise refresh) | Most expensive | Deep circuits only |

**Why linear models are fast under FHE:**
- Dot product = N scalar multiplies + N-1 additions = N cheap ops
- Sigmoid (degree 7) = 7 ciphertext multiplies = 7 expensive ops
- Total: N cheap + 7 expensive
- Compare to GBDT: 100 trees × depth comparisons × sign function evaluations

**Estimated multiplicative depth by model type:**

| Model | Depth | Bootstrapping Needed? |
|-------|-------|-----------------------|
| Linear regression (identity link) | 1 | No |
| Logistic regression (sigmoid deg 7) | 4 | Unlikely |
| GLM (exp deg 7) | 4 | Unlikely |
| Single tree (depth 4) | 4 | No |
| Random forest (100 trees, depth 6) | ~8 | Maybe |
| GBDT (100 trees, depth 6) | ~8 | Maybe |

---

## Risk Matrix

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sigmoid polynomial accuracy insufficient for regulatory PD calibration | High | Test against scipy.special.expit; use degree 9 if needed; document error bounds |
| RF oblivious conversion accuracy loss too high for challenger benchmarking | Medium | Make conversion optional; offer non-oblivious (slower but exact) mode |
| Preprocessing pipeline versioning drift between training and serving | High | Serialize pipeline in GBSP package; checksum verification at load time |
| Statsmodels export format instability across versions | Low | Pin supported versions; validate schema on parse |
| Adverse action reason codes lose fidelity under FHE noise | Medium | Single trees are low-depth (3-5) so noise is minimal; validate reason code accuracy in E2E tests |
