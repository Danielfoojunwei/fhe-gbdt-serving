"""
End-to-End Tests for Multi-Model FHE Platform

Tests the full pipeline for all 4 new model types:
1. Logistic Regression (credit scoring)
2. Linear/GLM (actuarial pricing)
3. Random Forest (fraud detection)
4. Single Decision Tree (adverse action)

Each test verifies:
- Model export → Parse → Compile → Execution plan generation
- Correctness: FHE polynomial approximation vs plaintext
- Innovation reuse: gradient noise, MOAI native, link functions
- Compliance metadata: reason codes, audit trails
"""

import json
import math
import os
import sys
import unittest

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if os.path.join(PROJECT_ROOT, 'services') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'services'))
if os.path.join(PROJECT_ROOT, 'sdk', 'python') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'sdk', 'python'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# Test fixtures: Create model JSON exports without sklearn/statsmodels
# ============================================================================

def make_logistic_regression_json(
    weights=None, intercept=0.5, num_features=5
):
    """Create a logistic regression model export JSON."""
    if weights is None:
        weights = [0.3, -0.5, 0.8, -0.2, 0.6][:num_features]
    data = {
        "model_type": "logistic_regression",
        "library": "manual",
        "coefficients": [weights],
        "intercept": [intercept],
        "classes": [0, 1],
        "num_features": len(weights),
        "feature_names": [f"feature_{i}" for i in range(len(weights))],
        "preprocessing": [],
    }
    return json.dumps(data).encode("utf-8")


def make_glm_json(
    weights=None, intercept=1.0, family="poisson", link="log", num_features=4
):
    """Create a GLM model export JSON."""
    if weights is None:
        weights = [0.2, -0.3, 0.5, 0.1][:num_features]
    params = [intercept] + weights
    param_names = ["const"] + [f"x{i}" for i in range(len(weights))]
    data = {
        "model_type": "glm",
        "library": "manual",
        "params": params,
        "param_names": param_names,
        "family": family,
        "link": link,
        "num_features": len(weights),
        "feature_names": [f"x{i}" for i in range(len(weights))],
        "preprocessing": [],
    }
    return json.dumps(data).encode("utf-8")


def make_random_forest_json(num_trees=3, num_features=4, max_depth=3):
    """Create a random forest model export JSON."""
    np.random.seed(42)
    trees = []
    for tree_id in range(num_trees):
        nodes = []
        node_id = 0

        def _build(depth, nid_start):
            nonlocal node_id
            nid = nid_start
            if depth >= max_depth:
                nodes.append({
                    "node_id": nid,
                    "depth": depth,
                    "is_leaf": True,
                    "value": float(np.random.uniform(0, 1)),
                })
                return nid
            feat = int(np.random.randint(0, num_features))
            thresh = float(np.random.uniform(0, 1))
            left_nid = nid + 1
            node_id = left_nid
            left_ret = _build(depth + 1, left_nid)
            right_nid = node_id + 1
            node_id = right_nid
            right_ret = _build(depth + 1, right_nid)
            nodes.append({
                "node_id": nid,
                "depth": depth,
                "is_leaf": False,
                "feature_index": feat,
                "threshold": thresh,
                "left_child": left_nid,
                "right_child": right_nid,
            })
            return nid

        _build(0, 0)
        # Sort by node_id for consistent ordering
        nodes.sort(key=lambda x: x["node_id"])
        trees.append({"tree_id": tree_id, "nodes": nodes})

    data = {
        "model_type": "random_forest",
        "library": "manual",
        "trees": trees,
        "num_features": num_features,
        "n_estimators": num_trees,
        "is_classifier": False,
        "feature_names": [f"f{i}" for i in range(num_features)],
        "preprocessing": [],
    }
    return json.dumps(data).encode("utf-8")


def make_decision_tree_json(num_features=5, max_depth=3):
    """Create a single decision tree model export JSON."""
    nodes = [
        {"node_id": 0, "depth": 0, "is_leaf": False,
         "feature_index": 2, "threshold": 0.5,
         "left_child": 1, "right_child": 2},
        {"node_id": 1, "depth": 1, "is_leaf": False,
         "feature_index": 0, "threshold": 0.3,
         "left_child": 3, "right_child": 4},
        {"node_id": 2, "depth": 1, "is_leaf": False,
         "feature_index": 4, "threshold": 0.7,
         "left_child": 5, "right_child": 6},
        {"node_id": 3, "depth": 2, "is_leaf": True, "value": 0.1},
        {"node_id": 4, "depth": 2, "is_leaf": True, "value": 0.4},
        {"node_id": 5, "depth": 2, "is_leaf": True, "value": 0.6},
        {"node_id": 6, "depth": 2, "is_leaf": True, "value": 0.9},
    ]
    data = {
        "model_type": "decision_tree",
        "library": "manual",
        "nodes": nodes,
        "num_features": num_features,
        "max_depth": max_depth,
        "is_classifier": True,
        "classes": [0, 1],
        "feature_names": ["age", "income", "credit_score", "dti", "employment_years"],
        "preprocessing": [],
    }
    return json.dumps(data).encode("utf-8")


# ============================================================================
# IR Extension Tests
# ============================================================================

class TestIRExtension(unittest.TestCase):
    """Test that the extended IR is backward-compatible."""

    def test_model_family_enum(self):
        from compiler.ir import ModelFamily
        self.assertEqual(ModelFamily.TREE_ENSEMBLE, "tree_ensemble")
        self.assertEqual(ModelFamily.LINEAR, "linear")
        self.assertEqual(ModelFamily.RANDOM_FOREST, "random_forest")
        self.assertEqual(ModelFamily.SINGLE_TREE, "single_tree")

    def test_link_function_enum(self):
        from compiler.ir import LinkFunction
        self.assertEqual(LinkFunction.IDENTITY, "identity")
        self.assertEqual(LinkFunction.LOGIT, "logit")
        self.assertEqual(LinkFunction.LOG, "log")
        self.assertEqual(LinkFunction.RECIPROCAL, "reciprocal")

    def test_aggregation_enum(self):
        from compiler.ir import Aggregation
        self.assertEqual(Aggregation.SUM, "sum")
        self.assertEqual(Aggregation.MEAN, "mean")
        self.assertEqual(Aggregation.NONE, "none")

    def test_linear_coefficients(self):
        from compiler.ir import LinearCoefficients
        lc = LinearCoefficients(weights=[0.5, -0.3, 0.8], intercept=1.0)
        self.assertEqual(lc.num_features, 3)
        self.assertEqual(lc.intercept, 1.0)

    def test_backward_compatible_model_ir(self):
        """Existing GBDT code should still work with extended ModelIR."""
        from compiler.ir import ModelIR, TreeIR, TreeNode, ModelFamily
        # Create a ModelIR the old way (without new fields)
        model = ModelIR(
            model_type="xgboost",
            trees=[TreeIR(tree_id=0, nodes={0: TreeNode(node_id=0, leaf_value=1.0)}, root_id=0, max_depth=0)],
            num_features=3,
            base_score=0.5,
        )
        # Default values should be backward-compatible
        self.assertEqual(model.model_family, ModelFamily.TREE_ENSEMBLE)
        self.assertIsNone(model.coefficients)
        self.assertEqual(model.aggregation.value, "sum")

    def test_linear_plan_ir(self):
        from compiler.ir import LinearPlanIR, PackingLayout
        layout = PackingLayout(layout_type="moai_column", feature_to_ciphertext={0: 0}, slots=1)
        plan = LinearPlanIR(
            compiled_model_id="test123",
            crypto_params_id="n2he_default",
            packing_layout=layout,
            coefficients=[0.5, -0.3],
            intercept=1.0,
            link_function="logit",
            poly_coeffs=[0.5, 0.25, 0.0, -0.02],
            poly_degree=3,
            num_features=2,
        )
        self.assertEqual(plan.link_function, "logit")
        self.assertEqual(plan.poly_degree, 3)
        json_str = plan.to_json()
        self.assertIn("logit", json_str)


# ============================================================================
# Parser Tests
# ============================================================================

class TestLogisticRegressionParser(unittest.TestCase):
    """Test scikit-learn LogisticRegression parser."""

    def test_parse_basic(self):
        from compiler.sklearn_parser import ScikitLearnLogisticRegressionParser
        from compiler.ir import ModelFamily, LinkFunction

        content = make_logistic_regression_json()
        parser = ScikitLearnLogisticRegressionParser()
        model = parser.parse(content)

        self.assertEqual(model.model_type, "logistic_regression")
        self.assertEqual(model.model_family, ModelFamily.LINEAR)
        self.assertEqual(model.link_function, LinkFunction.LOGIT)
        self.assertEqual(model.num_features, 5)
        self.assertIsNotNone(model.coefficients)
        self.assertEqual(len(model.coefficients.weights), 5)
        self.assertAlmostEqual(model.coefficients.intercept, 0.5)

    def test_parse_via_factory(self):
        from compiler.parser import get_parser
        parser = get_parser("logistic_regression")
        self.assertIsNotNone(parser)
        model = parser.parse(make_logistic_regression_json())
        self.assertEqual(model.model_type, "logistic_regression")

    def test_wrong_model_type(self):
        from compiler.sklearn_parser import ScikitLearnLogisticRegressionParser
        bad_json = json.dumps({"model_type": "xgboost"}).encode()
        parser = ScikitLearnLogisticRegressionParser()
        with self.assertRaises(ValueError):
            parser.parse(bad_json)


class TestGLMParser(unittest.TestCase):
    """Test statsmodels GLM parser."""

    def test_parse_poisson(self):
        from compiler.glm_parser import StatsmodelsGLMParser
        from compiler.ir import LinkFunction

        content = make_glm_json(family="poisson", link="log")
        parser = StatsmodelsGLMParser()
        model = parser.parse(content)

        self.assertEqual(model.model_type, "linear_glm")
        self.assertEqual(model.glm_family, "poisson")
        self.assertEqual(model.link_function, LinkFunction.LOG)
        self.assertEqual(model.num_features, 4)

    def test_parse_gaussian(self):
        from compiler.glm_parser import StatsmodelsGLMParser
        content = make_glm_json(family="gaussian", link="identity")
        model = StatsmodelsGLMParser().parse(content)
        self.assertEqual(model.glm_family, "gaussian")

    def test_parse_binomial(self):
        from compiler.glm_parser import StatsmodelsGLMParser
        content = make_glm_json(family="binomial", link="logit")
        model = StatsmodelsGLMParser().parse(content)
        self.assertEqual(model.glm_family, "binomial")

    def test_parse_via_factory(self):
        from compiler.parser import get_parser
        parser = get_parser("glm")
        self.assertIsNotNone(parser)
        model = parser.parse(make_glm_json())
        self.assertEqual(model.model_type, "linear_glm")


class TestRandomForestParser(unittest.TestCase):
    """Test scikit-learn RandomForest parser."""

    def test_parse_basic(self):
        from compiler.sklearn_parser import ScikitLearnRandomForestParser
        from compiler.ir import ModelFamily, Aggregation

        content = make_random_forest_json(num_trees=3)
        parser = ScikitLearnRandomForestParser()
        model = parser.parse(content)

        self.assertEqual(model.model_type, "random_forest")
        self.assertEqual(model.model_family, ModelFamily.RANDOM_FOREST)
        self.assertEqual(model.aggregation, Aggregation.MEAN)
        self.assertEqual(len(model.trees), 3)
        self.assertEqual(model.num_features, 4)

    def test_parse_via_factory(self):
        from compiler.parser import get_parser
        parser = get_parser("random_forest")
        model = parser.parse(make_random_forest_json())
        self.assertEqual(model.model_type, "random_forest")

    def test_tree_structure(self):
        from compiler.sklearn_parser import ScikitLearnRandomForestParser
        content = make_random_forest_json(num_trees=1, max_depth=2)
        model = ScikitLearnRandomForestParser().parse(content)
        tree = model.trees[0]
        # Should have both leaf and split nodes
        leaf_nodes = [n for n in tree.nodes.values() if n.leaf_value is not None]
        split_nodes = [n for n in tree.nodes.values() if n.feature_index is not None]
        self.assertGreater(len(leaf_nodes), 0)
        self.assertGreater(len(split_nodes), 0)


class TestDecisionTreeParser(unittest.TestCase):
    """Test scikit-learn DecisionTree parser."""

    def test_parse_basic(self):
        from compiler.sklearn_parser import ScikitLearnDecisionTreeParser
        from compiler.ir import ModelFamily, Aggregation

        content = make_decision_tree_json()
        parser = ScikitLearnDecisionTreeParser()
        model = parser.parse(content)

        self.assertEqual(model.model_type, "decision_tree")
        self.assertEqual(model.model_family, ModelFamily.SINGLE_TREE)
        self.assertEqual(model.aggregation, Aggregation.NONE)
        self.assertEqual(len(model.trees), 1)
        self.assertEqual(model.num_features, 5)
        self.assertTrue(model.metadata.get("supports_reason_codes"))

    def test_feature_names(self):
        from compiler.sklearn_parser import ScikitLearnDecisionTreeParser
        content = make_decision_tree_json()
        model = ScikitLearnDecisionTreeParser().parse(content)
        self.assertEqual(model.feature_names, ["age", "income", "credit_score", "dti", "employment_years"])

    def test_parse_via_factory(self):
        from compiler.parser import get_parser
        parser = get_parser("decision_tree")
        model = parser.parse(make_decision_tree_json())
        self.assertEqual(model.model_type, "decision_tree")


# ============================================================================
# Link Function Approximation Tests
# ============================================================================

class TestLinkFunctions(unittest.TestCase):
    """Test polynomial approximations of link functions."""

    def test_identity(self):
        from compiler.link_functions import get_link_approximation
        approx = get_link_approximation("identity")
        self.assertEqual(approx.degree, 1)
        self.assertEqual(approx.max_error, 0.0)
        x = np.linspace(-5, 5, 100)
        np.testing.assert_allclose(approx.evaluate(x), x, atol=1e-10)

    def test_logit_sigmoid(self):
        from compiler.link_functions import get_link_approximation
        approx = get_link_approximation("logit", degree=7)
        self.assertEqual(approx.degree, 7)

        x = np.linspace(-6, 6, 1000)
        y_true = 1.0 / (1.0 + np.exp(-x))
        y_approx = approx.evaluate(x)

        # Should be accurate to within 3% on [-6, 6] (well within FHE noise)
        max_err = np.max(np.abs(y_true - y_approx))
        self.assertLess(max_err, 0.03, f"Sigmoid approx error too high: {max_err}")

    def test_log_exp(self):
        from compiler.link_functions import get_link_approximation
        approx = get_link_approximation("log", degree=7, domain=(-3, 3))
        x = np.linspace(-2.5, 2.5, 500)
        y_true = np.exp(x)
        y_approx = approx.evaluate(x)
        rel_err = np.max(np.abs(y_true - y_approx) / np.abs(y_true))
        self.assertLess(rel_err, 0.05, f"Exp approx relative error too high: {rel_err}")

    def test_reciprocal(self):
        from compiler.link_functions import get_link_approximation
        approx = get_link_approximation("reciprocal", degree=5, domain=(0.5, 5.0))
        x = np.linspace(0.6, 4.5, 500)
        y_true = 1.0 / x
        y_approx = approx.evaluate(x)
        max_err = np.max(np.abs(y_true - y_approx))
        self.assertLess(max_err, 0.1)

    def test_probit(self):
        from compiler.link_functions import get_link_approximation
        approx = get_link_approximation("probit", degree=7)
        x = np.linspace(-3, 3, 500)
        y_true = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2)))
        y_approx = approx.evaluate(x)
        max_err = np.max(np.abs(y_true - y_approx))
        self.assertLess(max_err, 0.02)

    def test_multiplicative_depth(self):
        from compiler.link_functions import get_link_approximation
        approx = get_link_approximation("logit", degree=5)
        self.assertEqual(approx.multiplicative_depth, 5)


# ============================================================================
# Compiler / Optimizer Integration Tests
# ============================================================================

class TestLogisticRegressionCompiler(unittest.TestCase):
    """Test full compile pipeline for logistic regression."""

    def test_compile_end_to_end(self):
        from compiler.compiler import Compiler
        from compiler.ir import LinearPlanIR

        compiler = Compiler()
        content = make_logistic_regression_json()
        plan = compiler.compile(content, "logistic_regression", "latency")

        self.assertIsInstance(plan, LinearPlanIR)
        self.assertEqual(plan.link_function, "logit")
        self.assertEqual(plan.num_features, 5)
        self.assertGreater(plan.poly_degree, 0)
        self.assertGreater(len(plan.poly_coeffs), 0)

    def test_compile_has_correct_ops(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_logistic_regression_json(), "logistic_regression", "latency")

        op_types = [op.op_type for op in plan.ops]
        self.assertIn("DOT_PRODUCT", op_types)
        self.assertIn("ADD_BIAS", op_types)
        self.assertIn("LINK_FUNCTION", op_types)

    def test_compile_sigmoid_accuracy(self):
        """Verify compiled sigmoid polynomial matches true sigmoid."""
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_logistic_regression_json(), "logistic_regression", "latency")

        # Evaluate the polynomial on test points
        coeffs = plan.poly_coeffs
        x = np.linspace(-5, 5, 100)
        # Horner's method
        result = np.full_like(x, coeffs[-1])
        for i in range(len(coeffs) - 2, -1, -1):
            result = result * x + coeffs[i]

        y_true = 1.0 / (1.0 + np.exp(-x))
        max_err = np.max(np.abs(y_true - result))
        self.assertLess(max_err, 0.03)

    def test_metadata_includes_link_info(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_logistic_regression_json(), "logistic_regression", "latency")

        self.assertEqual(plan.metadata["link_function"], "logit")
        self.assertIn("link_max_error", plan.metadata)
        self.assertIn("multiplicative_depth", plan.metadata)
        self.assertEqual(plan.metadata["model_family"], "linear")

    def test_throughput_profile(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_logistic_regression_json(), "logistic_regression", "throughput")
        self.assertEqual(plan.metadata["profile"], "throughput")


class TestGLMCompiler(unittest.TestCase):
    """Test full compile pipeline for GLM models."""

    def test_compile_poisson(self):
        from compiler.compiler import Compiler
        from compiler.ir import LinearPlanIR

        compiler = Compiler()
        plan = compiler.compile(make_glm_json(family="poisson", link="log"), "glm", "latency")

        self.assertIsInstance(plan, LinearPlanIR)
        self.assertEqual(plan.link_function, "log")
        self.assertEqual(plan.metadata["glm_family"], "poisson")

    def test_compile_gaussian_identity(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(
            make_glm_json(family="gaussian", link="identity"), "glm", "latency"
        )
        self.assertEqual(plan.link_function, "identity")
        # Identity link should NOT have LINK_FUNCTION op
        op_types = [op.op_type for op in plan.ops]
        self.assertNotIn("LINK_FUNCTION", op_types)

    def test_compile_binomial(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(
            make_glm_json(family="binomial", link="logit"), "glm", "latency"
        )
        self.assertEqual(plan.link_function, "logit")
        op_types = [op.op_type for op in plan.ops]
        self.assertIn("LINK_FUNCTION", op_types)


class TestRandomForestCompiler(unittest.TestCase):
    """Test full compile pipeline for random forest."""

    def test_compile_end_to_end(self):
        from compiler.compiler import Compiler
        from compiler.ir import ObliviousPlanIR

        compiler = Compiler()
        content = make_random_forest_json(num_trees=5)
        plan = compiler.compile(content, "random_forest", "latency")

        self.assertIsInstance(plan, ObliviousPlanIR)
        self.assertEqual(plan.num_trees, 5)
        self.assertEqual(plan.metadata["model_family"], "random_forest")
        self.assertEqual(plan.metadata["aggregation"], "mean")

    def test_mean_aggregation_op(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_random_forest_json(), "random_forest", "latency")

        # Last schedule block should be AGGREGATE_MEAN
        last_block = plan.schedule[-1]
        self.assertEqual(last_block.depth_level, -1)
        self.assertEqual(last_block.ops[0].op_type, "AGGREGATE_MEAN")
        self.assertEqual(last_block.ops[0].params["num_trees"], 3)

    def test_rotation_savings(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(
            make_random_forest_json(num_trees=10), "random_forest", "latency"
        )
        savings = plan.metadata.get("rotation_savings", {})
        self.assertGreater(savings.get("savings_percent", 0), 90)


class TestDecisionTreeCompiler(unittest.TestCase):
    """Test full compile pipeline for single decision tree."""

    def test_compile_end_to_end(self):
        from compiler.compiler import Compiler
        from compiler.ir import ObliviousPlanIR

        compiler = Compiler()
        plan = compiler.compile(make_decision_tree_json(), "decision_tree", "latency")

        self.assertIsInstance(plan, ObliviousPlanIR)
        self.assertEqual(plan.num_trees, 1)
        self.assertEqual(plan.metadata["model_family"], "single_tree")

    def test_reason_codes(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_decision_tree_json(), "decision_tree", "latency")

        # Should have REASON_CODES operation
        reason_block = None
        for block in plan.schedule:
            for op in block.ops:
                if op.op_type == "REASON_CODES":
                    reason_block = op
        self.assertIsNotNone(reason_block, "Missing REASON_CODES operation")
        self.assertIn("features_by_depth", reason_block.params)
        self.assertIn("feature_names", reason_block.params)

    def test_adverse_action_metadata(self):
        from compiler.compiler import Compiler

        compiler = Compiler()
        plan = compiler.compile(make_decision_tree_json(), "decision_tree", "latency")
        self.assertTrue(plan.metadata.get("supports_adverse_action"))
        self.assertTrue(plan.metadata.get("reason_codes_enabled"))


# ============================================================================
# SDK Preprocessing Tests
# ============================================================================

class TestPreprocessing(unittest.TestCase):
    """Test client-side preprocessing pipeline."""

    def test_woe_transformer(self):
        from preprocessing import WoETransformer
        binning_table = {
            "age": [
                {"lo": 0, "hi": 25, "woe": -0.5, "iv": 0.1},
                {"lo": 25, "hi": 45, "woe": 0.3, "iv": 0.05},
                {"lo": 45, "hi": float("inf"), "woe": 0.8, "iv": 0.15},
            ]
        }
        woe = WoETransformer(binning_table)
        features = np.array([[20.0, 100.0], [30.0, 200.0], [50.0, 300.0]])
        result = woe.transform(features, ["age", "income"])

        self.assertAlmostEqual(result[0, 0], -0.5)  # age=20 → WoE=-0.5
        self.assertAlmostEqual(result[1, 0], 0.3)    # age=30 → WoE=0.3
        self.assertAlmostEqual(result[2, 0], 0.8)    # age=50 → WoE=0.8
        # income column unchanged (not in binning table)
        self.assertEqual(result[0, 1], 100.0)

    def test_woe_information_values(self):
        from preprocessing import WoETransformer
        binning_table = {
            "age": [{"lo": 0, "hi": 100, "woe": 0.5, "iv": 0.3}],
            "income": [{"lo": 0, "hi": 1e6, "woe": -0.2, "iv": 0.15}],
        }
        woe = WoETransformer(binning_table)
        ivs = woe.information_values
        self.assertAlmostEqual(ivs["age"], 0.3)
        self.assertAlmostEqual(ivs["income"], 0.15)

    def test_scorecard_points(self):
        from preprocessing import ScorecardPointsConverter
        converter = ScorecardPointsConverter(
            base_score=600, base_odds=50, pdo=20
        )
        # At base_odds=50 (good:bad = 50:1), p_default = 1/51 ≈ 0.0196
        p_base = 1.0 / 51.0
        score = converter.probability_to_score(np.array([p_base]))
        self.assertAlmostEqual(score[0], 600, delta=1)

        # Higher default probability → lower score (riskier)
        scores = converter.probability_to_score(np.array([0.1, 0.5, 0.9]))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_scorecard_roundtrip(self):
        from preprocessing import ScorecardPointsConverter
        converter = ScorecardPointsConverter()
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        scores = converter.probability_to_score(probs)
        recovered = converter.score_to_probability(scores.astype(float))
        np.testing.assert_allclose(probs, recovered, atol=0.01)

    def test_fhe_preprocessor_pipeline(self):
        from preprocessing import FHEPreprocessor
        steps = [
            {"type": "clip", "params": {"min": -8.0, "max": 8.0}},
        ]
        preprocessor = FHEPreprocessor(steps)
        features = np.array([[-100.0, 5.0, 20.0]])
        result = preprocessor.transform(features)
        self.assertEqual(result[0, 0], -8.0)   # Clipped
        self.assertEqual(result[0, 1], 5.0)     # Unchanged
        self.assertEqual(result[0, 2], 8.0)     # Clipped

    def test_fhe_preprocessor_standardize(self):
        from preprocessing import FHEPreprocessor
        steps = [
            {"type": "standardize", "params": {
                "mean": [10.0, 20.0],
                "std": [2.0, 5.0],
            }},
        ]
        preprocessor = FHEPreprocessor(steps)
        features = np.array([[12.0, 30.0]])
        result = preprocessor.transform(features)
        self.assertAlmostEqual(result[0, 0], 1.0)   # (12-10)/2
        self.assertAlmostEqual(result[0, 1], 2.0)   # (30-20)/5

    def test_fhe_preprocessor_from_plan(self):
        from preprocessing import FHEPreprocessor
        plan_metadata = {
            "link_domain": [-8.0, 8.0],
            "preprocessing": [],
        }
        preprocessor = FHEPreprocessor.from_plan(plan_metadata)
        features = np.array([[100.0]])
        result = preprocessor.transform(features)
        self.assertEqual(result[0, 0], 8.0)

    def test_credit_scorecard_preprocessor(self):
        from preprocessing import FHEPreprocessor
        binning = {
            "credit_score": [
                {"lo": 300, "hi": 600, "woe": -1.0, "iv": 0.5},
                {"lo": 600, "hi": 800, "woe": 0.5, "iv": 0.3},
            ]
        }
        preprocessor = FHEPreprocessor.for_credit_scorecard(binning)
        features = np.array([[500.0]])
        result = preprocessor.transform(features, ["credit_score"])
        self.assertAlmostEqual(result[0, 0], -1.0)  # WoE for credit_score=500


# ============================================================================
# Model Export Utility Tests
# ============================================================================

class TestModelExport(unittest.TestCase):
    """Test model export utilities."""

    def test_manual_logistic_regression(self):
        from model_export import create_manual_logistic_regression
        from compiler.parser import get_parser

        export = create_manual_logistic_regression(
            weights=[0.5, -0.3, 0.8],
            intercept=1.0,
            feature_names=["age", "income", "score"],
        )
        parser = get_parser("logistic_regression")
        model = parser.parse(export)
        self.assertEqual(model.num_features, 3)
        self.assertAlmostEqual(model.coefficients.intercept, 1.0)

    def test_manual_glm(self):
        from model_export import create_manual_glm
        from compiler.parser import get_parser

        export = create_manual_glm(
            weights=[0.2, -0.1],
            intercept=0.5,
            family="poisson",
            link="log",
        )
        parser = get_parser("glm")
        model = parser.parse(export)
        self.assertEqual(model.glm_family, "poisson")
        self.assertEqual(model.num_features, 2)


# ============================================================================
# Cross-Model Unified Pipeline Tests
# ============================================================================

class TestUnifiedPipeline(unittest.TestCase):
    """Test that all model types work through the unified compiler."""

    def test_all_models_compile(self):
        """Every supported model type should compile without error."""
        from compiler.compiler import Compiler
        compiler = Compiler()

        test_cases = [
            ("logistic_regression", make_logistic_regression_json()),
            ("glm", make_glm_json()),
            ("random_forest", make_random_forest_json()),
            ("decision_tree", make_decision_tree_json()),
        ]

        for lib_type, content in test_cases:
            with self.subTest(model=lib_type):
                plan = compiler.compile(content, lib_type, "latency")
                self.assertIsNotNone(plan)
                json_str = plan.to_json()
                self.assertGreater(len(json_str), 10)

    def test_gbdt_still_works(self):
        """Ensure existing GBDT compilation is unbroken."""
        from compiler.compiler import Compiler
        from compiler.ir import ObliviousPlanIR

        # Create a simple XGBoost-style model
        xgb_json = {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [{
                            "left_children": [-1, -1, 1],
                            "right_children": [-1, -1, 0],
                            "split_indices": [0, 0, 0],
                            "split_conditions": [0.0, 0.0, 0.5],
                            "base_weights": [0.1, 0.2, 0.0],
                        }]
                    }
                },
                "learner_model_param": {"base_score": "0.5"}
            }
        }
        content = json.dumps(xgb_json).encode()
        compiler = Compiler()
        plan = compiler.compile(content, "xgboost", "latency")
        self.assertIsInstance(plan, ObliviousPlanIR)

    def test_unsupported_library_type(self):
        from compiler.compiler import Compiler
        compiler = Compiler()
        with self.assertRaises(ValueError):
            compiler.compile(b'{}', "unsupported_type", "latency")

    def test_plan_serialization_all_models(self):
        """All plan types should serialize to JSON."""
        from compiler.compiler import Compiler
        compiler = Compiler()

        test_cases = [
            ("logistic_regression", make_logistic_regression_json()),
            ("glm", make_glm_json()),
            ("random_forest", make_random_forest_json()),
            ("decision_tree", make_decision_tree_json()),
        ]
        for lib_type, content in test_cases:
            with self.subTest(model=lib_type):
                plan = compiler.compile(content, lib_type, "latency")
                json_str = plan.to_json()
                parsed = json.loads(json_str)
                self.assertIn("compiled_model_id", parsed)


# ============================================================================
# Innovation Reuse Tests
# ============================================================================

class TestInnovationReuse(unittest.TestCase):
    """Test that existing innovations are properly leveraged."""

    def test_gradient_noise_with_lr(self):
        """Gradient noise allocation should work with LR coefficient magnitudes."""
        try:
            from services.innovations.gradient_noise import (
                GradientAwareNoiseAllocator,
                FeatureImportance,
            )
        except ImportError:
            self.skipTest("Gradient noise innovation not available")

        # Simulate LR coefficients as importance
        importance_map = {
            0: FeatureImportance(0, 0.9, 1, 0.9, 0.9),   # High importance
            1: FeatureImportance(1, 0.1, 1, 0.1, 0.1),   # Low importance
            2: FeatureImportance(2, 0.5, 1, 0.5, 0.5),   # Medium
        }
        allocator = GradientAwareNoiseAllocator()
        allocs = allocator.allocate_from_importance(importance_map, 3)

        # High importance feature should get more precision bits
        self.assertGreaterEqual(allocs[0].precision_bits, allocs[1].precision_bits)

    def test_moai_native_with_rf(self):
        """MOAI native converter should work with RF trees."""
        try:
            from services.innovations.moai_native import RotationOptimalConverter
        except ImportError:
            self.skipTest("MOAI native innovation not available")

        from compiler.sklearn_parser import ScikitLearnRandomForestParser

        content = make_random_forest_json(num_trees=2, max_depth=3)
        model = ScikitLearnRandomForestParser().parse(content)

        converter = RotationOptimalConverter()
        result = converter.convert_model(model)
        self.assertEqual(len(result.oblivious_trees), 2)
        self.assertGreater(
            result.rotation_savings.get("savings_percent", 0), 0
        )

    def test_polynomial_evaluator_for_sigmoid(self):
        """Polynomial leaves evaluator should work for sigmoid approximation."""
        try:
            from services.innovations.polynomial_leaves import FHEPolynomialEvaluator
        except ImportError:
            self.skipTest("Polynomial leaves innovation not available")

        evaluator = FHEPolynomialEvaluator(max_degree=7)
        # Verify the evaluator is compatible with our link function polynomials
        self.assertEqual(evaluator.max_degree, 7)
        noise_cost = evaluator.estimate_noise_cost(7)
        self.assertGreater(noise_cost, 0)


if __name__ == "__main__":
    unittest.main()
