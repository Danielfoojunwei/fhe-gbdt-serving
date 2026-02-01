import unittest
import json
from services.compiler.ir import TreeNode, TreeIR, ModelIR
from services.compiler.parser import XGBoostParser, LightGBMParser, CatBoostParser, get_parser

class TestCompilerIR(unittest.TestCase):
    def test_tree_node_creation(self):
        node = TreeNode(node_id=0, feature_index=1, threshold=0.5, left_child_id=1, right_child_id=2)
        self.assertEqual(node.node_id, 0)
        self.assertEqual(node.threshold, 0.5)

    def test_tree_node_leaf(self):
        """Test leaf node with no children."""
        node = TreeNode(node_id=1, feature_index=None, threshold=None,
                       left_child_id=None, right_child_id=None, leaf_value=0.5)
        self.assertIsNone(node.feature_index)
        self.assertEqual(node.leaf_value, 0.5)

    def test_xgboost_parser_valid_model(self):
        """Test parsing a valid minimal XGBoost model."""
        parser = XGBoostParser()

        # Minimal valid XGBoost JSON structure
        xgb_model = {
            "learner": {
                "gradient_booster": {
                    "model": {
                        "trees": [
                            {
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "split_indices": [0, 0, 0],
                                "split_conditions": [0.5, 0.0, 0.0],
                                "base_weights": [0.0, 0.3, 0.7]
                            }
                        ]
                    }
                },
                "learner_model_param": {
                    "base_score": "0.5"
                }
            }
        }

        model_ir = parser.parse(json.dumps(xgb_model).encode())
        self.assertEqual(model_ir.model_type, "xgboost")
        self.assertEqual(len(model_ir.trees), 1)
        self.assertEqual(model_ir.num_features, 1)
        self.assertEqual(model_ir.base_score, 0.5)

    def test_xgboost_parser_invalid_json_raises(self):
        """Test that invalid JSON raises appropriate error."""
        parser = XGBoostParser()

        with self.assertRaises(ValueError) as context:
            parser.parse(b'{"dummy": "data"}')

        self.assertIn("learner", str(context.exception))

    def test_xgboost_parser_empty_content_raises(self):
        """Test that empty content raises appropriate error."""
        parser = XGBoostParser()

        with self.assertRaises(ValueError):
            parser.parse(b'')

    def test_xgboost_parser_malformed_json_raises(self):
        """Test that malformed JSON raises appropriate error."""
        parser = XGBoostParser()

        with self.assertRaises(ValueError) as context:
            parser.parse(b'not json at all')

        self.assertIn("Invalid JSON", str(context.exception))

    def test_get_parser_xgboost(self):
        """Test parser factory for XGBoost."""
        parser = get_parser("xgboost")
        self.assertIsInstance(parser, XGBoostParser)

    def test_get_parser_lightgbm(self):
        """Test parser factory for LightGBM."""
        parser = get_parser("lightgbm")
        self.assertIsInstance(parser, LightGBMParser)

    def test_get_parser_catboost(self):
        """Test parser factory for CatBoost."""
        parser = get_parser("catboost")
        self.assertIsInstance(parser, CatBoostParser)

    def test_get_parser_unsupported(self):
        """Test parser factory for unsupported type."""
        parser = get_parser("unsupported")
        self.assertIsNone(parser)

if __name__ == '__main__':
    unittest.main()
