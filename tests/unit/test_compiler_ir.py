import unittest
from services.compiler.ir import TreeNode, TreeIR, ModelIR
from services.compiler.parser import XGBoostParser

class TestCompilerIR(unittest.TestCase):
    def test_tree_node_creation(self):
        node = TreeNode(node_id=0, feature_index=1, threshold=0.5, left_child_id=1, right_child_id=2)
        self.assertEqual(node.node_id, 0)
        self.assertEqual(node.threshold, 0.5)

    def test_xgboost_parser_skeleton(self):
        parser = XGBoostParser()
        model_ir = parser.parse(b'{"dummy": "data"}')
        self.assertEqual(model_ir.model_type, "xgboost")
        self.assertEqual(len(model_ir.trees), 0)

if __name__ == '__main__':
    unittest.main()
