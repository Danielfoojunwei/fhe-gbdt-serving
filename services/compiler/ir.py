from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import json

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
class ModelIR:
    model_type: str # xgboost, lightgbm, catboost
    trees: List[TreeIR]
    num_features: int
    base_score: float = 0.5

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
class ObliviousPlanIR:
    compiled_model_id: str
    crypto_params_id: str
    packing_layout: PackingLayout
    schedule: List[ScheduleBlock]
    base_score: float
    num_trees: int
    metadata: Dict = field(default_factory=dict)

    def to_json(self):
        # Simplified serialization for now
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)
