from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
import json

@dataclass
class FeatureSpec:
    feature_names: List[str]
    missing_value_policy: str = "zero" # zero, mean, median
    categorical_encoding: Dict[str, Dict[str, int]] = field(default_factory=dict)
    quantization_scale: float = 1.0

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, data: str):
        return cls(**json.loads(data))
