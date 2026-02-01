import pandas as pd
import numpy as np
from typing import List, Dict
from .features import FeatureSpec

class Preprocessor:
    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    def transform(self, data: List[Dict[str, float]]) -> np.ndarray:
        df = pd.DataFrame(data)
        # 1. Reorder features
        df = df.reindex(columns=self.spec.feature_names)
        
        # 2. Handle missing
        if self.spec.missing_value_policy == "zero":
            df = df.fillna(0.0)
            
        # 3. Quantize
        return df.values * self.spec.quantization_scale
