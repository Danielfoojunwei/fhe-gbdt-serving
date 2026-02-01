import os
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Use the real SDK client
from sdk.python.client import FHEGBDTClient as SDKClient

# Constants
GATEWAY_ADDR = "localhost:8080"
REGISTRY_ADDR = "localhost:8081"
TENANT_ID = "test-tenant-cookbook"

@dataclass
class CookbookResult:
    recipe_name: str
    model_type: str
    batch_size: int
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_eps: float
    plaintext_p50_ms: float
    plaintext_throughput_eps: float
    correctness_passed: bool
    correctness_metric: str
    correctness_value: float
    server_counters: Dict[str, int]

class BenchmarkHarness:
    def __init__(self, tenant_id=TENANT_ID):
        self.client = SDKClient(GATEWAY_ADDR, tenant_id)
        self.tenant_id = tenant_id

    def run_inference_cycle(self, compiled_model_id: str, features: List[Dict[str, float]], iterations: int) -> List[float]:
        latencies = []
        # Warmup
        self.client.predict_encrypted(compiled_model_id, features)
        
        for _ in range(iterations):
            start = time.perf_counter()
            self.client.predict_encrypted(compiled_model_id, features)
            latencies.append((time.perf_counter() - start) * 1000)
        return latencies

def load_sklearn_dataset(name: str):
    from sklearn import datasets
    if name == "breast_cancer":
        return datasets.load_breast_cancer()
    elif name == "diabetes":
        return datasets.load_diabetes()
    elif name == "california_housing":
        return datasets.fetch_california_housing()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def calculate_metrics(latencies: List[float], batch_size: int) -> Dict:
    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput": batch_size / (np.mean(arr) / 1000) if np.mean(arr) > 0 else 0
    }
