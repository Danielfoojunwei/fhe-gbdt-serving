import os
import time
import requests
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Constants
GATEWAY_URL = "http://localhost:8080"
API_KEY = "test-tenant.cookbook"

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

class FHEGBDTClient:
    def __init__(self, url=GATEWAY_URL, api_key=API_KEY):
        self.url = url
        self.headers = {"x-api-key": api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def register_model(self, name: str, library: str, content: str, spec: Dict) -> str:
        payload = {
            "name": name,
            "library": library,
            "content": content,
            "feature_spec": spec
        }
        resp = self.session.post(f"{self.url}/v1/models", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def compile_model(self, model_id: str, profile: str = "latency") -> str:
        resp = self.session.post(f"{self.url}/v1/models/{model_id}/compile", params={"profile": profile})
        resp.raise_for_status()
        return resp.json()["compiled_model_id"]

    def upload_keys(self, compiled_model_id: str, eval_keys: bytes):
        # In a real SDK, this would upload binary keys. 
        # For this cookbook harness, we mock the upload if we don't have the real SDK 
        # or call the SDK if available.
        # Assuming usage of real SDK in recipes, this helper might just be for verifying status.
        pass

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
