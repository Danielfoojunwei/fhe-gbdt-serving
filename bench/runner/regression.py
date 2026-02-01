import json
import numpy as np
import os

class RegressionHarness:
    def __init__(self, baseline_path: str):
        self.baseline_path = baseline_path
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                self.baselines = json.load(f)
        else:
            self.baselines = {}

    def run_regression(self, model_id: str, encrypted_results: np.ndarray, plaintext_results: np.ndarray):
        mae = np.mean(np.abs(encrypted_results - plaintext_results))
        correlation = np.corrcoef(encrypted_results, plaintext_results)[0, 1]
        
        print(f"Regression results for {model_id}:")
        print(f"  MAE: {mae}")
        print(f"  Correlation: {correlation}")
        
        threshold = self.baselines.get(model_id, {}).get("mae_threshold", 0.01)
        if mae > threshold:
            print(f"WARNING: MAE {mae} exceeds threshold {threshold}")
            return False
        return True

    def save_baseline(self, model_id: str, mae_threshold: float):
        self.baselines[model_id] = {"mae_threshold": mae_threshold}
        with open(self.baseline_path, 'w') as f:
            json.dump(self.baselines, f, indent=2)

if __name__ == "__main__":
    harness = RegressionHarness("bench/baselines.json")
    # Sample run
    harness.run_regression("toy-model", np.array([0.51, 0.12]), np.array([0.5, 0.1]))
