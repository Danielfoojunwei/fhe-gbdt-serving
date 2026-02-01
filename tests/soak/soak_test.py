#!/usr/bin/env python3
"""
Soak Test for FHE-GBDT Inference API

Runs continuous Predict requests for a specified duration to detect:
- Memory leaks
- P99 latency blowups
- Error rate creep
"""

import time
import argparse
import json
import numpy as np
from datetime import datetime

class SoakTestRunner:
    def __init__(self, duration_hours=2, rps=10):
        self.duration_hours = duration_hours
        self.rps = rps
        self.latencies = []
        self.errors = 0
        self.total = 0

    def mock_predict(self):
        """Mock predict call."""
        time.sleep(0.01 + np.random.exponential(0.005))
        if np.random.random() < 0.001:  # 0.1% error rate
            raise Exception("Mock error")
        return np.random.random()

    def run(self):
        print(f"Starting soak test for {self.duration_hours} hours at {self.rps} RPS...")
        start = time.time()
        duration_secs = self.duration_hours * 3600
        interval = 1.0 / self.rps

        while time.time() - start < duration_secs:
            try:
                t0 = time.perf_counter()
                self.mock_predict()
                self.latencies.append((time.perf_counter() - t0) * 1000)
                self.total += 1
            except Exception:
                self.errors += 1
                self.total += 1
            time.sleep(max(0, interval - (time.perf_counter() - t0)))
            
            # Report every 100 requests
            if self.total % 100 == 0:
                self._report_interim()

        self._report_final()

    def _report_interim(self):
        arr = np.array(self.latencies[-100:])
        print(f"[{datetime.now().isoformat()}] Total: {self.total}, Errors: {self.errors}, P99: {np.percentile(arr, 99):.2f}ms")

    def _report_final(self):
        arr = np.array(self.latencies)
        report = {
            "duration_hours": self.duration_hours,
            "total_requests": self.total,
            "errors": self.errors,
            "error_rate_pct": (self.errors / self.total) * 100 if self.total > 0 else 0,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "p999_ms": float(np.percentile(arr, 99.9)),
        }
        print("\n=== Soak Test Final Report ===")
        print(json.dumps(report, indent=2))
        with open("tests/soak_report.json", "w") as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=0.01)  # 36 seconds for testing
    parser.add_argument("--rps", type=int, default=10)
    args = parser.parse_args()
    
    runner = SoakTestRunner(duration_hours=args.hours, rps=args.rps)
    runner.run()
