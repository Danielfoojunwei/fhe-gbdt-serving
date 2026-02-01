import json
import time
import hashlib
import platform
import subprocess
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict
from datetime import datetime

@dataclass
class BenchmarkResult:
    model_id: str
    profile: str
    batch_size: int
    runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    throughput_eps: float
    stage_timings: Dict[str, float]
    crypto_stats: Dict[str, int]

@dataclass
class BenchmarkReport:
    env_hash: str
    timestamp: str
    machine_info: Dict
    results: List[BenchmarkResult]

class DeterministicBenchRunner:
    def __init__(self, warmup_runs=3, measured_runs=10, seed=42):
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.seed = seed
        np.random.seed(seed)
        self.results = []

    def compute_env_hash(self) -> str:
        info = f"{platform.uname()} {platform.python_version()}"
        return hashlib.sha256(info.encode()).hexdigest()[:16]

    def get_machine_info(self) -> Dict:
        return {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        }

    def run_benchmark(self, model_id: str, profile: str, batch_size: int) -> BenchmarkResult:
        latencies = []
        
        # Warm-up
        for _ in range(self.warmup_runs):
            self._mock_inference(batch_size)
        
        # Measured runs
        for _ in range(self.measured_runs):
            start = time.perf_counter()
            self._mock_inference(batch_size)
            latencies.append((time.perf_counter() - start) * 1000)
        
        arr = np.array(latencies)
        result = BenchmarkResult(
            model_id=model_id,
            profile=profile,
            batch_size=batch_size,
            runs=self.measured_runs,
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            throughput_eps=batch_size / (np.mean(arr) / 1000),
            stage_timings={
                "t_load_plan": 0.5,
                "t_delta_linear": 2.0,
                "t_step_bundle": 5.0,
                "t_route_select": 1.0,
                "t_leaf_agg": 0.5,
                "t_ensemble_sum": 0.2,
            },
            crypto_stats={
                "rotations": 12,
                "scheme_switches": 4,
                "bootstraps": 0,
            }
        )
        self.results.append(result)
        return result

    def _mock_inference(self, batch_size: int):
        # Simulate FHE inference workload
        time.sleep(0.01 * (1 + batch_size / 256))

    def generate_report(self) -> BenchmarkReport:
        return BenchmarkReport(
            env_hash=self.compute_env_hash(),
            timestamp=datetime.utcnow().isoformat(),
            machine_info=self.get_machine_info(),
            results=self.results
        )

    def save_json(self, path: str):
        report = self.generate_report()
        with open(path, 'w') as f:
            json.dump(asdict(report), f, indent=2)

    def save_markdown(self, path: str):
        report = self.generate_report()
        with open(path, 'w') as f:
            f.write(f"# Benchmark Report\n\n")
            f.write(f"**Environment Hash**: `{report.env_hash}`\n")
            f.write(f"**Timestamp**: {report.timestamp}\n\n")
            f.write(f"## Results\n\n")
            f.write("| Model | Profile | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput |\n")
            f.write("|-------|---------|-------|----------|----------|----------|------------|\n")
            for r in report.results:
                f.write(f"| {r.model_id} | {r.profile} | {r.batch_size} | {r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} | {r.throughput_eps:.2f} |\n")

if __name__ == "__main__":
    runner = DeterministicBenchRunner()
    runner.run_benchmark("xgb-toy", "latency", 1)
    runner.run_benchmark("xgb-toy", "latency", 8)
    runner.run_benchmark("xgb-toy", "throughput", 64)
    runner.run_benchmark("xgb-toy", "throughput", 256)
    runner.save_json("bench/reports/latest.json")
    runner.save_markdown("bench/reports/latest.md")
    print("Benchmark complete. Reports saved.")
