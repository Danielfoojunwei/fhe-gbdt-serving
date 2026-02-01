import time
import json
import numpy as np

class PerfBench:
    def __init__(self):
        self.results = []

    def measure(self, model_id: str, batch_size: int, profile: str):
        print(f"Benchmarking {model_id} (B={batch_size}, {profile})...")
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            # Simulate inference
            time.sleep(0.01) # Mock work
            latencies.append((time.perf_counter() - start) * 1000)
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        throughput = batch_size / (p50 / 1000)
        
        result = {
            "model_id": model_id,
            "batch_size": batch_size,
            "profile": profile,
            "p50_ms": p50,
            "p95_ms": p95,
            "throughput_eps": throughput
        }
        self.results.append(result)
        return result

    def report(self):
        print("\nPerformance Report:")
        print("-" * 60)
        print(f"{'Model':<15} {'B':<5} {'Profile':<12} {'p50 (ms)':<10} {'p95 (ms)':<10} {'Tput'}")
        for r in self.results:
            print(f"{r['model_id']:<15} {r['batch_size']:<5} {r['profile']:<12} {r['p50_ms']:<10.2f} {r['p95_ms']:<10.2f} {r['throughput_eps']:.2f}")

if __name__ == "__main__":
    bench = PerfBench()
    bench.measure("toy-model", 1, "latency")
    bench.measure("toy-model", 256, "throughput")
    bench.report()
