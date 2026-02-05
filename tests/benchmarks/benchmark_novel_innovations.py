"""
Performance Benchmark Suite for Novel FHE-GBDT Innovations

Comprehensive benchmarks comparing:
1. Baseline MOAI optimizer vs Novel Innovations
2. Individual innovation performance contributions
3. End-to-end latency and throughput measurements
4. Rotation savings verification
5. Accuracy impact analysis
"""

import sys
import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.compiler.ir import ModelIR, TreeIR, TreeNode


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    elapsed_ms: float
    operations: int
    throughput: float  # ops/sec
    memory_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison between baseline and innovation."""
    metric: str
    baseline_value: float
    innovation_value: float
    improvement_percent: float
    improvement_factor: float


class BenchmarkSuite:
    """Comprehensive benchmark suite for FHE-GBDT innovations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []

    def log(self, message: str):
        if self.verbose:
            print(f"[BENCH] {message}")

    def create_test_model(
        self,
        num_trees: int = 100,
        max_depth: int = 6,
        num_features: int = 50
    ) -> ModelIR:
        """Create a synthetic GBDT model for benchmarking."""
        trees = []
        np.random.seed(42)

        for tree_idx in range(num_trees):
            nodes = {}
            node_id = 0

            # Build complete binary tree
            for depth in range(max_depth):
                num_nodes_at_depth = 2 ** depth
                for _ in range(num_nodes_at_depth):
                    if depth < max_depth - 1:
                        nodes[node_id] = TreeNode(
                            node_id=node_id,
                            feature_index=np.random.randint(0, num_features),
                            threshold=np.random.uniform(-1, 1),
                            left_child_id=2 * node_id + 1,
                            right_child_id=2 * node_id + 2,
                            depth=depth
                        )
                    else:
                        nodes[node_id] = TreeNode(
                            node_id=node_id,
                            leaf_value=np.random.uniform(-0.5, 0.5),
                            depth=depth
                        )
                    node_id += 1

            trees.append(TreeIR(
                tree_id=tree_idx,
                nodes=nodes,
                root_id=0,
                max_depth=max_depth
            ))

        return ModelIR(
            model_type="xgboost",
            trees=trees,
            num_features=num_features,
            base_score=0.5
        )

    def create_oblivious_model(
        self,
        num_trees: int = 100,
        max_depth: int = 6,
        num_features: int = 50
    ) -> ModelIR:
        """Create an oblivious (CatBoost-style) model for benchmarking."""
        trees = []
        np.random.seed(42)

        for tree_idx in range(num_trees):
            nodes = {}
            node_id = 0

            # For oblivious trees, each depth level uses the same feature
            level_features = [np.random.randint(0, num_features) for _ in range(max_depth)]
            level_thresholds = [np.random.uniform(-1, 1) for _ in range(max_depth)]

            for depth in range(max_depth):
                num_nodes_at_depth = 2 ** depth
                for _ in range(num_nodes_at_depth):
                    if depth < max_depth - 1:
                        nodes[node_id] = TreeNode(
                            node_id=node_id,
                            feature_index=level_features[depth],  # Same feature at each depth
                            threshold=level_thresholds[depth],
                            left_child_id=2 * node_id + 1,
                            right_child_id=2 * node_id + 2,
                            depth=depth
                        )
                    else:
                        nodes[node_id] = TreeNode(
                            node_id=node_id,
                            leaf_value=np.random.uniform(-0.5, 0.5),
                            depth=depth
                        )
                    node_id += 1

            trees.append(TreeIR(
                tree_id=tree_idx,
                nodes=nodes,
                root_id=0,
                max_depth=max_depth
            ))

        return ModelIR(
            model_type="catboost",
            trees=trees,
            num_features=num_features,
            base_score=0.0
        )

    def generate_sample_data(
        self,
        num_samples: int,
        num_features: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(num_samples, num_features)
        y = np.sum(X[:, :5], axis=1) + 0.1 * np.random.randn(num_samples)
        return X, y

    def benchmark_baseline_optimizer(self, model: ModelIR, iterations: int = 10) -> BenchmarkResult:
        """Benchmark baseline MOAI optimizer."""
        from services.compiler.optimizer import MOAIOptimizer

        self.log(f"Benchmarking baseline optimizer ({iterations} iterations)...")

        optimizer = MOAIOptimizer(profile="latency", target="cpu")

        start = time.perf_counter()
        for _ in range(iterations):
            plan = optimizer.optimize(model)
        elapsed = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="baseline_optimizer",
            elapsed_ms=elapsed,
            operations=iterations,
            throughput=iterations / (elapsed / 1000),
            memory_mb=0,  # Not measured
            metadata={
                "num_trees": len(model.trees),
                "rotation_savings": plan.metadata.get("rotation_savings", {}),
            }
        )

        self.results.append(result)
        self.log(f"  Baseline: {elapsed:.2f}ms for {iterations} optimizations")
        return result

    def benchmark_innovation_optimizer(self, model: ModelIR, X: np.ndarray, y: np.ndarray, iterations: int = 10) -> BenchmarkResult:
        """Benchmark optimizer with all innovations."""
        from services.compiler.optimizer import optimize_model_with_innovations

        self.log(f"Benchmarking innovation optimizer ({iterations} iterations)...")

        start = time.perf_counter()
        for _ in range(iterations):
            plan, innovation_plan = optimize_model_with_innovations(
                model, X[:100], y[:100],
                enable_all=True,
                optimize_for="latency"
            )
        elapsed = (time.perf_counter() - start) * 1000

        metadata = {
            "num_trees": len(model.trees),
        }
        if innovation_plan:
            metadata["innovations_enabled"] = [i.name for i in innovation_plan.innovations]
            metadata["rotation_savings_percent"] = innovation_plan.rotation_savings_percent
            metadata["estimated_latency_ms"] = innovation_plan.estimated_latency_ms

        result = BenchmarkResult(
            name="innovation_optimizer",
            elapsed_ms=elapsed,
            operations=iterations,
            throughput=iterations / (elapsed / 1000),
            memory_mb=0,
            metadata=metadata
        )

        self.results.append(result)
        self.log(f"  Innovations: {elapsed:.2f}ms for {iterations} optimizations")
        return result

    def benchmark_leaf_centric(self, model: ModelIR, X: np.ndarray, iterations: int = 100) -> BenchmarkResult:
        """Benchmark leaf-centric encoding."""
        from services.innovations.leaf_centric import LeafCentricEncoder

        self.log(f"Benchmarking leaf-centric encoding ({iterations} iterations)...")

        encoder = LeafCentricEncoder()
        plan = encoder.encode_model(model)

        start = time.perf_counter()
        for _ in range(iterations):
            predictions = encoder.evaluate_plaintext(plan, X[:10], model.base_score)
        elapsed = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="leaf_centric",
            elapsed_ms=elapsed,
            operations=iterations * 10,  # 10 samples per iteration
            throughput=(iterations * 10) / (elapsed / 1000),
            memory_mb=0,
            metadata={
                "total_leaves": plan.total_leaves,
                "max_depth": plan.max_depth,
            }
        )

        self.results.append(result)
        self.log(f"  Leaf-Centric: {elapsed:.2f}ms, {result.throughput:.0f} samples/sec")
        return result

    def benchmark_moai_native_conversion(self, model: ModelIR, X: np.ndarray, y: np.ndarray) -> BenchmarkResult:
        """Benchmark MOAI-native tree conversion."""
        from services.innovations.moai_native import RotationOptimalConverter

        self.log("Benchmarking MOAI-native conversion...")

        converter = RotationOptimalConverter()

        start = time.perf_counter()
        result = converter.convert_model(model, X[:50], y[:50])
        elapsed = (time.perf_counter() - start) * 1000

        bench_result = BenchmarkResult(
            name="moai_native_conversion",
            elapsed_ms=elapsed,
            operations=len(model.trees),
            throughput=len(model.trees) / (elapsed / 1000),
            memory_mb=0,
            metadata={
                "num_oblivious_trees": len(result.oblivious_trees),
                "accuracy_loss": result.accuracy_loss,
                "rotation_savings": result.rotation_savings,
            }
        )

        self.results.append(bench_result)
        self.log(f"  MOAI Conversion: {elapsed:.2f}ms, {result.rotation_savings}")
        return bench_result

    def benchmark_bootstrap_aligned(self, model: ModelIR) -> BenchmarkResult:
        """Benchmark bootstrap-aligned partitioning."""
        from services.innovations.bootstrap_aligned import BootstrapAwareTreeBuilder

        self.log("Benchmarking bootstrap-aligned partitioning...")

        builder = BootstrapAwareTreeBuilder()

        start = time.perf_counter()
        for _ in range(10):
            forest = builder.partition_into_chunks(model)
        elapsed = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="bootstrap_aligned",
            elapsed_ms=elapsed,
            operations=10,
            throughput=10 / (elapsed / 1000),
            memory_mb=0,
            metadata={
                "num_chunks": len(forest.chunks),
                "bootstrap_points": len(forest.bootstrap_points),
            }
        )

        self.results.append(result)
        self.log(f"  Bootstrap-Aligned: {elapsed:.2f}ms, {len(forest.chunks)} chunks")
        return result

    def benchmark_gradient_noise(self, model: ModelIR) -> BenchmarkResult:
        """Benchmark gradient-informed noise allocation."""
        from services.innovations.gradient_noise import GradientAwareNoiseAllocator

        self.log("Benchmarking gradient-noise allocation...")

        allocator = GradientAwareNoiseAllocator()

        start = time.perf_counter()
        for _ in range(100):
            allocations = allocator.allocate(model, model.num_features)
        elapsed = (time.perf_counter() - start) * 1000

        precisions = [a.precision_bits for a in allocations.values()]

        result = BenchmarkResult(
            name="gradient_noise",
            elapsed_ms=elapsed,
            operations=100,
            throughput=100 / (elapsed / 1000),
            memory_mb=0,
            metadata={
                "num_features": len(allocations),
                "avg_precision": np.mean(precisions),
                "precision_range": [min(precisions), max(precisions)],
            }
        )

        self.results.append(result)
        self.log(f"  Gradient-Noise: {elapsed:.2f}ms, avg precision {np.mean(precisions):.1f} bits")
        return result

    def benchmark_homomorphic_pruning(self, num_trees: int = 100) -> BenchmarkResult:
        """Benchmark homomorphic pruning."""
        from services.innovations.homomorphic_pruning import HomomorphicEnsemblePruner

        self.log("Benchmarking homomorphic pruning...")

        pruner = HomomorphicEnsemblePruner()
        tree_outputs = np.random.randn(100, num_trees)

        start = time.perf_counter()
        for _ in range(100):
            aggregated, metadata = pruner.prune_plaintext(tree_outputs)
        elapsed = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="homomorphic_pruning",
            elapsed_ms=elapsed,
            operations=100 * 100,  # 100 iterations * 100 samples
            throughput=(100 * 100) / (elapsed / 1000),
            memory_mb=0,
            metadata={
                "num_trees": num_trees,
                "num_active_trees": metadata.get("num_active_trees", 0),
                "pruning_ratio": metadata.get("pruning_ratio", 0),
            }
        )

        self.results.append(result)
        self.log(f"  Homomorphic Pruning: {elapsed:.2f}ms, {metadata.get('pruning_ratio', 0)*100:.1f}% pruned")
        return result

    def benchmark_unified_engine(self, model: ModelIR, X: np.ndarray, y: np.ndarray) -> BenchmarkResult:
        """Benchmark unified innovation engine."""
        from services.innovations.unified_architecture import NovelFHEGBDTEngine

        self.log("Benchmarking unified engine...")

        engine = NovelFHEGBDTEngine()
        plan = engine.create_execution_plan(model, X[:50], y[:50])

        start = time.perf_counter()
        for _ in range(100):
            predictions = engine.predict(X[:10])
        elapsed = (time.perf_counter() - start) * 1000

        report = engine.get_optimization_report()

        result = BenchmarkResult(
            name="unified_engine",
            elapsed_ms=elapsed,
            operations=100 * 10,
            throughput=(100 * 10) / (elapsed / 1000),
            memory_mb=0,
            metadata={
                "innovations_enabled": report.get("innovations_enabled", []),
                "rotation_savings_percent": report.get("performance", {}).get("rotation_savings_percent", 0),
                "estimated_latency_ms": plan.estimated_latency_ms,
            }
        )

        self.results.append(result)
        self.log(f"  Unified Engine: {elapsed:.2f}ms, {result.throughput:.0f} samples/sec")
        return result

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("\n" + "=" * 70)
        print("FHE-GBDT NOVEL INNOVATIONS - PERFORMANCE BENCHMARK SUITE")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print()

        # Create test models
        self.log("Creating test models...")
        model_small = self.create_test_model(num_trees=20, max_depth=4, num_features=10)
        model_medium = self.create_test_model(num_trees=100, max_depth=6, num_features=50)
        model_large = self.create_test_model(num_trees=500, max_depth=8, num_features=100)
        model_oblivious = self.create_oblivious_model(num_trees=100, max_depth=6, num_features=50)

        X, y = self.generate_sample_data(1000, 100)

        print("\n" + "-" * 70)
        print("BENCHMARK 1: Optimizer Comparison (Medium Model: 100 trees, depth 6)")
        print("-" * 70)

        baseline = self.benchmark_baseline_optimizer(model_medium)
        innovation = self.benchmark_innovation_optimizer(model_medium, X, y)

        opt_speedup = baseline.elapsed_ms / innovation.elapsed_ms if innovation.elapsed_ms > 0 else 1
        self.log(f"  Optimization speedup: {opt_speedup:.2f}x")

        print("\n" + "-" * 70)
        print("BENCHMARK 2: Individual Innovation Performance")
        print("-" * 70)

        self.benchmark_leaf_centric(model_medium, X)
        self.benchmark_moai_native_conversion(model_medium, X, y)
        self.benchmark_bootstrap_aligned(model_medium)
        self.benchmark_gradient_noise(model_medium)
        self.benchmark_homomorphic_pruning(100)
        self.benchmark_unified_engine(model_medium, X, y)

        print("\n" + "-" * 70)
        print("BENCHMARK 3: Scaling Analysis")
        print("-" * 70)

        self.log("Small model (20 trees):")
        self.benchmark_baseline_optimizer(model_small, iterations=50)

        self.log("Medium model (100 trees):")
        self.benchmark_baseline_optimizer(model_medium, iterations=20)

        self.log("Large model (500 trees):")
        self.benchmark_baseline_optimizer(model_large, iterations=5)

        print("\n" + "-" * 70)
        print("BENCHMARK 4: Oblivious vs Non-Oblivious")
        print("-" * 70)

        from services.innovations.moai_native import RotationOptimalConverter
        converter = RotationOptimalConverter()

        self.log("Converting non-oblivious to oblivious:")
        conversion_result = converter.convert_model(model_medium, X[:50], y[:50])
        self.log(f"  Rotation savings: {conversion_result.rotation_savings}")
        self.log(f"  Accuracy loss: {conversion_result.accuracy_loss:.4f}")

        # Compute comparisons
        self._compute_comparisons(model_medium, baseline, innovation)

        # Generate report
        return self._generate_report()

    def _compute_comparisons(
        self,
        model: ModelIR,
        baseline: BenchmarkResult,
        innovation: BenchmarkResult
    ):
        """Compute comparison metrics."""
        # Rotation savings comparison
        baseline_rotations = sum(len(t.nodes) for t in model.trees)
        innovation_rotations = innovation.metadata.get("rotation_savings_percent", 0)

        self.comparisons.append(ComparisonResult(
            metric="Optimization Time",
            baseline_value=baseline.elapsed_ms,
            innovation_value=innovation.elapsed_ms,
            improvement_percent=((baseline.elapsed_ms - innovation.elapsed_ms) / baseline.elapsed_ms) * 100 if baseline.elapsed_ms > 0 else 0,
            improvement_factor=baseline.elapsed_ms / innovation.elapsed_ms if innovation.elapsed_ms > 0 else 1,
        ))

        self.comparisons.append(ComparisonResult(
            metric="Baseline Rotations",
            baseline_value=baseline_rotations,
            innovation_value=baseline_rotations * (1 - innovation_rotations / 100),
            improvement_percent=innovation_rotations,
            improvement_factor=100 / (100 - innovation_rotations) if innovation_rotations < 100 else float('inf'),
        ))

    def _generate_report(self) -> Dict[str, Any]:
        """Generate final benchmark report."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)

        report = {
            "timestamp": datetime.now().isoformat(),
            "results": [],
            "comparisons": [],
            "summary": {},
        }

        # Results table
        print("\nBenchmark Results:")
        print(f"{'Name':<30} {'Time (ms)':<12} {'Ops':<10} {'Throughput':<15}")
        print("-" * 70)

        for result in self.results:
            print(f"{result.name:<30} {result.elapsed_ms:<12.2f} {result.operations:<10} {result.throughput:<15.0f}")
            report["results"].append({
                "name": result.name,
                "elapsed_ms": result.elapsed_ms,
                "operations": result.operations,
                "throughput": result.throughput,
                "metadata": result.metadata,
            })

        # Comparisons table
        if self.comparisons:
            print("\nPerformance Comparisons:")
            print(f"{'Metric':<25} {'Baseline':<15} {'Innovation':<15} {'Improvement':<15}")
            print("-" * 70)

            for comp in self.comparisons:
                print(f"{comp.metric:<25} {comp.baseline_value:<15.2f} {comp.innovation_value:<15.2f} {comp.improvement_percent:>+.1f}%")
                report["comparisons"].append({
                    "metric": comp.metric,
                    "baseline": comp.baseline_value,
                    "innovation": comp.innovation_value,
                    "improvement_percent": comp.improvement_percent,
                    "improvement_factor": comp.improvement_factor,
                })

        # Summary
        leaf_centric_result = next((r for r in self.results if r.name == "leaf_centric"), None)
        unified_result = next((r for r in self.results if r.name == "unified_engine"), None)
        moai_result = next((r for r in self.results if r.name == "moai_native_conversion"), None)

        summary = {
            "total_benchmarks": len(self.results),
            "innovations_tested": len([r for r in self.results if r.name not in ["baseline_optimizer"]]),
        }

        if leaf_centric_result:
            summary["leaf_centric_throughput"] = leaf_centric_result.throughput

        if unified_result:
            summary["unified_engine_throughput"] = unified_result.throughput
            summary["rotation_savings_percent"] = unified_result.metadata.get("rotation_savings_percent", 0)

        if moai_result:
            summary["moai_rotation_savings"] = moai_result.metadata.get("rotation_savings", {})

        report["summary"] = summary

        print("\n" + "=" * 70)
        print("KEY FINDINGS:")
        print("=" * 70)

        if moai_result and "rotation_savings" in moai_result.metadata:
            savings = moai_result.metadata["rotation_savings"]
            print(f"✓ MOAI-Native Conversion: {savings.get('savings_percent', 0):.1f}% rotation savings")
            print(f"  - Baseline rotations: {savings.get('original_rotations', 'N/A')}")
            print(f"  - Oblivious rotations: {savings.get('oblivious_rotations', 'N/A')}")

        if unified_result:
            print(f"✓ Unified Engine Throughput: {unified_result.throughput:.0f} samples/sec")

        if leaf_centric_result:
            print(f"✓ Leaf-Centric Throughput: {leaf_centric_result.throughput:.0f} samples/sec")

        print("\n" + "=" * 70)
        print(f"Completed: {datetime.now().isoformat()}")
        print("=" * 70 + "\n")

        return report


def run_regression_tests() -> bool:
    """Run regression tests to verify correctness."""
    print("\n" + "=" * 70)
    print("REGRESSION TESTS - Verifying Correctness")
    print("=" * 70)

    all_passed = True
    test_results = []

    # Test 1: Leaf-centric produces valid predictions
    print("\n[TEST 1] Leaf-Centric Encoding Correctness")
    try:
        from services.innovations.leaf_centric import LeafCentricEncoder, LeafIndicatorComputer

        computer = LeafIndicatorComputer()
        signs = np.array([[0.2, 0.8], [0.9, 0.1]])
        indicators = computer.compute_leaf_indicators_tensor(signs, depth=2)

        # Indicators should sum to 1
        sums = indicators.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=0.01), f"Indicators don't sum to 1: {sums}"
        print("  ✓ Leaf indicators sum to 1")
        test_results.append(("Leaf-Centric", True, "Indicators sum to 1"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results.append(("Leaf-Centric", False, str(e)))
        all_passed = False

    # Test 2: Gradient-noise allocation produces valid allocations
    print("\n[TEST 2] Gradient-Noise Allocation Correctness")
    try:
        from services.innovations.gradient_noise import GradientAwareNoiseAllocator
        from services.compiler.ir import ModelIR, TreeIR, TreeNode

        # Create simple model
        nodes = {
            0: TreeNode(node_id=0, feature_index=0, threshold=0.5, left_child_id=1, right_child_id=2, depth=0),
            1: TreeNode(node_id=1, leaf_value=-0.3, depth=1),
            2: TreeNode(node_id=2, leaf_value=0.3, depth=1),
        }
        tree = TreeIR(tree_id=0, nodes=nodes, root_id=0, max_depth=2)
        model = ModelIR(model_type="xgboost", trees=[tree], num_features=4, base_score=0.5)

        allocator = GradientAwareNoiseAllocator()
        allocations = allocator.allocate(model, 4)

        assert len(allocations) == 4, f"Expected 4 allocations, got {len(allocations)}"
        for idx, alloc in allocations.items():
            assert 8 <= alloc.precision_bits <= 16, f"Precision out of range: {alloc.precision_bits}"
        print("  ✓ All allocations within valid range")
        test_results.append(("Gradient-Noise", True, "Valid allocations"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results.append(("Gradient-Noise", False, str(e)))
        all_passed = False

    # Test 3: MOAI-native conversion preserves structure
    print("\n[TEST 3] MOAI-Native Conversion Correctness")
    try:
        from services.innovations.moai_native import RotationOptimalConverter, ObliviousTree

        converter = RotationOptimalConverter()

        # Create test model
        nodes = {
            0: TreeNode(node_id=0, feature_index=0, threshold=0.5, left_child_id=1, right_child_id=2, depth=0),
            1: TreeNode(node_id=1, feature_index=1, threshold=0.3, left_child_id=3, right_child_id=4, depth=1),
            2: TreeNode(node_id=2, feature_index=2, threshold=0.7, left_child_id=5, right_child_id=6, depth=1),
            3: TreeNode(node_id=3, leaf_value=-0.5, depth=2),
            4: TreeNode(node_id=4, leaf_value=-0.1, depth=2),
            5: TreeNode(node_id=5, leaf_value=0.1, depth=2),
            6: TreeNode(node_id=6, leaf_value=0.5, depth=2),
        }
        tree = TreeIR(tree_id=0, nodes=nodes, root_id=0, max_depth=3)
        model = ModelIR(model_type="xgboost", trees=[tree], num_features=4, base_score=0.0)

        result = converter.convert_model(model)

        assert len(result.oblivious_trees) == 1, f"Expected 1 tree, got {len(result.oblivious_trees)}"
        assert result.oblivious_trees[0].max_depth == 3, "Depth not preserved"
        print("  ✓ Tree structure preserved after conversion")
        test_results.append(("MOAI-Native", True, "Structure preserved"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results.append(("MOAI-Native", False, str(e)))
        all_passed = False

    # Test 4: Bootstrap-aligned produces valid chunks
    print("\n[TEST 4] Bootstrap-Aligned Correctness")
    try:
        from services.innovations.bootstrap_aligned import BootstrapAwareTreeBuilder

        builder = BootstrapAwareTreeBuilder()

        # Create model with multiple trees
        trees = []
        for i in range(10):
            nodes = {
                0: TreeNode(node_id=0, feature_index=0, threshold=0.5, left_child_id=1, right_child_id=2, depth=0),
                1: TreeNode(node_id=1, leaf_value=-0.3, depth=1),
                2: TreeNode(node_id=2, leaf_value=0.3, depth=1),
            }
            trees.append(TreeIR(tree_id=i, nodes=nodes, root_id=0, max_depth=2))

        model = ModelIR(model_type="xgboost", trees=trees, num_features=4, base_score=0.5)

        forest = builder.partition_into_chunks(model)

        total_trees = sum(len(c.trees) for c in forest.chunks)
        assert total_trees == 10, f"Expected 10 trees, got {total_trees}"
        print("  ✓ All trees accounted for in chunks")
        test_results.append(("Bootstrap-Aligned", True, "Trees preserved"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results.append(("Bootstrap-Aligned", False, str(e)))
        all_passed = False

    # Test 5: Unified engine produces predictions
    print("\n[TEST 5] Unified Engine Correctness")
    try:
        from services.innovations.unified_architecture import NovelFHEGBDTEngine

        engine = NovelFHEGBDTEngine()

        # Create test model
        nodes = {
            0: TreeNode(node_id=0, feature_index=0, threshold=0.5, left_child_id=1, right_child_id=2, depth=0),
            1: TreeNode(node_id=1, leaf_value=-0.3, depth=1),
            2: TreeNode(node_id=2, leaf_value=0.3, depth=1),
        }
        tree = TreeIR(tree_id=0, nodes=nodes, root_id=0, max_depth=2)
        model = ModelIR(model_type="xgboost", trees=[tree], num_features=4, base_score=0.5)

        np.random.seed(42)
        X = np.random.randn(10, 4)
        y = np.sum(X[:, :2], axis=1)

        engine.create_execution_plan(model, X, y)
        predictions = engine.predict(X)

        assert predictions.shape == (10,), f"Expected shape (10,), got {predictions.shape}"
        assert np.all(np.isfinite(predictions)), "Predictions contain NaN/Inf"
        print("  ✓ Valid predictions produced")
        test_results.append(("Unified Engine", True, "Valid predictions"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results.append(("Unified Engine", False, str(e)))
        all_passed = False

    # Summary
    print("\n" + "-" * 70)
    print("REGRESSION TEST SUMMARY")
    print("-" * 70)
    passed = sum(1 for _, p, _ in test_results if p)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")

    for name, passed, msg in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name} - {msg}")

    return all_passed


if __name__ == "__main__":
    # Run regression tests first
    regression_passed = run_regression_tests()

    if not regression_passed:
        print("\n⚠ WARNING: Some regression tests failed!")
        print("Proceeding with benchmarks anyway...\n")

    # Run benchmarks
    suite = BenchmarkSuite(verbose=True)
    report = suite.run_full_benchmark()

    # Save report to file
    report_file = "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_file}")
