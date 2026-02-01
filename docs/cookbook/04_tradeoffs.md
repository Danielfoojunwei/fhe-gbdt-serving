# Trade-offs & Performance ⚖️

This document compares the characteristics of the three supported libraries in the context of FHE execution.

## Feature Comparison Matrix

| Feature | XGBoost | LightGBM | CatBoost |
| :--- | :--- | :--- | :--- |
| **Tree Structure** | Irregular (Greedy) | Irregular (Leaf-wise) | Symmetric (Oblivious) |
| **Model Format** | JSON | Text | JSON |
| **FHE Suitability** | Good | Moderate | **Excellent** |
| **Compilation** | Requires Levelization | Requires Levelization | Direct Mapping |
| **Rotations** | Moderate | High | **Low** |
| **Scheme Switches** | Dependent on Depth | Dependent on Depth | Minimal |

## Why CatBoost?
CatBoost's symmetric trees allow the FHE compiler to batch comparison operations ("Steps") more effectively. In an oblivious tree, all nodes at depth $d$ share the same splitting feature (or can be arranged so). This allows us to perform a single encrypted comparison for the entire level, rather than individual comparisons for each node.

## Empirical Benchmarks
*Results from `bench/reports/cookbook/combined.md`*

### Latency (Batch=1)
- **CatBoost**: ~8ms (Normalized: 1.0x)
- **XGBoost**: ~11ms (Normalized: 1.4x)
- **LightGBM**: ~14ms (Normalized: 1.7x)

*Note: Actual numbers depend on model depth/size.*

### Throughput (Batch=256)
- **CatBoost**: ~600 eps
- **XGBoost**: ~500 eps
- **LightGBM**: ~450 eps

## Recommendations

- **Use CatBoost** if you have full control over the training pipeline and want maximum FHE performance.
- **Use XGBoost** if you need compatibility with existing pipelines; the performance penalty is manageable.
- **Use LightGBM** for large datasets where training speed is critical, accepting slightly slower inference.
