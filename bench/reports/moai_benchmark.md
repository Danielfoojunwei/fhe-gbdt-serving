# MOAI Benchmark Report

**Timestamp**: 2026-02-08T17:19:25.537656
**Reference**: MOAI: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference (DTC NTU, IACR ePrint 2025/991)

## Summary

- **Benchmarks Run**: 11
- **Avg Rotation Reduction**: 99.5%
- **Avg Speedup**: 6.16x
- **Max Speedup**: 8.31x
- **Total Rotations Saved**: 2,342,269
- **MOAI Advantage**: Significant

## Detailed Results

| Benchmark | Trees | Depth | Features | Batch | Trad Rot | MOAI Rot | Reduction | Speedup |
|-----------|-------|-------|----------|-------|----------|----------|-----------|----------|
| Small-GBDT | 10 | 4 | 32 | 1 | 150 | 4 | 97.3% | 4.78x |
| Small-GBDT-Batch | 10 | 4 | 32 | 256 | 150 | 4 | 97.3% | 6.35x |
| Medium-GBDT | 100 | 6 | 128 | 1 | 6,300 | 7 | 99.9% | 5.00x |
| Medium-GBDT-Batch | 100 | 6 | 128 | 256 | 6,300 | 7 | 99.9% | 6.66x |
| Large-GBDT | 500 | 8 | 256 | 1 | 127,500 | 9 | 100.0% | 5.01x |
| Large-GBDT-Batch | 500 | 8 | 256 | 256 | 127,500 | 9 | 100.0% | 6.67x |
| XL-GBDT | 1000 | 10 | 512 | 1 | 1,023,000 | 10 | 100.0% | 5.01x |
| XL-GBDT-Batch | 1000 | 10 | 512 | 256 | 1,023,000 | 10 | 100.0% | 6.67x |
| Fraud-Detection | 200 | 6 | 50 | 1000 | 12,600 | 8 | 99.9% | 8.31x |
| Credit-Scoring | 100 | 5 | 30 | 500 | 3,100 | 7 | 99.8% | 7.45x |
| Medical-Diagnosis | 50 | 8 | 100 | 100 | 12,750 | 6 | 100.0% | 5.81x |

## Timing Breakdown

| Benchmark | Traditional (ms) | MOAI Total (ms) | Comparison (ms) | Aggregation (ms) |
|-----------|------------------|-----------------|-----------------|------------------|
| Small-GBDT | 226.78 | 47.49 | 45.00 | 2.40 |
| Small-GBDT-Batch | 451.80 | 71.10 | 45.00 | 2.40 |
| Medium-GBDT | 9496.85 | 1897.90 | 1890.00 | 4.20 |
| Medium-GBDT-Batch | 18919.80 | 2841.30 | 1890.00 | 4.20 |
| Large-GBDT | 192047.17 | 38330.12 | 38250.00 | 5.40 |
| Large-GBDT-Batch | 382599.80 | 57383.10 | 38250.00 | 5.40 |
| XL-GBDT | 1540594.43 | 307505.43 | 306900.00 | 6.00 |
| XL-GBDT-Batch | 3069199.80 | 460359.00 | 306900.00 | 6.00 |
| Fraud-Detection | 92825.76 | 11176.99 | 3780.00 | 4.80 |
| Credit-Scoring | 13761.27 | 1846.50 | 930.00 | 4.20 |
| Medical-Diagnosis | 26602.52 | 4576.37 | 3825.00 | 3.60 |

## Throughput Comparison

| Benchmark | Traditional (EPS) | MOAI (EPS) | Improvement |
|-----------|-------------------|------------|-------------|
| Small-GBDT | 4.4 | 21.1 | +377.5% |
| Small-GBDT-Batch | 566.6 | 3600.6 | +535.4% |
| Medium-GBDT | 0.1 | 0.5 | +-47.3% |
| Medium-GBDT-Batch | 13.5 | 90.1 | +565.9% |
| Large-GBDT | 0.0 | 0.0 | +-97.4% |
| Large-GBDT-Batch | 0.7 | 4.5 | +346.1% |
| XL-GBDT | 0.0 | 0.0 | +-99.7% |
| XL-GBDT-Batch | 0.1 | 0.6 | +-44.4% |
| Fraud-Detection | 10.8 | 89.5 | +730.5% |
| Credit-Scoring | 36.3 | 270.8 | +645.3% |
| Medical-Diagnosis | 3.8 | 21.9 | +481.3% |

## Key Insights

1. **Rotation Elimination**: Column packing eliminates ALL comparison rotations
2. **Consistent Packing**: No format conversions between tree levels
3. **Log-Reduction**: Interleaved aggregation reduces tree sum from O(n) to O(log n)
4. **Scalability**: Benefits increase with model size

## MOAI Paper Reference

```
MOAI: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference
Authors: Linru Zhang, Xiangning Wang, Jun Jie Sim, et al.
Affiliation: Digital Trust Centre, NTU Singapore
Publication: IACR ePrint 2025/991, NDSS 2025
GitHub: https://github.com/dtc2025ag/MOAI_GPU
```
