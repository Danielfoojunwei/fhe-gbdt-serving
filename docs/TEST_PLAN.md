# Test Plan

## Unit Tests
- Compiler: IR parsing and optimization.
- Runtime: Kernel logic (Step, Add, Mul).
- Proto: Serialization/Deserialization.

## Integration Tests
- End-to-end inference flow using toy datasets.

## Benchmarks
- Latency vs Throughput profiles.
- Scaling with batch size (B=1 to B=256).
