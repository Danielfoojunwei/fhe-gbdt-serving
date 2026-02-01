# Benchmark Environment Contract

This document defines the canonical benchmark environment for reproducible results.

## Reference Machine Class

| Component       | Specification                          |
|-----------------|----------------------------------------|
| CPU             | Intel Xeon Gold 6248 (20 cores, 2.5GHz)|
| RAM             | 256GB DDR4-2933                        |
| GPU (optional)  | NVIDIA A100 40GB                       |
| OS              | Ubuntu 22.04 LTS                       |
| Kernel          | 5.15.0-generic                         |
| Docker          | 24.0.x                                 |
| CUDA (if GPU)   | 12.2                                   |

## Compiler Versions

| Tool       | Version   |
|------------|-----------|
| GCC        | 11.4.0    |
| Go         | 1.21.x    |
| Python     | 3.10.x    |
| CMake      | 3.22+     |

## Environment Hash

To ensure reproducibility, compute an environment hash:
```bash
echo "$(uname -a) $(gcc --version | head -1) $(go version)" | sha256sum | cut -c1-16
```

Store this hash alongside benchmark results.

## Benchmark Execution Settings

- **Warm-up runs**: 3
- **Measured runs**: 10
- **Confidence level**: 95%
- **Power settings**: Performance governor, turbo enabled
- **Isolation**: No other workloads, CPU pinning for critical threads

## Dataset & Model Pinning

- Use fixed random seeds for dataset slicing: `SEED=42`
- Use fixed random seeds for model training: `SEED=42`
- Reference models stored in `bench/models/`

## Baseline Update Policy

- Baselines stored in `bench/baselines.json`
- Updates require explicit PR approval
- CI will fail if results drift > 10% without approval
