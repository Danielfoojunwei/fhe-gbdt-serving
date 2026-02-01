# FHE-GBDT Serving

Production-grade Fully Homomorphic Encryption (FHE) inference API for Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost).

## Features
- **Privacy-First**: No plaintext features ever leave the client.
- **Universal**: Supports XGBoost, LightGBM, and CatBoost.
- **Performance**: Optimized C++ runtime with N2HE-HEXL primitives.
- **Production Ready**: Security hardened, observable (OTel/Prometheus), and scalable (K8s/Helm).

## Quickstart

```bash
# Start the stack
make docker-up

# Run the Cookbook (E2E Recipe)
make cookbook
```

## Cookbook (End-to-End Examples)
Check out [docs/cookbook/README.md](docs/cookbook/README.md) for detailed recipes:
- 🚀 [Quickstart](docs/cookbook/00_quickstart.md)
- 🌲 [XGBoost](docs/cookbook/01_xgboost_classification.md)
- 🍃 [LightGBM](docs/cookbook/02_lightgbm_regression.md)
- 🐱 [CatBoost](docs/cookbook/03_catboost_classification.md)

## Development

```bash
# Build
make build

# Test
make test

# Run E2E Tests with Real Models
make e2e-real
```

## Correctness & Performance
We validate FHE correctness against plaintext baselines with strict tolerances.
Performance is measured in **Encrypted Prediction/s (EPS)**.
See [Performance Case Study](bench/dashboard.html) for latest metrics.

## Troubleshooting
See [docs/cookbook/05_troubleshooting.md](docs/cookbook/05_troubleshooting.md).
