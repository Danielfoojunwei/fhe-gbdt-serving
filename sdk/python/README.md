# FHE-GBDT Python SDK

A Python SDK for the FHE-GBDT (Fully Homomorphic Encryption - Gradient Boosted Decision Trees) Inference Service.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from fhe_gbdt_sdk import FHEGBDTClient

# Initialize client
client = FHEGBDTClient("localhost:8080", "your-tenant-id")

# Run encrypted inference
features = [{"feature_0": 1.5, "feature_1": 2.3}]
result = client.predict_encrypted("compiled-model-id", features)
print(f"Encrypted prediction: {result}")
```

## Features

- **End-to-End Encryption**: Features are encrypted client-side using RLWE-compatible crypto
- **gRPC Communication**: High-performance binary protocol
- **Tenant Isolation**: Multi-tenant support with API key authentication
- **Simulation Mode**: Fallback for testing without a live backend
