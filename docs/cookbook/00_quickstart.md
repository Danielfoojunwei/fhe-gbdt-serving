# Quickstart 🚀

This recipe demonstrates the canonical flow for using FHE-GBDT: starting the stack, training a model, registering it, and running encrypted inference.

## Prerequisites
- Docker & Docker Compose
- Python 3.10+
- `make` utility

## Step 1: Start the Stack
Start the local development stack (Gateway, Registry, Keystore, Runtime).
```bash
make docker-up
```
*Wait until all services are healthy (approx 30s).*

## Step 2: Train & Register a Model
We'll use a pre-built script to train a small XGBoost model and register it.
```bash
python bench/cookbook/run_recipe_xgboost.py --quick
```
This script performs the following:
1. Trains `XGBClassifier` on scikit-learn's breast cancer dataset.
2. Exports the model to `model.json`.
3. Uploads the model to the Registry (`POST /v1/models`).
4. Compiles the model for `latency` profile (`POST /v1/models/{id}/compile`).

## Step 3: Client Setup & Key Generation
The script acts as the client:
1. Generates FHE secret/eval keys using the SDK.
2. Uploads evaluation keys to the Keystore (`POST /v1/crypto/...`).

## Step 4: Encrypted Prediction
The client executes the prediction flow:
1. Encrypts a feature vector.
2. Sends the ciphertext to the Gateway (`GRPC Predict`).
3. Receives the encrypted result.
4. Decrypts the result locally.

## Expected Output
```text
[Client] Model uploaded: xgb-quickstart
[Client] Compilation complete. Plan ID: a1b2...
[Client] Keys generated and uploaded.
[Client] Encrypting input vector...
[Client] Sending prediction request...
[Client] Received encrypted result. Decrypting...
[Result] Decrypted Logit: 1.2534 (Class: Malignant)
[Verify] Plaintext Logit: 1.2534 (Diff: 0.0000)
```

## Next Steps
- Try the [XGBoost Recipe](01_xgboost_classification.md) for a deep dive.
- Check [Trade-offs](04_tradeoffs.md) to understand performance implications.
