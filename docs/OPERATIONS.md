# Operations Manual

This document is the **canonical operations guide** for running the current implementation in this repository.

## 1) Deployment Modes

- **Development**: Docker Compose
- **Production**: Kubernetes/Helm with externalized storage and secrets

## 2) Required Runtime Dependencies

### Gateway
- Runtime gRPC endpoint must be reachable.
- Registry control endpoint must be reachable.
- If runtime is disconnected, predict now fails closed (`Unavailable`) rather than simulating.

### Keystore
- `KEYSTORE_STORAGE_DIR` (default: `./data/keystore`) for encrypted eval-key records.
- Optional Vault transit integration via `VAULT_ADDR` and `VAULT_TOKEN`.

### Registry
- `REGISTRY_STORAGE_DIR` (default: `./data/registry`) for persisted model binaries/plans.
- PostgreSQL is recommended for metadata persistence.

## 3) Health and Readiness Semantics

- `/health`: process liveness.
- `/ready`: dependency readiness.
  - Returns `200` only when gateway + registry client + runtime client are initialized.
  - Returns `503` otherwise.

## 4) Inference Path (Canonical)

1. Client encrypts features.
2. REST `/v1/predict` accepts base64 ciphertext and forwards to gateway gRPC predict.
3. Gateway validates auth/license/ownership and forwards to runtime.
4. Runtime executes inference and returns encrypted outputs.
5. REST returns base64 ciphertext output and runtime stats.

> No production fallback simulation should be relied upon in gateway or REST paths.

## 5) Production Guardrails

- Keep SDK simulation disabled (`ALLOW_SDK_SIMULATION=false`, default).
- Treat local KEK mode in keystore as non-production fallback only.
- Ensure storage directories are mounted on durable volumes with backup policy.
- Alert on repeated `runtime unavailable` errors.

## 6) Canonical Validation Commands

```bash
go build ./...
go test ./...
python -m pytest tests/unit/test_no_plaintext_logs.py -q
```
