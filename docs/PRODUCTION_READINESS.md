# Production Readiness Checklist

This checklist is the canonical release gate for this repository.

## Security Gates
- [x] mTLS configured for service-to-service communication.
- [x] Tenant/model ownership authorization enforced.
- [x] Plaintext/ciphertext payload logging guardrails in place and tested.
- [x] Keystore eval-key encryption at rest implemented.
- [x] SBOM generation integrated in CI.

## Correctness Gates
- [x] REST `/v1/predict` forwards to real gRPC inference path (no mock response).
- [x] Gateway returns explicit failure when runtime is disconnected (no simulation fallback).
- [x] Keystore supports encrypted eval-key upload and retrieval/decryption.
- [x] Registry persists model binaries to durable path before metadata registration.
- [x] Readiness endpoint reflects dependency state, not constants.

## Reliability Gates
- [x] `go build ./...` passes.
- [x] `go test ./...` passes.
- [x] Unit guardrail test `test_no_plaintext_logs.py` passes.
- [ ] Full integration/E2E suites passing in network-enabled CI environment.

## Operability Gates
- [x] Runbooks include runtime, keystore, and registry incident handling.
- [x] Operations manual documents storage dirs and readiness semantics.
- [x] No-mock gap review tracked in `docs/PRODUCTION_NO_MOCKS_GAP_REVIEW.md`.

---

## Required Environment Variables

### Gateway
- `LICENSE_SIGNING_KEY` (recommended in production)
- `MTLS_CERT_FILE`, `MTLS_KEY_FILE`, `MTLS_CA_FILE` (recommended)

### Keystore
- `KEYSTORE_STORAGE_DIR` (durable volume path)
- `VAULT_ADDR`, `VAULT_TOKEN` (optional but recommended)

### Registry
- `REGISTRY_STORAGE_DIR` (durable volume path)
- DB connection environment for Postgres metadata store

### SDK
- `ALLOW_SDK_SIMULATION=false` in production

---

**Last Updated**: 2026-02-28
**Release Status**: Conditional (pending full E2E in unrestricted CI)
