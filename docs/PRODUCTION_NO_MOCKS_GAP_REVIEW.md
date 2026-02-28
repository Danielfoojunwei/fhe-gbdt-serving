# Production Gap Review (No Mocks / No Simulations / No Fakes)

## Scope and Method

This review was performed directly against the current repository code and runtime entry points, with emphasis on production-path services (`gateway`, `registry`, `keystore`, SDK crypto/client paths, and FHE backend integration).

### Checks executed

```bash
go build ./...
go test ./...
python -m pytest tests/unit/test_no_plaintext_logs.py -q
rg -n "simulate|simulated|mock|fake|TODO|FIXME|NotImplementedError|stub|placeholder" services sdk
```

## Executive Finding

The system compiles and baseline guardrail tests pass. Several critical no-mock gaps were remediated; remaining work is now concentrated in advanced/optional paths (for example batch APIs, broader simulation controls in non-gateway components, and full conformance testing).

---

## Critical Gaps (Blockers)

### 1) Gateway REST predict path returns mock output instead of forwarding to runtime

- `handlePredict` contains `TODO: Forward to gRPC gateway` and returns a hard-coded ciphertext/stat payload.
- This is a direct functional blocker for real encrypted inference through REST APIs.

**Evidence:** `services/gateway/rest_api.go` lines 302-318.

### 2) Gateway gRPC path silently falls back to simulation when runtime is unavailable

- If runtime is not connected, gateway sleeps 12ms and echoes request payload as "prediction output".
- This can mask outages and produce false positives in operational monitoring.

**Evidence:** `services/gateway/server.go` lines 97-115.

### 3) Keystore persistence/retrieval is incomplete

- Upload path encrypts but does not persist ciphertext (`TODO: Store ciphertext in MinIO/Postgres`).
- Retrieval path is unimplemented and returns not found.

**Evidence:** `services/keystore/server.go` lines 62-63 and 82-85.

### 4) Registry model artifact handling uses placeholder path and fallback memory store

- Register model path records a content path placeholder with TODO for object storage upload.
- In-memory fallback for errors creates durability and consistency risk.

**Evidence:** `services/registry/server.go` lines 61-74.

---

## High Gaps

### 5) SDK client falls back to simulated backend processing

- Python SDK catches prediction exceptions and simulates backend processing by local decrypt loopback.
- This can hide production integration failures from client applications.

**Evidence:** `sdk/python/client.py` lines 54-65.

### 6) SDK crypto defaults to simulated key/encryption path when native library is unavailable

- Keygen/export/encrypt/decrypt all route through simulated logic when native backend is absent.
- This is valid for dev/test but not for production assurance claims.

**Evidence:** `sdk/python/crypto.py` lines 65-110 and 131-167.

### 7) Readiness endpoint reports ready=true without dependency checks

- `/ready` currently hardcodes readiness true and does not verify gateway/runtime/registry dependencies.

**Evidence:** `services/gateway/rest_api.go` lines 260-274.

---

## Medium Gaps

### 8) Batch prediction REST endpoint not implemented

- Endpoint returns HTTP 501 and blocks production batch workloads.

**Evidence:** `services/gateway/rest_api.go` lines 321-327.

### 9) Concrete backend exposes simulation path that can be selected by config

- `predict_encrypted` can execute in `fhe="simulate"` mode depending on configuration.
- Requires strict production config controls to prevent accidental non-FHE operation.

**Evidence:** `services/fhe/concrete_backend.py` lines 309-319 and 349-362.

---


## Remediation Implemented in Current Revision

The following production-path gaps were closed in this revision:

1. Gateway gRPC no longer uses simulation fallback when runtime is disconnected (now returns `codes.Unavailable`).
2. REST `/v1/predict` now forwards requests to gRPC inference instead of returning a mock response.
3. REST `/ready` now checks core dependency initialization (gateway + registry client + runtime client).
4. Keystore now persists encrypted eval keys and supports retrieval/decryption round-trip.
5. Registry now persists model binary content to local durable storage path before metadata registration.
6. Python SDK now disables simulation fallback by default; simulation is opt-in via `ALLOW_SDK_SIMULATION=true`.

## What Is Working (Verified)

1. Go services compile and package-level tests execute successfully.
2. Unit guardrail test for plaintext log safety passes.
3. CI structure includes security checks and SBOM generation (from workflow definition).

---

## No-Mock Production Action Plan

### Phase 0 (Immediate, 1-2 weeks)

- Remove/feature-flag-disable simulation fallbacks in `gateway` and `sdk` for production builds.
- Change runtime-unavailable behavior to hard fail with explicit error code and alert signal.
- Implement real dependency checks in readiness endpoint.

### Phase 1 (2-4 weeks)

- Implement registry object storage upload path and checksum verification.
- Implement keystore encrypted key persistence + retrieval round-trip with integration tests.
- Replace REST `handlePredict` mock payload with real forwarding to gRPC inference path.

### Phase 2 (2-6 weeks)

- Implement batch predict endpoint with bounded resource controls.
- Add "prod-mode" startup assertions that reject simulation configs.
- Add conformance tests that validate **no simulation branches are reachable** in production mode.

### Phase 3 (ongoing)

- Add SLO-backed health/readiness probes tied to real dependencies.
- Add operational runbooks for runtime disconnect, keystore failure, and registry consistency incidents.
- Add periodic audit controls verifying storage durability and key lifecycle correctness.

---

## Acceptance Criteria for "No Mocks" Claim

A release can claim no-mock/no-simulation production behavior only when:

1. No production route returns canned/echoed inference output.
2. Runtime disconnect causes explicit failure (not simulation fallback).
3. Keystore upload/retrieve is durable and tested end-to-end.
4. Registry persists model artifacts with verified object-store writes.
5. Production profiles hard-disable simulation code paths with test coverage.
6. Readiness reflects actual dependency state, not constants.

