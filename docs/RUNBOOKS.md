# Runbooks

This document provides operational procedures for common alerts and incidents.

## Runtime Unavailable (Critical)

### Symptoms
- REST `/v1/predict` returning 503.
- gRPC `codes.Unavailable` from gateway predict path.
- Increased error-rate alerts.

### Diagnosis
1. Check gateway logs for `runtime unavailable`.
2. Verify runtime service endpoint and pod health.
3. Validate mTLS/network policy between gateway and runtime.
4. Confirm runtime process has loaded compiled plans and key access.

### Mitigation
1. Restore runtime connectivity.
2. Restart unhealthy runtime pods.
3. If rollout caused regression, rollback gateway/runtime release.
4. Validate with a real encrypted inference request (not simulation).

---

## Keystore Retrieval Failures

### Symptoms
- Errors retrieving eval keys.
- keystore `NotFound` or decrypt failures.

### Diagnosis
1. Check `KEYSTORE_STORAGE_DIR` mount and permissions.
2. If Vault mode: verify Vault transit health and token validity.
3. Verify tenant/model identifiers used for upload and retrieval match.

### Mitigation
1. Restore storage volume and permissions.
2. Restore Vault availability/credentials.
3. Re-upload eval keys for affected tenant/model.

---

## Registry Persistence Failures

### Symptoms
- Model registration returns internal errors.
- Metadata exists without usable artifact path.

### Diagnosis
1. Check `REGISTRY_STORAGE_DIR` writable and durable.
2. Check PostgreSQL connectivity and schema health.
3. Validate model content size and request integrity.

### Mitigation
1. Restore storage mount and permissions.
2. Restore DB connectivity.
3. Retry model registration and verify content path exists.

---

## High Error Rate (HighErrorRate)

### Symptoms
- Alert firing for elevated 5xxs.

### Diagnosis
1. Check gateway and runtime logs.
2. Check readiness endpoints for dependency regressions.
3. Correlate with recent deploy/change windows.

### Mitigation
1. Roll back recent release if needed.
2. Scale affected services.
3. Re-test with representative encrypted payloads.

---

## Rollback Procedure

```bash
helm history fhe-gbdt
helm rollback fhe-gbdt [REVISION]
kubectl rollout status deployment/fhe-gbdt-runtime-cpu
kubectl rollout status deployment/fhe-gbdt-gateway
```
