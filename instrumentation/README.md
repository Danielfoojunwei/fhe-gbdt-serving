# Instrumentation Guidelines

This document describes the observability practices for the FHE-GBDT Inference API.

## Tracing

We use OpenTelemetry for distributed tracing across all services.

### Required Span Attributes
| Attribute           | Description                           | ALLOWED in Prod |
|---------------------|---------------------------------------|-----------------|
| `tenant_id`         | Hashed/anonymized tenant identifier   | ✅              |
| `compiled_model_id` | The compiled model being executed     | ✅              |
| `profile`           | latency or throughput                 | ✅              |
| `batch_size`        | Number of samples in the batch        | ✅              |
| `scheme_id`         | FHE scheme identifier                 | ✅              |
| `request_id`        | Unique request correlation ID         | ✅              |

### FORBIDDEN Span Attributes
| Attribute           | Reason                                      |
|---------------------|---------------------------------------------|
| `payload`           | Contains ciphertext bytes                   |
| `feature_names`     | May leak data schema                        |
| `feature_values`    | NEVER allowed - plaintext data              |

## Structured Logging

All logs MUST use the allowlist-based structured logger.

### Allowed Log Fields
```go
var AllowedFields = []string{
    "request_id", "tenant_id", "compiled_model_id", "profile",
    "batch_size", "latency_ms", "status", "error_code",
}
```

### Forbidden Log Patterns
The following patterns are forbidden and will fail CI:
- `payload`, `ciphertext`, `feature`, `plaintext`
- Any raw byte array logging

## Request ID Propagation

Every request MUST have a `request_id` header/metadata.
- Gateway generates `request_id` if not present.
- All downstream services propagate and log with the same ID.

## CI Guardrails

The CI pipeline includes a job that greps for forbidden logging patterns.
See `.github/workflows/ci.yml` for the implementation.
