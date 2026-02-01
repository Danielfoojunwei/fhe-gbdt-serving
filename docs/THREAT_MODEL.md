# Threat Model (STRIDE Analysis)

## Assets
| Asset                | Confidentiality | Integrity | Availability |
|----------------------|-----------------|-----------|--------------|
| Plaintext Features   | Critical        | High      | N/A          |
| FHE Secret Keys      | Critical        | Critical  | High         |
| FHE Eval Keys        | Medium          | Critical  | High         |
| Compiled Plans       | Low             | Critical  | High         |
| Model Weights        | Medium          | Critical  | Medium       |

## Threat Categories (STRIDE)

### Spoofing
| Threat                           | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Attacker impersonates tenant     | API key + tenant_id binding                 |
| Attacker forges request_id       | Server-generated request_id only            |

### Tampering
| Threat                           | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Modified ciphertext payload      | FHE integrity (authenticated encryption)    |
| Modified compiled plan           | Content-addressed plan_id (SHA256 hash)     |
| Man-in-the-middle                | mTLS between services                       |

### Repudiation
| Threat                           | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Tenant denies action             | Audit logs with tenant_id, request_id       |
| Admin denies key access          | Keystore audit trail                        |

### Information Disclosure
| Threat                           | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Server learns plaintext features | FHE (feature vectors always encrypted)      |
| Logs leak sensitive data         | Allowlist logging, CI guardrails            |
| Memory dump exposes keys         | Secret keys client-only, never on server    |

### Denial of Service
| Threat                           | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Large payload flood              | Payload size limits (64MB)                  |
| Request rate flood               | Per-tenant rate limiting                    |
| Compute exhaustion (FHE)         | Queue depth limits, HPA                     |

### Elevation of Privilege
| Threat                           | Mitigation                                  |
|----------------------------------|---------------------------------------------|
| Access another tenant's model    | RBAC: compiled_model_id ownership check     |
| Access another tenant's keys     | Keystore: tenant_id scoping                 |

## mTLS Configuration
All service-to-service communication uses mutual TLS:
- Gateway ↔ Runtime
- Gateway ↔ Registry
- Gateway ↔ Keystore

## Supply Chain Security
- Dependencies pinned in lockfiles
- SBOM generated with CycloneDX
- Container images signed
- SAST + dependency scan in CI
