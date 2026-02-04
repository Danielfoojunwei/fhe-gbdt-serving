# FHE-GBDT Data Flow Documentation

> **Effective Date**: February 4, 2026
> **Version**: 1.0.0
> **Compliance**: ISO/IEC 27701, ISO/IEC 27001, SOC 2

## 1. Overview

This document describes data flows through the FHE-GBDT platform, a privacy-preserving machine learning inference service. Understanding these flows is critical for compliance audits, security assessments, and privacy impact analyses.

## 2. System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FHE-GBDT Platform                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Client    │───▶│   Gateway   │───▶│   Runtime   │───▶│  Response   │  │
│  │    SDK      │    │  (Auth/TLS) │    │  (FHE Ops)  │    │  (Encrypted)│  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                              │
│         │                  ▼                  ▼                              │
│         │           ┌─────────────┐    ┌─────────────┐                      │
│         │           │   Registry  │    │  Keystore   │                      │
│         │           │  (Models)   │    │ (Eval Keys) │                      │
│         │           └─────────────┘    └─────────────┘                      │
│         │                  │                  │                              │
│         │                  ▼                  ▼                              │
│         │           ┌─────────────┐    ┌─────────────┐                      │
│         │           │ PostgreSQL  │    │    Vault    │                      │
│         │           └─────────────┘    └─────────────┘                      │
│         │                                                                    │
│         │           ┌─────────────┐    ┌─────────────┐                      │
│         └──────────▶│  Training   │───▶│  Compiler   │                      │
│                     │   Service   │    │  (MOAI)     │                      │
│                     └─────────────┘    └─────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. Data Classification

| Classification | Description | Examples | Handling |
|----------------|-------------|----------|----------|
| **Secret** | Cryptographic keys | Secret keys, API keys | Client-only, never transmitted |
| **Confidential** | Encrypted user data | Feature ciphertexts | FHE encrypted, server processes |
| **Internal** | Platform operations | Eval keys, compiled models | Encrypted at rest, mTLS in transit |
| **Public** | Documentation | API specs, schemas | Open access |

## 4. Training Pipeline

### 4.1 Data Ingestion

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   External   │────▶│   Ingestion  │────▶│   Staging    │
│   Dataset    │     │   Validator  │     │   Storage    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ PII Scanner  │
                     │ (Optional)   │
                     └──────────────┘
```

| Stage | Data In | Data Out | Privacy Control |
|-------|---------|----------|-----------------|
| External | Raw dataset (CSV, Parquet) | Validated dataset | None (user responsibility) |
| Validator | Validated dataset | Staged dataset | Schema validation, size limits |
| PII Scanner | Staged dataset | Flagged dataset | PII detection, optional redaction |

### 4.2 Model Training

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Staged     │────▶│   GBDT       │────▶│   Trained    │
│   Dataset    │     │   Training   │     │    Model     │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │     DP       │
                     │  Accountant  │
                     └──────────────┘
```

**Privacy Mechanisms**:

| Mechanism | Implementation | Parameter |
|-----------|----------------|-----------|
| **DP Split Selection** | Laplace noise on split scores | ε per level |
| **DP Leaf Values** | Gaussian noise on leaf outputs | σ calibrated |
| **Gradient Clipping** | L2 norm bound | C = 1.0 default |
| **Secure Aggregation** | Federated training (optional) | Pairwise masking |

### 4.3 Model Compilation

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Trained    │────▶│   Compiler   │────▶│  Compiled    │
│    Model     │     │   (MOAI)     │     │    Plan      │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   GBSP       │
                     │  Packager    │
                     └──────────────┘
```

| Stage | Data In | Data Out | Privacy Control |
|-------|---------|----------|-----------------|
| Parser | XGBoost/LightGBM/CatBoost JSON | ModelIR | Input validation |
| Optimizer | ModelIR | ObliviousPlanIR | Column packing, levelization |
| Packager | ObliviousPlanIR + Evidence | .gbsp archive | Encryption, signing |

## 5. Inference Pipeline

### 5.1 Client-Side Encryption

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw        │────▶│   Feature    │────▶│  Encrypted   │
│   Features   │     │   Encoder    │     │  Ciphertext  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  N2HE Key    │
                     │   Manager    │
                     └──────────────┘
```

**Key Point**: Raw features NEVER leave the client. Encryption happens on-device.

| Data Element | Location | Encryption | Notes |
|--------------|----------|------------|-------|
| Raw features | Client only | None | Never transmitted |
| Secret key | Client only | None | Never transmitted |
| Ciphertext | Client → Server | N2HE (RLWE) | Encrypted features |
| Eval key | Client → Server | Envelope (AES) | For server computation |

### 5.2 Gateway Processing

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Incoming   │────▶│   Auth &     │────▶│   Validated  │
│   Request    │     │   Validate   │     │   Request    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Rate Limiter │
                     └──────────────┘
```

| Check | Action | Failure Response |
|-------|--------|------------------|
| mTLS | Verify client cert | 401 Unauthenticated |
| API Key | Validate against tenant | 401 Unauthenticated |
| Tenant ID | Extract from context | 400 Bad Request |
| Rate Limit | Token bucket check | 429 Too Many Requests |
| Input Size | Max 100MB | 413 Payload Too Large |

**Audit Logging**:
```json
{
  "timestamp": "2026-02-04T12:00:00Z",
  "request_id": "req-abc-123",
  "tenant_id": "tenant-123",
  "action": "Predict",
  "model_id": "model-xyz",
  "status": "SUCCESS",
  "latency_ms": 62,
  "ciphertext_size_bytes": 32768,
  "previous_hash": "sha256:abc...",
  "current_hash": "sha256:def..."
}
```

**Note**: Ciphertext payloads are NEVER logged, only metadata.

### 5.3 Runtime Execution

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Ciphertext  │────▶│   MOAI       │────▶│  Encrypted   │
│   Batch      │     │  Executor    │     │   Result     │
└──────────────┘     └──────────────┘     └──────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Eval Key    │     │  Compiled    │     │  Runtime     │
│   Cache      │     │    Plan      │     │   Stats      │
└──────────────┘     └──────────────┘     └──────────────┘
```

**FHE Operations**:

| Operation | Count (Typical) | Description |
|-----------|-----------------|-------------|
| Comparisons | ~6400 | Tree split evaluations |
| Scheme Switches | ~200 | RLWE ↔ LWE transitions |
| Rotations | ~12 | MOAI-optimized (vs. ~300 naive) |
| Bootstraps | 0-2 | Noise budget refresh |

**Data Isolation**:
- Each tenant's eval keys stored separately
- No cross-tenant data access possible
- Ciphertext processed without decryption

### 5.4 Response Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Encrypted   │────▶│   Gateway    │────▶│   Client     │
│   Result     │     │  (Response)  │     │  (Decrypt)   │
└──────────────┘     └──────────────┘     └──────────────┘
```

| Data Element | Location | Encryption | Notes |
|--------------|----------|------------|-------|
| Encrypted result | Server → Client | N2HE (LWE) | FHE ciphertext |
| Plaintext result | Client only | Decrypted | Never on server |

## 6. Key Management Flow

### 6.1 Client Key Generation

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Key Gen    │────▶│  Secret Key  │     │  Eval Key    │
│   (Client)   │     │  (Local)     │────▶│  (Upload)    │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 6.2 Server Key Storage

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Eval Key    │────▶│   Keystore   │────▶│    Vault     │
│  (Received)  │     │  (Envelope)  │     │  (Encrypted) │
└──────────────┘     └──────────────┘     └──────────────┘
```

| Key Type | Storage | Rotation | Access |
|----------|---------|----------|--------|
| Secret Key | Client device | User-controlled | Client only |
| Eval Key | Vault (encrypted) | 90 days | Runtime service |
| API Key | Vault | 365 days | Gateway |
| Signing Key | HSM | Annual | Packaging service |

## 7. Artifact Storage

### 7.1 Storage Locations

| Artifact | Storage | Encryption | Retention |
|----------|---------|------------|-----------|
| Compiled Plans | PostgreSQL | AES-256-GCM | Indefinite |
| GBSP Packages | MinIO/S3 | AES-256-GCM | Indefinite |
| Eval Keys | Vault | Envelope | 90 days after last use |
| Audit Logs | Loki | TLS in transit | 1 year |
| Metrics | Prometheus | None (no PII) | 30 days |
| Traces | Jaeger | None (no PII) | 7 days |

### 7.2 Secure Deletion

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Delete     │────▶│   Crypto     │────▶│   Verify     │
│   Request    │     │    Shred     │     │   Deletion   │
└──────────────┘     └──────────────┘     └──────────────┘
```

**Crypto-shred Process**:
1. Revoke encryption key in Vault
2. Overwrite encrypted data with zeros
3. Remove metadata from registry
4. Log deletion event (hash-chained)

## 8. Network Boundaries

### 8.1 Trust Zones

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL (Untrusted)                      │
│                                                              │
│    ┌──────────────┐                                         │
│    │   Clients    │                                         │
│    └──────────────┘                                         │
│            │                                                 │
│            │ mTLS + API Key                                  │
│            ▼                                                 │
├─────────────────────────────────────────────────────────────┤
│                      DMZ (Gateway)                           │
│                                                              │
│    ┌──────────────┐                                         │
│    │   Ingress    │                                         │
│    │   Gateway    │                                         │
│    └──────────────┘                                         │
│            │                                                 │
│            │ mTLS (internal)                                 │
│            ▼                                                 │
├─────────────────────────────────────────────────────────────┤
│                   INTERNAL (Trusted)                         │
│                                                              │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│    │   Runtime    │    │   Registry   │    │   Keystore   │ │
│    └──────────────┘    └──────────────┘    └──────────────┘ │
│            │                  │                  │           │
│            │                  ▼                  ▼           │
│            │           ┌──────────────┐    ┌──────────────┐ │
│            │           │  PostgreSQL  │    │    Vault     │ │
│            │           └──────────────┘    └──────────────┘ │
│            │                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Network Policies

| Source | Destination | Port | Protocol |
|--------|-------------|------|----------|
| External | Gateway | 443 | HTTPS/gRPC |
| Gateway | Runtime | 9000 | gRPC (mTLS) |
| Gateway | Registry | 8082 | gRPC (mTLS) |
| Gateway | Keystore | 8081 | gRPC (mTLS) |
| Registry | PostgreSQL | 5432 | PostgreSQL |
| Keystore | Vault | 8200 | HTTPS |
| * | Prometheus | 9090 | HTTP (metrics) |

## 9. Compliance Mapping

### 9.1 ISO 27701 Data Flow Requirements

| Requirement | Implementation |
|-------------|----------------|
| 7.4.1 Minimize collection | Only encrypted features transmitted |
| 7.4.2 Limit processing | FHE computation without decryption |
| 7.4.5 De-identification | Homomorphic encryption |
| 7.4.7 Retention | Configurable, 90-day default |
| 7.4.8 Disposal | Crypto-shred with audit |

### 9.2 SOC 2 Data Flow Requirements

| Criteria | Implementation |
|----------|----------------|
| CC6.1 Logical access | mTLS + API key + tenant isolation |
| CC6.6 Data transmission | TLS 1.3 all connections |
| CC6.7 Data removal | Crypto-shred + verification |
| C1.1 Confidential data | FHE encryption client-side |
| P4.2 Retention | 90-day default, configurable |

## 10. Incident Data Flow

In case of security incident:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Detect     │────▶│   Isolate    │────▶│  Investigate │
│   (Alert)    │     │  (Freeze)    │     │  (Audit Log) │
└──────────────┘     └──────────────┘     └──────────────┘
        │                                         │
        ▼                                         ▼
┌──────────────┐                          ┌──────────────┐
│   Notify     │                          │   Remediate  │
│  (Tenant)    │                          │  (Rotate)    │
└──────────────┘                          └──────────────┘
```

| Phase | Data Access | Logging |
|-------|-------------|---------|
| Detection | Metrics, alerts | Automated |
| Isolation | Freeze affected tenant | Hash-chained |
| Investigation | Audit logs only | Read-only |
| Notification | Contact info | Encrypted channel |
| Remediation | Key rotation | Full audit |

---

*Last Updated*: February 4, 2026
*Next Review*: May 4, 2026
*Owner*: Security & Compliance Team
