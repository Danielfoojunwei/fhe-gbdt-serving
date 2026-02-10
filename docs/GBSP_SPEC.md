# GBDT Secure Package (GBSP) Specification v1.0

> **Effective Date**: February 4, 2026
> **Version**: 1.0.0
> **Status**: Production

## 1. Overview

The GBDT Secure Package (GBSP) format (`.gbsp`) is a ZIP-based container designed for securely transporting compiled GBDT models between training environments and FHE inference servers. This specification is aligned with TenSafe's TGSP format to ensure consistent security posture across the company's product line.

## 2. Design Goals

| Goal | Description |
|------|-------------|
| **Confidentiality** | Model weights and split points encrypted at rest |
| **Integrity** | Content hashed with SHA-256; tampering detectable |
| **Provenance** | Hybrid signatures (Ed25519 + Dilithium3) for post-quantum security |
| **Optimization** | FHE execution hints for MOAI column packing and levelization |

## 3. Security Guarantees

### 3.1 Encryption

- **Algorithm**: AES-256-GCM or ChaCha20-Poly1305
- **Key Derivation**: HKDF-SHA256 with per-package salt
- **Key Hierarchy**:
  - Producer maintains Key Encryption Key (KEK)
  - Data Encryption Key (DEK) wrapped per recipient
  - Recipients use KEM (Kyber-768 or X25519) to unwrap

### 3.2 Integrity

- **Hash Algorithm**: SHA-256
- **Scope**: All files in manifest individually hashed
- **Verification**: Hash chain linking manifest to signature

### 3.3 Provenance

- **Classical Signature**: Ed25519 (128-bit security)
- **Post-Quantum Signature**: Dilithium3 (NIST Level 3)
- **Hybrid Mode**: Both signatures required for verification
- **Certificate Chain**: X.509 certificates with OCSP stapling

## 4. File Structure

A valid GBSP archive MUST contain the following files:

```
model.gbsp
├── manifest.json           # Integrity hashes for all files
├── manifest.sig            # Detached hybrid signature
├── policy.rego             # OPA Gatekeeper policy
├── model.enc               # Encrypted compiled model (ObliviousPlanIR)
├── optimization.json       # MOAI execution hints
├── evidence.json           # Training telemetry and audit chain
├── dp_certificate.json     # Differential privacy certificate (if applicable)
└── metadata.json           # Model metadata and versioning
```

### 4.1 manifest.json

```json
{
  "version": "1.0.0",
  "created_at": "2026-02-04T12:00:00Z",
  "producer": {
    "tenant_id": "tenant-123",
    "signing_key_id": "key-abc",
    "certificate_chain": ["base64-encoded-cert"]
  },
  "files": {
    "model.enc": {
      "sha256": "abcd1234...",
      "size_bytes": 1048576,
      "encrypted": true
    },
    "optimization.json": {
      "sha256": "efgh5678...",
      "size_bytes": 2048,
      "encrypted": false
    }
  },
  "encryption": {
    "algorithm": "AES-256-GCM",
    "kdf": "HKDF-SHA256",
    "kem": "Kyber-768"
  }
}
```

### 4.2 manifest.sig

Detached signature file containing:

```json
{
  "ed25519": "base64-encoded-ed25519-signature",
  "dilithium3": "base64-encoded-dilithium3-signature",
  "signed_at": "2026-02-04T12:00:00Z",
  "certificate_fingerprint": "sha256:abcd1234..."
}
```

### 4.3 policy.rego

OPA Gatekeeper policy for deployment validation:

```rego
package gbsp.v1

default allow = false

allow {
    input.tenant_id == data.allowed_tenants[_]
    valid_signature
    valid_dp_bounds
    compatible_runtime
}

valid_signature {
    crypto.verify_ed25519(input.manifest, input.signature.ed25519, data.public_key)
    crypto.verify_dilithium3(input.manifest, input.signature.dilithium3, data.pq_public_key)
}

valid_dp_bounds {
    input.dp_certificate.epsilon <= data.max_epsilon
    input.dp_certificate.delta <= data.max_delta
}

compatible_runtime {
    input.optimization.moai_version >= data.min_moai_version
    input.optimization.n2he_version >= data.min_n2he_version
}
```

### 4.4 model.enc

Encrypted ObliviousPlanIR containing:

- Compiled decision trees
- Feature split points
- Leaf values
- Column packing layout
- Levelized execution schedule

### 4.5 optimization.json

MOAI execution hints:

```json
{
  "moai_version": "1.0.0",
  "n2he_version": "2.1.0",
  "packing": {
    "strategy": "column",
    "max_slots": 32768,
    "feature_layout": [0, 1, 2, 3]
  },
  "schedule": {
    "type": "levelized",
    "batch_size": 256,
    "rotation_budget": 12
  },
  "crypto_params": {
    "ring_dimension": 4096,
    "ciphertext_modulus": "2^64",
    "security_level": "high"
  }
}
```

### 4.6 evidence.json

Training provenance and audit chain:

```json
{
  "training_id": "train-xyz-789",
  "started_at": "2026-02-04T10:00:00Z",
  "completed_at": "2026-02-04T11:30:00Z",
  "library": "xgboost",
  "library_version": "2.0.0",
  "dataset": {
    "name": "customer_churn",
    "rows": 100000,
    "features": 50,
    "hash": "sha256:dataset-hash"
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
  },
  "metrics": {
    "train_auc": 0.95,
    "val_auc": 0.92
  },
  "audit_chain": {
    "previous_hash": "sha256:prev-evidence-hash",
    "current_hash": "sha256:current-evidence-hash"
  }
}
```

### 4.7 dp_certificate.json

Differential privacy certificate (when DP training is used):

```json
{
  "version": "1.0.0",
  "mechanism": "DP-GBDT",
  "epsilon": 1.0,
  "delta": 1e-5,
  "noise_type": "laplace",
  "sensitivity": {
    "split_candidates": 0.1,
    "leaf_values": 0.01
  },
  "accountant": {
    "type": "rdp",
    "orders": [2, 4, 8, 16, 32, 64],
    "epsilon_per_order": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
  },
  "certified_at": "2026-02-04T11:30:00Z",
  "certifier": "fhe-gbdt-training-service"
}
```

### 4.8 metadata.json

Model metadata:

```json
{
  "model_id": "model-abc-123",
  "model_name": "customer_churn_predictor",
  "version": "1.0.0",
  "library": "xgboost",
  "task": "binary_classification",
  "features": {
    "count": 50,
    "names": ["age", "tenure", "monthly_charges"],
    "types": ["numeric", "numeric", "numeric"]
  },
  "trees": {
    "count": 100,
    "max_depth": 6,
    "total_leaves": 6400
  },
  "fhe_compatible": true,
  "created_by": "tenant-123",
  "created_at": "2026-02-04T11:30:00Z"
}
```

## 5. Packaging Workflow

### 5.1 Producer Side (Training Service)

```
1. Train GBDT model (XGBoost/LightGBM/CatBoost)
2. Apply differential privacy (if enabled)
3. Compile to ObliviousPlanIR
4. Generate MOAI optimization hints
5. Collect training evidence
6. Generate DP certificate (if applicable)
7. Create metadata
8. Encrypt model with DEK
9. Wrap DEK with recipient KEMs
10. Hash all files → manifest.json
11. Sign manifest → manifest.sig
12. Create policy.rego
13. Bundle into .gbsp archive
```

### 5.2 Consumer Side (Runtime Service)

```
1. Extract .gbsp archive
2. Verify hybrid signature (Ed25519 + Dilithium3)
3. Validate manifest hashes
4. Execute policy.rego checks
5. Verify DP bounds (if applicable)
6. Check runtime compatibility
7. Unwrap DEK using private KEM key
8. Decrypt model.enc
9. Load ObliviousPlanIR
10. Initialize MOAI executor
```

## 6. API Integration

### 6.1 Packaging API

```protobuf
service PackagingService {
  rpc CreatePackage(CreatePackageRequest) returns (CreatePackageResponse);
  rpc VerifyPackage(VerifyPackageRequest) returns (VerifyPackageResponse);
  rpc ExtractPackage(ExtractPackageRequest) returns (ExtractPackageResponse);
}

message CreatePackageRequest {
  string tenant_id = 1;
  string model_id = 2;
  bytes compiled_model = 3;
  OptimizationHints optimization = 4;
  TrainingEvidence evidence = 5;
  DPCertificate dp_certificate = 6;
  repeated string recipient_public_keys = 7;
}

message VerifyPackageRequest {
  string tenant_id = 1;
  bytes package_data = 2;
  bytes private_kem_key = 3;
}
```

### 6.2 CLI Commands

```bash
# Create a GBSP package
fhe-gbdt package create \
  --model model.json \
  --output model.gbsp \
  --sign-key signing.key \
  --encrypt-for recipient.pub

# Verify a GBSP package
fhe-gbdt package verify \
  --package model.gbsp \
  --trusted-keys trusted/

# Extract a GBSP package
fhe-gbdt package extract \
  --package model.gbsp \
  --decrypt-key private.key \
  --output ./extracted/
```

## 7. Security Considerations

### 7.1 Key Management

- Signing keys MUST be stored in HSM or Vault
- KEKs MUST be rotated every 90 days
- Recipient public keys MUST be verified before encryption

### 7.2 Audit Trail

- All packaging operations logged with request ID
- Hash chain links packages to training evidence
- Immutable audit logs stored separately

### 7.3 Threat Mitigations

| Threat | Mitigation |
|--------|------------|
| Model tampering | SHA-256 hashes + hybrid signatures |
| Key compromise | Post-quantum signatures (Dilithium3) |
| Unauthorized deployment | OPA policy enforcement |
| Privacy leakage | DP certificate bounds checking |

## 8. Compatibility

### 8.1 Version Matrix

| GBSP Version | MOAI Version | N2HE Version | Status |
|--------------|--------------|--------------|--------|
| 1.0.x | 1.0.x | 2.0.x+ | Current |

### 8.2 Migration Path

When upgrading GBSP versions:
1. New runtimes MUST support previous GBSP versions
2. Deprecation notice 6 months before version sunset
3. Migration tools provided for package re-signing

## 9. References

- [TGSP Specification v2.4](https://github.com/Danielfoojunwei/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation/blob/main/docs/TSSP_SPEC.md) - TenSafe's equivalent for LoRA
- [MOAI Paper (NDSS 2025)](https://www.ndss-symposium.org/ndss-paper/moai/) - Optimization techniques
- [Dilithium NIST Standard](https://pq-crystals.org/dilithium/) - Post-quantum signatures
- [Kyber NIST Standard](https://pq-crystals.org/kyber/) - Post-quantum KEM
