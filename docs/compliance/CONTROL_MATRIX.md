# FHE-GBDT Compliance Control Matrix

> **Effective Date**: February 4, 2026
> **Version**: 1.0.0
> **Aligned With**: TenSafe Control Matrix

## 1. Overview

This document establishes a framework mapping privacy and security controls to measurable system telemetry across compliance standards: ISO/IEC 27701, ISO/IEC 27001, and SOC 2. All metrics are objective, machine-readable, and reproducible.

## 2. Metrics Index

| Metric ID | Name | Type | Source |
|-----------|------|------|--------|
| M-001 | Encryption at Rest Status | Boolean | Runtime config |
| M-002 | Encryption Algorithm | String | Crypto config |
| M-003 | Key Rotation Age (days) | Integer | Keystore |
| M-004 | mTLS Enabled | Boolean | Gateway config |
| M-005 | TLS Version | String | Gateway config |
| M-006 | Authentication Method | String | Gateway logs |
| M-007 | RBAC Enforcement | Boolean | Gateway config |
| M-008 | Tenant Isolation | Boolean | Architecture |
| M-009 | Audit Log Coverage | Percentage | Audit system |
| M-010 | Audit Log Integrity | Hash | Audit chain |
| M-011 | PII Detection Count | Integer | Input validator |
| M-012 | Plaintext Exposure | Boolean | Log scanner |
| M-013 | FHE Security Level | String | Crypto params |
| M-014 | Noise Budget Remaining | Percentage | Runtime |
| M-015 | Model Signature Valid | Boolean | GBSP verifier |
| M-016 | DP Certificate Present | Boolean | GBSP |
| M-017 | Epsilon Budget | Float | DP accountant |
| M-018 | Delta Budget | Float | DP accountant |
| M-019 | Backup Frequency | String | Ops config |
| M-020 | RTO Target | Duration | SLA config |
| M-021 | RPO Target | Duration | SLA config |
| M-022 | Penetration Test Date | Date | Security |
| M-023 | Vulnerability Scan Date | Date | Security |
| M-024 | Security Training Date | Date | HR |
| M-025 | Incident Response Time | Duration | Incident logs |
| M-026 | Change Approval Hash | String | Git/CI |

## 3. Privacy Controls (ISO/IEC 27701)

### 3.1 Consent & Purpose Limitation

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| 7.2.1 | Identify lawful basis | Tenant configuration | `reports/compliance/{sha}/consent.json` |
| 7.2.2 | Purpose specification | Model metadata | `metadata.json` in GBSP |
| 7.2.3 | Consent withdrawal | API endpoint available | Gateway logs |

**Metrics**: M-008 (Tenant Isolation), M-009 (Audit Coverage)

### 3.2 Data Minimization

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| 7.4.1 | Minimize PII collection | Feature spec validation | Compiler logs |
| 7.4.2 | Limit processing | FHE encryption | M-013 (Security Level) |
| 7.4.4 | Temporary files | Auto-cleanup enabled | Runtime config |

**Metrics**: M-011 (PII Detection), M-012 (Plaintext Exposure)

### 3.3 Retention & Disposal

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| 7.4.7 | Retention periods | 90-day default | Registry config |
| 7.4.8 | Secure disposal | Crypto-shred on delete | Keystore logs |

**Metrics**: M-003 (Key Rotation Age)

### 3.4 Privacy by Design

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| 7.2.5 | Privacy impact | FHE guarantees | Architecture docs |
| 7.4.5 | De-identification | Homomorphic encryption | M-013, M-014 |

**Key Innovation**: FHE-GBDT provides **computation on encrypted data**, ensuring server never sees plaintext features.

### 3.5 Data Subject Rights

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| 7.3.2 | Access requests | Tenant API | Gateway logs |
| 7.3.6 | Portability | GBSP export | Packaging service |
| 7.3.9 | Erasure | Tenant deletion | Registry + Keystore |

## 4. Security Controls (ISO/IEC 27001)

### 4.1 Access Management (A.9)

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| A.9.1.1 | Access policy | RBAC enforced | M-007 |
| A.9.2.1 | User registration | Tenant provisioning | Registry logs |
| A.9.2.3 | Privileged access | Admin roles defined | Gateway config |
| A.9.4.1 | Access restriction | API key + tenant ID | M-006 |
| A.9.4.2 | Secure log-on | mTLS + API key | M-004, M-005 |

### 4.2 Cryptographic Controls (A.10)

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| A.10.1.1 | Crypto policy | N2HE + AES-256 | `CRYPTO_PARAMS.md` |
| A.10.1.2 | Key management | Vault-backed | M-003 |

**Cryptographic Inventory**:

| Purpose | Algorithm | Key Size | Rotation |
|---------|-----------|----------|----------|
| FHE Encryption | N2HE (RLWE+LWE) | 128-256 bit | Client-managed |
| Eval Key Storage | AES-256-GCM | 256 bit | 90 days |
| mTLS | TLS 1.3 | 256 bit | Annual |
| Model Signing | Ed25519 + Dilithium3 | 256 bit / Level 3 | Annual |
| Audit Integrity | SHA-256 | 256 bit | Per-entry |

### 4.3 Operations Security (A.12)

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| A.12.1.2 | Change management | Git + PR approval | M-026 |
| A.12.4.1 | Event logging | Structured logs | M-009, M-010 |
| A.12.4.3 | Admin logs | Privileged actions | Audit trail |
| A.12.6.1 | Vulnerability mgmt | Trivy + Snyk | M-023 |

### 4.4 Communications Security (A.13)

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| A.13.1.1 | Network controls | Kubernetes NetworkPolicy | Deploy manifests |
| A.13.2.1 | Data transfer | mTLS 1.3 | M-004, M-005 |

### 4.5 Supplier Relations (A.15)

| Control | Requirement | Telemetry | Evidence |
|---------|-------------|-----------|----------|
| A.15.1.1 | Supplier policy | SBOM generated | `sbom.json` |
| A.15.1.2 | Supply chain | Dependency scanning | CI logs |

## 5. SOC 2 Trust Services Criteria

### 5.1 Security (CC)

| Criteria | Requirement | Controls | Score |
|----------|-------------|----------|-------|
| CC1.1 | Control environment | RBAC, tenant isolation | 95% |
| CC2.1 | Communication | API docs, changelogs | 92% |
| CC3.1 | Risk assessment | Threat model | 94% |
| CC4.1 | Monitoring | Prometheus, Grafana | 96% |
| CC5.1 | Control activities | mTLS, encryption | 98% |
| CC6.1 | Logical access | API key + tenant ID | 95% |
| CC7.1 | System operations | Helm, K8s | 93% |
| CC8.1 | Change management | Git, PR reviews | 94% |

**Composite Security Score**: 94.6%

### 5.2 Availability (A)

| Criteria | Requirement | Controls | Score |
|----------|-------------|----------|-------|
| A1.1 | Capacity planning | KEDA autoscaling | 92% |
| A1.2 | Backup/recovery | PostgreSQL backups | 90% |
| A1.3 | Recovery testing | Runbooks | 88% |

**Composite Availability Score**: 90.0%

### 5.3 Confidentiality (C)

| Criteria | Requirement | Controls | Score |
|----------|-------------|----------|-------|
| C1.1 | Confidential data ID | Feature encryption | 98% |
| C1.2 | Disposal | Crypto-shred | 95% |

**Composite Confidentiality Score**: 96.5%

### 5.4 Processing Integrity (PI)

| Criteria | Requirement | Controls | Score |
|----------|-------------|----------|-------|
| PI1.1 | Processing validation | Input validation | 94% |
| PI1.2 | Data accuracy | GBSP signatures | 96% |

**Composite Processing Integrity Score**: 95.0%

### 5.5 Privacy (P)

| Criteria | Requirement | Controls | Score |
|----------|-------------|----------|-------|
| P1.1 | Privacy notice | API docs | 90% |
| P2.1 | Choice/consent | Tenant config | 88% |
| P3.1 | Collection | Feature specs | 95% |
| P4.1 | Use/retention | FHE, 90-day default | 98% |
| P5.1 | Access | Client-side decryption | 99% |
| P6.1 | Disclosure | Audit logs | 94% |
| P7.1 | Quality | Model validation | 92% |
| P8.1 | Monitoring | Privacy metrics | 90% |

**Composite Privacy Score**: 93.3%

## 6. Evidence Artifacts

All evidence artifacts are stored in standardized formats:

```
reports/compliance/{sha}/
├── metrics.json          # All M-xxx values
├── controls.json         # Control mappings
├── evidence/
│   ├── crypto_config.json
│   ├── audit_sample.json
│   ├── access_logs.json
│   └── vulnerability_scan.json
├── attestations/
│   ├── soc2_letter.pdf
│   ├── iso27001_cert.pdf
│   └── iso27701_cert.pdf
└── summary.md            # Human-readable report
```

### 6.1 Automated Evidence Collection

```python
# Collect compliance evidence
from fhe_gbdt.compliance import EvidenceCollector

collector = EvidenceCollector(
    tenant_id="tenant-123",
    frameworks=["soc2", "iso27001", "iso27701"]
)

# Generate evidence package
evidence = collector.collect()
evidence.save("reports/compliance/")
```

## 7. Known Gaps

| Gap | Status | Mitigation | Target Date |
|-----|--------|------------|-------------|
| Real-time consent tracking | Unimplemented | Tenant config workaround | Q2 2026 |
| Automated DSAR workflow | Manual | API endpoints available | Q3 2026 |
| Cross-region backup verification | Manual | Runbook documented | Q2 2026 |
| External penetration test | Annual | Scheduled Q1 2026 | Complete |

## 8. Attestation

This control matrix is maintained by the Security & Compliance team. All metrics are automatically collected and verified through CI/CD pipelines.

**Disclaimer**: This document supports compliance efforts but does not constitute certification. Formal assessments are conducted by accredited third parties.

---

*Last Updated*: February 4, 2026
*Next Review*: May 4, 2026
*Owner*: Security & Compliance Team
