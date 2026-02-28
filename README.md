# FHE Model Serving Platform

**Confidential AI inference for regulated enterprises: keep sensitive features encrypted, enforce tenant isolation, and produce audit-ready evidence.**

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](/.github/workflows/ci.yml)
[![Security](https://img.shields.io/badge/security-hardened-blue)](/docs/THREAT_MODEL.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE)

---

## Executive Brief for CISOs

Most classical ML deployments still decrypt sensitive inputs in application memory. That design creates a broad blast radius across logs, caches, APM tooling, support workflows, and incident response handling. Our platform changes the trust boundary: data is encrypted client-side, processed in encrypted form, and only decrypted by the data owner.

### Why this matters now

Security and privacy guidance increasingly emphasizes governance, accountability, and resilience for AI systems:

- NIST states that “**AI actors are not all equally responsible or accountable**,” underscoring the need for enforceable technical controls, not policy-only promises (NIST, 2023).
- CISA and international partners note that “**data security is one of the most common vulnerabilities in AI systems**,” highlighting practical attack surfaces in modern AI pipelines (CISA et al., 2024).
- OWASP identifies Sensitive Information Disclosure as a recurring application risk class, which becomes harder to control when ML inference requires plaintext in server memory (OWASP Foundation, 2021).

Our platform is built to reduce exposure from that exact class of risk.

---

## The Problem We Solve (Mapped to CISO Priorities)

### 1) Confidentiality risk in AI inference paths
**Observed in practice:** sensitive features often flow through API gateways, model servers, debug traces, exception stacks, and BI pipelines.

**What we do:**
- Encrypt features on the client side.
- Execute inference with FHE-compatible plans in runtime.
- Return encrypted outputs so only the key owner can decrypt.

**CISO benefit:** materially reduces plaintext handling scope and insider-access exposure.

### 2) Compliance friction across jurisdictions and sectors
**Observed in literature and guidance:** cross-border, cross-entity data handling and retention create legal/compliance overhead.

**What we do:**
- Tenant-scoped keys and model boundaries.
- Strong audit-chain support and structured operational controls.
- Explicit controls for mTLS, RBAC, and key lifecycle integration.

**CISO benefit:** easier control mapping for GDPR/HIPAA/PCI-oriented programs and lower evidence-collection effort during audits.

### 3) Third-party and supply-chain trust gap
**Observed in media and incident reporting:** dependency compromise and configuration drift are persistent operational threats.

**What we do:**
- CI with guardrails, security scanning, and SBOM generation.
- Service boundary hardening and traceability-ready architecture.
- Least-privilege defaults and reproducible deployment patterns.

**CISO benefit:** improved attestability for vendor assessments and board-level risk reporting.

---

## What Makes This Platform Novel (and Why It Matters)

Our innovation set is implemented in `services/innovations/` and documented in depth in [docs/INNOVATIONS.md](docs/INNOVATIONS.md).

### Innovation portfolio (production relevance)

1. **Leaf-Centric Encoding**  
   Parallelized leaf-indicator computation for encrypted trees to reduce sequential branching overhead.

2. **Gradient-Aware Noise Allocation**  
   Precision allocation by feature importance to preserve inference quality within FHE constraints.

3. **Homomorphic Ensemble Pruning**  
   FHE-friendly significance scoring to reduce computation where tree contribution is low.

4. **Polynomial Leaf Functions**  
   Additive leaf-level correction terms with validation safeguards.

5. **MOAI-Native Tree Conversion**  
   Rotation-minimizing conversion path for tree execution plans.

6. **Bootstrap-Aligned Architecture**  
   Runtime design aligned to noise-budget and bootstrapping economics.

7. **Federated Multi-Key Protocol**  
   Enables multi-party encrypted collaboration patterns.

8. **Streaming Encrypted Gradients**  
   Streaming-compatible encrypted update flows for operational adaptability.

9. **Unified Architecture Integration**  
   Coherent integration across compiler/runtime/ops controls.

### Why we classify these as state-of-the-art in this repo context

- They target real bottlenecks in practical FHE inference (rotations, precision/noise trade-offs, per-tree cost).
- They are validated by dedicated integration and benchmark suites in this codebase.
- They provide architecture-level options to trade off latency, accuracy retention, and operational risk.

> Production note: some optimization families have workload-dependent behavior. We document limitations and failure modes explicitly, including regression-sensitive cases, to support risk-informed rollout decisions rather than “one-size-fits-all” claims.

---

## Security Architecture (CISO Lens)

### Control layers

- **Transport security:** mTLS between services.
- **Identity and authorization:** API key + tenant scoping + RBAC pathways.
- **Key security:** evaluation-key management with envelope-encryption integrations.
- **Operational assurance:** telemetry, health signals, and security scan hooks.
- **Governance evidence:** audit-chain and control-matrix supporting artifacts in `docs/compliance/`.

### Data handling posture

- Plaintext feature vectors are not required in the server-side inference path by design.
- Cryptographic context is separated from application-level model metadata.
- Tenant separation is enforced at API and model-plan boundaries.

---

## Deployment Readiness

### Core services

- **Gateway (Go):** authn/authz, routing, rate controls, observability.
- **Registry (Go + SQL):** model metadata and compiled-plan references.
- **Compiler (Python):** model parsing and IR transformation.
- **Runtime (C++):** encrypted execution path.
- **Keystore (Go):** key lifecycle and secure retrieval boundaries.
- **SDKs (Python/TypeScript):** client integration for encryption and inference workflows.

### Supported model families

- XGBoost, LightGBM, CatBoost
- Scikit-learn Decision Trees and Random Forests
- Scikit-learn Logistic Regression
- Statsmodels GLM pathways

---

## Fast Start (for Security + Platform Teams)

```bash
git clone https://github.com/your-org/fhe-gbdt-serving.git
cd fhe-gbdt-serving
make docker-up
make test
```

Then review:
- [Threat Model](docs/THREAT_MODEL.md)
- [Production Readiness](docs/PRODUCTION_READINESS.md)
- [Control Matrix](docs/compliance/CONTROL_MATRIX.md)
- [Runbooks](docs/RUNBOOKS.md)

---

## References (APA 7th)

CISA, U.K. National Cyber Security Centre, Australian Cyber Security Centre, New Zealand National Cyber Security Centre, & Canadian Centre for Cyber Security. (2024). *Guidelines for secure AI system development*. https://www.cisa.gov/resources-tools/resources/guidelines-secure-ai-system-development

NIST. (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (NIST AI 100-1). National Institute of Standards and Technology. https://www.nist.gov/itl/ai-risk-management-framework

OWASP Foundation. (2021). *OWASP Top 10: The ten most critical web application security risks*. https://owasp.org/www-project-top-ten/

Verizon. (2024). *2024 Data Breach Investigations Report*. https://www.verizon.com/business/resources/reports/dbir/

European Union Agency for Cybersecurity (ENISA). (2023). *ENISA threat landscape 2023*. https://www.enisa.europa.eu/publications/enisa-threat-landscape-2023

Gentry, C. (2009). *A fully homomorphic encryption scheme* (Doctoral dissertation, Stanford University). https://crypto.stanford.edu/craig/

---

## CISO Deep Dive: Threat Scenarios, Business Impact, and Control Outcomes

This section **extends** (not replaces) the executive brief above with a deeper, risk-program view for security leadership.

### A. High-probability threat scenarios in plaintext AI inference

#### Scenario A1 — Memory-scraping and insider abuse in model-serving tiers
In conventional ML serving, features are decrypted in memory to perform inference. That creates exposure through:
- Debug dumps
- Tracing payload capture
- Crash diagnostics
- Privileged admin shell access

**Why CISOs care:** these are hard-to-detect, high-impact pathways that frequently bypass preventive DLP controls.

**Platform control response:** encrypted feature handling from client to inference pipeline reduces plaintext residency in server memory.

#### Scenario A2 — Observability stack overcollection
Modern reliability stacks often ingest request metadata and payload-adjacent values by default.

**Why CISOs care:** telemetry systems are high-value aggregators that are broadly accessible to SRE/platform roles.

**Platform control response:** design posture minimizes sensitive plaintext in transit through telemetry surfaces and enforces structured control boundaries.

#### Scenario A3 — Cross-tenant data governance failures
Multi-tenant AI services can drift into keying, policy, or routing mistakes under operational pressure.

**Why CISOs care:** cross-tenant access is an existential trust event for regulated buyers.

**Platform control response:** tenant-scoped key/material boundaries plus model/plan scoping and access control enforcement.

---

## CISO Value Realization Framework

### 1) Risk Reduction (Security Outcome)
- **Before:** confidentiality depends heavily on process and privileged-user restraint.
- **After:** confidentiality is reinforced by cryptographic execution boundaries.

### 2) Auditability (Assurance Outcome)
- **Before:** proving that plaintext was not over-exposed is difficult and costly.
- **After:** audit-chain and control artifacts simplify evidence narratives for internal/external audit.

### 3) Resilience (Operational Outcome)
- **Before:** incidents in model-serving tiers can implicate broad data classes.
- **After:** reduced plaintext concentration can lower potential blast radius in compromise scenarios.

### 4) Procurement Confidence (Commercial Outcome)
- **Before:** enterprise buyers require extensive compensating controls for AI data handling.
- **After:** stronger security architecture improves trust posture for vendor assessments and DPAs.

---

## Evidence from Guidance, Literature, and Industry Reporting

Security leaders increasingly require alignment to recognized frameworks and observed attack patterns.

- NIST AI RMF highlights governance/accountability complexity and emphasizes measurable risk controls (NIST, 2023).
- CISA-led secure-AI development guidance calls out data security and secure-by-design implementation needs (CISA et al., 2024).
- ENISA threat landscape reporting reinforces that systemic cyber threats continue to evolve in ways that stress data-centric controls (ENISA, 2023).
- Verizon DBIR provides ongoing empirical evidence that data compromise remains a central incident impact domain (Verizon, 2024).
- OWASP Top 10 keeps sensitive information exposure as a practical and recurring risk class in internet-facing systems (OWASP Foundation, 2021).

> Board-level interpretation: the strategic issue is no longer “whether AI is valuable,” but whether security architecture can keep confidentiality risk within organizational risk appetite as AI scales.

---

## Innovation Novelty Narrative (Expanded)

Below is an expanded explanation of why the implemented innovations are operationally significant.

### 1. Leaf-Centric Encoding
**Novelty angle:** replaces sequential tree traversal assumptions with parallelized encrypted indicator computation patterns, improving viability under homomorphic constraints.

### 2. Gradient-Aware Noise Allocation
**Novelty angle:** treats precision as a budgeted security-performance variable informed by model structure, rather than uniform static quantization.

### 3. Homomorphic Ensemble Pruning
**Novelty angle:** enables significance-aware compute reduction without requiring plaintext model execution at runtime.

### 4. Polynomial Leaf Functions
**Novelty angle:** augments leaf expressiveness while preserving guardrailed validation logic for production safety.

### 5. MOAI-Native Tree Conversion
**Novelty angle:** focuses on rotation-cost minimization as a first-class systems problem in encrypted inference.

### 6. Bootstrap-Aligned Architecture
**Novelty angle:** aligns model/runtime orchestration with practical noise-budget economics and bootstrapping realities.

### 7. Federated Multi-Key Protocol
**Novelty angle:** addresses enterprise collaboration patterns where one-party trust assumptions are unacceptable.

### 8. Streaming Encrypted Gradients
**Novelty angle:** introduces encrypted adaptation pathways needed for near-real-time operational environments.

### 9. Unified Architecture Integration
**Novelty angle:** integrates innovations into an end-to-end deployable stack, reducing “research-only” fragmentation.

### SOTA positioning statement (practical)
In this repository’s scope, SOTA means:
1. The approach addresses known FHE bottlenecks,
2. The implementation is integrated into deployable services,
3. Bench/test artifacts exist to evaluate trade-offs,
4. Limitations are explicitly documented for safe rollout decisions.

---

## Implementation Guidance for Security Programs

### Phase 1 — Pilot with strict guardrails
- Select one high-value, high-sensitivity inference workload.
- Enforce tenant-level key separation and logging minimization policy.
- Define quantitative acceptance criteria (latency, error rate, evidence completeness).

### Phase 2 — Controlled production rollout
- Introduce canary cohorts and per-tenant policy gates.
- Require runbook validation and incident playbook updates.
- Establish routine evidence export for compliance/audit workflows.

### Phase 3 — Enterprise scaling
- Integrate third-party risk and architecture review checkpoints.
- Formalize SLO/SLA and escalation pathways with platform owners.
- Continuously review innovation toggles by workload risk profile.

---

## CISO FAQ

### Q1: Does this eliminate all privacy risk?
No. It materially reduces specific plaintext-exposure classes, but governance, IAM hygiene, software supply chain, and operational controls remain essential.

### Q2: Is this suitable for regulated workloads?
It is designed for regulated environments with control evidence in mind; final suitability depends on your legal/regulatory scope and control mapping.

### Q3: How should we evaluate it against alternatives?
Use a structured scorecard: confidentiality boundary strength, evidence quality, performance envelope, operational complexity, and incident blast-radius profile.

### Q4: What is the biggest deployment risk?
Treating advanced optimization modes as universally safe. Some model/task combinations require conservative settings and staged validation.

---

## Additional APA References (for security leadership context)

National Institute of Standards and Technology. (2024). *The NIST Cybersecurity Framework (CSF) 2.0*. https://www.nist.gov/cyberframework

European Union. (2016). *Regulation (EU) 2016/679 (General Data Protection Regulation)*. https://eur-lex.europa.eu/eli/reg/2016/679/oj

U.S. Department of Health & Human Services. (2013). *Summary of the HIPAA Security Rule*. https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html


---

## Preserved Legacy README (No Content Removed)

The following section is intentionally preserved from the prior README version to ensure no previously available documentation is removed.

# FHE Model Serving Platform

**Production-Grade Privacy-Preserving Inference for Trees, Linear Models, and GLMs**

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](/.github/workflows/ci.yml)
[![Security](https://img.shields.io/badge/security-hardened-blue)](/docs/THREAT_MODEL.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE)

---

## Why Us. Why Now. Why This Matters.

### The Privacy Crisis in Machine Learning

Every day, billions of machine learning predictions are made on sensitive data—medical diagnoses, financial risk scores, fraud detection, personalized recommendations. In every case, **your raw data is exposed to the server**. Healthcare providers see your symptoms. Banks see your transaction patterns. Advertisers see your preferences.

This isn't a bug—it's how ML has always worked. Until now.

### Why No One Has Done This Before

Fully Homomorphic Encryption (FHE) has been theoretically possible since 2009, but practical deployment has been blocked by three fundamental barriers:

| Barrier | Traditional FHE | Our Solution |
|---------|-----------------|--------------|
| **Performance** | 1,000,000x slowdown | **60ms latency** (100x improvement via MOAI optimizations) |
| **Complexity** | PhD-level cryptography required | **Drop-in SDK** for XGBoost/LightGBM/CatBoost |
| **Production Readiness** | Research prototypes only | **Enterprise-grade** with mTLS, RBAC, observability |

Previous attempts failed because they treated FHE as a cryptographic problem. We solved it as a **systems engineering problem**—optimizing the entire stack from model compilation to encrypted execution.

### Our Technological Moat

**1. MOAI (Model-Oblivious Architecture Interpreter)**
Our proprietary compiler transforms supported ML models (GBDTs, decision trees, random forests, logistic regression, and GLMs) into an FHE-optimized execution plan:
- **Column Packing**: Eliminates 90% of costly rotations through intelligent feature layout
- **Polynomial Step Functions**: Replaces lookup tables with smooth approximations
- **Interleaved Aggregation**: Processes all trees in parallel, not sequentially

**2. N2HE-HEXL Integration**
We've built the first production integration of the N2HE hybrid encryption scheme with Intel HEXL acceleration:
- **Hardware Acceleration**: AVX-512 and GPU NTT for 10x throughput
- **Noise Management**: Automatic bootstrapping to prevent decryption failures
- **128-bit Security**: Provably secure against quantum-resistant lattice attacks

**3. Universal Model Support**
Production support across tree ensembles and generalized linear models:
- **XGBoost / LightGBM / CatBoost** for gradient-boosted trees
- **Scikit-learn Decision Trees & Random Forests** for classical tabular ML
- **Scikit-learn Logistic Regression & Statsmodels GLM** for linear/probabilistic scoring

### What Our Product Does

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Your Data     │         │   Our Cloud     │         │   Your Result   │
│   (Encrypted)   │ ──────▶ │   (Blind)       │ ──────▶ │   (Decrypted)   │
│                 │         │                 │         │                 │
│ features=[...]  │         │ 🔒 Computes on  │         │ prediction=0.87 │
│ 🔐 AES+RLWE     │         │    ciphertext   │         │ 🔓 Only you can │
│                 │         │    only         │         │    decrypt      │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

**The server never sees your data. Ever.**

Your features are encrypted on your device using lattice-based cryptography. Our servers perform the entire GBDT inference—comparisons, tree traversals, aggregations—on encrypted data. Only you hold the secret key to decrypt the result.

### Who This Is For

| Use Case | Benefit |
|----------|---------|
| **Healthcare** | Run diagnostic models without exposing patient records |
| **Finance** | Credit scoring without revealing transaction history |
| **Insurance** | Risk assessment without accessing sensitive claims |
| **Government** | Citizen services without centralized data collection |
| **Enterprise** | Deploy ML models without data residency concerns |

---

## Performance Metrics

### Benchmark Results (2026-02-01)

| Model | Library | Encrypted P50 | Plaintext P50 | Overhead | Throughput |
|-------|---------|---------------|---------------|----------|------------|
| Classification | CatBoost | **62.03ms** | 0.18ms | 339x | ~600 eps |
| Classification | XGBoost | **61.68ms** | 0.63ms | 97x | ~500 eps |
| Regression | LightGBM | **62.47ms** | 0.72ms | 86x | ~450 eps |

*eps = encrypted predictions per second*

### Why CatBoost Performs Best

CatBoost uses **oblivious (symmetric) decision trees** where all nodes at the same depth use identical split conditions. This maps perfectly to FHE's SIMD operations:

| Metric | CatBoost | XGBoost | LightGBM |
|--------|----------|---------|----------|
| Crypto Rotations | **8** | 12 | 28 |
| Scheme Switches | **2** | 4 | 6 |
| Memory Transfers | **Low** | Medium | High |

**Recommendation**: Use CatBoost for new models. XGBoost/LightGBM supported for migration.

### Service Level Objectives

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency (p50) | < 50ms | Latency profile |
| Latency (p95) | < 100ms | Latency profile |
| Latency (p99) | < 200ms | Latency profile |
| Availability | 99.9% | Monthly uptime |
| Error Rate | < 0.1% | 5xx responses |
| Queue Depth | < 100 | Pending requests |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT SIDE                                    │
│  ┌──────────────┐                                                          │
│  │ Python SDK   │  • Key generation (secret + eval keys)                   │
│  │              │  • Feature encryption (RLWE ciphertexts)                 │
│  │              │  • Result decryption                                     │
│  └──────┬───────┘                                                          │
└─────────┼──────────────────────────────────────────────────────────────────┘
          │ gRPC + mTLS
          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              SERVER SIDE                                    │
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │   Gateway    │───▶│   Registry   │    │   Keystore   │                 │
│  │              │    │              │    │              │                 │
│  │ • Auth/AuthZ │    │ • Model meta │    │ • Eval keys  │                 │
│  │ • Rate limit │    │ • Plan store │    │ • Envelope   │                 │
│  │ • OTel trace │    │ • Versioning │    │   encryption │                 │
│  └──────┬───────┘    └──────────────┘    └──────────────┘                 │
│         │                                                                  │
│         ▼                                                                  │
│  ┌──────────────┐    ┌──────────────┐                                     │
│  │   Compiler   │───▶│   Runtime    │                                     │
│  │              │    │              │                                     │
│  │ • GBDT libs  │    │ • C++ engine │                                     │
│  │ • RF / DT    │    │ • N2HE-HEXL  │                                     │
│  │ • LogReg/GLM │    │ • GPU accel  │                                     │
│  │ • MOAI optim │    │ • SIMD/AVX2  │                                     │
│  └──────────────┘    └──────────────┘                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Gateway** | Go + gRPC | Authentication, rate limiting, request routing |
| **Registry** | Go + PostgreSQL | Model metadata, compiled plan storage |
| **Keystore** | Go + Vault | Evaluation key management with envelope encryption |
| **Compiler** | Python | Multi-model (GBDT/RF/DT/LogReg/GLM) → ObliviousPlanIR transformation |
| **Runtime** | C++ | FHE execution engine with N2HE-HEXL |
| **SDK** | Python | Client-side crypto and API interface |

---

## Quickstart

### 5-Minute Setup

```bash
# Clone the repository
git clone https://github.com/your-org/fhe-gbdt-serving.git
cd fhe-gbdt-serving

# Start all services
make docker-up

# Run the complete cookbook (trains models, encrypts, predicts)
make cookbook
```

### Python SDK Example

```python
from fhe_gbdt_sdk import FHEGBDTClient
from fhe_gbdt_sdk.crypto import N2HEKeyManager

# Initialize crypto (client-side)
key_manager = N2HEKeyManager("my-tenant-id")
key_manager.generate_keys()

# Connect to gateway
client = FHEGBDTClient(
    gateway_url="localhost:8080",
    key_manager=key_manager
)

# Upload evaluation keys (one-time)
client.upload_eval_keys()

# Register and compile your model
model_id = client.register_model("my_xgboost_model.json", "xgboost")
compiled_id = client.compile_model(model_id)

# Encrypt features and predict
features = [5.1, 3.5, 1.4, 0.2]  # Iris flower measurements
encrypted_features = key_manager.encrypt(features)

# Server computes on encrypted data
encrypted_result = client.predict(compiled_id, encrypted_features)

# Only you can decrypt
prediction = key_manager.decrypt(encrypted_result)
print(f"Prediction: {prediction}")  # e.g., [0.92, 0.05, 0.03]
```

---

## Cookbook Recipes

Detailed end-to-end examples for common use cases:

| Recipe | Description | Dataset |
|--------|-------------|---------|
| [00 Quickstart](docs/cookbook/00_quickstart.md) | 5-minute introduction | Synthetic |
| [01 XGBoost Classification](docs/cookbook/01_xgboost_classification.md) | Binary classification | Breast Cancer |
| [02 LightGBM Regression](docs/cookbook/02_lightgbm_regression.md) | Continuous prediction | California Housing |
| [03 CatBoost Classification](docs/cookbook/03_catboost_classification.md) | Multi-class with oblivious trees | Iris |
| [04 Trade-offs](docs/cookbook/04_tradeoffs.md) | Performance comparison guide | - |
| [05 Troubleshooting](docs/cookbook/05_troubleshooting.md) | Common issues & solutions | - |

---

## Security

### Cryptographic Foundation

| Parameter | Value | Security Level |
|-----------|-------|----------------|
| Scheme | N2HE (RLWE + LWE hybrid) | 128-bit |
| Ring Dimension | 4096 / 8192 | Post-quantum secure |
| Ciphertext Modulus | 2^32 | - |
| Error Distribution | Gaussian, σ=3.2 | - |

### Defense in Depth

| Layer | Control | Implementation |
|-------|---------|----------------|
| **Transport** | mTLS | All service-to-service communication |
| **Authentication** | API Key + Tenant ID | Gateway enforcement |
| **Authorization** | RBAC | Tenant → Model → Plan scoping |
| **Data at Rest** | Envelope Encryption | Keystore (Vault backend) |
| **Audit** | Structured Logging | Request ID + Tenant ID tracing |
| **Supply Chain** | SBOM + Scanning | SAST, dependency, container scans |

### Threat Model

We follow [STRIDE](docs/THREAT_MODEL.md) methodology:

| Threat | Mitigation |
|--------|------------|
| **Spoofing** | API key binding to tenant_id |
| **Tampering** | Content-addressed plan IDs (SHA-256) |
| **Repudiation** | Immutable audit logs |
| **Information Disclosure** | FHE encryption (features never plaintext on server) |
| **Denial of Service** | Per-tenant rate limiting (100 req/s, burst 200) |
| **Elevation of Privilege** | Tenant isolation, no cross-tenant access |

---

## Production Deployment

### Kubernetes (Recommended)

```bash
# Install with Helm
helm install fhe-gbdt ./deploy/helm \
  --set gateway.replicas=3 \
  --set runtime.replicas=5 \
  --set runtime.gpu.enabled=true
```

### Configuration

See [config/production.py](config/production.py) for all options:

```python
from config import ProductionConfig, SecurityLevel

config = ProductionConfig.from_env()
config.crypto.security_level = SecurityLevel.HIGH  # 192-bit
config.tls.require_client_cert = True
config.rate_limit.requests_per_second = 100
```

### Observability

| Tool | Purpose | Dashboard |
|------|---------|-----------|
| **Prometheus** | Metrics collection | `dashboards/grafana/slo.json` |
| **Grafana** | Visualization | SLO dashboard included |
| **Jaeger** | Distributed tracing | OpenTelemetry integration |
| **AlertManager** | Alerting | SLO-based rules |

---

## Testing

### Test Suite

```bash
# Unit tests (20 tests)
make test

# Integration tests - AI Engineer workflow (17 tests)
python -m pytest tests/integration/test_ai_engineer_workflow.py -v

# E2E tests with real models (4 tests)
make e2e-real

# Full test suite (44 tests)
python -m pytest tests/ -v
```

### Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Unit** | 20 | Parser, IR, kernel algebra, security patterns |
| **Integration** | 17 | Full AI engineer workflow (train → encrypt → predict) |
| **E2E** | 10 | Real XGBoost/LightGBM/CatBoost models |
| **Fuzz** | 2 | Parser robustness against malformed input |
| **Metamorphic** | 3 | Property-based testing (Hypothesis) |

### CI/CD Pipeline

```yaml
Jobs:
  - build-and-test    # Build + unit tests + security scan + SBOM
  - guardrails        # Forbidden pattern detection (no plaintext logging)
  - cookbook-tests    # E2E with real ML models
  - integration-tests # AI engineer workflow validation
  - unit-tests        # Fast feedback on PRs
```

---

## Production Readiness

### Checklist Status: 27/27 ✅

| Category | Items | Status |
|----------|-------|--------|
| **Security** | mTLS, AuthZ, secrets, scanning | 8/8 ✅ |
| **Correctness** | E2E regression, metamorphic tests | 5/5 ✅ |
| **Performance** | Benchmarks, baselines, monitoring | 4/4 ✅ |
| **Reliability** | Load tests, soak tests, error budget | 4/4 ✅ |
| **Operability** | Dashboards, alerts, runbooks | 6/6 ✅ |

See [PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for full details.

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and component details |
| [THREAT_MODEL.md](docs/THREAT_MODEL.md) | STRIDE security analysis |
| [SLO.md](docs/SLO.md) | Service level objectives and error budgets |
| [CRYPTO_PARAMS.md](docs/CRYPTO_PARAMS.md) | Cryptographic parameter selection |
| [RUNBOOKS.md](docs/RUNBOOKS.md) | Operational procedures |
| [OPERATIONS.md](docs/OPERATIONS.md) | Day-to-day operational guide |

---

## Development

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Go 1.21+
- (Optional) CUDA 11+ for GPU acceleration

### Commands

```bash
make build          # Build all Docker images
make docker-up      # Start development stack
make stop           # Stop all services
make test           # Run unit tests
make e2e-real       # Run E2E tests with real models
make cookbook       # Run all cookbook recipes
make clean          # Clean build artifacts
```

### Project Structure

```
fhe-gbdt-serving/
├── services/
│   ├── gateway/        # Go - API gateway
│   ├── runtime/        # C++ - FHE execution engine
│   ├── compiler/       # Python - Model compiler
│   ├── registry/       # Go - Model registry
│   └── keystore/       # Go - Key management
├── sdk/python/         # Python client SDK
├── proto/              # gRPC service definitions
├── tests/              # Test suite
├── docs/               # Documentation
├── bench/              # Benchmarking tools
├── deploy/             # Kubernetes/Helm charts
├── dashboards/         # Grafana dashboards
└── config/             # Configuration modules
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Security Vulnerabilities

Please report security issues to security@example.com. We respond within 48 hours and fix critical issues within 14 days.

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [N2HE](https://github.com/n2he) - Hybrid FHE library
- [Intel HEXL](https://github.com/intel/hexl) - Hardware acceleration
- [XGBoost](https://xgboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/) - GBDT libraries

---

<p align="center">
  <b>Privacy is not a feature. It's a right.</b><br>
  <i>FHE-GBDT Serving - Inference without exposure.</i>
</p>

---

## Data Collaboration Without Data Sharing: Why This Unblocks Banking and Healthcare

For many CISOs, the core blocker in enterprise ML is not model quality—it is **data-sharing risk** across subsidiaries, counterparties, clinical partners, and vendors. Teams can build models, but legal, security, and regulatory constraints often stop deployment where cross-organization data is required.

This platform addresses that bottleneck by enabling inference on encrypted features so data owners do not need to disclose raw inputs to the serving environment.

### What this changes for security programs

- **Traditional pattern:** centralize or replicate sensitive data, then protect it with compensating controls.
- **Confidential-compute pattern here:** keep feature-level confidentiality boundary at the data owner, execute encrypted inference, and only decrypt results where policy allows.

For security leadership, this shifts control design from “who can access plaintext” to “where plaintext is ever required.”

### Evidence from industry and policy reporting

#### 1) Financial services: data sharing and fraud/AML collaboration barriers

- The World Economic Forum documents that financial crime controls depend on institution-to-institution collaboration and that data-sharing constraints are a major friction point for collective defense programs (World Economic Forum, 2020).
- The UK FCA/ICO joint guidance on privacy-enhancing technologies highlights PETs as mechanisms to enable analytics while reducing personal data exposure in regulated contexts (Information Commissioner’s Office & Financial Conduct Authority, 2023).
- Major banking privacy incidents continue to show high impact from data exposure events, reinforcing CISO focus on minimizing plaintext proliferation in analytic pipelines (Verizon, 2024).

**Operational implication:** encrypted inference pathways can support collaborative risk scoring and model-serving use cases with lower cross-entity disclosure pressure.

#### 2) Healthcare: interoperability demand vs. PHI disclosure risk

- Industry and policy literature consistently notes that healthcare AI value depends on multi-institution data breadth, while PHI constraints and breach risks impede central data pooling (ENISA, 2023; HHS, 2013).
- HHS OCR breach reporting trends and healthcare incident analyses repeatedly show the sensitivity and legal impact of unauthorized health data disclosure, increasing the burden on CISOs to reduce exposure-by-design (U.S. Department of Health & Human Services, n.d.; Verizon, 2024).
- PET guidance from regulators and standards bodies points to privacy-preserving analytics as a practical route for extracting insight without equivalent raw-data movement (Information Commissioner’s Office & Financial Conduct Authority, 2023; NIST, 2023).

**Operational implication:** encrypted inference supports diagnostic and risk models where hospitals/partners can consume model value without handing over raw patient features to central serving tiers.

### Illustrative deployment patterns (production-oriented)

1. **Bank consortium risk scoring**  
   Member institutions submit encrypted feature vectors to a shared model-serving fabric; only authorized members decrypt final scores.

2. **Cross-hospital clinical decision support**  
   Participating providers run encrypted inference against shared models without exposing raw PHI to the central runtime operator.

3. **Payer-provider fraud analytics**  
   High-risk claims are scored through encrypted workflows, reducing bilateral data transfer and plaintext handling obligations.

### CISO decision criteria for this capability

Evaluate readiness with a security governance scorecard:

- **Plaintext minimization:** Is plaintext avoided in shared serving infrastructure?
- **Tenant/key isolation:** Are keys and model scopes partitioned by policy boundary?
- **Evidence generation:** Can audit/compliance teams trace controls without disclosing sensitive data?
- **Incident containment:** Does architecture reduce blast radius if one control plane is compromised?
- **Regulatory fit:** Can control narratives map to HIPAA/GDPR/sector expectations for confidentiality and accountability?

### Implementation guardrails (recommended)

- Start with high-sensitivity, bounded-scope models (fraud triage, readmission risk, claims routing).
- Enforce strict key lifecycle ownership and dual-control for key management changes.
- Add policy checks to prevent fallback routes that reintroduce plaintext at integration boundaries.
- Run tabletop incident exercises specifically for encrypted-inference failure modes (key loss, ciphertext replay, mis-scoped tenant access).

---

## Additional References for Data-Sharing Bottlenecks (APA 7th)

Information Commissioner’s Office, & Financial Conduct Authority. (2023). *Privacy-enhancing technologies guidance*. https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/privacy-enhancing-technologies/

U.S. Department of Health & Human Services. (n.d.). *Breach portal: Notice to the Secretary of HHS breach of unsecured protected health information*. https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf

World Economic Forum. (2020). *The new frontier in financial crime: From defensive to offensive measures*. https://www.weforum.org/reports/the-new-frontier-in-financial-crime/


## System Gap Review (No Mocks)

For a code-referenced production gap assessment and remediation plan, see [docs/PRODUCTION_NO_MOCKS_GAP_REVIEW.md](docs/PRODUCTION_NO_MOCKS_GAP_REVIEW.md).

---

## Step-by-Step Case Study: Organization A Lets Organization B Compute on A's Data Without Seeing It

### Scenario

- **Organization A (Data Owner):** A regional hospital network that has sensitive patient records.
- **Organization B (Model Operator):** A specialized analytics company with a validated sepsis-risk model.
- **Goal:** Organization B runs inference on Organization A’s patient features to generate clinical risk insights, **without B ever seeing patient plaintext data**.

### Trust Model

- A does **not** share raw feature vectors with B.
- B does **not** hold A’s decryption key.
- Runtime operators only process ciphertext.
- Final predictions are only decrypted by A.

### Architecture Mapping in This Repository

- A uses SDK + client-side encryption and key management.
- B hosts model registration/compilation and runtime services.
- Keystore stores evaluation keys only (not secret keys).
- Gateway enforces tenant/model controls and forwards encrypted inference.

---

### Implementation Walkthrough (Production Sequence)

#### Step 1) Legal, governance, and technical boundary definition

A and B agree on:
1. A remains data controller/owner of raw patient features.
2. B operates model-serving infrastructure but cannot decrypt A data.
3. Allowed model purpose, retention, and audit requirements.
4. Incident response ownership and escalation paths.

**Deliverable:** data processing agreement + control matrix mapping to technical controls.

#### Step 2) Organization B onboards model as a bounded service

1. B registers the model in registry.
2. B compiles it to encrypted execution plan.
3. B publishes model/version/profile metadata to A.

**Deliverable:** `model_id`, `compiled_model_id`, expected feature schema.

#### Step 3) Organization A generates keys locally

1. A uses local SDK key manager to generate secret key + evaluation keys.
2. Secret key remains in A’s controlled environment.
3. A uploads evaluation keys to keystore.

**Security property:** B can evaluate encrypted data but cannot decrypt input or output.

#### Step 4) A encrypts inference features on-prem

1. A’s EHR pipeline extracts required feature vector.
2. A normalizes/transforms locally using approved preprocessing.
3. A encrypts the vector client-side before network transfer.

**Security property:** no raw patient features cross trust boundary.

#### Step 5) Gateway and runtime process ciphertext only

1. Gateway authenticates tenant, validates model scope, forwards request.
2. Runtime loads compiled plan and evaluation keys.
3. Runtime computes prediction over ciphertext.
4. Service returns encrypted result payload.

**Operational property:** runtime disconnect/failure returns explicit error, not simulated success.

#### Step 6) A decrypts and operationalizes insight

1. A decrypts response locally.
2. A writes risk score/insight into care workflow.
3. A stores only policy-approved metadata in audit systems.

**Outcome:** A gets model insight; B never sees patient plaintext.

---

### Example Data Flow (Single Inference)

1. A input (plaintext): `[age, lactate, heart_rate, wbc, ...]` (local only)
2. A output (ciphertext): `C(features)` -> sent to B runtime
3. B runtime output (ciphertext): `C(risk_score)` -> returned to A
4. A decrypts: `risk_score = 0.82` and triggers care pathway

At no point does B receive plaintext features or plaintext risk context tied to patient-level raw inputs.

---

### Measurable Success Criteria for the Pilot

- **Confidentiality KPI:** 0 plaintext feature payloads outside A boundary.
- **Reliability KPI:** >=99.9% successful encrypted inference over pilot window.
- **Security KPI:** 100% requests with tenant/model audit trace.
- **Clinical/Business KPI:** agreed threshold for precision/recall and intervention lift.

### Production Hardening Checklist for This Use Case

- Disable SDK simulation in production (`ALLOW_SDK_SIMULATION=false`).
- Enforce durable storage for keystore and registry paths.
- Require dependency-aware readiness checks for go-live.
- Add runbook drills for runtime unavailable, key retrieval failures, and rollback.
- Pin CI dependencies and require green build before model/version promotion.

### Why CISOs Like This Pattern

- It enables collaboration value (A uses B’s model expertise) without data surrender.
- It materially reduces breach blast radius from centralized plaintext ML serving.
- It creates auditable separation of duties between data owner and compute operator.
