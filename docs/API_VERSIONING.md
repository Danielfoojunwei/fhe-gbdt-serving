# FHE-GBDT API Versioning and Deprecation Policy

> **Effective Date**: February 4, 2026
> **Version**: 1.0.0
> **Aligned With**: TenSafe API Versioning Policy

## 1. Overview

This document establishes the API versioning strategy, backwards compatibility guarantees, and deprecation procedures for the FHE-GBDT platform. Our goal is to provide a stable, predictable API while enabling continuous improvement.

## 2. Versioning Strategy

### 2.1 URL Path Versioning

All REST API endpoints use URL path versioning:

```
https://api.fhe-gbdt.example.com/v1/predict
https://api.fhe-gbdt.example.com/v1/models
https://api.fhe-gbdt.example.com/v2/predict  (future)
```

### 2.2 gRPC Service Versioning

gRPC services include version in package name:

```protobuf
package fhe_gbdt.v1;

service InferenceService {
  rpc Predict(PredictRequest) returns (PredictResponse);
}
```

### 2.3 SDK Semantic Versioning

SDKs follow [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

Examples:
- 1.0.0 → 1.0.1  (Patch: bug fix)
- 1.0.1 → 1.1.0  (Minor: new feature, backwards compatible)
- 1.1.0 → 2.0.0  (Major: breaking change)
```

## 3. Version Lifecycle

### 3.1 Version States

| State | Description | Support Level |
|-------|-------------|---------------|
| **Current** | Latest stable version | Full support, new features |
| **Supported** | Previous major version | Security patches, critical bugs |
| **Deprecated** | Scheduled for removal | Security patches only, migration warnings |
| **Sunset** | No longer available | Returns 410 Gone |

### 3.2 Support Timeline

| Version | State | Support End Date |
|---------|-------|------------------|
| v1 | Current | At least February 4, 2028 |

### 3.3 Minimum Support Period

- **Major versions**: Minimum 24 months support
- **Minor versions**: Minimum 12 months support
- **Deprecation notice**: Minimum 12 months before sunset

## 4. Backwards Compatibility

### 4.1 Compatibility Guarantees

We guarantee the following will NOT change within a major version:

| Guarantee | Description |
|-----------|-------------|
| **Endpoint URLs** | Existing endpoints remain accessible |
| **Required Parameters** | No new required parameters added |
| **Response Fields** | Existing fields not removed or renamed |
| **Field Types** | Data types remain consistent |
| **Error Codes** | Existing error codes maintain meaning |
| **Authentication** | Auth methods remain supported |

### 4.2 Non-Breaking Changes

The following changes are considered non-breaking and may occur in minor releases:

| Change Type | Example |
|-------------|---------|
| New optional parameters | Adding `timeout_ms` parameter |
| New response fields | Adding `metrics.rotations` |
| New endpoints | Adding `/v1/batch/predict` |
| New error codes | Adding `NOISE_BUDGET_EXCEEDED` |
| Performance improvements | Faster inference |
| Documentation updates | Clarifying behavior |

### 4.3 Breaking Changes

The following require a new major version:

| Change Type | Example |
|-------------|---------|
| Endpoint removal | Removing `/v1/legacy/predict` |
| Field removal | Removing `response.deprecated_field` |
| Type changes | Changing `count` from int to string |
| Required parameters | Making `profile` required |
| Authentication changes | Requiring new auth method |
| Error format changes | Restructuring error responses |
| Behavioral changes | Changing default batch size |

## 5. Deprecation Process

### 5.1 Timeline

```
Day 0          Month 6        Month 12       Month 12+
  │              │              │              │
  ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Announce  │  │ Warning  │  │Migration │  │ Sunset   │
│          │──│  Period  │──│  Period  │──│          │
│          │  │          │  │          │  │410 Gone  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 5.2 Deprecation Announcement

At announcement (Day 0):
- Changelog entry published
- Documentation updated with deprecation notice
- Email notification to affected users
- Status page announcement

### 5.3 Warning Period (Months 1-6)

During warning period:
- Deprecated features continue working
- Response headers indicate deprecation:
  ```
  Deprecation: true
  Sunset: Sat, 04 Feb 2028 00:00:00 GMT
  Link: <https://docs.fhe-gbdt.example.com/migration/v2>; rel="deprecation"
  ```
- SDK warnings logged on deprecated method usage

### 5.4 Migration Period (Months 7-12)

During migration period:
- Reduced SLA (99.5% vs 99.9%)
- Security patches only
- Active migration support
- Monthly reminders

### 5.5 Sunset

At sunset:
- Endpoint returns `410 Gone`
- Response body includes migration guidance:
  ```json
  {
    "error": {
      "code": "ENDPOINT_SUNSET",
      "message": "This endpoint has been sunset. Please migrate to v2.",
      "migration_guide": "https://docs.fhe-gbdt.example.com/migration/v2"
    }
  }
  ```

## 6. Current API Status

### 6.1 v1 (Current)

**Released**: February 4, 2026
**Status**: Current
**Supported Until**: At least February 4, 2028

#### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/predict` | POST | Encrypted prediction |
| `/v1/batch/predict` | POST | Batch encrypted prediction |
| `/v1/models` | GET | List models |
| `/v1/models` | POST | Register model |
| `/v1/models/{id}` | GET | Get model details |
| `/v1/models/{id}` | DELETE | Delete model |
| `/v1/models/{id}/compile` | POST | Compile model for FHE |
| `/v1/keys` | POST | Upload evaluation keys |
| `/v1/keys/{id}` | DELETE | Revoke keys |
| `/v1/training/jobs` | POST | Start training job |
| `/v1/training/jobs/{id}` | GET | Get training status |
| `/v1/packages` | POST | Create GBSP package |
| `/v1/packages/{id}/verify` | POST | Verify package |
| `/v1/health` | GET | Health check |
| `/v1/ready` | GET | Readiness check |

#### gRPC Services

| Service | Methods |
|---------|---------|
| `InferenceService` | Predict, BatchPredict |
| `ControlService` | RegisterModel, CompileModel, DeleteModel |
| `CryptoKeyService` | UploadEvalKeys, RotateKeys, RevokeKeys |
| `TrainingService` | StartTraining, GetStatus, StopTraining |
| `PackagingService` | CreatePackage, VerifyPackage, ExtractPackage |

### 6.2 v2 (Planned)

**Expected Release**: Q3 2026
**Status**: Planning

Planned enhancements:
- GraphQL API option
- Enhanced streaming inference
- Batch operations optimization
- Improved error handling with more granular codes
- Async training callbacks
- Multi-region support

## 7. Migration Support

### 7.1 Support by Tier

| Tier | Migration Support |
|------|-------------------|
| **Free** | Documentation only |
| **Pro** | Documentation + email support |
| **Business** | Dedicated migration guide + priority support |
| **Enterprise** | Dedicated engineer + custom timeline |

### 7.2 Migration Tools

We provide tools to assist migration:

```bash
# Check API compatibility
fhe-gbdt api check-compatibility --version v2

# Dry-run migration
fhe-gbdt api migrate --from v1 --to v2 --dry-run

# Generate migration report
fhe-gbdt api migration-report --output report.md
```

### 7.3 Compatibility Mode

During migration, enable compatibility mode:

```python
from fhe_gbdt import Client

client = Client(
    api_version="v2",
    compatibility_mode="v1"  # Accept v1 responses
)
```

## 8. SDK Versioning

### 8.1 Python SDK

```python
# Check SDK version
import fhe_gbdt
print(fhe_gbdt.__version__)  # "1.0.0"

# Check API version
client = fhe_gbdt.Client()
print(client.api_version)  # "v1"
```

### 8.2 TypeScript SDK

```typescript
import { FHEGBDTClient, VERSION, API_VERSION } from '@fhe-gbdt/sdk';

console.log(VERSION);      // "1.0.0"
console.log(API_VERSION);  // "v1"
```

### 8.3 CLI

```bash
# Check versions
fhe-gbdt version
# CLI Version: 1.0.0
# API Version: v1
# Server Version: 1.2.3
```

## 9. Changelog

### v1.0.0 (February 4, 2026)

**Initial Release**

- REST API with `/v1/` prefix
- gRPC services (Inference, Control, CryptoKey, Training, Packaging)
- Python SDK
- TypeScript SDK
- CLI tool
- GBSP packaging format

## 10. Notification Channels

Stay informed about API changes:

| Channel | URL |
|---------|-----|
| **Changelog** | https://docs.fhe-gbdt.example.com/changelog |
| **Status Page** | https://status.fhe-gbdt.example.com |
| **GitHub Releases** | https://github.com/example/fhe-gbdt/releases |
| **Developer Portal** | https://developers.fhe-gbdt.example.com |
| **Email List** | Subscribe at developer portal |

## 11. Contact

For API versioning questions:
- **Email**: api-support@fhe-gbdt.example.com
- **GitHub Issues**: https://github.com/example/fhe-gbdt/issues
- **Developer Slack**: #fhe-gbdt-api

---

*Last Updated*: February 4, 2026
*Next Review*: August 4, 2026
*Owner*: Platform Engineering Team
