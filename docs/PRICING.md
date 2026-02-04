# FHE-GBDT Pricing

> **Effective Date**: February 4, 2026
> **Version**: 1.0.0
> **Aligned With**: TenSafe Pricing Structure

## 1. Overview

FHE-GBDT offers flexible pricing tiers designed to support teams from experimentation through enterprise deployment. All tiers include our core privacy-preserving GBDT inference capabilities.

## 2. Pricing Tiers

### 2.1 Free Tier

**$0/month**

Perfect for experimentation and development.

| Feature | Limit |
|---------|-------|
| API Requests | 10,000/month |
| Encrypted Predictions | 5,000/month |
| Models | 3 |
| Storage | 1 GB |
| Compiled Plans | 5 |
| Support | Community |
| SLA | None |

**Includes**:
- Full FHE inference capabilities
- Python SDK
- Basic documentation
- Community forum access

### 2.2 Pro Tier

**$99/month** or **$79/month** (billed annually)

For individuals and small teams building privacy-first applications.

| Feature | Limit |
|---------|-------|
| API Requests | 500,000/month |
| Encrypted Predictions | 250,000/month |
| Models | 25 |
| Storage | 50 GB |
| Compiled Plans | 50 |
| Support | Email (48h response) |
| SLA | 99.5% uptime |

**Includes everything in Free, plus**:
- Priority compilation queue
- TypeScript SDK
- CLI tool
- API versioning guarantees
- Basic analytics dashboard
- Email support

### 2.3 Business Tier

**$499/month** or **$399/month** (billed annually)

For growing teams requiring production-grade features.

| Feature | Limit |
|---------|-------|
| API Requests | 5,000,000/month |
| Encrypted Predictions | 2,500,000/month |
| Models | Unlimited |
| Storage | 500 GB |
| Compiled Plans | Unlimited |
| Training Jobs | 100/month |
| Support | Priority (4h response) |
| SLA | 99.9% uptime |

**Includes everything in Pro, plus**:
- Privacy-preserving GBDT training
- Differential privacy support
- GBSP secure packaging
- SSO/SAML authentication
- Full RBAC
- Advanced analytics
- Custom model compilation profiles
- Dedicated account manager

### 2.4 Enterprise Tier

**Custom Pricing**

For organizations with advanced security and compliance requirements.

| Feature | Limit |
|---------|-------|
| API Requests | Custom |
| Encrypted Predictions | Unlimited |
| Models | Unlimited |
| Storage | Custom |
| Training Jobs | Unlimited |
| Support | 24/7 (15min response) |
| SLA | 99.99% uptime |

**Includes everything in Business, plus**:
- Dedicated infrastructure
- On-premise deployment option
- Custom SLA terms
- HIPAA BAA
- SOC 2 Type II report access
- ISO 27001/27701 certification
- Custom compliance controls
- Dedicated security review
- Custom integrations
- Training and onboarding
- Quarterly business reviews

## 3. Feature Comparison

| Feature | Free | Pro | Business | Enterprise |
|---------|:----:|:---:|:--------:|:----------:|
| **Inference** |
| FHE Encryption | ✓ | ✓ | ✓ | ✓ |
| XGBoost Support | ✓ | ✓ | ✓ | ✓ |
| LightGBM Support | ✓ | ✓ | ✓ | ✓ |
| CatBoost Support | ✓ | ✓ | ✓ | ✓ |
| Batch Predictions | - | ✓ | ✓ | ✓ |
| Custom Profiles | - | - | ✓ | ✓ |
| **Training** |
| DP-GBDT Training | - | - | ✓ | ✓ |
| Federated Training | - | - | - | ✓ |
| Custom Loss Functions | - | - | ✓ | ✓ |
| **Security** |
| API Key Auth | ✓ | ✓ | ✓ | ✓ |
| mTLS | - | ✓ | ✓ | ✓ |
| SSO/SAML | - | - | ✓ | ✓ |
| RBAC | Basic | Basic | Full | Custom |
| Audit Logs | 7 days | 30 days | 1 year | Custom |
| GBSP Packaging | - | - | ✓ | ✓ |
| **Compliance** |
| SOC 2 | - | - | Report | Report + Audit |
| HIPAA | - | - | - | BAA Available |
| ISO 27001 | - | - | - | Certificate |
| ISO 27701 | - | - | - | Certificate |
| **SDKs** |
| Python SDK | ✓ | ✓ | ✓ | ✓ |
| TypeScript SDK | - | ✓ | ✓ | ✓ |
| CLI Tool | - | ✓ | ✓ | ✓ |
| **Support** |
| Documentation | ✓ | ✓ | ✓ | ✓ |
| Community Forum | ✓ | ✓ | ✓ | ✓ |
| Email Support | - | ✓ | ✓ | ✓ |
| Priority Support | - | - | ✓ | ✓ |
| Phone Support | - | - | - | ✓ |
| Dedicated Engineer | - | - | - | ✓ |

## 4. Usage-Based Pricing

### 4.1 Prediction Pricing (Beyond Tier Limits)

| Volume (monthly) | Price per 1,000 predictions |
|------------------|----------------------------|
| First 100,000 | Included in tier |
| 100,001 - 500,000 | $0.50 |
| 500,001 - 2,000,000 | $0.40 |
| 2,000,001 - 10,000,000 | $0.30 |
| 10,000,001+ | Custom |

### 4.2 Training Pricing (Business & Enterprise)

| Resource | Price |
|----------|-------|
| Training job (per hour) | $2.00 |
| GPU training (per hour) | $5.00 |
| DP accountant compute | $0.10/job |

### 4.3 Storage Pricing (Beyond Tier Limits)

| Storage Type | Price per GB/month |
|--------------|-------------------|
| Model storage | $0.10 |
| Compiled plan storage | $0.15 |
| Audit log retention | $0.05 |

## 5. Billing Details

### 5.1 Payment Methods

- Credit/Debit Card (Visa, Mastercard, Amex)
- ACH Bank Transfer (Business & Enterprise)
- Wire Transfer (Enterprise)
- Invoice (Enterprise, Net-30)

### 5.2 Billing Cycle

- **Monthly**: Billed on subscription start date
- **Annual**: Billed upfront, 20% discount

### 5.3 Overage Handling

| Tier | Overage Policy |
|------|----------------|
| Free | Service paused until next month |
| Pro | Automatic usage-based billing |
| Business | Automatic usage-based billing |
| Enterprise | Custom arrangement |

### 5.4 Proration

- Upgrades: Prorated immediately
- Downgrades: Applied at next billing cycle
- Cancellations: No refund for partial month

## 6. Free Trial

### 6.1 Pro Trial

- **Duration**: 14 days
- **Credit Card**: Not required
- **Limits**: Full Pro tier access
- **Conversion**: Automatic downgrade to Free if not converted

### 6.2 Enterprise Evaluation

- **Duration**: 30 days (negotiable)
- **Setup**: Dedicated onboarding call
- **Limits**: Custom based on evaluation scope
- **Conversion**: Custom contract negotiation

## 7. Discounts

### 7.1 Standard Discounts

| Discount Type | Amount | Eligibility |
|---------------|--------|-------------|
| Annual billing | 20% | All paid tiers |
| Startup program | 50% | <$5M funding, <50 employees |
| Academic | 75% | Verified .edu email |
| Non-profit | 50% | 501(c)(3) verified |

### 7.2 Volume Discounts

| Annual Commitment | Discount |
|-------------------|----------|
| $10,000 - $50,000 | 10% |
| $50,001 - $100,000 | 15% |
| $100,001 - $500,000 | 20% |
| $500,001+ | Custom |

## 8. Refund Policy

### 8.1 Monthly Subscriptions

- **Cancellation**: Service continues until end of billing period
- **Refunds**: Not available for partial months
- **Downgrades**: Applied at next billing cycle

### 8.2 Annual Subscriptions

- **30-day guarantee**: Full refund within 30 days
- **After 30 days**: Prorated refund minus 2-month penalty
- **Enterprise**: Per contract terms

## 9. Service Level Agreements

### 9.1 Uptime Guarantees

| Tier | Uptime SLA | Monthly Credit |
|------|------------|----------------|
| Free | None | N/A |
| Pro | 99.5% | 10% at 99.0%, 25% at 98.0% |
| Business | 99.9% | 10% at 99.5%, 25% at 99.0%, 50% at 98.0% |
| Enterprise | 99.99% | Per contract |

### 9.2 Performance SLAs (Business & Enterprise)

| Metric | Target |
|--------|--------|
| Inference latency (p50) | <100ms |
| Inference latency (p99) | <500ms |
| Compilation time (p50) | <30s |
| API response time (p50) | <200ms |

### 9.3 Support Response SLAs

| Severity | Pro | Business | Enterprise |
|----------|-----|----------|------------|
| Critical (P1) | 24h | 4h | 15min |
| High (P2) | 48h | 8h | 1h |
| Medium (P3) | 5 days | 24h | 4h |
| Low (P4) | 10 days | 5 days | 24h |

## 10. Contact Sales

For Enterprise pricing or custom requirements:

- **Email**: sales@fhe-gbdt.example.com
- **Phone**: +1 (555) 123-4567
- **Schedule Demo**: https://fhe-gbdt.example.com/demo

## 11. FAQ

**Q: Can I switch tiers mid-cycle?**
A: Yes. Upgrades are immediate and prorated. Downgrades apply at the next billing cycle.

**Q: What happens if I exceed my limits?**
A: Free tier pauses. Paid tiers automatically bill for overage at published rates.

**Q: Do you offer reserved capacity?**
A: Yes, for Enterprise customers. Contact sales for reserved pricing.

**Q: Is there a minimum commitment?**
A: No minimum for Free, Pro, or Business. Enterprise typically requires 12-month commitment.

**Q: Can I get a custom tier?**
A: Enterprise tier is fully customizable. Contact sales for requirements.

---

*Last Updated*: February 4, 2026
*Next Review*: August 4, 2026
*Owner*: Product & Sales Team
