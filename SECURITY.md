# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities via email to:

📧 **security@fhe-gbdt.dev**

### What to Include

Please include the following in your report:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Affected versions**
4. **Potential impact**
5. **Any suggested fixes** (optional)

### Response Timeline

| Action | Timeline |
|--------|----------|
| Initial acknowledgment | Within 24 hours |
| Initial assessment | Within 72 hours |
| Status update | Every 7 days |
| Resolution target | Within 90 days |

### Scope

The following are in scope for security reports:

- FHE-GBDT-Serving services (Gateway, Registry, Keystore, Runtime, Compiler)
- SDKs (Python, TypeScript)
- CLI tool
- Customer portal
- Official Docker images
- Cryptographic implementations

### Out of Scope

- Vulnerabilities in dependencies (report to upstream maintainers)
- Issues in third-party services we integrate with
- Social engineering attacks
- Physical security issues
- Denial of service attacks (unless caused by a specific bug)

## Security Measures

### Cryptographic Security

- **FHE Scheme**: N2HE with RLWE-based encryption
- **Security Parameter**: 128-bit post-quantum security
- **Key Management**: Envelope encryption with HashiCorp Vault
- **Transport**: mTLS between all services

### Application Security

- **Authentication**: API key + tenant binding
- **Authorization**: Role-based access control (RBAC)
- **Input Validation**: Strict validation on all inputs
- **Audit Logging**: All security-relevant events logged

### Infrastructure Security

- **Network**: Service mesh with mTLS
- **Secrets**: Managed via HashiCorp Vault
- **Containers**: Scanned for vulnerabilities
- **Dependencies**: Automated security updates

## Security Best Practices for Users

### API Key Management

```
✅ DO:
- Store API keys in environment variables or secret managers
- Rotate keys regularly
- Use different keys for development and production
- Revoke keys immediately if compromised

❌ DON'T:
- Commit API keys to version control
- Share keys between users
- Use keys in client-side code
- Log API keys
```

### Key Management

```
✅ DO:
- Store secret keys securely (encrypted at rest)
- Use hardware security modules (HSM) for production
- Implement key rotation policies
- Back up keys securely

❌ DON'T:
- Share secret keys
- Store keys in plaintext
- Transmit keys over insecure channels
- Keep keys in source code
```

### Network Security

```
✅ DO:
- Use TLS for all API connections
- Implement IP allowlisting where possible
- Use VPC peering for production deployments
- Monitor for unusual traffic patterns

❌ DON'T:
- Expose services directly to the internet
- Disable TLS certificate verification
- Use HTTP instead of HTTPS
```

## Vulnerability Disclosure Policy

### Timeline

1. **Day 0**: Vulnerability reported
2. **Day 1-3**: Initial assessment and acknowledgment
3. **Day 4-14**: Develop and test fix
4. **Day 15-30**: Prepare security advisory
5. **Day 30-90**: Coordinated disclosure

### Public Disclosure

We follow coordinated disclosure:

1. Fix is developed and tested
2. Security advisory is prepared
3. Fix is released
4. Advisory is published
5. CVE is requested (if applicable)

### Credit

We believe in giving credit to security researchers. With your permission, we will:

- Credit you in the security advisory
- Add you to our Hall of Fame
- Provide a letter of appreciation (upon request)

## Security Contacts

| Role | Contact |
|------|---------|
| Security Team | security@fhe-gbdt.dev |
| Lead Maintainer | maintainer@fhe-gbdt.dev |
| Emergency Contact | +1-XXX-XXX-XXXX (24/7) |

## Bug Bounty Program

We are planning to launch a bug bounty program. Details coming soon.

### Planned Rewards

| Severity | Reward Range |
|----------|--------------|
| Critical | $5,000 - $15,000 |
| High | $1,000 - $5,000 |
| Medium | $250 - $1,000 |
| Low | $50 - $250 |

## Security Updates

Subscribe to security announcements:

- GitHub Security Advisories (Watch this repo)
- Mailing list: security-announce@fhe-gbdt.dev
- Status page: status.fhe-gbdt.dev

## Compliance

FHE-GBDT-Serving is designed to help customers meet:

- SOC 2 Type II requirements
- HIPAA technical safeguards
- GDPR data protection requirements
- ISO 27001 controls

See our [Compliance Documentation](docs/compliance/) for details.

---

Thank you for helping keep FHE-GBDT-Serving secure!
