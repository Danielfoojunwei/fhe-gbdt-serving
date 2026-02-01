# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in FHE-GBDT Serving, please report it responsibly:

1. **Do NOT** create a public GitHub issue.
2. Email security@example.com with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested mitigations

We aim to respond within 48 hours and provide a fix within 14 days for critical issues.

## Scope

The following are in scope:
- Gateway, Compiler, Runtime, Registry, Keystore services
- Python SDK
- C++ client library

Out of scope:
- Third-party dependencies (report upstream)
- Social engineering attacks

## Our Commitments

- We will not take legal action against good-faith security researchers.
- We will acknowledge your contribution (with permission) in release notes.
- We will provide a timeline for fixes.

## Security Best Practices

### For Operators
- Enable mTLS for all service-to-service communication.
- Rotate API keys regularly.
- Monitor audit logs for anomalies.
- Keep dependencies updated.

### For SDK Users
- Never share secret keys.
- Store eval keys securely (they are sensitive).
- Validate server certificates.
