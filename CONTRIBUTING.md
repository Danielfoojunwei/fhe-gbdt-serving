# Contributing to FHE-GBDT-Serving

Thank you for your interest in contributing to FHE-GBDT-Serving! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@fhe-gbdt.dev](mailto:conduct@fhe-gbdt.dev).

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Go 1.21+
- Python 3.9+
- Node.js 18+
- Docker and Docker Compose
- PostgreSQL 15+
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/fhe-gbdt/fhe-gbdt-serving.git
   cd fhe-gbdt-serving
   ```

2. **Start development environment**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Install dependencies**
   ```bash
   # Go services
   go mod download

   # Python SDK and compiler
   cd sdk/python && pip install -e ".[dev]"
   cd services/compiler && pip install -e ".[dev]"

   # TypeScript SDK
   cd sdk/typescript && npm install

   # CLI
   cd cli && pip install -e ".[dev]"
   ```

4. **Run tests**
   ```bash
   make test
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, versions, etc.)
- **Logs or error messages**

### Suggesting Features

Feature requests are welcome! Please:

1. Check if the feature has already been requested
2. Clearly describe the use case
3. Explain why this would be valuable
4. Consider implementation complexity

### Your First Contribution

Look for issues labeled:
- `good-first-issue` - Good for newcomers
- `help-wanted` - Extra attention needed
- `documentation` - Documentation improvements

### Pull Requests

1. **Fork the repository**
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Write/update tests**
5. **Update documentation**
6. **Submit a pull request**

## Pull Request Process

1. **Before submitting:**
   - Run `make lint` to check code style
   - Run `make test` to ensure tests pass
   - Update the README if needed
   - Add entries to CHANGELOG.md

2. **PR Requirements:**
   - Clear description of changes
   - Reference related issues
   - Include test coverage
   - Pass CI checks

3. **Review Process:**
   - At least one maintainer approval required
   - Address all review comments
   - Keep PR scope focused

## Coding Standards

### Go Code

- Follow [Effective Go](https://golang.org/doc/effective_go.html)
- Use `gofmt` and `golint`
- Write meaningful comments
- Handle errors explicitly

```go
// Good
if err != nil {
    return fmt.Errorf("failed to process request: %w", err)
}

// Avoid
if err != nil {
    return err
}
```

### Python Code

- Follow PEP 8
- Use type hints
- Use `black` for formatting
- Use `ruff` for linting

```python
# Good
def process_model(model_path: Path, config: Config) -> Model:
    """Process a GBDT model for FHE compilation."""
    ...

# Avoid
def process_model(p, c):
    ...
```

### TypeScript Code

- Follow the project's ESLint configuration
- Use TypeScript strict mode
- Prefer explicit types over `any`

```typescript
// Good
interface PredictRequest {
  modelId: string;
  features: number[];
}

// Avoid
const request: any = { ... };
```

## Testing Guidelines

### Unit Tests

- Test individual functions/methods
- Use descriptive test names
- Cover edge cases

```go
func TestParseXGBoostModel_ValidJSON(t *testing.T) {
    // ...
}

func TestParseXGBoostModel_InvalidJSON(t *testing.T) {
    // ...
}
```

### Integration Tests

- Test component interactions
- Use test fixtures
- Clean up test data

### End-to-End Tests

- Test complete workflows
- Use realistic data
- Document test scenarios

### Test Coverage

- Aim for 80%+ coverage
- Focus on critical paths
- Don't sacrifice quality for coverage numbers

## Documentation

### Code Documentation

- Document all public APIs
- Include examples where helpful
- Keep documentation up to date

### User Documentation

- Located in `/docs`
- Use clear, simple language
- Include practical examples
- Test all code samples

## Project Structure

```
fhe-gbdt-serving/
├── services/           # Go microservices
│   ├── gateway/        # API Gateway
│   ├── registry/       # Model Registry
│   ├── keystore/       # Key Management
│   ├── compiler/       # Python Compiler
│   ├── runtime/        # C++ Runtime
│   ├── billing/        # Billing Service
│   └── metering/       # Usage Metering
├── sdk/
│   ├── python/         # Python SDK
│   └── typescript/     # TypeScript SDK
├── cli/                # CLI Tool
├── portal/             # Web Portal
├── proto/              # Protocol Buffers
├── docs/               # Documentation
├── tests/              # Test Suite
└── deploy/             # Deployment Configs
```

## Community

### Communication Channels

- **GitHub Discussions**: Questions and discussions
- **Discord**: Real-time chat (coming soon)
- **Mailing List**: Announcements

### Getting Help

- Check existing documentation
- Search GitHub issues
- Ask in GitHub Discussions
- Join our Discord

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md
- Release notes
- Project website

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to FHE-GBDT-Serving! Your efforts help advance privacy-preserving machine learning.
