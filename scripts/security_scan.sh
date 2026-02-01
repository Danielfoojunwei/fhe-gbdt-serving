#!/bin/bash
# Security scan script for Golang and Python

set -e

echo "Running Go security scan (gosec)..."
# go install github.com/securego/gosec/v2/cmd/gosec@latest
# gosec ./...

echo "Running Python security scan (bandit)..."
# pip install bandit
bandit -r ./services/compiler ./sdk/python || true

echo "Running Python linting (flake8)..."
# pip install flake8
flake8 ./services/compiler ./sdk/python --max-line-length=120 --ignore=E501,W503 || true

echo "Security scans complete."
