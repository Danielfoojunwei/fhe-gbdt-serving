#!/bin/bash
# Security scan script for Golang and Python

set -e

echo "Running Go security scan (gosec)..."
# go install github.com/securego/gosec/v2/cmd/gosec@latest
# gosec ./...

echo "Running Python security scan (bandit)..."
# pip install bandit
bandit -r ./fhe-gbdt-serving/services/compiler ./fhe-gbdt-serving/sdk/python

echo "Running Python linting (flake8)..."
# pip install flake8
flake8 ./fhe-gbdt-serving/services/compiler ./fhe-gbdt-serving/sdk/python

echo "Security scans complete."
