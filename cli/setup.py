#!/usr/bin/env python3
"""
Setup script for fhe-gbdt-cli package.

This file is provided for backwards compatibility with older pip versions
and editable installs. The primary configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="fhe-gbdt-cli",
        version="1.0.0",
        packages=find_packages(),
        install_requires=[
            "click>=8.0.0",
            "rich>=13.0.0",
            "httpx>=0.24.0",
            "pyyaml>=6.0",
            "pydantic>=2.0.0",
            "python-dotenv>=1.0.0",
        ],
        entry_points={
            "console_scripts": [
                "fhe-gbdt=fhe_gbdt_cli.main:cli",
            ],
        },
        python_requires=">=3.8",
    )
