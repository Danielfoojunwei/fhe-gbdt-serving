#!/usr/bin/env python3
"""
FHE-GBDT CLI Tool

Command-line interface for interacting with the FHE-GBDT platform.
Aligned with TenSafe CLI design.

Usage:
    fhe-gbdt [OPTIONS] COMMAND [ARGS]...

Commands:
    train       Train GBDT models with differential privacy
    predict     Run encrypted predictions
    model       Model management (register, compile, list, delete)
    keys        Key management (upload, rotate, revoke)
    package     GBSP package management
    config      Configuration management
    verify      Verify backend availability
    version     Show version information
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Version info
__version__ = "1.0.0"
__api_version__ = "v1"


class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    TABLE = "table"


@dataclass
class CLIConfig:
    """CLI configuration."""
    api_url: str = "https://api.fhe-gbdt.example.com"
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    output_format: OutputFormat = OutputFormat.TEXT
    verbose: bool = False
    timeout: int = 120


def load_config() -> CLIConfig:
    """Load configuration from file or environment."""
    config = CLIConfig()

    # Load from config file
    config_file = Path.home() / ".fhe-gbdt" / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            data = json.load(f)
            config.api_url = data.get("api_url", config.api_url)
            config.api_key = data.get("api_key")
            config.tenant_id = data.get("tenant_id")

    # Override with environment variables
    config.api_url = os.environ.get("FHE_GBDT_API_URL", config.api_url)
    config.api_key = os.environ.get("FHE_GBDT_API_KEY", config.api_key)
    config.tenant_id = os.environ.get("FHE_GBDT_TENANT_ID", config.tenant_id)

    return config


def save_config(config: CLIConfig):
    """Save configuration to file."""
    config_dir = Path.home() / ".fhe-gbdt"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump({
            "api_url": config.api_url,
            "api_key": config.api_key,
            "tenant_id": config.tenant_id,
        }, f, indent=2)

    print(f"Configuration saved to {config_file}")


class CLIClient:
    """HTTP client for CLI operations."""

    def __init__(self, config: CLIConfig):
        self.config = config
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "X-API-Key": self.config.api_key or "",
                "X-Tenant-ID": self.config.tenant_id or "",
                "Content-Type": "application/json",
            })
        return self._session

    def request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.config.api_url}/{__api_version__}{path}"

        if self.config.verbose:
            print(f"[DEBUG] {method} {url}")

        response = self.session.request(
            method, url,
            timeout=self.config.timeout,
            **kwargs
        )

        if self.config.verbose:
            print(f"[DEBUG] Status: {response.status_code}")

        if response.status_code >= 400:
            try:
                error = response.json().get("error", {})
                raise CLIError(
                    error.get("message", "Unknown error"),
                    code=error.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise CLIError(f"HTTP {response.status_code}: {response.text}")

        if response.status_code == 204:
            return {}

        return response.json()


class CLIError(Exception):
    """CLI error with code."""

    def __init__(self, message: str, code: str = "ERROR"):
        super().__init__(message)
        self.code = code


def output(data: Any, format: OutputFormat = OutputFormat.TEXT):
    """Output data in specified format."""
    if format == OutputFormat.JSON:
        print(json.dumps(data, indent=2, default=str))
    elif format == OutputFormat.TABLE:
        if isinstance(data, list) and data:
            # Print as table
            headers = data[0].keys()
            print("\t".join(headers))
            print("-" * 80)
            for row in data:
                print("\t".join(str(row.get(h, "")) for h in headers))
        else:
            print(data)
    else:
        # Text format
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"{k}: {v}")
        elif isinstance(data, list):
            for item in data:
                print(item)
        else:
            print(data)


# ============== Command Handlers ==============

def handle_version(args, config: CLIConfig):
    """Show version information."""
    output({
        "cli_version": __version__,
        "api_version": __api_version__,
        "api_url": config.api_url,
    }, config.output_format)


def handle_config(args, config: CLIConfig):
    """Handle config commands."""
    if args.config_cmd == "show":
        output({
            "api_url": config.api_url,
            "api_key": "***" if config.api_key else None,
            "tenant_id": config.tenant_id,
        }, config.output_format)

    elif args.config_cmd == "set":
        if args.key == "api_url":
            config.api_url = args.value
        elif args.key == "api_key":
            config.api_key = args.value
        elif args.key == "tenant_id":
            config.tenant_id = args.value
        else:
            raise CLIError(f"Unknown config key: {args.key}")
        save_config(config)

    elif args.config_cmd == "init":
        config.api_url = args.api_url or input("API URL [https://api.fhe-gbdt.example.com]: ") or config.api_url
        config.api_key = args.api_key or input("API Key: ")
        config.tenant_id = args.tenant_id or input("Tenant ID: ")
        save_config(config)


def handle_verify(args, config: CLIConfig):
    """Verify backend availability."""
    print("Checking backend availability...")

    checks = {
        "api": False,
        "n2he": False,
        "training": False,
    }

    client = CLIClient(config)

    # Check API
    try:
        result = client.request("GET", "/health")
        checks["api"] = result.get("status") == "healthy"
    except Exception as e:
        if config.verbose:
            print(f"[DEBUG] API check failed: {e}")

    # TODO: Check N2HE and training backends

    for name, status in checks.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}: {'available' if status else 'unavailable'}")

    if all(checks.values()):
        print("\nAll backends available!")
        return 0
    else:
        print("\nSome backends unavailable.")
        return 1


def handle_train(args, config: CLIConfig):
    """Handle training commands."""
    client = CLIClient(config)

    if args.train_cmd == "start":
        # Build training config
        training_config = {
            "name": args.name,
            "dataset_path": args.dataset,
            "library": args.library,
            "hyperparameters": {},
        }

        # Parse hyperparameters
        if args.n_estimators:
            training_config["hyperparameters"]["n_estimators"] = args.n_estimators
        if args.max_depth:
            training_config["hyperparameters"]["max_depth"] = args.max_depth
        if args.learning_rate:
            training_config["hyperparameters"]["learning_rate"] = args.learning_rate

        # DP config
        if args.dp:
            training_config["dp_config"] = {
                "enabled": True,
                "epsilon": args.epsilon or 1.0,
                "delta": args.delta or 1e-5,
            }

        # Load from config file if provided
        if args.config_file:
            with open(args.config_file) as f:
                file_config = json.load(f)
                training_config.update(file_config)

        result = client.request("POST", "/training/jobs", json=training_config)

        print(f"Training job started: {result['data']['id']}")

        if args.wait:
            # Wait for completion
            job_id = result['data']['id']
            while True:
                status = client.request("GET", f"/training/jobs/{job_id}")
                job = status.get("data", status)

                print(f"\rProgress: {job.get('progress', 0):.1f}%", end="")

                if job.get("status") in ("completed", "failed"):
                    print()
                    output(job, config.output_format)
                    break

                time.sleep(5)
        else:
            output(result.get("data", result), config.output_format)

    elif args.train_cmd == "status":
        result = client.request("GET", f"/training/jobs/{args.job_id}")
        output(result.get("data", result), config.output_format)

    elif args.train_cmd == "list":
        result = client.request("GET", "/training/jobs")
        output(result.get("data", []), config.output_format)

    elif args.train_cmd == "stop":
        client.request("DELETE", f"/training/jobs/{args.job_id}")
        print(f"Training job {args.job_id} stopped.")


def handle_predict(args, config: CLIConfig):
    """Handle prediction commands."""
    client = CLIClient(config)

    # Read ciphertext
    if args.ciphertext_file:
        with open(args.ciphertext_file, "rb") as f:
            import base64
            ciphertext = base64.b64encode(f.read()).decode()
    else:
        ciphertext = args.ciphertext

    request = {
        "model_id": args.model_id,
        "ciphertext": ciphertext,
    }

    if args.profile:
        request["profile"] = args.profile

    result = client.request("POST", "/predict", json=request)

    if args.output_file:
        import base64
        with open(args.output_file, "wb") as f:
            f.write(base64.b64decode(result["data"]["ciphertext"]))
        print(f"Result written to {args.output_file}")
    else:
        output(result.get("data", result), config.output_format)


def handle_model(args, config: CLIConfig):
    """Handle model commands."""
    client = CLIClient(config)

    if args.model_cmd == "list":
        result = client.request("GET", "/models")
        output(result.get("data", []), config.output_format)

    elif args.model_cmd == "register":
        # Read model file
        with open(args.model_file, "rb") as f:
            import base64
            model_data = base64.b64encode(f.read()).decode()

        request = {
            "name": args.name,
            "library": args.library,
            "model": model_data,
        }

        result = client.request("POST", "/models", json=request)
        output(result.get("data", result), config.output_format)

    elif args.model_cmd == "get":
        result = client.request("GET", f"/models/{args.model_id}")
        output(result.get("data", result), config.output_format)

    elif args.model_cmd == "delete":
        client.request("DELETE", f"/models/{args.model_id}")
        print(f"Model {args.model_id} deleted.")

    elif args.model_cmd == "compile":
        request = {}
        if args.profile:
            request["profile"] = args.profile

        result = client.request("POST", f"/models/{args.model_id}/compile", json=request)

        print(f"Compilation started: {result.get('job_id')}")

        if args.wait:
            # Wait for completion
            while True:
                status = client.request("GET", f"/models/{args.model_id}/compile/status")
                print(f"\rProgress: {status.get('progress', 0)}%", end="")

                if status.get("status") in ("completed", "failed"):
                    print()
                    output(status, config.output_format)
                    break

                time.sleep(2)
        else:
            output(result, config.output_format)


def handle_keys(args, config: CLIConfig):
    """Handle key commands."""
    client = CLIClient(config)

    if args.keys_cmd == "upload":
        with open(args.key_file, "rb") as f:
            import base64
            key_data = base64.b64encode(f.read()).decode()

        result = client.request("POST", "/keys", json={"key": key_data})
        output(result, config.output_format)

    elif args.keys_cmd == "status":
        result = client.request("GET", f"/keys/{args.key_id}")
        output(result, config.output_format)

    elif args.keys_cmd == "rotate":
        result = client.request("POST", f"/keys/{args.key_id}/rotate")
        output(result, config.output_format)

    elif args.keys_cmd == "revoke":
        client.request("DELETE", f"/keys/{args.key_id}")
        print(f"Key {args.key_id} revoked.")


def handle_package(args, config: CLIConfig):
    """Handle package commands."""
    client = CLIClient(config)

    if args.package_cmd == "create":
        request = {
            "model_id": args.model_id,
        }
        if args.recipients:
            request["recipients"] = args.recipients.split(",")

        result = client.request("POST", "/packages", json=request)
        output(result.get("data", result), config.output_format)

    elif args.package_cmd == "verify":
        result = client.request("POST", f"/packages/{args.package_id}/verify")
        output(result, config.output_format)

    elif args.package_cmd == "get":
        result = client.request("GET", f"/packages/{args.package_id}")
        output(result.get("data", result), config.output_format)


# ============== Main Entry Point ==============

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="fhe-gbdt",
        description="FHE-GBDT Command Line Interface",
    )

    # Global options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["text", "json", "table"], default="text", help="Output format")
    parser.add_argument("--api-url", help="API URL override")
    parser.add_argument("--api-key", help="API key override")
    parser.add_argument("--tenant-id", help="Tenant ID override")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_sub = config_parser.add_subparsers(dest="config_cmd")

    config_sub.add_parser("show", help="Show current configuration")

    config_set = config_sub.add_parser("set", help="Set configuration value")
    config_set.add_argument("key", help="Configuration key")
    config_set.add_argument("value", help="Configuration value")

    config_init = config_sub.add_parser("init", help="Initialize configuration")
    config_init.add_argument("--api-url", help="API URL")
    config_init.add_argument("--api-key", help="API key")
    config_init.add_argument("--tenant-id", help="Tenant ID")

    # Verify command
    subparsers.add_parser("verify", help="Verify backend availability")

    # Train command
    train_parser = subparsers.add_parser("train", help="Training operations")
    train_sub = train_parser.add_subparsers(dest="train_cmd")

    train_start = train_sub.add_parser("start", help="Start training job")
    train_start.add_argument("--name", required=True, help="Job name")
    train_start.add_argument("--dataset", required=True, help="Dataset path")
    train_start.add_argument("--library", required=True, choices=["xgboost", "lightgbm", "catboost"])
    train_start.add_argument("--config-file", help="Training config file (JSON)")
    train_start.add_argument("--n-estimators", type=int, help="Number of trees")
    train_start.add_argument("--max-depth", type=int, help="Max tree depth")
    train_start.add_argument("--learning-rate", type=float, help="Learning rate")
    train_start.add_argument("--dp", action="store_true", help="Enable differential privacy")
    train_start.add_argument("--epsilon", type=float, help="DP epsilon budget")
    train_start.add_argument("--delta", type=float, help="DP delta budget")
    train_start.add_argument("--wait", action="store_true", help="Wait for completion")

    train_status = train_sub.add_parser("status", help="Get training job status")
    train_status.add_argument("job_id", help="Job ID")

    train_sub.add_parser("list", help="List training jobs")

    train_stop = train_sub.add_parser("stop", help="Stop training job")
    train_stop.add_argument("job_id", help="Job ID")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run encrypted prediction")
    predict_parser.add_argument("--model-id", required=True, help="Model ID")
    predict_parser.add_argument("--ciphertext", help="Base64 encoded ciphertext")
    predict_parser.add_argument("--ciphertext-file", help="Ciphertext file path")
    predict_parser.add_argument("--output-file", help="Output file for result")
    predict_parser.add_argument("--profile", choices=["latency", "throughput"], help="Execution profile")

    # Model command
    model_parser = subparsers.add_parser("model", help="Model management")
    model_sub = model_parser.add_subparsers(dest="model_cmd")

    model_sub.add_parser("list", help="List models")

    model_register = model_sub.add_parser("register", help="Register model")
    model_register.add_argument("--name", required=True, help="Model name")
    model_register.add_argument("--library", required=True, choices=["xgboost", "lightgbm", "catboost"])
    model_register.add_argument("--model-file", required=True, help="Model file path")

    model_get = model_sub.add_parser("get", help="Get model details")
    model_get.add_argument("model_id", help="Model ID")

    model_delete = model_sub.add_parser("delete", help="Delete model")
    model_delete.add_argument("model_id", help="Model ID")

    model_compile = model_sub.add_parser("compile", help="Compile model for FHE")
    model_compile.add_argument("model_id", help="Model ID")
    model_compile.add_argument("--profile", choices=["latency", "throughput"], help="Compilation profile")
    model_compile.add_argument("--wait", action="store_true", help="Wait for completion")

    # Keys command
    keys_parser = subparsers.add_parser("keys", help="Key management")
    keys_sub = keys_parser.add_subparsers(dest="keys_cmd")

    keys_upload = keys_sub.add_parser("upload", help="Upload evaluation keys")
    keys_upload.add_argument("--key-file", required=True, help="Key file path")

    keys_status = keys_sub.add_parser("status", help="Get key status")
    keys_status.add_argument("key_id", help="Key ID")

    keys_rotate = keys_sub.add_parser("rotate", help="Rotate keys")
    keys_rotate.add_argument("key_id", help="Key ID")

    keys_revoke = keys_sub.add_parser("revoke", help="Revoke keys")
    keys_revoke.add_argument("key_id", help="Key ID")

    # Package command
    package_parser = subparsers.add_parser("package", help="GBSP package management")
    package_sub = package_parser.add_subparsers(dest="package_cmd")

    package_create = package_sub.add_parser("create", help="Create GBSP package")
    package_create.add_argument("--model-id", required=True, help="Model ID")
    package_create.add_argument("--recipients", help="Comma-separated recipient public keys")

    package_verify = package_sub.add_parser("verify", help="Verify GBSP package")
    package_verify.add_argument("package_id", help="Package ID")

    package_get = package_sub.add_parser("get", help="Get package details")
    package_get.add_argument("package_id", help="Package ID")

    # Parse arguments
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Apply overrides
    if args.verbose:
        config.verbose = True
    if args.format:
        config.output_format = OutputFormat(args.format)
    if args.api_url:
        config.api_url = args.api_url
    if args.api_key:
        config.api_key = args.api_key
    if args.tenant_id:
        config.tenant_id = args.tenant_id

    # Route to handler
    try:
        if args.command == "version":
            handle_version(args, config)
        elif args.command == "config":
            handle_config(args, config)
        elif args.command == "verify":
            sys.exit(handle_verify(args, config))
        elif args.command == "train":
            handle_train(args, config)
        elif args.command == "predict":
            handle_predict(args, config)
        elif args.command == "model":
            handle_model(args, config)
        elif args.command == "keys":
            handle_keys(args, config)
        elif args.command == "package":
            handle_package(args, config)
        else:
            parser.print_help()
            sys.exit(1)

    except CLIError as e:
        print(f"Error [{e.code}]: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(130)
    except Exception as e:
        if config.verbose:
            import traceback
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
