#!/usr/bin/env python3
"""
QA Test Suite: CLI Tool

Tests the fhe-gbdt CLI tool for:
1. Version command
2. Config command
3. Help text
4. Argument parsing
"""

import subprocess
import sys
import tempfile
import json
from pathlib import Path


class TestResult:
    """Test result tracker."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.duration = 0

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        result = f"{status} {self.name}"
        if self.error:
            result += f"\n    Error: {self.error}"
        if self.details:
            result += f"\n    Details: {self.details}"
        return result


def run_cli(*args, env=None):
    """Run CLI command and return output."""
    cli_path = Path(__file__).parent.parent.parent / "cli" / "main.py"
    cmd = [sys.executable, str(cli_path)] + list(args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    return result.returncode, result.stdout, result.stderr


def test_version_command():
    """Test version command."""
    result = TestResult("Version Command")

    try:
        returncode, stdout, stderr = run_cli("version")

        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "cli_version" in stdout or "1.0.0" in stdout, "Version not found in output"

        result.passed = True
        result.details = {"output": stdout.strip()[:100]}

    except Exception as e:
        result.error = str(e)

    return result


def test_version_json_format():
    """Test version command with JSON output."""
    result = TestResult("Version JSON Format")

    try:
        returncode, stdout, stderr = run_cli("--format", "json", "version")

        assert returncode == 0, f"Expected return code 0, got {returncode}"

        # Should be valid JSON
        data = json.loads(stdout)
        assert "cli_version" in data or "cliVersion" in data, "Version key not found"

        result.passed = True
        result.details = {"parsed_json": True}

    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
    except Exception as e:
        result.error = str(e)

    return result


def test_help_command():
    """Test help output."""
    result = TestResult("Help Command")

    try:
        returncode, stdout, stderr = run_cli("--help")

        # Help might go to stdout or stderr depending on argparse version
        output = stdout + stderr

        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "train" in output.lower(), "train command not in help"
        assert "predict" in output.lower(), "predict command not in help"
        assert "model" in output.lower(), "model command not in help"

        result.passed = True
        result.details = {"commands_found": ["train", "predict", "model"]}

    except Exception as e:
        result.error = str(e)

    return result


def test_config_show():
    """Test config show command."""
    result = TestResult("Config Show Command")

    try:
        returncode, stdout, stderr = run_cli("config", "show")

        # Should show config even if empty
        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "api_url" in stdout.lower() or "api-url" in stdout.lower(), "API URL not shown"

        result.passed = True
        result.details = {"output": stdout.strip()[:100]}

    except Exception as e:
        result.error = str(e)

    return result


def test_train_help():
    """Test train subcommand help."""
    result = TestResult("Train Help")

    try:
        returncode, stdout, stderr = run_cli("train", "--help")

        output = stdout + stderr

        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "start" in output or "status" in output, "Train subcommands not in help"

        result.passed = True
        result.details = {"has_subcommands": True}

    except Exception as e:
        result.error = str(e)

    return result


def test_model_help():
    """Test model subcommand help."""
    result = TestResult("Model Help")

    try:
        returncode, stdout, stderr = run_cli("model", "--help")

        output = stdout + stderr

        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "list" in output or "register" in output, "Model subcommands not in help"

        result.passed = True
        result.details = {"has_subcommands": True}

    except Exception as e:
        result.error = str(e)

    return result


def test_keys_help():
    """Test keys subcommand help."""
    result = TestResult("Keys Help")

    try:
        returncode, stdout, stderr = run_cli("keys", "--help")

        output = stdout + stderr

        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "upload" in output or "rotate" in output, "Keys subcommands not in help"

        result.passed = True
        result.details = {"has_subcommands": True}

    except Exception as e:
        result.error = str(e)

    return result


def test_package_help():
    """Test package subcommand help."""
    result = TestResult("Package Help")

    try:
        returncode, stdout, stderr = run_cli("package", "--help")

        output = stdout + stderr

        assert returncode == 0, f"Expected return code 0, got {returncode}"
        assert "create" in output or "verify" in output, "Package subcommands not in help"

        result.passed = True
        result.details = {"has_subcommands": True}

    except Exception as e:
        result.error = str(e)

    return result


def test_verify_command():
    """Test verify command (backend availability check)."""
    result = TestResult("Verify Command")

    try:
        # This will fail to connect but should still run
        returncode, stdout, stderr = run_cli("verify")

        output = stdout + stderr

        # Check that it attempted verification
        assert "api" in output.lower() or "backend" in output.lower() or "checking" in output.lower(), \
            "Verify output doesn't show backend checks"

        result.passed = True
        result.details = {"ran_checks": True}

    except Exception as e:
        result.error = str(e)

    return result


def test_invalid_command():
    """Test invalid command error handling."""
    result = TestResult("Invalid Command Handling")

    try:
        returncode, stdout, stderr = run_cli("nonexistent_command")

        # Should return non-zero or show help
        output = stdout + stderr

        # Either shows help or error
        assert returncode != 0 or "usage" in output.lower() or "error" in output.lower(), \
            "Should show error or help for invalid command"

        result.passed = True
        result.details = {"handled_gracefully": True}

    except Exception as e:
        result.error = str(e)

    return result


def test_verbose_flag():
    """Test verbose flag."""
    result = TestResult("Verbose Flag")

    try:
        returncode, stdout, stderr = run_cli("-v", "version")

        # Should still work with verbose
        assert returncode == 0, f"Expected return code 0, got {returncode}"

        result.passed = True
        result.details = {"verbose_works": True}

    except Exception as e:
        result.error = str(e)

    return result


def run_all_tests():
    """Run all CLI tests."""
    print("=" * 70)
    print("FHE-GBDT CLI Tool QA Tests")
    print("=" * 70)
    print()

    tests = [
        test_version_command,
        test_version_json_format,
        test_help_command,
        test_config_show,
        test_train_help,
        test_model_help,
        test_keys_help,
        test_package_help,
        test_verify_command,
        test_invalid_command,
        test_verbose_flag,
    ]

    results = []

    for test_func in tests:
        print(f"Running: {test_func.__name__}...", end=" ", flush=True)
        result = test_func()
        results.append(result)

        if result.passed:
            print("PASS")
        else:
            print("FAIL")

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for result in results:
        print(result)
        print()

    print("-" * 70)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("-" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
