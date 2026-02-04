#!/usr/bin/env python3
"""
QA Test Suite: Audit Chain Functionality

Tests the hash-chained audit log system for:
1. Entry creation and hashing
2. Chain integrity verification
3. Sensitive data filtering
4. Export functionality
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.audit.audit_chain import (
    AuditChain,
    AuditEntry,
    AuditAction,
    AuditSeverity,
    audit_log,
    get_audit_chain,
)


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
        result = f"{status} {self.name} ({self.duration:.2f}s)"
        if self.error:
            result += f"\n    Error: {self.error}"
        if self.details:
            result += f"\n    Details: {self.details}"
        return result


def test_entry_creation():
    """Test audit entry creation and hashing."""
    result = TestResult("Audit Entry Creation")
    start = time.time()

    try:
        entry = AuditEntry(
            entry_id="test-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=AuditAction.PREDICT_REQUEST.value,
            tenant_id="tenant-123",
            previous_hash="sha256:genesis",
            request_id="req-abc",
            resource_id="model-xyz",
        )

        # Verify hash is computed
        assert entry.current_hash.startswith("sha256:"), "Hash should start with sha256:"
        assert len(entry.current_hash) > 10, "Hash should be non-trivial"

        # Verify to_dict works
        entry_dict = entry.to_dict()
        assert "entry_id" in entry_dict
        assert "current_hash" in entry_dict
        assert "_sensitive_fields" not in entry_dict

        # Verify from_dict works
        restored = AuditEntry.from_dict(entry_dict)
        assert restored.entry_id == entry.entry_id
        assert restored.current_hash == entry.current_hash

        result.passed = True
        result.details = {"hash": entry.current_hash[:30] + "..."}

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_chain_integrity():
    """Test hash chain integrity verification."""
    result = TestResult("Chain Integrity Verification")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = AuditChain(
                storage_path=Path(tmpdir),
                auto_flush=False,
            )

            # Add entries
            entries = []
            for i in range(5):
                entry = chain.append(
                    action=AuditAction.PREDICT_REQUEST,
                    tenant_id="tenant-123",
                    request_id=f"req-{i}",
                    metadata={"iteration": i},
                )
                entries.append(entry)

            # Verify chain
            assert chain.verify_chain(entries), "Chain should be valid"

            # Verify each entry links correctly
            for i in range(1, len(entries)):
                assert entries[i].previous_hash == entries[i - 1].current_hash, \
                    f"Entry {i} should link to entry {i-1}"

            result.passed = True
            result.details = {"entries": len(entries)}

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_sensitive_data_filtering():
    """Test that sensitive data is filtered from audit logs."""
    result = TestResult("Sensitive Data Filtering")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = AuditChain(
                storage_path=Path(tmpdir),
                auto_flush=False,
            )

            # Add entry with sensitive data
            entry = chain.append(
                action=AuditAction.PREDICT_SUCCESS,
                tenant_id="tenant-123",
                metadata={
                    "model_id": "model-123",  # Safe
                    "api_key": "secret-key-12345",  # Should be redacted
                    "ciphertext": "encrypted-data",  # Should be redacted
                    "plaintext": "sensitive-value",  # Should be redacted
                    "password": "my-password",  # Should be redacted
                    "nested": {
                        "secret": "hidden-value",  # Should be redacted
                        "normal": "visible",  # Safe
                    }
                },
            )

            # Check filtering
            metadata = entry.metadata
            assert metadata.get("model_id") == "model-123", "Safe field should be preserved"
            assert metadata.get("api_key") == "[REDACTED]", "API key should be redacted"
            assert metadata.get("ciphertext") == "[REDACTED]", "Ciphertext should be redacted"
            assert metadata.get("plaintext") == "[REDACTED]", "Plaintext should be redacted"
            assert metadata.get("password") == "[REDACTED]", "Password should be redacted"
            assert metadata.get("nested", {}).get("secret") == "[REDACTED]", "Nested secret should be redacted"
            assert metadata.get("nested", {}).get("normal") == "visible", "Nested normal should be preserved"

            result.passed = True
            result.details = {"redacted_fields": 5}

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_flush_and_persistence():
    """Test that entries are persisted to storage."""
    result = TestResult("Flush and Persistence")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create chain and add entries
            chain = AuditChain(
                storage_path=storage_path,
                auto_flush=False,
            )

            for i in range(10):
                chain.append(
                    action=AuditAction.PREDICT_REQUEST,
                    tenant_id="tenant-123",
                    request_id=f"req-{i}",
                )

            # Flush to disk
            chain.flush()

            # Check files created
            log_files = list(storage_path.glob("audit-*.jsonl"))
            assert len(log_files) >= 1, "Log file should be created"

            # Verify content
            with open(log_files[0]) as f:
                lines = f.readlines()
                assert len(lines) == 10, f"Expected 10 entries, got {len(lines)}"

                # Parse first entry
                first_entry = json.loads(lines[0])
                assert "entry_id" in first_entry
                assert "current_hash" in first_entry
                assert "previous_hash" in first_entry

            # Check chain state saved
            state_file = storage_path / "chain_state.json"
            assert state_file.exists(), "Chain state should be saved"

            result.passed = True
            result.details = {
                "log_files": len(log_files),
                "entries_persisted": 10,
            }

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_export_for_compliance():
    """Test audit log export for compliance."""
    result = TestResult("Compliance Export")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = AuditChain(
                storage_path=Path(tmpdir),
                auto_flush=False,
            )

            # Add entries
            for i in range(20):
                chain.append(
                    action=AuditAction.PREDICT_REQUEST if i % 2 == 0 else AuditAction.PREDICT_SUCCESS,
                    tenant_id="tenant-123" if i % 3 == 0 else "tenant-456",
                    request_id=f"req-{i}",
                )

            chain.flush()

            # Export all
            export = chain.export_for_compliance(
                start_date=datetime.now(timezone.utc) - timedelta(hours=1),
                end_date=datetime.now(timezone.utc) + timedelta(hours=1),
            )

            assert export["entry_count"] == 20
            assert export["chain_valid"] is True
            assert "export_hash" in export
            assert len(export["entries"]) == 20

            # Export filtered by tenant
            export_filtered = chain.export_for_compliance(
                start_date=datetime.now(timezone.utc) - timedelta(hours=1),
                end_date=datetime.now(timezone.utc) + timedelta(hours=1),
                tenant_id="tenant-123",
            )

            assert export_filtered["entry_count"] < 20
            assert all(e["tenant_id"] == "tenant-123" for e in export_filtered["entries"])

            result.passed = True
            result.details = {
                "total_exported": export["entry_count"],
                "filtered_exported": export_filtered["entry_count"],
            }

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_action_types():
    """Test all audit action types."""
    result = TestResult("All Action Types")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = AuditChain(
                storage_path=Path(tmpdir),
                auto_flush=False,
            )

            actions_tested = []

            # Test each action type
            for action in AuditAction:
                entry = chain.append(
                    action=action,
                    tenant_id="tenant-123",
                )
                actions_tested.append(action.value)
                assert entry.action == action.value

            result.passed = True
            result.details = {
                "actions_tested": len(actions_tested),
                "sample_actions": actions_tested[:5],
            }

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_callback_registration():
    """Test audit entry callbacks."""
    result = TestResult("Callback Registration")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = AuditChain(
                storage_path=Path(tmpdir),
                auto_flush=False,
            )

            callback_entries = []

            def my_callback(entry):
                callback_entries.append(entry.entry_id)

            chain.register_callback(my_callback)

            # Add entries
            for i in range(5):
                chain.append(
                    action=AuditAction.PREDICT_REQUEST,
                    tenant_id="tenant-123",
                )

            assert len(callback_entries) == 5, f"Expected 5 callbacks, got {len(callback_entries)}"

            result.passed = True
            result.details = {"callbacks_received": len(callback_entries)}

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def test_get_recent_entries():
    """Test getting recent entries from memory."""
    result = TestResult("Get Recent Entries")
    start = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = AuditChain(
                storage_path=Path(tmpdir),
                auto_flush=False,
            )

            # Add entries
            for i in range(50):
                chain.append(
                    action=AuditAction.PREDICT_REQUEST,
                    tenant_id="tenant-123",
                    request_id=f"req-{i}",
                )

            # Get recent
            recent = chain.get_recent_entries(count=10)
            assert len(recent) == 10
            assert recent[-1].metadata.get("request_id") is None  # We didn't set request_id in metadata

            # Get all
            all_recent = chain.get_recent_entries(count=100)
            assert len(all_recent) == 50

            result.passed = True
            result.details = {
                "recent_10": len(recent),
                "all_entries": len(all_recent),
            }

    except Exception as e:
        result.error = str(e)

    result.duration = time.time() - start
    return result


def run_all_tests():
    """Run all audit chain tests."""
    print("=" * 70)
    print("FHE-GBDT Audit Chain QA Tests")
    print("=" * 70)
    print()

    tests = [
        test_entry_creation,
        test_chain_integrity,
        test_sensitive_data_filtering,
        test_flush_and_persistence,
        test_export_for_compliance,
        test_action_types,
        test_callback_registration,
        test_get_recent_entries,
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
