"""
FHE-GBDT Hash-Chained Audit Log System

Provides immutable, cryptographically-linked audit trail for all platform operations.
Aligned with TenSafe's audit trail implementation.
"""

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import os


class AuditAction(Enum):
    """Audit action types."""
    # Authentication
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_REVOKE = "auth.revoke"

    # Model operations
    MODEL_REGISTER = "model.register"
    MODEL_COMPILE = "model.compile"
    MODEL_DELETE = "model.delete"
    MODEL_EXPORT = "model.export"

    # Training operations
    TRAINING_START = "training.start"
    TRAINING_COMPLETE = "training.complete"
    TRAINING_FAIL = "training.fail"
    TRAINING_CHECKPOINT = "training.checkpoint"

    # Inference operations
    PREDICT_REQUEST = "predict.request"
    PREDICT_SUCCESS = "predict.success"
    PREDICT_FAILURE = "predict.failure"

    # Key management
    KEY_UPLOAD = "key.upload"
    KEY_ROTATE = "key.rotate"
    KEY_REVOKE = "key.revoke"
    KEY_DELETE = "key.delete"

    # Package operations
    PACKAGE_CREATE = "package.create"
    PACKAGE_VERIFY = "package.verify"
    PACKAGE_EXTRACT = "package.extract"

    # Administrative
    TENANT_CREATE = "tenant.create"
    TENANT_DELETE = "tenant.delete"
    CONFIG_CHANGE = "config.change"

    # Compliance
    EVIDENCE_COLLECT = "compliance.evidence_collect"
    AUDIT_EXPORT = "compliance.audit_export"


class AuditSeverity(Enum):
    """Audit entry severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """Single audit log entry with hash chain linkage."""

    # Required fields
    entry_id: str
    timestamp: str
    action: str
    tenant_id: str

    # Chain integrity
    previous_hash: str
    current_hash: str = ""

    # Context
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None

    # Details (never include sensitive data)
    severity: str = "info"
    status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Metrics (safe to log)
    latency_ms: Optional[float] = None
    payload_size_bytes: Optional[int] = None

    # DO NOT log these - they are stripped before storage
    _sensitive_fields: List[str] = field(default_factory=lambda: [
        "ciphertext", "plaintext", "secret_key", "api_key",
        "password", "token", "features", "predictions"
    ])

    def __post_init__(self):
        """Compute hash after initialization."""
        if not self.current_hash:
            self.current_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of entry content."""
        # Create deterministic JSON representation
        content = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "tenant_id": self.tenant_id,
            "previous_hash": self.previous_hash,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "severity": self.severity,
            "status": self.status,
            "metadata": self.metadata,
            "latency_ms": self.latency_ms,
            "payload_size_bytes": self.payload_size_bytes,
        }

        # Sort keys for deterministic hashing
        json_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"

    def verify_chain(self, previous_entry: Optional['AuditEntry']) -> bool:
        """Verify this entry's hash chain integrity."""
        # Verify previous hash matches
        if previous_entry:
            if self.previous_hash != previous_entry.current_hash:
                return False
        else:
            # Genesis entry should have empty previous hash
            if self.previous_hash != "sha256:genesis":
                return False

        # Verify current hash is correct
        expected_hash = self._compute_hash()
        return self.current_hash == expected_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        del d['_sensitive_fields']
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create entry from dictionary."""
        data.pop('_sensitive_fields', None)
        return cls(**data)


class AuditChain:
    """
    Hash-chained audit log with integrity verification.

    Features:
    - Immutable append-only log
    - SHA-256 hash chain linking entries
    - Automatic sensitive data filtering
    - Batch verification
    - Export for compliance audits
    """

    GENESIS_HASH = "sha256:genesis"

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_entries_in_memory: int = 10000,
        auto_flush: bool = True,
        flush_interval_seconds: int = 60,
    ):
        self.storage_path = storage_path or Path("/var/log/fhe-gbdt/audit")
        self.max_entries_in_memory = max_entries_in_memory
        self.auto_flush = auto_flush
        self.flush_interval_seconds = flush_interval_seconds

        self._entries: List[AuditEntry] = []
        self._last_hash: str = self.GENESIS_HASH
        self._lock = threading.Lock()
        self._entry_counter = 0
        self._callbacks: List[Callable[[AuditEntry], None]] = []

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load last hash from storage
        self._load_last_hash()

        # Start auto-flush thread if enabled
        if self.auto_flush:
            self._start_auto_flush()

    def _load_last_hash(self):
        """Load the last hash from storage for chain continuity."""
        chain_state_file = self.storage_path / "chain_state.json"
        if chain_state_file.exists():
            with open(chain_state_file, 'r') as f:
                state = json.load(f)
                self._last_hash = state.get("last_hash", self.GENESIS_HASH)
                self._entry_counter = state.get("entry_counter", 0)

    def _save_chain_state(self):
        """Save chain state for recovery."""
        chain_state_file = self.storage_path / "chain_state.json"
        with open(chain_state_file, 'w') as f:
            json.dump({
                "last_hash": self._last_hash,
                "entry_counter": self._entry_counter,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }, f)

    def _start_auto_flush(self):
        """Start background thread for auto-flushing."""
        def flush_loop():
            while True:
                time.sleep(self.flush_interval_seconds)
                self.flush()

        thread = threading.Thread(target=flush_loop, daemon=True)
        thread.start()

    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        self._entry_counter += 1
        timestamp = int(time.time() * 1000)
        return f"audit-{timestamp}-{self._entry_counter:08d}"

    def _filter_sensitive_data(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from metadata."""
        sensitive_keys = {
            "ciphertext", "plaintext", "secret_key", "api_key",
            "password", "token", "features", "predictions",
            "secret", "private_key", "credential"
        }

        def filter_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for k, v in d.items():
                if k.lower() in sensitive_keys:
                    result[k] = "[REDACTED]"
                elif isinstance(v, dict):
                    result[k] = filter_dict(v)
                elif isinstance(v, list):
                    result[k] = [
                        filter_dict(item) if isinstance(item, dict) else item
                        for item in v
                    ]
                else:
                    result[k] = v
            return result

        return filter_dict(metadata)

    def append(
        self,
        action: AuditAction,
        tenant_id: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        payload_size_bytes: Optional[int] = None,
    ) -> AuditEntry:
        """
        Append a new entry to the audit chain.

        Args:
            action: The audit action type
            tenant_id: Tenant identifier
            request_id: Optional request correlation ID
            user_id: Optional user identifier
            resource_id: Optional resource identifier
            resource_type: Optional resource type
            severity: Entry severity level
            status: Operation status
            metadata: Additional metadata (sensitive data auto-filtered)
            latency_ms: Operation latency in milliseconds
            payload_size_bytes: Payload size in bytes

        Returns:
            The created AuditEntry
        """
        with self._lock:
            # Filter sensitive data from metadata
            safe_metadata = self._filter_sensitive_data(metadata or {})

            # Create entry
            entry = AuditEntry(
                entry_id=self._generate_entry_id(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action.value,
                tenant_id=tenant_id,
                previous_hash=self._last_hash,
                request_id=request_id,
                user_id=user_id,
                resource_id=resource_id,
                resource_type=resource_type,
                severity=severity.value,
                status=status,
                metadata=safe_metadata,
                latency_ms=latency_ms,
                payload_size_bytes=payload_size_bytes,
            )

            # Update chain state
            self._last_hash = entry.current_hash
            self._entries.append(entry)

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(entry)
                except Exception:
                    pass  # Don't let callback failures affect logging

            # Auto-flush if memory limit reached
            if len(self._entries) >= self.max_entries_in_memory:
                self._flush_to_storage()

            return entry

    def flush(self):
        """Flush in-memory entries to storage."""
        with self._lock:
            self._flush_to_storage()

    def _flush_to_storage(self):
        """Internal flush implementation (must hold lock)."""
        if not self._entries:
            return

        # Create daily log file
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit-{date_str}.jsonl"

        # Append entries to file
        with open(log_file, 'a') as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

        # Clear in-memory entries
        self._entries = []

        # Save chain state
        self._save_chain_state()

    def verify_chain(self, entries: Optional[List[AuditEntry]] = None) -> bool:
        """
        Verify integrity of the hash chain.

        Args:
            entries: Optional list of entries to verify. If None, verifies in-memory entries.

        Returns:
            True if chain is valid, False otherwise
        """
        entries = entries or self._entries

        if not entries:
            return True

        # Verify first entry
        if entries[0].previous_hash != self.GENESIS_HASH:
            # Check if it chains from previous storage
            pass

        # Verify chain integrity
        for i, entry in enumerate(entries):
            if i == 0:
                continue
            if not entry.verify_chain(entries[i - 1]):
                return False

        return True

    def export_for_compliance(
        self,
        start_date: datetime,
        end_date: datetime,
        tenant_id: Optional[str] = None,
        actions: Optional[List[AuditAction]] = None,
    ) -> Dict[str, Any]:
        """
        Export audit logs for compliance review.

        Args:
            start_date: Start of export range
            end_date: End of export range
            tenant_id: Optional tenant filter
            actions: Optional action type filter

        Returns:
            Export package with entries and integrity proof
        """
        entries = []

        # Collect entries from storage
        for log_file in sorted(self.storage_path.glob("audit-*.jsonl")):
            file_date_str = log_file.stem.replace("audit-", "")
            try:
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
                if file_date < start_date.date() or file_date > end_date.date():
                    continue
            except (ValueError, AttributeError):
                continue

            with open(log_file, 'r') as f:
                for line in f:
                    entry_dict = json.loads(line)
                    entry = AuditEntry.from_dict(entry_dict)

                    # Apply filters
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    if entry_time < start_date or entry_time > end_date:
                        continue
                    if tenant_id and entry.tenant_id != tenant_id:
                        continue
                    if actions and entry.action not in [a.value for a in actions]:
                        continue

                    entries.append(entry)

        # Add in-memory entries
        with self._lock:
            for entry in self._entries:
                entry_time = datetime.fromisoformat(entry.timestamp)
                if entry_time < start_date or entry_time > end_date:
                    continue
                if tenant_id and entry.tenant_id != tenant_id:
                    continue
                if actions and entry.action not in [a.value for a in actions]:
                    continue
                entries.append(entry)

        # Verify chain integrity
        is_valid = self.verify_chain(entries)

        # Compute export hash
        export_content = json.dumps([e.to_dict() for e in entries], sort_keys=True)
        export_hash = f"sha256:{hashlib.sha256(export_content.encode()).hexdigest()}"

        return {
            "export_id": f"export-{int(time.time() * 1000)}",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "tenant_id": tenant_id,
            "entry_count": len(entries),
            "chain_valid": is_valid,
            "export_hash": export_hash,
            "first_entry_hash": entries[0].current_hash if entries else None,
            "last_entry_hash": entries[-1].current_hash if entries else None,
            "entries": [e.to_dict() for e in entries]
        }

    def register_callback(self, callback: Callable[[AuditEntry], None]):
        """Register callback for new audit entries."""
        self._callbacks.append(callback)

    def get_recent_entries(self, count: int = 100) -> List[AuditEntry]:
        """Get recent entries from memory."""
        with self._lock:
            return self._entries[-count:]


# Global audit chain instance
_audit_chain: Optional[AuditChain] = None


def get_audit_chain() -> AuditChain:
    """Get or create global audit chain instance."""
    global _audit_chain
    if _audit_chain is None:
        storage_path = Path(os.environ.get(
            "FHE_GBDT_AUDIT_PATH",
            "/var/log/fhe-gbdt/audit"
        ))
        _audit_chain = AuditChain(storage_path=storage_path)
    return _audit_chain


def audit_log(
    action: AuditAction,
    tenant_id: str,
    **kwargs
) -> AuditEntry:
    """Convenience function to log audit entry."""
    return get_audit_chain().append(action, tenant_id, **kwargs)


# Decorators for automatic audit logging
def audit_operation(action: AuditAction, resource_type: str):
    """Decorator to automatically audit function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract context
            tenant_id = kwargs.get('tenant_id', 'unknown')
            request_id = kwargs.get('request_id')
            resource_id = kwargs.get('resource_id') or kwargs.get('model_id')

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                audit_log(
                    action=action,
                    tenant_id=tenant_id,
                    request_id=request_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    status="success",
                    latency_ms=latency_ms,
                )

                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                audit_log(
                    action=action,
                    tenant_id=tenant_id,
                    request_id=request_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    severity=AuditSeverity.ERROR,
                    status="failure",
                    metadata={"error": str(e)},
                    latency_ms=latency_ms,
                )

                raise

        return wrapper
    return decorator
