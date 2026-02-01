"""
Security test to detect accidental logging of sensitive data.

This test checks for patterns that could indicate sensitive FHE data being logged.
It focuses on LOGGING statements (print, log.Printf, etc.) that reference sensitive data,
NOT the legitimate use of these terms in crypto code.

Allowed:
- Using 'plaintext' as a parameter name in crypto functions
- Accessing .Payload in proto message handling (not logged)
- Variables named secret_key, eval_key in crypto implementation

Forbidden:
- Logging raw feature/payload/ciphertext data: log.Printf("data: %v", req.Payload)
- Printing secret keys or plaintext values
"""
import unittest
import re
import os
import glob

# Patterns that indicate LOGGING of sensitive data (not just usage)
FORBIDDEN_LOG_PATTERNS = [
    # Go: logging sensitive data values
    r'log\.Printf?\([^)]*%[vsdfxXp][^)]*,\s*(payload|ciphertext|feature|secret|plaintext)',
    r'log\.Printf?\([^)]*,\s*[^)]*\.(Payload|Ciphertext|Feature|SecretKey)\)',
    r'log\.Print[fl]?n?\([^)]*\.(Payload|Ciphertext|Feature|SecretKey)',
    # Python: print statements with sensitive data
    r'print\([^)]*\b(payload|ciphertext|secret_key|eval_key)\s*[,\)]',
    r'print\(f["\'].*\{[^}]*\b(payload|ciphertext|secret_key|eval_key)',
    # Go: fmt.Print with sensitive data
    r'fmt\.Print[fl]?n?\([^)]*\.(Payload|Ciphertext|Feature|SecretKey)',
]

# Files that are allowed to use sensitive terminology (crypto implementations)
ALLOWED_FILES = [
    'sdk/python/crypto.py',           # Core crypto library - needs these terms
    'services/keystore/crypto.go',    # Envelope encryption implementation
    'services/keystore/vault/client.go',  # Vault integration
    'sdk/python/client.py',           # Client SDK - accesses proto fields
    'services/gateway/server.go',     # Gateway - accesses proto fields (not logged)
]

class TestNoPlaintextLogs(unittest.TestCase):
    """
    Security guardrail tests to prevent sensitive data leakage in logs.

    These tests look for logging statements that might inadvertently expose:
    - Plaintext feature values
    - Ciphertext payloads (encrypted data)
    - Secret keys or evaluation keys

    Note: Legitimate crypto code that uses these terms for variables/parameters
    is explicitly allowed. We only flag LOGGING of this data.
    """

    def test_no_forbidden_patterns_in_go(self):
        """Check Go services don't log sensitive FHE data."""
        go_files = glob.glob('services/**/*.go', recursive=True)
        violations = []
        for filepath in go_files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    for pattern in FORBIDDEN_LOG_PATTERNS:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if it's a comment
                            stripped = line.strip()
                            if stripped.startswith('//') or stripped.startswith('/*'):
                                continue
                            violations.append(f"{filepath}:{line_num}: '{line.strip()}'")
        self.assertEqual(violations, [], f"Forbidden logging of sensitive data found:\n" + "\n".join(violations))

    def test_no_forbidden_patterns_in_python(self):
        """Check Python SDK/services don't log sensitive FHE data."""
        py_files = glob.glob('sdk/python/**/*.py', recursive=True) + glob.glob('services/compiler/**/*.py', recursive=True)
        violations = []
        for filepath in py_files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    for pattern in FORBIDDEN_LOG_PATTERNS:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if it's a comment or docstring
                            stripped = line.strip()
                            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                                continue
                            violations.append(f"{filepath}:{line_num}: '{line.strip()}'")
        self.assertEqual(violations, [], f"Forbidden logging of sensitive data found:\n" + "\n".join(violations))

    def test_audit_logs_redact_data(self):
        """Verify that audit logs in gateway only log metadata, not payloads."""
        gateway_file = 'services/gateway/server.go'
        with open(gateway_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check that AUDIT logs exist (good)
        self.assertIn('AUDIT:', content, "Gateway should have audit logging")

        # Check that audit logs don't include payload data
        audit_pattern = r'log\.Printf\("AUDIT:[^"]*",\s*[^)]+\)'
        for match in re.finditer(audit_pattern, content):
            audit_line = match.group()
            # Ensure payload/ciphertext is not in the format args
            self.assertNotIn('Payload', audit_line,
                f"Audit log should not include payload: {audit_line}")
            self.assertNotIn('Ciphertext', audit_line,
                f"Audit log should not include ciphertext: {audit_line}")

if __name__ == '__main__':
    unittest.main()
