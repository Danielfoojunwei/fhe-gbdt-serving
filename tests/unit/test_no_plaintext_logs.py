import unittest
import re
import os
import glob

FORBIDDEN_PATTERNS = [
    r'\.payload\b',
    r'\.ciphertext\b',
    r'\.feature\b',
    r'plaintext',
    r'secret_key',
    r'eval_key',
    r'print\(.*(feature|payload|ciphertext)',
    r'log\..*(feature|payload|ciphertext)',
]

class TestNoPlaintextLogs(unittest.TestCase):
    def test_no_forbidden_patterns_in_go(self):
        go_files = glob.glob('services/**/*.go', recursive=True)
        violations = []
        for filepath in go_files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in FORBIDDEN_PATTERNS:
                    if re.search(pattern, content, re.IGNORECASE):
                        violations.append(f"{filepath}: matches '{pattern}'")
        self.assertEqual(violations, [], f"Forbidden patterns found:\n" + "\n".join(violations))

    def test_no_forbidden_patterns_in_python(self):
        py_files = glob.glob('sdk/python/**/*.py', recursive=True) + glob.glob('services/compiler/**/*.py', recursive=True)
        violations = []
        for filepath in py_files:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in FORBIDDEN_PATTERNS:
                    if re.search(pattern, content, re.IGNORECASE):
                        violations.append(f"{filepath}: matches '{pattern}'")
        self.assertEqual(violations, [], f"Forbidden patterns found:\n" + "\n".join(violations))

if __name__ == '__main__':
    unittest.main()
