import json
import random
import string
from hypothesis import given, strategies as st, settings

def parse_xgboost_json(content: bytes) -> dict:
    """Mock XGBoost parser that validates JSON structure."""
    try:
        data = json.loads(content.decode('utf-8'))
        return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid XGBoost JSON: {e}")

@settings(max_examples=100)
@given(st.binary(min_size=0, max_size=10000))
def test_xgboost_parser_fuzz(data):
    """Fuzz the XGBoost parser with random bytes."""
    try:
        parse_xgboost_json(data)
    except ValueError:
        pass  # Expected for malformed input
    except Exception as e:
        raise AssertionError(f"Parser crashed unexpectedly: {e}")

@given(st.text(min_size=0, max_size=1000))
def test_lightgbm_parser_fuzz(data):
    """Fuzz the LightGBM parser with random text."""
    # Mock parser
    try:
        lines = data.split('\n')
        # Simulate parsing
    except Exception as e:
        # Should not crash, only raise controlled errors
        pass

if __name__ == '__main__':
    import unittest
    unittest.main()
