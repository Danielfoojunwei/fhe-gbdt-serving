import pytest
from bench.cookbook.run_recipe_xgboost import run_recipe_xgboost

def test_xgboost_recipe_e2e():
    """
    Runs the XGBoost cookbook recipe in 'quick' mode as an E2E test.
    Asserts that the result is successful and metrics are sanity-checked.
    """
    result = run_recipe_xgboost(quick=True, output_dir="bench/reports/test_run")
    
    assert result.model_type == "xgboost"
    assert result.correctness_passed is True
    assert result.p50_latency_ms > 0
    assert result.server_counters["rotations"] > 0
