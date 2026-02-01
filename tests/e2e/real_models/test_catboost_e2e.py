import pytest
from bench.cookbook.run_recipe_catboost import run_recipe_catboost

def test_catboost_recipe_e2e():
    """
    Runs the CatBoost cookbook recipe in 'quick' mode as an E2E test.
    """
    result = run_recipe_catboost(quick=True, output_dir="bench/reports/test_run")
    
    assert result.model_type == "catboost"
    assert result.correctness_passed is True
    
    # CatBoost specific checks (symmetric trees usually imply fewer schemes switches if depth low)
    assert result.p95_latency_ms > 0
