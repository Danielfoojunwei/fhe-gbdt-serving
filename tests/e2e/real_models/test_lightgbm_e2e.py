import pytest
from bench.cookbook.run_recipe_lightgbm import run_recipe_lightgbm

def test_lightgbm_recipe_e2e():
    """
    Runs the LightGBM cookbook recipe in 'quick' mode as an E2E test.
    """
    result = run_recipe_lightgbm(quick=True, output_dir="bench/reports/test_run")
    
    assert result.model_type == "lightgbm"
    assert result.correctness_passed is True
    # Sanity checks
    assert result.throughput_eps > 0
    assert result.server_counters["switches"] > 0
