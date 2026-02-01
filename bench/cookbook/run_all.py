import argparse
import os
import time
from .run_recipe_xgboost import run_recipe_xgboost
from .run_recipe_lightgbm import run_recipe_lightgbm
from .run_recipe_catboost import run_recipe_catboost
from .report import generate_combined_report

def run_all(quick=False):
    print("=== Starting Cookbook Suite ===")
    
    output_dir = f"bench/reports/cookbook/{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    results.append(run_recipe_xgboost(quick, output_dir))
    results.append(run_recipe_lightgbm(quick, output_dir))
    results.append(run_recipe_catboost(quick, output_dir))
    
    report_path = generate_combined_report(results, output_dir)
    print(f"=== Cookbook Suite Complete ===")
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_all(quick=args.quick)
