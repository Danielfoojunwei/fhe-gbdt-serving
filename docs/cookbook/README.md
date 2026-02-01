# FHE-GBDT Cookbook 🍳

Welcome to the FHE-GBDT Cookbook! This guide provides recipe-style instructions for running end-to-end encrypted inference with real models.

## Recipes

### 🚀 [Quickstart](00_quickstart.md)
Go from zero to encrypted prediction in 5 minutes using the canonical flow.

### 🌲 [XGBoost Classification](01_xgboost_classification.md)
Train an XGBoost model on the Breast Cancer dataset and run encrypted inference.

### 🍃 [LightGBM Regression](02_lightgbm_regression.md)
Train a LightGBM regressor on the California Housing dataset and validate continuous predictions.

### 🐱 [CatBoost Classification](03_catboost_classification.md)
Train a CatBoost classifier and leverage symmetric trees for optimized FHE performance.

---

## Technical Deep Dives

### ⚖️ [Trade-offs & Performance](04_tradeoffs.md)
Understand the performance and structural differences between libraries in the FHE context.

### 🔧 [Troubleshooting](05_troubleshooting.md)
Common issues, error codes, and solutions.

---

## Running the Recipes

All recipes have corresponding runnable scripts in `bench/cookbook/`. You can run them individually or all together.

```bash
# Run all recipes
make cookbook

# Run specific recipe
python bench/cookbook/run_recipe_xgboost.py
```
