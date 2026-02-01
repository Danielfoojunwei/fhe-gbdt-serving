# Recipe: CatBoost Classification 🐱

**Goal**: Leverage CatBoost's symmetric trees for optimal FHE performance.

## 1. Train Model
CatBoost builds oblivious (symmetric) trees by default, which map perfectly to FHE vectorization.
```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
model = CatBoostClassifier(iterations=10, depth=4, verbose=False)
model.fit(data.data, data.target)

# Export as JSON
model.save_model("catboost_model.json", format="json")
```

## 2. Performance Advantage
Because CatBoost trees are symmetric:
- **Levelization is trivial**: Every path is the same length.
- **Batching is maximized**: All nodes at the same depth can be evaluated in a single SIMD batch (StepBundle).
- **Lower Rotations**: Fewer permutations required to align ciphertexts.

## 3. Compile & Run
```bash
python bench/cookbook/run_recipe_catboost.py
```

## Comparison
In our benchmarks, CatBoost models often achieve **20-30% lower latency** per tree compared to equivalent depth XGBoost models due to the regular structure.
