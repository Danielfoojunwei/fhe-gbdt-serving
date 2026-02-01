# Recipe: LightGBM Regression 🍃

**Goal**: Train a LightGBM regressor, serve it, and validate regression metrics.

## 1. Train Model
```python
import lightgbm as lgb
from sklearn.datasets import load_diabetes

data = load_diabetes()
model = lgb.LGBMRegressor(n_estimators=20, num_leaves=15)
model.fit(data.data, data.target)

# Export as text (Standard LightGBM format)
model.booster_.save_model("lgb_model.txt")
```

## 2. Register & Compile
```python
# Using Python script instead of CLI
import requests

# 1. Register
resp = requests.post("http://localhost:8080/v1/models", json={
    "name": "lgbm-reg",
    "library": "lightgbm",
    "content": open("lgb_model.txt").read()
})
model_id = resp.json()["id"]

# 2. Compile
requests.post(f"http://localhost:8080/v1/models/{model_id}/compile?profile=throughput")
```

## 3. Verify Correctness
For regression, we verify the Mean Absolute Error (MAE) between the plaintext prediction and the decrypted FHE prediction.
```python
# ... (client setup) ...
fhe_pred = client.predict_encrypted(x_sample)
plain_pred = model.predict([x_sample])[0]

assert abs(fhe_pred - plain_pred) < 1e-4
print("Correctness check passed!")
```

## Pitfalls
- **Missing Values**: LightGBM handles NaNs by default. Ensure the `FeatureSpec` matches the client-side preprocessing policy (e.g., `nan_policy="zero"` or `nan_policy="mean"`).
