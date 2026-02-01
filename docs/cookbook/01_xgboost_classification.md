# Recipe: XGBoost Classification 🌲

**Goal**: Train an XGBoost model, serve it over FHE, and validate correctness.

## 1. Train Model
Use the standard `xgboost` library.
```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
model.fit(data.data, data.target)

# Export as JSON (Crucial for FHE compiler)
model.save_model("xgb_model.json")
```

## 2. Create Feature Spec
Define the expected input schema for the SDK.
```python
feature_spec = {
    "feature_names": list(data.feature_names),
    "quantization_scale": 1.0
}
# Save as feature_spec.json
```

## 3. Register & Compile
Use the FHE-GBDT CLI or scripts.
```bash
# Upload and compile
python -m fhe_gbdt_cli register \
  --model xgb_model.json \
  --spec feature_spec.json \
  --profile latency
```

## 4. Encrypted Prediction (Python SDK)
```python
from fhe_gbdt_sdk import Client

client = Client("http://localhost:8080", api_key="test-tenant.key")
client.load_model("xgb-model-id")

# Generate keys (one-time)
client.keygen()
client.upload_keys()

# Predict
x_sample = data.data[0]
prediction = client.predict_encrypted(x_sample)
print(f"Decrypted Probability: {prediction}")
```

## Performance Note
XGBoost trees are often irregular. The MOAI compiler will attempt to levelize them, but deep, unbalanced trees may result in higher depth and latency.
