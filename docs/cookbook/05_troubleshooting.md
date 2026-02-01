# Troubleshooting 🔧

## Common Issues

### 1. `ErrUnauthorized` / "missing API key"
**Cause**: The Gateway requires an `x-api-key` header for all requests.
**Fix**: Ensure your client is initialized with a valid key format `tenant_id.random_string`.
```python
client = Client(..., api_key="my-tenant.abc1234")
```

### 2. "Payload too large"
**Cause**: The request batch size creates a ciphertext exceeding 64MB.
**Fix**: Reduce `batch_size` in your predict request. The default limit is conservative.
- Try `B=1` or `B=32`.

### 3. Model Compilation Failed
**Cause**: Unsupported feature types or extremely deep trees.
**Fix**: 
- Inspect compiler logs.
- Ensure all features are numeric (float/int). Categorical string features must be pre-encoded.
- Limit `max_depth` to <= 8 for reasonable FHE performance.

### 4. Decryption Error / Garbage Output
**Cause**: Mismatch between the keys uploaded to Keystore and the keys used by the client.
**Fix**: Regenerate keys and re-upload. Ensure `compiled_model_id` matches.

## Error Codes

| Code | Meaning | Action |
| :--- | :--- | :--- |
| `401` | Unauthorized | Check API Key |
| `403` | Forbidden | Check Resource Ownership |
| `400` | Bad Request | Check Input Schema / Proto |
| `500` | Internal Error | Check Server Logs (Audit/OTel) |
