# Architecture Overview

## System Components
1. **Gateway**: gRPC/HTTP Proxy, Authn/Authz, Rate Limiting.
2. **Registry**: Model and Plan metadata storage.
3. **Keystore**: Evaluation key storage with envelope encryption.
4. **Compiler**: Transpiles GBDT models (XGBoost, LightGBM, CatBoost) into ObliviousPlanIR.
5. **Runtime**: C++ engine executing encrypted inference using N2HE-hexl.

## Data Flow
1. Client generates keys and encrypts features.
2. Client sends encrypted request to Gateway.
3. Gateway fetches Plan and Keys from Registry/Keystore.
4. Gateway forwards request to Runtime.
5. Runtime executes plan and returns encrypted result.
6. Client decrypts result.
