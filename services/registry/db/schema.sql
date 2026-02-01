-- Registry Database Schema

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    name TEXT NOT NULL,
    library_type TEXT NOT NULL, -- xgboost, lightgbm, catboost
    model_content_path TEXT NOT NULL, -- MinIO path
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS compiled_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id),
    profile TEXT NOT NULL, -- latency, throughput
    plan_id TEXT NOT NULL, -- content hash
    plan_content_path TEXT NOT NULL, -- MinIO path
    status TEXT NOT NULL DEFAULT 'pending', -- pending, successful, failed
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
