-- Metering Service Database Schema
-- Requires TimescaleDB extension for time-series hypertables
-- PostgreSQL 14+ with TimescaleDB 2.x

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- Core Usage Events Table (TimescaleDB Hypertable)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS usage_events (
    event_id UUID DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    model_id TEXT,

    -- Timestamp is the partition key for TimescaleDB
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Metrics
    predictions_count BIGINT DEFAULT 0,
    compute_ms BIGINT DEFAULT 0,
    input_bytes BIGINT DEFAULT 0,
    output_bytes BIGINT DEFAULT 0,

    -- Flexible metadata as JSONB
    metadata JSONB DEFAULT '{}',

    -- Ingestion timestamp for debugging
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (event_id, timestamp)
);

-- Convert to TimescaleDB hypertable (partition by time)
-- Chunk interval of 1 day for high-throughput ingestion
SELECT create_hypertable('usage_events', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Enable compression after 7 days
ALTER TABLE usage_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy: compress chunks older than 7 days
SELECT add_compression_policy('usage_events', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy: keep data for 2 years
SELECT add_retention_policy('usage_events', INTERVAL '2 years', if_not_exists => TRUE);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_usage_events_tenant_time
    ON usage_events (tenant_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_usage_events_tenant_model
    ON usage_events (tenant_id, model_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_usage_events_event_type
    ON usage_events (event_type, timestamp DESC);

-- GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_usage_events_metadata
    ON usage_events USING GIN (metadata);

-- ==============================================================================
-- Continuous Aggregates (Materialized Views for Real-Time Analytics)
-- ==============================================================================

-- Hourly aggregates per tenant
CREATE MATERIALIZED VIEW IF NOT EXISTS usage_hourly
WITH (timescaledb.continuous) AS
SELECT
    tenant_id,
    model_id,
    event_type,
    time_bucket('1 hour', timestamp) AS bucket,
    COUNT(*) AS event_count,
    SUM(predictions_count) AS total_predictions,
    SUM(compute_ms) AS total_compute_ms,
    AVG(compute_ms) AS avg_compute_ms,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY compute_ms) AS p50_compute_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY compute_ms) AS p95_compute_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY compute_ms) AS p99_compute_ms,
    SUM(input_bytes) AS total_input_bytes,
    SUM(output_bytes) AS total_output_bytes
FROM usage_events
GROUP BY tenant_id, model_id, event_type, bucket
WITH NO DATA;

-- Refresh policy: refresh every 5 minutes, keeping up to date
SELECT add_continuous_aggregate_policy('usage_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- Daily aggregates for longer-term trends
CREATE MATERIALIZED VIEW IF NOT EXISTS usage_daily
WITH (timescaledb.continuous) AS
SELECT
    tenant_id,
    model_id,
    event_type,
    time_bucket('1 day', timestamp) AS bucket,
    COUNT(*) AS event_count,
    SUM(predictions_count) AS total_predictions,
    SUM(compute_ms) AS total_compute_ms,
    AVG(compute_ms) AS avg_compute_ms,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY compute_ms) AS p50_compute_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY compute_ms) AS p95_compute_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY compute_ms) AS p99_compute_ms,
    SUM(input_bytes) AS total_input_bytes,
    SUM(output_bytes) AS total_output_bytes
FROM usage_events
GROUP BY tenant_id, model_id, event_type, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('usage_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Monthly aggregates for billing
CREATE MATERIALIZED VIEW IF NOT EXISTS usage_monthly
WITH (timescaledb.continuous) AS
SELECT
    tenant_id,
    model_id,
    event_type,
    time_bucket('1 month', timestamp) AS bucket,
    COUNT(*) AS event_count,
    SUM(predictions_count) AS total_predictions,
    SUM(compute_ms) AS total_compute_ms,
    AVG(compute_ms) AS avg_compute_ms,
    SUM(input_bytes) AS total_input_bytes,
    SUM(output_bytes) AS total_output_bytes
FROM usage_events
GROUP BY tenant_id, model_id, event_type, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('usage_monthly',
    start_offset => INTERVAL '60 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ==============================================================================
-- Quotas Table
-- ==============================================================================

CREATE TABLE IF NOT EXISTS quotas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    quota_type TEXT NOT NULL,

    -- Limit configuration
    limit_value BIGINT NOT NULL,
    warning_threshold_percent INT DEFAULT 80,

    -- Current period
    period_start TIMESTAMPTZ NOT NULL DEFAULT date_trunc('month', NOW()),
    period_end TIMESTAMPTZ NOT NULL DEFAULT date_trunc('month', NOW()) + INTERVAL '1 month',

    -- Webhook configuration for notifications
    webhook_url TEXT,
    webhook_secret TEXT,

    -- Status tracking
    last_warning_sent_at TIMESTAMPTZ,
    last_exceeded_sent_at TIMESTAMPTZ,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(tenant_id, quota_type)
);

CREATE INDEX IF NOT EXISTS idx_quotas_tenant ON quotas (tenant_id);
CREATE INDEX IF NOT EXISTS idx_quotas_period ON quotas (period_start, period_end);

-- ==============================================================================
-- Quota Usage Cache (Updated Periodically from Aggregates)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS quota_usage_cache (
    tenant_id TEXT NOT NULL,
    quota_type TEXT NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,

    -- Cached usage value
    current_usage BIGINT NOT NULL DEFAULT 0,

    -- Last update timestamp
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (tenant_id, quota_type, period_start)
);

CREATE INDEX IF NOT EXISTS idx_quota_cache_tenant ON quota_usage_cache (tenant_id);

-- ==============================================================================
-- Webhook Notification Log
-- ==============================================================================

CREATE TABLE IF NOT EXISTS webhook_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    quota_type TEXT NOT NULL,
    notification_type TEXT NOT NULL, -- 'warning', 'exceeded', 'resolved'

    -- Notification details
    threshold_percent INT,
    current_usage BIGINT,
    limit_value BIGINT,

    -- Webhook delivery
    webhook_url TEXT NOT NULL,
    payload JSONB NOT NULL,

    -- Delivery status
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'sent', 'failed', 'retrying'
    attempts INT DEFAULT 0,
    last_attempt_at TIMESTAMPTZ,
    last_error TEXT,
    delivered_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_webhook_notifications_tenant
    ON webhook_notifications (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_notifications_status
    ON webhook_notifications (status, created_at);

-- ==============================================================================
-- Helper Functions
-- ==============================================================================

-- Function to get current month usage for a tenant/quota type
CREATE OR REPLACE FUNCTION get_current_month_usage(
    p_tenant_id TEXT,
    p_quota_type TEXT
) RETURNS BIGINT AS $$
DECLARE
    v_usage BIGINT;
    v_period_start TIMESTAMPTZ;
    v_period_end TIMESTAMPTZ;
BEGIN
    v_period_start := date_trunc('month', NOW());
    v_period_end := v_period_start + INTERVAL '1 month';

    CASE p_quota_type
        WHEN 'PREDICTIONS_PER_MONTH' THEN
            SELECT COALESCE(SUM(predictions_count), 0) INTO v_usage
            FROM usage_events
            WHERE tenant_id = p_tenant_id
              AND timestamp >= v_period_start
              AND timestamp < v_period_end
              AND event_type = 'PREDICTION';

        WHEN 'COMPUTE_MS_PER_MONTH' THEN
            SELECT COALESCE(SUM(compute_ms), 0) INTO v_usage
            FROM usage_events
            WHERE tenant_id = p_tenant_id
              AND timestamp >= v_period_start
              AND timestamp < v_period_end;

        WHEN 'DATA_BYTES_PER_MONTH' THEN
            SELECT COALESCE(SUM(input_bytes + output_bytes), 0) INTO v_usage
            FROM usage_events
            WHERE tenant_id = p_tenant_id
              AND timestamp >= v_period_start
              AND timestamp < v_period_end;

        WHEN 'MODELS_TOTAL' THEN
            SELECT COUNT(DISTINCT model_id) INTO v_usage
            FROM usage_events
            WHERE tenant_id = p_tenant_id
              AND model_id IS NOT NULL;

        ELSE
            v_usage := 0;
    END CASE;

    RETURN v_usage;
END;
$$ LANGUAGE plpgsql;

-- Function to update quota usage cache
CREATE OR REPLACE FUNCTION update_quota_usage_cache(
    p_tenant_id TEXT
) RETURNS VOID AS $$
DECLARE
    v_quota RECORD;
    v_usage BIGINT;
    v_period_start TIMESTAMPTZ;
BEGIN
    v_period_start := date_trunc('month', NOW());

    FOR v_quota IN
        SELECT quota_type FROM quotas WHERE tenant_id = p_tenant_id
    LOOP
        v_usage := get_current_month_usage(p_tenant_id, v_quota.quota_type);

        INSERT INTO quota_usage_cache (tenant_id, quota_type, period_start, current_usage, updated_at)
        VALUES (p_tenant_id, v_quota.quota_type, v_period_start, v_usage, NOW())
        ON CONFLICT (tenant_id, quota_type, period_start)
        DO UPDATE SET current_usage = v_usage, updated_at = NOW();
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- Triggers
-- ==============================================================================

-- Trigger to update updated_at on quotas
CREATE OR REPLACE FUNCTION update_quotas_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_quotas_updated_at
    BEFORE UPDATE ON quotas
    FOR EACH ROW
    EXECUTE FUNCTION update_quotas_updated_at();

-- ==============================================================================
-- Initial Data / Seed Data
-- ==============================================================================

-- Insert default quota templates (can be customized per tenant)
-- These are examples, not enforced until explicitly set per tenant

COMMENT ON TABLE usage_events IS 'High-throughput usage event storage using TimescaleDB hypertable';
COMMENT ON TABLE quotas IS 'Per-tenant quota configurations and limits';
COMMENT ON TABLE quota_usage_cache IS 'Cached quota usage for fast lookups (updated periodically)';
COMMENT ON TABLE webhook_notifications IS 'Audit log of quota notification webhooks';
