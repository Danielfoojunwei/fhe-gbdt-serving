-- Billing Database Schema
-- FHE-GBDT-Serving Platform

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Plans Table
-- Defines available subscription tiers
-- ============================================================================
CREATE TABLE IF NOT EXISTS plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    price_cents BIGINT NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD',
    prediction_limit BIGINT NOT NULL DEFAULT 0,  -- 0 = unlimited
    overage_price_micros BIGINT NOT NULL DEFAULT 0,  -- Price per prediction in microdollars (1/1,000,000 of dollar)
    features JSONB DEFAULT '{}',
    stripe_price_id TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for active plans lookup
CREATE INDEX IF NOT EXISTS idx_plans_active ON plans(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_plans_stripe_price_id ON plans(stripe_price_id) WHERE stripe_price_id IS NOT NULL;

-- ============================================================================
-- Subscriptions Table
-- Tracks tenant subscription state and Stripe integration
-- ============================================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL UNIQUE,
    plan_id UUID NOT NULL REFERENCES plans(id),
    status TEXT NOT NULL DEFAULT 'active',  -- active, canceled, past_due, trialing, paused
    stripe_subscription_id TEXT UNIQUE,
    stripe_customer_id TEXT,
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    canceled_at TIMESTAMP WITH TIME ZONE,
    cancel_at_period_end BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_subscriptions_tenant_id ON subscriptions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe_sub_id ON subscriptions(stripe_subscription_id) WHERE stripe_subscription_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe_cust_id ON subscriptions(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_subscriptions_plan_id ON subscriptions(plan_id);

-- ============================================================================
-- Usage Records Table
-- Tracks prediction usage per billing period
-- ============================================================================
CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    subscription_id UUID REFERENCES subscriptions(id),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    predictions_count BIGINT NOT NULL DEFAULT 0,
    compute_time_ms BIGINT NOT NULL DEFAULT 0,
    data_transfer_bytes BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, period_start, period_end)
);

-- Indexes for usage queries
CREATE INDEX IF NOT EXISTS idx_usage_records_tenant_period ON usage_records(tenant_id, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_usage_records_subscription_id ON usage_records(subscription_id);

-- ============================================================================
-- Usage Events Table
-- Individual usage events for auditing and replay (append-only)
-- ============================================================================
CREATE TABLE IF NOT EXISTS usage_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    idempotency_key TEXT,
    predictions_count BIGINT NOT NULL DEFAULT 0,
    compute_time_ms BIGINT NOT NULL DEFAULT 0,
    data_transfer_bytes BIGINT NOT NULL DEFAULT 0,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, idempotency_key)
);

-- Index for idempotency checks
CREATE INDEX IF NOT EXISTS idx_usage_events_idempotency ON usage_events(tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_usage_events_tenant_time ON usage_events(tenant_id, recorded_at);

-- ============================================================================
-- Invoices Table
-- Stores invoice records with Stripe integration
-- ============================================================================
CREATE TABLE IF NOT EXISTS invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT NOT NULL,
    subscription_id UUID REFERENCES subscriptions(id),
    stripe_invoice_id TEXT UNIQUE,
    status TEXT NOT NULL DEFAULT 'draft',  -- draft, open, paid, void, uncollectible
    currency TEXT NOT NULL DEFAULT 'USD',
    subtotal_cents BIGINT NOT NULL DEFAULT 0,
    tax_cents BIGINT NOT NULL DEFAULT 0,
    total_cents BIGINT NOT NULL DEFAULT 0,
    amount_paid_cents BIGINT NOT NULL DEFAULT 0,
    amount_due_cents BIGINT NOT NULL DEFAULT 0,
    line_items JSONB DEFAULT '[]',
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    due_date TIMESTAMP WITH TIME ZONE,
    paid_at TIMESTAMP WITH TIME ZONE,
    hosted_invoice_url TEXT,
    pdf_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for invoice queries
CREATE INDEX IF NOT EXISTS idx_invoices_tenant_id ON invoices(tenant_id);
CREATE INDEX IF NOT EXISTS idx_invoices_subscription_id ON invoices(subscription_id);
CREATE INDEX IF NOT EXISTS idx_invoices_stripe_id ON invoices(stripe_invoice_id) WHERE stripe_invoice_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_created_at ON invoices(created_at DESC);

-- ============================================================================
-- Webhook Events Table
-- Stores processed Stripe webhook events for idempotency and auditing
-- ============================================================================
CREATE TABLE IF NOT EXISTS webhook_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stripe_event_id TEXT UNIQUE NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT false,
    error_message TEXT,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for event lookup
CREATE INDEX IF NOT EXISTS idx_webhook_events_stripe_id ON webhook_events(stripe_event_id);
CREATE INDEX IF NOT EXISTS idx_webhook_events_type ON webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_events_processed ON webhook_events(processed) WHERE processed = false;

-- ============================================================================
-- Audit Log Table
-- Immutable audit trail for billing operations
-- ============================================================================
CREATE TABLE IF NOT EXISTS billing_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,  -- subscription, invoice, usage, plan
    resource_id TEXT,
    old_value JSONB,
    new_value JSONB,
    actor TEXT,  -- system, webhook, api, tenant_id
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for audit queries
CREATE INDEX IF NOT EXISTS idx_audit_log_tenant ON billing_audit_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON billing_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON billing_audit_log(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON billing_audit_log(created_at DESC);

-- ============================================================================
-- Seed Data: Plans
-- ============================================================================

-- Free Tier
INSERT INTO plans (id, name, description, price_cents, currency, prediction_limit, overage_price_micros, features, is_active)
VALUES (
    'a0000000-0000-0000-0000-000000000001',
    'free',
    'Free tier for development and testing',
    0,
    'USD',
    1000,  -- 1,000 predictions/month
    0,     -- No overage allowed
    '{
        "support": "community",
        "sla": "none",
        "data_retention_days": 7,
        "max_models": 1,
        "max_concurrent_requests": 1,
        "encryption": "standard"
    }',
    true
) ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    price_cents = EXCLUDED.price_cents,
    prediction_limit = EXCLUDED.prediction_limit,
    overage_price_micros = EXCLUDED.overage_price_micros,
    features = EXCLUDED.features,
    updated_at = CURRENT_TIMESTAMP;

-- Pro Tier
INSERT INTO plans (id, name, description, price_cents, currency, prediction_limit, overage_price_micros, features, is_active)
VALUES (
    'a0000000-0000-0000-0000-000000000002',
    'pro',
    'Professional tier for production workloads',
    9900,   -- $99/month
    'USD',
    100000, -- 100,000 predictions/month
    1000,   -- $0.001 per prediction overage (1000 microdollars)
    '{
        "support": "email",
        "sla": "99.9%",
        "data_retention_days": 30,
        "max_models": 10,
        "max_concurrent_requests": 10,
        "encryption": "standard",
        "priority_queue": true,
        "analytics_dashboard": true
    }',
    true
) ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    price_cents = EXCLUDED.price_cents,
    prediction_limit = EXCLUDED.prediction_limit,
    overage_price_micros = EXCLUDED.overage_price_micros,
    features = EXCLUDED.features,
    updated_at = CURRENT_TIMESTAMP;

-- Enterprise Tier
INSERT INTO plans (id, name, description, price_cents, currency, prediction_limit, overage_price_micros, features, is_active)
VALUES (
    'a0000000-0000-0000-0000-000000000003',
    'enterprise',
    'Enterprise tier with unlimited usage and premium support',
    0,  -- Custom pricing (contact sales)
    'USD',
    0,  -- Unlimited (0 = no limit)
    0,  -- No overage (included in custom pricing)
    '{
        "support": "dedicated",
        "sla": "99.99%",
        "data_retention_days": 365,
        "max_models": -1,
        "max_concurrent_requests": -1,
        "encryption": "customer_managed_keys",
        "priority_queue": true,
        "analytics_dashboard": true,
        "custom_integrations": true,
        "dedicated_infrastructure": true,
        "hipaa_compliant": true,
        "soc2_compliant": true
    }',
    true
) ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    price_cents = EXCLUDED.price_cents,
    prediction_limit = EXCLUDED.prediction_limit,
    overage_price_micros = EXCLUDED.overage_price_micros,
    features = EXCLUDED.features,
    updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- Functions
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_plans_updated_at ON plans;
CREATE TRIGGER update_plans_updated_at BEFORE UPDATE ON plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_subscriptions_updated_at ON subscriptions;
CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_usage_records_updated_at ON usage_records;
CREATE TRIGGER update_usage_records_updated_at BEFORE UPDATE ON usage_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_invoices_updated_at ON invoices;
CREATE TRIGGER update_invoices_updated_at BEFORE UPDATE ON invoices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate overage for a usage record
CREATE OR REPLACE FUNCTION calculate_overage(
    p_predictions_count BIGINT,
    p_prediction_limit BIGINT,
    p_overage_price_micros BIGINT
) RETURNS TABLE(overage_count BIGINT, overage_cost_cents BIGINT) AS $$
BEGIN
    IF p_prediction_limit = 0 THEN
        -- Unlimited plan, no overage
        RETURN QUERY SELECT 0::BIGINT, 0::BIGINT;
    ELSIF p_predictions_count > p_prediction_limit THEN
        -- Calculate overage
        overage_count := p_predictions_count - p_prediction_limit;
        -- Convert microdollars to cents: (count * microdollars) / 10000
        overage_cost_cents := (overage_count * p_overage_price_micros) / 10000;
        RETURN QUERY SELECT overage_count, overage_cost_cents;
    ELSE
        RETURN QUERY SELECT 0::BIGINT, 0::BIGINT;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Views
-- ============================================================================

-- View for subscription details with plan info
CREATE OR REPLACE VIEW subscription_details AS
SELECT
    s.id,
    s.tenant_id,
    s.status,
    s.stripe_subscription_id,
    s.stripe_customer_id,
    s.current_period_start,
    s.current_period_end,
    s.canceled_at,
    s.cancel_at_period_end,
    s.created_at,
    s.updated_at,
    p.id as plan_id,
    p.name as plan_name,
    p.price_cents as plan_price_cents,
    p.prediction_limit as plan_prediction_limit,
    p.overage_price_micros as plan_overage_price_micros,
    p.features as plan_features
FROM subscriptions s
JOIN plans p ON s.plan_id = p.id;

-- View for current usage with overage calculation
CREATE OR REPLACE VIEW current_usage AS
SELECT
    u.id,
    u.tenant_id,
    u.subscription_id,
    u.period_start,
    u.period_end,
    u.predictions_count,
    u.compute_time_ms,
    u.data_transfer_bytes,
    p.prediction_limit,
    CASE
        WHEN p.prediction_limit = 0 THEN 0
        WHEN u.predictions_count > p.prediction_limit THEN u.predictions_count - p.prediction_limit
        ELSE 0
    END as overage_count,
    CASE
        WHEN p.prediction_limit = 0 THEN 0
        WHEN u.predictions_count > p.prediction_limit THEN
            ((u.predictions_count - p.prediction_limit) * p.overage_price_micros) / 10000
        ELSE 0
    END as overage_cost_cents,
    CASE
        WHEN p.prediction_limit = 0 THEN 0
        ELSE ROUND((u.predictions_count::NUMERIC / p.prediction_limit::NUMERIC) * 100, 2)
    END as usage_percentage,
    u.updated_at
FROM usage_records u
JOIN subscriptions s ON u.subscription_id = s.id
JOIN plans p ON s.plan_id = p.id
WHERE CURRENT_TIMESTAMP BETWEEN u.period_start AND u.period_end;
