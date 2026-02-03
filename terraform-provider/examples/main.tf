# FHE-GBDT Terraform Provider Examples
# This file demonstrates how to use the provider to manage FHE-GBDT resources

terraform {
  required_providers {
    fhegbdt = {
      source  = "registry.terraform.io/fhe-gbdt/fhe-gbdt"
      version = "~> 0.1.0"
    }
  }
}

# Provider configuration
provider "fhegbdt" {
  endpoint  = var.fhe_gbdt_endpoint
  api_key   = var.fhe_gbdt_api_key
  tenant_id = var.tenant_id
  region    = "us-east-1"
}

# Variables
variable "fhe_gbdt_endpoint" {
  description = "FHE-GBDT API endpoint"
  default     = "https://api.fhe-gbdt.dev"
}

variable "fhe_gbdt_api_key" {
  description = "API key for authentication"
  sensitive   = true
}

variable "tenant_id" {
  description = "Tenant/Organization ID"
}

# ============================================================================
# Data Sources - Read existing resources
# ============================================================================

# List all available regions
data "fhegbdt_regions" "available" {}

# List all models
data "fhegbdt_models" "all" {}

# Get current usage metrics
data "fhegbdt_usage" "current" {}

# ============================================================================
# Model Resources
# ============================================================================

# Deploy a fraud detection model
resource "fhegbdt_model" "fraud_detector" {
  name           = "fraud-detector-v2"
  description    = "Credit card fraud detection model"
  library_type   = "xgboost"
  model_path     = "s3://my-bucket/models/fraud_detector.json"
  compile_profile = "balanced"

  regions = ["us-east-1", "eu-west-1"]

  labels = {
    team        = "risk"
    environment = "production"
    compliance  = "pci-dss"
  }
}

# Deploy a credit scoring model
resource "fhegbdt_model" "credit_scorer" {
  name           = "credit-scorer-v3"
  description    = "Consumer credit scoring model"
  library_type   = "lightgbm"
  model_path     = "s3://my-bucket/models/credit_scorer.txt"
  compile_profile = "accurate"

  regions = ["us-east-1"]

  labels = {
    team        = "underwriting"
    environment = "production"
  }
}

# ============================================================================
# Key Resources
# ============================================================================

# Generate FHE encryption keys
resource "fhegbdt_key" "production_keys" {
  name      = "production-keys-2024"
  key_type  = "evaluation"
  algorithm = "tfhe"
}

# ============================================================================
# Monitoring Alerts
# ============================================================================

# Alert for high latency
resource "fhegbdt_alert" "fraud_latency" {
  name           = "fraud-detector-high-latency"
  model_id       = fhegbdt_model.fraud_detector.id
  metric         = "latency_p95"
  condition      = "gt"
  threshold      = 100  # 100ms
  window_minutes = 5

  notification_channels = ["slack", "email"]
}

# Alert for error rate
resource "fhegbdt_alert" "fraud_errors" {
  name           = "fraud-detector-error-rate"
  model_id       = fhegbdt_model.fraud_detector.id
  metric         = "error_rate"
  condition      = "gt"
  threshold      = 1  # 1%
  window_minutes = 10

  notification_channels = ["pagerduty"]
}

# Alert for model drift
resource "fhegbdt_alert" "credit_drift" {
  name           = "credit-scorer-drift"
  model_id       = fhegbdt_model.credit_scorer.id
  metric         = "drift"
  condition      = "gt"
  threshold      = 0.2  # PSI > 0.2
  window_minutes = 60

  notification_channels = ["email"]
}

# ============================================================================
# Webhooks
# ============================================================================

# Webhook for model events
resource "fhegbdt_webhook" "model_events" {
  name = "model-lifecycle-webhook"
  url  = "https://my-app.example.com/webhooks/fhe-gbdt"

  events = [
    "model.deployed",
    "model.retired",
    "drift.detected",
    "alert.triggered",
  ]

  headers = {
    "X-Custom-Header" = "my-value"
  }

  enabled = true
}

# Webhook for billing events
resource "fhegbdt_webhook" "billing_events" {
  name = "billing-webhook"
  url  = "https://my-app.example.com/webhooks/billing"

  events = [
    "billing.invoice.created",
    "billing.payment.failed",
  ]

  enabled = true
}

# ============================================================================
# Team Management
# ============================================================================

# Create a team
resource "fhegbdt_team" "ml_platform" {
  name        = "ML Platform Team"
  description = "Team responsible for ML infrastructure"
}

# Add team members
resource "fhegbdt_team_member" "alice" {
  team_id = fhegbdt_team.ml_platform.id
  email   = "alice@example.com"
  role    = "admin"
}

resource "fhegbdt_team_member" "bob" {
  team_id = fhegbdt_team.ml_platform.id
  email   = "bob@example.com"
  role    = "member"
}

# ============================================================================
# Outputs
# ============================================================================

output "fraud_detector_id" {
  description = "Fraud detector model ID"
  value       = fhegbdt_model.fraud_detector.id
}

output "fraud_detector_status" {
  description = "Fraud detector model status"
  value       = fhegbdt_model.fraud_detector.status
}

output "available_regions" {
  description = "Available deployment regions"
  value       = data.fhegbdt_regions.available.regions[*].code
}

output "current_usage" {
  description = "Current billing period usage"
  value = {
    predictions    = data.fhegbdt_usage.current.total_predictions
    compute_hours  = data.fhegbdt_usage.current.total_compute_hours
    storage_gb     = data.fhegbdt_usage.current.total_storage_gb
    period_start   = data.fhegbdt_usage.current.period_start
    period_end     = data.fhegbdt_usage.current.period_end
  }
}

output "webhook_secret" {
  description = "Webhook signing secret (for verification)"
  value       = fhegbdt_webhook.model_events.secret
  sensitive   = true
}
